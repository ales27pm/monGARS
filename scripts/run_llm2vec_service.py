#!/usr/bin/env python3
"""Serve the Dolphin 3.0 LLM2Vec wrapper over HTTP."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from scripts.export_llm2vec_wrapper import load_wrapper_config

LOGGER = logging.getLogger("llm2vec.service")


class EmbedRequest(BaseModel):
    inputs: Sequence[str] = Field(..., description="Texts to embed using Dolphin 3.0")
    normalise: Optional[bool] = Field(
        None,
        description="Override the default L2 normalisation configured in the wrapper",
    )


class EmbedResponse(BaseModel):
    vectors: List[List[float]] = Field(..., description="Embedding matrix")
    dims: int = Field(..., description="Embedding dimensionality")
    count: int = Field(..., description="Number of embeddings returned")
    backend: str = Field(..., description="Backend implementation identifier")
    model: str = Field(..., description="Base model powering the embeddings")
    normalised: bool = Field(..., description="Whether vectors were L2 normalised")


class EmbeddingService:
    """Lazy loader around the generated LLM2Vec wrapper."""

    def __init__(
        self,
        model_dir: Path,
        *,
        prefer_merged: bool = False,
        device: str | None = None,
        load_in_4bit: bool | None = None,
        wrapper_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.prefer_merged = prefer_merged
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.config = load_wrapper_config(model_dir)
        self._wrapper_factory = wrapper_factory or self._load_wrapper
        self._wrapper: Any | None = None
        self._lock = asyncio.Lock()

    def _load_wrapper(self) -> Any:
        wrapper_dir = self.model_dir / "wrapper"
        wrapper_path = wrapper_dir / "llm2vec_wrapper.py"
        if not wrapper_path.exists():
            raise RuntimeError(
                f"Wrapper Python module missing at {wrapper_path}. Run export_llm2vec_wrapper first."
            )

        spec = importlib.util.spec_from_file_location(
            "monGARS_llm2vec_wrapper", wrapper_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import wrapper from {wrapper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config_path = wrapper_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError(
                f"Wrapper config missing at {config_path}. Re-run export_llm2vec_wrapper."
            )
        config = json.loads(config_path.read_text())

        LOGGER.info(
            "Loading LLM2Vec wrapper",
            extra={
                "base_model": config.get("base_model_id"),
                "prefer_merged": self.prefer_merged,
                "device": self.device,
            },
        )

        return module.LLM2Vec(
            self.model_dir,
            prefer_merged=self.prefer_merged,
            device=self.device,
            load_in_4bit=self.load_in_4bit,
            config=config,
        )

    async def _ensure_wrapper(self) -> Any:
        if self._wrapper is not None:
            return self._wrapper
        async with self._lock:
            if self._wrapper is None:
                self._wrapper = await asyncio.to_thread(self._wrapper_factory)
        return self._wrapper

    async def embed(
        self, texts: Sequence[str], *, normalise: bool | None
    ) -> tuple[List[List[float]], bool]:
        if not texts:
            raise HTTPException(status_code=400, detail="inputs must not be empty")

        wrapper = await self._ensure_wrapper()
        tensor = await asyncio.to_thread(
            wrapper.embed, list(texts), normalise=normalise
        )
        matrix = tensor.tolist()
        dims = tensor.shape[-1]
        LOGGER.debug(
            "Generated embeddings",
            extra={"count": len(matrix), "dims": dims, "normalise": normalise},
        )
        return matrix, bool(
            normalise
            if normalise is not None
            else self.config["embedding_options"].get("normalise", False)
        )


def create_app(service: EmbeddingService) -> FastAPI:
    app = FastAPI(title="monGARS Dolphin 3 Embedding Service", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        payload = {
            "status": "ok",
            "model": service.config.get("base_model_id"),
            "backend": service.config.get("embedding_backend"),
            "pooling": service.config.get("embedding_options", {}).get("pooling_mode"),
            "max_length": service.config.get("embedding_options", {}).get("max_length"),
            "normalise": service.config.get("embedding_options", {}).get(
                "normalise", False
            ),
        }
        return JSONResponse(payload)

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        vectors, normalised = await service.embed(
            request.inputs, normalise=request.normalise
        )
        dims = len(vectors[0]) if vectors else 0
        return EmbedResponse(
            vectors=vectors,
            dims=dims,
            count=len(vectors),
            backend=service.config.get("embedding_backend", "huggingface"),
            model=service.config.get("base_model_id", "unknown"),
            normalised=normalised,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory that contains the fine-tuned Dolphin 3.0 artifacts",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to bind",
    )
    parser.add_argument(
        "--prefer-merged",
        action="store_true",
        help="Load the merged FP16 weights when available instead of PEFT adapters",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit torch device to load the model on (defaults to auto detection)",
    )
    bit_group = parser.add_mutually_exclusive_group()
    bit_group.add_argument(
        "--force-4bit",
        action="store_true",
        help="Force 4-bit loading even if the wrapper metadata disables it",
    )
    bit_group.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit loading and always use full precision",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model directory {model_dir} does not exist")

    load_in_4bit: bool | None
    if args.force_4bit:
        load_in_4bit = True
    elif args.disable_4bit:
        load_in_4bit = False
    else:
        load_in_4bit = None

    service = EmbeddingService(
        model_dir,
        prefer_merged=args.prefer_merged,
        device=args.device,
        load_in_4bit=load_in_4bit,
    )
    app = create_app(service)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
