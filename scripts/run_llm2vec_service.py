#!/usr/bin/env python3
"""
HTTP embedding service for LLM2Vec-adapted Dolphin models.

This script loads a Transformer checkpoint (e.g. a Dolphin-X1-8B model that
has been adapted with LLM2Vec MNTP + SimCSE) from a Hugging Face-style
folder and exposes a FastAPI service that returns sentence embeddings.

Key features
============
* Mean- or CLS-pooling with attention-mask awareness.
* Optional L2 normalisation of embeddings.
* Batched inference with configurable batch sizes.
* Automatic device selection with explicit CPU/CUDA/MPS overrides.
* Lightweight health endpoint exposing model metadata.

Example usage
-------------
    python scripts/run_llm2vec_service.py \
        --model-dir runs/dolphin_x1_llm2vec_v1/simcse \
        --host 0.0.0.0 \
        --port 8080 \
        --max-length 128 \
        --batch-size 8

Request payload:
    POST /embed
    {"texts": ["hello world", "monGARS is alive"]}

Response payload:
    {
        "embeddings": [[0.01, 0.02, ...], [0.05, -0.03, ...]],
        "dimension": 4096,
        "model": "runs/dolphin_x1_llm2vec_v1/simcse"
    }

This file is designed to be invoked by
`scripts/all_in_one_dolphin_llm2vec_pipeline.py` after training completes.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger("llm2vec_service")


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


@dataclass
class ServiceConfig:
    model_dir: str
    host: str
    port: int
    device: str
    max_length: int
    batch_size: int
    normalize: bool
    pooling: str
    log_level: str


def parse_args() -> ServiceConfig:
    parser = argparse.ArgumentParser(
        description="Run an embedding HTTP service for a LLM2Vec-adapted model."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the Hugging Face model directory (config.json, tokenizer, weights).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the HTTP server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Device to use: 'auto' (prefer CUDA, then MPS, else CPU), "
            "'cpu', 'cuda', or 'mps'."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (default: 128).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum batch size per forward pass (default: 8).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization of embeddings.",
    )
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy over token embeddings (default: mean).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info).",
    )

    ns = parser.parse_args()

    if ns.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    if ns.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    return ServiceConfig(
        model_dir=ns.model_dir,
        host=ns.host,
        port=ns.port,
        device=ns.device,
        max_length=ns.max_length,
        batch_size=ns.batch_size,
        normalize=ns.normalize,
        pooling=ns.pooling,
        log_level=ns.log_level.upper(),
    )


# ---------------------------------------------------------------------------
# Embedding wrapper
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Wrapper around a Transformer model that returns sentence embeddings."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "auto",
        max_length: int = 128,
        pooling: str = "mean",
        normalize: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize

        LOGGER.info("Loading tokenizer from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        LOGGER.info("Loading model from %s", model_dir)
        torch_dtype: Optional[torch.dtype] = None
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            torch_dtype = torch.float16
        self.model = AutoModel.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        self.device = self._resolve_device(device)
        LOGGER.info("Using device: %s", self.device)
        self.model.to(self.device)
        self.model.eval()

        # Cache embedding dimension by running a single dummy forward pass.
        with torch.inference_mode():
            dummy_inputs = self.tokenizer(
                ["dummy"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
            outputs = self.model(**dummy_inputs)
            last_hidden = self._extract_last_hidden(outputs)
            pooled = self._pool(last_hidden, dummy_inputs["attention_mask"])
            self.embedding_dim = pooled.shape[-1]
        LOGGER.info("Detected embedding dimension: %d", self.embedding_dim)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            return torch.device("cuda")
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available.")
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _extract_last_hidden(outputs) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        raise ValueError("Cannot find last_hidden_state in model outputs.")

    def _pool(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return token_embeddings[:, 0, :]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def embed_texts(self, texts: List[str], batch_size: int) -> List[List[float]]:
        if not texts:
            raise ValueError("texts must be a non-empty list")

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = [str(item) for item in texts[start : start + batch_size]]
            LOGGER.debug("Embedding batch %d:%d", start, start + len(chunk))
            with torch.inference_mode():
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                last_hidden = self._extract_last_hidden(outputs)
                pooled = self._pool(last_hidden, encoded["attention_mask"])
                if self.normalize:
                    pooled = self._l2_normalize(pooled)
                embeddings.extend(pooled.to("cpu").tolist())

        return embeddings


# ---------------------------------------------------------------------------
# FastAPI models
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_app(cfg: ServiceConfig) -> FastAPI:
    app = FastAPI(title="LLM2Vec Embedding Service")

    model_wrapper = EmbeddingModel(
        model_dir=cfg.model_dir,
        device=cfg.device,
        max_length=cfg.max_length,
        pooling=cfg.pooling,
        normalize=cfg.normalize,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=cfg.model_dir,
            dimension=model_wrapper.embedding_dim,
        )

    @app.post("/embed", response_model=EmbedResponse)
    def embed(request: EmbedRequest) -> EmbedResponse:
        if not request.texts:
            raise HTTPException(
                status_code=400, detail="Field 'texts' must be a non-empty list"
            )

        embeddings = model_wrapper.embed_texts(request.texts, batch_size=cfg.batch_size)
        return EmbedResponse(
            embeddings=embeddings,
            dimension=model_wrapper.embedding_dim,
            model=cfg.model_dir,
        )

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not os.path.isdir(cfg.model_dir):
        raise SystemExit(f"Model directory does not exist: {cfg.model_dir}")

    LOGGER.info("Starting LLM2Vec embedding service with model: %s", cfg.model_dir)

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
