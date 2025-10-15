#!/usr/bin/env python3
"""Export a lightweight LLM2Vec-style wrapper for the fine-tuned model."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

DEFAULT_BASE_MODEL = "dphn/Dolphin3.0-Llama3.1-8B"

WRAPPER_PY_TEMPLATE = '''# llm2vec_wrapper.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - PEFT may be unavailable on CPU-only setups
    PeftModel = None


try:
    CONFIG_PATH = Path(__file__).with_name("config.json")
except NameError:  # pragma: no cover - falls back during dynamic exec in tests
    CONFIG_PATH = Path("config.json")


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {
            "base_model_id": None,
            "embedding_options": {"max_length": 512, "normalise": False},
            "generation_defaults": {
                "temperature": 0.2,
                "top_p": 0.9,
                "max_new_tokens": 512,
            },
            "artifacts": {},
            "quantization": "",
        }
    return json.loads(CONFIG_PATH.read_text())


def _resolve_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


class LLM2Vec:
    """Utility class that exposes :meth:`generate` and :meth:`embed` helpers."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        prefer_merged: bool = False,
        device: str | None = None,
        load_in_4bit: bool | None = None,
        config: dict | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Model directory {self.base_dir} does not exist")

        self.config = config or _load_config()
        self.device = _resolve_device(device)
        self.prefer_merged = prefer_merged
        self.load_in_4bit = load_in_4bit
        artifacts = self.config.get("artifacts", {})
        self.tokenizer_dir = Path(artifacts.get("tokenizer_dir", self.base_dir / "tokenizer"))
        if not self.tokenizer_dir.exists():
            raise FileNotFoundError("Tokenizer directory not found. Run training first.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._max_length = int(
            self.config.get("embedding_options", {}).get("max_length", 512)
        )
        self._default_generation = self.config.get("generation_defaults", {})
        self._normalise = bool(
            self.config.get("embedding_options", {}).get("normalise", False)
        )

        self.model = self._load_model(artifacts)

        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, artifacts: dict) -> AutoModelForCausalLM:
        merged_subdir = artifacts.get("merged_subdir", "merged_fp16")
        adapter_subdir = artifacts.get("adapter_subdir", "lora_adapter")
        merged_path = self.base_dir / merged_subdir
        adapter_dir = self.base_dir / adapter_subdir
        base_model_id = self.config.get("base_model_id")

        if self.prefer_merged and merged_path.exists():
            dtype_name = self.config.get("embedding_options", {}).get("dtype")
            torch_dtype = torch.float32
            if dtype_name == "bfloat16":
                torch_dtype = torch.bfloat16
            elif dtype_name == "float16":
                torch_dtype = torch.float16

            return AutoModelForCausalLM.from_pretrained(
                str(merged_path),
                torch_dtype=torch_dtype if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_dir}; run training before exporting the wrapper."
            )

        target_model = base_model_id or "__BASE_MODEL_ID__"
        load_in_4bit = (
            self.load_in_4bit
            if self.load_in_4bit is not None
            else "4bit" in str(self.config.get("quantization", "")).lower()
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                target_model,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                target_model,
                device_map="auto",
                trust_remote_code=True,
            )

        if PeftModel is None:
            raise RuntimeError("peft is required to load the LoRA adapter")

        return PeftModel.from_pretrained(model, str(adapter_dir))

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        defaults = self._default_generation
        max_new_tokens = max_new_tokens or defaults.get("max_new_tokens", 512)
        temperature = defaults.get("temperature", 0.2) if temperature is None else temperature
        top_p = defaults.get("top_p", 0.9) if top_p is None else top_p

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.inference_mode()
    def embed(
        self, texts: str | Iterable[str], *, normalise: bool | None = None
    ) -> torch.Tensor:
        if isinstance(texts, str):
            batch_texts = [texts]
        else:
            batch_texts = list(texts)
        if not batch_texts:
            raise ValueError("texts must not be empty")

        encoded = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self.device)

        forward_kwargs = {
            **encoded,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }

        candidates = [self.model]
        for attr in ("model", "base_model", "transformer"):
            candidate = getattr(self.model, attr, None)
            if candidate is not None:
                candidates.append(candidate)

        seen: list[object] = []
        final_hidden = None
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.append(candidate)
            try:
                outputs = candidate(**forward_kwargs)
            except TypeError:
                continue
            except AttributeError:
                continue

            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is not None:
                final_hidden = hidden_states[-1]
                break

            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is not None:
                final_hidden = last_hidden_state
                break

        if final_hidden is None:
            raise RuntimeError(
                "model does not expose hidden states required for embedding"
            )

        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (final_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts

        should_normalise = self._normalise if normalise is None else normalise
        if should_normalise:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu()
'''

WRAPPER_PY = WRAPPER_PY_TEMPLATE.replace("__BASE_MODEL_ID__", DEFAULT_BASE_MODEL)

README_MD = """# LLM2Vec Wrapper (monGARS)

This wrapper exposes:

- `LLM2Vec.generate(prompt, ...)` → text generation with Dolphin 3.0 adapters
- `LLM2Vec.embed(texts, normalise=False)` → embeddings via mean-pooled hidden states

## Quickstart

```python
from llm2vec_wrapper import LLM2Vec
wrapper = LLM2Vec(base_dir="..", prefer_merged=False)
print(wrapper.generate("Bonjour [MOD=Hippocampus] Rappelle-moi le dernier contexte."))
vector = wrapper.embed("Allô, ça va?")
print(vector.shape)
```

## Serve over HTTP

The `scripts/run_llm2vec_service.py` utility spins up a FastAPI server using the
exported wrapper:

```bash
python scripts/run_llm2vec_service.py --model-dir .. --host 0.0.0.0 --port 8081
```

The service exposes `/healthz` and `/embed` endpoints so retrieval pipelines can
reuse the same Dolphin 3.0 checkpoint that powers chat.
"""

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "monGARS-LLM2Vec",
    "base_model_id": DEFAULT_BASE_MODEL,
    "embedding_backend": "huggingface",
    "embedding_options": {
        "pooling_mode": "mean",
        "normalise": False,
        "attention_mask_weighting": "mean",
        "dtype": "float32",
        "do_sample": False,
        "top_p": 1.0,
        "max_length": 512,
    },
    "generation_defaults": {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 512,
    },
    "artifacts": {
        "tokenizer_dir": None,
        "adapter_subdir": "lora_adapter",
        "merged_subdir": "merged_fp16",
    },
    "wrapper_metadata": {
        "version": 2,
        "generated_by": "scripts/export_llm2vec_wrapper.py",
    },
}


def _deep_update(target: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_wrapper_config(model_dir: Path) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config_path = model_dir / "wrapper_config.json"
    if config_path.exists():
        disk_config = json.loads(config_path.read_text())
        _deep_update(config, disk_config)

    artifacts = config.setdefault("artifacts", {})
    if not artifacts.get("tokenizer_dir"):
        artifacts["tokenizer_dir"] = str(model_dir / "tokenizer")
    if not artifacts.get("adapter_subdir"):
        artifacts["adapter_subdir"] = "lora_adapter"
    if not artifacts.get("merged_subdir"):
        artifacts["merged_subdir"] = "merged_fp16"
    config.setdefault("wrapper_metadata", {})
    config["wrapper_metadata"].update(
        {
            "version": 2,
            "generated_by": "scripts/export_llm2vec_wrapper.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "export_root": str(model_dir),
        }
    )

    if not config.get("base_model_id"):
        config["base_model_id"] = DEFAULT_BASE_MODEL

    return config


def write_wrapper(model_dir: Path, *, base_model: str | None = None) -> Path:
    wrap_dir = model_dir / "wrapper"
    wrap_dir.mkdir(parents=True, exist_ok=True)

    config = load_wrapper_config(model_dir)
    if base_model:
        config["base_model_id"] = base_model

    (wrap_dir / "llm2vec_wrapper.py").write_text(WRAPPER_PY, encoding="utf-8")
    (wrap_dir / "README.md").write_text(README_MD, encoding="utf-8")
    (wrap_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return wrap_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="out/monGARS_dolphin_finetuned",
        help="Directory containing the fine-tuned model artifacts",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model identifier stored in the wrapper config",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"Model directory {model_dir} does not exist. Train a model first."
        )

    wrap_dir = write_wrapper(model_dir, base_model=args.base_model)
    print(f"[DONE] Wrapper exported to {wrap_dir}")


if __name__ == "__main__":
    main()
