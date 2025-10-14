#!/usr/bin/env python3
"""Export a lightweight LLM2Vec-style wrapper for the fine-tuned model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WRAPPER_PY = '''# llm2vec_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - PEFT may be unavailable on CPU-only setups
    PeftModel = None


class LLM2Vec:
    """Utility class that exposes :meth:`generate` and :meth:`embed` helpers."""

    def __init__(
        self,
        base_dir: str | Path,
        prefer_merged: bool = False,
        device: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Model directory {self.base_dir} does not exist")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_dir = self.base_dir / "tokenizer"
        if not tokenizer_dir.exists():
            raise FileNotFoundError("Tokenizer directory not found. Run training first.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if prefer_merged and (self.base_dir / "merged").exists():
            model_dir = self.base_dir / "merged"
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        else:
            adapter_dir = self.base_dir / "lora_adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError("LoRA adapter not found; run training before exporting the wrapper.")
            base_model = "dphn/Dolphin3.0-Llama3.1-8B"
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=load_in_4bit,
                device_map="auto",
                trust_remote_code=True,
            )
            if PeftModel is None:
                raise RuntimeError("peft is required to load the LoRA adapter")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
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
    def embed(self, texts: str | Iterable[str]):
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
        return embeddings.cpu()
'''

README_MD = """# LLM2Vec Wrapper (monGARS)

This wrapper exposes:

- `LLM2Vec.generate(prompt, ...)` → text generation
- `LLM2Vec.embed(texts)` → embeddings via mean-pooled hidden states

## Quickstart

```python
from llm2vec_wrapper import LLM2Vec
wrapper = LLM2Vec(base_dir="..", prefer_merged=False)
print(wrapper.generate("Bonjour [MOD=Hippocampus] Rappelle-moi le dernier contexte."))
vector = wrapper.embed("Allô, ça va?")
print(vector.shape)
```
"""


CONFIG_JSON = {
    "name": "monGARS-LLM2Vec",
    "backbone": "Dolphin3.0-Llama3.1-8B",
    "adapter": "lora_adapter",
    "supports_merged": True,
    "embed_strategy": "last_hidden_mean_pool",
    "prompt_tag": "[MOD=<Module>]",
}


def write_wrapper(model_dir: Path) -> None:
    wrap_dir = model_dir / "wrapper"
    wrap_dir.mkdir(parents=True, exist_ok=True)

    (wrap_dir / "llm2vec_wrapper.py").write_text(WRAPPER_PY, encoding="utf-8")
    (wrap_dir / "README.md").write_text(README_MD, encoding="utf-8")
    (wrap_dir / "config.json").write_text(
        json.dumps(CONFIG_JSON, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        default="out/monGARS_dolphin_finetuned",
        help="Directory containing the fine-tuned model artifacts",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"Model directory {model_dir} does not exist. Train a model first."
        )

    write_wrapper(model_dir)
    print(f"[DONE] Wrapper exported to {model_dir / 'wrapper'}")


if __name__ == "__main__":
    main()
