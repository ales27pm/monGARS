from __future__ import annotations

import json
from pathlib import Path

import pytest

from monGARS.mlops.wrapper_loader import WrapperBundleError, load_wrapper_bundle


def _write_wrapper(tmp_path: Path, name: str = "wrapper") -> Path:
    wrapper_dir = tmp_path / name
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    (wrapper_dir / "project_wrapper.py").write_text(
        "class ChatAndEmbed:\n"
        "    def __init__(self):\n"
        "        self.calls = []\n"
        "    def embed(self, texts):\n"
        "        if isinstance(texts, str):\n"
        "            texts = [texts]\n"
        "        self.calls.append(list(texts))\n"
        "        return [[float(len(text)), 0.0] for text in texts]\n"
    )
    (wrapper_dir / "config.json").write_text(
        json.dumps(
            {
                "base_model_id": "sample-base",
                "lora_dir": (tmp_path / "adapter").as_posix(),
                "max_seq_len": 512,
                "quantized_4bit": True,
                "vram_budget_mb": 4096,
                "offload_dir": (tmp_path / "offload").as_posix(),
            }
        )
    )
    return wrapper_dir


def test_load_wrapper_bundle_success(tmp_path: Path) -> None:
    wrapper_dir = _write_wrapper(tmp_path)
    bundle = load_wrapper_bundle(wrapper_dir)

    assert bundle.config.base_model_id == "sample-base"
    assert bundle.module_path == wrapper_dir / "project_wrapper.py"

    instance = bundle.create_instance()
    vectors = instance.embed(["hello"])
    assert vectors == [[5.0, 0.0]]


def test_load_wrapper_bundle_requires_files(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    with pytest.raises(WrapperBundleError):
        load_wrapper_bundle(missing_dir)


def test_load_wrapper_bundle_validates_module(tmp_path: Path) -> None:
    wrapper_dir = _write_wrapper(tmp_path)
    (wrapper_dir / "project_wrapper.py").write_text("class NotChat:\n    pass\n")

    with pytest.raises(WrapperBundleError):
        load_wrapper_bundle(wrapper_dir)
