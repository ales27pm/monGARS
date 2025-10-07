#!/usr/bin/env python3
"""Headless training pipeline for Dolphin 3.0 Llama 3.1-8B using Unsloth.

This script automates end-to-end supervised fine-tuning (SFT) with features that
make it suitable for unattended, headless servers:

* Optional preparation of Ubuntu systems for headless mode (multi-user target)
  with an automatic reboot and resumable state tracking.
* Automatic probing of GPU VRAM to determine an initial batch size and a retry
  loop that halves the batch size when CUDA out-of-memory (OOM) errors occur.
* Optional CPU fallback that reloads the model without 4-bit quantisation to
  guarantee completion on systems without a working GPU.
* LoRA training with Unsloth's ``FastLanguageModel`` in 4-bit precision to
  reduce VRAM usage, including gradient checkpointing to lower memory pressure.
* Support for Hugging Face datasets or JSON/JSONL files containing either
  ``prompt``/``response`` pairs or chat-style ``messages`` arrays.  The
  tokenizer's chat template is used when available to stay aligned with Llama 3
  formatting.
* Optional conversion of the resulting checkpoint into an LLM2Vec encoder for
  embedding use-cases.

When no dataset flags are supplied the script falls back to a bundled sample
dataset under ``scripts/data`` so smoke tests can run without external
downloads. Provide ``--dataset-name`` or ``--train-file`` for real fine-tuning
jobs.

Example usage (first run):

```
python scripts/train_dolphin_unsloth.py \
    --hf-token <TOKEN> \
    --train-file data/train.jsonl \
    --validation-file data/val.jsonl \
    --prepare-headless --reboot
```

After the reboot, add an ``@reboot`` cron entry pointing to the same command or
run it manually again.  The script detects the persisted state file and skips
the headless preparation step before resuming training.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable, Optional

# Import Unsloth before Transformers so its patches take effect.
try:
    from unsloth import FastLanguageModel
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise RuntimeError(
        "Unsloth is required for this script. Install it with 'pip install unsloth'."
    ) from exc

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SAMPLE_DATA_DIR = ROOT_DIR / "scripts" / "data"
SAMPLE_TRAIN_FILE = SAMPLE_DATA_DIR / "dolphin_sft_sample_train.jsonl"
SAMPLE_VALIDATION_FILE = SAMPLE_DATA_DIR / "dolphin_sft_sample_validation.jsonl"

from modules.neurons.registry import update_manifest
from monGARS.mlops.artifacts import (
    WrapperConfig,
    build_adapter_summary,
    write_wrapper_bundle,
)

try:  # pragma: no cover - optional dependency
    from llm2vec import LLM2VecModel
except Exception:  # pragma: no cover - only needed when conversion requested
    LLM2VecModel = None  # type: ignore[assignment]


LOGGER = logging.getLogger("dolphin_autotrain")

DEFAULT_HEADLESS_TARGET = "multi-user.target"
DEFAULT_GRAPHICAL_TARGET = "graphical.target"
STATE_FILE = Path.home() / ".cache" / "monGARS" / "dolphin_autotrain_state.json"


def _ensure_state_dir() -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("State file is corrupt; ignoring it and starting fresh.")
    return {}


def _save_state(state: dict[str, Any]) -> None:
    _ensure_state_dir()
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def configure_logging(log_file: Optional[Path]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def _maybe_wrap_with_sudo(command: list[str]) -> list[str]:
    if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0:
        if shutil.which("sudo"):
            return ["sudo", *command]
        LOGGER.warning(
            "sudo is unavailable; attempting to run '%s' without elevation.",
            " ".join(command),
        )
    return command


def _locate_adapter_weights(adapter_dir: Path) -> Optional[Path]:
    candidates = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    LOGGER.debug(
        "No adapter weights detected in output directory",
        extra={"path": str(adapter_dir)},
    )
    return None


def _safe_len(dataset: Optional[Dataset]) -> Optional[int]:
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def _resolve_sample_dataset_files() -> Optional[dict[str, str]]:
    if not SAMPLE_TRAIN_FILE.exists():
        return None

    files: dict[str, str] = {"train": str(SAMPLE_TRAIN_FILE)}
    if SAMPLE_VALIDATION_FILE.exists():
        files["validation"] = str(SAMPLE_VALIDATION_FILE)
    return files


def generate_chat_and_embed_wrapper(
    *,
    base_model_id: str,
    output_dir: Path,
    max_seq_len: int,
    vram_budget_mb: int,
    activation_buffer_mb: int,
    offload_dir: Path,
) -> Path:
    wrapper_config = WrapperConfig(
        base_model_id=base_model_id,
        lora_dir=output_dir,
        max_seq_len=max_seq_len,
        vram_budget_mb=vram_budget_mb,
        activation_buffer_mb=activation_buffer_mb,
        offload_dir=offload_dir,
    )
    paths = write_wrapper_bundle(wrapper_config, output_dir)
    wrapper_dir = paths["module"].parent
    LOGGER.info("Wrapper bundle created at %s", wrapper_dir)
    return wrapper_dir


def build_training_summary(
    *,
    config: "TrainingConfig",
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    wrapper_dir: Optional[Path],
    gpu_free_mb: int,
    gpu_total_mb: int,
    wrapper_settings: Optional[dict[str, Any]],
) -> dict[str, Any]:
    weights_path = _locate_adapter_weights(config.output_dir)
    summary = build_adapter_summary(
        adapter_dir=config.output_dir,
        weights_path=weights_path,
        wrapper_dir=wrapper_dir,
        status="success",
        labels={"pipeline": "unsloth_autotrain", "base_model": config.base_model_id},
        metrics={},
        training={},
    )

    metrics = summary.setdefault("metrics", {})
    train_len = _safe_len(train_dataset)
    if train_len is not None:
        metrics["train_dataset_size"] = train_len
    eval_len = _safe_len(eval_dataset)
    if eval_len is not None:
        metrics["eval_dataset_size"] = eval_len
    metrics["per_device_train_batch_size"] = config.per_device_train_batch_size
    metrics["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    metrics["learning_rate"] = config.learning_rate
    metrics["num_train_epochs"] = config.num_train_epochs
    metrics["max_seq_length"] = config.max_seq_length

    training_meta = summary.setdefault("training", {})
    training_meta.update(
        {
            "base_model": config.base_model_id,
            "dataset_name": config.dataset_name,
            "dataset_config": config.dataset_config,
            "train_split": config.train_split,
            "eval_split": config.eval_split,
            "max_seq_len": config.max_seq_length,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "weight_decay": config.weight_decay,
            "warmup_steps": config.warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "lora": {
                "r": config.lora_r,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
            },
            "precision": {
                "load_in_4bit": config.load_in_4bit,
                "bf16": config.bf16,
                "fp16": config.fp16,
            },
        }
    )

    data_files_summary: dict[str, str] = {}
    if config.train_file is not None:
        data_files_summary["train"] = str(config.train_file)
    if config.validation_file is not None:
        data_files_summary["validation"] = str(config.validation_file)
    if data_files_summary:
        training_meta["data_files"] = data_files_summary

    if config.using_sample_dataset and config.sample_dataset_files is not None:
        training_meta["dataset_fallback"] = {
            "type": "bundled_sample",
            "files": dict(config.sample_dataset_files),
        }

    if wrapper_settings is not None:
        training_meta["wrapper"] = wrapper_settings

    labels = summary.setdefault("labels", {})
    if config.using_sample_dataset:
        labels["dataset_source"] = "bundled_sample"
    elif config.dataset_name is not None:
        labels["dataset_source"] = "huggingface"
    elif data_files_summary:
        labels["dataset_source"] = "custom_files"
    labels["llm2vec"] = "enabled" if wrapper_dir is not None else "not_configured"
    if wrapper_dir is not None:
        labels["wrapper"] = "chat_and_embed"

    summary.setdefault("analysis", {}).setdefault(
        "gpu_vram", {"free_mb": gpu_free_mb, "total_mb": gpu_total_mb}
    )

    return summary


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Training summary written to %s", path)


def switch_systemctl_target(target: str, persist: bool) -> None:
    action = "set-default" if persist else "isolate"
    command = _maybe_wrap_with_sudo(["systemctl", action, target])
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except (
        FileNotFoundError
    ) as exc:  # pragma: no cover - systemctl not available on tests
        raise RuntimeError("systemctl is not available on this system") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(
            f"Failed to switch systemd target to {target!r}: {stderr.strip() or exc}"
        ) from exc


def reboot_system() -> None:
    command = _maybe_wrap_with_sudo(["reboot"])
    LOGGER.info("Rebooting system to apply headless mode...")
    try:
        subprocess.run(command, check=True)
    except (
        subprocess.CalledProcessError
    ) as exc:  # pragma: no cover - reboot rarely tested
        raise RuntimeError(f"Failed to reboot the machine: {exc}") from exc


def detect_gpu_memory() -> tuple[int, int]:
    """Return (free_mb, total_mb) for the first CUDA device, or (0, 0)."""

    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return int(free // (1024 * 1024)), int(total // (1024 * 1024))
        except RuntimeError:
            pass

    if shutil.which("nvidia-smi"):
        try:
            free_mb = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .decode("utf-8")
                .strip()
                .split("\n")[0]
            )
            total_mb = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                )
                .decode("utf-8")
                .strip()
                .split("\n")[0]
            )
            return free_mb, total_mb
        except (subprocess.CalledProcessError, ValueError):
            LOGGER.warning("Failed to parse nvidia-smi output; treating as no GPU.")

    return 0, 0


def recommend_batch_size(free_mb: int) -> int:
    if free_mb >= 24000:
        return 8
    if free_mb >= 16000:
        return 6
    if free_mb >= 12000:
        return 4
    if free_mb >= 8000:
        return 2
    return 1


def format_conversation(
    example: dict[str, Any],
    tokenizer,
    args: "TrainingConfig",
) -> str:
    if args.text_column in example and example[args.text_column]:
        text_value = example[args.text_column]
        if isinstance(text_value, str) and text_value.strip():
            return text_value
        if isinstance(text_value, (list, tuple)):
            return "\n".join(str(item) for item in text_value)

    messages = []
    if args.messages_column and args.messages_column in example:
        raw_messages = example.get(args.messages_column) or []
        if isinstance(raw_messages, str):
            try:
                raw_messages = json.loads(raw_messages)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON messages from column '{args.messages_column}'."
                ) from exc
        if not isinstance(raw_messages, Iterable):
            raise ValueError(
                f"Column '{args.messages_column}' must contain a list of chat messages."
            )
        for message in raw_messages:
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                raise ValueError(
                    "Each chat message must be an object with 'role' and 'content' keys."
                )
            messages.append({"role": message["role"], "content": message["content"]})
    else:
        prompt = example.get(args.prompt_column)
        response = example.get(args.response_column)
        system_prompt = (
            example.get(args.system_column)
            if args.system_column and args.system_column in example
            else args.system_prompt
        )
        if prompt is None or response is None:
            raise ValueError(
                "Dataset rows must contain both prompt and response information when"
                " messages_column is not provided."
            )
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt)})
        messages.append({"role": "assistant", "content": str(response)})

    try:
        return tokenizer.apply_chat_template(  # type: ignore[return-value]
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except AttributeError:
        # Fallback to a minimal conversation format.
        rendered = []
        for message in messages:
            rendered.append(
                f"### {message['role'].capitalize()}:\n{textwrap.dedent(str(message['content'])).strip()}"
            )
        return "\n\n".join(rendered) + "\n\n### Assistant:\n"


@dataclasses.dataclass
class TrainingConfig:
    output_dir: Path
    base_model_id: str
    max_seq_length: int
    learning_rate: float
    num_train_epochs: float
    weight_decay: float
    warmup_steps: int
    gradient_accumulation_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    logging_steps: int
    save_strategy: str
    save_steps: Optional[int]
    evaluation_strategy: str
    eval_steps: Optional[int]
    load_in_4bit: bool
    bf16: bool
    fp16: bool
    lr_scheduler_type: str
    gradient_checkpointing: bool
    max_grad_norm: float
    per_device_train_batch_size: int
    dataset_name: Optional[str]
    dataset_config: Optional[str]
    train_split: str
    eval_split: Optional[str]
    train_file: Optional[Path]
    validation_file: Optional[Path]
    text_column: str
    prompt_column: str
    response_column: str
    system_column: Optional[str]
    system_prompt: Optional[str]
    messages_column: Optional[str]
    seed: int
    report_to: tuple[str, ...]
    dataset_cache_dir: Optional[Path]
    resume_from_checkpoint: Optional[Path]
    deepspeed: Optional[Path]
    allow_tf32: bool
    using_sample_dataset: bool = False
    sample_dataset_files: Optional[dict[str, str]] = dataclasses.field(default=None)


def build_training_arguments(config: TrainingConfig, device: str) -> TrainingArguments:
    args_kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "evaluation_strategy": config.evaluation_strategy,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "seed": config.seed,
        "report_to": list(config.report_to) or ["none"],
        "gradient_checkpointing": config.gradient_checkpointing,
        "bf16": config.bf16 and device != "cpu",
        "fp16": config.fp16 and device != "cpu",
        "tf32": config.allow_tf32,
    }

    if config.save_steps is not None:
        args_kwargs["save_steps"] = config.save_steps
    if config.eval_steps is not None:
        args_kwargs["eval_steps"] = config.eval_steps
    if config.resume_from_checkpoint is not None:
        args_kwargs["resume_from_checkpoint"] = str(config.resume_from_checkpoint)
    if config.deepspeed is not None:
        args_kwargs["deepspeed"] = str(config.deepspeed)

    init_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    remapped_args: dict[str, str] = {"evaluation_strategy": "eval_strategy"}
    for old_key, new_key in remapped_args.items():
        if (
            old_key in args_kwargs
            and old_key not in init_params
            and new_key in init_params
        ):
            args_kwargs[new_key] = args_kwargs.pop(old_key)

    return TrainingArguments(**args_kwargs)


def load_datasets_for_training(
    config: TrainingConfig,
    tokenizer,
) -> tuple[Dataset, Optional[Dataset]]:
    config.using_sample_dataset = False
    config.sample_dataset_files = None

    data_files: dict[str, str] | None = None
    if config.train_file is not None:
        data_files = {"train": str(config.train_file)}
        if config.validation_file is not None:
            data_files["validation"] = str(config.validation_file)

    if config.dataset_name is None and data_files is None:
        sample_files = _resolve_sample_dataset_files()
        if sample_files is None:
            raise ValueError(
                "Either --dataset-name or --train-file must be provided to supply training data."
            )

        LOGGER.warning(
            "No dataset provided; using bundled sample dataset at %s. Provide --dataset-name or --train-file for production runs.",
            sample_files["train"],
        )
        data_files = dict(sample_files)
        config.using_sample_dataset = True
        config.sample_dataset_files = sample_files
        config.train_file = Path(sample_files["train"])
        if "validation" in sample_files:
            config.validation_file = Path(sample_files["validation"])

    load_kwargs: dict[str, Any] = {}
    if config.dataset_cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.dataset_cache_dir)

    if config.dataset_name is not None:
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            **load_kwargs,
        )
    else:
        dataset = load_dataset("json", data_files=data_files, **load_kwargs)

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Loaded dataset must be a DatasetDict with named splits.")

    if config.train_split not in dataset:
        raise ValueError(f"Train split '{config.train_split}' not found in dataset.")
    train_dataset = dataset[config.train_split]

    eval_dataset: Optional[Dataset] = None
    if config.eval_split and config.eval_split in dataset:
        eval_dataset = dataset[config.eval_split]
    elif config.validation_file is not None and "validation" in dataset:
        eval_dataset = dataset["validation"]

    def _map_example(example: dict[str, Any]) -> dict[str, Any]:
        text = format_conversation(example, tokenizer, config)
        return {"text": text}

    LOGGER.info("Tokenising training dataset...")
    train_dataset = train_dataset.map(
        _map_example, remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        LOGGER.info("Tokenising evaluation dataset...")
        eval_dataset = eval_dataset.map(
            _map_example,
            remove_columns=eval_dataset.column_names,
        )

    return train_dataset, eval_dataset


class SupervisedFineTuningCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [feature["text"] for feature in features]
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        return {
            **encoded,
            "labels": labels,
        }


def prepare_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    LOGGER.info("Loading base model %s...", config.base_model_id)
    load_kwargs: dict[str, Any] = {
        "model_name": config.base_model_id,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if config.load_in_4bit:
        load_kwargs["dtype"] = torch.float16
    else:
        load_kwargs["dtype"] = torch.float32

    if torch.cuda.is_available():
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_mb = int(free_bytes // (1024 * 1024))
            budget_mb = max(1024, free_mb - 512)
            load_kwargs["device_map"] = {"": 0}
            load_kwargs["max_memory"] = {0: f"{budget_mb}MiB"}
        except Exception:
            load_kwargs["device_map"] = {"": 0}

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.use_cache = False

    LOGGER.info(
        "Configuring LoRA adapters (r=%s, alpha=%s, dropout=%s)...",
        config.lora_r,
        config.lora_alpha,
        config.lora_dropout,
    )
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=(
            "unsloth" if config.gradient_checkpointing else False
        ),
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model, tokenizer


def run_training_with_retries(
    base_model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: TrainingConfig,
    max_retries: int,
    allow_cpu_fallback: bool,
) -> Trainer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_device_batch_size = config.per_device_train_batch_size
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        LOGGER.info(
            "Starting training attempt %s/%s with batch size %s on %s",
            attempt + 1,
            max_retries + 1,
            per_device_batch_size,
            device,
        )
        config.per_device_train_batch_size = per_device_batch_size
        training_args = build_training_arguments(config, device)
        data_collator = SupervisedFineTuningCollator(tokenizer, config.max_seq_length)

        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        try:
            trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
            LOGGER.info("Training completed successfully on attempt %s", attempt + 1)
            return trainer
        except RuntimeError as err:
            error_message = str(err).lower()
            last_error = err
            if (
                "out of memory" in error_message
                and per_device_batch_size > 1
                and device != "cpu"
            ):
                per_device_batch_size = max(1, per_device_batch_size // 2)
                LOGGER.warning(
                    "CUDA OOM encountered; reducing batch size to %s",
                    per_device_batch_size,
                )
                torch.cuda.empty_cache()
                continue
            LOGGER.exception("Training failed on attempt %s", attempt + 1)
            break

    if allow_cpu_fallback and device != "cpu":
        LOGGER.info("Falling back to CPU training with full-precision weights.")
        del base_model
        torch.cuda.empty_cache()

        cpu_config = dataclasses.replace(
            config,
            per_device_train_batch_size=1,
            load_in_4bit=False,
            bf16=False,
            fp16=False,
        )
        base_model_cpu, tokenizer_cpu = prepare_model_and_tokenizer(cpu_config)
        training_args = build_training_arguments(cpu_config, "cpu")
        data_collator = SupervisedFineTuningCollator(
            tokenizer_cpu, cpu_config.max_seq_length
        )

        trainer = Trainer(
            model=base_model_cpu,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer_cpu,
            data_collator=data_collator,
        )
        trainer.train(resume_from_checkpoint=cpu_config.resume_from_checkpoint)
        return trainer

    assert last_error is not None
    raise last_error


def convert_to_llm2vec(output_dir: Path, tokenizer_dir: Path) -> None:
    if LLM2VecModel is None:
        raise RuntimeError(
            "llm2vec is not installed. Install it with 'pip install llm2vec' to enable conversion."
        )
    LOGGER.info("Converting fine-tuned checkpoint into an LLM2Vec encoder...")
    encoder = LLM2VecModel.from_pretrained(str(output_dir))
    target_dir = output_dir / "llm2vec_encoder"
    encoder.save_pretrained(str(target_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tokenizer.save_pretrained(str(target_dir))


def parse_arguments(
    argv: Optional[list[str]] = None,
) -> tuple[argparse.Namespace, TrainingConfig]:
    parser = argparse.ArgumentParser(
        description="Automated Dolphin 3.0 fine-tuning pipeline"
    )
    parser.add_argument("--hf-token", type=str, help="Hugging Face access token")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="cognitivecomputations/Dolphin3.0-Llama3.1-8B",
        help="Base Hugging Face model identifier to fine-tune.",
    )
    parser.add_argument(
        "--prepare-headless",
        action="store_true",
        help="Persistently switch the system to the multi-user (CLI) target and exit.",
    )
    parser.add_argument(
        "--reboot",
        action="store_true",
        help="Reboot the machine after preparing headless mode.",
    )
    parser.add_argument(
        "--restore-graphical-target",
        action="store_true",
        help="Restore the graphical.target after successful training.",
    )
    parser.add_argument(
        "--headless-target",
        type=str,
        default=DEFAULT_HEADLESS_TARGET,
        help="Systemd target used for headless mode (default: multi-user.target)",
    )
    parser.add_argument(
        "--graphical-target",
        type=str,
        default=DEFAULT_GRAPHICAL_TARGET,
        help="Systemd target used when restoring GUI (default: graphical.target)",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Retry training on CPU if all GPU attempts fail.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of GPU retries on CUDA OOM errors.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        help="Path to a JSON or JSONL file with training data (prompt/response pairs).",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        help="Path to a JSON or JSONL validation file for evaluation.",
    )
    parser.add_argument(
        "--dataset-name", type=str, help="Hugging Face dataset identifier."
    )
    parser.add_argument(
        "--dataset-config", type=str, help="Optional dataset configuration name."
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split used for training when --dataset-name is specified.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        help="Dataset split used for evaluation (defaults to validation split).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenisation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Peak learning rate for AdamW.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay applied during training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Linear warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps per device.",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Interval (in steps) between logging updates.",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Checkpoint saving strategy.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        help="Number of steps between checkpoints when save-strategy=steps.",
    )
    parser.add_argument(
        "--evaluation-strategy",
        type=str,
        default="epoch",
        choices=("no", "steps", "epoch"),
        help="Evaluation scheduling strategy.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        help="Number of steps between evaluations when evaluation-strategy=steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dolphin3_finetuned"),
        help="Directory where the fine-tuned checkpoint will be written.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume training from a specific checkpoint directory.",
    )
    parser.add_argument(
        "--deepspeed",
        type=Path,
        help="Path to a DeepSpeed configuration file.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 mixed precision (requires Ampere or newer GPUs).",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Force FP16 mixed precision (default when 4-bit quantisation is active).",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 mixed precision.",
    )
    parser.set_defaults(fp16=None)
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing even when running on GPU.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type supported by transformers.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing preformatted text; used when other columns missing.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column containing user prompts when messages-column is absent.",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="response",
        help="Column containing assistant responses when messages-column is absent.",
    )
    parser.add_argument(
        "--system-column",
        type=str,
        help="Column containing optional system prompts when using prompt/response format.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Fallback system prompt when dataset rows do not define one.",
    )
    parser.add_argument(
        "--messages-column",
        type=str,
        help="Column containing a list of chat messages compatible with chat templates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialisation and shuffling.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        nargs="*",
        default=("none",),
        help="Reporting integrations (e.g. wandb, tensorboard).",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        help="Optional cache directory for Hugging Face datasets.",
    )
    parser.add_argument(
        "--no-generate-wrapper",
        action="store_true",
        help="Skip generating the ChatAndEmbed wrapper bundle after training.",
    )
    parser.add_argument(
        "--wrapper-vram-budget-mb",
        type=int,
        default=7424,
        help="VRAM budget (MiB) to encode in the generated wrapper.",
    )
    parser.add_argument(
        "--wrapper-activation-buffer-mb",
        type=int,
        default=1024,
        help="Activation buffer (MiB) reserved in the wrapper for runtime spikes.",
    )
    parser.add_argument(
        "--wrapper-offload-dir",
        type=Path,
        help="Directory the wrapper should use for CPU offloading (defaults to <output>/offload).",
    )
    parser.add_argument(
        "--wrapper-max-seq-length",
        type=int,
        help="Override maximum sequence length encoded in the wrapper bundle.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        help="Optional adapter registry path; update manifest metadata when provided.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Location for the training summary JSON (defaults to <output>/training_summary.json).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write detailed logs to the specified file in addition to stdout.",
    )
    parser.add_argument(
        "--convert-to-llm2vec",
        action="store_true",
        help="Convert the fine-tuned checkpoint into an LLM2Vec encoder after training.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Override the default state file used for reboot detection.",
    )
    parser.add_argument(
        "--no-autoresume",
        action="store_true",
        help="Ignore any stored autoresume state and run exactly as configured.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 acceleration on Ampere+ GPUs.",
    )

    parsed = parser.parse_args(argv)

    state_file = Path(parsed.state_file) if parsed.state_file else STATE_FILE
    globals()["STATE_FILE"] = state_file

    default_fp16 = True
    if parsed.fp16 is None:
        fp16_value = default_fp16
    else:
        fp16_value = parsed.fp16

    config = TrainingConfig(
        output_dir=parsed.output_dir,
        base_model_id=parsed.base_model_id,
        max_seq_length=parsed.max_seq_length,
        learning_rate=parsed.learning_rate,
        num_train_epochs=parsed.num_train_epochs,
        weight_decay=parsed.weight_decay,
        warmup_steps=parsed.warmup_steps,
        gradient_accumulation_steps=parsed.gradient_accumulation_steps,
        lora_r=parsed.lora_r,
        lora_alpha=parsed.lora_alpha,
        lora_dropout=parsed.lora_dropout,
        logging_steps=parsed.logging_steps,
        save_strategy=parsed.save_strategy,
        save_steps=parsed.save_steps,
        evaluation_strategy=parsed.evaluation_strategy,
        eval_steps=parsed.eval_steps,
        load_in_4bit=True,
        bf16=parsed.bf16,
        fp16=fp16_value,
        lr_scheduler_type=parsed.lr_scheduler_type,
        gradient_checkpointing=not parsed.disable_gradient_checkpointing,
        max_grad_norm=parsed.max_grad_norm,
        per_device_train_batch_size=1,  # placeholder; updated after GPU probing
        dataset_name=parsed.dataset_name,
        dataset_config=parsed.dataset_config,
        train_split=parsed.train_split,
        eval_split=parsed.eval_split,
        train_file=parsed.train_file,
        validation_file=parsed.validation_file,
        text_column=parsed.text_column,
        prompt_column=parsed.prompt_column,
        response_column=parsed.response_column,
        system_column=parsed.system_column,
        system_prompt=parsed.system_prompt,
        messages_column=parsed.messages_column,
        seed=parsed.seed,
        report_to=tuple(parsed.report_to),
        dataset_cache_dir=parsed.dataset_cache_dir,
        resume_from_checkpoint=parsed.resume_from_checkpoint,
        deepspeed=parsed.deepspeed,
        allow_tf32=parsed.allow_tf32,
    )

    return parsed, config


def main(argv: Optional[list[str]] = None) -> None:
    args, config = parse_arguments(argv)
    configure_logging(args.log_file)

    state = _load_state() if not args.no_autoresume else {}
    autoresume_pending = bool(state.get("pending_training"))

    if autoresume_pending and args.prepare_headless:
        LOGGER.info("Detected autoresume state; skipping headless preparation phase.")
        args.prepare_headless = False
        args.reboot = False

    if args.prepare_headless and not autoresume_pending:
        LOGGER.info("Setting persistent systemd target to %s", args.headless_target)
        switch_systemctl_target(args.headless_target, persist=True)
        state["pending_training"] = True
        _save_state(state)
        if args.reboot:
            LOGGER.info("Reboot requested; syncing disks before restart.")
            subprocess.run(["sync"], check=False)
            reboot_system()
        LOGGER.info("Headless mode configured. Re-run the script to start training.")
        return

    if autoresume_pending:
        LOGGER.info("Resuming training after reboot; clearing autoresume flag.")
        state.pop("pending_training", None)
        _save_state(state)

    set_seed(config.seed)

    free_mb, total_mb = detect_gpu_memory()
    LOGGER.info("GPU VRAM free/total: %s/%s MB", free_mb, total_mb)

    if free_mb == 0 or total_mb == 0:
        LOGGER.warning("No compatible GPU detected; defaulting to CPU mode.")
        config.per_device_train_batch_size = 1
        device = "cpu"
    else:
        config.per_device_train_batch_size = recommend_batch_size(free_mb)

    if args.hf_token:
        LOGGER.info("Authenticating with Hugging Face hub...")
        login(token=args.hf_token, add_to_git_credential=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = prepare_model_and_tokenizer(config)

    train_dataset, eval_dataset = load_datasets_for_training(config, tokenizer)

    trainer = run_training_with_retries(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        config,
        max_retries=args.max_retries,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )

    LOGGER.info("Saving fine-tuned model to %s", config.output_dir)
    trainer.model.save_pretrained(config.output_dir)
    trainer.tokenizer.save_pretrained(config.output_dir)

    wrapper_dir: Optional[Path] = None
    wrapper_settings: Optional[dict[str, Any]] = None
    if args.no_generate_wrapper:
        LOGGER.info("Wrapper generation disabled via --no-generate-wrapper.")
    else:
        wrapper_max_seq_len = args.wrapper_max_seq_length or config.max_seq_length
        offload_dir = (
            args.wrapper_offload_dir.resolve()
            if args.wrapper_offload_dir is not None
            else (config.output_dir / "offload").resolve()
        )
        wrapper_dir = generate_chat_and_embed_wrapper(
            base_model_id=config.base_model_id,
            output_dir=config.output_dir,
            max_seq_len=wrapper_max_seq_len,
            vram_budget_mb=args.wrapper_vram_budget_mb,
            activation_buffer_mb=args.wrapper_activation_buffer_mb,
            offload_dir=offload_dir,
        )
        wrapper_settings = {
            "vram_budget_mb": args.wrapper_vram_budget_mb,
            "activation_buffer_mb": args.wrapper_activation_buffer_mb,
            "offload_dir": str(offload_dir),
            "max_seq_len": wrapper_max_seq_len,
        }

    if args.convert_to_llm2vec:
        convert_to_llm2vec(config.output_dir, config.output_dir)

    summary = build_training_summary(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        wrapper_dir=wrapper_dir,
        gpu_free_mb=free_mb,
        gpu_total_mb=total_mb,
        wrapper_settings=wrapper_settings,
    )

    artifacts = summary.setdefault("artifacts", {})
    if args.convert_to_llm2vec:
        artifacts["llm2vec_encoder"] = str(config.output_dir / "llm2vec_encoder")
    labels = summary.setdefault("labels", {})
    labels["llm2vec_export"] = "true" if args.convert_to_llm2vec else "false"

    summary_path = args.summary_file or (config.output_dir / "training_summary.json")
    write_summary(summary, summary_path)

    if args.registry_path:
        try:
            manifest = update_manifest(args.registry_path, summary)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive logging for orchestration
            LOGGER.error(
                "Failed to update adapter registry at %s: %s", args.registry_path, exc
            )
        else:
            LOGGER.info(
                "Adapter manifest updated",
                extra={
                    "path": str(manifest.path),
                    "active_version": (
                        manifest.current.version if manifest.current else None
                    ),
                },
            )

    if args.restore_graphical_target:
        LOGGER.info("Restoring graphical target %s", args.graphical_target)
        try:
            switch_systemctl_target(args.graphical_target, persist=True)
        except RuntimeError as exc:
            LOGGER.warning("Failed to restore graphical target: %s", exc)

    state["last_success"] = {
        "output_dir": str(config.output_dir),
    }
    _save_state(state)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
