#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unified_dolphin_x1_enhanced.py

Enhanced all-in-one script for Dolphin-X1 unified model building.

Features
--------
- Production-grade configuration with validation (Pydantic)
- Advanced error handling and logging (Rich + logging)
- Memory-optimized training and inference (Unsloth + 4-bit)
- Multi-format dataset support (Alpaca, ChatML, custom)
- System monitoring and metrics (psutil + GPUtil)
- Model checkpointing and saving with metadata
- Unified model: generation + embeddings (LLM2Vec)
- Internal reasoning loop for chain-of-thought style refinement
- REPL with session history and optional 3D embedding visualisation

Usage
-----
Quick start (train + REPL using default config):
    python build_unified_dolphin_x1_enhanced.py --train

Train with custom config:
    python build_unified_dolphin_x1_enhanced.py --train --config my_config.json

Use existing model with REPL:
    python build_unified_dolphin_x1_enhanced.py --model-dir ./models/my_model

Export default config:
    python build_unified_dolphin_x1_enhanced.py --export-config default_config.json
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import time
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)

console = Console()

# ---------------------------------------------------------------------------
# Core ML imports (fail-fast if missing)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    from datasets import load_dataset
    from llm2vec import LLM2Vec
except ImportError as e:
    console.print(f"[red]Import error: {e}[/red]")
    console.print("[yellow]Install required packages, for example:[/yellow]")
    console.print(
        """
    pip install "unsloth[torch]" "transformers>=4.42.0" "datasets" "accelerate" \
               "bitsandbytes" "llm2vec" "rich" "scikit-learn" "matplotlib" "trl" \
               "pydantic<3" "psutil" "GPUtil"
    """
    )
    raise SystemExit(1)

import GPUtil
import psutil
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("dolphin_x1_enhanced")

# ---------------------------------------------------------------------------
# Pydantic configuration models
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Dataset configuration"""

    source_type: str = Field(..., description="local_jsonl or hf_hub")
    path: Optional[str] = None  # for local_jsonl
    hf_name: Optional[str] = None  # for hf_hub
    hf_subset: Optional[str] = None
    split: str = "train"
    input_field: str = "instruction"
    output_field: str = "output"
    validation_size: float = 0.1
    max_samples: Optional[int] = None

    @validator("source_type")
    def validate_source_type(cls, v: str) -> str:
        allowed = ["local_jsonl", "hf_hub"]
        if v not in allowed:
            raise ValueError(f"source_type must be one of {allowed}")
        return v

    @validator("validation_size")
    def validate_val_size(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("validation_size must be in [0, 1)")
        return v


class TrainingConfig(BaseModel):
    """Training configuration"""

    batch_size: int = Field(gt=0, le=128)
    gradient_accumulation_steps: int = Field(gt=0, le=128)
    num_epochs: int = Field(gt=0, le=100)
    learning_rate: float = Field(gt=0, le=1e-2)
    warmup_ratio: float = Field(ge=0, le=0.5)
    weight_decay: float = Field(ge=0, le=0.1)
    max_grad_norm: float = Field(gt=0, le=10.0)

    lora_r: int = Field(gt=0, le=256)
    lora_alpha: int = Field(gt=0, le=512)
    lora_dropout: float = Field(ge=0, le=0.5)
    lora_target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    @validator("batch_size")
    def validate_effective_batch(cls, v: int, values: Dict[str, Any]) -> int:
        gas = values.get("gradient_accumulation_steps", 1)
        if v * gas > 1024:
            raise ValueError(
                "Effective batch size (batch_size * gradient_accumulation_steps) too large"
            )
        return v


class ModelConfig(BaseModel):
    """Model configuration"""

    base_model: str
    max_seq_len: int = Field(ge=256, le=32768)
    model_dtype: str = "bfloat16"  # float16, bfloat16, float32
    use_gradient_checkpointing: bool = True

    quantize_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    @validator("model_dtype")
    def validate_dtype(cls, v: str) -> str:
        allowed = ["float16", "bfloat16", "float32"]
        if v not in allowed:
            raise ValueError(f"model_dtype must be one of {allowed}")
        return v


class InferenceConfig(BaseModel):
    """Inference & reasoning configuration"""

    max_new_tokens: int = Field(gt=0, le=4096)
    temperature: float = Field(ge=0, le=2.0)
    top_p: float = Field(ge=0, le=1.0)
    top_k: int = Field(ge=0, le=100)
    repetition_penalty: float = Field(ge=1.0, le=2.0)
    do_sample: bool = True
    num_beams: int = Field(ge=1, le=8)

    reasoning_steps: int = Field(ge=0, le=10)
    reasoning_max_tokens: int = Field(gt=0, le=512)

    embedding_pooling: str = "mean"
    embedding_max_length: int = 512


class RunConfig(BaseModel):
    """Master configuration"""

    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig

    work_dir: str = "./runs/finetune_dolphin_x1"
    run_name: str = "dolphin_x1_enhanced"
    unified_model_dir: str = "./models/dolphin_x1_unified_enhanced"

    use_wandb: bool = False
    wandb_project: Optional[str] = None

    trust_remote_code: bool = False
    use_auth_token: bool = False

    save_embedding_plots: bool = True

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# System monitoring & memory management
# ---------------------------------------------------------------------------


class SystemMonitor:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info: Dict[str, Any] = {"cpu": {}, "gpu": []}

        vm = psutil.virtual_memory()
        info["cpu"] = {
            "cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage": psutil.cpu_percent(interval=0.5),
            "memory": {
                "total": vm.total / (1024**3),
                "available": vm.available / (1024**3),
                "used": vm.used / (1024**3),
                "percent": vm.percent,
            },
        }

        try:
            gpus = GPUtil.getGPUs()
            for g in gpus:
                info["gpu"].append(
                    {
                        "id": g.id,
                        "name": g.name,
                        "load": g.load * 100,
                        "memory_used": g.memoryUsed,
                        "memory_total": g.memoryTotal,
                        "memory_percent": g.memoryUtil * 100,
                        "temperature": g.temperature,
                    }
                )
        except Exception as e:  # pragma: no cover (best-effort only)
            logger.warning(f"GPU monitoring failed: {e}")

        return info

    @staticmethod
    def print_system_status() -> None:
        info = SystemMonitor.get_system_info()
        table = Table(title="System Status")
        table.add_column("Resource", style="cyan")
        table.add_column("Details", style="white")

        cpu = info["cpu"]
        table.add_row(
            "CPU",
            f"{cpu['cores']} cores, {cpu['usage']:.1f}% usage, "
            f"{cpu['memory']['used']:.1f}/{cpu['memory']['total']:.1f} GB RAM "
            f"({cpu['memory']['percent']:.1f}%)",
        )

        for gpu in info["gpu"]:
            table.add_row(
                f"GPU {gpu['id']} ({gpu['name']})",
                f"{gpu['load']:.1f}% load, {gpu['memory_used']}/{gpu['memory_total']} MB "
                f"({gpu['memory_percent']:.1f}%), {gpu['temperature']}°C",
            )

        console.print(table)

    @staticmethod
    def check_system_requirements(
        min_memory_gb: float = 16.0,
        min_gpu_memory_gb: float = 8.0,
    ) -> bool:
        """Warn if the system looks too small, but do not hard-fail."""

        info = SystemMonitor.get_system_info()

        if info["cpu"]["memory"]["total"] < min_memory_gb:
            console.print(
                f"[yellow]Total RAM is {info['cpu']['memory']['total']:.1f} GB < {min_memory_gb} GB. "
                "Training may be slow or unstable.[/yellow]"
            )

        if info["gpu"]:
            best_gpu = max(info["gpu"], key=lambda g: g["memory_total"])
            available_gb = (best_gpu["memory_total"] - best_gpu["memory_used"]) / 1024
            if available_gb < min_gpu_memory_gb:
                console.print(
                    f"[yellow]Available GPU memory is {available_gb:.1f} GB < {min_gpu_memory_gb} GB. "
                    "Consider smaller batch size / seq len.[/yellow]"
                )
        else:
            console.print(
                "[yellow]No GPU detected; training will be extremely slow.[/yellow]"
            )

        return True


@contextmanager
def memory_cleanup():
    """Context manager that aggressively cleans up GPU & CPU memory."""

    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------


class DatasetFormat(Enum):
    ALPACA = "alpaca"
    CHATML = "chatml"
    CUSTOM = "custom"


class SmartDatasetLoader:
    """Loader that auto-detects a few common formats."""

    @staticmethod
    def detect_format(sample: Dict[str, Any]) -> DatasetFormat:
        if "messages" in sample and isinstance(sample["messages"], list):
            return DatasetFormat.CHATML
        if "instruction" in sample and ("output" in sample or "response" in sample):
            return DatasetFormat.ALPACA
        return DatasetFormat.CUSTOM

    @staticmethod
    def load(
        cfg: DatasetConfig,
    ) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
        console.print("[cyan]Loading dataset...[/cyan]")

        if cfg.source_type == "local_jsonl":
            records = SmartDatasetLoader._load_local_jsonl(cfg.path, cfg.max_samples)
        elif cfg.source_type == "hf_hub":
            records = SmartDatasetLoader._load_hf_dataset(cfg)
        else:
            raise ValueError(f"Unsupported source_type: {cfg.source_type}")

        if not records:
            raise ValueError("No records loaded from dataset")

        fmt = SmartDatasetLoader.detect_format(records[0])
        console.print(f"[green]Detected dataset format: {fmt.value}[/green]")

        formatted = SmartDatasetLoader._convert_to_training_format(records, fmt, cfg)

        val_data: Optional[List[Dict[str, str]]] = None
        if cfg.validation_size > 0 and len(formatted) > 1:
            split_idx = int(len(formatted) * (1 - cfg.validation_size))
            val_data = formatted[split_idx:]
            formatted = formatted[:split_idx]
            console.print(
                f"[green]Dataset split into {len(formatted)} train / {len(val_data)} validation[/green]"
            )
        else:
            console.print(
                f"[green]Using {len(formatted)} samples for training (no validation split).[/green]"
            )

        return formatted, val_data

    @staticmethod
    def _load_local_jsonl(
        path: Optional[str], max_samples: Optional[int]
    ) -> List[Dict[str, Any]]:
        if not path:
            raise ValueError("dataset.path must be set for local_jsonl source_type")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {i + 1}: {e}")
        console.print(f"[green]Loaded {len(records)} records from {path}[/green]")
        return records

    @staticmethod
    def _load_hf_dataset(cfg: DatasetConfig) -> List[Dict[str, Any]]:
        if not cfg.hf_name:
            raise ValueError("dataset.hf_name must be set for hf_hub source_type")

        ds = load_dataset(cfg.hf_name, cfg.hf_subset, split=cfg.split)
        if cfg.max_samples is not None:
            ds = ds.select(range(cfg.max_samples))
        console.print(
            f"[green]Loaded {len(ds)} records from HF dataset {cfg.hf_name}[/green]"
        )
        return list(ds)

    @staticmethod
    def _convert_to_training_format(
        records: List[Dict[str, Any]],
        fmt: DatasetFormat,
        cfg: DatasetConfig,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for rec in records:
            try:
                if fmt == DatasetFormat.ALPACA:
                    prompt, response = SmartDatasetLoader._convert_alpaca(rec, cfg)
                elif fmt == DatasetFormat.CHATML:
                    prompt, response = SmartDatasetLoader._convert_chatml(rec)
                else:
                    prompt, response = SmartDatasetLoader._convert_custom(rec, cfg)
                if prompt and response:
                    out.append({"prompt": prompt, "response": response})
            except Exception as e:  # pragma: no cover (robustness only)
                logger.warning(f"Failed to convert record: {e}")
        console.print(f"[green]Prepared {len(out)} training examples.[/green]")
        return out

    @staticmethod
    def _convert_alpaca(rec: Dict[str, Any], cfg: DatasetConfig) -> Tuple[str, str]:
        instruction = rec.get(cfg.input_field, "").strip()
        output = rec.get(cfg.output_field, rec.get("response", "")).strip()
        if not instruction or not output:
            return "", ""
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt, output

    @staticmethod
    def _convert_chatml(rec: Dict[str, Any]) -> Tuple[str, str]:
        messages = rec.get("messages", [])
        if len(messages) < 2:
            return "", ""
        prompt_parts: List[str] = []
        for msg in messages[:-1]:
            role = msg.get("role", "user").title()
            content = msg.get("content", "").strip()
            prompt_parts.append(f"{role}: {content}")
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        response = messages[-1].get("content", "").strip()
        return prompt, response

    @staticmethod
    def _convert_custom(rec: Dict[str, Any], cfg: DatasetConfig) -> Tuple[str, str]:
        inp = rec.get(cfg.input_field, "").strip()
        out = rec.get(cfg.output_field, "").strip()
        if not inp or not out:
            return "", ""
        prompt = f"### Instruction:\n{inp}\n\n### Response:\n"
        return prompt, out


# ---------------------------------------------------------------------------
# Enhanced Trainer (Unsloth + TRL)
# ---------------------------------------------------------------------------


class EnhancedTrainer:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self) -> str:
        console.print(
            Panel.fit(
                f"[bold]Starting Enhanced Training for {self.cfg.model.base_model}[/bold]",
                title="TRAINING",
                border_style="cyan",
            )
        )

        SystemMonitor.check_system_requirements()
        SystemMonitor.print_system_status()

        train_data, val_data = SmartDatasetLoader.load(self.cfg.dataset)
        if not train_data:
            raise ValueError("No training data available")

        model, tokenizer = self._prepare_model()
        training_args = self._create_training_args(has_validation=val_data is not None)
        trainer = self._create_trainer(
            model, tokenizer, train_data, val_data, training_args
        )

        self._train_with_progress(trainer)
        save_dir = self._save_model(trainer, tokenizer)
        return save_dir

    def _prepare_model(self):
        console.print("[cyan]Loading base model with Unsloth...[/cyan]")

        if self.cfg.model.model_dtype == "bfloat16" and is_bfloat16_supported():
            dtype = torch.bfloat16
        elif self.cfg.model.model_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model.base_model,
            max_seq_length=self.cfg.model.max_seq_len,
            dtype=dtype,
            load_in_4bit=self.cfg.model.quantize_4bit,
            token=self.cfg.use_auth_token,
            trust_remote_code=self.cfg.trust_remote_code,
            device_map="auto",
        )

        FastLanguageModel.for_training(model)

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.cfg.training.lora_r,
            target_modules=self.cfg.training.lora_target_modules,
            lora_alpha=self.cfg.training.lora_alpha,
            lora_dropout=self.cfg.training.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.cfg.model.use_gradient_checkpointing,
            random_state=3407,
        )

        return model, tokenizer

    def _create_training_args(self, has_validation: bool) -> TrainingArguments:
        out_dir = os.path.join(self.cfg.work_dir, self.cfg.run_name)
        os.makedirs(out_dir, exist_ok=True)

        return TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=self.cfg.training.num_epochs,
            per_device_train_batch_size=self.cfg.training.batch_size,
            per_device_eval_batch_size=self.cfg.training.batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            warmup_ratio=self.cfg.training.warmup_ratio,
            weight_decay=self.cfg.training.weight_decay,
            max_grad_norm=self.cfg.training.max_grad_norm,
            logging_steps=50,
            eval_steps=self.cfg.training.eval_steps if has_validation else None,
            save_steps=self.cfg.training.save_steps,
            save_total_limit=self.cfg.training.save_total_limit,
            evaluation_strategy="steps" if has_validation else "no",
            load_best_model_at_end=has_validation,
            metric_for_best_model="eval_loss" if has_validation else None,
            greater_is_better=False,
            report_to="wandb" if self.cfg.use_wandb else None,
            run_name=self.cfg.run_name if self.cfg.use_wandb else None,
            bf16=self.cfg.model.model_dtype == "bfloat16",
            fp16=self.cfg.model.model_dtype == "float16",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

    def _create_trainer(
        self,
        model,
        tokenizer,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]],
        training_args: TrainingArguments,
    ) -> SFTTrainer:
        class TRLDataset(Dataset):
            def __init__(self, data: List[Dict[str, str]]):
                self.data = data

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> Dict[str, str]:
                rec = self.data[idx]
                return {"text": rec["prompt"] + rec["response"]}

        train_ds = TRLDataset(train_data)
        eval_ds = TRLDataset(val_data) if val_data else None

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            packing=True,
            max_seq_length=self.cfg.model.max_seq_len,
        )
        return trainer

    def _train_with_progress(self, trainer: SFTTrainer) -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=None)

            class ProgressCallback:
                def __init__(self, progress_obj: Progress, task_id: int) -> None:
                    self.progress = progress_obj
                    self.task = task_id

                def __call__(self, args, state, control, **kwargs) -> None:
                    if state.is_local_process_zero and state.max_steps:
                        self.progress.update(
                            self.task,
                            description=f"Training [Step {state.global_step}/{state.max_steps}]",
                            completed=state.global_step,
                            total=state.max_steps,
                        )

            trainer.add_callback(ProgressCallback(progress, task))
            trainer.train()

        console.print("[green]Training completed successfully![/green]")

    def _save_model(self, trainer: SFTTrainer, tokenizer) -> str:
        out_dir = os.path.join(self.cfg.work_dir, self.cfg.run_name, "final_model")
        os.makedirs(out_dir, exist_ok=True)
        console.print(f"[cyan]Saving model to {out_dir}...[/cyan]")

        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

        with open(
            os.path.join(out_dir, "training_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.cfg.dict(), f, indent=2)

        self._create_model_card(out_dir)

        console.print(
            Panel.fit(
                f"[bold]Model saved to:[/bold]\n{out_dir}",
                title="TRAINING COMPLETE",
                border_style="green",
            )
        )
        return out_dir

    def _create_model_card(self, out_dir: str) -> None:
        content = f"""---
license: apache-2.0
base_model: {self.cfg.model.base_model}
tags:
- unified-model
- text-generation
- text-embedding
- llm2vec
- unsloth
- fine-tuned
---

# Unified Dolphin-X1 Model

Fine-tuned from `{self.cfg.model.base_model}` with LoRA and Unsloth.

## Training Configuration

```json
{json.dumps(self.cfg.dict(), indent=2)}
```

## Capabilities

- Text generation with internal reasoning loop
- Text embeddings via LLM2Vec
- 4-bit quantization support
"""
        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Unified model (generation + embeddings + reasoning)
# ---------------------------------------------------------------------------


class EnhancedUnifiedModel:
    def __init__(self, model_dir: str, cfg: RunConfig) -> None:
        self.model_dir = model_dir
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        console.print(
            Panel.fit(
                f"[bold]Loading Enhanced Unified Model from {model_dir}[/bold]",
                title="MODEL LOADING",
                border_style="blue",
            )
        )

        self.tokenizer = self._load_tokenizer()
        self.gen_model = self._load_generation_model()
        self.embed_model = self._load_embedding_model()

        console.print("[green]✓ Unified model loaded successfully[/green]")

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_dir,
            use_fast=True,
            trust_remote_code=self.cfg.trust_remote_code,
            padding_side="left",
        )

    def _load_generation_model(self):
        console.print("[cyan]Loading generation model...[/cyan]")

        if self.cfg.model.quantize_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.cfg.model.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(
                    torch, self.cfg.model.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=self.cfg.model.bnb_4bit_use_double_quant,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=self.cfg.trust_remote_code,
                torch_dtype=torch.float16,
            )
        else:
            dtype = getattr(torch, self.cfg.model.model_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=self.cfg.trust_remote_code,
            )

        model.eval()
        return model

    def _load_embedding_model(self):
        console.print("[cyan]Loading embedding (LLM2Vec) model...[/cyan]")
        return LLM2Vec.from_pretrained(
            self.model_dir,
            peft_model_name_or_path=self.model_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            pooling_mode=self.cfg.inference.embedding_pooling,
            max_length=self.cfg.inference.embedding_max_length,
        )

    @memory_cleanup
    def generate(self, prompt: str, **kwargs) -> str:
        gen_cfg: Dict[str, Any] = {
            "max_new_tokens": self.cfg.inference.max_new_tokens,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
            "top_k": self.cfg.inference.top_k,
            "repetition_penalty": self.cfg.inference.repetition_penalty,
            "do_sample": self.cfg.inference.do_sample,
            "num_beams": self.cfg.inference.num_beams,
        }
        gen_cfg.update(kwargs)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gen_model.generate(
                **inputs,
                **gen_cfg,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text

    @memory_cleanup
    def generate_with_reasoning(self, user_input: str) -> str:
        steps = self.cfg.inference.reasoning_steps
        if steps <= 0:
            return self.generate(user_input)

        console.print("[dim]Engaging internal reasoning...[/dim]")
        thoughts: List[str] = []
        for step in range(steps):
            reasoning_prompt = (
                "You are thinking through a problem step by step. "
                "Analyse the user's query and break down your reasoning.\n\n"
                f"User Query: {user_input}\n\n"
                f"Reasoning Step {step + 1}/{steps}:\n"
            )
            thought = self.generate(
                reasoning_prompt,
                max_new_tokens=self.cfg.inference.reasoning_max_tokens,
                temperature=0.3,
                do_sample=True,
            )
            thoughts.append(thought.strip())
            console.print(f"[dim]Step {step + 1}: {thought[:100]}...[/dim]")

        reasoning_block = "\n".join(
            [f"Thought {i + 1}: {t}" for i, t in enumerate(thoughts)]
        )
        final_prompt = (
            "You have reasoned internally as follows:\n"
            f"{reasoning_block}\n\n"
            "Based on this reasoning, provide your final helpful answer to the user.\n\n"
            f"User: {user_input}\nAssistant:"
        )
        return self.generate(final_prompt)

    @memory_cleanup
    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.embed_model.encode(texts)
        return np.asarray(emb, dtype="float32")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "device": self.device,
            "generation_params": {
                "max_new_tokens": self.cfg.inference.max_new_tokens,
                "temperature": self.cfg.inference.temperature,
                "reasoning_steps": self.cfg.inference.reasoning_steps,
            },
            "embedding_params": {
                "pooling": self.cfg.inference.embedding_pooling,
                "max_length": self.cfg.inference.embedding_max_length,
            },
        }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


class AdvancedVisualizer:
    @staticmethod
    def plot_embeddings_3d(
        embeddings: np.ndarray,
        labels: List[str],
        title: str,
        save_path: str,
    ) -> None:
        if embeddings.shape[0] < 3:
            console.print("[yellow]Need at least 3 points for 3D plot.[/yellow]")
            return

        pca = PCA(n_components=3)
        coords = pca.fit_transform(embeddings)

        # Avoid import errors at top-level in some environments
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=range(len(coords)),
            cmap="viridis",
            s=80,
            alpha=0.8,
        )

        for i, label in enumerate(labels):
            ax.text(
                coords[i, 0],
                coords[i, 1],
                coords[i, 2],
                label,
                fontsize=8,
                ha="center",
                va="center",
            )

        ax.set_title(f"{title} (PCA 3D Projection)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.colorbar(scatter, ax=ax, label="Index")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        console.print(f"[green]3D embedding plot saved to {save_path}[/green]")


# ---------------------------------------------------------------------------
# REPL session
# ---------------------------------------------------------------------------


class REPLSession:
    def __init__(self, unified_model: EnhancedUnifiedModel, cfg: RunConfig) -> None:
        self.model = unified_model
        self.cfg = cfg
        self.history: List[Tuple[str, str, int]] = []
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def start(self) -> None:
        console.print(
            Panel.fit(
                "[bold]Enhanced Dolphin-X1 Unified Assistant[/bold]\n"
                f"Session ID: {self.session_id}\n"
                "Type /help for available commands.",
                title="REPL SESSION",
                border_style="magenta",
            )
        )

        while True:
            try:
                user_input = console.input("[bold cyan]>>>[/bold cyan] ").strip()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                else:
                    self._process_input(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit or /quit to end session.[/yellow]")
            except Exception as e:  # pragma: no cover (interactive)
                logger.exception("REPL error")
                console.print(f"[red]Error: {e}[/red]")

    def _handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()
        if cmd in ("/exit", "/quit"):
            console.print("[cyan]Ending session. Goodbye![/cyan]")
            return True

        if cmd == "/help":
            self._show_help()
        elif cmd == "/history":
            self._show_history()
        elif cmd == "/clear":
            self.history.clear()
            console.print("[green]History cleared.[/green]")
        elif cmd == "/status":
            self._show_status()
        elif cmd == "/export":
            self._export_session()
        elif cmd.startswith("/reasoning"):
            self._toggle_reasoning(cmd)
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Type /help for available commands.")
        return False

    def _show_help(self) -> None:
        help_text = """
[bold]Commands:[/bold]
[cyan]/help[/cyan]        Show this help message
[cyan]/history[/cyan]     Show conversation history summary
[cyan]/clear[/cyan]       Clear conversation history
[cyan]/status[/cyan]      Show model & system status
[cyan]/export[/cyan]      Export session to JSON
[cyan]/reasoning on|off|N[/cyan]  Toggle reasoning, or set #steps (0–10)
[cyan]/exit[/cyan], [cyan]/quit[/cyan]  End session
"""
        console.print(Panel.fit(help_text, title="HELP", border_style="blue"))

    def _show_history(self) -> None:
        if not self.history:
            console.print("[yellow]No history yet.[/yellow]")
            return

        table = Table(title=f"Conversation History (Session {self.session_id})")
        table.add_column("Turn", style="cyan")
        table.add_column("User", style="magenta")
        table.add_column("Assistant", style="green")
        table.add_column("Tokens (approx.)", style="yellow")

        for i, (user, assistant, tokens) in enumerate(self.history):
            u = user if len(user) <= 60 else user[:57] + "..."
            a = assistant if len(assistant) <= 60 else assistant[:57] + "..."
            table.add_row(str(i + 1), u, a, str(tokens))

        console.print(table)

    def _show_status(self) -> None:
        SystemMonitor.print_system_status()
        info = self.model.get_model_info()
        table = Table(title="Model Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model dir", info["model_dir"])
        table.add_row("Device", info["device"])
        for k, v in info["generation_params"].items():
            table.add_row(f"Generation.{k}", str(v))
        for k, v in info["embedding_params"].items():
            table.add_row(f"Embedding.{k}", str(v))

        console.print(table)

    def _export_session(self) -> None:
        export = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "history": self.history,
            "config": self.cfg.dict(),
        }
        filename = f"session_{self.session_id}_{int(time.time())}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        console.print(f"[green]Session exported to {filename}[/green]")

    def _toggle_reasoning(self, command: str) -> None:
        parts = command.split()
        if len(parts) == 1:
            console.print(
                f"[cyan]Current reasoning steps: {self.cfg.inference.reasoning_steps}[/cyan]"
            )
            return

        arg = parts[1].lower()
        if arg == "on":
            self.cfg.inference.reasoning_steps = 3
            console.print("[green]Reasoning enabled (3 steps).[/green]")
        elif arg == "off":
            self.cfg.inference.reasoning_steps = 0
            console.print("[green]Reasoning disabled.[/green]")
        else:
            try:
                steps = int(arg)
                if 0 <= steps <= 10:
                    self.cfg.inference.reasoning_steps = steps
                    console.print(f"[green]Reasoning steps set to {steps}.[/green]")
                else:
                    console.print(
                        "[red]Reasoning steps must be between 0 and 10.[/red]"
                    )
            except ValueError:
                console.print("[red]Invalid argument for /reasoning.[/red]")

    def _process_input(self, user_input: str) -> None:
        start = time.time()
        if self.cfg.inference.reasoning_steps > 0:
            response = self.model.generate_with_reasoning(user_input)
        else:
            response = self.model.generate(user_input)
        elapsed = time.time() - start
        approx_tokens = len(user_input.split()) + len(response.split())
        self.history.append((user_input, response, approx_tokens))

        console.print(
            Panel(
                response,
                title=(
                    f"[bold green]Assistant[/bold green] "
                    f"({elapsed:.2f}s, ~{approx_tokens} tokens)"
                ),
                border_style="green",
            )
        )

        if self.cfg.save_embedding_plots and len(self.history) >= 3:
            self._update_embeddings_plot()

    def _update_embeddings_plot(self) -> None:
        try:
            recent = [h[0] for h in self.history[-6:]]
            if len(recent) < 3:
                return
            emb = self.model.encode(recent)
            labels = [f"U{i}" for i in range(len(recent))]
            path = f"embeddings_session_{self.session_id}.png"
            AdvancedVisualizer.plot_embeddings_3d(
                emb,
                labels,
                "User Input Embeddings",
                path,
            )
        except Exception as e:  # pragma: no cover (visualisation is optional)
            logger.warning(f"Embedding plot update failed: {e}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def create_default_config() -> RunConfig:
    dataset_cfg = DatasetConfig(
        source_type="local_jsonl",
        path="./data/training.jsonl",
        input_field="instruction",
        output_field="output",
        max_samples=1000,
        validation_size=0.1,
    )

    training_cfg = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=4,
        num_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    model_cfg = ModelConfig(
        base_model="dphn/Dolphin-X1-8B",
        max_seq_len=2048,
        model_dtype="bfloat16",
        quantize_4bit=True,
    )

    inference_cfg = InferenceConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        reasoning_steps=2,
        reasoning_max_tokens=128,
    )

    return RunConfig(
        dataset=dataset_cfg,
        training=training_cfg,
        model=model_cfg,
        inference=inference_cfg,
        work_dir="./runs/finetune_dolphin_x1",
        run_name="dolphin_x1_enhanced",
        unified_model_dir="./models/dolphin_x1_unified_enhanced",
    )


def load_config_from_file(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_cfg = DatasetConfig(**data.get("dataset", {}))
    training_cfg = TrainingConfig(**data.get("training", {}))
    model_cfg = ModelConfig(**data.get("model", {}))
    inference_cfg = InferenceConfig(**data.get("inference", {}))

    extra = {
        k: v
        for k, v in data.items()
        if k not in ["dataset", "training", "model", "inference"]
    }

    return RunConfig(
        dataset=dataset_cfg,
        training=training_cfg,
        model=model_cfg,
        inference=inference_cfg,
        **extra,
    )


def save_config_to_file(cfg: RunConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhanced Unified Dolphin-X1 Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train with custom config:
    python build_unified_dolphin_x1_enhanced.py --train --config config.json

  Use existing model with REPL:
    python build_unified_dolphin_x1_enhanced.py --model-dir ./models/my_model

  Quick start with defaults:
    python build_unified_dolphin_x1_enhanced.py --train --quick
""",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training before building unified model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Use existing model directory (skip training)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick settings for faster experimentation",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Skip interactive REPL",
    )
    parser.add_argument(
        "--export-config",
        type=str,
        default=None,
        help="Export default configuration to file and exit",
    )

    args = parser.parse_args()

    # Export default config and exit
    if args.export_config:
        cfg = create_default_config()
        save_config_to_file(cfg, args.export_config)
        console.print(
            f"[green]Default configuration exported to {args.export_config}[/green]"
        )
        return

    # Load configuration
    if args.config:
        cfg = load_config_from_file(args.config)
        console.print(f"[green]Loaded configuration from {args.config}[/green]")
    else:
        cfg = create_default_config()
        console.print("[yellow]Using default configuration.[/yellow]")

    # Quick mode tweaks
    if args.quick:
        cfg.training.num_epochs = 1
        if cfg.dataset.max_samples is None:
            cfg.dataset.max_samples = 200
        else:
            cfg.dataset.max_samples = min(cfg.dataset.max_samples, 200)
        cfg.inference.reasoning_steps = 1
        console.print("[cyan]Applied quick settings for faster experimentation.[/cyan]")

    console.print(
        Panel.fit(
            "[bold]Enhanced Unified Dolphin-X1 Builder[/bold]\n"
            "Production-grade fine-tuning with unified encode/decode.",
            title="🚀 ENHANCED BUILDER",
            border_style="cyan",
        )
    )

    model_dir = args.model_dir

    # Training phase
    if args.train:
        console.print(
            Panel.fit("[bold]Starting Training Phase[/bold]", border_style="green")
        )
        try:
            trainer = EnhancedTrainer(cfg)
            model_dir = trainer.train()
        except Exception as e:  # pragma: no cover (runtime)
            console.print(f"[red]Training failed: {e}[/red]")
            logger.exception("Training error")
            return

    # Model directory resolution
    if not model_dir:
        if os.path.isdir(cfg.unified_model_dir):
            model_dir = cfg.unified_model_dir
        else:
            console.print(
                "[red]No model directory specified and unified_model_dir does not exist.[/red]"
            )
            console.print(
                "Use --train to train a model or --model-dir to point to an existing model."
            )
            return

    # Load unified model + start REPL
    try:
        console.print(
            Panel.fit("[bold]Loading Unified Model[/bold]", border_style="blue")
        )
        unified_model = EnhancedUnifiedModel(model_dir, cfg)

        summary = Table(title="Final Configuration Summary")
        summary.add_column("Component", style="cyan")
        summary.add_column("Key Settings", style="white")

        summary.add_row(
            "Dataset",
            f"Source={cfg.dataset.source_type}, "
            f"Samples={cfg.dataset.max_samples or 'all'}",
        )
        summary.add_row(
            "Training",
            f"Epochs={cfg.training.num_epochs}, LR={cfg.training.learning_rate}, "
            f"LoRA r={cfg.training.lora_r}",
        )
        summary.add_row(
            "Model",
            f"Base={cfg.model.base_model}, SeqLen={cfg.model.max_seq_len}, "
            f"4bit={cfg.model.quantize_4bit}",
        )
        summary.add_row(
            "Inference",
            f"Reasoning={cfg.inference.reasoning_steps} steps, "
            f"MaxTokens={cfg.inference.max_new_tokens}, Temp={cfg.inference.temperature}",
        )
        console.print(summary)

        if not args.no_repl:
            console.print(
                Panel.fit("[bold]Starting Enhanced REPL[/bold]", border_style="magenta")
            )
            session = REPLSession(unified_model, cfg)
            session.start()
        else:
            console.print("[yellow]REPL skipped (--no-repl).[/yellow]")

    except Exception as e:  # pragma: no cover (runtime)
        console.print(f"[red]Model loading failed: {e}[/red]")
        logger.exception("Model loading error")
        return

    console.print(
        Panel.fit(
            "[bold]Enhanced Unified Builder Completed Successfully![/bold]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
