#!/usr/bin/env python3
"""
monGARS multi-stage pipeline
============================

This module contains a set of helper functions and a command-line interface to
execute the full workflow needed to build, fine-tune and export large language
models for the monGARS project.  It mirrors the high-level sequence described
in the project documentation:

1. **Dataset Generation** – create multilingual French datasets covering
   instruction, reasoning, conversation and retrieval/embedding tasks.  This
   stage calls the consolidated French dataset pipeline provided by the
   repository and also runs monGARS internal static analysis to generate
   module-specific instruction datasets.

2. **Supervised Fine-tuning (SFT)** – fine-tune a base model on the
   instruction/reasoning/conversation datasets using QLoRA via Unsloth.

3. **LLM2Vec Adaptation** – apply masked next token prediction (MNTP) and
   SimCSE contrastive learning to convert the fine-tuned decoder-only model
   into a strong text encoder using the retrieval/embedding dataset.

4. **Retrieval Augmented Fine-tuning (RAFT) [Optional]** – optionally run
   retrieval-augmented supervised fine-tuning where the model sees retrieved
   context alongside the prompt.  This stage is left unimplemented here but
   provided as a hook for future work.

5. **Export and Deployment** – merge LoRA adapters into full weights, export
   to GGUF (or other formats) and generate wrapper bundles for deployment.

This script can be executed as a whole to run the pipeline end to end, or
individual stages can be invoked via CLI flags.  It is intentionally
structured to allow resumption and re-use of intermediate artifacts.

Example usage
-------------

Generate datasets and perform SFT, LLM2Vec and export in a single run::

    python mongars_multistage_pipeline.py \
        --run-id myexperiment \
        --repo-root /path/to/monGARS \
        --base-model mistralai/Mistral-7B-v0.2 \
        --val-split 0.06 \
        --mntp-config path/to/mntp_config.json \
        --simcse-config path/to/simcse_config.json \
        --llm2vec-root /path/to/llm2vec \
        --runs-root runs

To run only dataset generation::

    python mongars_multistage_pipeline.py --run-id myexperiment --build-datasets

To fine-tune modules after datasets exist::

    python mongars_multistage_pipeline.py --run-id myexperiment --sft

See ``--help`` for full argument listing.

Notes
-----

This script depends on monGARS being importable on the Python path (e.g.
using ``pip install -e .`` or running from the repository root).  It also
assumes that the consolidated French dataset pipeline script lives in
``scripts/consolidated_french_dataset_pipeline.py`` relative to the repository
root, and that the LLM2Vec MNTP and SimCSE scripts live under
``experiments/`` in the LLM2Vec repository.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from monGARS.mlops.artifacts import WrapperConfig, write_wrapper_bundle
from monGARS.mlops.code_analysis import (
    scan_llm_usage,
    scan_module_interactions,
)
from monGARS.mlops.dataset import (
    build_module_interaction_dataset,
    build_mongars_strategy_dataset,
    prepare_local_instruction_dataset,
)
from monGARS.mlops.exporters import export_gguf
from monGARS.mlops.pipelines.unsloth import run_unsloth_finetune

LOGGER = logging.getLogger("mongars_multistage_pipeline")


def _stringify_paths(value: Any) -> Any:
    """Recursively convert :class:`pathlib.Path` objects to strings for JSON serialization."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    return value


def save_state(path: Path, state: Dict[str, Any]) -> None:
    """Persist pipeline state to disk with path values converted to strings."""

    serializable_state = _stringify_paths(state)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_state, f, indent=2)


def run_subprocess(
    cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> None:
    """Execute a command via subprocess and raise on failure."""

    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
    LOGGER.info("Running command: %s", cmd_str)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    result = subprocess.run(cmd, cwd=cwd, env=proc_env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {cmd_str}"
        )


def build_french_multitask_datasets(
    repo_root: Path,
    output_dir: Path,
    languages: List[str],
    max_examples: Optional[int] = None,
    trust_remote_code: bool = True,
    workers: int = 4,
) -> Dict[str, Path]:
    """
    Invoke the consolidated French dataset pipeline to produce multiple task datasets.

    Parameters
    ----------
    repo_root: Path
        Root of the monGARS repository where the ``scripts`` directory lives.
    output_dir: Path
        Directory into which the dataset pipeline will write its outputs.  It will
        contain subdirectories for instruction, reasoning, conversation and retrieval datasets.
    languages: list of str
        Languages to include in the dataset (e.g. ["fr"]).  Only French is supported
        in the current pipeline, but this parameter allows for extension.
    max_examples: int, optional
        Maximum number of examples to draw from each source (for debugging/testing).
    trust_remote_code: bool
        Passed through to the dataset pipeline to allow remote code execution when
        loading HuggingFace datasets.
    workers: int
        Number of worker processes for dataset loading.

    Returns
    -------
    Dict[str, Path]
        A mapping from dataset type (e.g. "instruction", "reasoning", "conversation", "retrieval")
        to the path of its JSONL file in the output directory.

    Notes
    -----
    This helper function constructs a subprocess command that calls the
    ``consolidated_french_dataset_pipeline.py`` script residing in
    ``scripts/`` under ``repo_root``.  If you wish to customise the
    pipeline further, modify the command line arguments accordingly.
    """
    pipeline_script = repo_root / "scripts" / "consolidated_french_dataset_pipeline.py"
    if not pipeline_script.is_file():
        raise FileNotFoundError(
            f"Could not find consolidated_french_dataset_pipeline.py at {pipeline_script}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--output-dir",
        str(output_dir),
        "--languages",
        ",".join(languages),
        "--trust-remote-code" if trust_remote_code else "--no-trust-remote-code",
        "--workers",
        str(workers),
    ]
    if max_examples is not None:
        cmd += ["--max-examples", str(max_examples)]
    run_subprocess(cmd, cwd=repo_root)

    # Collect resulting files.  The pipeline writes files with predictable names.
    datasets = {}
    for task in ["instruction", "reasoning", "conversation", "retrieval"]:
        jsonl_path = output_dir / f"{task}.jsonl"
        if jsonl_path.is_file():
            datasets[task] = jsonl_path
        else:
            LOGGER.warning(
                "Expected dataset %s.jsonl not found in %s", task, output_dir
            )
    return datasets


def build_internal_instruction_datasets(
    repo_root: Path,
    output_dir: Path,
    val_split: float = 0.06,
    split_seed: int = 42,
    min_val_examples: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Build monGARS internal datasets by statically analysing the codebase.

    This function scans for LLM call-sites and module interactions, builds
    strategy and interaction datasets, then splits each into train/val sets.

    Parameters
    ----------
    repo_root: Path
        Root of the monGARS codebase to analyse.
    output_dir: Path
        Directory into which per-module subdirectories and JSONL files will be written.
    val_split: float
        Fraction of examples to reserve for validation.  Must be in [0, 1).
    split_seed: int
        Seed for the random shuffling used during splitting.
    min_val_examples: int
        Minimum number of validation examples to allocate per module (if available).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A mapping from module name to a dictionary containing paths and counts for
        the training and validation files.  The returned structure matches that
        produced by the interactive pipeline’s ``build_datasets`` helper.
    """
    LOGGER.info("Scanning repository for LLM usage...")
    usages = scan_llm_usage(repo_root)
    LOGGER.info("Found %d LLM call-sites", len(usages))

    LOGGER.info("Scanning repository for module interactions...")
    interactions = scan_module_interactions(repo_root)
    LOGGER.info("Found %d module interactions", len(interactions))

    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = tmp_dir / "strategy_data.jsonl"
    interaction_path = tmp_dir / "interaction_data.jsonl"

    build_mongars_strategy_dataset(usages, interactions, strategy_path)
    build_module_interaction_dataset(interactions, interaction_path)

    # Load and group by [MOD=module] annotation
    records: List[Dict[str, Any]] = []
    for path in [strategy_path, interaction_path]:
        for rec in prepare_local_instruction_dataset(path):
            records.append(rec)

    examples_by_module: Dict[str, List[Dict[str, str]]] = {}
    for rec in records:
        instr: str = rec["instruction"]
        if instr.startswith("[MOD="):
            end = instr.find("]")
            module = instr[5:end]
            payload = {
                "instruction": instr[end + 2 :],
                "input": rec.get("input", ""),
                "output": rec["output"],
            }
            examples_by_module.setdefault(module, []).append(payload)
        else:
            # Records without a [MOD=...] tag are grouped under "General"
            examples_by_module.setdefault("General", []).append(rec)

    random.seed(split_seed)
    module_info: Dict[str, Dict[str, Any]] = {}
    for module, recs in examples_by_module.items():
        if not recs:
            continue
        random.shuffle(recs)
        total = len(recs)
        n_val = max(min_val_examples, int(total * val_split)) if total > 1 else 0
        n_val = min(n_val, total)
        val_recs = recs[:n_val]
        train_recs = recs[n_val:]
        mod_dir = output_dir / module.lower()
        mod_dir.mkdir(parents=True, exist_ok=True)
        train_path = mod_dir / "train.jsonl"
        val_path = mod_dir / "val.jsonl"
        # Write train and validation
        with train_path.open("w", encoding="utf-8") as tf:
            for ex in train_recs:
                tf.write(json.dumps(ex, ensure_ascii=False) + "\n")
        with val_path.open("w", encoding="utf-8") as vf:
            for ex in val_recs:
                vf.write(json.dumps(ex, ensure_ascii=False) + "\n")
        module_info[module] = {
            "train_path": train_path,
            "val_path": val_path,
            "n_train": len(train_recs),
            "n_val": len(val_recs),
        }
        LOGGER.info(
            "Module %s → %d train examples, %d val examples",
            module,
            len(train_recs),
            len(val_recs),
        )
    # Clean up temporaries
    try:
        strategy_path.unlink(missing_ok=True)
        interaction_path.unlink(missing_ok=True)
        tmp_dir.rmdir()
    except Exception:
        pass
    return module_info


def perform_sft(
    modules: Dict[str, Dict[str, Any]],
    run_dir: Path,
    base_model: str,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    epochs: float = 3.0,
    batch_size: int = 64,
    vram_budget_mb: int = 24000,
    hf_token: Optional[str] = None,
) -> None:
    """
    Supervised fine-tune each monGARS module on its instruction dataset.

    Parameters
    ----------
    modules: dict
        Mapping of module names to dataset information as returned by
        ``build_internal_instruction_datasets``.
    run_dir: Path
        Directory under which per-module outputs will be created.
    base_model: str
        HuggingFace model identifier to use as the base for fine-tuning.
    lora_r, lora_alpha, lora_dropout: hyperparameters for LoRA.
    epochs: float
        Number of epochs to train for (can be fractional for early stopping).
    batch_size: int
        Batch size per device.
    vram_budget_mb: int
        Budget for GPU memory; passed through to Unsloth.
    hf_token: str, optional
        Optional HuggingFace token for private models.

    This function writes LoRA adapters and wrapper bundles to ``run_dir/<module>``.
    The resulting adapter directories are stored back into the ``modules`` mapping
    for later stages.
    """
    for module, info in modules.items():
        LOGGER.info("Starting SFT for module %s", module)
        mod_dir = run_dir / module
        mod_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = mod_dir / "chat_lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        train_path = Path(info["train_path"]).expanduser()
        val_path = Path(info["val_path"]).expanduser()
        dataset_spec = {
            "train": str(train_path),
            "validation": str(val_path),
        }
        run_unsloth_finetune(
            model_id=base_model,
            dataset=dataset_spec,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            epochs=epochs,
            train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_dataset=None,
            skip_smoke_test=True,
            output_dir=str(adapter_dir),
            merge_full_weights=True,
            update_manifest=False,
            save_merged=True,
            eval_steps=0,
            warmup_ratio=0.1,
            lr=2e-4,
            weight_decay=0.01,
            max_seq_len=1024,
            vram_budget_mb=vram_budget_mb,
            hf_token=hf_token,
        )
        LOGGER.info("Finished SFT for module %s", module)
        # Write wrapper bundle for the chat model
        wrapper_cfg = WrapperConfig(
            base_model_id=base_model,
            lora_dir=adapter_dir,
            max_seq_len=1024,
            vram_budget_mb=vram_budget_mb,
            offload_dir=mod_dir / "offload",
            activation_buffer_mb=1024,
        )
        write_wrapper_bundle(wrapper_cfg, mod_dir)
        info["adapter_dir"] = str(adapter_dir)
        info["current_model_dir"] = str(adapter_dir)


def perform_llm2vec(
    modules: Dict[str, Dict[str, Any]],
    run_dir: Path,
    llm2vec_root: Path,
    mntp_config: Path,
    simcse_config: Path,
) -> None:
    """
    Run LLM2Vec adaptation (MNTP then SimCSE) on each fine-tuned module.

    Parameters
    ----------
    modules: dict
        Mapping of module names to dataset/adaptation information.  The mapping
        will be updated in place with ``current_model_dir`` pointing to the
        SimCSE-adapted model directory.
    run_dir: Path
        Directory containing per-module subdirectories.
    llm2vec_root: Path
        Root of the LLM2Vec repository (contains ``experiments/run_mntp.py`` and
        ``experiments/run_simcse.py`` scripts).
    mntp_config: Path
        JSON config file for MNTP adaptation.
    simcse_config: Path
        JSON config file for SimCSE adaptation.
    """
    mntp_script = llm2vec_root / "experiments" / "run_mntp.py"
    simcse_script = llm2vec_root / "experiments" / "run_simcse.py"
    if not mntp_script.is_file() or not simcse_script.is_file():
        raise FileNotFoundError(
            f"Could not locate LLM2Vec scripts at {mntp_script} and {simcse_script}."
        )
    for module, info in modules.items():
        model_dir = info.get("current_model_dir")
        if not model_dir:
            LOGGER.warning("No model dir for module %s; skipping LLM2Vec", module)
            continue
        model_dir_path = Path(model_dir).expanduser()
        mod_dir = run_dir / module
        LOGGER.info("Running MNTP for module %s", module)
        mntp_out = mod_dir / "mntp"
        mntp_out.mkdir(parents=True, exist_ok=True)
        env = {
            "LLM2VEC_BASE_MODEL_DIR": str(model_dir_path),
            "LLM2VEC_OUTPUT_DIR": str(mntp_out),
        }
        run_subprocess(
            [sys.executable, str(mntp_script), str(mntp_config)],
            cwd=llm2vec_root,
            env=env,
        )
        LOGGER.info("MNTP complete for module %s", module)
        # SimCSE
        LOGGER.info("Running SimCSE for module %s", module)
        simcse_out = mod_dir / "simcse"
        simcse_out.mkdir(parents=True, exist_ok=True)
        env = {
            "LLM2VEC_BASE_MODEL_DIR": str(mntp_out),
            "LLM2VEC_OUTPUT_DIR": str(simcse_out),
        }
        run_subprocess(
            [sys.executable, str(simcse_script), str(simcse_config)],
            cwd=llm2vec_root,
            env=env,
        )
        LOGGER.info("SimCSE complete for module %s", module)
        info["current_model_dir"] = str(simcse_out)


def perform_raft(
    modules: Dict[str, Dict[str, Any]],
    run_dir: Path,
    retrieval_corpus: Optional[Path] = None,
    context_length: int = 1024,
) -> None:
    """
    Placeholder for retrieval-augmented fine-tuning (RAFT).

    RAFT involves retrieving relevant context documents for each training
    instruction and concatenating them to the input before fine-tuning.  This
    function is currently a stub.  It should be implemented by the user
    depending on their retrieval engine and task requirements.

    Parameters
    ----------
    modules: dict
        Mapping of module names to dataset/adaptation information.
    run_dir: Path
        Directory containing per-module subdirectories.
    retrieval_corpus: Path, optional
        Path to a corpus of documents that can be retrieved.  If None, the
        function does nothing.
    context_length: int
        Maximum number of tokens from retrieved documents to prepend to each
        instruction.
    """
    LOGGER.warning(
        "RAFT is not implemented in this script.  If you require retrieval-augmented fine-tuning, "
        "please implement this function using your preferred retrieval engine and fine-tuning strategy."
    )
    # Implementation would go here.
    return None


def perform_export(
    modules: Dict[str, Dict[str, Any]],
    run_dir: Path,
    base_model: str,
    method: str = "auto",
    export_cmd_template: Optional[str] = None,
) -> None:
    """
    Export each module’s current model to GGUF or another format.

    Parameters
    ----------
    modules: dict
        Mapping of module names to dataset/adaptation information.  It will be
        updated with ``gguf_path`` pointing to the exported file.
    run_dir: Path
        Directory containing per-module subdirectories.
    base_model: str
        Identifier of the base model (used when merging LoRA weights via the
        built-in exporter).
    method: str
        Export method for the built-in ``export_gguf`` function.  Default is
        "auto" which chooses a method based on available packages.
    export_cmd_template: str, optional
        If provided, this string is formatted with ``model_dir`` and ``gguf_path``
        and executed as a shell command to perform the export.  Use this when
        you have a custom exporter not covered by the built-in ``export_gguf``.
    """
    for module, info in modules.items():
        model_dir = info.get("current_model_dir")
        if not model_dir:
            LOGGER.warning("No model directory for module %s; skipping export", module)
            continue
        model_dir_path = Path(model_dir).expanduser()
        mod_dir = run_dir / module
        gguf_dir = mod_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        gguf_path = gguf_dir / f"{module}.gguf"
        if export_cmd_template:
            cmd = export_cmd_template.format(
                model_dir=model_dir_path, gguf_path=gguf_path
            )
            run_subprocess(shlex.split(cmd), cwd=mod_dir)
        else:
            export_gguf(
                model=base_model,
                lora_dir=str(model_dir_path),
                output_dir=str(gguf_dir),
                method=method,
            )
        LOGGER.info("Exported module %s to %s", module, gguf_path)
        info["gguf_path"] = str(gguf_path)


def parse_cli() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="monGARS multi-stage pipeline")
    parser.add_argument(
        "--run-id", required=True, help="Unique identifier for this pipeline run"
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Root of the monGARS repository containing scripts and code.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory under which run directories will be created.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="HuggingFace model identifier to use as the base model for SFT.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="fr",
        help="Comma separated list of languages for the French dataset pipeline (default: 'fr').",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.06,
        help="Validation split fraction for internal instruction datasets.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for splitting internal datasets.",
    )
    parser.add_argument(
        "--mntp-config",
        type=str,
        required=True,
        help="Path to LLM2Vec MNTP config JSON",
    )
    parser.add_argument(
        "--simcse-config",
        type=str,
        required=True,
        help="Path to LLM2Vec SimCSE config JSON",
    )
    parser.add_argument(
        "--llm2vec-root",
        type=str,
        required=True,
        help="Root directory of the LLM2Vec repository (contains experiments/).",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="Run dataset generation stages (French multi-task + internal).",
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Run supervised fine-tuning for each module.",
    )
    parser.add_argument(
        "--llm2vec",
        action="store_true",
        help="Run LLM2Vec adaptation (MNTP + SimCSE).",
    )
    parser.add_argument(
        "--raft",
        action="store_true",
        help="Run retrieval augmented fine-tuning (not implemented by default).",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export current model checkpoints to GGUF.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HuggingFace token for private models.",
    )
    parser.add_argument(
        "--vram-budget-mb",
        type=int,
        default=24000,
        help="VRAM budget (MB) for training and wrappers.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank for SFT.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha for SFT.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout for SFT.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of epochs for SFT.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SFT.",
    )
    parser.add_argument(
        "--export-method",
        type=str,
        default="auto",
        help="Method for built-in GGUF export.",
    )
    parser.add_argument(
        "--export-cmd",
        type=str,
        default=None,
        help="Custom shell command template for export.  Use {model_dir} and {gguf_path}.",
    )
    return parser, parser.parse_args()


def main() -> None:
    parser, args = parse_cli()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    )
    repo_root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root).resolve()
    run_dir = runs_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prepare state storage
    state_path = run_dir / "state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state: Dict[str, Any] = json.load(f)
    else:
        state = {"datasets": {}, "modules": {}}

    # Stage 1: dataset generation
    if args.build_datasets:
        # French multi-task datasets
        french_output = run_dir / "french_datasets"
        languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
        ds_paths = build_french_multitask_datasets(
            repo_root=repo_root,
            output_dir=french_output,
            languages=languages,
        )
        state["datasets"]["french"] = {k: str(v) for k, v in ds_paths.items()}
        # Internal instruction datasets
        internal_output = run_dir / "internal_datasets"
        module_info = build_internal_instruction_datasets(
            repo_root=repo_root,
            output_dir=internal_output,
            val_split=args.val_split,
            split_seed=args.split_seed,
        )
        state["modules"] = {
            m: {
                "train_path": str(info["train_path"]),
                "val_path": str(info["val_path"]),
                "n_train": info["n_train"],
                "n_val": info["n_val"],
            }
            for m, info in module_info.items()
        }
        save_state(state_path, state)
        LOGGER.info(
            "Dataset generation complete.  Stored dataset info in %s", state_path
        )

    # Stage 2: SFT
    if args.sft:
        if not state.get("modules"):
            raise RuntimeError(
                "No modules found in state.  Run --build-datasets first."
            )
        perform_sft(
            modules=state["modules"],
            run_dir=run_dir,
            base_model=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            vram_budget_mb=args.vram_budget_mb,
            hf_token=args.hf_token,
        )
        save_state(state_path, state)
        LOGGER.info(
            "SFT completed for all modules.  Updated state saved to %s", state_path
        )

    # Stage 3: LLM2Vec
    if args.llm2vec:
        if not state.get("modules"):
            raise RuntimeError(
                "No modules available for LLM2Vec adaptation.  Run previous stages first."
            )
        perform_llm2vec(
            modules=state["modules"],
            run_dir=run_dir,
            llm2vec_root=Path(args.llm2vec_root).resolve(),
            mntp_config=Path(args.mntp_config).resolve(),
            simcse_config=Path(args.simcse_config).resolve(),
        )
        save_state(state_path, state)
        LOGGER.info(
            "LLM2Vec adaptation completed for all modules.  Updated state saved to %s",
            state_path,
        )

    # Stage 4: RAFT
    if args.raft:
        perform_raft(
            modules=state.get("modules", {}),
            run_dir=run_dir,
        )
        save_state(state_path, state)

    # Stage 5: Export
    if args.export:
        if not state.get("modules"):
            raise RuntimeError(
                "No modules available for export.  Run previous stages first."
            )
        perform_export(
            modules=state["modules"],
            run_dir=run_dir,
            base_model=args.base_model,
            method=args.export_method,
            export_cmd_template=args.export_cmd,
        )
        save_state(state_path, state)
        LOGGER.info("Export completed for all modules.  State saved to %s", state_path)

    if not any([args.build_datasets, args.sft, args.llm2vec, args.raft, args.export]):
        parser.print_help()


if __name__ == "__main__":
    main()
