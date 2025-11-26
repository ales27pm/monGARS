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
import inspect
import json
import logging
import os
import random
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from monGARS.mlops.code_analysis import (
    scan_llm_usage,
    scan_module_interactions,
)
from monGARS.mlops.dataset import (
    _load_jsonl_records,
    build_module_interaction_dataset,
    build_mongars_strategy_dataset,
)
from monGARS.mlops.exporters import export_gguf
from monGARS.mlops.pipelines.unsloth import run_unsloth_finetune

LOGGER = logging.getLogger("mongars_multistage_pipeline")


def _call_with_supported_kwargs(func, **kwargs):
    """Invoke ``func`` with only the keyword arguments it supports.

    This guards the pipeline against signature drift in lower-level helpers by
    removing unsupported parameters rather than propagating a ``TypeError``
    mid-run. Any skipped arguments are logged for visibility so that callers can
    keep configurations aligned with the current API.
    """

    signature = inspect.signature(func)
    accepted = {}
    skipped = []
    for key, value in kwargs.items():
        if key in signature.parameters:
            accepted[key] = value
        else:
            skipped.append(key)

    if skipped:
        LOGGER.warning(
            "Skipping unsupported parameters for %s: %s",
            func.__name__,
            ", ".join(skipped),
        )

    return func(**accepted)


@dataclass
class ModuleState:
    train_path: Path
    val_path: Path
    n_train: int
    n_val: int
    adapter_dir: Optional[Path] = None
    current_model_dir: Optional[Path] = None
    gguf_path: Optional[Path] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in payload.items()
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ModuleState":
        return cls(
            train_path=Path(data["train_path"]),
            val_path=Path(data["val_path"]),
            n_train=int(data["n_train"]),
            n_val=int(data["n_val"]),
            adapter_dir=Path(data["adapter_dir"]) if data.get("adapter_dir") else None,
            current_model_dir=(
                Path(data["current_model_dir"])
                if data.get("current_model_dir")
                else None
            ),
            gguf_path=Path(data["gguf_path"]) if data.get("gguf_path") else None,
        )


ModulesDict = Dict[str, ModuleState]


def save_state(path: Path, datasets: Dict[str, Any], modules: ModulesDict) -> None:
    """Persist pipeline state to disk with path values converted to strings."""

    state = {
        "datasets": datasets,
        "modules": {name: module.to_json() for name, module in modules.items()},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_state(path: Path) -> Tuple[Dict[str, Any], ModulesDict]:
    if not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as f:
        raw_state = json.load(f)
    modules = {
        name: ModuleState.from_json(data)
        for name, data in raw_state.get("modules", {}).items()
    }
    return raw_state.get("datasets", {}), modules


def parse_module_from_instruction(instr: str) -> Tuple[str, str]:
    """Extract module name and instruction text from a ``[MOD=...]`` prefix.

    Returns a tuple of ``(module_name, stripped_instruction)``. If no prefix is
    present or the prefix is malformed, the module defaults to "General" and the
    original instruction is returned unchanged.
    """

    if instr.startswith("[MOD="):
        end = instr.find("]")
        if end > 5:
            module = instr[5:end]
            stripped = instr[end + 1 :].lstrip()
            return module, stripped
        LOGGER.warning("Malformed module prefix found in instruction: %s", instr)
    return "General", instr


def run_subprocess(
    cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None
) -> None:
    """Execute a command via subprocess and raise on failure."""

    if not cmd:
        raise ValueError("Command cannot be empty")
    cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
    LOGGER.info("Running command: %s", cmd_str)
    proc_env = os.environ.copy()
    if env:
        proc_env |= env
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

    # Construct command using underscore-style arguments expected by the dataset pipeline.
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--output_dir",
        str(output_dir),
        "--langs",
        ",".join(languages),
        "--num_workers",
        str(workers),
        "--ignore_failed_datasets",
    ]
    if trust_remote_code:
        cmd.append("--allow_trust_remote_code")
    # Pass through optional maximum examples per dataset using the --max_per_dataset flag.
    if max_examples is not None:
        cmd += ["--max_per_dataset", str(max_examples)]
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
) -> ModulesDict:
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
    ModulesDict
        A mapping from module name to its dataset metadata. The returned
        structure matches that produced by the interactive pipeline’s
        ``build_datasets`` helper.
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
        records.extend(_load_jsonl_records(path))

    examples_by_module: Dict[str, List[Dict[str, str]]] = {}
    for rec in records:
        module, stripped_instr = parse_module_from_instruction(rec["instruction"])
        payload = {
            **rec,
            "instruction": stripped_instr,
            "input": rec.get("input", ""),
            "output": rec["output"],
        }
        examples_by_module.setdefault(module, []).append(payload)

    rng = random.Random(split_seed)
    module_info: ModulesDict = {}
    for module, recs in examples_by_module.items():
        if not recs:
            continue
        rng.shuffle(recs)
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
        module_info[module] = ModuleState(
            train_path=train_path,
            val_path=val_path,
            n_train=len(train_recs),
            n_val=len(val_recs),
        )
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug(
            "Cleanup failed for %s, %s, %s: %s",
            strategy_path,
            interaction_path,
            tmp_dir,
            exc,
            exc_info=True,
        )
    return module_info


def perform_sft(
    modules: ModulesDict,
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
    modules: ModulesDict
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
    for module, state in modules.items():
        LOGGER.info("Starting SFT for module %s", module)
        mod_dir = run_dir / module
        mod_dir.mkdir(parents=True, exist_ok=True)
        result = _call_with_supported_kwargs(
            run_unsloth_finetune,
            model_id=base_model,
            output_dir=mod_dir,
            dataset_path=state.train_path,
            max_seq_len=1024,
            vram_budget_mb=vram_budget_mb,
            batch_size=batch_size,
            epochs=epochs,
            lora_rank=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            eval_dataset_path=state.val_path,
            eval_batch_size=batch_size,
            run_smoke_tests=False,
        )
        LOGGER.info("Finished SFT for module %s", module)
        adapter_dir = Path(result["chat_lora_dir"])
        state.adapter_dir = adapter_dir
        state.current_model_dir = adapter_dir


def perform_llm2vec(
    modules: ModulesDict,
    run_dir: Path,
    llm2vec_root: Path,
    mntp_config: Path,
    simcse_config: Path,
) -> None:
    """
    Run LLM2Vec adaptation (MNTP then SimCSE) on each fine-tuned module.

    Parameters
    ----------
    modules: ModulesDict
        Mapping of module names to dataset/adaptation information. The mapping
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
    for module, state in modules.items():
        model_dir = state.current_model_dir
        if not model_dir:
            LOGGER.warning("No model dir for module %s; skipping LLM2Vec", module)
            continue
        mod_dir = run_dir / module
        LOGGER.info("Running MNTP for module %s", module)
        mntp_out = mod_dir / "mntp"
        mntp_out.mkdir(parents=True, exist_ok=True)
        env = {
            "LLM2VEC_BASE_MODEL_DIR": str(model_dir),
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
        state.current_model_dir = simcse_out


def perform_raft(
    modules: ModulesDict,
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
    modules: ModulesDict
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
    modules: ModulesDict,
    run_dir: Path,
    base_model: str,
    method: str = "auto",
    export_cmd_template: Optional[str] = None,
) -> None:
    """
    Export each module’s current model to GGUF or another format.

    Parameters
    ----------
    modules: ModulesDict
        Mapping of module names to dataset/adaptation information. It will be
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
    for module, state in modules.items():
        model_dir = state.current_model_dir
        if not model_dir:
            LOGGER.warning("No model directory for module %s; skipping export", module)
            continue
        mod_dir = run_dir / module
        gguf_dir = mod_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        exported_files: List[Path]
        if export_cmd_template:
            gguf_path = gguf_dir / f"{module}.gguf"
            cmd = export_cmd_template.format(model_dir=model_dir, gguf_path=gguf_path)
            run_subprocess(shlex.split(cmd), cwd=mod_dir)
            exported_files = [gguf_path] if gguf_path.exists() else []
        else:
            _call_with_supported_kwargs(
                export_gguf,
                source_dir=model_dir,
                gguf_dir=gguf_dir,
                quantization_method=method,
            )
            exported_files = sorted(gguf_dir.glob("*.gguf"))
        if exported_files:
            state.gguf_path = exported_files[0]
            LOGGER.info("Exported module %s to %s", module, exported_files[0])
        else:
            state.gguf_path = None
            LOGGER.warning(
                "No GGUF artefacts found for module %s in %s", module, gguf_dir
            )


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
        default=None,
        help=(
            "HuggingFace model identifier to use as the base model for SFT and export. "
            "Required when running --sft or --export."
        ),
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
        default=None,
        help=(
            "Path to LLM2Vec MNTP config JSON. Required when running the LLM2Vec stage."
        ),
    )
    parser.add_argument(
        "--simcse-config",
        type=str,
        default=None,
        help=(
            "Path to LLM2Vec SimCSE config JSON. Required when running the LLM2Vec stage."
        ),
    )
    parser.add_argument(
        "--llm2vec-root",
        type=str,
        default=None,
        help=(
            "Root directory of the LLM2Vec repository (contains experiments/). "
            "Required when running the LLM2Vec stage."
        ),
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
        default=7000,
        help=(
            "VRAM budget (MB) for training and wrappers.  The default is optimised "
            "for an 8 GB GPU (e.g. RTX 2070).  Adjust this value if your GPU has "
            "more or less memory."
        ),
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Run the pipeline in interactive mode.  When this flag is present or no stage flags "
            "are provided, a guided menu will prompt for configuration values instead of using "
            "command-line arguments."
        ),
    )
    return parser, parser.parse_args()

def run_interactive_menu(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Interactive menu for configuring and running the monGARS pipeline.

    This helper prompts the user for all necessary parameters and stage
    selections.  It mirrors the command-line interface but is more user
    friendly, providing defaults and explanations for each option.  The
    function persists state across stages via the ``state.json`` file in
    the chosen run directory.
    """
    print("\n=== monGARS Multi‑Stage Pipeline Interactive Menu ===")
    print("This menu will guide you through selecting which stages to run and")
    print("entering configuration values.  Press Enter to accept the defaults.\n")

    # Determine sensible defaults from provided args.
    default_run_id = getattr(args, "run_id", None) or "001"
    run_id = input(f"Enter a unique run identifier [default: {default_run_id}]: ").strip()
    if not run_id:
        run_id = default_run_id

    default_repo_root = Path(getattr(args, "repo_root", ".")).resolve()
    repo_root_in = input(
        f"Repository root containing the monGARS codebase [default: {default_repo_root}]: "
    ).strip()
    repo_root = Path(repo_root_in or default_repo_root).resolve()

    default_runs_root = Path(getattr(args, "runs_root", "runs")).resolve()
    runs_root_in = input(
        f"Root directory where run outputs will be stored [default: {default_runs_root}]: "
    ).strip()
    runs_root = Path(runs_root_in or default_runs_root).resolve()

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load existing state if present.
    state_path = run_dir / "state.json"
    datasets, modules = load_state(state_path)

    # Prompt for stage selection
    print("\nSelect stages to run (answer yes/no for each):")
    def yes_no(prompt: str, default: bool = False) -> bool:
        resp = input(prompt).strip().lower()
        if not resp:
            return default
        return resp.startswith("y")

    build_ds = yes_no("1. Build datasets? [y/N]: ", default=False)
    run_sft = yes_no("2. Supervised fine‑tuning (SFT)? [y/N]: ", default=False)
    run_llm2vec = yes_no("3. LLM2Vec adaptation? [y/N]: ", default=False)
    run_raft = yes_no("4. Retrieval‑augmented fine‑tuning (RAFT)? [y/N]: ", default=False)
    run_export = yes_no("5. Export to GGUF? [y/N]: ", default=False)

    # Dataset parameters
    languages = None
    max_per_dataset = None
    val_split = 0.06
    split_seed = 42
    if build_ds:
        lang_default = getattr(args, "languages", "fr")
        languages_in = input(
            f"Languages for French dataset (comma‑separated) [default: {lang_default}]: "
        ).strip()
        languages = [lang.strip() for lang in (languages_in or lang_default).split(",") if lang.strip()]

        max_in = input(
            "Maximum examples per dataset (leave blank for unlimited/default): "
        ).strip()
        if max_in:
            try:
                max_per_dataset = int(max_in)
            except ValueError:
                print("Invalid number for max examples; ignoring.")
                max_per_dataset = None

        val_default = getattr(args, "val_split", 0.06)
        val_in = input(
            f"Validation split fraction for internal instruction datasets [default: {val_default}]: "
        ).strip()
        if val_in:
            try:
                val_split = float(val_in)
            except ValueError:
                print("Invalid value for validation split; using default.")
                val_split = val_default
        else:
            val_split = val_default

        seed_default = getattr(args, "split_seed", 42)
        seed_in = input(
            f"Random seed for splitting internal datasets [default: {seed_default}]: "
        ).strip()
        if seed_in:
            try:
                split_seed = int(seed_in)
            except ValueError:
                print("Invalid seed; using default.")
                split_seed = seed_default
        else:
            split_seed = seed_default

    # SFT parameters
    base_model = None
    lora_r = getattr(args, "lora_r", 64)
    lora_alpha = getattr(args, "lora_alpha", 16)
    lora_dropout = getattr(args, "lora_dropout", 0.05)
    epochs = getattr(args, "epochs", 3.0)
    batch_size = getattr(args, "batch_size", 64)
    vram_budget_mb = getattr(args, "vram_budget_mb", 7000)
    hf_token = getattr(args, "hf_token", None)
    if run_sft or run_raft:
        # When performing SFT or RAFT we require a base model
        base_default = getattr(args, "base_model", "") or "mistralai/Mistral-7B-v0.2"
        base_in = input(
            f"Base model identifier for SFT and export [default: {base_default}]: "
        ).strip()
        base_model = base_in or base_default

        r_in = input(f"LoRA rank (r) [default: {lora_r}]: ").strip()
        if r_in:
            try:
                lora_r = int(r_in)
            except ValueError:
                print("Invalid LoRA rank; using default.")

        alpha_in = input(f"LoRA alpha [default: {lora_alpha}]: ").strip()
        if alpha_in:
            try:
                lora_alpha = int(alpha_in)
            except ValueError:
                print("Invalid LoRA alpha; using default.")

        dropout_in = input(f"LoRA dropout [default: {lora_dropout}]: ").strip()
        if dropout_in:
            try:
                lora_dropout = float(dropout_in)
            except ValueError:
                print("Invalid dropout value; using default.")

        epochs_in = input(f"Number of epochs [default: {epochs}]: ").strip()
        if epochs_in:
            try:
                epochs = float(epochs_in)
            except ValueError:
                print("Invalid number of epochs; using default.")

        bs_in = input(f"Batch size [default: {batch_size}]: ").strip()
        if bs_in:
            try:
                batch_size = int(bs_in)
            except ValueError:
                print("Invalid batch size; using default.")

        vram_in = input(f"VRAM budget (MB) [default: {vram_budget_mb}]: ").strip()
        if vram_in:
            try:
                vram_budget_mb = int(vram_in)
            except ValueError:
                print("Invalid VRAM budget; using default.")

        token_in = input("Optional HuggingFace token (press Enter to skip): ").strip()
        hf_token = token_in or None

    # LLM2Vec parameters
    llm2vec_root = None
    mntp_config = None
    simcse_config = None
    if run_llm2vec:
        root_in = input(
            "Path to LLM2Vec repository root (containing experiments/) (required): "
        ).strip()
        llm2vec_root = Path(root_in).resolve() if root_in else None
        mntp_in = input(
            "Path to LLM2Vec MNTP config JSON (required): "
        ).strip()
        mntp_config = Path(mntp_in).resolve() if mntp_in else None
        simcse_in = input(
            "Path to LLM2Vec SimCSE config JSON (required): "
        ).strip()
        simcse_config = Path(simcse_in).resolve() if simcse_in else None

        if not llm2vec_root or not mntp_config or not simcse_config:
            parser.error(
                "LLM2Vec root and both MNTP and SimCSE configs must be provided when running LLM2Vec."
            )

    # RAFT parameters
    retrieval_corpus = None
    context_length = 1024
    if run_raft:
        # Determine a default retrieval corpus: use French retrieval dataset if available.
        default_corpus = None
        french_ds = datasets.get("french", {})
        if isinstance(french_ds, dict) and "retrieval" in french_ds:
            default_corpus = Path(french_ds["retrieval"])
        corpus_in = input(
            f"Path to retrieval corpus for RAFT [default: {default_corpus or ''}]: "
        ).strip()
        if corpus_in:
            retrieval_corpus = Path(corpus_in).resolve()
        else:
            retrieval_corpus = default_corpus
        ctx_in = input(
            f"Max tokens from retrieved context to prepend [default: {context_length}]: "
        ).strip()
        if ctx_in:
            try:
                context_length = int(ctx_in)
            except ValueError:
                print("Invalid context length; using default.")

    # Export parameters
    export_method = getattr(args, "export_method", "auto")
    export_cmd = getattr(args, "export_cmd", None)
    if run_export:
        method_in = input(
            f"Export quantization method (auto/none/q4_0/q4_1/etc.) [default: {export_method}]: "
        ).strip()
        if method_in:
            export_method = method_in
        cmd_in = input(
            "Custom export command template (leave blank to use built-in exporter): "
        ).strip()
        if cmd_in:
            export_cmd = cmd_in

    # Stage 1: Dataset generation
    if build_ds:
        print("\n[DATASET] Building French multi-task datasets...")
        french_output = run_dir / "french_datasets"
        ds_paths = build_french_multitask_datasets(
            repo_root=repo_root,
            output_dir=french_output,
            languages=languages or ["fr"],
            max_examples=max_per_dataset,
            trust_remote_code=True,
            workers=4,
        )
        datasets["french"] = {k: str(v) for k, v in ds_paths.items()}
        print(f"[DATASET] French datasets created at {french_output}")

        print("[DATASET] Building internal instruction datasets...")
        internal_output = run_dir / "internal_datasets"
        modules = build_internal_instruction_datasets(
            repo_root=repo_root,
            output_dir=internal_output,
            val_split=val_split,
            split_seed=split_seed,
        )
        save_state(state_path, datasets, modules)
        print("[DATASET] Internal instruction datasets generated.")

    # Stage 2: SFT
    if run_sft:
        if not modules:
            raise RuntimeError(
                "No modules found in state.  Run the dataset stage first."
            )
        print("\n[SFT] Running supervised fine-tuning...")
        perform_sft(
            modules=modules,
            run_dir=run_dir,
            base_model=base_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            epochs=epochs,
            batch_size=batch_size,
            vram_budget_mb=vram_budget_mb,
            hf_token=hf_token,
        )
        save_state(state_path, datasets, modules)
        print("[SFT] Completed supervised fine-tuning.")

    # Stage 3: LLM2Vec
    if run_llm2vec:
        if not modules:
            raise RuntimeError(
                "No modules available for LLM2Vec adaptation.  Run previous stages first."
            )
        print("\n[LLM2VEC] Running LLM2Vec adaptation...")
        perform_llm2vec(
            modules=modules,
            run_dir=run_dir,
            llm2vec_root=llm2vec_root,
            mntp_config=mntp_config,
            simcse_config=simcse_config,
        )
        save_state(state_path, datasets, modules)
        print("[LLM2VEC] Completed LLM2Vec adaptation.")

    # Stage 4: RAFT
    if run_raft:
        print("\n[RAFT] Running retrieval-augmented fine-tuning...")
        perform_raft(
            modules=modules,
            run_dir=run_dir,
            retrieval_corpus=retrieval_corpus,
            context_length=context_length,
        )
        save_state(state_path, datasets, modules)
        print("[RAFT] Completed retrieval-augmented fine-tuning.")

    # Stage 5: Export
    if run_export:
        if not modules:
            raise RuntimeError(
                "No modules available for export.  Run previous stages first."
            )
        print("\n[EXPORT] Exporting current model checkpoints...")
        perform_export(
            modules=modules,
            run_dir=run_dir,
            base_model=base_model,
            method=export_method,
            export_cmd=export_cmd,
        )
        save_state(state_path, datasets, modules)
        print("[EXPORT] Export completed.")

    print("\nPipeline finished.  State saved to", state_path)


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
    datasets, modules = load_state(state_path)

    # If interactive flag is set or no stage flags provided, enter interactive menu.
    stage_flags_provided = any(
        [
            args.build_datasets,
            args.sft,
            args.llm2vec,
            args.raft,
            args.export,
        ]
    )
    if args.interactive or not stage_flags_provided:
        run_interactive_menu(args, parser)
        return

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
        datasets["french"] = {k: str(v) for k, v in ds_paths.items()}
        # Internal instruction datasets
        internal_output = run_dir / "internal_datasets"
        modules = build_internal_instruction_datasets(
            repo_root=repo_root,
            output_dir=internal_output,
            val_split=args.val_split,
            split_seed=args.split_seed,
        )
        save_state(state_path, datasets, modules)
        LOGGER.info(
            "Dataset generation complete.  Stored dataset info in %s", state_path
        )

    # Stage 2: SFT
    if args.sft:
        if not modules:
            raise RuntimeError(
                "No modules found in state.  Run --build-datasets first."
            )
        if not args.base_model:
            parser.error("--base-model is required when running SFT")
        perform_sft(
            modules=modules,
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
        save_state(state_path, datasets, modules)
        LOGGER.info(
            "SFT completed for all modules.  Updated state saved to %s", state_path
        )

    # Stage 3: LLM2Vec
    if args.llm2vec:
        if not modules:
            raise RuntimeError(
                "No modules available for LLM2Vec adaptation.  Run previous stages first."
            )
        if not args.llm2vec_root:
            parser.error("--llm2vec-root is required when running --llm2vec")
        if not args.mntp_config:
            parser.error("--mntp-config is required when running --llm2vec")
        if not args.simcse_config:
            parser.error("--simcse-config is required when running --llm2vec")
        perform_llm2vec(
            modules=modules,
            run_dir=run_dir,
            llm2vec_root=Path(args.llm2vec_root).resolve(),
            mntp_config=Path(args.mntp_config).resolve(),
            simcse_config=Path(args.simcse_config).resolve(),
        )
        save_state(state_path, datasets, modules)
        LOGGER.info(
            "LLM2Vec adaptation completed for all modules.  Updated state saved to %s",
            state_path,
        )

    # Stage 4: RAFT
    if args.raft:
        perform_raft(
            modules=modules,
            run_dir=run_dir,
        )
        save_state(state_path, datasets, modules)

    # Stage 5: Export
    if args.export:
        if not modules:
            raise RuntimeError(
                "No modules available for export.  Run previous stages first."
            )
        if not args.base_model:
            parser.error("--base-model is required when running --export")
        perform_export(
            modules=modules,
            run_dir=run_dir,
            base_model=args.base_model,
            method=args.export_method,
            export_cmd_template=args.export_cmd,
        )
        save_state(state_path, datasets, modules)
        LOGGER.info("Export completed for all modules.  State saved to %s", state_path)

    if not any([args.build_datasets, args.sft, args.llm2vec, args.raft, args.export]):
        parser.print_help()


if __name__ == "__main__":
    main()
