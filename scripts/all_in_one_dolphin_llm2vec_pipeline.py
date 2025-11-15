#!/usr/bin/env python3
"""
All-in-one pipeline for Dolphin + Unsloth + LLM2Vec + monGARS integration.

This script orchestrates the full lifecycle:

    1. Supervised fine-tuning (SFT) of a Dolphin 8B model with Unsloth
       using `scripts/train_dolphin_unsloth.py`.

    2. LLM2Vec adaptation of that model via:
       - Masked Next Token Prediction (MNTP)
       - Unsupervised SimCSE contrastive learning

       using the official LLM2Vec training entry points:

         - experiments/run_mntp.py
         - experiments/run_simcse.py

       and JSON configuration files like:

         - train_configs/mntp/DolphinX1.json
         - train_configs/simcse/DolphinX1.json

    3. Export of the final merged model (Hugging Face directory + optional
       GGUF export via a user-provided command).

    4. Integration into monGARS by launching the LLM2Vec embedding service
       implemented in `scripts/run_llm2vec_service.py`.

The pipeline is opinionated but deterministic: each stage is a real
subprocess with explicit logging and clear inputs/outputs.

Example (full run):

    python scripts/all_in_one_dolphin_llm2vec_pipeline.py \
        --run-id dolphin_x1_llm2vec_v1 \
        --hf-token $HF_TOKEN \
        --train-file data/train.jsonl \
        --validation-file data/val.jsonl \
        --mntp-config llm2vec/train_configs/mntp/DolphinX1.json \
        --simcse-config llm2vec/train_configs/simcse/DolphinX1.json \
        --llm2vec-model-name dolphin-x1-llm2vec-v1 \
        --export-gguf-cmd \
            "python -m llamafile.export --model {model_dir} --out {gguf_path}" \
        --serve -vv

You can also stop earlier, e.g. after SFT:

    python scripts/all_in_one_dolphin_llm2vec_pipeline.py \
        --run-id dolphin_sft_only \
        --hf-token $HF_TOKEN \
        --train-file data/train.jsonl \
        --validation-file data/val.jsonl \
        --stop-after sft

Assumptions
-----------
- This file lives in the `scripts/` directory of your repo.
- `train_dolphin_unsloth.py` is present in the same directory and already
  implements a full Unsloth-powered SFT pipeline.
- The LLM2Vec repo is available in `llm2vec/` (or any path passed via
  --llm2vec-root) and provides `experiments/run_mntp.py` and
  `experiments/run_simcse.py` with JSON configs.
- `run_llm2vec_service.py` is present in scripts/ and can serve embeddings
  when given a model directory.

"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("all_in_one_dolphin_llm2vec")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # Identifiers
    run_id: str
    base_model: str
    llm2vec_model_name: str

    # Data
    train_file: str
    validation_file: Optional[str]

    # Tokens / auth
    hf_token: Optional[str]

    # Paths
    repo_root: Path
    scripts_dir: Path
    llm2vec_root: Path
    runs_root: Path

    # LLM2Vec configs
    mntp_config: Optional[str]
    simcse_config: Optional[str]

    # Export
    export_gguf_cmd_template: Optional[str]

    # Control flags
    stop_after: str
    skip_mntp: bool
    skip_simcse: bool
    dry_run: bool
    serve: bool

    # Serving options
    serve_host: str
    serve_port: int
    serve_extra_args: str

    # Verbosity
    verbosity: int


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


def run_subprocess(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    check: bool = True,
) -> int:
    """Run a subprocess with logging and error handling."""
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    LOGGER.info("Running command%s: %s", " [DRY-RUN]" if dry_run else "", cmd_str)

    if dry_run:
        return 0

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=proc_env,
    )
    if check and proc.returncode != 0:
        raise PipelineError(
            f"Command failed with exit code {proc.returncode}: {cmd_str}"
        )
    return proc.returncode


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    # scripts/ is one level below repo root
    return here.parents[1]


# ---------------------------------------------------------------------------
# Stage 1 – SFT with Unsloth (train_dolphin_unsloth.py)
# ---------------------------------------------------------------------------


def stage_sft(cfg: PipelineConfig) -> Path:
    """
    Run supervised fine-tuning with Unsloth using train_dolphin_unsloth.py.

    Output:
        Path to the SFT run directory, which is assumed to be:
        runs_root / run_id / "sft"
    """
    LOGGER.info("=== Stage 1 / 4: Supervised fine-tuning with Unsloth ===")

    sft_dir = cfg.runs_root / cfg.run_id / "sft"
    ensure_directory(sft_dir)

    script_path = cfg.scripts_dir / "train_dolphin_unsloth.py"
    if not script_path.is_file():
        raise PipelineError(f"train_dolphin_unsloth.py not found at {script_path}")

    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--train-file",
        cfg.train_file,
        "--output-dir",
        str(sft_dir),
    ]

    if cfg.validation_file:
        cmd.extend(["--validation-file", cfg.validation_file])

    if cfg.hf_token:
        cmd.extend(["--hf-token", cfg.hf_token])

    run_subprocess(cmd, cwd=cfg.repo_root, dry_run=cfg.dry_run)

    LOGGER.info("SFT stage complete → %s", sft_dir)
    return sft_dir


# ---------------------------------------------------------------------------
# Stage 2 – LLM2Vec MNTP
# ---------------------------------------------------------------------------


def stage_mntp(cfg: PipelineConfig, sft_dir: Path) -> Path:
    """
    Run LLM2Vec MNTP adaptation on top of the SFT model.

    Uses:

        python experiments/run_mntp.py <config.json>

    The config can:

      - read the base model directory from LLM2VEC_BASE_MODEL_DIR
      - write outputs into LLM2VEC_OUTPUT_DIR

    Output:
        Path to the MNTP-adapted model directory.
    """
    LOGGER.info("=== Stage 2 / 4: LLM2Vec MNTP adaptation ===")

    if cfg.skip_mntp:
        LOGGER.info("MNTP stage skipped by configuration.")
        return sft_dir

    if not cfg.mntp_config:
        raise PipelineError(
            "MNTP stage enabled but --mntp-config was not provided. "
            "Create a LLM2Vec JSON config and pass its path."
        )

    llm2vec_root = cfg.llm2vec_root
    mntp_script = llm2vec_root / "experiments" / "run_mntp.py"
    if not mntp_script.is_file():
        raise PipelineError(f"LLM2Vec MNTP script not found at {mntp_script}")

    mntp_out_dir = cfg.runs_root / cfg.run_id / "mntp"
    ensure_directory(mntp_out_dir)

    env: Dict[str, str] = {}
    env["LLM2VEC_BASE_MODEL_DIR"] = str(sft_dir)
    env["LLM2VEC_OUTPUT_DIR"] = str(mntp_out_dir)

    cmd: List[str] = [
        sys.executable,
        str(mntp_script),
        cfg.mntp_config,
    ]

    run_subprocess(cmd, cwd=llm2vec_root, env=env, dry_run=cfg.dry_run)

    LOGGER.info("MNTP stage complete → %s", mntp_out_dir)
    return mntp_out_dir


# ---------------------------------------------------------------------------
# Stage 3 – LLM2Vec SimCSE
# ---------------------------------------------------------------------------


def stage_simcse(cfg: PipelineConfig, mntp_dir: Path) -> Path:
    """
    Run LLM2Vec unsupervised SimCSE contrastive learning.

    Uses:

        python experiments/run_simcse.py <config.json>

    The config can:

      - read base model from LLM2VEC_BASE_MODEL_DIR
      - write outputs to LLM2VEC_OUTPUT_DIR

    Output:
        Path to final LLM2Vec-adapted model directory.
    """
    LOGGER.info("=== Stage 3 / 4: LLM2Vec SimCSE adaptation ===")

    if cfg.skip_simcse:
        LOGGER.info("SimCSE stage skipped by configuration.")
        return mntp_dir

    if not cfg.simcse_config:
        raise PipelineError(
            "SimCSE stage enabled but --simcse-config was not provided. "
            "Create a LLM2Vec JSON config and pass its path."
        )

    llm2vec_root = cfg.llm2vec_root
    simcse_script = llm2vec_root / "experiments" / "run_simcse.py"
    if not simcse_script.is_file():
        raise PipelineError(f"LLM2Vec SimCSE script not found at {simcse_script}")

    simcse_out_dir = cfg.runs_root / cfg.run_id / "simcse"
    ensure_directory(simcse_out_dir)

    env: Dict[str, str] = {}
    env["LLM2VEC_BASE_MODEL_DIR"] = str(mntp_dir)
    env["LLM2VEC_OUTPUT_DIR"] = str(simcse_out_dir)

    cmd: List[str] = [
        sys.executable,
        str(simcse_script),
        cfg.simcse_config,
    ]

    run_subprocess(cmd, cwd=llm2vec_root, env=env, dry_run=cfg.dry_run)

    LOGGER.info("SimCSE stage complete → %s", simcse_out_dir)
    return simcse_out_dir


# ---------------------------------------------------------------------------
# Stage 4 – Export (HF + optional GGUF)
# ---------------------------------------------------------------------------


def stage_export(cfg: PipelineConfig, final_model_dir: Path) -> Dict[str, Path]:
    """
    Export the model.

    - final_model_dir is expected to be a Hugging Face-compatible
      directory (config.json, tokenizer.json, model.safetensors, etc.).
    - Optionally perform GGUF export using a user-supplied command template.

    The GGUF command template may reference:

        {model_dir} – the HF directory to export from
        {gguf_path} – the desired output GGUF file path

    Returns:
        Mapping with at least {"hf_dir": final_model_dir}. May also include
        {"gguf": gguf_path} if export was requested.
    """
    LOGGER.info("=== Stage 4 / 4: Export ===")

    outputs: Dict[str, Path] = {"hf_dir": final_model_dir}

    if cfg.export_gguf_cmd_template:
        gguf_dir = cfg.runs_root / cfg.run_id / "exports"
        ensure_directory(gguf_dir)

        gguf_path = gguf_dir / f"{cfg.llm2vec_model_name}.gguf"

        cmd_str = cfg.export_gguf_cmd_template.format(
            model_dir=str(final_model_dir),
            gguf_path=str(gguf_path),
        )
        cmd = shlex.split(cmd_str)

        run_subprocess(cmd, cwd=cfg.repo_root, dry_run=cfg.dry_run)
        LOGGER.info("GGUF export complete → %s", gguf_path)
        outputs["gguf"] = gguf_path

    LOGGER.info("Export stage complete.")
    return outputs


# ---------------------------------------------------------------------------
# Stage 5 – Integration: run_llm2vec_service.py
# ---------------------------------------------------------------------------


def stage_serve(cfg: PipelineConfig, hf_model_dir: Path) -> None:
    """
    Launch the monGARS LLM2Vec embedding service for the exported model.

    We assume a script:

        scripts/run_llm2vec_service.py

    which accepts:

        --model-dir  <hf_model_dir>
        --host       <host>
        --port       <port>

    plus any extra CLI arguments passed through --serve-extra-args.
    """
    if not cfg.serve:
        LOGGER.info("Serving stage disabled (no --serve flag).")
        return

    LOGGER.info("=== Stage 5: Starting LLM2Vec embedding service ===")

    service_script = cfg.scripts_dir / "run_llm2vec_service.py"
    if not service_script.is_file():
        raise PipelineError(f"run_llm2vec_service.py not found at {service_script}")

    cmd: List[str] = [
        sys.executable,
        str(service_script),
        "--model-dir",
        str(hf_model_dir),
        "--host",
        cfg.serve_host,
        "--port",
        str(cfg.serve_port),
    ]

    if cfg.serve_extra_args.strip():
        extra = shlex.split(cfg.serve_extra_args)
        cmd.extend(extra)

    try:
        run_subprocess(cmd, cwd=cfg.repo_root, dry_run=cfg.dry_run, check=False)
    except KeyboardInterrupt:
        LOGGER.info("Embedding service interrupted by user.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="All-in-one Dolphin + Unsloth + LLM2Vec pipeline."
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier for this run; used to create runs/<run_id>/ subdirectories.",
    )
    parser.add_argument(
        "--base-model",
        default="dphn/Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf",
        help="Base model identifier used by the SFT step (if needed by underlying scripts).",
    )
    parser.add_argument(
        "--llm2vec-model-name",
        required=True,
        help="Logical name for the final LLM2Vec model; used for export filenames.",
    )

    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to the SFT training dataset (JSON/JSONL), forwarded to train_dolphin_unsloth.py.",
    )
    parser.add_argument(
        "--validation-file",
        help="Optional validation dataset for SFT.",
    )

    parser.add_argument(
        "--hf-token",
        help="Hugging Face token for model/dataset access, if required.",
    )

    parser.add_argument(
        "--llm2vec-root",
        default="llm2vec",
        help="Path to the llm2vec repo root (where experiments/ lives).",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Base directory for all pipeline outputs.",
    )

    parser.add_argument(
        "--mntp-config",
        help="Path to the LLM2Vec MNTP JSON config (train_configs/mntp/...).",
    )
    parser.add_argument(
        "--simcse-config",
        help="Path to the LLM2Vec SimCSE JSON config (train_configs/simcse/...).",
    )

    parser.add_argument(
        "--export-gguf-cmd",
        dest="export_gguf_cmd_template",
        help=(
            "Optional GGUF export command template. "
            "Use {model_dir} and {gguf_path} placeholders, e.g.: "
            '"python -m llamafile.export --model {model_dir} --out {gguf_path}"'
        ),
    )

    parser.add_argument(
        "--stop-after",
        choices=["sft", "mntp", "simcse", "export"],
        default="export",
        help="Stop the pipeline after the given stage.",
    )

    parser.add_argument(
        "--skip-mntp",
        action="store_true",
        help="Skip MNTP stage and feed SFT output directly into SimCSE or export.",
    )
    parser.add_argument(
        "--skip-simcse",
        action="store_true",
        help="Skip SimCSE stage and export MNTP (or SFT) output directly.",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="After export, start the run_llm2vec_service.py embedding server.",
    )
    parser.add_argument(
        "--serve-host",
        default="127.0.0.1",
        help="Host for the embedding service.",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8080,
        help="Port for the embedding service.",
    )
    parser.add_argument(
        "--serve-extra-args",
        default="",
        help="Extra CLI args passed to run_llm2vec_service.py (single string, shell-split).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv).",
    )

    ns = parser.parse_args(argv)

    repo_root = resolve_repo_root()
    scripts_dir = repo_root / "scripts"
    llm2vec_root = (repo_root / ns.llm2vec_root).resolve()
    runs_root = (repo_root / ns.runs_root).resolve()

    cfg = PipelineConfig(
        run_id=ns.run_id,
        base_model=ns.base_model,
        llm2vec_model_name=ns.llm2vec_model_name,
        train_file=ns.train_file,
        validation_file=ns.validation_file,
        hf_token=ns.hf_token,
        repo_root=repo_root,
        scripts_dir=scripts_dir,
        llm2vec_root=llm2vec_root,
        runs_root=runs_root,
        mntp_config=ns.mntp_config,
        simcse_config=ns.simcse_config,
        export_gguf_cmd_template=ns.export_gguf_cmd_template,
        stop_after=ns.stop_after,
        skip_mntp=ns.skip_mntp,
        skip_simcse=ns.skip_simcse,
        dry_run=ns.dry_run,
        serve=ns.serve,
        serve_host=ns.serve_host,
        serve_port=ns.serve_port,
        serve_extra_args=ns.serve_extra_args,
        verbosity=ns.verbose,
    )
    return cfg


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_pipeline(cfg: PipelineConfig) -> None:
    LOGGER.info("Starting all-in-one pipeline with run_id=%s", cfg.run_id)

    # Stage 1: SFT
    sft_dir = stage_sft(cfg)
    if cfg.stop_after == "sft":
        LOGGER.info("Stopping after SFT as requested.")
        return

    # Stage 2: MNTP
    current_dir = sft_dir
    if not cfg.skip_mntp:
        current_dir = stage_mntp(cfg, sft_dir)
    else:
        LOGGER.info("Skipping MNTP stage; passing SFT output forward.")

    if cfg.stop_after == "mntp":
        LOGGER.info("Stopping after MNTP as requested.")
        return

    # Stage 3: SimCSE
    if not cfg.skip_simcse:
        current_dir = stage_simcse(cfg, current_dir)
    else:
        LOGGER.info("Skipping SimCSE stage; passing previous output to export.")

    if cfg.stop_after == "simcse":
        LOGGER.info("Stopping after SimCSE as requested.")
        return

    # Stage 4: Export
    exports = stage_export(cfg, current_dir)
    hf_dir = exports["hf_dir"]

    if cfg.stop_after == "export":
        LOGGER.info("Stopping after export as requested.")
        if cfg.serve:
            stage_serve(cfg, hf_dir)
        return

    if cfg.serve:
        stage_serve(cfg, hf_dir)


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv)
    setup_logging(cfg.verbosity)

    try:
        run_pipeline(cfg)
    except PipelineError as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        LOGGER.warning("Pipeline interrupted by user.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
