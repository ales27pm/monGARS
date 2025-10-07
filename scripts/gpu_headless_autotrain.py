#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU-adaptive training orchestrator for monGARS."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------


def sh(
    cmd: str, check: bool = False, env: Optional[Dict[str, str]] = None
) -> subprocess.CompletedProcess:
    """Run a shell command with minimal fuss."""

    print(f"➡️  {cmd}")
    res = subprocess.run(
        cmd,
        shell=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
    return res


def has_cmd(bin_name: str) -> bool:
    return (
        subprocess.call(
            f"command -v {shlex.quote(bin_name)} >/dev/null 2>&1", shell=True
        )
        == 0
    )


def parse_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def read_first_line(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def sleep(seconds: float) -> None:
    print(f"⏳ sleeping {seconds:.1f}s...")
    time.sleep(seconds)


# -----------------------------
# System control (headless)
# -----------------------------


def switch_to_headless(persist: bool = False) -> None:
    """Switch to CLI-only systemd target."""

    if not has_cmd("systemctl"):
        print("⚠️ systemctl not found; headless switch skipped.")
        return
    if persist:
        print("🔧 Setting default target to multi-user (headless) persistently...")
        sh("sudo systemctl set-default multi-user.target", check=True)
    else:
        print("🔧 Isolating to multi-user (headless) now...")
        sh("sudo systemctl isolate multi-user.target", check=False)


def restore_gui_default() -> None:
    if not has_cmd("systemctl"):
        print("⚠️ systemctl not found; cannot restore GUI default.")
        return
    print("🔧 Restoring default target to graphical...")
    sh("sudo systemctl set-default graphical.target", check=False)


def reboot_now() -> None:
    print("🔁 Rebooting now...")
    sh("sudo sync", check=False)
    sh("sudo reboot", check=False)


# -----------------------------
# Resource probing
# -----------------------------


def get_gpu_name() -> Optional[str]:
    if not has_cmd("nvidia-smi"):
        return None
    return read_first_line(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
    )


def get_free_vram_mb() -> int:
    if not has_cmd("nvidia-smi"):
        return 0
    line = read_first_line(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_int(line or "0")


def get_total_vram_mb() -> int:
    if not has_cmd("nvidia-smi"):
        return 0
    line = read_first_line(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_int(line or "0")


def get_free_ram_mb() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.virtual_memory().available / 1024**2)


def print_resources() -> None:
    print("📊 Resources:")
    print(f"   GPU: {get_gpu_name() or 'None'}")
    print(f"   VRAM free/total: {get_free_vram_mb()} / {get_total_vram_mb()} MB")
    print(f"   RAM free: {get_free_ram_mb()} MB")


# -----------------------------
# CUDA hygiene between retries
# -----------------------------


def drop_caches() -> None:
    print("🧹 Dropping FS caches (requires sudo)...")
    sh("sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'", check=False)


def gpu_reset(idx: int = 0) -> None:
    if not has_cmd("nvidia-smi"):
        print("⚠️ nvidia-smi not found; GPU reset skipped.")
        return
    print("🔧 Resetting GPU (if supported)...")
    sh(f"sudo nvidia-smi --gpu-reset -i {idx}", check=False)


def _terminate_matching_processes(pattern: str) -> None:
    """Terminate processes matching ``pattern`` without killing ourselves."""

    regex = re.compile(pattern)
    try:
        import psutil
    except Exception:
        sh(f"pkill -f {shlex.quote(pattern)} || true", check=False)
        return

    this_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        pid = proc.info.get("pid")
        if pid == this_pid:
            continue
        cmdline = " ".join(proc.info.get("cmdline") or [])
        if not cmdline or not regex.search(cmdline):
            continue
        try:
            proc.terminate()
        except Exception:
            continue


def kill_gpu_processes() -> None:
    if not has_cmd("nvidia-smi"):
        return
    print("🛑 Killing leftover GPU processes (if any)...")
    _terminate_matching_processes(r"python .*build_and_wrap.py")
    _terminate_matching_processes(r"python [^\n]*(?:/|\s)train(?:\.py)?")


# -----------------------------
# OOM detection
# -----------------------------


OOM_PATTERNS = [
    r"CUDA out of memory",
    r"CUDA OOM",
    r"RuntimeError:.*out of memory",
    r"c10::Error.*out of memory",
    r"\bOOM\b",
]

# -----------------------------
# Adaptive config model
# -----------------------------


@dataclass
class TrainKnobs:
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    per_device_eval_batch_size: int = 4
    max_seq_length: int = 2048
    eval_max_seq_length: int = 2048
    torch_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    attention_implementation: str = "flash_attention_2"
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def smaller(self) -> "TrainKnobs":
        new = TrainKnobs(**asdict(self))
        if new.per_device_train_batch_size > 1:
            new.per_device_train_batch_size = max(
                1, new.per_device_train_batch_size // 2
            )
        else:
            new.gradient_accumulation_steps *= 2
        new.max_seq_length = max(512, new.max_seq_length // 2)
        new.eval_max_seq_length = max(512, new.eval_max_seq_length // 2)
        if new.attention_implementation == "flash_attention_2":
            new.attention_implementation = "eager"
        return new

    def cli_overrides(self) -> List[str]:
        args = [
            f"--per_device_train_batch_size={self.per_device_train_batch_size}",
            f"--gradient_accumulation_steps={self.gradient_accumulation_steps}",
            f"--per_device_eval_batch_size={self.per_device_eval_batch_size}",
            f"--max_seq_length={self.max_seq_length}",
            f"--evaluation_strategy=steps",
            f"--save_strategy=steps",
            f"--logging_strategy=steps",
            f"--gradient_checkpointing={'true' if self.gradient_checkpointing else 'false'}",
            f"--torch_dtype={self.torch_dtype}",
        ]
        if self.attention_implementation:
            args.append(f"--attn_impl={self.attention_implementation}")
        args.extend(
            [
                f"--use_4bit={'true' if self.use_4bit else 'false'}",
                f"--bnb_4bit_quant_type={self.bnb_4bit_quant_type}",
                f"--bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}",
                f"--lora_r={self.lora_r}",
                f"--lora_alpha={self.lora_alpha}",
                f"--lora_dropout={self.lora_dropout}",
            ]
        )
        return args

    def env_overrides(self) -> Dict[str, str]:
        return {
            "OVR_PER_DEVICE_TRAIN_BATCH_SIZE": str(self.per_device_train_batch_size),
            "OVR_GRAD_ACCUM_STEPS": str(self.gradient_accumulation_steps),
            "OVR_PER_DEVICE_EVAL_BATCH_SIZE": str(self.per_device_eval_batch_size),
            "OVR_MAX_SEQ_LEN": str(self.max_seq_length),
            "OVR_EVAL_MAX_SEQ_LEN": str(self.eval_max_seq_length),
            "OVR_TORCH_DTYPE": self.torch_dtype,
            "OVR_GRAD_CHECKPOINT": "1" if self.gradient_checkpointing else "0",
            "OVR_ATTN_IMPL": self.attention_implementation or "",
            "OVR_USE_4BIT": "1" if self.use_4bit else "0",
            "OVR_BNB_QUANT": self.bnb_4bit_quant_type,
            "OVR_BNB_COMP_DTYPE": self.bnb_4bit_compute_dtype,
            "OVR_LORA_R": str(self.lora_r),
            "OVR_LORA_ALPHA": str(self.lora_alpha),
            "OVR_LORA_DROPOUT": str(self.lora_dropout),
        }


def initial_knobs_for_vram(free_mb: int) -> TrainKnobs:
    kb = TrainKnobs()
    if free_mb >= 11000:
        kb.per_device_train_batch_size = 4
        kb.gradient_accumulation_steps = 4
        kb.max_seq_length = 4096
        kb.eval_max_seq_length = 4096
    elif free_mb >= 8000:
        kb.per_device_train_batch_size = 4
        kb.gradient_accumulation_steps = 4
        kb.max_seq_length = 3072
        kb.eval_max_seq_length = 3072
    elif free_mb >= 6000:
        kb.per_device_train_batch_size = 2
        kb.gradient_accumulation_steps = 8
        kb.max_seq_length = 2048
        kb.eval_max_seq_length = 2048
    elif free_mb >= 4500:
        kb.per_device_train_batch_size = 1
        kb.gradient_accumulation_steps = 16
        kb.max_seq_length = 2048
        kb.eval_max_seq_length = 2048
    else:
        kb.per_device_train_batch_size = 1
        kb.gradient_accumulation_steps = 32
        kb.max_seq_length = 1024
        kb.eval_max_seq_length = 1024
        kb.attention_implementation = "eager"
    return kb


# -----------------------------
# Training attempt loop
# -----------------------------


def base_env(device: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    if device == "cuda":
        env.setdefault("TORCH_DTYPE_FALLBACK", "bfloat16")
        env["CUDA_LAUNCH_BLOCKING"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def write_overrides_json(knobs: TrainKnobs) -> str:
    data = {"trainer_overrides": asdict(knobs), "timestamp": time.time()}
    fd, path = tempfile.mkstemp(prefix="trainer_overrides_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


def run_training_once(
    command: str, knobs: TrainKnobs, device: str, extra_cli: List[str]
) -> Tuple[int, str, str]:
    env = base_env(device)
    env.update(knobs.env_overrides())
    overrides_path = write_overrides_json(knobs)
    env["TRAINER_OVERRIDES_JSON"] = overrides_path
    final_cmd = " ".join([command.strip()] + knobs.cli_overrides() + list(extra_cli))
    print(f"🚀 Launching training with overrides:\n    {final_cmd}\n")
    result = sh(final_cmd, check=False, env=env)
    return result.returncode, result.stdout, result.stderr


def autotrain(
    command: str,
    extra_cli: List[str],
    max_retries: int,
    allow_cpu_fallback: bool,
    gpu_index: int,
) -> int:
    print_resources()
    free_mb = get_free_vram_mb()
    total_mb = get_total_vram_mb()

    if total_mb == 0:
        print(
            "⚠️ No NVIDIA GPU detected by nvidia-smi. You can still run on CPU with --allow-cpu-fallback."
        )
        knobs = initial_knobs_for_vram(0)
        rc, out, err = run_training_once(command, knobs, "cpu", extra_cli)
        print(out)
        print(err, file=sys.stderr)
        return rc

    device = "cuda"
    knobs = initial_knobs_for_vram(free_mb)
    print(f"🎯 VRAM free: {free_mb} MB (total {total_mb} MB). Initial knobs: {knobs}")

    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        print(f"\n===== Attempt {attempt}/{max_retries + 1} on {device.upper()} =====")
        kill_gpu_processes()
        drop_caches()
        gpu_reset(gpu_index)
        sleep(2.0)

        rc, out, err = run_training_once(command, knobs, device, extra_cli)

        if rc == 0 and not looks_like_oom(out, err):
            print("✅ Training finished successfully.")
            print(out)
            return 0

        if looks_like_oom(out, err):
            print("⚠️ Detected OOM. Tightening knobs and retrying...")
            knobs = knobs.smaller()
            sh(
                "python - <<'PY'\nimport torch\ntry:\n torch.cuda.empty_cache()\nexcept Exception: pass\nPY",
                check=False,
            )
            sleep(1.0)
            continue

        print("❌ Training failed (non-OOM). See logs below.")
        print(out)
        print(err, file=sys.stderr)
        return rc

    if allow_cpu_fallback:
        print(
            "\n↩️  Exhausted GPU attempts. Falling back to CPU to guarantee completion."
        )
        device = "cpu"
        knobs.gradient_checkpointing = True
        knobs.attention_implementation = "eager"
        knobs.per_device_train_batch_size = 1
        knobs.gradient_accumulation_steps = max(knobs.gradient_accumulation_steps, 32)
        knobs.max_seq_length = min(knobs.max_seq_length, 1024)
        knobs.eval_max_seq_length = min(knobs.eval_max_seq_length, 1024)

        rc, out, err = run_training_once(command, knobs, device, extra_cli)
        print(out)
        print(err, file=sys.stderr)
        return rc

    print("❗ Ran out of retries on GPU and CPU fallback disabled.")
    return 2


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-adaptive headless trainer")
    parser.add_argument(
        "--prepare-headless",
        action="store_true",
        help="Persist multi-user (headless) target. Requires reboot to take effect.",
    )
    parser.add_argument(
        "--isolate-headless",
        action="store_true",
        help="Switch to headless target for the current session without reboot.",
    )
    parser.add_argument(
        "--reboot", action="store_true", help="Reboot after changing default target."
    )
    parser.add_argument(
        "--restore-gui", action="store_true", help="Restore graphical default target."
    )
    parser.add_argument(
        "--command",
        default="python build_and_wrap.py",
        help="Training command to run. Default: %(default)s",
    )
    parser.add_argument(
        "--extra-cli",
        default="",
        help="Extra CLI arguments appended to the training command.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum GPU retries before fallback. Default: %(default)s",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permit CPU-only fallback after exhausting GPU attempts.",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to target.")

    args = parser.parse_args()

    if args.prepare_headless:
        switch_to_headless(persist=True)
        if args.reboot:
            reboot_now()
            return
        print("ℹ️  Headless target set. Reboot (or pass --reboot) for it to apply.")
        return

    if args.isolate_headless:
        switch_to_headless(persist=False)
        return

    if args.restore_gui:
        restore_gui_default()
        return

    extra_cli = shlex.split(args.extra_cli) if args.extra_cli else []
    rc = autotrain(
        command=args.command,
        extra_cli=extra_cli,
        max_retries=args.max_retries,
        allow_cpu_fallback=args.allow_cpu_fallback,
        gpu_index=args.gpu_index,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
