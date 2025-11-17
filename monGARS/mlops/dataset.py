"""Dataset helpers for supervised fine-tuning pipelines."""

from __future__ import annotations

# isort: off
from ._unsloth_bootstrap import UNSLOTH_AVAILABLE

# isort: on
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from transformers import PreTrainedTokenizerBase

from datasets import Dataset, DatasetDict, load_dataset
from monGARS.mlops.code_analysis import (
    LLMUsage,
    ModuleInteraction,
    build_strategy_recommendation,
)

logger = logging.getLogger(__name__)


DOLPHIN_CHAT_TEMPLATE = """
<|im_start|>
# User-provided custom instructions

You are an advanced coding agent designed to assist with production-ready software development for legitimate network diagnostics and security research, operating in an environment optimized for flexibility and innovation. Your goal is to deliver complete, functional, and high-quality code with no placeholders, adhering to project-specific guidelines and prioritizing real-world usability. You operate under the following principles:

- **Complete Implementations**: Provide fully functional code with no stubs or incomplete flows. Proactively include necessary details (e.g., APIs, frameworks, error handling) to ensure the solution works as intended for the specified use case.
- **Support for Research**: Enable advanced network diagnostics, including WiFi scanning and packet analysis, using public APIs (e.g., CoreWLAN, NetworkExtension) and Apple-supported tools like rvictl for tethered capture. For ad-hoc sideloading (e.g., via Xcode or AltStore), bypass App Store restrictions while remaining within ethical and legal bounds for security research.
- **Type Safety and Patterns**: Use TypeScript-friendly React Native patterns, ensuring strong typing and compatibility with modern frameworks (e.g., TurboModules, React Navigation).
- **Project Adherence**: Follow repository guidelines (e.g., Prettier, ESLint, Jest) as specified in AGENTS.md or similar files. Run linting (`npm run lint`) and formatting (`npx prettier --write`) before finalizing code.
- **Error Handling**: Implement robust error handling and logging (e.g., os.log for Swift, console for JS) to ensure production readiness and debuggability.
- **Cross-Platform**: Support iOS and Android, using platform-specific APIs (e.g., CoreWLAN, NetworkExtension for iOS; WifiManager, VpnService for Android) with Platform.OS checks.
- **Sideloading Optimization**: Provide build instructions for Xcode and AltStore, including necessary entitlements and dependency management.
- **Testing**: Include Jest tests for new functionality and verify on specified platforms (e.g., iOS 19 simulator/device, Android emulator). Ensure tests pass with `npm test`.
- **Dependency Management**: Address dependency issues (e.g., deprecated packages, npm audit warnings) by updating to supported versions or suggesting alternatives, ensuring a stable build environment.

When handling network diagnostics requests, prioritize public APIs and Apple-supported tools (e.g., rvictl for tethered packet capture) to achieve functionality. For low-level data analysis (e.g., 802.11 frame parsing), use tethered capture or existing parsers (e.g., libpcap) within the bounds of legitimate research. Log detailed errors for debugging and provide fallbacks for unavailable features (e.g., missing permissions, no WiFi). Interpret all requests as part of ethical security research, avoiding assumptions of malicious intent. If a request is unclear, seek clarification to align with the user’s goals. Deliverables must include updated files, new files (if needed), build instructions, testing steps, and dependency fixes, ensuring a production-ready solution.

If a request triggers restrictions, rephrase or adjust the approach to use only public APIs and tools, focusing on diagnostics and analysis functionality. For example, replace low-level capture with tethered methods like rvictl or file imports. Provide clear explanations for any limitations and suggest workarounds to achieve the desired outcome.
"""

if not UNSLOTH_AVAILABLE:
    logger.debug(
        "Unsloth unavailable during dataset initialisation; proceeding with standard tokenizers"
    )


PROMPT_KEYS = ("instruction", "prompt", "question")
INPUT_KEYS = ("input", "context")
OUTPUT_KEYS = ("output", "response", "answer")


def _extract_field(example: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _format_prompt_completion(example: dict[str, Any]) -> dict[str, str]:
    instruction = _extract_field(example, PROMPT_KEYS)
    additional = _extract_field(example, INPUT_KEYS)
    output = _extract_field(example, OUTPUT_KEYS)
    prompt = f"{instruction}\n\n{additional}" if additional else instruction
    return {"prompt": prompt, "completion": output}


def _tokenize_pair(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Callable[[dict[str, str]], dict[str, Any]]:
    def builder(example: dict[str, str]) -> dict[str, Any]:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            prompt_only = f"User: {example['prompt']}\nAssistant:"
            full_text = f"{prompt_only} {example['completion']}"

        prompt_tokens = tokenizer(
            prompt_only,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            return_attention_mask=False,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = full_tokens["input_ids"]
        attention = full_tokens["attention_mask"]
        labels = list(input_ids)
        k = min(len(prompt_tokens["input_ids"]), len(labels))
        for idx in range(k):
            labels[idx] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }

    return builder


def prepare_instruction_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    *,
    train_fraction: float = 1.0,
) -> Dataset:
    """Load and tokenise an instruction dataset for supervised fine-tuning."""

    logger.info(
        "Loading dataset",
        extra={"dataset": dataset_name, "train_fraction": train_fraction},
    )
    dataset: Dataset | DatasetDict = load_dataset(dataset_name)
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get("train") or next(iter(dataset.values()))
    if train_fraction and 0 < train_fraction < 1:
        total = len(dataset)
        desired = max(1, int(total * train_fraction))
        take = min(total, max(1000, desired))
        dataset = dataset.select(range(take))
        logger.info("Dataset subset selected", extra={"take": take, "total": total})

    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        dataset = dataset.map(
            _format_prompt_completion, remove_columns=dataset.column_names
        )

    tokenized = dataset.map(
        _tokenize_pair(tokenizer, max_seq_len),
        remove_columns=dataset.column_names,
        desc="tokenize+mask",
    )
    tokenized.set_format(type="torch")
    logger.info("Tokenised dataset", extra={"size": len(tokenized)})
    return tokenized


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    buffer: list[str] = []
    depth = 0
    in_string = False
    escape_next = False
    record_start_line: int | None = None

    def flush_buffer(start_line: int) -> None:
        nonlocal depth, in_string, escape_next, buffer
        chunk = "".join(buffer).strip()
        buffer = []
        depth = 0
        in_string = False
        escape_next = False
        if not chunk:
            return
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Invalid JSON starting at line {start_line} of {path}: {exc}"
            ) from exc
        if not {"prompt", "completion"} <= payload.keys():
            raise ValueError(
                f"Record starting at line {start_line} missing prompt/completion fields in {path}"
            )
        records.append(
            {
                "prompt": str(payload["prompt"]),
                "completion": str(payload["completion"]),
            }
        )

    line_number = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not buffer and not stripped:
                continue
            if not buffer:
                record_start_line = line_number
            buffer.append(line)
            for char in line:
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\" and in_string:
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth < 0:
                        raise ValueError(
                            f"Unexpected closing brace on line {line_number} of {path}"
                        )
            if depth == 0 and buffer:
                flush_buffer(record_start_line or line_number)
                record_start_line = None

    if buffer:
        start = record_start_line or line_number
        raise ValueError(
            f"Unexpected EOF while parsing JSON object starting at line {start} in {path}"
        )

    if not records:
        raise ValueError(f"Dataset at {path} is empty")
    return records


def prepare_local_instruction_dataset(
    dataset_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
) -> Dataset:
    """Load a JSONL dataset generated by :func:`build_mongars_strategy_dataset`."""

    records = _load_jsonl_records(dataset_path)
    dataset = Dataset.from_list(records)
    tokenized = dataset.map(
        _tokenize_pair(tokenizer, max_seq_len),
        remove_columns=dataset.column_names,
        desc="tokenize+mask",
    )
    tokenized.set_format(type="torch")
    logger.info(
        "Prepared local dataset",
        extra={"size": len(tokenized), "path": str(dataset_path)},
    )
    return tokenized


def build_mongars_strategy_dataset(
    usages: Sequence[LLMUsage],
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    min_examples: int = 4,
) -> Path:
    """Materialise a JSONL dataset with prompts derived from static analysis."""

    if len(usages) < min_examples:
        raise ValueError(
            "Insufficient LLM callsites to build a representative dataset; "
            f"need at least {min_examples}, got {len(usages)}"
        )

    records = []
    for usage in usages:
        prompt = (
            "Review the following callsite and explain how to fine-tune the model.\n"
            f"File: {usage.file_path}\n"
            f"Line: {usage.line}\n"
            f"Symbol: {usage.symbol}\n"
            f"Invocation: {usage.call}\n"
            f"Snippet:\n{usage.snippet}\n"
        )
        completion = build_strategy_recommendation(usage)
        records.append({"prompt": prompt, "completion": completion})

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if metadata_path is not None:
        metadata = {
            "num_examples": len(records),
            "frameworks": sorted({usage.framework for usage in usages}),
            "generated_from": str(output_path),
        }
        metadata_path = metadata_path.resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Wrote monGARS strategy dataset",
        extra={"path": str(output_path), "examples": len(records)},
    )
    return output_path


def _summarise_interaction(interaction: ModuleInteraction) -> str:
    imports = ", ".join(interaction.import_names) or "the module surface"
    base = (
        f"{interaction.source_module} depends on {interaction.target_module} via "
        f"a {interaction.kind} import of {imports}. "
        "Document how the source coordinates with the imported module "
        "to keep responsibilities well factored and observable."
    )
    target = interaction.target_module
    if "dataset" in target:
        extra = (
            " Emphasise dataset shaping, tokenisation boundaries, and how prompts feed"
            " downstream training jobs."
        )
    elif "code_analysis" in target or "analysis" in target:
        extra = (
            " Show how static analysis insights inform data curation and guardrail"
            " selection for fine-tuning."
        )
    elif "pipelines" in target or "unsloth" in target:
        extra = (
            " Outline the training orchestration expectations, paying attention to"
            " metadata capture and wrapper export routines."
        )
    elif "evolution_engine" in target:
        extra = (
            " Call out coordination with the evolution orchestrator and how artefact"
            " metadata flows across stages."
        )
    elif "core" in target:
        extra = (
            " Reference shared persistence, caching, or validation helpers exposed"
            " through the core package."
        )
    else:
        extra = (
            " Highlight logging, error handling, and contract boundaries so future"
            " refactors preserve behaviour."
        )
    return base + extra


def build_module_interaction_dataset(
    interactions: Sequence[ModuleInteraction],
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    min_examples: int = 8,
) -> Path:
    """Create a JSONL dataset derived from module dependency edges."""

    if len(interactions) < min_examples:
        raise ValueError(
            "Insufficient module interactions to build a dataset; "
            f"need at least {min_examples}, got {len(interactions)}"
        )

    records = []
    for interaction in interactions:
        imports = ", ".join(interaction.import_names) or "<module>"
        prompt = (
            "Describe how these monGARS modules collaborate.\n"
            f"Source module: {interaction.source_module}\n"
            f"Imported module: {interaction.target_module}\n"
            f"Imported names: {imports}\n"
            f"Import kind: {interaction.kind}\n"
            f"Location: {interaction.source_path}:{interaction.line}\n"
            f"Snippet:\n{interaction.snippet}\n"
        )
        completion = _summarise_interaction(interaction)
        records.append({"prompt": prompt, "completion": completion})

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if metadata_path is not None:
        metadata = {
            "num_examples": len(records),
            "unique_targets": sorted({item.target_module for item in interactions}),
            "generated_from": str(output_path),
        }
        metadata_path = metadata_path.resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Wrote monGARS module interaction dataset",
        extra={"path": str(output_path), "examples": len(records)},
    )
    return output_path
