#!/usr/bin/env python3
"""
Consolidated Advanced Dataset Pipeline for Multi-Task Learning (French-Only)
===============================================================
A robust, production-grade pipeline for aggregating, processing, and validating
instruction and retrieval datasets in French.
Features comprehensive error handling, memory optimization, license compliance,
real-time monitoring, and deduplication.

Combined features from both original scripts:
- Advanced logging, configuration management, and error handling (from script 2)
- Simple but effective progress tracking with ETA calculations (from script 1)
- Comprehensive PII detection and license compliance (from script 2)
- Multiple dataset loaders with checkpointing (from script 2)
- Memory optimization and deduplication strategies (both scripts)
- French-specific optimizations and dataset configurations
"""

import argparse
import gc
import hashlib
import itertools
import json
import logging
import logging.handlers
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
import psutil
from tqdm import tqdm

# Third-party imports
import datasets
from datasets import (
    Dataset,
    DatasetDict,
    DownloadMode,
    VerificationMode,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

# Optional dependency on datasketch for near-duplicate detection
try:
    from datasketch import MinHash, MinHashLSH

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    MinHash = None
    MinHashLSH = None

# Optional dependency on boto3 for cloud storage
try:
    import boto3

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None

# Optional resource module for memory tracking (Unix-like systems)
try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False
    resource = None


# -----------------------------
# Advanced Logging Configuration
# -----------------------------
class ColorFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(
    log_file: str = "dataset_pipeline.log",
    verbose: bool = False,
    log_level: str = "INFO",
) -> logging.Logger:
    """
    Configure comprehensive logging with rotation, color coding, and structured output.
    Args:
        log_file: Path to log file (rotated at 100MB)
        verbose: Enable debug level logging
        log_level: Base logging level
    Returns:
        Configured logger instance
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Set log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(log_level.upper(), logging.INFO)
    if verbose:
        log_level = logging.DEBUG

    root_logger.setLevel(log_level)

    # File formatter (structured JSON for machine parsing)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )

    # Console formatter (color-coded for human readability)
    console_formatter = ColorFormatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Rotating file handler (100MB max size, 10 backups)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("dataset_pipeline")


logger = setup_logging()


# -----------------------------
# Configuration Management
# -----------------------------
class DatasetType(str, Enum):
    """Enum for dataset types with clear categorization."""

    INSTRUCTION = "instruction"
    REASONING = "reasoning"
    DIALOG = "dialog"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class LicenseType(str, Enum):
    """Common license types with predefined compliance rules."""

    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    CC_BY_4_0 = "CC BY 4.0"
    CC_BY_SA_3_0 = "CC BY-SA 3.0"
    CC_BY_NC_4_0 = "CC BY-NC 4.0"
    CC0_1_0 = "CC0-1.0"
    RESEARCH_ONLY = "research-only"
    CUSTOM = "custom"


@dataclass
class LicenseCompliance:
    """License compliance configuration with validation rules."""

    license_type: LicenseType = LicenseType.CUSTOM
    requires_attribution: bool = True
    allows_commercial_use: bool = False
    allows_modification: bool = True
    attribution_text: str = ""
    custom_terms: str = ""

    def validate(self, config: "PipelineConfig") -> List[str]:
        """Validate license compliance against pipeline configuration."""
        issues = []
        if self.requires_attribution and not self.attribution_text:
            issues.append("Missing attribution text for license requiring attribution")
        if not self.allows_commercial_use and config.enable_commercial_use:
            issues.append(
                "Commercial use not allowed by license but enabled in pipeline"
            )
        if not self.allows_modification and config.enable_modification:
            issues.append("Modification not allowed by license but enabled in pipeline")
        return issues


@dataclass
class DatasetConfig:
    """Configuration for a single dataset source."""

    name: str
    hf_path: str
    hf_config: Optional[str] = None
    dataset_type: DatasetType = DatasetType.INSTRUCTION
    languages: List[str] = field(default_factory=lambda: ["fr"])  # French only
    license: LicenseType = LicenseType.CUSTOM
    license_compliance: LicenseCompliance = field(default_factory=LicenseCompliance)
    source_url: str = ""
    notes: str = ""
    streaming: bool = False
    max_examples: Optional[int] = None
    filter_function: Optional[Callable[[Dict], bool]] = None
    transform_function: Optional[Callable[[Dict], Dict]] = None
    requires_auth: bool = False
    timeout: int = 300
    quality_weight: float = 1.0
    sampling_strategy: Literal["uniform", "quality_weighted", "stratified"] = "uniform"
    batch_size: int = 1000
    cache_strategy: Literal["memory", "disk", "hybrid"] = "hybrid"
    requires_trust_remote_code: bool = False
    allow_trust_remote_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization, excluding functions."""
        return {
            "name": self.name,
            "hf_path": self.hf_path,
            "hf_config": self.hf_config,
            "dataset_type": self.dataset_type.value,
            "languages": self.languages,
            "license": self.license.value,
            "source_url": self.source_url,
            "notes": self.notes,
            "streaming": self.streaming,
            "max_examples": self.max_examples,
            "requires_auth": self.requires_auth,
            "timeout": self.timeout,
            "quality_weight": self.quality_weight,
            "sampling_strategy": self.sampling_strategy,
            "batch_size": self.batch_size,
            "cache_strategy": self.cache_strategy,
            "requires_trust_remote_code": self.requires_trust_remote_code,
            "allow_trust_remote_code": self.allow_trust_remote_code,
            "license_compliance": {
                "license_type": self.license_compliance.license_type.value,
                "requires_attribution": self.license_compliance.requires_attribution,
                "allows_commercial_use": self.license_compliance.allows_commercial_use,
                "allows_modification": self.license_compliance.allows_modification,
                "attribution_text": self.license_compliance.attribution_text,
                "custom_terms": self.license_compliance.custom_terms,
            },
        }


@dataclass
class MemoryConfig:
    """Memory management configuration for large-scale processing."""

    max_memory_gb: float = 32.0
    gc_threshold_mb: float = 512.0  # Trigger GC when memory exceeds this threshold
    batch_memory_limit_mb: float = 256.0
    cache_clear_interval: int = 1000  # Clear cache every N examples
    aggressive_gc: bool = False
    memory_check_interval: int = 100  # Check memory every N examples
    memory_release_strategy: Literal["aggressive", "balanced", "conservative"] = (
        "aggressive"
    )


@dataclass
class DedupConfig:
    """Configuration for deduplication strategies."""

    enable_exact_dedup: bool = True
    enable_near_dedup: bool = False
    near_dedup_threshold: float = 0.8
    minhash_permutations: int = 128
    preserve_high_quality: bool = True
    quality_threshold: float = 0.7


@dataclass
class QualityConfig:
    """Configuration for data quality filtering."""

    min_chars: int = 5  # Reduced for French
    max_chars: int = 10000
    min_words: int = 1  # Reduced for French
    max_words: int = 2000
    min_quality_score: float = 0.3  # Reduced to capture more data initially
    language_detection_threshold: float = 0.5  # Reduced for more flexibility
    enable_spam_detection: bool = True
    enable_toxicity_filtering: bool = False


@dataclass
class PIIConfig:
    """Configuration for PII detection and filtering."""

    enabled: bool = True
    sensitive_patterns: List[str] = field(
        default_factory=lambda: [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
            r"\b\d{9}\b",  # Passport
            r"\b[A-Z]{2}\d{6}\b",  # Passport variant
            r"\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b",  # Driver's license
            r"(?i)(password|pwd|pass)\s*[:=]\s*[^\s]+",  # Password patterns
            r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[^\s]+",  # API keys
            r"(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[=:]",  # AWS credentials
            r"(?i)(private[_-]?key|ssh[_-]?key)\s*[:=]\s*",  # Private keys
        ]
    )
    redaction_strategy: Literal["remove", "mask", "replace"] = "remove"
    context_window: int = 50  # Characters around PII to include in detection context


@dataclass
class PipelineConfig:
    """Main pipeline configuration with comprehensive settings."""

    output_dir: str
    langs: List[str] = field(default_factory=lambda: ["fr"])  # French only
    max_per_dataset: int = 50000
    metadata_file: str = "dataset_metadata.json"
    seed: int = 42
    cache_dir: Optional[str] = None
    num_workers: int = 4
    download_mode: str = "reuse_dataset_if_exists"
    quality_config: QualityConfig = field(default_factory=QualityConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    dedup_config: DedupConfig = field(default_factory=DedupConfig)
    pii_config: PIIConfig = field(default_factory=PIIConfig)
    enable_progress_bar: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 5000
    enable_commercial_use: bool = False
    enable_modification: bool = True
    dashboard_port: Optional[int] = None
    log_level: str = "INFO"
    # Remove trust_remote_code as it's deprecated
    force_download: bool = False
    allow_trust_remote_code: bool = False
    trusted_remote_code_datasets: Set[str] = field(default_factory=set)
    state_file: str = "pipeline_state.pkl"
    dedup_state_file: str = "dedup_state.pkl"
    resume_from_checkpoint: bool = False
    ignore_failed_datasets: bool = False
    sampling_strategy: Literal["uniform", "quality_weighted", "adaptive"] = "uniform"
    quality_threshold: float = 0.6
    min_examples_per_language: int = 1000
    max_examples_per_language: int = 100000
    enable_validation: bool = True
    validation_split: float = 0.05
    validation_max_size: int = 5000
    verbose: bool = False
    unsloth_export_name: str = "unsloth_prompt_completion.jsonl"
    cloud_storage: Optional[Dict[str, str]] = None
    distributed_mode: bool = False
    node_rank: int = 0
    world_size: int = 1
    health_check_interval: int = 60  # seconds
    max_runtime_hours: float = 24.0
    auto_scaling: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.distributed_mode:
            if self.world_size <= 0:
                raise ValueError(
                    "world_size must be greater than 0 when distributed mode is enabled"
                )
            if self.node_rank < 0 or self.node_rank >= self.world_size:
                raise ValueError(
                    "node_rank must be within [0, world_size) when distributed mode is enabled"
                )

        # Ensure French is in selected languages
        if "fr" not in self.langs:
            self.langs.append("fr")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["trusted_remote_code_datasets"] = sorted(self.trusted_remote_code_datasets)
        return data


# -----------------------------
# Core Utilities with Enhanced Error Handling
# -----------------------------
T = TypeVar("T")
U = TypeVar("U")


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    pass


class DatasetLoadingError(PipelineError):
    """Exception raised when dataset loading fails."""

    pass


class DataProcessingError(PipelineError):
    """Exception raised when data processing fails."""

    pass


class MemoryLimitExceeded(PipelineError):
    """Exception raised when memory limits are exceeded."""

    pass


class LicenseComplianceError(PipelineError):
    """Exception raised when license compliance issues are detected."""

    pass


def normalize_text(text: str) -> str:
    """
    Normalize text for deduplication with advanced cleaning.
    Args:
        text: Input text to normalize
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    # Convert to lowercase and normalize whitespace
    normalized = " ".join(text.lower().split())
    # Remove common punctuation that doesn't affect meaning
    normalized = re.sub(r"[^\w\s\'-]", " ", normalized)
    # Collapse multiple spaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def make_instruct_key(instr: str, out: str) -> str:
    """Create a normalized key for instruction-output pairs."""
    return normalize_text(f"{instr.strip()} ||| {out.strip()}")


def make_retrieval_key(t1: str, t2: str, symmetric: bool = True) -> str:
    """
    Create a normalized key for retrieval pairs.
    Args:
        t1: First text
        t2: Second text
        symmetric: If True, order doesn't matter (default: True)
    Returns:
        Normalized key string
    """
    if symmetric:
        a, b = sorted([t1.strip(), t2.strip()])
        return normalize_text(f"{a} ||| {b}")
    return normalize_text(f"{t1.strip()} ||| {t2.strip()}")


def safe_get(item: Dict[str, Any], *keys: str, min_length: int = 3) -> Optional[str]:
    """
    Safely get the first non-empty string value from multiple keys.
    Args:
        item: Dictionary to search
        keys: Keys to try in order
        min_length: Minimum length for valid string
    Returns:
        First valid string found or None
    """
    for k in keys:
        v = item.get(k)
        if isinstance(v, str) and len(v.strip()) >= min_length:
            return v.strip()
    return None


def normalise_instruction_record(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Transform heterogeneous instruction records into prompt/completion pairs."""
    prompt = record.get("prompt")
    completion = record.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        prompt = prompt.strip()
        completion = completion.strip()
        if prompt and completion:
            return {"prompt": prompt, "completion": completion}

    instruction = safe_get(
        record,
        "instruction",
        "question",
        "inputs",
        "query",
        min_length=1,
    )
    supplemental = safe_get(record, "input", "context", "source", "text", min_length=1)
    output = safe_get(record, "output", "response", "answer", min_length=1)

    prompt_parts = []
    if instruction:
        prompt_parts.append(instruction.strip())
    if supplemental:
        prompt_parts.append(supplemental.strip())

    prompt_text = "\n".join(part for part in prompt_parts if part)
    if prompt_text and output:
        return {"prompt": prompt_text, "completion": output.strip()}

    return None


def validate_text(
    text: str, config: PipelineConfig, quality_score: float = 1.0
) -> bool:
    """
    Validate text quality based on configurable thresholds and quality score.
    Args:
        text: Text to validate
        config: Pipeline configuration
        quality_score: Current quality score (0.0-1.0)
    Returns:
        True if text passes validation
    """
    if not text or not isinstance(text, str):
        return False

    text_clean = text.strip()
    length = len(text_clean)

    # Basic length checks (relaxed for French)
    if (
        length < config.quality_config.min_chars
        or length > config.quality_config.max_chars
    ):
        return False

    # Word count checks (relaxed for French)
    words = text_clean.split()
    word_count = len(words)
    if (
        word_count < config.quality_config.min_words
        or word_count > config.quality_config.max_words
    ):
        return False

    # Quality score threshold
    if quality_score < config.quality_threshold:
        return False

    return True


def load_dataset_with_retry(
    hf_path: str,
    hf_config: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    max_retries: int = 5,
    base_retry_delay: float = 1.0,
    timeout: int = 300,
    requires_auth: bool = False,
    trust_remote_code: bool = False,
    download_mode: DownloadMode = DownloadMode.REUSE_DATASET_IF_EXISTS,
) -> Union[Dataset, DatasetDict, Iterable[Dict]]:
    """
    Load dataset with robust retry logic and error handling.
    Args:
        hf_path: HuggingFace dataset path
        hf_config: Dataset configuration name
        split: Dataset split to load
        streaming: Enable streaming mode
        cache_dir: Directory to cache datasets
        max_retries: Maximum retry attempts
        base_retry_delay: Base delay between retries (exponential backoff)
        timeout: Timeout for operations
        requires_auth: Whether authentication is required
        download_mode: Dataset download mode
    Returns:
        Loaded dataset object
    Raises:
        DatasetLoadingError: If dataset fails to load after retries
    """
    logger.info(f"Loading dataset: {hf_path} ({hf_config or 'default'})")

    # Set environment variables for caching
    if cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir

    token = True if requires_auth else None

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            ds_kwargs: Dict[str, Any] = dict(
                path=hf_path,
                name=hf_config,
                streaming=streaming,
                cache_dir=cache_dir,
                download_mode=download_mode,
                verification_mode=VerificationMode.NO_CHECKS,
            )
            if trust_remote_code:
                ds_kwargs["trust_remote_code"] = True
            if token:
                ds_kwargs["token"] = token

            if split:
                ds = load_dataset(split=split, **ds_kwargs)
            else:
                ds = load_dataset(**ds_kwargs)

            elapsed = time.time() - start_time
            logger.info(f"Successfully loaded {hf_path} in {elapsed:.2f} seconds")
            return ds
        except Exception as e:
            error_str = str(e)
            if (
                "trust_remote_code" in error_str
                or "Dataset scripts are no longer supported" in error_str
            ):
                logger.error(
                    "Dataset %s requires trust_remote_code; rerun with --allow_trust_remote_code if you trust the source."
                    % hf_path
                )
                raise DatasetLoadingError(
                    f"Dataset {hf_path} requires trust_remote_code. Pass allow_trust_remote_code to enable."
                ) from e

            logger.error(
                f"Failed to load {hf_path} (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            if attempt < max_retries - 1:
                delay = base_retry_delay * (2**attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue

            error_msg = f"Failed to load dataset {hf_path} after {max_retries} attempts: {str(e)}"
            logger.error(error_msg)
            raise DatasetLoadingError(error_msg) from e

    raise DatasetLoadingError(f"Dataset loading failed for {hf_path} after all retries")


def estimate_dataset_quality(
    examples: Iterable[Dict], config: PipelineConfig, sample_size: int = 1000
) -> float:
    """
    Estimate dataset quality by sampling examples and applying validation rules.
    Args:
        examples: Iterable of dataset examples
        config: Pipeline configuration
        sample_size: Number of examples to sample
    Returns:
        Quality score between 0.0 and 1.0
    """
    valid_count = 0
    total_count = 0

    # Convert to list if needed for sampling
    if hasattr(examples, "__len__"):
        examples = list(examples)
        sample_size = min(sample_size, len(examples))
        sampled_examples = random.sample(examples, sample_size)
    else:
        # For streaming datasets, take first N examples
        sampled_examples = []
        for i, ex in enumerate(examples):
            if i >= sample_size:
                break
            sampled_examples.append(ex)

    for ex in sampled_examples:
        total_count += 1
        try:
            # Check for instruction-output or text pair structure
            if any(k in ex for k in ["instruction", "input", "prompt", "question"]):
                instr = safe_get(ex, "instruction", "inputs", "prompt", "question")
                out = safe_get(ex, "output", "targets", "response", "answer")
                if instr and out:
                    if validate_text(instr, config) and validate_text(out, config):
                        valid_count += 1
            elif any(k in ex for k in ["text_1", "sentence1", "query"]):
                t1 = safe_get(ex, "text_1", "sentence1", "query")
                t2 = safe_get(ex, "text_2", "sentence2", "answer")
                if t1 and t2:
                    if validate_text(t1, config) and validate_text(t2, config):
                        valid_count += 1
        except Exception as e:
            logger.debug(f"Error validating example: {str(e)}")
            continue

    quality_score = valid_count / total_count if total_count > 0 else 0.0
    logger.info(
        f"Dataset quality score: {quality_score:.3f} ({valid_count}/{total_count} valid)"
    )
    return quality_score


def memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / (1024**3)
    return mem_gb


def check_memory_limit(config: MemoryConfig) -> bool:
    """
    Check if memory usage is within configured limits.
    Args:
        config: Memory configuration
    Returns:
        True if within limits, False otherwise
    """
    usage_gb = memory_usage()
    if usage_gb > config.max_memory_gb:
        logger.warning(
            f"Memory limit exceeded: {usage_gb:.2f}GB / {config.max_memory_gb}GB"
        )
        return False
    return True


def force_gc(aggressive: bool = False) -> float:
    """
    Force garbage collection and return memory freed.
    Args:
        aggressive: Enable more aggressive collection
    Returns:
        Memory freed in GB
    """
    before = memory_usage()
    if aggressive:
        gc.collect(2)  # Full collection
    else:
        gc.collect(1)  # Standard collection

    after = memory_usage()
    freed = before - after
    if freed > 0.1:  # Log if more than 100MB freed
        logger.debug(f"Garbage collection freed {freed:.2f}GB memory")
    return freed


# -----------------------------
# Progress Tracking with ETA and Dashboard
# -----------------------------
@dataclass
class ProgressStats:
    """Advanced progress tracking with ETA, memory, and deduplication statistics."""

    started_at: float
    instruction_total: int = 0
    retrieval_total: int = 0
    by_source_instruction: dict = None
    by_source_retrieval: dict = None
    duplicate_instruction: int = 0
    duplicate_retrieval: int = 0
    expected_instruction_total: Optional[int] = None
    expected_retrieval_total: Optional[int] = None
    last_render_ts: float = 0.0

    def __post_init__(self):
        if self.by_source_instruction is None:
            self.by_source_instruction = {}
        if self.by_source_retrieval is None:
            self.by_source_retrieval = {}
        if not self.last_render_ts:
            self.last_render_ts = self.started_at

    # --- counters-----------------------------------------------------------
    def inc_instruction(self, source: str, n: int = 1) -> None:
        """Increment instruction counter for a source."""
        self.instruction_total += n
        self.by_source_instruction[source] = (
            self.by_source_instruction.get(source, 0) + n
        )

    def inc_retrieval(self, source: str, n: int = 1) -> None:
        """Increment retrieval counter for a source."""
        self.retrieval_total += n
        self.by_source_retrieval[source] = self.by_source_retrieval.get(source, 0) + n

    def mark_duplicate_instruction(self, n: int = 1) -> None:
        """Mark duplicate instruction examples."""
        self.duplicate_instruction += n

    def mark_duplicate_retrieval(self, n: int = 1) -> None:
        """Mark duplicate retrieval pairs."""
        self.duplicate_retrieval += n

    # --- configuration helpers---------------------------------------------
    def set_expected_instruction_total(self, total: Optional[int]) -> None:
        """Set expected total for ETA calculation."""
        self.expected_instruction_total = total

    def set_expected_retrieval_total(self, total: Optional[int]) -> None:
        """Set expected total for ETA calculation."""
        self.expected_retrieval_total = total

    # --- derived metrics----------------------------------------------------
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return max(0.0, time.time() - self.started_at)

    def dedup_ratio_instruction(self) -> float:
        """Calculate deduplication ratio for instructions."""
        total = float(self.instruction_total + self.duplicate_instruction)
        if total <= 0.0:
            return 0.0
        return float(self.duplicate_instruction) / total

    def dedup_ratio_retrieval(self) -> float:
        """Calculate deduplication ratio for retrieval pairs."""
        total = float(self.retrieval_total + self.duplicate_retrieval)
        if total <= 0.0:
            return 0.0
        return float(self.duplicate_retrieval) / total

    def current_memory_mb(self) -> Optional[float]:
        """Best-effort RSS in MiB using resource module if available."""
        if not HAS_RESOURCE:
            return None

        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = float(usage.ru_maxrss)
            # On Linux this is kilobytes; on macOS it may already be bytes
            return rss_kb / 1024.0
        except Exception:
            return None

    def _eta_line(self) -> List[str]:
        """Calculate ETA lines for rendering."""
        lines = []
        elapsed = self.elapsed_seconds()
        if elapsed <= 0.0:
            return lines

        # Instruction ETA
        if self.expected_instruction_total is not None and self.instruction_total > 0:
            instr_left = self.expected_instruction_total - self.instruction_total
            if instr_left > 0:
                instr_rate = float(self.instruction_total) / elapsed
                if instr_rate > 0.0:
                    eta_instr = float(instr_left) / instr_rate
                    lines.append(f"[DASHBOARD] ETA(instruction): {eta_instr:.1f}s")

        # Retrieval ETA
        if self.expected_retrieval_total is not None and self.retrieval_total > 0:
            retr_left = self.expected_retrieval_total - self.retrieval_total
            if retr_left > 0:
                retr_rate = float(self.retrieval_total) / elapsed
                if retr_rate > 0.0:
                    eta_retr = float(retr_left) / retr_rate
                    lines.append(f"[DASHBOARD] ETA(retrieval): {eta_retr:.1f}s")

        return lines

    # --- rendering----------------------------------------------------------
    def render(self) -> str:
        """Render a static snapshot of all tracked metrics."""
        lines = []
        lines.append(f"[DASHBOARD] Elapsed: {self.elapsed_seconds():.1f}s")
        lines.append(f"[DASHBOARD] Instruction examples: {self.instruction_total}")
        lines.append(f"[DASHBOARD] Retrieval pairs: {self.retrieval_total}")

        if self.duplicate_instruction or self.duplicate_retrieval:
            instr_pct = self.dedup_ratio_instruction() * 100.0
            retr_pct = self.dedup_ratio_retrieval() * 100.0
            lines.append(
                f"[DASHBOARD] Dedup(instr): {self.duplicate_instruction} dup ({instr_pct:.2f}%)"
            )
            lines.append(
                f"[DASHBOARD] Dedup(retr): {self.duplicate_retrieval} dup ({retr_pct:.2f}%)"
            )

        mem_mb = self.current_memory_mb()
        if mem_mb is not None:
            lines.append(f"[DASHBOARD] RSS approx: {mem_mb:.1f} MiB")

        eta_lines = self._eta_line()
        lines.extend(eta_lines)

        if self.by_source_instruction:
            lines.append("[DASHBOARD] By source(instruction):")
            for src, count in sorted(
                self.by_source_instruction.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f" - {src}: {count}")

        if self.by_source_retrieval:
            lines.append("[DASHBOARD] By source(retrieval):")
            for src, count in sorted(
                self.by_source_retrieval.items(), key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(f" - {src}: {count}")

        return "\n".join(lines)

    def render_streaming(self, min_interval: float = 1.0, stream=None) -> None:
        """
        Render to a stream no more than once every min_interval seconds.
        Intended usage inside long-running dataset loops.
        """
        now = time.time()
        if now - self.last_render_ts < float(min_interval):
            return

        self.last_render_ts = now

        if stream is None:
            stream = sys.stdout

        try:
            stream.write(self.render() + "\n")
            if hasattr(stream, "flush"):
                stream.flush()
        except Exception:
            # Dashboard must never break the pipeline; swallow I/O errors
            pass


# -----------------------------
# Advanced PII Detection System
# -----------------------------
class PIIDetector:
    """Advanced PII detection system with context-aware pattern matching."""

    def __init__(self, config: PIIConfig):
        """
        Initialize PII detector with custom patterns and strategies.
        Args:
            config: PII configuration object
        """
        self.config = config
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in config.sensitive_patterns
        ]
        self.false_positive_patterns = [
            re.compile(r"\b(year|date|time)\b", re.IGNORECASE),
            re.compile(
                r"\b(phone|email|address)\b\s+(number|field|column)\b", re.IGNORECASE
            ),
            re.compile(r"\b(example|sample)\b", re.IGNORECASE),
        ]

    def detect_pii(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect PII in text with detailed findings.
        Args:
            text: Input text to scan
        Returns:
            Tuple of (contains_pii, list_of_findings)
        """
        if not text or not isinstance(text, str):
            return False, []

        text_lower = text.lower()
        findings = []

        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text_lower):
                start, end = match.span()
                # Extract context around the match
                context_start = max(0, start - self.config.context_window)
                context_end = min(len(text), end + self.config.context_window)
                context = text[context_start:context_end]

                # Check for false positives
                is_false_positive = any(
                    fp_pattern.search(context.lower())
                    for fp_pattern in self.false_positive_patterns
                )

                if not is_false_positive:
                    findings.append(
                        {
                            "pattern": pattern.pattern,
                            "match": text[start:end],
                            "start": start,
                            "end": end,
                            "context": context,
                            "sensitivity_level": self._assess_sensitivity(
                                pattern.pattern
                            ),
                        }
                    )

        return len(findings) > 0, findings

    def _assess_sensitivity(self, pattern: str) -> str:
        """Assess sensitivity level of detected PII."""
        high_sensitivity = ["ssn", "credit", "password", "passport", "license"]
        medium_sensitivity = ["email", "phone", "address"]
        pattern_lower = pattern.lower()

        if any(term in pattern_lower for term in high_sensitivity):
            return "high"
        elif any(term in pattern_lower for term in medium_sensitivity):
            return "medium"
        return "low"

    def redact_text(self, text: str, findings: List[Dict[str, Any]]) -> str:
        """Redact detected PII according to configured strategy."""
        if not findings or self.config.redaction_strategy == "remove":
            return text

        # Sort findings by start position (descending to avoid index shifting)
        findings.sort(key=lambda x: x["start"], reverse=True)
        redacted = text

        for finding in findings:
            start, end = finding["start"], finding["end"]

            if self.config.redaction_strategy == "mask":
                replacement = "*" * (end - start)
            elif self.config.redaction_strategy == "replace":
                sensitivity = finding["sensitivity_level"]
                if sensitivity == "high":
                    replacement = "[REDACTED_HIGH]"
                elif sensitivity == "medium":
                    replacement = "[REDACTED_MEDIUM]"
                else:
                    replacement = "[REDACTED_LOW]"
            else:
                replacement = ""

            redacted = redacted[:start] + replacement + redacted[end:]

        return redacted


# -----------------------------
# Advanced Deduplication System
# -----------------------------
class DeduplicationEngine:
    """Advanced deduplication engine with state persistence and recovery."""

    def __init__(
        self,
        config: DedupConfig,
        state_file: Optional[Path] = None,
        load_existing_state: bool = False,
    ):
        """
        Initialize deduplication engine with configurable strategies.
        Args:
            config: Deduplication configuration
            state_file: Path to state file for persistence
            load_existing_state: Whether to hydrate dedup state from disk (used when
                resuming from a checkpoint)
        """
        self.config = config
        self.state_file = Path(state_file) if state_file else None
        self.exact_dedup_keys: Set[str] = set()
        self.near_dedup_engine: Optional[MinHashLSH] = None
        self.minhashes: Dict[str, MinHash] = {}
        self.quality_scores: Dict[str, float] = {}

        # Load state if available and requested
        if load_existing_state:
            self.load_state()
        elif self.state_file and self.state_file.exists():
            logger.info(
                "Deduplication state file present but resume_from_checkpoint is false; "
                "starting with a fresh deduplication cache",
            )

        # Initialize near-dedup engine if enabled and datasketch is available
        if config.enable_near_dedup and HAS_DATASKETCH:
            self.near_dedup_engine = MinHashLSH(
                threshold=config.near_dedup_threshold,
                num_perm=config.minhash_permutations,
            )

    def load_state(self) -> bool:
        """Load deduplication state from file if available."""
        if not self.state_file or not self.state_file.exists():
            return False

        try:
            logger.info(f"Loading deduplication state from {self.state_file}")
            with open(self.state_file, "rb") as f:
                state = pickle.load(f)

            self.exact_dedup_keys = state.get("exact_dedup_keys", set())
            self.quality_scores = state.get("quality_scores", {})

            # Rebuild near-dedup engine if needed
            if self.config.enable_near_dedup and HAS_DATASKETCH:
                self.minhashes = state.get("minhashes", {})
                self.near_dedup_engine = MinHashLSH(
                    threshold=self.config.near_dedup_threshold,
                    num_perm=self.config.minhash_permutations,
                )
                # Re-insert all minhashes
                for key, minhash in self.minhashes.items():
                    self.near_dedup_engine.insert(key, minhash)

            logger.info(
                f"Loaded state with {len(self.exact_dedup_keys)} exact keys "
                f"and {len(self.minhashes)} near-dup entries"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load deduplication state: {e}")
            return False

    def save_state(self) -> bool:
        """Save deduplication state to file atomically."""
        if not self.state_file:
            return False

        try:
            # Create temp file first
            temp_file = self.state_file.with_suffix(".tmp")
            state = {
                "exact_dedup_keys": self.exact_dedup_keys,
                "quality_scores": self.quality_scores,
                "minhashes": self.minhashes if self.config.enable_near_dedup else {},
                "timestamp": time.time(),
                "config": asdict(self.config),
            }
            with open(temp_file, "wb") as f:
                pickle.dump(state, f)

            # Atomic rename
            if self.state_file.exists():
                self.state_file.unlink()
            temp_file.rename(self.state_file)

            logger.debug(f"Saved deduplication state to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save deduplication state: {e}")
            return False

    def get_minhash(self, text: str) -> MinHash:
        """Generate MinHash for text with consistent tokenization."""
        if not text:
            text = ""
        m = MinHash(num_perm=self.config.minhash_permutations)
        tokens = re.findall(r"\b\w+\b", text.lower())
        for token in tokens:
            if token.strip():
                m.update(token.encode("utf-8"))
        return m

    def is_duplicate(
        self, key: str, text: str, quality_score: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Check if item is a duplicate and determine deduplication action.
        Args:
            key: Unique key for exact deduplication
            text: Text content for near-deduplication
            quality_score: Quality score of the item (higher = better)
        Returns:
            Tuple of (is_duplicate, dedup_type)
        """
        dedup_type = "none"

        # Exact deduplication
        if self.config.enable_exact_dedup:
            if key in self.exact_dedup_keys:
                return True, "exact"
            self.exact_dedup_keys.add(key)
            dedup_type = "exact_added"

        # Near deduplication
        if self.config.enable_near_dedup and self.near_dedup_engine and HAS_DATASKETCH:
            minhash = self.get_minhash(text)
            text_id = str(uuid.uuid4())

            # Query for similar items
            results = self.near_dedup_engine.query(minhash)
            if results:
                # Check if any similar item has better quality
                for result_id in results:
                    existing_quality = self.quality_scores.get(result_id, 0.0)
                    if (
                        existing_quality > quality_score
                        and existing_quality > self.config.quality_threshold
                    ):
                        return True, "near"

                # If we have higher quality, update the index
                if quality_score > self.config.quality_threshold:
                    self.near_dedup_engine.insert(text_id, minhash)
                    self.minhashes[text_id] = minhash
                    self.quality_scores[text_id] = quality_score
                    return False, "near_added_higher_quality"
                return True, "near"

            # No similar items found, add to index
            self.near_dedup_engine.insert(text_id, minhash)
            self.minhashes[text_id] = minhash
            self.quality_scores[text_id] = quality_score
            return False, "near_added_new"

        return False, dedup_type

    def estimate_similarity(self, text1: str, text2: str) -> float:
        """Estimate Jaccard similarity between two texts using MinHash."""
        if (
            not self.config.enable_near_dedup
            or not self.near_dedup_engine
            or not HAS_DATASKETCH
        ):
            return 0.0
        m1 = self.get_minhash(text1)
        m2 = self.get_minhash(text2)
        return m1.jaccard(m2)


# -----------------------------
# Base Dataset Loader Class
# -----------------------------
class DatasetLoaderError(RuntimeError):
    """Raised when a dataset loader cannot complete its operation."""


class DatasetLoader:
    """Base class for dataset loaders with common functionality."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize dataset loader with pipeline configuration.
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metadata = {
            "source": "",
            "type": "",
            "languages": defaultdict(int),
            "examples_loaded": 0,
            "examples_used": 0,
            "duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "filtered_by_quality": 0,
            "filtered_by_pii": 0,
            "loading_errors": 0,
            "processing_time": 0.0,
            "quality_score": 0.0,
            "memory_peak_mb": 0,
            "license_compliance_issues": [],
            "pii_instances_found": 0,
        }

        self.pii_detector = (
            PIIDetector(config.pii_config) if config.pii_config.enabled else None
        )
        self.dedup_engine: Optional[DeduplicationEngine] = None

    def _is_trust_remote_code_allowed(self, dataset_config: DatasetConfig) -> bool:
        """Check whether trust_remote_code can be enabled for a dataset."""

        if not dataset_config.allow_trust_remote_code:
            return False

        # Respect global opt-in or dataset-specific allowlist
        dataset_keys = {
            dataset_config.name.lower(),
            dataset_config.hf_path.lower(),
            dataset_config.name.lower().split("/")[-1],
            dataset_config.hf_path.lower().split("/")[-1],
        }

        return self.config.allow_trust_remote_code or bool(
            dataset_keys & self.config.trusted_remote_code_datasets
        )

    def set_dedup_engine(self, engine: DeduplicationEngine):
        """Set shared deduplication engine."""
        self.dedup_engine = engine

    def check_memory_usage(self, force_gc_threshold: float = 0.8) -> float:
        """
        Check memory usage and perform cleanup if needed.
        Args:
            force_gc_threshold: Memory usage ratio that triggers GC
        Returns:
            Current memory usage in GB
        """
        mem_gb = memory_usage()
        # Update peak memory tracking
        current_mb = mem_gb * 1024
        self.metadata["memory_peak_mb"] = max(
            self.metadata["memory_peak_mb"], current_mb
        )

        # Check if we need to trigger garbage collection
        if mem_gb > self.config.memory_config.max_memory_gb * force_gc_threshold:
            logger.warning(
                f"High memory usage: {mem_gb:.2f}GB. Forcing garbage collection..."
            )
            force_gc(self.config.memory_config.aggressive_gc)

        # Check if we exceed absolute limit
        if mem_gb > self.config.memory_config.max_memory_gb:
            raise MemoryLimitExceeded(
                f"Memory limit exceeded: {mem_gb:.2f}GB / {self.config.memory_config.max_memory_gb}GB"
            )

        return mem_gb

    def validate_license_compliance(self, dataset_config: DatasetConfig) -> List[str]:
        """
        Validate license compliance for dataset configuration.
        Args:
            dataset_config: Dataset configuration to validate
        Returns:
            List of compliance issues found
        """
        license_value = (
            dataset_config.license.value
            if isinstance(dataset_config.license, LicenseType)
            else str(dataset_config.license)
        )
        high_risk_licenses = {
            LicenseType.RESEARCH_ONLY.value,
            LicenseType.CUSTOM.value,
            "custom",
        }
        allow_high_risk = (
            os.environ.get("ALLOW_HIGH_RISK_LICENSES", "false").lower() == "true"
        )

        if license_value in high_risk_licenses and not allow_high_risk:
            return [f"High-risk license {license_value} requires explicit approval"]

        if not dataset_config.license_compliance:
            return ["Missing license compliance configuration"]

        return dataset_config.license_compliance.validate(self.config)

    def process_example(
        self,
        example: Dict[str, Any],
        dataset_config: DatasetConfig,
        selected_langs: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single example with filtering and transformation.
        Args:
            example: Raw example dictionary
            dataset_config: Dataset configuration
            selected_langs: Set of selected languages
        Returns:
            Processed example dictionary or None if filtered out
        """
        try:
            # Apply custom filter if provided
            if dataset_config.filter_function and not dataset_config.filter_function(
                example
            ):
                return None

            # Apply transformation
            if dataset_config.transform_function:
                result = dataset_config.transform_function(example)
                if result is None:
                    return None
            else:
                # Default transformation based on dataset type
                result = self._default_transform(example, dataset_config)
                if result is None:
                    return None

            # Language filtering - only process French examples
            lang = result.get("language", "en")[:2].lower()
            if lang != "fr":
                return None

            # Quality filtering
            if self.config.quality_config.enable_spam_detection:
                if self._is_spam(result):
                    self.metadata["filtered_by_quality"] += 1
                    return None

            return result
        except Exception as e:
            self.logger.debug(f"Error processing example: {str(e)}")
            self.metadata["loading_errors"] += 1
            return None

    def _default_transform(
        self, example: Dict[str, Any], dataset_config: DatasetConfig
    ) -> Optional[Dict[str, Any]]:
        """Default transformation based on dataset type."""
        if dataset_config.dataset_type in [
            DatasetType.INSTRUCTION,
            DatasetType.REASONING,
            DatasetType.DIALOG,
        ]:
            instr = safe_get(
                example, "instruction", "inputs", "prompt", "question", "text"
            )
            out = safe_get(
                example,
                "output",
                "targets",
                "response",
                "answer",
                "completion",
                "solution",
            )
            if not instr or not out:
                return None

            # Explicitly set language to French for French-only pipeline
            return {
                "instruction": instr,
                "output": out,
                "source": dataset_config.name,
                "language": "fr",  # Hardcode French for French-only pipeline
                "metadata": {
                    "original_id": example.get("id", example.get("_id", "")),
                    "dataset": dataset_config.name,
                },
            }
        elif dataset_config.dataset_type in [
            DatasetType.RETRIEVAL,
            DatasetType.EMBEDDING,
        ]:
            t1 = safe_get(example, "text_1", "sentence1", "query", "source")
            t2 = safe_get(example, "text_2", "sentence2", "answer", "target")
            if not t1 or not t2:
                return None

            # Explicitly set language to French for French-only pipeline
            return {
                "text_1": t1,
                "text_2": t2,
                "source": dataset_config.name,
                "language": "fr",  # Hardcode French for French-only pipeline
                "metadata": {
                    "original_id": example.get("id", example.get("_id", "")),
                    "dataset": dataset_config.name,
                },
            }

        return None

    def _is_spam(self, example: Dict[str, Any]) -> bool:
        """Basic spam detection heuristics."""
        if "instruction" in example and "output" in example:
            text = f"{example['instruction']} {example['output']}"
        elif "text_1" in example and "text_2" in example:
            text = f"{example['text_1']} {example['text_2']}"
        else:
            return False

        # Check for excessive repetition
        words = text.lower().split()
        if len(words) > 10:
            word_counts = Counter(words)
            if max(word_counts.values()) / len(words) > 0.5:  # More than 50% repetition
                return True

        # Check for suspicious patterns
        spam_patterns = [
            r"(buy|cheap|free|offer|discount|sale)\b",
            r"click here",
            r"http[s]?://",
            r"@\w+\.\w+",
            r"!!!+$",
            r"\b(?:win|winner|prize|congrat)\b",
        ]
        text_lower = text.lower()
        spam_count = sum(
            1 for pattern in spam_patterns if re.search(pattern, text_lower)
        )
        return spam_count >= 2

    def add_to_sink(
        self,
        item: Dict[str, Any],
        sink: List[Dict[str, Any]],
        seen_keys: Set[str],
        key_func: Callable[[Dict], str],
        dataset_type: str,
        quality_score: float = 1.0,
    ) -> bool:
        """
        Add item to sink with comprehensive filtering and deduplication.
        Args:
            item: Item to add
            sink: Target list
            seen_keys: Set of seen keys for deduplication
            key_func: Function to generate deduplication key
            dataset_type: Type of dataset
            quality_score: Quality score of the item
        Returns:
            True if item was added, False otherwise
        """
        try:
            # PII filtering
            if self.config.pii_config.enabled and self.pii_detector:
                texts_to_check = []
                if dataset_type in ["instruction", "reasoning", "dialog"]:
                    texts_to_check = [item["instruction"], item["output"]]
                elif dataset_type in ["retrieval", "embedding"]:
                    texts_to_check = [item["text_1"], item["text_2"]]

                for text in texts_to_check:
                    has_pii, findings = self.pii_detector.detect_pii(text)
                    if has_pii:
                        self.metadata["filtered_by_pii"] += 1
                        self.metadata["pii_instances_found"] += len(findings)
                        return False

            # Quality filtering
            if self.config.quality_config:
                texts_to_validate = []
                if dataset_type in ["instruction", "reasoning", "dialog"]:
                    texts_to_validate = [item["instruction"], item["output"]]
                elif dataset_type in ["retrieval", "embedding"]:
                    texts_to_validate = [item["text_1"], item["text_2"]]

                if not all(
                    validate_text(text, self.config, quality_score)
                    for text in texts_to_validate
                ):
                    self.metadata["filtered_by_quality"] += 1
                    return False

            # Deduplication
            key = key_func(item)
            if self.dedup_engine:
                is_dup, dup_type = self.dedup_engine.is_duplicate(
                    key, " ".join(texts_to_validate), quality_score
                )
                if is_dup:
                    if dup_type == "exact" or dup_type == "near":
                        self.metadata[
                            (
                                "duplicates_removed"
                                if dup_type == "exact"
                                else "near_duplicates_removed"
                            )
                        ] += 1
                        return False
            elif key in seen_keys:
                self.metadata["duplicates_removed"] += 1
                return False

            seen_keys.add(key)
            sink.append(item)
            self.metadata["examples_used"] += 1

            # Track language distribution
            lang = item.get("language", "unknown")
            self.metadata["languages"][lang] += 1
            return True
        except Exception as e:
            self.logger.error(f"Error processing item: {str(e)}")
            self.metadata["loading_errors"] += 1
            return False

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Main loading method to be implemented by subclasses."""
        raise DatasetLoaderError(
            f"{type(self).__name__} does not implement a load() strategy"
        )

    def _estimate_quality(
        self, dataset: Union[Dataset, Iterable], dataset_config: DatasetConfig
    ) -> float:
        """Estimate dataset quality with sampling."""
        try:
            # Skip quality estimation for streaming datasets to avoid double iteration
            if dataset_config.streaming:
                return 1.0

            # Estimate quality using sampling
            return estimate_dataset_quality(dataset, self.config)
        except Exception as e:
            self.logger.warning(f"Failed to estimate dataset quality: {str(e)}")
            return 0.5  # Default medium quality


# -----------------------------
# Concrete Loader Implementations
# -----------------------------
class InstructionDatasetLoader(DatasetLoader):
    """Loader for instruction-style datasets with advanced features."""

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """
        Load instruction dataset with comprehensive error handling and optimization.
        Args:
            selected_langs: Set of languages to include
            seen_keys: Set of seen keys for deduplication
            sink: Target list to accumulate results
            dataset_config: Dataset configuration
        """
        start_time = time.time()
        self.metadata.update(
            {
                "source": dataset_config.hf_path,
                "type": dataset_config.dataset_type.value,
                "license": dataset_config.license.value,
            }
        )

        # Respect trust_remote_code guardrails
        needs_trust = dataset_config.requires_trust_remote_code
        allow_trust = self._is_trust_remote_code_allowed(dataset_config)
        if needs_trust and not allow_trust:
            self.logger.warning(
                (
                    "Dataset %s requires trust_remote_code but it is disabled; "
                    "re-run with --allow_trust_remote_code or "
                    "--allow_trust_remote_code_for %s"
                ),
                dataset_config.hf_path,
                dataset_config.name,
            )
            self.metadata["skipped_due_to_script_requirement"] = True
            self.metadata["loading_errors"] += 1
            self.metadata["error"] = (
                "trust_remote_code disabled for dataset requiring remote code execution"
            )
            return

        try:
            # Load dataset with retry logic
            ds = load_dataset_with_retry(
                dataset_config.hf_path,
                dataset_config.hf_config,
                split="train",
                streaming=dataset_config.streaming,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=allow_trust,
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if self.config.force_download
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )

            # Estimate dataset quality
            if not dataset_config.streaming:
                self.metadata["quality_score"] = self._estimate_quality(
                    ds, dataset_config
                )

            # Handle DatasetDict (multiple splits)
            if isinstance(ds, DatasetDict):
                iterators = []
                for split_name, split_ds in ds.items():
                    if dataset_config.streaming:
                        iterators.append(split_ds)
                    else:
                        iterators.append(
                            tqdm(split_ds, desc=f"{dataset_config.name} [{split_name}]")
                        )
                iterator = itertools.chain(*iterators)
            else:
                iterator = ds
                if not dataset_config.streaming and self.config.enable_progress_bar:
                    iterator = tqdm(list(ds), desc=dataset_config.name)

            # Process examples in batches
            batch = []
            processed_count = 0

            for ex in iterator:
                processed_count += 1
                self.metadata["examples_loaded"] += 1

                # Process example
                item = self.process_example(ex, dataset_config, selected_langs)
                if item:
                    batch.append(
                        (item, make_instruct_key(item["instruction"], item["output"]))
                    )

                # Process batch when full or at end
                if (
                    len(batch) >= dataset_config.batch_size
                    or processed_count % dataset_config.batch_size == 0
                ):
                    self._process_batch_with_memory_control(
                        batch, seen_keys, sink, dataset_config
                    )
                    batch = []

                # Memory and checkpoint management
                if (
                    processed_count % self.config.memory_config.cache_clear_interval
                    == 0
                ):
                    self.check_memory_usage()

                if (
                    self.config.enable_checkpointing
                    and processed_count % self.config.checkpoint_interval == 0
                ):
                    self._save_checkpoint(dataset_config.name, processed_count)

                # Early stopping if we've reached the limit
                if (
                    dataset_config.max_examples
                    and self.metadata["examples_used"] >= dataset_config.max_examples
                ):
                    break

            # Process remaining batch
            if batch:
                self._process_batch_with_memory_control(
                    batch, seen_keys, sink, dataset_config
                )

            logger.info(
                f"Loaded {self.metadata['examples_used']}/{processed_count} examples from {dataset_config.name} "
                f"({self.metadata['duplicates_removed']} duplicates, "
                f"{self.metadata['filtered_by_quality']} quality filtered, "
                f"{self.metadata['filtered_by_pii']} PII filtered)"
            )
        except Exception as e:
            self.logger.error(
                f"Error loading {dataset_config.name}: {str(e)}", exc_info=True
            )
            self.metadata["loading_errors"] += 1
            self.metadata["error"] = str(e)
        finally:
            self.metadata["processing_time"] = time.time() - start_time
            self.logger.info(
                f"{dataset_config.name} processing completed in {self.metadata['processing_time']:.2f}s"
            )
            # Save final checkpoint
            if self.config.enable_checkpointing:
                self._save_checkpoint(
                    dataset_config.name, self.metadata["examples_loaded"]
                )

    def _process_batch(
        self,
        batch: List[Tuple[Dict[str, Any], str]],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """
        Process a batch of examples in parallel.
        Args:
            batch: List of (item, key) tuples
            seen_keys: Set of seen keys
            sink: Target list
            dataset_config: Dataset configuration
        """
        with ThreadPoolExecutor(
            max_workers=min(self.config.num_workers, 4)
        ) as executor:
            futures = []
            for item, key in batch:
                # Submit processing task
                future = executor.submit(
                    self.add_to_sink,
                    item,
                    sink,
                    seen_keys,
                    lambda x: key,
                    dataset_config.dataset_type.value,
                    self.metadata["quality_score"],
                )
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error processing batch item: {str(e)}")
                    self.metadata["loading_errors"] += 1

    def _process_batch_with_memory_control(
        self,
        batch: List[Tuple[Dict[str, Any], str]],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Process batch with active memory monitoring and adaptive sizing."""
        current_batch: List[Tuple[Dict[str, Any], str]] = []
        aggressive_gc = (
            self.config.memory_config.memory_release_strategy == "aggressive"
            or self.config.memory_config.aggressive_gc
        )

        for item in batch:
            current_batch.append(item)
            if len(current_batch) % 10 == 0:
                usage_gb = memory_usage()
                max_allowed = self.config.memory_config.max_memory_gb * 0.85
                if usage_gb > max_allowed:
                    self.logger.warning(
                        "Memory usage %.2fGB exceeded soft limit %.2fGB; flushing partial batch",
                        usage_gb,
                        max_allowed,
                    )
                    self._process_batch(current_batch, seen_keys, sink, dataset_config)
                    current_batch = []
                    force_gc(aggressive=aggressive_gc)
                    if self.config.auto_scaling and dataset_config.batch_size > 1:
                        new_batch_size = max(1, dataset_config.batch_size // 2)
                        if new_batch_size != dataset_config.batch_size:
                            self.logger.info(
                                "Auto-scaling %s batch size from %d to %d due to memory pressure",
                                dataset_config.name,
                                dataset_config.batch_size,
                                new_batch_size,
                            )
                            dataset_config.batch_size = new_batch_size

        if current_batch:
            self._process_batch(current_batch, seen_keys, sink, dataset_config)

    def _save_checkpoint(self, dataset_name: str, examples_processed: int) -> None:
        """Save pipeline checkpoint."""
        if not self.config.enable_checkpointing:
            return

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "dataset_name": dataset_name,
            "examples_processed": examples_processed,
            "examples_used": self.metadata["examples_used"],
            "timestamp": time.time(),
            "config": self.config.to_dict(),
        }

        checkpoint_file = checkpoint_dir / f"{dataset_name}_checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            self.logger.debug(
                f"Checkpoint saved for {dataset_name} at {examples_processed} examples"
            )
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {str(e)}")


class XP3MultiLanguageLoader(InstructionDatasetLoader):
    """
    Loader for BigScience xP3 – we explicitly extract French examples
    and convert them into instruction/output pairs.

    We only use the 'fr' config (already set in DatasetConfig.hf_config),
    so everything that passes quality checks is treated as French.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        start_time = time.time()
        self.logger.info(
            f"Loading xP3 from {dataset_config.hf_path} (config={dataset_config.hf_config})"
        )

        # We only care about French
        if "fr" not in {l.lower() for l in selected_langs}:
            self.logger.info("Skipping xP3 because 'fr' is not in selected languages")
            return

        # Handle trust_remote_code allowlist
        allow_trust = False
        if dataset_config.requires_trust_remote_code:
            if not self.config.allow_trust_remote_code:
                msg = (
                    "xP3 requires trust_remote_code but global "
                    "--allow_trust_remote_code is disabled; skipping dataset."
                )
                self.logger.error(msg)
                self.metadata["error"] = msg
                self.metadata["loading_errors"] += 1
                return

            normalized_ids = {
                dataset_config.hf_path.lower(),
                dataset_config.name.lower(),
            }
            trusted = {d.lower() for d in self.config.trusted_remote_code_datasets}
            allow_trust = bool(normalized_ids & trusted)
            if not allow_trust:
                msg = (
                    "xP3 requires trust_remote_code but dataset is not in "
                    "trusted_remote_code_datasets; skipping dataset."
                )
                self.logger.error(msg)
                self.metadata["error"] = msg
                self.metadata["loading_errors"] += 1
                return

        # Actually load the dataset
        try:
            ds = load_dataset_with_retry(
                hf_path=dataset_config.hf_path,
                hf_config=dataset_config.hf_config,
                split=dataset_config.hf_split or "train",
                streaming=dataset_config.streaming,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=allow_trust,
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if self.config.force_download
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )
        except DatasetLoadingError as e:
            msg = f"Failed to load xP3: {e}"
            self.logger.error(msg)
            self.metadata["error"] = msg
            self.metadata["loading_errors"] += 1
            return

        # Metadata/init
        self.metadata["total_records"] = 0
        self.metadata["examples_used"] = 0
        self.metadata["duplicates_removed"] = 0
        self.metadata["quality_score"] = dataset_config.quality_weight

        max_examples = dataset_config.max_examples or float("inf")

        # Iterate over examples (streaming or not)
        if dataset_config.streaming:
            iterator = iter(ds)
        else:
            iterator = ds

        for example in iterator:
            self.metadata["total_records"] += 1

            # xP3 uses "inputs" and "targets"
            instr = (example.get("inputs") or "").strip()
            out = (example.get("targets") or "").strip()

            if not instr or not out:
                continue

            # Quick quality check on combined text
            combined_text = instr + "\n" + out
            if not validate_text(
                combined_text,
                self.config,
                quality_score=dataset_config.quality_weight,
            ):
                continue

            item = {
                "instruction": instr,
                "output": out,
                "language": "fr",  # this config is French
                "task": "instruction",
                "context": None,
                "source": "bigscience/xP3",
            }

            key = make_instruct_key(instr, out)

            if self.add_to_sink(
                item=item,
                sink=sink,
                seen_keys=seen_keys,
                key_func=lambda _x, k=key: k,
                dataset_type=dataset_config.dataset_type.value,
                quality_score=dataset_config.quality_weight,
            ):
                self.metadata["examples_used"] += 1

            if self.metadata["examples_used"] >= max_examples:
                break

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed xp3 in {elapsed:.2f}s with {self.metadata['examples_used']} examples"
        )


class AyaCollectionLoader(InstructionDatasetLoader):
    """
    Loader for Cohere Aya Collection (aya_dataset config).

    The config is multilingual; we explicitly filter for French-language
    examples using the `language` field, then convert to instruction/output
    pairs.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    @staticmethod
    def _is_french_language_tag(raw: str) -> bool:
        """Heuristic check whether Aya language tag is French."""
        if not raw:
            return False
        tag = raw.strip().lower()
        if tag.startswith("fr"):
            return True
        if "french" in tag:
            return True
        return False

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        start_time = time.time()
        self.logger.info(
            f"Loading Aya collection from {dataset_config.hf_path} "
            f"(config={dataset_config.hf_config})"
        )

        if "fr" not in {l.lower() for l in selected_langs}:
            self.logger.info("Skipping Aya because 'fr' is not in selected languages")
            return

        # trust_remote_code allowlist handling
        allow_trust = False
        if dataset_config.requires_trust_remote_code:
            if not self.config.allow_trust_remote_code:
                msg = (
                    "Aya collection requires trust_remote_code but global "
                    "--allow_trust_remote_code is disabled; skipping dataset."
                )
                self.logger.error(msg)
                self.metadata["error"] = msg
                self.metadata["loading_errors"] += 1
                return

            normalized_ids = {
                dataset_config.hf_path.lower(),
                dataset_config.name.lower(),
            }
            trusted = {d.lower() for d in self.config.trusted_remote_code_datasets}
            allow_trust = bool(normalized_ids & trusted)
            if not allow_trust:
                msg = (
                    "Aya collection requires trust_remote_code but dataset is not in "
                    "trusted_remote_code_datasets; skipping dataset."
                )
                self.logger.error(msg)
                self.metadata["error"] = msg
                self.metadata["loading_errors"] += 1
                return

        # Load dataset (non-streaming)
        try:
            ds = load_dataset_with_retry(
                hf_path=dataset_config.hf_path,
                hf_config=dataset_config.hf_config,
                split=dataset_config.hf_split or "train",
                streaming=False,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=allow_trust,
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if self.config.force_download
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )
        except DatasetLoadingError as e:
            msg = f"Failed to load Aya collection: {e}"
            self.logger.error(msg)
            self.metadata["error"] = msg
            self.metadata["loading_errors"] += 1
            return

        self.metadata["total_records"] = 0
        self.metadata["examples_used"] = 0
        self.metadata["duplicates_removed"] = 0
        self.metadata["quality_score"] = dataset_config.quality_weight

        max_examples = dataset_config.max_examples or float("inf")

        for example in ds:
            self.metadata["total_records"] += 1

            raw_lang = (example.get("language") or example.get("lang") or "").strip()
            if not self._is_french_language_tag(raw_lang):
                continue

            instr = (example.get("inputs") or "").strip()
            out = (example.get("targets") or "").strip()

            if not instr or not out:
                continue

            combined_text = instr + "\n" + out
            if not validate_text(
                combined_text,
                self.config,
                quality_score=dataset_config.quality_weight,
            ):
                continue

            # Task name as context, if present
            context = example.get("task_name") or None
            source_tag = example.get("source") or "aya_collection"

            item = {
                "instruction": instr,
                "output": out,
                "language": "fr",
                "task": "instruction",
                "context": context,
                "source": f"CohereForAI/aya_collection::{source_tag}",
            }

            key = make_instruct_key(instr, out)

            if self.add_to_sink(
                item=item,
                sink=sink,
                seen_keys=seen_keys,
                key_func=lambda _x, k=key: k,
                dataset_type=dataset_config.dataset_type.value,
                quality_score=dataset_config.quality_weight,
            ):
                self.metadata["examples_used"] += 1

            if self.metadata["examples_used"] >= max_examples:
                break

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed aya in {elapsed:.2f}s with {self.metadata['examples_used']} examples"
        )


class OpenAssistantLoader(InstructionDatasetLoader):
    """Loader for OpenAssistant datasets (OASST1, OASST2)."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Load OpenAssistant dataset with conversation threading."""
        # Load dataset
        ds = load_dataset_with_retry(
            dataset_config.hf_path,
            dataset_config.hf_config,
            split="train",
            streaming=False,
            cache_dir=self.config.cache_dir,
            timeout=dataset_config.timeout,
            requires_auth=dataset_config.requires_auth,
            download_mode=(
                DownloadMode.FORCE_REDOWNLOAD
                if self.config.force_download
                else DownloadMode.REUSE_DATASET_IF_EXISTS
            ),
        )

        # Build conversation trees with explicit child links so that French
        # replies nested under non-French roots are still reachable.
        messages_by_id: Dict[str, Dict[str, Any]] = {}
        root_messages: List[Dict[str, Any]] = []

        for msg in tqdm(ds, desc="Building French conversation trees"):
            msg_id = msg.get("message_id")
            if not msg_id:
                continue

            # Clone to avoid mutating the dataset object and ensure we always
            # have a replies list for tree construction.
            msg_copy = dict(msg)
            msg_copy.setdefault("replies", [])
            messages_by_id[msg_id] = msg_copy

        # Populate child links and identify roots
        for msg in messages_by_id.values():
            parent_id = msg.get("parent_id")
            if parent_id and parent_id in messages_by_id:
                messages_by_id[parent_id].setdefault("replies", []).append(
                    msg["message_id"]
                )
            else:
                root_messages.append(msg)

        # Process French conversations
        processed_count = 0
        self.metadata["examples_loaded"] = len(root_messages)

        for root in tqdm(root_messages, desc="Processing French conversations"):
            conversation = self._extract_conversation(
                root, messages_by_id, selected_langs
            )
            if conversation:
                for turn in conversation:
                    item = {
                        "instruction": turn["instruction"],
                        "output": turn["output"],
                        "source": dataset_config.name,
                        "language": "fr",  # Explicitly set to French
                        "metadata": {
                            "conversation_id": root.get("conversation_id", ""),
                            "message_id": turn.get("message_id", ""),
                            "depth": turn.get("depth", 0),
                        },
                    }
                    key = make_instruct_key(item["instruction"], item["output"])
                    if self.add_to_sink(
                        item,
                        sink,
                        seen_keys,
                        lambda x: key,
                        dataset_config.dataset_type.value,
                    ):
                        processed_count += 1

            # Check limits and memory
            if (
                dataset_config.max_examples
                and processed_count >= dataset_config.max_examples
            ):
                break

            if processed_count % self.config.memory_config.cache_clear_interval == 0:
                self.check_memory_usage()

        self.metadata["examples_used"] = processed_count
        self.logger.info(
            f"Processed {processed_count} French conversation turns from OpenAssistant"
        )

    def _extract_conversation(
        self,
        message: Dict[str, Any],
        messages_by_id: Dict[str, Dict[str, Any]],
        selected_langs: Set[str],
    ) -> List[Dict[str, Any]]:
        """Extract conversation turns from a message tree."""
        turns = []
        stack = [(message, [], 0)]  # (message, context_history, depth)

        while stack:
            current, context, depth = stack.pop()
            role = current.get("role", "")
            text = current.get("text", "").strip()
            lang = current.get("lang", "")
            lang_matches = _lang_matches(lang, selected_langs)

            if role == "assistant" and context and lang_matches and text:
                # Create instruction-output pair from context and response
                instruction = "\n".join(
                    [
                        f"{ctx['role'].title()}: {ctx['text']}"
                        for ctx in context
                        if ctx["text"]
                    ]
                )
                turns.append(
                    {
                        "instruction": instruction,
                        "output": text,
                        "language": lang.lower() if lang else "unknown",
                        "message_id": current.get("message_id", ""),
                        "depth": depth,
                    }
                )

            # Add children to stack, preserving context regardless of language so
            # French replies under non-French parents are still explored.
            replies = current.get("replies", [])
            for reply_id in reversed(replies):  # Process in order
                if reply_id in messages_by_id:
                    reply = messages_by_id[reply_id]
                    new_context = context
                    if text:
                        new_context = context + [{"role": role, "text": text}]
                    stack.append((reply, new_context, depth + 1))

        return turns


class RetrievalDatasetLoader(DatasetLoader):
    """Base loader for retrieval and embedding datasets."""

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Load retrieval dataset with pair validation."""
        start_time = time.time()
        self.metadata.update(
            {
                "source": dataset_config.hf_path,
                "type": dataset_config.dataset_type.value,
                "license": dataset_config.license.value,
            }
        )

        try:
            allow_trust = self._is_trust_remote_code_allowed(dataset_config)

            needs_trust = dataset_config.requires_trust_remote_code
            if needs_trust and not allow_trust:
                self.logger.warning(
                    (
                        "Skipping %s because trust_remote_code is required but disabled; "
                        "use --allow_trust_remote_code or --allow_trust_remote_code_for %s"
                    ),
                    dataset_config.hf_path,
                    dataset_config.name,
                )
                self.metadata["skipped_due_to_script_requirement"] = True
                self.metadata["loading_errors"] += 1
                self.metadata["error"] = (
                    "trust_remote_code disabled for dataset requiring remote code execution"
                )
                return
            # Load dataset
            ds = load_dataset_with_retry(
                dataset_config.hf_path,
                dataset_config.hf_config,
                split="train",
                streaming=dataset_config.streaming,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=allow_trust,
                download_mode=(
                    DownloadMode.FORCE_REDOWNLOAD
                    if self.config.force_download
                    else DownloadMode.REUSE_DATASET_IF_EXISTS
                ),
            )

            # Estimate quality to avoid defaulting to zero (which triggers blanket filtering)
            if not dataset_config.streaming:
                estimated_quality = self._estimate_quality(ds, dataset_config)
                if estimated_quality <= 0:
                    fallback = max(dataset_config.quality_weight, 1.0)
                    self.logger.warning(
                        "Quality estimation returned %.3f for %s; defaulting to %.2f to keep candidates",
                        estimated_quality,
                        dataset_config.name,
                        fallback,
                    )
                    estimated_quality = fallback
                self.metadata["quality_score"] = estimated_quality
            else:
                # Streaming datasets skip estimation; assume neutral quality
                self.metadata["quality_score"] = max(dataset_config.quality_weight, 1.0)

            # Create iterator with progress bar if needed
            iterator = ds
            if not dataset_config.streaming and self.config.enable_progress_bar:
                iterator = tqdm(list(ds), desc=dataset_config.name)

            processed_count = 0
            for ex in iterator:
                processed_count += 1
                self.metadata["examples_loaded"] += 1

                # Process example
                item = self.process_example(ex, dataset_config, selected_langs)
                if not item:
                    continue

                # Add to sink
                key = make_retrieval_key(item["text_1"], item["text_2"])
                if self.add_to_sink(
                    item,
                    sink,
                    seen_keys,
                    lambda x: key,
                    dataset_config.dataset_type.value,
                    self.metadata["quality_score"],
                ):
                    pass  # Already counted in add_to_sink

                # Memory and checkpoint management
                if (
                    processed_count % self.config.memory_config.cache_clear_interval
                    == 0
                ):
                    self.check_memory_usage()

                if (
                    self.config.enable_checkpointing
                    and processed_count % self.config.checkpoint_interval == 0
                ):
                    self._save_checkpoint(dataset_config.name, processed_count)

                # Early stopping
                if (
                    dataset_config.max_examples
                    and self.metadata["examples_used"] >= dataset_config.max_examples
                ):
                    break

            logger.info(
                f"Loaded {self.metadata['examples_used']}/{processed_count} pairs from {dataset_config.name} "
                f"({self.metadata['duplicates_removed']} duplicates, "
                f"{self.metadata['filtered_by_quality']} quality filtered)"
            )
        except Exception as e:
            self.logger.error(
                f"Error loading {dataset_config.name}: {str(e)}", exc_info=True
            )
            self.metadata["loading_errors"] += 1
            self.metadata["error"] = str(e)
        finally:
            self.metadata["processing_time"] = time.time() - start_time

    def _save_checkpoint(self, dataset_name: str, examples_processed: int) -> None:
        """Save checkpoint for retrieval datasets."""
        if not self.config.enable_checkpointing:
            return

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "dataset_name": dataset_name,
            "examples_processed": examples_processed,
            "pairs_used": self.metadata["examples_used"],
            "timestamp": time.time(),
        }

        checkpoint_file = checkpoint_dir / f"{dataset_name}_retrieval_checkpoint.json"
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            self.logger.debug(f"Retrieval checkpoint saved for {dataset_name}")
        except Exception as e:
            self.logger.warning(f"Failed to save retrieval checkpoint: {str(e)}")


class MIRACLLoader(RetrievalDatasetLoader):
    """Loader for MIRACL retrieval dataset with multi-language support."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Load MIRACL dataset for multiple languages."""
        # MIRACL language codes - focus only on French
        miracl_langs = {
            "fr": "fr",  # French
        }
        relevant_langs = [lang for lang in selected_langs if lang in miracl_langs]

        if not relevant_langs:
            self.logger.info(
                f"Skipping MIRACL - no relevant languages in {selected_langs}"
            )
            return

        total_pairs = 0
        start_time = time.time()

        for lang in relevant_langs:
            self.logger.info(f"Loading MIRACL for language: {lang}")

            # Language-specific config
            lang_config = DatasetConfig(
                name=f"MIRACL-{lang}",
                hf_path="miracl/miracl",
                hf_config=miracl_langs[lang],
                dataset_type=DatasetType.RETRIEVAL,
                languages=[lang],
                license=LicenseType.APACHE_2_0,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="MIRACL dataset",
                ),
                max_examples=(
                    dataset_config.max_examples // len(relevant_langs)
                    if dataset_config.max_examples
                    else None
                ),
            )

            # Custom transform for MIRACL - focus on French
            def transform(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                query = safe_get(ex, "query")
                positives = ex.get("positive_passages", [])
                if not query or not positives:
                    return None

                # Take first positive passage
                pos = positives[0]
                text = pos.get("text", "")
                title = pos.get("title", "")
                if not text:
                    return None

                # Combine title and text if available
                if title:
                    text = f"{title}\n{text}"

                return {
                    "text_1": query,
                    "text_2": text,
                    "source": f"MIRACL-{lang}",
                    "language": "fr",  # Explicitly set to French
                    "metadata": {
                        "query_id": ex.get("query_id", ""),
                        "passage_id": pos.get("docid", ""),
                        "relevance_score": 1.0,
                    },
                }

            lang_config.transform_function = transform

            # Create temporary loader for this language
            lang_loader = RetrievalDatasetLoader(self.config)
            if self.dedup_engine:
                lang_loader.set_dedup_engine(self.dedup_engine)

            # Load language data
            lang_loader.load(selected_langs, seen_keys, sink, lang_config)

            # Merge metadata
            for key, value in lang_loader.metadata.items():
                if key == "languages":
                    for l, count in value.items():
                        self.metadata["languages"][l] += count
                elif isinstance(value, int) or isinstance(value, float):
                    self.metadata[key] = self.metadata.get(key, 0) + value

            total_pairs += lang_loader.metadata["examples_used"]

        self.metadata["processing_time"] = time.time() - start_time
        self.logger.info(
            f"MIRACL total pairs across {len(relevant_langs)} languages: {total_pairs}"
        )


# -----------------------------
# Pipeline State Management
# -----------------------------
@dataclass
class PipelineState:
    """Serializable pipeline state for checkpointing and recovery."""

    config_hash: str
    datasets_processed: List[str]
    instruct_count: int = 0
    retrieval_count: int = 0
    checkpoint_time: float = field(default_factory=time.time)
    memory_usage_gb: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        return cls(**data)


class PipelineStateManager:
    """Manages pipeline state persistence and recovery."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state_file = Path(config.output_dir) / config.state_file
        self.state: Optional[PipelineState] = None

    def _compute_config_hash(self) -> str:
        """Compute a stable hash for the current configuration."""

        return hashlib.md5(
            json.dumps(self.config.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def load_state(self) -> bool:
        """Load pipeline state from file."""
        if not self.config.resume_from_checkpoint or not self.state_file.exists():
            return False

        try:
            with open(self.state_file, "rb") as f:
                state_dict = pickle.load(f)
            self.state = PipelineState.from_dict(state_dict)
            expected_hash = self._compute_config_hash()

            if self.state.config_hash != expected_hash:
                logger.warning(
                    "Checkpoint configuration hash %s does not match current hash %s; "
                    "ignoring saved state to avoid inconsistent resumes",
                    self.state.config_hash,
                    expected_hash,
                )
                self.state = None
                return False
            logger.info(
                f"Loaded pipeline state: {self.state.datasets_processed} datasets processed"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load pipeline state: {e}")
            return False

    def save_state(self, pipeline: "DatasetPipeline") -> bool:
        """Save pipeline state to file."""
        try:
            config_hash = self._compute_config_hash()

            state = PipelineState(
                config_hash=config_hash,
                datasets_processed=list(pipeline.stats["datasets_processed"].keys()),
                instruct_count=len(pipeline.instruct_data),
                retrieval_count=len(pipeline.retrieval_data),
                memory_usage_gb=memory_usage(),
                errors=pipeline.stats.get("errors", []),
            )

            # Create temp file first
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                pickle.dump(state.to_dict(), f)

            # Atomic rename
            if self.state_file.exists():
                self.state_file.unlink()
            temp_file.rename(self.state_file)

            self.state = state
            logger.debug(f"Saved pipeline state to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
            return False

    def should_skip_dataset(self, dataset_name: str) -> bool:
        """Check if dataset should be skipped based on saved state."""
        if not self.state:
            return False
        return dataset_name in self.state.datasets_processed


# -----------------------------
# Monitoring Dashboard
# -----------------------------
class DashboardServer:
    """Flask-based monitoring dashboard with real-time metrics."""

    def __init__(self, pipeline: "DatasetPipeline", port: int = 8080):
        """
        Initialize dashboard server.
        Args:
            pipeline: Pipeline instance to monitor
            port: Port to run dashboard on
        """
        self.pipeline = pipeline
        self.port = port
        self.app = None
        self.server = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> bool:
        """Start dashboard server in background thread."""
        try:
            from flask import Flask, jsonify
            from werkzeug.serving import make_server
        except ImportError:
            logger.warning(
                "Flask is not installed; dashboard server will not be started"
            )
            return False

        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.ERROR)  # Reduce Flask logging noise

        @self.app.route("/")
        def dashboard() -> str:
            """Main dashboard page with a quick stats view."""

            stats = self.pipeline.get_stats()
            dataset_stats = self.pipeline.get_dataset_stats()
            dataset_rows = "".join(
                f"<li><strong>{name}</strong>: {info.get('count', 0)} records</li>"
                for name, info in dataset_stats.items()
            )

            return f"""
            <html>
            <head>
                <title>Dataset Pipeline Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 24px; }}
                    .metric {{ margin: 8px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
                    .header {{ color: #2c3e50; font-size: 24px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 18px; }}
                    ul {{ padding-left: 20px; }}
                </style>
            </head>
            <body>
                <div class="header">Dataset Pipeline Dashboard</div>
                <div class="section">
                    <div class="metric">Elapsed Time: {stats['elapsed_time']:.2f} seconds</div>
                    <div class="metric">Instruction Examples: {stats['instruct_count']}</div>
                    <div class="metric">Retrieval Pairs: {stats['retrieval_count']}</div>
                    <div class="metric">Memory Usage: {stats['memory_usage_mb']:.2f} MB</div>
                    <div class="metric">Peak Memory: {stats['memory_peak_mb']:.2f} MB</div>
                    <div class="metric">Throughput (instr/retr): {stats['throughput_instruct']:.2f} / {stats['throughput_retrieval']:.2f} per sec</div>
                </div>
                <div class="section">
                    <h3>Datasets Processed</h3>
                    <ul>{dataset_rows}</ul>
                </div>
            </body>
            </html>
            """

        @self.app.route("/stats")
        def stats_endpoint():
            """Return raw stats as JSON."""

            return jsonify(self.pipeline.get_stats())

        try:
            self.server = make_server("0.0.0.0", self.port, self.app)
        except (
            OSError
        ) as exc:  # pragma: no cover - depends on runtime port availability
            logger.error(
                "Failed to start dashboard server on port %s: %s", self.port, exc
            )
            return False

        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.running = True
        logger.info("Dashboard server started on port %s", self.port)
        return True

    def stop(self) -> None:
        """Stop dashboard server if running."""

        if self.server:
            self.server.shutdown()
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

        self.server = None
        self.server_thread = None
        self.running = False
        logger.info("Dashboard server stopped")


# -----------------------------
# Main Pipeline Class
# -----------------------------
class DatasetPipeline:
    """Main pipeline class for multi-task dataset preparation."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize data stores
        self.instruct_data: List[Dict[str, Any]] = []
        self.retrieval_data: List[Dict[str, Any]] = []

        # Deduplication sets
        self.seen_instruct: Set[str] = set()
        self.seen_retrieval: Set[str] = set()

        # Statistics tracking
        self.stats: Dict[str, Any] = {
            "start_time": time.time(),
            "datasets_processed": defaultdict(dict),
            "instruct_count": 0,
            "retrieval_count": 0,
            "memory_peak_mb": 0,
            "errors": [],
        }

        # Progress tracking with ETA
        self.progress_stats = ProgressStats(started_at=time.time())

        self.output_artifacts: Dict[str, Dict[str, Any]] = {}
        self._health_thread: Optional[threading.Thread] = None
        self._health_stop_event = threading.Event()
        self._health_violation: Optional[BaseException] = None

        # Initialize deduplication engine
        self.dedup_engine = None
        if config.dedup_config.enable_exact_dedup or (
            config.dedup_config.enable_near_dedup and HAS_DATASKETCH
        ):
            dedup_state_file = Path(config.output_dir) / config.dedup_state_file
            self.dedup_engine = DeduplicationEngine(
                config.dedup_config,
                state_file=dedup_state_file,
                load_existing_state=config.resume_from_checkpoint,
            )

        # Initialize state manager
        self.state_manager = PipelineStateManager(config)
        state_loaded = self.state_manager.load_state()

        # Rehydrate previously saved results when resuming so datasets already
        # marked as processed are also present in memory and outputs stay stable.
        if state_loaded:
            self._rehydrate_previous_results()

        # Initialize dataset loaders
        self.loaders = self._initialize_loaders()

        # Initialize dashboard if requested
        self.dashboard = None
        if config.dashboard_port:
            self.dashboard = DashboardServer(self, config.dashboard_port)
            self.dashboard.start()

        # Initialize output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory created: {config.output_dir}")

    def _rehydrate_previous_results(self) -> None:
        """Reload previously written outputs when resuming from a checkpoint."""

        instruct_path = Path(self.config.output_dir) / "combined_instruct.jsonl"
        retrieval_path = Path(self.config.output_dir) / "combined_retrieval.jsonl"

        if instruct_path.exists():
            try:
                with open(instruct_path, "r", encoding="utf-8") as f:
                    self.instruct_data = [
                        json.loads(line) for line in f if line.strip()
                    ]
                self.seen_instruct = {
                    make_instruct_key(
                        item.get("instruction", ""), item.get("output", "")
                    )
                    for item in self.instruct_data
                }
                self.logger.info(
                    "Rehydrated %d instruction examples from %s",
                    len(self.instruct_data),
                    instruct_path,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to rehydrate instruction outputs from %s: %s",
                    instruct_path,
                    exc,
                )

        if retrieval_path.exists():
            try:
                with open(retrieval_path, "r", encoding="utf-8") as f:
                    self.retrieval_data = [
                        json.loads(line) for line in f if line.strip()
                    ]
                self.seen_retrieval = {
                    make_retrieval_key(
                        item.get("text_1", ""),
                        item.get("text_2", ""),
                        symmetric=item.get("symmetric", True),
                    )
                    for item in self.retrieval_data
                }
                self.logger.info(
                    "Rehydrated %d retrieval pairs from %s",
                    len(self.retrieval_data),
                    retrieval_path,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to rehydrate retrieval outputs from %s: %s",
                    retrieval_path,
                    exc,
                )

        # Keep stats aligned with recovered data
        self.stats["instruct_count"] = len(self.instruct_data)
        self.stats["retrieval_count"] = len(self.retrieval_data)

        if self.state_manager.state:
            for dataset in self.state_manager.state.datasets_processed:
                self.stats["datasets_processed"].setdefault(dataset, {})[
                    "status"
                ] = "resumed"

    def _initialize_loaders(self) -> Dict[str, DatasetLoader]:
        """Initialize all dataset loaders based on configuration."""
        loaders = {}

        # Instruction datasets
        loaders["xp3"] = XP3MultiLanguageLoader(self.config)
        loaders["aya"] = AyaCollectionLoader(self.config)
        loaders["oasst"] = OpenAssistantLoader(self.config)

        # Retrieval datasets
        loaders["miracl"] = MIRACLLoader(self.config)

        # Set dedup engine for all loaders
        if self.dedup_engine:
            for loader in loaders.values():
                loader.set_dedup_engine(self.dedup_engine)

        return loaders

    def get_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics for dashboard."""
        current_time = time.time()
        elapsed = current_time - self.stats["start_time"]

        return {
            "elapsed_time": elapsed,
            "instruct_count": len(self.instruct_data),
            "retrieval_count": len(self.retrieval_data),
            "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "memory_peak_mb": self.stats["memory_peak_mb"],
            "datasets_processed": dict(self.stats["datasets_processed"]),
            "throughput_instruct": (
                len(self.instruct_data) / elapsed if elapsed > 0 else 0
            ),
            "throughput_retrieval": (
                len(self.retrieval_data) / elapsed if elapsed > 0 else 0
            ),
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for all datasets.

        Returns the same structure as `self.stats["datasets_processed"]`
        so that metadata generation and the summary report can rely on it.
        """
        # Ensure it's a plain dict, not a defaultdict, for JSON serialization
        return {
            name: dict(stats)
            for name, stats in self.stats["datasets_processed"].items()
        }

    def _start_health_monitor(self) -> None:
        """Start background health monitoring if enabled."""
        if self.config.health_check_interval <= 0:
            return

        if self._health_thread and self._health_thread.is_alive():
            return

        self._health_stop_event.clear()
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="dataset-health-monitor",
            daemon=True,
        )
        self._health_thread.start()

    def _stop_health_monitor(self) -> None:
        """Stop the health monitoring thread if running."""
        self._health_stop_event.set()
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)
        self._health_thread = None

    def _health_monitor_loop(self) -> None:
        """Continuously check runtime constraints and memory usage."""
        interval = max(1, self.config.health_check_interval)
        while not self._health_stop_event.wait(interval):
            try:
                usage_gb = memory_usage()
                usage_mb = usage_gb * 1024
                self.stats["memory_peak_mb"] = max(
                    self.stats.get("memory_peak_mb", 0), usage_mb
                )

                if usage_gb > self.config.memory_config.max_memory_gb:
                    self._health_violation = MemoryLimitExceeded(
                        f"Memory usage {usage_gb:.2f}GB exceeded limit of {self.config.memory_config.max_memory_gb}GB"
                    )
                    self.logger.error(str(self._health_violation))
                    self._health_stop_event.set()
                    break

                runtime_hours = (time.time() - self.stats["start_time"]) / 3600
                if (
                    self.config.max_runtime_hours > 0
                    and runtime_hours > self.config.max_runtime_hours
                ):
                    self._health_violation = RuntimeError(
                        f"Max runtime {self.config.max_runtime_hours}h exceeded"
                    )
                    self.logger.error(str(self._health_violation))
                    self._health_stop_event.set()
                    break
            except Exception as e:
                self.logger.error("Health monitor encountered an error: %s", e)
                self._health_stop_event.set()
                break

    def _check_health_status(self) -> None:
        """Raise recorded health violations in the main thread."""
        if self._health_violation:
            raise self._health_violation

    def run(self) -> None:
        """Execute the main pipeline."""
        try:
            self.logger.info("Starting dataset pipeline execution")
            self.logger.info(
                f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}"
            )

            if self.config.distributed_mode:
                self.logger.info(
                    "Distributed mode enabled (rank %d/%d)",
                    self.config.node_rank,
                    self.config.world_size,
                )

            self._start_health_monitor()
            self._check_health_status()

            # Set random seed
            random.seed(self.config.seed)

            # Load datasets
            self._load_all_datasets()
            self._check_health_status()

            # Validate final results
            if self.config.enable_validation:
                self._validate_results()
            self._check_health_status()

            # Write outputs
            self._write_outputs()
            self._sync_outputs_to_cloud()

            self.logger.info("Pipeline execution completed successfully")
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats["errors"].append(error_msg)
            raise
        finally:
            # Save final state
            self.state_manager.save_state(self)

            # Save final deduplication state
            if self.dedup_engine:
                self.dedup_engine.save_state()

            # Stop dashboard if running
            if self.dashboard:
                self.dashboard.stop()

            self._stop_health_monitor()

            # Log final statistics
            duration = time.time() - self.stats["start_time"]
            self.logger.info(f"Pipeline completed in {duration:.2f} seconds")
            self.logger.info(f"Total instruction examples: {len(self.instruct_data)}")
            self.logger.info(f"Total retrieval pairs: {len(self.retrieval_data)}")

    def _load_all_datasets(self) -> None:
        """Load all configured datasets with progress tracking."""
        selected_langs = set(self.config.langs)

        # Process instruction datasets
        self.logger.info("Loading instruction datasets...")

        for name, loader in self.loaders.items():
            self._check_health_status()

            if self.state_manager.should_skip_dataset(name):
                self.logger.info(f"Skipping {name} (already processed in checkpoint)")
                continue

            if isinstance(loader, InstructionDatasetLoader):
                self.logger.info(f"Processing instruction dataset: {name}")
                start_time = time.time()

                # Get dataset config
                dataset_config = self._get_dataset_config(name)
                if not dataset_config:
                    continue

                # Load dataset
                loader.load(
                    selected_langs,
                    self.seen_instruct,
                    self.instruct_data,
                    dataset_config,
                )

                # Update progress stats
                self.progress_stats.set_expected_instruction_total(
                    len(self.seen_instruct) + loader.metadata["examples_used"]
                )
                self.progress_stats.inc_instruction(
                    name, loader.metadata["examples_used"]
                )
                self.progress_stats.mark_duplicate_instruction(
                    loader.metadata["duplicates_removed"]
                )

                # Update statistics
                elapsed = time.time() - start_time
                self.stats["datasets_processed"][name] = {
                    "type": "instruction",
                    "count": loader.metadata["examples_used"],
                    "time": elapsed,
                    "metadata": loader.metadata,
                }

                self.logger.info(
                    f"Completed {name} in {elapsed:.2f}s with {loader.metadata['examples_used']} examples"
                )

                # Save checkpoint after each dataset
                self.state_manager.save_state(self)
                self._check_health_status()

                # Render progress dashboard
                self.progress_stats.render_streaming(min_interval=1.0)

            elif isinstance(loader, RetrievalDatasetLoader):
                self.logger.info(f"Processing retrieval dataset: {name}")
                start_time = time.time()

                # Get dataset config
                dataset_config = self._get_dataset_config(name)
                if not dataset_config:
                    continue

                # Load dataset
                loader.load(
                    selected_langs,
                    self.seen_retrieval,
                    self.retrieval_data,
                    dataset_config,
                )

                # Update progress stats
                self.progress_stats.set_expected_retrieval_total(
                    len(self.seen_retrieval) + loader.metadata["examples_used"]
                )
                self.progress_stats.inc_retrieval(
                    name, loader.metadata["examples_used"]
                )
                self.progress_stats.mark_duplicate_retrieval(
                    loader.metadata["duplicates_removed"]
                )

                # Update statistics
                elapsed = time.time() - start_time
                self.stats["datasets_processed"][name] = {
                    "type": "retrieval",
                    "count": loader.metadata["examples_used"],
                    "time": elapsed,
                    "metadata": loader.metadata,
                }

                self.logger.info(
                    f"Completed {name} in {elapsed:.2f}s with {loader.metadata['examples_used']} pairs"
                )

                # Save checkpoint after each dataset
                self.state_manager.save_state(self)
                self._check_health_status()

                # Render progress dashboard
                self.progress_stats.render_streaming(min_interval=1.0)

    def _get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration based on name."""
        configs = {
            "xp3": DatasetConfig(
                name="bigscience/xP3",
                hf_path="bigscience/xP3",
                hf_config="fr",
                dataset_type=DatasetType.INSTRUCTION,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
                requires_trust_remote_code=True,
                allow_trust_remote_code=True,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="BigScience xP3 Dataset",
                ),
                streaming=True,
                max_examples=self.config.max_per_dataset,
                quality_weight=1.0,
            ),
            "aya": DatasetConfig(
                name="CohereForAI/aya_collection",
                hf_path="CohereForAI/aya_collection",
                hf_config="aya_dataset",  # Specify sub-config
                dataset_type=DatasetType.INSTRUCTION,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
                requires_trust_remote_code=True,
                allow_trust_remote_code=True,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="Cohere Aya Collection",
                ),
                streaming=False,
                max_examples=self.config.max_per_dataset,
                quality_weight=1.2,
            ),
            "oasst": DatasetConfig(
                name="OpenAssistant/oasst2",
                hf_path="OpenAssistant/oasst2",
                dataset_type=DatasetType.DIALOG,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="OpenAssistant Dataset",
                ),
                streaming=False,
                max_examples=self.config.max_per_dataset,
                quality_weight=0.9,
            ),
            "miracl": DatasetConfig(
                name="miracl/miracl",
                hf_path="miracl/miracl",
                dataset_type=DatasetType.RETRIEVAL,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="MIRACL Dataset",
                ),
                streaming=False,
                max_examples=self.config.max_per_dataset,
                quality_weight=1.0,
            ),
        }
        return configs.get(dataset_name)

    def _register_output(
        self,
        key: str,
        path: Path,
        records: int,
        *,
        schema: Optional[Dict[str, str]] = None,
    ) -> None:
        self.output_artifacts[key] = {
            "path": str(path),
            "records": int(records),
        }
        if schema:
            self.output_artifacts[key]["schema"] = dict(schema)

    def _sync_outputs_to_cloud(self) -> None:
        """Upload generated artifacts to configured cloud storage."""
        if not self.config.cloud_storage:
            return

        provider = (self.config.cloud_storage.get("provider") or "").lower()
        if provider != "aws":
            self.logger.warning(
                "Unsupported cloud storage provider '%s'; skipping sync", provider
            )
            return

        bucket = self.config.cloud_storage.get("bucket")
        if not bucket:
            self.logger.warning("AWS cloud storage configuration missing bucket name")
            return

        prefix = (self.config.cloud_storage.get("prefix") or "").strip("/")

        if not HAS_BOTO3:
            self.logger.warning(
                "boto3 not installed; cannot upload artifacts to s3://%s", bucket
            )
            return

        s3 = boto3.client("s3")
        for name, artifact in self.output_artifacts.items():
            path = artifact.get("path")
            if not path or not os.path.exists(path):
                continue

            file_name = os.path.basename(path)
            key = f"{prefix}/{file_name}" if prefix else file_name

            try:
                s3.upload_file(path, bucket, key)
                self.logger.info(
                    "Uploaded %s artifact to s3://%s/%s", name, bucket, key
                )
            except Exception as exc:
                error_msg = f"Failed to upload {path} to s3://{bucket}/{key}: {exc}"
                self.logger.error(error_msg)
                self.stats["errors"].append(error_msg)

    def _write_unsloth_ready_dataset(self, instruct_path: Path) -> Optional[Path]:
        """Materialize a prompt/completion dataset for downstream fine-tuning."""
        if not self.instruct_data:
            return None

        converted: List[Dict[str, str]] = []
        dropped = 0

        for record in self.instruct_data:
            normalised = normalise_instruction_record(record)
            if normalised:
                converted.append(normalised)
            else:
                dropped += 1

        if not converted:
            self.logger.warning(
                "Unable to derive prompt/completion records from instruction dataset"
            )
            return None

        export_path = instruct_path.parent / self.config.unsloth_export_name
        self._safe_write_jsonl(converted, export_path)

        if dropped:
            self.logger.info(
                "Skipped %d instruction items without prompt/completion fields",
                dropped,
            )

        self._register_output(
            "unsloth_prompt_completion",
            export_path,
            len(converted),
            schema={"prompt": "str", "completion": "str"},
        )

        self.logger.info("Wrote unsloth-ready dataset to %s", export_path)
        return export_path

    def _safe_write_jsonl(self, data: List[Dict[str, Any]], path: Path) -> None:
        """Write JSONL file with atomic operations."""
        temp_file = path.with_suffix(".tmp")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            if path.exists():
                path.unlink()
            temp_file.rename(path)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to write {path}: {str(e)}")

    def _write_outputs(self) -> None:
        """Write output files with atomic operations."""
        try:
            # Write instruction data
            instruct_path = Path(self.config.output_dir) / "combined_instruct.jsonl"
            self._safe_write_jsonl(self.instruct_data, instruct_path)
            self._register_output(
                "instruction_jsonl",
                instruct_path,
                len(self.instruct_data),
                schema={"instruction": "str", "output": "str"},
            )

            # Write retrieval data
            retrieval_path = Path(self.config.output_dir) / "combined_retrieval.jsonl"
            self._safe_write_jsonl(self.retrieval_data, retrieval_path)
            self._register_output(
                "retrieval_jsonl",
                retrieval_path,
                len(self.retrieval_data),
                schema={"text_1": "str", "text_2": "str"},
            )

            # Derive prompt/completion export for downstream pipelines
            self._write_unsloth_ready_dataset(instruct_path)

            # Write validation splits if enabled
            if self.config.enable_validation:
                self._write_validation_splits()

            self.logger.info(
                f"Written {len(self.instruct_data)} instruction examples to {instruct_path}"
            )
            self.logger.info(
                f"Written {len(self.retrieval_data)} retrieval pairs to {retrieval_path}"
            )

            # Final progress dashboard
            self.progress_stats.render_streaming(min_interval=0.0)

        except Exception as e:
            self.logger.error(f"Error writing output files: {str(e)}")
            self.stats["errors"].append(f"Output writing failed: {str(e)}")
            raise

    def _write_validation_splits(self) -> None:
        """Write validation splits for model evaluation."""
        if not self.instruct_data or not self.retrieval_data:
            return

        # Create validation directory
        val_dir = Path(self.config.output_dir) / "validation"
        val_dir.mkdir(exist_ok=True)

        # Instruction validation split
        instruct_val_size = min(
            int(len(self.instruct_data) * self.config.validation_split),
            self.config.validation_max_size,
        )
        if instruct_val_size > 0:
            instruct_val = random.sample(self.instruct_data, instruct_val_size)
            instruct_train = [ex for ex in self.instruct_data if ex not in instruct_val]

            self._safe_write_jsonl(instruct_train, val_dir / "instruct_train.jsonl")
            self._safe_write_jsonl(instruct_val, val_dir / "instruct_val.jsonl")

            self.logger.info(
                "Created instruction validation split: %d train, %d val",
                len(instruct_train),
                len(instruct_val),
            )

        # Retrieval validation split
        retrieval_val_size = min(
            int(len(self.retrieval_data) * self.config.validation_split),
            self.config.validation_max_size,
        )
        if retrieval_val_size > 0:
            retrieval_val = random.sample(self.retrieval_data, retrieval_val_size)
            retrieval_train = [
                ex for ex in self.retrieval_data if ex not in retrieval_val
            ]

            self._safe_write_jsonl(retrieval_train, val_dir / "retrieval_train.jsonl")
            self._safe_write_jsonl(retrieval_val, val_dir / "retrieval_val.jsonl")

            self.logger.info(
                "Created retrieval validation split: %d train, %d val",
                len(retrieval_train),
                len(retrieval_val),
            )

    def _validate_results(self) -> Dict[str, Any]:
        """Validate final results for quality and consistency."""
        self.logger.info("Validating final results...")

        validation_results: Dict[str, Any] = {
            "instruct_stats": {
                "total": len(self.instruct_data),
                "by_language": defaultdict(int),
                "avg_instruction_length": 0.0,
                "avg_output_length": 0.0,
            },
            "retrieval_stats": {
                "total": len(self.retrieval_data),
                "by_language": defaultdict(int),
                "avg_text1_length": 0.0,
                "avg_text2_length": 0.0,
            },
            "quality_issues": [],
            "compliance_issues": [],
        }

        # Instruction stats
        instr_lengths: List[int] = []
        out_lengths: List[int] = []
        for ex in self.instruct_data:
            lang = ex.get("language", "unknown")
            validation_results["instruct_stats"]["by_language"][lang] += 1
            instr_lengths.append(len(ex.get("instruction", "")))
            out_lengths.append(len(ex.get("output", "")))

        if instr_lengths:
            validation_results["instruct_stats"]["avg_instruction_length"] = sum(
                instr_lengths
            ) / len(instr_lengths)
        if out_lengths:
            validation_results["instruct_stats"]["avg_output_length"] = sum(
                out_lengths
            ) / len(out_lengths)

        # Retrieval stats
        t1_lengths: List[int] = []
        t2_lengths: List[int] = []
        for ex in self.retrieval_data:
            lang = ex.get("language", "unknown")
            validation_results["retrieval_stats"]["by_language"][lang] += 1
            t1_lengths.append(len(ex.get("text_1", "")))
            t2_lengths.append(len(ex.get("text_2", "")))

        if t1_lengths:
            validation_results["retrieval_stats"]["avg_text1_length"] = sum(
                t1_lengths
            ) / len(t1_lengths)
        if t2_lengths:
            validation_results["retrieval_stats"]["avg_text2_length"] = sum(
                t2_lengths
            ) / len(t2_lengths)

        # Global sanity checks
        min_total = self.config.min_examples_per_language * len(self.config.langs)
        if validation_results["instruct_stats"]["total"] < min_total:
            validation_results["quality_issues"].append(
                f"Low instruction example count: {validation_results['instruct_stats']['total']}"
            )
        if validation_results["retrieval_stats"]["total"] < min_total:
            validation_results["quality_issues"].append(
                f"Low retrieval pair count: {validation_results['retrieval_stats']['total']}"
            )

        self.logger.info(
            "Validation results: %d instruction examples, %d retrieval pairs",
            validation_results["instruct_stats"]["total"],
            validation_results["retrieval_stats"]["total"],
        )
        for issue in validation_results["quality_issues"]:
            self.logger.warning("Quality issue: %s", issue)

        return validation_results

    def _generate_metadata(self) -> None:
        """Generate comprehensive metadata with performance metrics."""
        metadata_path = Path(self.config.output_dir) / self.config.metadata_file
        summary_path = Path(self.config.output_dir) / "pipeline_summary.txt"

        try:
            exec_stats = {
                "start_time": self.stats["start_time"],
                "end_time": time.time(),
                "duration": time.time() - self.stats["start_time"],
                "peak_memory_mb": self.stats["memory_peak_mb"],
                "instruct_count": len(self.instruct_data),
                "retrieval_count": len(self.retrieval_data),
                "progress_stats": {
                    "instruction_total": self.progress_stats.instruction_total,
                    "retrieval_total": self.progress_stats.retrieval_total,
                    "duplicate_instruction": self.progress_stats.duplicate_instruction,
                    "duplicate_retrieval": self.progress_stats.duplicate_retrieval,
                    "elapsed_seconds": self.progress_stats.elapsed_seconds(),
                },
            }

            metadata: Dict[str, Any] = {
                "pipeline_config": self.config.to_dict(),
                "execution_stats": exec_stats,
                "dataset_stats": self.get_dataset_stats(),
                "validation_results": self._validate_results(),
                "license_compliance_summary": self._generate_license_summary(),
                "pii_summary": self._generate_pii_summary(),
                "derived_outputs": self.output_artifacts,
                "errors": self.stats.get("errors", []),
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            with open(summary_path, "w", encoding="utf-8") as f:
                self._write_summary_report(f, metadata)

            self._register_output("metadata", metadata_path, 0)
            self._register_output("summary_report", summary_path, 0)

            self.logger.info("Metadata written to %s", metadata_path)
            self.logger.info("Summary report written to %s", summary_path)

            print("\n" + "=" * 80)
            print("FINAL PIPELINE SUMMARY")
            print("=" * 80)
            print(self.progress_stats.render())
            print("=" * 80)

        except Exception as e:
            self.logger.error(f"Error generating meta {str(e)}")
            self.stats.setdefault("errors", []).append(
                f"Metadata generation failed: {str(e)}"
            )

    def _generate_license_summary(self) -> Dict[str, Any]:
        """Generate license compliance summary."""
        summary = {
            "datasets_by_license": defaultdict(list),
            "commercial_use_compatible": [],
            "attribution_required": [],
            "modification_allowed": [],
            "compliance_issues": [],
        }

        for name, loader in self.loaders.items():
            if not hasattr(loader, "metadata"):
                continue

            license_type = loader.metadata.get("license", "unknown")
            summary["datasets_by_license"][license_type].append(name)

            compliance = loader.metadata.get("license_compliance_issues", [])
            if compliance:
                summary["compliance_issues"].extend(compliance)

            # Check specific license properties
            if (
                hasattr(loader, "dataset_config")
                and loader.dataset_config.license_compliance
            ):
                compliance = loader.dataset_config.license_compliance
                if compliance.allows_commercial_use:
                    summary["commercial_use_compatible"].append(name)
                if compliance.requires_attribution:
                    summary["attribution_required"].append(name)
                if compliance.allows_modification:
                    summary["modification_allowed"].append(name)

        return summary

    def _generate_pii_summary(self) -> Dict[str, Any]:
        """Generate PII detection summary."""
        summary = {
            "total_instances_found": 0,
            "datasets_with_pii": {},
            "pii_by_type": defaultdict(int),
        }

        for name, loader in self.loaders.items():
            if not hasattr(loader, "metadata"):
                continue

            pii_count = loader.metadata.get("pii_instances_found", 0)
            if pii_count > 0:
                summary["datasets_with_pii"][name] = pii_count
                summary["total_instances_found"] += pii_count

            # Get PII types if available
            if hasattr(loader, "pii_detector") and loader.pii_detector:
                for pattern in loader.pii_detector.config.sensitive_patterns:
                    # This is a simplified approach - in practice you'd track specific types
                    if "ssn" in pattern.lower():
                        summary["pii_by_type"]["ssn"] += pii_count // 3  # Estimate
                    elif "credit" in pattern.lower() or "card" in pattern.lower():
                        summary["pii_by_type"]["credit_card"] += pii_count // 3
                    elif "email" in pattern.lower():
                        summary["pii_by_type"]["email"] += pii_count // 3

        return summary

    def _write_summary_report(self, f, metadata: Dict[str, Any]) -> None:
        """Write human-readable summary report."""
        f.write("DATASET PIPELINE EXECUTION SUMMARY\n")
        f.write("=" * 60 + "\n")

        exec_stats = metadata["execution_stats"]
        duration = exec_stats["duration"]

        f.write(f"Execution Time: {duration:.2f} seconds\n")
        f.write(f"Peak Memory Usage: {exec_stats['peak_memory_mb']:.2f} MB\n")
        f.write(f"Instruction Examples: {exec_stats['instruct_count']}\n")
        f.write(f"Retrieval Pairs: {exec_stats['retrieval_count']}\n")
        if duration > 0:
            f.write(
                "Throughput: "
                f"{exec_stats['instruct_count'] / duration:.2f} instruct/sec, "
                f"{exec_stats['retrieval_count'] / duration:.2f} retrieval/sec\n"
            )

        progress_stats = exec_stats.get("progress_stats", {})
        f.write("\nProgress Statistics:\n")
        f.write(
            f"- Total Instruction Examples: {progress_stats.get('instruction_total', 0)}\n"
        )
        f.write(
            f"- Total Retrieval Pairs: {progress_stats.get('retrieval_total', 0)}\n"
        )
        f.write(
            f"- Duplicate Instructions Removed: {progress_stats.get('duplicate_instruction', 0)}\n"
        )
        f.write(
            f"- Duplicate Retrieval Pairs Removed: {progress_stats.get('duplicate_retrieval', 0)}\n"
        )
        f.write(
            f"- Elapsed Time: {progress_stats.get('elapsed_seconds', 0.0):.2f} seconds\n"
        )

        # Dataset breakdown
        f.write("\nDATASETS PROCESSED:\n")
        f.write("-" * 30 + "\n")

        for name, stats in metadata["dataset_stats"].items():
            meta = stats.get("metadata", {})
            f.write(f"- {name} ({stats.get('type', 'unknown')}):\n")
            f.write(f"  * Examples used: {stats.get('count', 0)}\n")
            f.write(f"  * Processing time: {stats.get('time', 0.0):.2f}s\n")
            f.write(f"  * Quality score: {meta.get('quality_score', 0.0):.3f}\n")
            f.write(f"  * Duplicates removed: {meta.get('duplicates_removed', 0)}\n")
            if meta.get("near_duplicates_removed"):
                f.write(
                    f"  * Near duplicates removed: {meta.get('near_duplicates_removed', 0)}\n"
                )
            f.write(f"  * Quality filtered: {meta.get('filtered_by_quality', 0)}\n")
            if meta.get("filtered_by_pii"):
                f.write(f"  * PII filtered: {meta.get('filtered_by_pii', 0)}\n")
            f.write(f"  * Errors: {meta.get('loading_errors', 0)}\n")
            if meta.get("skipped_due_to_script_requirement"):
                f.write(
                    "  * Skipped: requires trust_remote_code (enable via "
                    "--allow_trust_remote_code or --allow_trust_remote_code_for)\n"
                )
            if meta.get("languages"):
                f.write(f"  * Languages: {dict(meta['languages'])}\n")
            f.write("\n")

        # Quality summary
        f.write("DATA QUALITY SUMMARY:\n")
        f.write("-" * 30 + "\n")

        val_results = metadata["validation_results"]
        instr_stats = val_results["instruct_stats"]
        retr_stats = val_results["retrieval_stats"]

        f.write(f"Instruction Examples: {instr_stats['total']}\n")
        f.write(
            f"  - Avg instruction length: {instr_stats['avg_instruction_length']:.1f} chars\n"
        )
        f.write(
            f"  - Avg output length: {instr_stats['avg_output_length']:.1f} chars\n"
        )
        f.write(f"  - By language: {dict(instr_stats['by_language'])}\n")

        f.write(f"\nRetrieval Pairs: {retr_stats['total']}\n")
        f.write(f"  - Avg text1 length: {retr_stats['avg_text1_length']:.1f} chars\n")
        f.write(f"  - Avg text2 length: {retr_stats['avg_text2_length']:.1f} chars\n")
        f.write(f"  - By language: {dict(retr_stats['by_language'])}\n")

        # License compliance
        f.write("\nLICENSE COMPLIANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")

        license_summary = metadata["license_compliance_summary"]
        for license_type, datasets in license_summary["datasets_by_license"].items():
            f.write(f"- {license_type}: {len(datasets)} datasets\n")

        if license_summary["compliance_issues"]:
            f.write("\nCOMPLIANCE ISSUES FOUND:\n")
            for issue in license_summary["compliance_issues"]:
                f.write(f"  - {issue}\n")
        else:
            f.write("\nNo license compliance issues detected.\n")

        if license_summary["commercial_use_compatible"]:
            f.write("\nDatasets compatible with commercial use:\n")
            for name in license_summary["commercial_use_compatible"]:
                f.write(f"  - {name}\n")

        if license_summary["attribution_required"]:
            f.write("\nDatasets requiring attribution:\n")
            for name in license_summary["attribution_required"]:
                f.write(f"  - {name}\n")

        if license_summary["modification_allowed"]:
            f.write("\nDatasets allowing modification:\n")
            for name in license_summary["modification_allowed"]:
                f.write(f"  - {name}\n")

        # PII summary
        if self.config.pii_config.enabled:
            f.write("\nPII SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Total PII instances detected (before filtering): "
                f"{metadata['pii_summary']['total_instances_found']}\n"
            )
            if metadata["pii_summary"]["datasets_with_pii"]:
                f.write("Datasets with PII detections:\n")
                for name, count in metadata["pii_summary"]["datasets_with_pii"].items():
                    f.write(f"  - {name}: {count}\n")
            if metadata["pii_summary"]["pii_by_type"]:
                f.write("PII by approximate type:\n")
                for pii_type, count in metadata["pii_summary"]["pii_by_type"].items():
                    f.write(f"  - {pii_type}: {count}\n")

        # Errors
        errors = metadata.get("errors", [])
        if errors:
            f.write("\nPIPELINE ERRORS:\n")
            f.write("-" * 30 + "\n")
            for err in errors:
                f.write(f"- {err}\n")


# -----------------------------
# Command-line Interface
# -----------------------------
def _normalize_langs(lang_arg: str) -> List[str]:
    """Parse comma-separated language codes and normalize casing/format."""

    raw_langs = [part.strip() for part in lang_arg.split(",") if part.strip()]
    if not raw_langs:
        return ["fr"]

    normalized = []
    for code in raw_langs:
        code_norm = code.lower().replace("_", "-")
        # Keep region-specific variants (e.g., fr-ca) but also allow the base code
        normalized.append(code_norm)
    if "fr" not in normalized:
        normalized.append("fr")
    return list(dict.fromkeys(normalized))


def _parse_trusted_dataset_allowlist(raw: str) -> Set[str]:
    """Normalize comma-separated dataset identifiers for trust_remote_code allowlist."""

    return {part.strip().lower() for part in raw.split(",") if part and part.strip()}


def _lang_matches(lang_value: str, selected_langs: Set[str]) -> bool:
    """Return True when a language code matches the selected set (exact or base code)."""

    normalized = lang_value.lower().replace("_", "-") if lang_value else ""
    candidates = {normalized}
    if "-" in normalized:
        candidates.add(normalized.split("-", 1)[0])

    return any(candidate in selected_langs for candidate in candidates if candidate)


def parse_args() -> PipelineConfig:
    """Parse command line arguments and build a PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Consolidated Advanced Dataset Pipeline for Multi-Task Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core configuration
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="fr",
        help="Comma-separated languages to include (French only for this pipeline)",
    )
    parser.add_argument(
        "--max_per_dataset",
        type=int,
        default=50000,
        help="Maximum examples to keep per dataset",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="dataset_metadata.json",
        help="Name of metadata JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling/shuffling"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face datasets cache directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Maximum number of worker threads for batch processing",
    )

    # Behaviour toggles
    parser.add_argument(
        "--enable_commercial_use",
        action="store_true",
        help="Enable commercial-use mode (will enforce license checks)",
    )
    parser.add_argument(
        "--disable_modification",
        dest="enable_modification",
        action="store_false",
        help="Disable modification of licensed content",
    )
    parser.set_defaults(enable_modification=True)

    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Enable checkpointing for crash recovery",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Save checkpoint every N processed examples",
    )
    parser.add_argument(
        "--resume",
        dest="resume_from_checkpoint",
        action="store_true",
        help="Resume from last pipeline checkpoint if present",
    )

    parser.add_argument(
        "--ignore_failed_datasets",
        action="store_true",
        help="Skip datasets that fail to load instead of raising",
    )

    parser.add_argument(
        "--enable_validation",
        action="store_true",
        help="Run validation on final merged dataset",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation splits",
    )
    parser.add_argument(
        "--validation_max_size",
        type=int,
        default=5000,
        help="Maximum validation set size per task",
    )

    # Logging / monitoring
    parser.add_argument(
        "--dashboard_port",
        type=int,
        default=None,
        help="Port for optional HTTP dashboard (Flask)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Base logging level",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Shortcut for DEBUG logging on console",
    )

    # HF + downloads (trust_remote_code guarded)
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of datasets instead of reusing cache",
    )
    parser.add_argument(
        "--allow_trust_remote_code",
        action="store_true",
        help=(
            "Allow datasets that require trust_remote_code (only enable for vetted sources)."
        ),
    )
    parser.add_argument(
        "--allow_trust_remote_code_for",
        type=str,
        default="",
        help=(
            "Comma-separated dataset keys/paths to allow trust_remote_code for (e.g., xp3,aya)."
        ),
    )

    # Memory config
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=32.0,
        help="Maximum allowed RSS memory in GB before hard stop",
    )
    parser.add_argument(
        "--gc_threshold_mb",
        type=float,
        default=512.0,
        help="Soft threshold (MB) above which garbage collection is triggered",
    )
    parser.add_argument(
        "--batch_memory_limit_mb",
        type=float,
        default=256.0,
        help="Target max memory per batch before flushing",
    )
    parser.add_argument(
        "--aggressive_gc",
        action="store_true",
        help="Use more aggressive GC when under memory pressure",
    )
    parser.add_argument(
        "--health_check_interval",
        type=int,
        default=60,
        help="Seconds between health checks",
    )
    parser.add_argument(
        "--max_runtime_hours",
        type=float,
        default=24.0,
        help="Maximum allowed runtime before forced shutdown",
    )

    # Outputs / extras
    parser.add_argument(
        "--unsloth_export_name",
        type=str,
        default="unsloth_prompt_completion.jsonl",
        help="Filename for derived prompt/completion export",
    )
    parser.add_argument(
        "--cloud_storage",
        type=str,
        default=None,
        help='JSON blob for cloud target (e.g. {"provider":"aws","bucket":"my-bucket","prefix":"datasets/"})',
    )

    args = parser.parse_args()

    langs = _normalize_langs(args.langs)
    trusted_allowlist = _parse_trusted_dataset_allowlist(
        args.allow_trust_remote_code_for
    )

    # Nested configs – you can later expose CLI flags for these if you want
    quality_config = QualityConfig(
        min_chars=5,  # Relaxed for French
        min_words=1,  # Relaxed for French
        min_quality_score=0.3,  # Relaxed to capture more data initially
        language_detection_threshold=0.5,  # Relaxed for flexibility
    )
    pii_config = PIIConfig()
    dedup_config = DedupConfig()
    memory_config = MemoryConfig(
        max_memory_gb=args.max_memory_gb,
        gc_threshold_mb=args.gc_threshold_mb,
        batch_memory_limit_mb=args.batch_memory_limit_mb,
        aggressive_gc=args.aggressive_gc,
    )
    cloud_storage = json.loads(args.cloud_storage) if args.cloud_storage else None

    cfg = PipelineConfig(
        output_dir=args.output_dir,
        langs=langs,  # Enforce French only
        max_per_dataset=args.max_per_dataset,
        metadata_file=args.metadata_file,
        seed=args.seed,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        download_mode="reuse_dataset_if_exists",
        quality_config=quality_config,
        memory_config=memory_config,
        dedup_config=dedup_config,
        pii_config=pii_config,
        enable_progress_bar=True,
        enable_checkpointing=args.enable_checkpointing,
        checkpoint_interval=args.checkpoint_interval,
        enable_commercial_use=args.enable_commercial_use,
        enable_modification=args.enable_modification,
        dashboard_port=args.dashboard_port,
        log_level=args.log_level,
        force_download=args.force_download,
        allow_trust_remote_code=args.allow_trust_remote_code,
        trusted_remote_code_datasets=trusted_allowlist,
        state_file="pipeline_state.pkl",
        dedup_state_file="dedup_state.pkl",
        resume_from_checkpoint=args.resume_from_checkpoint,
        ignore_failed_datasets=args.ignore_failed_datasets,
        sampling_strategy="uniform",
        quality_threshold=0.6,
        min_examples_per_language=1000,
        max_examples_per_language=100000,
        enable_validation=args.enable_validation,
        validation_split=args.validation_split,
        validation_max_size=args.validation_max_size,
        verbose=args.verbose,
        unsloth_export_name=args.unsloth_export_name,
        cloud_storage=cloud_storage,
        distributed_mode=False,
        node_rank=0,
        world_size=1,
        health_check_interval=args.health_check_interval,
        max_runtime_hours=args.max_runtime_hours,
        auto_scaling=True,
    )

    return cfg


def main() -> int:
    """Main entry point."""
    try:
        config = parse_args()

        # Ensure output dir exists before logger tries to open files
        os.makedirs(config.output_dir, exist_ok=True)

        global logger
        logger = setup_logging(
            log_file=os.path.join(config.output_dir, "dataset_pipeline.log"),
            verbose=config.verbose,
            log_level=config.log_level,
        )

        # Nice little CONFIG banner like your first script
        print(
            f"[CONFIG] langs={config.langs} "
            f"max_per_dataset={config.max_per_dataset} "
            f"seed={config.seed} "
            f"output_dir='{config.output_dir}'"
        )

        pipeline = DatasetPipeline(config)
        pipeline.run()

        print("[INFO] Real pipeline finished")
        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
