#!/usr/bin/env python3
"""
Advanced Dataset Pipeline for Multi-Task Learning
===============================================

A robust, production-grade pipeline for aggregating, processing, and validating
instruction and retrieval datasets across multiple languages and domains.
Features comprehensive error handling, memory optimization, license compliance,
and real-time monitoring.

Key Features:
- 🔄 Resilient checkpointing with crash recovery
- 🛡️ Advanced PII detection and filtering
- 📊 Real-time monitoring dashboard with metrics
- 🧠 Near-duplicate detection with MinHash/LSH
- 📈 Automatic dataset quality scoring
- 🔐 License compliance validation
- 💾 Atomic file operations to prevent data corruption
- 🚡 Memory pressure handling with GC tuning
- ⚡ Parallel processing for optimal throughput
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
    TypeVar,
    Union,
)

import psutil
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from datasets import (
    Dataset,
    DatasetDict,
    DownloadMode,
    VerificationMode,
    load_dataset,
)

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
    log_file: str = "dataset_pipeline.log", verbose: bool = False
) -> logging.Logger:
    """
    Configure comprehensive logging with rotation, color coding, and structured output.

    Args:
        log_file: Path to log file (rotated at 100MB)
        verbose: Enable debug level logging

    Returns:
        Configured logger instance
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO
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
    languages: List[str] = field(default_factory=lambda: ["en"])
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

    min_chars: int = 10
    max_chars: int = 10000
    min_words: int = 3
    max_words: int = 2000
    min_quality_score: float = 0.5
    language_detection_threshold: float = 0.8
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
    langs: List[str] = field(default_factory=lambda: ["en", "fr"])
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
    trust_remote_code: bool = True
    force_download: bool = False
    state_file: str = "pipeline_state.pkl"
    dedup_state_file: str = "dedup_state.pkl"
    resume_from_checkpoint: bool = False
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


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

    def __init__(self, config: DedupConfig, state_file: Optional[Path] = None):
        """
        Initialize deduplication engine with configurable strategies.

        Args:
            config: Deduplication configuration
            state_file: Path to state file for persistence
        """
        self.config = config
        self.state_file = Path(state_file) if state_file else None
        self.exact_dedup_keys: Set[str] = set()
        self.near_dedup_engine: Optional[MinHashLSH] = None
        self.minhashes: Dict[str, MinHash] = {}
        self.quality_scores: Dict[str, float] = {}

        # Load state if available
        self.load_state()

        # Initialize near-dedup engine if enabled
        if config.enable_near_dedup:
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
            if self.config.enable_near_dedup:
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
        if self.config.enable_near_dedup and self.near_dedup_engine:
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
        if not self.config.enable_near_dedup or not self.near_dedup_engine:
            return 0.0

        m1 = self.get_minhash(text1)
        m2 = self.get_minhash(text2)
        return m1.jaccard(m2)


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

    prompt_text = "\n\n".join(part for part in prompt_parts if part)
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

    # Basic length checks
    if (
        length < config.quality_config.min_chars
        or length > config.quality_config.max_chars
    ):
        return False

    # Word count checks
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
    trust_remote_code: bool = True,
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
        trust_remote_code: Trust remote code execution
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
                trust_remote_code=trust_remote_code,
            )

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

            # Language filtering
            lang = result.get("language", "en")
            if lang not in selected_langs:
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

            lang = example.get("language", example.get("lang", "en"))[:2].lower()

            return {
                "instruction": instr,
                "output": out,
                "source": dataset_config.name,
                "language": lang,
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

            lang = example.get("language", example.get("lang", "en"))[:2].lower()

            return {
                "text_1": t1,
                "text_2": t2,
                "source": dataset_config.name,
                "language": lang,
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

        try:
            # License compliance check
            compliance_issues = self.validate_license_compliance(dataset_config)
            if compliance_issues:
                self.metadata["license_compliance_issues"] = compliance_issues
                if (
                    not os.environ.get("IGNORE_LICENSE_COMPLIANCE", "false").lower()
                    == "true"
                ):
                    self.logger.warning(
                        f"License compliance issues for {dataset_config.name}: {compliance_issues}"
                    )
                    if self.config.enable_commercial_use:
                        raise LicenseComplianceError(
                            f"License compliance issues: {compliance_issues}"
                        )

            # Load dataset with retry logic
            ds = load_dataset_with_retry(
                dataset_config.hf_path,
                dataset_config.hf_config,
                split="train",
                streaming=dataset_config.streaming,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=self.config.trust_remote_code,
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


# -----------------------------
# Specific Dataset Loader Implementations
# -----------------------------


class XP3MultiLanguageLoader(InstructionDatasetLoader):
    """Loader for BigScience xP3 dataset with multi-language support."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Load xP3 for multiple languages with per-language processing."""
        # Filter languages to only those available in xP3
        available_langs = [
            "en",
            "fr",
            "es",
            "de",
            "zh",
            "ru",
            "ar",
            "hi",
            "vi",
            "bg",
            "ca",
            "hr",
        ]
        relevant_langs = [lang for lang in selected_langs if lang in available_langs]

        if not relevant_langs:
            self.logger.info(
                f"Skipping xP3 - no relevant languages in {selected_langs}"
            )
            return

        # Process each language separately
        total_examples = 0
        start_time = time.time()

        for lang in relevant_langs:
            self.logger.info(f"Processing xP3 for language: {lang}")

            # Create language-specific config
            lang_config = DatasetConfig(
                name=f"xP3-{lang}",
                hf_path="bigscience/xP3",
                hf_config=lang,
                dataset_type=DatasetType.INSTRUCTION,
                languages=[lang],
                license=LicenseType.APACHE_2_0,
                license_compliance=LicenseCompliance(
                    license_type=LicenseType.APACHE_2_0,
                    requires_attribution=True,
                    allows_commercial_use=True,
                    allows_modification=True,
                    attribution_text="BigScience xP3 dataset",
                ),
                streaming=True,
                max_examples=(
                    dataset_config.max_examples // len(relevant_langs)
                    if dataset_config.max_examples
                    else None
                ),
                batch_size=dataset_config.batch_size,
            )

            # Custom transform function for xP3
            def transform(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                instr = safe_get(ex, "inputs")
                out = safe_get(ex, "targets")
                if not instr or not out:
                    return None
                return {
                    "instruction": instr,
                    "output": out,
                    "source": f"bigscience/xP3-{lang}",
                    "language": lang,
                    "metadata": {"task": ex.get("task", ""), "language": lang},
                }

            lang_config.transform_function = transform

            # Create temporary loader for this language
            lang_loader = InstructionDatasetLoader(self.config)
            if self.dedup_engine:
                lang_loader.set_dedup_engine(self.dedup_engine)

            # Load language-specific data
            lang_loader.load(selected_langs, seen_keys, sink, lang_config)

            # Merge metadata
            for key, value in lang_loader.metadata.items():
                if key == "languages":
                    for lang_code, count in value.items():
                        self.metadata["languages"][lang_code] += count
                elif isinstance(value, (int, float)):
                    self.metadata[key] = self.metadata.get(key, 0) + value

            total_examples += lang_loader.metadata["examples_used"]

        self.metadata["processing_time"] = time.time() - start_time
        self.logger.info(
            f"xP3 total examples across {len(relevant_langs)} languages: {total_examples}"
        )


class AyaCollectionLoader(InstructionDatasetLoader):
    """Loader for Cohere Aya Collection dataset."""

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    def load(
        self,
        selected_langs: Set[str],
        seen_keys: Set[str],
        sink: List[Dict[str, Any]],
        dataset_config: DatasetConfig,
    ) -> None:
        """Load Aya Collection dataset with advanced filtering."""
        # Create specific config for Aya
        aya_config = DatasetConfig(
            name="CohereAya",
            hf_path="CohereForAI/aya_collection",
            dataset_type=DatasetType.INSTRUCTION,
            languages=dataset_config.languages,
            license=LicenseType.APACHE_2_0,
            license_compliance=LicenseCompliance(
                license_type=LicenseType.APACHE_2_0,
                requires_attribution=True,
                allows_commercial_use=True,
                allows_modification=True,
                attribution_text="Cohere Aya Collection",
            ),
            streaming=False,
            max_examples=dataset_config.max_examples,
            quality_weight=1.2,  # Higher quality weight
        )

        # Custom transform function for Aya
        def transform(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Extract instruction and output with fallbacks
            instr = safe_get(ex, "instruction", "prompt", "inputs", "question")
            out = safe_get(ex, "output", "response", "targets", "answer", "completion")

            if not instr or not out:
                return None

            # Determine language
            lang = ex.get("language", ex.get("lang", ex.get("locale", "en"))).lower()[
                :2
            ]

            # Skip if not in selected languages
            if lang not in selected_langs:
                return None

            # Additional quality checks specific to Aya
            if len(instr.split()) < 2 or len(out.split()) < 2:
                return None

            return {
                "instruction": instr,
                "output": out,
                "source": "CohereForAI/aya_collection",
                "language": lang,
                "metadata": {
                    "collection": ex.get("collection", ""),
                    "task_type": ex.get("task_type", ""),
                    "original_language": ex.get("original_language", lang),
                },
            }

        aya_config.transform_function = transform
        aya_config.filter_function = lambda ex: ex.get(
            "is_response_matching_human", True
        )  # Only high-quality responses

        super().load(selected_langs, seen_keys, sink, aya_config)


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
            trust_remote_code=self.config.trust_remote_code,
        )

        # Build conversation trees
        messages_by_id = {}
        root_messages = []

        for msg in tqdm(ds, desc="Building conversation trees"):
            msg_id = msg.get("message_id")
            parent_id = msg.get("parent_id")
            lang = msg.get("lang", "en")[:2].lower()

            if lang not in selected_langs:
                continue

            messages_by_id[msg_id] = msg

            if not parent_id or parent_id not in messages_by_id:
                root_messages.append(msg)

        # Process conversations
        processed_count = 0
        self.metadata["examples_loaded"] = len(root_messages)

        for root in tqdm(root_messages, desc="Processing conversations"):
            conversation = self._extract_conversation(
                root, messages_by_id, selected_langs
            )
            if conversation:
                for turn in conversation:
                    item = {
                        "instruction": turn["instruction"],
                        "output": turn["output"],
                        "source": dataset_config.name,
                        "language": turn["language"],
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
            f"Processed {processed_count} conversation turns from OpenAssistant"
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
            lang = current.get("lang", "en")[:2].lower()

            if not text or lang not in selected_langs:
                continue

            if role == "assistant" and context:
                # Create instruction-output pair from context and response
                instruction = "\n".join(
                    [f"{ctx['role'].title()}: {ctx['text']}" for ctx in context]
                )
                turns.append(
                    {
                        "instruction": instruction,
                        "output": text,
                        "language": lang,
                        "message_id": current.get("message_id", ""),
                        "depth": depth,
                    }
                )

            # Add children to stack
            replies = current.get("replies", [])
            for reply_id in reversed(replies):  # Process in order
                if reply_id in messages_by_id:
                    reply = messages_by_id[reply_id]
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
            # Load dataset
            ds = load_dataset_with_retry(
                dataset_config.hf_path,
                dataset_config.hf_config,
                split="train",
                streaming=dataset_config.streaming,
                cache_dir=self.config.cache_dir,
                timeout=dataset_config.timeout,
                requires_auth=dataset_config.requires_auth,
                trust_remote_code=self.config.trust_remote_code,
            )

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
        # MIRACL language codes
        miracl_langs = {
            "en": "en",
            "fr": "fr",
            "es": "es",
            "de": "de",
            "zh": "zh",
            "ar": "ar",
            "bn": "bn",
            "fi": "fi",
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

            # Custom transform for MIRACL
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
                    "language": lang,
                    "metadata": {
                        "query_id": ex.get("query_id", ""),
                        "passage_id": pos.get("docid", ""),
                        "relevance_score": 1.0,
                    },
                }

            lang_config.transform_function = transform

            # Create temporary loader
            lang_loader = RetrievalDatasetLoader(self.config)
            if self.dedup_engine:
                lang_loader.set_dedup_engine(self.dedup_engine)

            # Load language data
            lang_loader.load(selected_langs, seen_keys, sink, lang_config)

            # Merge metadata
            for key, value in lang_loader.metadata.items():
                if key == "languages":
                    for lang_code, count in value.items():
                        self.metadata["languages"][lang_code] += count
                elif isinstance(value, (int, float)):
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

    def load_state(self) -> bool:
        """Load pipeline state from file."""
        if not self.config.resume_from_checkpoint or not self.state_file.exists():
            return False

        try:
            with open(self.state_file, "rb") as f:
                state_dict = pickle.load(f)

            self.state = PipelineState.from_dict(state_dict)
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
            config_hash = hashlib.md5(
                json.dumps(self.config.to_dict()).encode()
            ).hexdigest()

            state = PipelineState(
                config_hash=config_hash,
                datasets_processed=list(pipeline.stats["datasets_processed"].keys()),
                instruct_count=len(pipeline.instruct_data),
                retrieval_count=len(pipeline.retrieval_data),
                memory_usage_gb=memory_usage(),
                errors=pipeline.errors,
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
        self.server_thread = None
        self.running = False

    def start(self) -> bool:
        """Start dashboard server in background thread."""
        try:
            import threading

            from flask import Flask, jsonify, render_template

            self.app = Flask(__name__)
            self.app.logger.setLevel(logging.ERROR)  # Reduce Flask logging noise

            @self.app.route("/")
            def dashboard():
                """Main dashboard page."""
                stats = self.pipeline.get_stats()
                return render_template("dashboard.html", stats=stats)

            @self.app.route("/api/stats")
            def api_stats():
                """JSON API for statistics."""
                return jsonify(self.pipeline.get_stats())

            @self.app.route("/api/datasets")
            def api_datasets():
                """JSON API for dataset information."""
                return jsonify(self.pipeline.get_dataset_stats())

            def run_server():
                """Run Flask server."""
                self.running = True
                try:
                    self.app.run(
                        host="0.0.0.0", port=self.port, debug=False, use_reloader=False
                    )
                finally:
                    self.running = False

            # Start server in background thread
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

            # Wait for server to start
            time.sleep(1)

            if self.running:
                logger.info(f"Dashboard available at http://localhost:{self.port}")
                return True

            return False

        except ImportError:
            logger.warning("Flask not installed. Install with: pip install flask")
            return False
        except Exception as e:
            logger.warning(f"Failed to start dashboard: {e}")
            return False

    def stop(self):
        """Stop dashboard server."""
        if self.server_thread and self.server_thread.is_alive():
            # Note: Flask doesn't provide a clean shutdown mechanism
            # This is a limitation of the current implementation
            logger.info("Stopping dashboard server (may require manual termination)")


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
        self.output_artifacts: Dict[str, Dict[str, Any]] = {}
        self._health_thread: Optional[threading.Thread] = None
        self._health_stop_event = threading.Event()
        self._health_violation: Optional[BaseException] = None

        # Initialize deduplication engine
        self.dedup_engine = None
        if (
            config.dedup_config.enable_exact_dedup
            or config.dedup_config.enable_near_dedup
        ):
            dedup_state_file = Path(config.output_dir) / config.dedup_state_file
            self.dedup_engine = DeduplicationEngine(
                config.dedup_config, state_file=dedup_state_file
            )

        # Initialize state manager
        self.state_manager = PipelineStateManager(config)
        self.state_manager.load_state()

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
        """Get detailed statistics for all datasets."""
        dataset_stats = {}

        for name, loader in self.loaders.items():
            dataset_stats[name] = {
                "examples_used": loader.metadata["examples_used"],
                "examples_loaded": loader.metadata["examples_loaded"],
                "duplicates_removed": loader.metadata["duplicates_removed"],
                "near_duplicates_removed": loader.metadata.get(
                    "near_duplicates_removed", 0
                ),
                "filtered_by_quality": loader.metadata["filtered_by_quality"],
                "filtered_by_pii": loader.metadata.get("filtered_by_pii", 0),
                "loading_errors": loader.metadata["loading_errors"],
                "processing_time": loader.metadata["processing_time"],
                "quality_score": loader.metadata.get("quality_score", 0.0),
                "languages": dict(loader.metadata["languages"]),
                "license_compliance_issues": loader.metadata.get(
                    "license_compliance_issues", []
                ),
                "pii_instances_found": loader.metadata.get("pii_instances_found", 0),
            }

        return dataset_stats

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

            except BaseException as exc:  # Catch all to avoid silent thread death
                self._health_violation = exc
                self.logger.error("Health monitor encountered an error: %s", exc)
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
            self._check_health_status()

            # Generate comprehensive metadata
            self._generate_metadata()
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

            # Stop dashboard if running
            if self.dashboard:
                self.dashboard.stop()

            self._stop_health_monitor()

            # Log final statistics
            duration = time.time() - self.stats["start_time"]
            self.logger.info(f"Pipeline completed in {duration:.2f} seconds")
            self.logger.info(f"Total instruction examples: {len(self.instruct_data)}")
            self.logger.info(f"Total retrieval pairs: {len(self.retrieval_data)}")

            # Save dedup state if enabled
            if self.dedup_engine:
                self.dedup_engine.save_state()

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
                self.state_manager.save_state(
                    self
                )  # Save checkpoint after each dataset
                self._check_health_status()

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
                self.state_manager.save_state(
                    self
                )  # Save checkpoint after each dataset
                self._check_health_status()

    def _get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """Get dataset configuration based on name."""
        configs = {
            "xp3": DatasetConfig(
                name="bigscience/xP3",
                hf_path="bigscience/xP3",
                dataset_type=DatasetType.INSTRUCTION,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
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
                dataset_type=DatasetType.INSTRUCTION,
                languages=self.config.langs,
                license=LicenseType.APACHE_2_0,
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

        try:
            import boto3
        except ImportError:
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

        except Exception as e:
            self.logger.error(f"Error writing output files: {str(e)}")
            self.stats["errors"].append(f"Output writing failed: {str(e)}")
            raise

    def _safe_write_jsonl(self, data: List[Dict[str, Any]], path: Path) -> None:
        """Write JSONL file with atomic operations."""
        # Create temp file
        temp_file = path.with_suffix(".tmp")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Atomic rename
            if path.exists():
                path.unlink()
            temp_file.rename(path)

        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise IOError(f"Failed to write {path}: {str(e)}")

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
                f"Created instruction validation split: {len(instruct_train)} train, {len(instruct_val)} val"
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
                f"Created retrieval validation split: {len(retrieval_train)} train, {len(retrieval_val)} val"
            )

    def _validate_results(self) -> None:
        """Validate final results for quality and consistency."""
        self.logger.info("Validating final results...")

        validation_results = {
            "instruct_stats": {
                "total": len(self.instruct_data),
                "by_language": defaultdict(int),
                "avg_instruction_length": 0,
                "avg_output_length": 0,
            },
            "retrieval_stats": {
                "total": len(self.retrieval_data),
                "by_language": defaultdict(int),
                "avg_text1_length": 0,
                "avg_text2_length": 0,
            },
            "quality_issues": [],
            "compliance_issues": [],
        }

        # Validate instruction data
        instr_lengths = []
        output_lengths = []

        for ex in self.instruct_data:
            lang = ex.get("language", "unknown")
            validation_results["instruct_stats"]["by_language"][lang] += 1

            instr_lengths.append(len(ex["instruction"]))
            output_lengths.append(len(ex["output"]))

        if instr_lengths:
            validation_results["instruct_stats"]["avg_instruction_length"] = sum(
                instr_lengths
            ) / len(instr_lengths)
            validation_results["instruct_stats"]["avg_output_length"] = sum(
                output_lengths
            ) / len(output_lengths)

        # Validate retrieval data
        text1_lengths = []
        text2_lengths = []

        for ex in self.retrieval_data:
            lang = ex.get("language", "unknown")
            validation_results["retrieval_stats"]["by_language"][lang] += 1

            text1_lengths.append(len(ex["text_1"]))
            text2_lengths.append(len(ex["text_2"]))

        if text1_lengths:
            validation_results["retrieval_stats"]["avg_text1_length"] = sum(
                text1_lengths
            ) / len(text1_lengths)
            validation_results["retrieval_stats"]["avg_text2_length"] = sum(
                text2_lengths
            ) / len(text2_lengths)

        # Check for potential issues
        if validation_results["instruct_stats"][
            "total"
        ] < self.config.min_examples_per_language * len(self.config.langs):
            validation_results["quality_issues"].append(
                f"Low instruction example count: {validation_results['instruct_stats']['total']}"
            )

        if validation_results["retrieval_stats"][
            "total"
        ] < self.config.min_examples_per_language * len(self.config.langs):
            validation_results["quality_issues"].append(
                f"Low retrieval pair count: {validation_results['retrieval_stats']['total']}"
            )

        # Log validation results
        self.logger.info("Validation results:")
        self.logger.info(
            f"Instruction examples: {validation_results['instruct_stats']['total']}"
        )
        self.logger.info(
            f"Retrieval pairs: {validation_results['retrieval_stats']['total']}"
        )

        for issue in validation_results["quality_issues"]:
            self.logger.warning(f"Quality issue: {issue}")

        return validation_results

    def _generate_metadata(self) -> None:
        """Generate comprehensive metadata with performance metrics."""
        metadata_path = Path(self.config.output_dir) / self.config.metadata_file
        summary_path = Path(self.config.output_dir) / "pipeline_summary.txt"

        try:
            # Prepare metadata
            metadata = {
                "pipeline_config": self.config.to_dict(),
                "execution_stats": {
                    "start_time": self.stats["start_time"],
                    "end_time": time.time(),
                    "duration": time.time() - self.stats["start_time"],
                    "peak_memory_mb": self.stats["memory_peak_mb"],
                    "instruct_count": len(self.instruct_data),
                    "retrieval_count": len(self.retrieval_data),
                },
                "dataset_stats": self.get_dataset_stats(),
                "validation_results": self._validate_results(),
                "license_compliance_summary": self._generate_license_summary(),
                "pii_summary": self._generate_pii_summary(),
                "derived_outputs": self.output_artifacts,
            }

            # Write metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Write human-readable summary
            with open(summary_path, "w", encoding="utf-8") as f:
                self._write_summary_report(f, metadata)

            self._register_output("metadata", metadata_path, 0)
            self._register_output("summary_report", summary_path, 0)

            self.logger.info(f"Metadata written to {metadata_path}")
            self.logger.info(f"Summary report written to {summary_path}")

        except Exception as e:
            self.logger.error(f"Error generating metadata: {str(e)}")
            self.stats["errors"].append(f"Metadata generation failed: {str(e)}")

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
        f.write("=" * 60 + "\n\n")

        # Execution summary
        exec_stats = metadata["execution_stats"]
        duration = exec_stats["duration"]
        f.write(f"Execution Time: {duration:.2f} seconds\n")
        f.write(f"Peak Memory Usage: {exec_stats['peak_memory_mb']:.2f} MB\n")
        f.write(f"Instruction Examples: {exec_stats['instruct_count']}\n")
        f.write(f"Retrieval Pairs: {exec_stats['retrieval_count']}\n")
        f.write(
            f"Throughput: {exec_stats['instruct_count'] / duration:.2f} instruct/sec, "
            f"{exec_stats['retrieval_count'] / duration:.2f} retrieval/sec\n\n"
        )

        # Dataset breakdown
        f.write("DATASETS PROCESSED:\n")
        f.write("-" * 30 + "\n")

        for name, stats in metadata["dataset_stats"].items():
            f.write(f"- {name} ({stats['type']}):\n")
            f.write(f"  * Examples used: {stats['count']}\n")
            f.write(f"  * Processing time: {stats['time']:.2f}s\n")
            f.write(f"  * Quality score: {stats.get('quality_score', 0.0):.3f}\n")
            f.write(f"  * Duplicates removed: {stats['duplicates_removed']}\n")
            if stats.get("near_duplicates_removed", 0):
                f.write(
                    f"  * Near duplicates removed: {stats['near_duplicates_removed']}\n"
                )
            f.write(f"  * Quality filtered: {stats['filtered_by_quality']}\n")
            if stats.get("filtered_by_pii", 0):
                f.write(f"  * PII filtered: {stats['filtered_by_pii']}\n")
            f.write(f"  * Errors: {stats['loading_errors']}\n")
            if stats["languages"]:
                f.write(f"  * Languages: {dict(stats['languages'])}\n")
            f.write("\n")

        # Quality summary
        f.write("DATA QUALITY SUMMARY:\n")
        f.write("-" * 30 + "\n")

        val_results = metadata["validation_results"]

        f.write(f"Instruction Examples: {val_results['instruct_stats']['total']}\n")
        f.write(
            f"  - Avg instruction length: {val_results['instruct_stats']['avg_instruction_length']:.1f} chars\n"
        )
        f.write(
            f"  - Avg output length: {val_results['instruct_stats']['avg_output_length']:.1f} chars\n"
        )
        f.write(
            f"  - By language: {dict(val_results['instruct_stats']['by_language'])}\n\n"
        )

        f.write(f"Retrieval Pairs: {val_results['retrieval_stats']['total']}\n")
        f.write(
            f"  - Avg text1 length: {val_results['retrieval_stats']['avg_text1_length']:.1f} chars\n"
        )
        f.write(
            f"  - Avg text2 length: {val_results['retrieval_stats']['avg_text2_length']:.1f} chars\n"
        )
        f.write(
            f"  - By language: {dict(val_results['retrieval_stats']['by_language'])}\n\n"
        )

        # Compliance summary
        f.write("LICENSE COMPLIANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")

        license_summary = metadata["license_compliance_summary"]
        for license_type, datasets in license_summary["datasets_by_license"].items():
            f.write(f"- {license_type}: {len(datasets)} datasets\n")

        if license_summary["compliance_issues"]:
            f.write("\nCOMPLIANCE ISSUES FOUND:\n")
            for issue in license_summary["compliance_issues"]:
                f.write(f"- {issue}\n")
        else:
            f.write("\nNo license compliance issues detected.\n")

        # PII summary
        if self.config.pii_config.enabled:
            f.write("\nPII DETECTION SUMMARY:\n")
            f.write("-" * 30 + "\n")

            pii_summary = metadata["pii_summary"]
            f.write(
                f"Total PII instances found: {pii_summary['total_instances_found']}\n"
            )

            if pii_summary["datasets_with_pii"]:
                f.write("Datasets with PII:\n")
                for dataset, count in pii_summary["datasets_with_pii"].items():
                    f.write(f"- {dataset}: {count} instances\n")

            if pii_summary["pii_by_type"]:
                f.write("PII by type:\n")
                for pii_type, count in pii_summary["pii_by_type"].items():
                    f.write(f"- {pii_type}: {count} instances\n")

        # Errors summary
        if self.stats["errors"]:
            f.write("\nERRORS ENCOUNTERED:\n")
            f.write("-" * 30 + "\n")
            for error in self.stats["errors"]:
                f.write(f"- {error}\n")


# -----------------------------
# CLI Interface
# -----------------------------


def parse_args() -> PipelineConfig:
    """Parse command line arguments with validation."""
    parser = argparse.ArgumentParser(
        description="Advanced multi-task dataset preparation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core configuration
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="en,fr",
        help="Comma-separated languages to include",
    )
    parser.add_argument(
        "--max_per_dataset",
        type=int,
        default=50000,
        help="Maximum examples/pairs per dataset",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="dataset_metadata.json",
        help="Metadata output filename",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Performance and memory
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("HF_HOME"),
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing",
    )
    parser.add_argument(
        "--max_memory_gb", type=float, default=32.0, help="Maximum memory usage in GB"
    )

    # Quality and filtering
    parser.add_argument(
        "--min_quality_score",
        type=float,
        default=0.5,
        help="Minimum quality score to keep examples",
    )
    parser.add_argument(
        "--disable_quality_filter",
        action="store_true",
        help="Disable heuristic quality filtering",
    )
    parser.add_argument(
        "--disable_pii_filtering", action="store_true", help="Disable PII filtering"
    )

    # Deduplication
    parser.add_argument(
        "--enable_near_dedup",
        action="store_true",
        help="Enable near-duplicate detection",
    )
    parser.add_argument(
        "--near_dedup_threshold",
        type=float,
        default=0.8,
        help="Threshold for near-duplicate detection (0.0-1.0)",
    )

    # Checkpointing and recovery
    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Enable checkpointing for crash recovery",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Save checkpoint every N examples",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume_from_checkpoint",
        help="Resume from last checkpoint",
    )

    # Monitoring and logging
    parser.add_argument(
        "--dashboard_port", type=int, help="Port for monitoring dashboard"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # License and compliance
    parser.add_argument(
        "--enable_commercial_use",
        action="store_true",
        help="Enable commercial use (check licenses first!)",
    )
    parser.add_argument(
        "--enable_modification",
        action="store_true",
        default=True,
        help="Enable dataset modification",
    )

    # Validation
    parser.add_argument(
        "--enable_validation",
        action="store_true",
        help="Enable validation split creation",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--validation_max_size",
        type=int,
        default=5000,
        help="Maximum validation set size",
    )

    # Advanced options
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code execution",
    )
    parser.add_argument(
        "--force_download", action="store_true", help="Force re-download of datasets"
    )
    parser.add_argument(
        "--unsloth_export_name",
        type=str,
        default="unsloth_prompt_completion.jsonl",
        help="Filename for the derived prompt/completion dataset",
    )
    parser.add_argument(
        "--cloud_storage",
        type=str,
        help='JSON blob describing cloud storage target (e.g. {"provider": "aws", "bucket": "my-bucket"})',
    )
    parser.add_argument(
        "--distributed_mode",
        action="store_true",
        help="Enable distributed coordination with node/world ranks",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of this node when distributed mode is enabled",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total nodes participating in distributed mode",
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
    parser.add_argument(
        "--disable_auto_scaling",
        action="store_true",
        help="Disable automatic batch down-scaling when memory pressure is detected",
    )

    args = parser.parse_args()

    # Parse languages
    langs = [lang.strip().lower() for lang in args.langs.split(",") if lang.strip()]
    if not langs:
        raise ValueError("At least one language must be specified")

    # Create quality config
    quality_config = QualityConfig(
        min_chars=10,
        max_chars=10000,
        min_words=3,
        max_words=2000,
        min_quality_score=args.min_quality_score,
        enable_spam_detection=not args.disable_quality_filter,
    )

    # Create PII config
    pii_config = PIIConfig(
        enabled=not args.disable_pii_filtering, redaction_strategy="remove"
    )

    # Create dedup config
    dedup_config = DedupConfig(
        enable_exact_dedup=True,
        enable_near_dedup=args.enable_near_dedup,
        near_dedup_threshold=args.near_dedup_threshold,
    )

    # Create memory config
    memory_config = MemoryConfig(max_memory_gb=args.max_memory_gb)

    # Validate distributed options
    if args.distributed_mode:
        if args.world_size <= 0:
            raise ValueError(
                "world_size must be greater than 0 when distributed mode is enabled"
            )
        if args.node_rank < 0 or args.node_rank >= args.world_size:
            raise ValueError(
                "node_rank must be within [0, world_size) when distributed mode is enabled"
            )

    cloud_storage = None
    if args.cloud_storage:
        try:
            loaded = json.loads(args.cloud_storage)
        except json.JSONDecodeError as exc:
            raise ValueError("cloud_storage must be valid JSON") from exc

        if not isinstance(loaded, dict):
            raise ValueError("cloud_storage must decode to an object/dict")
        cloud_storage = loaded

    return PipelineConfig(
        output_dir=args.output_dir,
        langs=langs,
        max_per_dataset=args.max_per_dataset,
        metadata_file=args.metadata_file,
        seed=args.seed,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        quality_config=quality_config,
        pii_config=pii_config,
        dedup_config=dedup_config,
        memory_config=memory_config,
        enable_progress_bar=True,
        enable_checkpointing=args.enable_checkpointing,
        checkpoint_interval=args.checkpoint_interval,
        enable_commercial_use=args.enable_commercial_use,
        enable_modification=args.enable_modification,
        dashboard_port=args.dashboard_port,
        log_level=args.log_level,
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download,
        resume_from_checkpoint=args.resume,
        enable_validation=args.enable_validation,
        validation_split=args.validation_split,
        validation_max_size=args.validation_max_size,
        verbose=args.verbose,
        unsloth_export_name=args.unsloth_export_name,
        cloud_storage=cloud_storage,
        distributed_mode=args.distributed_mode,
        node_rank=args.node_rank,
        world_size=args.world_size,
        health_check_interval=args.health_check_interval,
        max_runtime_hours=args.max_runtime_hours,
        auto_scaling=not args.disable_auto_scaling,
    )


def main() -> int:
    """Main entry point."""
    try:
        # Parse arguments
        config = parse_args()

        # Reconfigure logging based on arguments
        global logger
        logger = setup_logging(
            log_file=os.path.join(config.output_dir, "pipeline.log"),
            verbose=config.log_level == "DEBUG" or config.verbose,
        )

        # Initialize and run pipeline
        pipeline = DatasetPipeline(config)
        pipeline.run()

        logger.info("Pipeline executed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130  # Standard exit code for KeyboardInterrupt

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
