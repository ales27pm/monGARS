from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

_original_simplefilter = warnings.simplefilter


def _suppress_awq_simplefilter(
    action: str,
    category: type[Warning] | None = None,
    lineno: int = 0,
    append: bool = False,
) -> None:
    if action == "default" and category is DeprecationWarning:
        _original_simplefilter("ignore", category, lineno, append)
        return
    _original_simplefilter(action, category, lineno, append)


warnings.simplefilter = _suppress_awq_simplefilter
try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:  # pragma: no cover - dependency missing at import time
    raise RuntimeError(
        "peft is required for style fine-tuning. Install the 'peft' package."
    ) from exc
finally:
    warnings.simplefilter = _original_simplefilter


logger = logging.getLogger(__name__)


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*AutoAWQ is officially deprecated.*",
)

try:  # pragma: no cover - deterministic signature inspection
    _LORA_CONFIG_SUPPORTS_FAN_IN_OUT = (
        "fan_in_fan_out" in inspect.signature(LoraConfig.__init__).parameters
    )
except (TypeError, ValueError):  # pragma: no cover - defensive guard
    _LORA_CONFIG_SUPPORTS_FAN_IN_OUT = False


def _resolve_hidden_size(config: AutoConfig) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "n_embd"):
        return int(config.n_embd)
    raise AttributeError("Unable to determine hidden size from model configuration")


def _default_device() -> str:
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _fingerprint_interactions(interactions: Sequence[dict[str, str]]) -> str:
    payload = json.dumps(interactions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_requires_fan_in_fan_out(model: PreTrainedModel) -> bool:
    """Detect whether the underlying model relies on GPT-style Conv1D layers."""

    for module in model.modules():
        if module.__class__.__name__ == "Conv1D":
            return True
    return False


@dataclass
class StyleFineTuningConfig:
    base_model: str = os.getenv(
        "STYLE_BASE_MODEL", "hf-internal-testing/tiny-random-gpt2"
    )
    adapter_repository: Path = Path(
        os.getenv(
            "STYLE_ADAPTER_DIR", os.path.join(tempfile.gettempdir(), "mongars_style")
        )
    )
    max_history_messages: int = int(os.getenv("STYLE_MAX_HISTORY", 20))
    min_samples: int = int(os.getenv("STYLE_MIN_SAMPLES", 2))
    micro_batch_size: int = int(os.getenv("STYLE_MICRO_BATCH_SIZE", 1))
    training_epochs: float = float(os.getenv("STYLE_TRAIN_EPOCHS", 1.0))
    max_steps: int = int(os.getenv("STYLE_MAX_STEPS", 6))
    learning_rate: float = float(os.getenv("STYLE_LEARNING_RATE", 5e-4))
    lora_r: int = int(os.getenv("STYLE_LORA_R", 8))
    lora_alpha: int = int(os.getenv("STYLE_LORA_ALPHA", 16))
    lora_dropout: float = float(os.getenv("STYLE_LORA_DROPOUT", 0.05))
    use_qlora: bool = os.getenv("STYLE_USE_QLORA", "False").lower() in ("true", "1")
    max_sequence_length: int = int(os.getenv("STYLE_MAX_SEQUENCE_LENGTH", 256))
    temperature: float = float(os.getenv("STYLE_GENERATION_TEMPERATURE", 0.7))
    top_p: float = float(os.getenv("STYLE_GENERATION_TOP_P", 0.9))
    max_new_tokens: int = int(os.getenv("STYLE_MAX_NEW_TOKENS", 96))
    seed: int = int(os.getenv("STYLE_ANALYSIS_SEED", 7))
    max_concurrent_trainings: int = int(os.getenv("STYLE_MAX_CONCURRENT_TRAININGS", 2))
    adapter_cache_ttl_seconds: int = int(os.getenv("STYLE_ADAPTER_TTL", 3600))
    adapter_cache_maxsize: int = int(os.getenv("STYLE_ADAPTER_MAXSIZE", 64))


@dataclass
class StyleAnalysis:
    traits: dict[str, float]
    style: dict[str, float]
    context_preferences: dict[str, float]
    confidence: float
    sample_count: int

    @classmethod
    def default(cls, sample_count: int) -> "StyleAnalysis":
        return cls(
            traits={
                "openness": 0.55,
                "conscientiousness": 0.55,
                "extraversion": 0.55,
                "agreeableness": 0.55,
                "neuroticism": 0.45,
            },
            style={
                "formality": 0.5,
                "humor": 0.5,
                "enthusiasm": 0.5,
                "directness": 0.5,
            },
            context_preferences={
                "technical": 0.5,
                "casual": 0.5,
                "professional": 0.5,
            },
            confidence=0.2 if sample_count == 0 else min(0.4 + sample_count * 0.1, 0.9),
            sample_count=sample_count,
        )


class ConversationDataset(Dataset):
    """Minimal dataset for LoRA fine-tuning from conversation snippets."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        samples: Sequence[str],
        *,
        max_length: int,
    ) -> None:
        self._inputs = []
        for text in samples:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoded["labels"] = encoded["input_ids"].clone()
            self._inputs.append({k: v.squeeze(0) for k, v in encoded.items()})

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._inputs[index]


@dataclass
class StyleAdapterState:
    fingerprint: str
    model: PreTrainedModel | None
    tokenizer: PreTrainedTokenizerBase
    sample_count: int
    updated_at: float


class PromptBuilder:
    """Construct generation prompts from personality dimensions."""

    def __init__(self, style_keys: Sequence[str]) -> None:
        self._style_keys = list(style_keys)

    def build(self, base_text: str, personality: Mapping[str, float]) -> str:
        descriptors: list[str] = []
        for key in self._style_keys:
            value = personality.get(key)
            if value is None:
                continue
            if value > 0.66:
                descriptors.append(f"très {key}")
            elif value < 0.33:
                descriptors.append(f"peu {key}")
        tone = ", ".join(descriptors) if descriptors else "équilibré"
        return (
            "Tu es un assistant personnalisé. Ajuste la réponse suivante pour adopter "
            f"un ton {tone}. Réponse originale: {base_text}\nRéponse adaptée:"
        )


class StyleAdapterCache:
    """In-memory adapter cache with TTL and LRU eviction."""

    def __init__(self, *, ttl_seconds: int, maxsize: int) -> None:
        self._ttl_seconds = ttl_seconds if ttl_seconds > 0 else None
        self._maxsize = max(1, maxsize)
        self._entries: OrderedDict[str, StyleAdapterState] = OrderedDict()

    def get(self, user_id: str) -> StyleAdapterState | None:
        state = self._entries.get(user_id)
        if state is None:
            return None
        if self._ttl_seconds is not None:
            now = time.monotonic()
            if now - state.updated_at > self._ttl_seconds:
                self._entries.pop(user_id, None)
                return None
            state.updated_at = now
        self._entries.move_to_end(user_id)
        return state

    def set(self, user_id: str, state: StyleAdapterState) -> None:
        self._entries[user_id] = state
        self._entries.move_to_end(user_id)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)


class AdapterTrainer:
    """Train and materialise LoRA/QLoRA adapters for inference."""

    def __init__(
        self,
        config: StyleFineTuningConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer

    def train(
        self,
        user_id: str,
        fingerprint: str,
        samples: Sequence[str],
    ) -> StyleAdapterState:
        logger.info(
            "Training style adapter for %s with %s samples (fingerprint=%s)",
            user_id,
            len(samples),
            fingerprint,
        )
        training_model = self._create_model(trainable=True)
        lora_kwargs = {
            "r": self._config.lora_r,
            "lora_alpha": self._config.lora_alpha,
            "lora_dropout": self._config.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if _LORA_CONFIG_SUPPORTS_FAN_IN_OUT:
            lora_kwargs["fan_in_fan_out"] = _model_requires_fan_in_fan_out(
                training_model
            )
        lora_config = LoraConfig(**lora_kwargs)
        training_model = get_peft_model(training_model, lora_config)
        dataset = ConversationDataset(
            self._tokenizer,
            samples,
            max_length=self._config.max_sequence_length,
        )
        batch_size = min(self._config.micro_batch_size, max(1, len(dataset)))
        adapter_path = self._config.adapter_repository / user_id
        tmp_output = adapter_path / "trainer"
        training_args = TrainingArguments(
            output_dir=str(tmp_output),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self._config.learning_rate,
            num_train_epochs=self._config.training_epochs,
            max_steps=self._config.max_steps,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            optim="adamw_torch",
            fp16=False,
            dataloader_pin_memory=torch.cuda.is_available(),
        )
        trainer = Trainer(
            model=training_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )
        trainer.train()
        training_model.eval()
        adapter_path.mkdir(parents=True, exist_ok=True)
        training_model.save_pretrained(str(adapter_path))
        self._tokenizer.save_pretrained(str(adapter_path))

        inference_model = self._create_model(trainable=False)
        inference_adapter = PeftModel.from_pretrained(
            inference_model,
            str(adapter_path),
            is_trainable=False,
        )
        inference_adapter.to("cpu")
        inference_adapter.eval()

        return StyleAdapterState(
            fingerprint=fingerprint,
            model=inference_adapter,
            tokenizer=self._tokenizer,
            sample_count=len(samples),
            updated_at=time.monotonic(),
        )

    def _create_model(self, *, trainable: bool) -> PreTrainedModel:
        quantization_config = None
        if trainable and self._config.use_qlora:
            try:  # pragma: no cover - depends on optional bitsandbytes
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                logger.warning("Failed to enable QLoRA, falling back to LoRA: %s", exc)
                quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model,
            dtype=torch.float32,
            device_map=None,
            quantization_config=quantization_config,
        )
        if not trainable:
            model.eval()
        return model


class StyleInferenceService:
    """Run inference for style extraction and adaptation."""

    def __init__(
        self,
        config: StyleFineTuningConfig,
        *,
        device: str,
        trait_projection: torch.Tensor,
        style_projection: torch.Tensor,
        context_projection: torch.Tensor,
    ) -> None:
        self._config = config
        self._device = device
        self._trait_projection = trait_projection
        self._style_projection = style_projection
        self._context_projection = context_projection

    def extract_personality(
        self,
        state: StyleAdapterState,
        interactions: Sequence[dict[str, str]],
    ) -> StyleAnalysis:
        if state.model is None:
            return StyleAnalysis.default(sample_count=state.sample_count)

        cpu_device = torch.device("cpu")
        current_device = next(state.model.parameters()).device
        model = (
            state.model.to(cpu_device) if current_device != cpu_device else state.model
        )
        model.eval()
        vectors: list[torch.Tensor] = []
        for item in interactions[-self._config.max_history_messages :]:
            candidate = item.get("response") or item.get("message")
            if not candidate:
                continue
            tokens = state.tokenizer(
                candidate,
                return_tensors="pt",
                truncation=True,
                max_length=self._config.max_sequence_length,
            )
            with torch.no_grad():
                outputs = model(
                    **tokens,
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states[-1][:, -1, :]
            vectors.append(hidden_states.squeeze(0))

        if not vectors:
            return StyleAnalysis.default(sample_count=state.sample_count)

        stacked = torch.stack(vectors)
        mean_vector = stacked.mean(dim=0)
        trait_scores = torch.sigmoid(self._trait_projection @ mean_vector)
        style_scores = torch.sigmoid(self._style_projection @ mean_vector)
        context_scores = torch.sigmoid(self._context_projection @ mean_vector)

        traits = {
            key: float(trait_scores[idx].item())
            for idx, key in enumerate(StyleFineTuner.TRAIT_KEYS)
        }
        style = {
            key: float(style_scores[idx].item())
            for idx, key in enumerate(StyleFineTuner.STYLE_KEYS)
        }
        context_preferences = {
            key: float(context_scores[idx].item())
            for idx, key in enumerate(StyleFineTuner.CONTEXT_KEYS)
        }

        confidence = min(0.9, 0.4 + 0.1 * state.sample_count)
        return StyleAnalysis(
            traits=traits,
            style=style,
            context_preferences=context_preferences,
            confidence=confidence,
            sample_count=state.sample_count,
        )

    def apply_style(
        self,
        state: StyleAdapterState,
        prompt: str,
    ) -> str:
        if state.model is None:
            return prompt

        tokens = state.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_sequence_length,
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}
        model = state.model.to(self._device)
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **tokens,
                max_new_tokens=self._config.max_new_tokens,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                do_sample=True,
                pad_token_id=state.tokenizer.eos_token_id,
            )
        return state.tokenizer.decode(output[0], skip_special_tokens=True)


class StyleFineTuner:
    """Manage LoRA/QLoRA adapters to personalise responses per user."""

    TRAIT_KEYS = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    STYLE_KEYS = ["formality", "humor", "enthusiasm", "directness"]
    CONTEXT_KEYS = ["technical", "casual", "professional"]

    def __init__(
        self,
        config: StyleFineTuningConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        if config is None:
            try:
                from monGARS.config import get_settings

                settings = get_settings()
                config = StyleFineTuningConfig(
                    base_model=settings.style_base_model,
                    adapter_repository=Path(settings.style_adapter_dir),
                    max_history_messages=settings.style_max_history,
                    min_samples=settings.style_min_samples,
                    max_steps=settings.style_max_steps,
                    learning_rate=settings.style_learning_rate,
                    use_qlora=settings.style_use_qlora,
                    max_concurrent_trainings=settings.style_max_concurrent_trainings,
                    adapter_cache_ttl_seconds=settings.style_adapter_ttl_seconds,
                    adapter_cache_maxsize=settings.style_adapter_maxsize,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - settings unavailable during import
                logger.debug(
                    "Falling back to environment defaults for style tuning config: %s",
                    exc,
                )
                config = StyleFineTuningConfig()
        self.config = config
        self.device = device or _default_device()
        self.config.adapter_repository.mkdir(parents=True, exist_ok=True)
        self._base_config = AutoConfig.from_pretrained(self.config.base_model)
        self._hidden_size = _resolve_hidden_size(self._base_config)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        generator = torch.Generator().manual_seed(self.config.seed)
        self._trait_projection = torch.randn(
            len(self.TRAIT_KEYS), self._hidden_size, generator=generator
        )
        self._style_projection = torch.randn(
            len(self.STYLE_KEYS), self._hidden_size, generator=generator
        )
        self._context_projection = torch.randn(
            len(self.CONTEXT_KEYS), self._hidden_size, generator=generator
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self._adapter_cache = StyleAdapterCache(
            ttl_seconds=self.config.adapter_cache_ttl_seconds,
            maxsize=self.config.adapter_cache_maxsize,
        )
        self._training_semaphore = asyncio.Semaphore(
            max(1, self.config.max_concurrent_trainings)
        )
        self._prompt_builder = PromptBuilder(self.STYLE_KEYS)
        self._trainer = AdapterTrainer(self.config, self._tokenizer)
        self._inference = StyleInferenceService(
            self.config,
            device=self.device,
            trait_projection=self._trait_projection,
            style_projection=self._style_projection,
            context_projection=self._context_projection,
        )
        logger.info(
            "StyleFineTuner initialised (base_model=%s, device=%s)",
            self.config.base_model,
            self.device,
        )

    async def estimate_personality(
        self, user_id: str, interactions: Sequence[dict[str, str]]
    ) -> StyleAnalysis:
        """Train (if required) the LoRA adapter and extract trait estimates."""

        interactions = [
            {
                "message": item.get("message", ""),
                "response": item.get("response", ""),
            }
            for item in interactions
            if item.get("message") or item.get("response")
        ]
        fingerprint = _fingerprint_interactions(interactions)
        lock = self._locks.setdefault(user_id, asyncio.Lock())

        async with lock:
            state = self._adapter_cache.get(user_id)
            needs_training = (
                state is None or state.fingerprint != fingerprint or state.model is None
            )
            samples = self._build_training_corpus(interactions)

            if not samples:
                state = StyleAdapterState(
                    fingerprint=fingerprint,
                    model=None,
                    tokenizer=self._tokenizer,
                    sample_count=0,
                    updated_at=time.monotonic(),
                )
                self._adapter_cache.set(user_id, state)
                return StyleAnalysis.default(sample_count=0)

            if needs_training and len(samples) >= self.config.min_samples:
                async with self._training_semaphore:
                    state = await asyncio.to_thread(
                        self._trainer.train,
                        user_id,
                        fingerprint,
                        samples,
                    )
                self._adapter_cache.set(user_id, state)
            elif state is None:
                # Not enough samples to train yet.
                state = StyleAdapterState(
                    fingerprint=fingerprint,
                    model=None,
                    tokenizer=self._tokenizer,
                    sample_count=len(samples),
                    updated_at=time.monotonic(),
                )
                self._adapter_cache.set(user_id, state)

        if state.model is None:
            return StyleAnalysis.default(sample_count=state.sample_count)

        return await asyncio.to_thread(
            self._inference.extract_personality,
            state,
            interactions,
        )

    def apply_style(
        self,
        user_id: str,
        base_text: str,
        personality: dict[str, float] | None,
    ) -> str:
        state = self._adapter_cache.get(user_id)
        if state is None or state.model is None:
            logger.debug("No trained adapter for %s; returning base text", user_id)
            return base_text

        prompt = self._prompt_builder.build(base_text, personality or {})
        generated = self._inference.apply_style(state, prompt)
        if generated.startswith(prompt):
            adapted = generated[len(prompt) :].strip()
            return adapted or generated.strip()
        return generated.strip()

    def _build_training_corpus(
        self, interactions: Sequence[dict[str, str]]
    ) -> list[str]:
        samples: list[str] = []
        for item in interactions[-self.config.max_history_messages :]:
            message = item.get("message", "").strip()
            if response := item.get("response", "").strip():
                samples.append(response)
            elif message:
                samples.append(message)
        return samples
