from os import PathLike
from pathlib import Path
from typing import Union

from peft import PeftModel
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2RMSNorm,
    Qwen2SdpaAttention,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ModifiedQwen2Attention(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2FlashAttention2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedQwen2SdpaAttention(Qwen2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


QWEN2_ATTENTION_CLASSES = {
    "eager": ModifiedQwen2Attention,
    "flash_attention_2": ModifiedQwen2FlashAttention2,
    "sdpa": ModifiedQwen2SdpaAttention,
}


class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        """Initialise a decoder layer that disables causal masking."""

        super().__init__(config, layer_idx)

        try:
            attention_cls = QWEN2_ATTENTION_CLASSES[config._attn_implementation]
        except KeyError as exc:  # pragma: no cover - defensive guard
            supported = ", ".join(sorted(QWEN2_ATTENTION_CLASSES))
            raise ValueError(
                f"Unsupported attention implementation: {config._attn_implementation!r}. "
                f"Supported implementations are: {supported}."
            ) from exc

        self.self_attn = attention_cls(config=config, layer_idx=layer_idx)


class Qwen2BiModel(Qwen2Model):
    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2BiForMNTP(Qwen2ForCausalLM):
    def __init__(self, config):
        """Initialise the bidirectional MNTP head around the Qwen2 backbone."""

        Qwen2PreTrainedModel.__init__(self, config)

        if not isinstance(
            config, Qwen2Config
        ):  # pragma: no cover - defensive type check
            raise TypeError("config must be an instance of Qwen2Config")

        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self) -> nn.Module:
        """Return the model instance that adapters should wrap."""

        if getattr(self, "model", None) is None:
            raise RuntimeError("PEFT model has not been initialised")
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel) -> None:
        """Attach a PEFT-adapted model, validating it exposes the expected API."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module instance")
        save_fn = getattr(model, "save_pretrained", None)
        if not callable(save_fn):
            raise TypeError("model must expose a callable save_pretrained method")

        self.model = model

    # save the PEFT model
    def save_peft_model(self, path: Union[str, PathLike[str]]) -> None:
        """Persist the currently attached PEFT model to ``path``."""

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.get_model_for_peft()
        save_fn = getattr(model, "save_pretrained", None)
        if not callable(save_fn):
            raise RuntimeError("Attached model does not support save_pretrained")

        save_fn(str(target_path))
