import importlib.metadata

from packaging import version
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available

logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.1"
    )


def resolve_attention_classes(
    modeling_module,
    *,
    attention_name: str,
    flash_attention_name: str,
    sdpa_attention_name: str,
    model_label: str,
):
    """Resolve attention classes across transformer releases.

    Some recent Transformers builds stop exporting specialised FlashAttention2
    and SDPA subclasses while keeping the eager attention class available. The
    bidirectional llm2vec adapters only require a non-causal attention module,
    so they can safely fall back to the eager class when those symbols are
    absent.
    """

    attention_cls = getattr(modeling_module, attention_name)
    flash_cls = getattr(modeling_module, flash_attention_name, attention_cls)
    sdpa_cls = getattr(modeling_module, sdpa_attention_name, attention_cls)

    missing = []
    if not hasattr(modeling_module, flash_attention_name):
        missing.append(flash_attention_name)
    if not hasattr(modeling_module, sdpa_attention_name):
        missing.append(sdpa_attention_name)

    if missing:
        logger.info(
            "%s attention compatibility fallback enabled",
            model_label,
            extra={"missing": ",".join(missing)},
        )

    return attention_cls, flash_cls, sdpa_cls
