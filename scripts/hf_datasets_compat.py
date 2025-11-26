# scripts/hf_datasets_compat.py
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_hf_datasets_for_list_feature() -> None:
    """
    Make datasets 3.x understand feature type 'List' used by datasets>=4.

    - On datasets 3.x: registers 'List' as an alias of Sequence in _FEATURE_TYPES.
    - On datasets 4.x: 'List' already exists, so this becomes a no-op.

    Must be called BEFORE the first `datasets.load_dataset` in the process.
    """
    try:
        import datasets  # noqa: F401
    except Exception as e:
        logger.warning(
            "huggingface-datasets not installed; skipping List->Sequence compat patch: %s",
            e,
        )
        return

    try:
        # Old versions use Sequence for list-like features.
        from datasets.features import Sequence

        # Different layouts between versions; try both.
        try:
            from datasets.features import features as features_module  # type: ignore[attr-defined]
        except Exception:  # datasets<=3.6 fallback
            import datasets.features.features as features_module  # type: ignore[import]

        mapping = getattr(features_module, "_FEATURE_TYPES", None)
        if not isinstance(mapping, dict):
            logger.warning(
                "HF datasets compat: _FEATURE_TYPES not found; cannot register 'List' feature."
            )
            return

        if "List" not in mapping:
            mapping["List"] = Sequence
            logger.info(
                "HF datasets compat: registered 'List' feature as alias of 'Sequence' "
                "for datasets<=3.6.0."
            )
        else:
            # On 4.x+ 'List' is already present; nothing to do.
            logger.debug("HF datasets compat: 'List' feature already registered; no-op.")

    except Exception as e:
        logger.exception(
            "HF datasets compat: failed to patch 'List' feature into datasets: %s", e
        )
