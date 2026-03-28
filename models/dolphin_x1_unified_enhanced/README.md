# Dolphin-X1 Unified Bundle Placeholder

This directory is intentionally tracked as a scaffold only. It satisfies the
default repository layout, but it is not a runnable model bundle.

Replace this scaffold with a real exported Dolphin-X1 unified bundle before
starting local inference. After copying or training the real artefacts, remove
`bundle.placeholder.json`.

Expected top-level artefacts usually include:

- `chat_lora/`
- `merged_fp16/`
- `wrapper/`

Use one of the existing training/export flows to populate this directory, such
as `python build_unified_dolphin_x1_enhanced.py --train` or the repo's
Unsloth/LLM2Vec pipeline scripts.
