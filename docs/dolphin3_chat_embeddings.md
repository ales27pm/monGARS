# Reusing Dolphin 3.0 for Chat and Embeddings

> **Last updated:** 2025-10-15 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The Dolphin 3.0 (Llama-3.1-8B) checkpoint already powers monGARS chat flows via
Ollama. This guide explains how to reuse the very same weights for retrieval
embeddings so the assistant and the vector index stay aligned.

## Single Checkpoint Strategy
- `dolphin_llm2vec_pipeline.py` loads the base model in 4-bit, attaches LoRA
  adapters, fine-tunes, and finally persists both the adapters and an
  LLM2Vec-compatible wrapper configuration. The wrapper metadata now records the
  Hugging Face backend and deterministic embedding options (mean pooling,
  `do_sample=false`, `top_p=1.0`, capped length) so downstream services can
  reuse the configuration without guessing defaults.【F:dolphin_llm2vec_pipeline.py†L40-L126】【F:dolphin_llm2vec_pipeline.py†L118-L140】
- The `LLM2Vec` smoke test performed inside the pipeline confirms that pooling
  the final hidden layer is viable for embeddings immediately after training.
  No additional adapter conversion is required before serving.【F:dolphin_llm2vec_pipeline.py†L142-L177】

## Chat Serving via Ollama
- The default profile binds the `general` role to the `dolphin3` Ollama model so
  conversational traffic continues to flow through the dedicated chat runtime.
  This path preserves streaming responses and sampling controls optimised for
  dialogue.【F:configs/llm_models.json†L4-L20】

## Embeddings via Hugging Face (LLM2Vec)
- The same pipeline exports a lightweight wrapper that can reload the base
  checkpoint plus LoRA adapters through Hugging Face Transformers and produce
  embeddings with mean pooling over the last hidden state. The embedded config
  documents the backend and deterministic parameters for consumers that expect a
  sentence-vector API.【F:scripts/export_llm2vec_wrapper.py†L1-L229】
- Under the hood, the wrapper tokenises batches, requests hidden states, and
  averages them using the attention mask so padding tokens are ignored. This
  keeps the embeddings consistent with the chat model’s tokenisation and avoids
  stochasticity during vector extraction.【F:scripts/export_llm2vec_wrapper.py†L57-L168】
- When running training locally, the pipeline relies on
  `load_4bit_causal_lm(...)` to fit Dolphin 3.0 within commodity GPUs by using
  NF4 quantisation, device maps, and offloading heuristics. This means the same
  model directory can be mounted by both the chat service and the embedding
  wrapper without duplicating checkpoints.【F:monGARS/mlops/model.py†L41-L120】

### Wrapper Manifest & Metadata
- The training pipeline now captures tokenizer details, chat defaults, embedding
  pooling behaviour, and artifact layout in `wrapper_config.json`. The metadata
  includes max sequence length, deterministic pooling, dtype hints, and
  references to the tokenizer and merged FP16 directories so downstream systems
  can reload the checkpoint without reverse engineering the training run.【F:dolphin_llm2vec_pipeline.py†L88-L139】
- `scripts/export_llm2vec_wrapper.py` deep-merges the training metadata with a
  wrapper manifest (versioned, timestamped, and annotated with the export
  directory) before emitting `wrapper/config.json`. Consumers can override the
  base model identifier (for example, if they renamed the checkpoint in a model
  registry) by passing `--base-model` to the exporter; the script persists the
  override for future loads.【F:scripts/export_llm2vec_wrapper.py†L1-L229】
- The generated `llm2vec_wrapper.py` respects the manifest at runtime. It uses
  the metadata to decide whether to load merged FP16 weights or attach LoRA
  adapters, applies the configured max length during tokenisation, and exposes a
  `normalise` toggle that defaults to the manifest’s deterministic setting. That
  keeps embedding requests aligned with the same tokenizer configuration used in
  chat.【F:scripts/export_llm2vec_wrapper.py†L57-L168】

### Serving Embeddings via FastAPI
- `scripts/run_llm2vec_service.py` provides a FastAPI façade that imports the
  generated wrapper, lazily initialises the model, and exposes `/embed` plus a
  `/healthz` endpoint. The CLI toggles merged-weight loading, device targeting,
  and 4-bit preferences, making it simple to stand up an embedding sidecar next
  to the Ollama chat runtime.【F:scripts/run_llm2vec_service.py†L1-L204】
- The service returns vectors as JSON (with count, dimensionality, backend, and
  normalisation flags) so a retrieval pipeline can stream results directly into
  a vector index. `--prefer-merged` lets you reuse the FP16 snapshot exported by
  the training pipeline, while `--force-4bit` keeps VRAM budgets low when GGUF
  export is unnecessary.【F:scripts/run_llm2vec_service.py†L33-L204】
- Example launch when running on the same host as Ollama:
  ```bash
  python scripts/run_llm2vec_service.py \
    --model-dir outputs_dolphin8b \
    --host 0.0.0.0 --port 8081 \
    --prefer-merged --log-level info
  ```
  ```bash
  curl -X POST http://localhost:8081/embed \
    -H 'content-type: application/json' \
    -d '{"inputs": ["How do we share weights?"], "normalise": true}'
  ```
  The same Dolphin 3.0 weights now power both chat (via Ollama) and deterministic
  embeddings (via FastAPI/Hugging Face) without duplicating checkpoints.

## Optional llama.cpp / GGUF Export
- If you need a lighter-weight embedding daemon, call the pipeline with
  `EXPORT_GGUF=1`. The exporter converts the merged weights into a GGUF file,
  which llama.cpp can serve in `--embeddings` mode. This gives you the same
  vector semantics while using llama.cpp’s optimised runtime.【F:dolphin_llm2vec_pipeline.py†L99-L116】【F:monGARS/mlops/exporters.py†L28-L74】

## Operational Notes
- Keep sampling parameters (temperature, nucleus sampling) for chat requests via
  Ollama. Embedding callers should honour the deterministic options stored in
  the wrapper metadata to avoid divergence between index updates and retrieval
  queries.【F:scripts/export_llm2vec_wrapper.py†L122-L149】
- Because both chat and embedding services consume the same tokenizer, document
  updates only once and validate them in the pipeline smoke tests before
  redeploying the wrapper.【F:dolphin_llm2vec_pipeline.py†L142-L177】
