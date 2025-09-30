# Encoder Artefact Registry

This directory stores trained LLM2Vec adapter weights and metadata consumed by
`monGARS.core.llm_integration` and the evolution engine.

## Layout
- `*/` – adapter directories named after the model family and version, e.g.
  `mistral-7b-v1/`.
- `adapter_manifest.json` – optional manifest produced by the evolution engine to
  advertise the latest adapter revision and checksum to inference services.
- `README.md` – this document.

## Usage Guidelines
1. Keep adapter directories self-contained: include the model weights, tokenizer
   files, and any supplementary config required to load the adapter.
2. Record provenance in a `METADATA.json` file inside each adapter directory
   (training dataset, hyperparameters, metrics). This ensures evolution cycles can
   be audited.
3. Do not commit large checkpoints unless they are quantised for distribution or
   explicitly requested by maintainers. Store original weights in object storage
   and document retrieval instructions.
4. Update `docs/advanced_fine_tuning.md` and `README.md` when you add or remove
   adapters so runtime operators know which artefacts are available.
