# MonGARSCoreML

This local Swift package runs a pinned, stateful Qwen3 1.7B Core ML model on a
physical iPhone. The model is downloaded after installation and is never
bundled in the app binary.

## Runtime contract

- iOS 18 or newer on a physical device
- `mariocde/Qwen3-1.7B-CoreML-LUT6` at revision
  `51c5bc038afa962216e3880bf870e92b219328e6`
- source model `Qwen/Qwen3-1.7B` at revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- 512-token context, with up to 192 new tokens per response
- 64-token stateful prefill and one-token stateful decode
- 16 segmented logits outputs for a 151,936-token vocabulary

The downloader requests only the files listed in `ModelManifest.swift`, pins
the Hub revision, and verifies the size and SHA-256 digest of the tokenizer and
compiled model payload before loading them. Files live under Application
Support and are excluded from iCloud backup.

## Provenance and licensing

The bootstrap Core ML artifact is maintained by a third party on Hugging Face.
The source Qwen3 model is published under Apache-2.0. Before distributing a
production build, review the artifact repository and source-model terms, retain
the required notices, and preferably publish a reproducible monGARS-owned Core
ML conversion.

- https://huggingface.co/mariocde/Qwen3-1.7B-CoreML-LUT6
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://github.com/huggingface/swift-transformers

## Device verification

The simulator intentionally reports this backend as unavailable. Validate a
release build on a recent iPhone, including interrupted downloads, first model
load, multi-turn generation, cancellation, backgrounding, memory pressure, low
power mode, and sustained thermal load.
