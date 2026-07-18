# MonGARSCoreML

This local Swift package runs the pinned, stateful INT4 conversion of Dolphin
3.0 Llama 3.2 3B on a physical iPhone. The model is downloaded after app
installation and is never bundled in the application binary.

## Runtime contract

- iOS 18 or newer on a physical device
- `ales27pm/Dolphin3.0-CoreML` at revision
  `95671cf9a2f56d2a381816ae264cd9aae335d96f`
- artifact `Dolphin3.0-Llama3.2-3B-stateful-int4.mlpackage`
- source `dphn/Dolphin3.0-Llama3.2-3B` at revision
  `392a6f57223e7ccfe6ef4ebdb2ff101a42d57364`
- 2,048-token state capacity and prefill chunks of at most 512 tokens
- `inputIds` and dynamic `causalMask` inputs
- `keyCache` and `valueCache` FP16 states with shape
  `[28, 1, 8, 2048, 128]`
- one FP16 `logits` output over the 128,258-token vocabulary

The downloader requests only the files listed in `ModelManifest.swift`, pins
the Hub revision, and verifies every file's exact size and SHA-256 digest. The
verified download is 1,825,812,981 bytes. Core ML compiles the source package
on the device and caches the derived `mlmodelc`; preparation requires about
5 GB of free space for the source, compiled model, and compiler scratch data.

Both the downloaded repository and the derived compilation cache live under
Application Support. Backup exclusion is applied recursively and any failure
to enforce it makes preparation fail. The compiled cache is discarded whenever
the verified source revision or manifest changes, or if Core ML cannot load it.

## Provenance and licensing

The conversion and its reproducible export/validation reports are maintained
in the `ales27pm/Dolphin3.0-CoreML` Hugging Face repository. The source model and
conversion are governed by the Llama 3.2 Community License and acceptable-use
policy. Retain the required "Built with Llama" attribution and review the
repository's `LICENSE`, `NOTICE`, and `USE_POLICY.md` before distribution.

- https://huggingface.co/ales27pm/Dolphin3.0-CoreML
- https://huggingface.co/dphn/Dolphin3.0-Llama3.2-3B
- https://github.com/huggingface/swift-transformers

## Device verification

The simulator intentionally reports this backend as unavailable. Validate a
release build on each supported iPhone class, including download interruption,
first compilation, cold and warm load, multi-chunk prefill, multi-turn
generation, cancellation, backgrounding, memory pressure, low-power mode, and
sustained thermal load. The Hub release reports validate the package schema and
Apple compiler, but do not replace generation and performance tests on an
actual iPhone.
