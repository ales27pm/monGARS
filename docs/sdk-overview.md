# monGARS SDK Overview

> **Last updated:** 2025-10-06 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The monGARS platform now ships with first-party SDKs for Python and TypeScript.
These libraries wrap the public FastAPI surface with strong typing, resilient
error handling, and ergonomic helpers so integrators can focus on building
creative experiences.

## Packages

| Language   | Location               | Highlights                                           |
| ---------- | ---------------------- | ---------------------------------------------------- |
| Python     | `sdks/python`          | Sync + async clients using `httpx`, typed models, CLI examples |
| TypeScript | `sdks/typescript`      | Promise-based client with ESM output and fetch polyfill |

## Supported features

Both SDKs cover the authentication, conversation, review, UI, peer, and model
management endpoints exposed by the FastAPI service. All requests return typed
responses and raise structured exceptions when the API indicates an error.

## Reference clients

- `sdks/python/examples/chat_cli.py`: interactive terminal chat.
- `sdks/python/examples/peer_metrics.py`: periodic peer telemetry publisher.
- `sdks/typescript/examples/chat.ts`: minimal Node.js chat script.

## Versioning

The SDKs follow semantic versioning. Each release is tagged in the repository
and published to the respective package registry.

## Contributing

1. Update or extend the SDK functionality.
2. Add or adjust tests in `tests/test_python_sdk.py` or the TypeScript example
   suite.
3. Run the shared formatting and linting commands (`black`, `isort`, `npm run lint`).
4. Document new capabilities in this file or the language-specific READMEs.

## Release packaging

- Build artefacts with `python -m scripts.sdk_release --output dist/sdk`.
- Follow the detailed checklist in
  [`docs/sdk-release-guide.md`](sdk-release-guide.md) before publishing to PyPI
  and npm.
