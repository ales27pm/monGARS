# monGARS SDK Release Guide

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This guide documents the end-to-end process for publishing the Python and
TypeScript SDKs. Follow it whenever you cut a release so external consumers
receive reproducible packages that match the deployed FastAPI contract.

## Prerequisites
- Access to the PyPI account that owns `monGARS-sdk`.
- Access to the npm organisation that manages `@mongars/sdk`.
- Local toolchain with Python 3.11, Node.js 18+, `twine`, and npm 9+.
- Two-factor authentication enabled for both package registries.

## 1. Prepare the repository
1. Update the SDK versions:
   - `sdks/python/pyproject.toml`
   - `sdks/typescript/package.json`
2. Regenerate `package-lock.json` if dependencies changed:
   ```bash
   cd sdks/typescript
   npm install
   ```
3. Document notable changes in `docs/sdk-overview.md` and the language-specific
   READMEs.

## 2. Run validation
Execute the shared quality gates from the repository root:

```bash
python -m isort .
python -m black .
pytest -q
cd sdks/typescript && npm run lint
```

Resolve any failures before continuing.

## 3. Build release artefacts
Use the consolidated helper to create distributable packages for both SDKs:

```bash
python -m scripts.sdk_release --output dist/sdk
```

This command produces:
- Python wheels and source distributions in `dist/sdk/python/`
- An npm tarball in `dist/sdk/typescript/`

Run `twine check dist/sdk/python/*` to verify Python metadata before uploading.

## 4. Smoke test the reference clients
- Python CLI: `python sdks/python/examples/chat_cli.py`
- TypeScript example: `node sdks/typescript/dist/examples/chat.js` (after running `npm run build`)

Both should authenticate against a staging environment and complete a chat
exchange.

## 5. Publish the packages
1. Python (replace the file globs with the actual filenames):
   ```bash
   twine upload dist/sdk/python/*
   ```
2. TypeScript:
   ```bash
   npm publish dist/sdk/typescript/*.tgz
   ```

## 6. Tag the release
Create a signed tag and push it alongside the changelog update:

```bash
git tag -s sdk-v<version> -m "SDK release <version>"
git push origin sdk-v<version>
```

Finally, announce the release in the chosen communications channel and notify
partners that updated packages are available.
