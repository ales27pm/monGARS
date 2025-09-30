# monGARS API Catalogue

This directory contains the generated OpenAPI specification and example client
implementations for the FastAPI surface exposed by `monGARS.api.web_api`.

## Contents
- [`openapi.json`](openapi.json) – canonical machine-readable description of
  every route, request/response model, and authentication requirement.
- [`clients/`](clients) – typed client examples for Python and TypeScript.

## Regenerating the Specification
Ensure `SECRET_KEY` is set (a throwaway value is fine) so the FastAPI app can
boot while generating the schema.

```bash
SECRET_KEY="dev-secret" python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

from monGARS.api.web_api import app

spec = app.openapi()
output_path = Path("docs/api/openapi.json")
output_path.write_text(json.dumps(spec, indent=2))
print(f"wrote {output_path}")
PY
```

Commit the updated `openapi.json` alongside any API change; downstream client
code depends on it.

## Client Guidance
- [Python example](clients/python.md) – async `httpx` client backed by Pydantic
  models with optional SDK generation instructions.
- [TypeScript example](clients/typescript.md) – `openapi-typescript-codegen`
  workflow for Node.js or React Native consumers.

Keep examples updated when endpoints, schemas, or authentication flows evolve.
