# monGARS OpenAPI Catalogue

This directory contains generated artefacts and client guides for the public
FastAPI surface exposed by `monGARS.api.web_api`. The specification is updated
from the running application to ensure feature parity with the deployed
behaviour.

## Specification Files

- [`openapi.json`](openapi.json) – canonical machine-readable description of
  every route, request/response model, and authentication requirement.

## Regenerating the Specification

The schema depends on a configured `SECRET_KEY`. When regenerating, provide a
throwaway value via the environment so the FastAPI application can boot:

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

Commit the updated `openapi.json` alongside any API changes. Downstream client
generators (see below) rely on the canonical file.

## Example Client Libraries

Sample integrations for popular stacks are documented under
[`clients/`](clients). Each client example includes dependency instructions,
authentication helpers, and typed request wrappers that mirror the generated
schema.
