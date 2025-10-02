# RAG Context Enrichment

## Overview

RAG context enrichment augments monGARS chat and review workflows with references
from external code repositories. When enabled (`rag_enabled=true`), FastAPI
routes call `monGARS.core.rag.context_enricher.RagContextEnricher` to retrieve
focus areas and code snippets that align with the current query.

## Prerequisites

- Available for deployments that can reach the configured enrichment service
  (`rag_service_url` or `DOC_RETRIEVAL_URL`).
- Requires a prepared database and indexed repositories on the enrichment side.
- Production deployments should ensure the enrichment service enforces
  authentication/authorization before exposing organisation code.

## Configuration

Set the following keys in `.env` (or Kubernetes secrets):

| Setting | Description |
| --- | --- |
| `rag_enabled` | Enable or disable repository enrichment. Defaults to `false`. |
| `rag_service_url` | Base URL for the enrichment service. Falls back to `DOC_RETRIEVAL_URL` when omitted. |
| `rag_repo_list` | Default repositories queried for enrichment. Use `all` to search every project exposed by the service. |
| `rag_max_results` | Maximum number of references requested per query. Values above this limit are clamped. |

The FastAPI endpoint normalises overrides per request and clamps `max_results`
against the configured limit.

## Applications

Enriched context is surfaced in the following flows:

- `/ask`, `/review`, `/implement`, and `/compliance` assistants within the
  operator console.
- `POST /api/v1/review/rag-context` for programmatic access.

### API Endpoint

`POST /api/v1/review/rag-context`

Request body:

- `query` (string, required): description of the change or ticket under review.
- `repositories` (array of strings, optional): overrides the configured repository
  list for this request. Use `all` to search every indexed project.
- `max_results` (integer, optional): requested number of references. Values above
  `rag_max_results` are clamped server-side.

Successful responses include:

```json
{
  "enabled": true,
  "focus_areas": ["Refactor cache invalidation"],
  "references": [
    {
      "repository": "acme/api",
      "file_path": "src/cache.py",
      "summary": "Ensure invalidation runs on background workers",
      "score": 0.92,
      "url": "https://git.example.com/acme/api/blob/main/src/cache.py"
    }
  ]
}
```

If RAG is disabled, the endpoint returns `{ "enabled": false, "focus_areas": [],
"references": [] }` with HTTP 200.

## Failure Modes & Fallbacks

- Missing or blank queries raise a validation error before reaching the external
  service.
- Network failures (`httpx.HTTPError`) or non-2xx responses raise
  `RagServiceError`, which surfaces as a structured error and logs the failing
  endpoint (`rag.context_enrichment.*`).
- Invalid payloads (e.g. non-JSON responses) degrade to an empty enrichment
  result so downstream consumers remain functional.

## Recommended Practices

- Keep repository indexing up to date to maintain high-quality results.
- Curate `rag_repo_list` to balance context coverage against retrieval noise.
- Document data-retention policies for the curated datasets stored in
  `models/datasets/curated/` to ensure privacy expectations are met when RAG is
  active.
