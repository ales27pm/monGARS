# Next Implementation Priority

## Summary
The SDK release closed the final open item for Phase 5, shifting the immediate
focus to Retrieval-Augmented Generation (RAG) governance. We must codify dataset
retention policies, automate scrubbing checks, and document operator workflows
before expanding partner access to curated artefacts.

## Supporting Signals from Existing Documentation
- **Implementation Status Report** – Calls out RAG governance as the remaining
  contradiction in Phase 5 now that SDK packaging is complete.【F:docs/implementation_status.md†L96-L140】
- **Codebase Status Report** – Highlights dataset governance as a key risk and
  recommends prioritising controls before scaling partner integrations.【F:docs/codebase_status_report.md†L40-L120】
- **RAG Context Enrichment Guide** – Notes that curated datasets demand explicit
  retention policies to protect sensitive context.【F:docs/rag_context_enrichment.md†L60-L96】

## Rationale for Prioritising RAG Governance
1. **Compliance Readiness** – Clearly documented retention windows and scrubbing
   automation ensure the enrichment datasets align with privacy and contractual
   requirements.
2. **Operational Safety** – Guardrails around export workflows prevent the
   accidental release of sensitive content as more teams rely on RAG-sourced
   suggestions.
3. **Ecosystem Trust** – Partners need assurance that enrichment data is curated
   and auditable before embedding it into their own workflows.

## Implementation Outline
1. Catalogue existing datasets under `models/datasets/curated/` and define
   metadata fields (provenance, expiry, sensitivity tags).
2. Extend the enrichment service or background jobs to validate datasets against
   the new policy (expiry checks, automated redaction, audit trails).
3. Document operator procedures for onboarding new repositories, handling
   takedown requests, and exporting artefacts safely.
4. Integrate reporting into the telemetry pipeline so governance metrics surface
   alongside existing peer and RAG monitoring.

## Success Criteria & Validation
- Every curated dataset includes metadata covering provenance, expiry, and
  review history.
- Automated jobs flag or quarantine datasets that fall out of compliance, and
  alerts surface through existing observability channels.
- Updated runbooks enable operators and partners to follow the governance flow
  without relying on ad-hoc institutional knowledge.

## Follow-On Work Once Governance Lands
- Resume reinforcement-learning integration workstreams for Phase 6.
- Revisit SDK telemetry to capture partner usage patterns in line with the new
  governance metrics.
