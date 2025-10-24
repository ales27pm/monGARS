# monGARS Documentation Hub

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

The monGARS docs collection is organised so that contributors, operators, and
researchers can land on the right guidance without paging through unrelated
runbooks. This hub groups the existing notes, highlights their intended
audiences, and calls out the live sources of truth inside the repository.

## How to use this hub
- **New contributors** should start with the project [README](../README.md) for a
  high-level tour, then dive into the Developer Essentials listed below.
- **Operators** should focus on the Deployment & Operations section and the
  security guidance embedded in each linked runbook.
- **Research teams** should reference the Research & ML track for adapter
  lifecycle management, dataset governance, and reinforcement loops.
- When a document has a "Last updated" banner, run
  `python scripts/update_docs_metadata.py` as part of your change. This keeps the
  docs dynamic and prevents stale operational advice by reading actual Git
  history.

## Architecture & Core Services

| Reference | Purpose |
| --- | --- |
| [architecture/module_interactions.md](architecture/module_interactions.md) | Diagrams and component-level walkthroughs for the FastAPI, cognition, and UI layers. |
| [conversation_workflow.md](conversation_workflow.md) | End-to-end chat pipeline tracing OAuth, memory enrichment, adaptive response synthesis, and WebSocket streaming. |
| [model_management.md](model_management.md) | Manifest structure, model lifecycle, and adapter provisioning hooks in `monGARS/core/model_manager.py`. |
| [repo_memory_alignment.md](repo_memory_alignment.md) | Mapping between repository modules and persisted state in Hippocampus/SQLModel. |

## Developer Essentials

| Reference | Why it matters |
| --- | --- |
| [testing.md](testing.md) | Complete test matrix, coverage thresholds, and targeted command recipes. |
| [workflow_reference.md](workflow_reference.md) | GitHub Actions fan-out, quality gates, and artefact inventory automation. |
| [code_audit_summary.md](code_audit_summary.md) | Security-focused audit outcomes with remediation cross-links. |
| [implementation_status.md](implementation_status.md) | Roadmap alignment across shipped and planned phases. |
| [advanced_fine_tuning.md](advanced_fine_tuning.md) | Strategic backlog for MNTP training hardening and distributed inference. |
| [documentation_maintenance.md](documentation_maintenance.md) | Style rules, review hooks, and "Last updated" expectations that keep guidance dynamic. |

## Deployment & Operations

| Runbook | Highlights |
| --- | --- |
| [deployment_automation.md](deployment_automation.md) | Guided visual installer, command switches, and hand-off into manual workflows. |
| [deployment_simulation.md](deployment_simulation.md) | Fault-injection playbook for testing idle optimisation and recovery routines. |
| [ray_serve_deployment.md](ray_serve_deployment.md) | Ray Serve cluster preparation, adapter refresh hooks, and telemetry integration. |
| [workflow_reference.md](workflow_reference.md) | CI/CD entry points, inventory checks, and container smoke-test coverage. |
| [security guidance in README](../README.md#security--observability) | Baseline operational hardening steps for secrets, WebSockets, and telemetry. |

## Research & ML Tooling

| Document | Scope |
| --- | --- |
| [rag_context_enrichment.md](rag_context_enrichment.md) | How RAG enrichment integrates with FastAPI review flows and fallback behaviour. |
| [rag_dataset_governance.md](rag_dataset_governance.md) | Dataset retention policies, scrubbing tools, and export controls. |
| [dolphin3_chat_embeddings.md](dolphin3_chat_embeddings.md) | Dolphin 3 stack reuse for chat and embeddings with deployment parameters. |
| [reinforcement_rollout_runbook.md](reinforcement_rollout_runbook.md) | Operator actions for the reinforcement loop, approvals, and soak testing. |
| [advanced_fine_tuning.md](advanced_fine_tuning.md) | (Also in Developer Essentials) — adapter training and Ray Serve automation backlog. |

## SDKs & Client Integrations

| Reference | Details |
| --- | --- |
| [sdk-overview.md](sdk-overview.md) | Supported languages, feature parity, and release cadence. |
| [sdk-release-guide.md](sdk-release-guide.md) | Versioning workflow, signing, and publication checklists. |
| [api/](api/README.md) | REST/WebSocket endpoint catalogue and request/response schemas. |

## Documentation Maintenance
- Track runbook edits in [documentation_maintenance.md](documentation_maintenance.md) so reviewers can see which checklists changed this release.
- Update diagrams in [`docs/images`](images/) when architecture or flows change; the README embeds these assets directly.
- Cross-link PRs to the relevant documentation section so reviewers verify the guidance moved with the code.
- Use [`scripts/manage_agents.py refresh`](../scripts/manage_agents.py) after updating scoped documentation rules in `configs/agents/agents_config.json`.
- Regenerate OpenAPI artefacts (`openapi.json`, `openapi.lock.json`) when FastAPI routes change so the API reference stays current with the service.

## Need something else?
If you cannot find an answer here, search the repository for an `AGENTS.md`
file scoped to the area you plan to modify. The nested charter explains the
expected style and testing obligations for that part of the tree.
