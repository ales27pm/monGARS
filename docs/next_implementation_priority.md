# Next Implementation Priority

> **Last updated:** 2025-10-24 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

## Summary

Reinforcement validation is automated and the sustainability dashboards now
receive energy tracker metrics and observability snapshots via the
`SustainabilityDashboardBridge`. Carbon-aware gating is live, so the next
priority is to operationalise the remaining signals—quantify cross-node
artefact reuse and surface hardware-aware placement guidance so the dashboards
drive tangible energy savings across deployments.【F:modules/evolution_engine/sustainability.py†L1-L235】【F:modules/evolution_engine/orchestrator.py†L260-L360】【F:docs/implementation_status.md†L130-L200】

## Supporting Signals

- **Implementation Status Report** – carbon-aware gating has shipped, leaving
  reuse analytics and placement guidance as the remaining sustainability
  deliverables.【F:docs/implementation_status.md†L130-L200】
- **Codebase Status Report** – highlights cross-node reuse and hardware-aware
  placement as the next evolution now that telemetry is unified.【F:docs/codebase_status_report.md†L169-L214】
- **Energy Tracker & Sustainability Bridge** – persist per-cycle energy data and
  reinforcement summaries for dashboards and downstream automation hooks, ready
  to support reuse analytics.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/sustainability_dashboard.py†L1-L260】

## Rationale for Prioritising Sustainability Telemetry

1. **Operational Stewardship** – grounding approvals in consumption data ensures
   reinforcement experiments align with sustainability goals.
2. **Capacity Planning** – dashboards that unify reward curves and energy trends
   let operators schedule GPU-intensive work during favourable energy windows.
3. **Partner Readiness** – downstream teams need actionable sustainability
   metrics before enabling reinforcement-driven features in production.

## Implementation Outline

1. **Cross-Node Artefact Reuse Metrics** – enrich the sustainability dashboard
   payload with reuse ratios and amortised energy savings so operators can
   prioritise adapters that reduce retraining cost.【F:monGARS/core/long_haul_validation.py†L520-L660】【F:monGARS/core/sustainability_dashboard.py†L200-L320】
2. **Hardware-Aware Placement Guidance** – translate sustainability telemetry
   into scheduler hints (time windows, replica mixes, GPU affinity) that feed the
   evolution orchestrator and distributed scheduler.【F:modules/evolution_engine/orchestrator.py†L260-L360】【F:monGARS/core/distributed_scheduler.py†L1-L220】
3. **Operator Playbooks** – extend sustainability runbooks with reuse analytics
   and placement recommendations so on-call staff can act on the richer
   telemetry without manual calculations.【F:docs/reinforcement_rollout_runbook.md†L1-L200】【F:docs/codebase_status_report.md†L169-L214】

## Success Criteria & Validation

- Dashboards expose energy savings from artefact reuse, guiding operators toward
  efficient experimentation pathways.【F:monGARS/core/long_haul_validation.py†L520-L660】【F:docs/codebase_status_report.md†L169-L214】
- Scheduler hints reflect carbon-aware placement recommendations derived from
  sustainability telemetry.【F:modules/evolution_engine/orchestrator.py†L260-L360】【F:monGARS/core/distributed_scheduler.py†L1-L220】
- Updated runbooks walk on-call staff through interpreting reuse metrics,
  placement hints, and remediation actions without specialist
  knowledge.【F:docs/reinforcement_rollout_runbook.md†L1-L200】

## Follow-On Work Once Sustainability Dashboards Ship

- Iterate on energy-efficiency experiments (carbon-aware bin-packing, GPU
  affinity policies) once reuse metrics surface baseline opportunities.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/sustainability_dashboard.py†L1-L260】
- Expand partner telemetry to correlate reinforcement-driven responses with
  engagement metrics leveraging the sustainability dashboards and policy
  outputs.【F:docs/codebase_status_report.md†L169-L214】
