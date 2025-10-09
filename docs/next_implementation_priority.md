# Next Implementation Priority

## Summary

Reinforcement validation is automated and the sustainability dashboards now
receive energy tracker metrics and observability snapshots via the
`SustainabilityDashboardBridge`. The next priority is to convert that unified
signal into actionable guidance—carbon-aware rollout policies, cross-node
artefact reuse, and hardware recommendations that keep consumption within
targets while scaling deployments.【F:monGARS/core/sustainability_dashboard.py†L1-L260】【F:docs/implementation_status.md†L130-L160】

## Supporting Signals

- **Implementation Status Report** – sustainability dashboards now publish
  `llm.sustainability.*` metrics, freeing the roadmap to focus on actionable
  policies.【F:docs/implementation_status.md†L130-L160】
- **Codebase Status Report** – highlights carbon-aware rollouts and cross-node
  reuse as the next evolution now that the telemetry feed is unified.【F:docs/codebase_status_report.md†L169-L214】
- **Energy Tracker & Sustainability Bridge** – persist per-cycle energy data and
  reinforcement summaries for dashboards and downstream automation hooks.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/sustainability_dashboard.py†L1-L260】

## Rationale for Prioritising Sustainability Telemetry

1. **Operational Stewardship** – grounding approvals in consumption data ensures
   reinforcement experiments align with sustainability goals.
2. **Capacity Planning** – dashboards that unify reward curves and energy trends
   let operators schedule GPU-intensive work during favourable energy windows.
3. **Partner Readiness** – downstream teams need actionable sustainability
   metrics before enabling reinforcement-driven features in production.

## Implementation Outline

1. **Carbon-Aware Rollouts** – build policies that read the sustainability
   dashboard feed and adjust rollout timing, replica placement, or approval
   thresholds based on emissions intensity.【F:monGARS/core/sustainability_dashboard.py†L1-L260】【F:modules/evolution_engine/orchestrator.py†L1-L160】
2. **Cross-Node Artefact Reuse** – quantify adapter reuse and amortised energy
   savings using the per-cycle energy history, feeding optimisation heuristics
   in the evolution engine.【F:monGARS/core/long_haul_validation.py†L120-L520】【F:modules/evolution_engine/orchestrator.py†L1-L160】
3. **Operator Playbooks** – extend the sustainability runbooks with new
   automation hooks, carbon thresholds, and remediation flows powered by the
   dashboard feed.【F:docs/reinforcement_rollout_runbook.md†L1-L160】【F:docs/codebase_status_report.md†L169-L214】

## Success Criteria & Validation

- Carbon-aware rollout policies adjust deployment timing or replica mix based on
  energy intensity and approval backlog surfaced in the dashboards.【F:modules/evolution_engine/orchestrator.py†L1-L160】【F:monGARS/core/sustainability_dashboard.py†L1-L260】
- Dashboards expose energy savings from artefact reuse, guiding operators toward
  efficient experimentation pathways.【F:monGARS/core/long_haul_validation.py†L120-L520】【F:docs/codebase_status_report.md†L169-L214】
- Updated runbooks walk on-call staff through interpreting sustainability
  metrics, carbon thresholds, and remediation actions without specialist
  knowledge.【F:docs/reinforcement_rollout_runbook.md†L1-L160】

## Follow-On Work Once Sustainability Dashboards Ship

- Iterate on energy-efficiency experiments (carbon-aware scheduling,
  replica bin-packing) once rollout policies have baseline data to act
  upon.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/sustainability_dashboard.py†L1-L260】
- Expand partner telemetry to correlate reinforcement-driven responses with
  engagement metrics leveraging the sustainability dashboards and policy
  outputs.【F:docs/codebase_status_report.md†L169-L214】
