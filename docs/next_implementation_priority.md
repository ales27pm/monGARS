# Next Implementation Priority

## Summary

Reinforcement validation is now automated and observable; the next priority is
connecting the sustainability data exhaust—energy tracker metrics,
reinforcement observability snapshots, and hardware-aware rollouts—to shared
dashboards and alerts so operators can act on consumption trends in real
time.【F:docs/implementation_status.md†L130-L160】【F:docs/codebase_status_report.md†L169-L214】

## Supporting Signals

- **Implementation Status Report** – Phase 6 closed with observability feeds and
  now shifts focus to energy dashboards for the sustainability
  milestone.【F:docs/implementation_status.md†L130-L160】
- **Codebase Status Report** – lists sustainability dashboards as the top risk
  now that long-haul validation runs autonomously.【F:docs/codebase_status_report.md†L169-L214】
- **Energy Tracker & Observability Store** – already emit the consumption data
  that dashboards must surface for operators.【F:modules/evolution_engine/energy.py†L1-L160】【F:monGARS/core/reinforcement_observability.py†L1-L168】

## Rationale for Prioritising Sustainability Telemetry

1. **Operational Stewardship** – grounding approvals in consumption data ensures
   reinforcement experiments align with sustainability goals.
2. **Capacity Planning** – dashboards that unify reward curves and energy trends
   let operators schedule GPU-intensive work during favourable energy windows.
3. **Partner Readiness** – downstream teams need actionable sustainability
   metrics before enabling reinforcement-driven features in production.

## Implementation Outline

1. **Dashboards & Alerts** – surface reinforcement observability snapshots,
   energy tracker metrics, and replica analytics in shared dashboards with alert
   thresholds for regressions.【F:monGARS/core/reinforcement_observability.py†L1-L168】【F:modules/evolution_engine/energy.py†L1-L160】
2. **Hardware-Aware Rollouts** – correlate the dashboard signals with evolution
   engine scheduling so energy-heavy experiments favour efficient replicas.【F:modules/evolution_engine/orchestrator.py†L1-L160】
3. **Operator Runbooks** – extend the sustainability playbook with dashboard
   links, remediation flows, and escalation paths once metrics are live.【F:docs/reinforcement_rollout_runbook.md†L1-L160】

## Success Criteria & Validation

- Reinforcement and sustainability metrics co-exist in the shared dashboards
  alongside self-training events.
- Energy consumption trends trigger actionable alerts with documented
  remediation steps for operators.【F:docs/reinforcement_rollout_runbook.md†L1-L160】
- Updated documentation enables on-call staff to balance reinforcement outcomes
  with sustainability objectives without specialist knowledge.【F:docs/reinforcement_rollout_runbook.md†L1-L160】

## Follow-On Work Once Sustainability Dashboards Ship

- Resume advanced energy-efficiency experiments (carbon-aware scheduling,
  replica bin-packing) after dashboards surface baseline trends.【F:modules/evolution_engine/energy.py†L1-L160】
- Expand partner telemetry to correlate reinforcement-driven responses with
  engagement metrics leveraging the new sustainability dashboards.【F:docs/codebase_status_report.md†L169-L214】
