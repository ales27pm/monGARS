# Next Implementation Priority

## Summary

Reinforcement-learning (RL) runs now stream approvals, telemetry, and artefacts
through the orchestrator and long-haul validator. The next priority is
operationalising that data: shared dashboards and sustained multi-replica soak
tests need to land so operators can track RL output alongside self-training.
【F:docs/implementation_status.md†L96-L140】【F:docs/codebase_status_report.md†L104-L160】

## Supporting Signals

- **Implementation Status Report** – highlights the need for durable
  observability before Phase 6 can close.【F:docs/implementation_status.md†L124-L160】
- **Codebase Status Report** – flags long-haul observability as the primary
  remaining risk and calls for multi-replica soak coverage.【F:docs/codebase_status_report.md†L96-L160】
- **Evolution Engine Orchestrator & Validator** – already emit approvals,
  rewards, and energy metrics, providing the data feeds dashboards must
  visualise.【F:modules/evolution_engine/orchestrator.py†L360-L440】【F:monGARS/core/long_haul_validation.py†L156-L226】

## Rationale for Prioritising RL Operationalisation

1. **Experiment Velocity** – moving RL into the automated cycle reduces the
   manual work currently required to evaluate policy improvements.
2. **Safety & Observability** – explicit telemetry and guardrails prevent
   regressions when exploring novel reward models.
3. **Partner Readiness** – RL-backed suggestions unlock new product
   capabilities, but only once the rollout and rollback story matches the rest
   of the platform.

## Implementation Outline

1. **Dashboards & Alerts** – wire the long-haul metrics (reward curves, approval
   counts, energy usage) into shared dashboards with alert thresholds for
   regressions.【F:monGARS/core/long_haul_validation.py†L156-L226】
2. **Soak & Multi-Replica Tests** – extend the long-haul integration suite to
   exercise concurrent RL and MNTP runs with multiple Ray replicas to flush out
   scheduler and manifest contention.【F:tests/test_long_haul_validation.py†L1-L220】
3. **Operator Runbooks** – update the reinforcement rollout guide with the new
   dashboard links and alert handling procedures so on-call staff can respond
   quickly.【F:docs/reinforcement_rollout_runbook.md†L1-L160】

## Success Criteria & Validation

- RL training cycles emit metrics and logs that surface in the same dashboards
  as self-training events.
- Deployment of RL artefacts requires an explicit approval or automated policy
  pass, with clear rollback instructions surfaced alongside dashboards.【F:monGARS/core/long_haul_validation.py†L156-L226】
- Updated documentation (including runbooks) enables on-call operators to triage
  RL incidents without specialist knowledge.【F:docs/reinforcement_rollout_runbook.md†L1-L160】

## Follow-On Work Once RL Is Operational

- Resume energy-efficiency research outlined for the sustainability phase once
  RL observability is standardised across deployments.【F:modules/evolution_engine/energy.py†L1-L120】
- Expand partner telemetry to correlate RL-driven responses with engagement and
  satisfaction metrics using the shared dashboards as the foundation.【F:docs/codebase_status_report.md†L137-L188】
