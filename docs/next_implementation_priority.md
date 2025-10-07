# Next Implementation Priority

## Summary

Instrumentation, rollout safeguards, and operator approvals have now been
implemented for the reinforcement-learning (RL) research loop. The loop emits
OpenTelemetry spans, streams metrics, and records approval requests before
deploying artefacts, closing the operational gap highlighted in the status
reports.【F:docs/implementation_status.md†L96-L140】【F:modules/neurons/training/reinforcement_loop.py†L520-L760】

## Supporting Signals

- **Implementation Status Report** – calls out RL automation as the remaining
  contradiction after the governance milestone.【F:docs/implementation_status.md†L96-L140】
- **Codebase Status Report** – recommends prioritising telemetry and operational
  controls before graduating RL into production workflows.【F:docs/codebase_status_report.md†L96-L140】
- **Evolution Engine Orchestrator** – already integrates curated datasets,
  making it the natural landing spot once RL artefacts can be generated and
  approved safely.【F:modules/evolution_engine/orchestrator.py†L1-L320】

## Rationale for Prioritising RL Operationalisation

1. **Experiment Velocity** – moving RL into the automated cycle reduces the
   manual work currently required to evaluate policy improvements.
2. **Safety & Observability** – explicit telemetry and guardrails prevent
   regressions when exploring novel reward models.
3. **Partner Readiness** – RL-backed suggestions unlock new product
   capabilities, but only once the rollout and rollback story matches the rest
   of the platform.

## Implementation Outline

1. **Runbooks** – publish the RL rollout guide alongside the RAG governance
   documentation so operators understand how to review approval queues and
   rollback artefacts if needed.【F:docs/reinforcement_rollout_runbook.md†L1-L160】
2. **Integration Tests** – schedule long-haul evaluation covering sustained RL
   runs, ensuring telemetry and approval checkpoints stay healthy under load.
3. **Metrics Dashboards** – extend observability dashboards to include the new
   `reinforcement.loop.*` metrics emitted by the telemetry sink.

## Success Criteria & Validation

- RL training cycles emit metrics and logs that surface in the same dashboards
  as self-training events.
- Deployment of RL artefacts requires an explicit approval or automated policy
  pass, with clear rollback instructions.
- Updated documentation (including runbooks) enables on-call operators to triage
  RL incidents without specialist knowledge.

## Follow-On Work Once RL Is Operational

- Resume energy-efficiency research outlined for the sustainability phase once
  RL experiments can run safely in production.
- Expand partner telemetry to correlate RL-driven responses with engagement and
  satisfaction metrics.
