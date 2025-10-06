# Next Implementation Priority

## Summary

With Retrieval-Augmented Generation governance complete, the immediate focus
shifts to productionising the reinforcement-learning (RL) research loop. The
loop exists in `modules/neurons/training/reinforcement_loop.py` but still lacks
telemetry, rollout safeguards, and operator controls required for unattended
operation.【F:docs/implementation_status.md†L96-L140】【F:modules/neurons/training/reinforcement_loop.py†L320-L520】

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

1. **Instrumentation** – emit structured metrics and events across the RL loop
   (policy selection, reward aggregation, adapter export) tied into the existing
   OpenTelemetry pipeline.【F:modules/neurons/training/reinforcement_loop.py†L320-L520】
2. **Safeguards** – add approval hooks to the evolution orchestrator so RL
   results require an operator sign-off before deployment, mirroring curated
   dataset governance.【F:modules/evolution_engine/orchestrator.py†L320-L520】
3. **Runbooks** – document the RL workflow alongside the new governance guide so
   operators understand how to stage, review, and roll back experiments.【F:docs/rag_dataset_governance.md†L1-L160】
4. **Testing** – extend the existing unit tests to cover RL edge cases and add a
   targeted integration scenario verifying telemetry and approval gates.

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
