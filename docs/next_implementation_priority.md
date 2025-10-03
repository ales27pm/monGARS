# Next Implementation Priority

> **Note:** This planning document is intentionally temporary. Archive or remove
> it once the migration initiative completes so the permanent documentation
> stays focused on long-lived workflows.

## Summary
Documentation across the roadmap and implementation audit identifies schema
evolution gaps as the most pressing blocker to production readiness. The
immediate focus should be expanding Alembic migrations so recent SQLModel
additions deploy cleanly without relying on ad-hoc bootstrap scripts.

## Supporting Signals from Existing Documentation
- **Roadmap**: Phase 3 highlights the need to "extend Alembic migrations for the
  newest SQLModel tables" before the hardware and performance milestone can be
  closed.
- **Implementation Status Report**: Lists expanded Alembic migrations as the
  outstanding item for Phase 3 and flags schema evolution as a key
  contradiction, warning that deployments still depend on `init_db.py` instead
  of structured migrations.
- **Security & Stability Charter**: Treats schema drift as a reliability risk
  alongside telemetry gaps and credential hardening, reinforcing its priority in
  the hotlist.

## Rationale for Prioritising Alembic Migration Coverage
1. **Deployment Safety** – Environments that skip `init_db.py` (e.g. production
   rollouts that assume migrations) currently miss new tables, risking runtime
   failures and data loss.
2. **Upgrade Automation** – Formal migrations are prerequisites for blue/green
   and rolling upgrades across Kubernetes clusters.
3. **Downstream Tasks Depend on It** – Credential hardening and database-backed
   auth flows require consistent schemas; completing migrations unblocks those
   efforts.

## Implementation Outline
1. **Inventory Current Models** – Enumerate SQLModel tables introduced since the
   last Alembic revision (e.g. via `SQLModel.metadata.tables`).
2. **Generate Revisions** – Use `alembic revision --autogenerate -m "<summary>"`
   to capture schema deltas, then review and hand-tune for deterministic
   defaults and indexes.
3. **Backfill Data** – Introduce migration steps that populate critical columns
   or seed reference data previously inserted by `init_db.py`.
4. **Update Bootstrap Scripts** – Trim redundant table creation from
   `init_db.py` once migrations cover the same structures; keep idempotent data
   seeding where necessary.
5. **Document Rollout** – Record upgrade steps in `docs/ray_serve_deployment.md`
   and ops runbooks so operators know how to apply the new migrations.
6. **Add Tests** – Extend integration tests (or add new fixtures) that spin up an
   empty database, apply migrations up and down to `head` and back to `base`, and
   verify the ORM can interact with the resulting schema at each stage.

## Success Criteria & Validation
- Alembic `upgrade head` completes on a blank database without manual scripts.
- Existing test suites pass using only migrations for schema creation.
- Documentation reflects the new workflow, including rollback guidance if a
  migration fails in staging.

## Follow-On Work Once Migrations Ship
- Emit Ray Serve success/failure counters through OpenTelemetry as captured in
  the roadmap and audit documents.
- Replace demo credential stores with the database-backed authentication flow,
  now safe to deploy thanks to consistent schema evolution.
- Continue credential and secrets hardening once the database layer is stable.
