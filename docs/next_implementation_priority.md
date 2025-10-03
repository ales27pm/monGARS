# Next Implementation Priority

## Summary
Now that schema evolution and telemetry upgrades are complete, the next
high-impact milestone is publishing first-party SDKs and reference clients while
credential hardening work wraps up.
Operators and partner teams currently rely on OpenAPI scraping or ad-hoc scripts;
packaged SDKs will harden integrations and shrink the support surface area as
the project moves into long-term maintenance.

## Supporting Signals from Existing Documentation
- **Roadmap**: Phase 5 keeps SDKs and reference clients as the final deliverable
  but still flags the presence of demo credential defaults that must be removed.
- **Implementation Status Report**: Highlights SDK packaging as the new
  contradiction to resolve, notes the outstanding credential cleanup, and calls
  out the need for clear RAG governance to accompany client distribution.
- **Roadmap Charter (AGENTS.md)**: Security & Stability items are satisfied,
  shifting attention to sustainable integration stories for external teams.

## Rationale for Prioritising SDK Publication
1. **Developer Experience** – Typed clients for Python/TypeScript remove the need
   for manual request crafting and ensure authentication, retries, and ticket
   handling remain consistent with production expectations.
2. **Operational Safety** – Providing vetted tooling reduces the temptation to
   embed long-lived tokens or bypass telemetry hooks, keeping observability and
   governance intact.
3. **Ecosystem Enablement** – Documented SDKs accelerate integrations with peer
   services, dashboards, and research notebooks without exposing internal
   modules or requiring Django coupling.

## Implementation Outline
1. **Define API Surfaces** – Lock the minimal stable routes (chat, history,
   review, peer management) and document any feature flags they depend on. Fold
   the credential cleanup into this pass by removing the `DEFAULT_USERS`
   bootstrap so SDK consumers cannot rely on demo logins.
2. **Generate Typed Clients** – Use `openapi-python-client` and `openapi-typescript`
   (or equivalent) to scaffold SDKs, layering ergonomic helpers for
   authentication flows, ticket refresh, and streaming responses.
3. **Harden Tooling** – Add CI jobs that build the SDK packages, run lint/tests,
   and verify compatibility with the published OpenAPI schema.
4. **Document Usage** – Provide quick-start guides in `docs/` plus inline docstrings
   covering auth bootstrapping, WebSocket streaming, and RAG toggles.
5. **Distribution Plan** – Decide on publication channels (private index, GitHub
   releases, internal registry) and automate version bumps alongside backend
   releases.
6. **Feedback Loop** – Capture telemetry and user feedback from SDK adoption to
   drive future roadmap refinements.

## Success Criteria & Validation
- Python and TypeScript SDKs install via package manager and authenticate against
  a fresh deployment without manual token crafting.
- Streaming chat flows, history pagination, and RAG review helpers function
  end-to-end using the SDK abstractions.
- SDK documentation and code examples are verified to be accurate and sufficient
  for a developer to integrate the client successfully.

## Follow-On Work Once SDKs Ship
- Finalise RAG dataset retention guidance and automation ahead of wider partner
  adoption.
- Resume experimentation on reinforcement learning loops and sustainability
  metrics as outlined in Phases 6 and 7 of the roadmap.
