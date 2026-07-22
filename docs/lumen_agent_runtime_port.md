# Lumen Agent Runtime Port

> **Last updated:** 2026-07-21

This document defines the production boundary for the Lumen-derived agent
runtime in monGARS. The port keeps monGARS' pinned Core ML inference stack and
adopts Lumen's structured tool, routing, approval, memory, and dataset
contracts. It does not copy Lumen's legacy GGUF runtime or claim an MLX path.

## Runtime ownership

| Layer | Source of truth | Responsibility |
| --- | --- | --- |
| Core ML model | `mobile-app/ios/MonGARSCoreML/Sources/MonGARSCoreML/ModelManifest.swift` | Pinned Hugging Face revision, exact file hashes, tokenizer, stateful generation |
| Native policy kernel | `mobile-app/ios/MonGARSCoreML/Sources/MonGARSCoreML/Agent*.swift` | Typed JSON, routing, schema validation, permissions, approvals, bounded execution |
| React Native mirror | `mobile-app/src/agent/` | Deterministic preflight, routed prompt construction, UI-safe result types, parity tests |
| iOS host tools | `mobile-app/ios/AgentTools/` | Apple-framework and Microsoft Graph implementations registered at runtime |
| Server policy | `monGARS/core/operator_approvals.py` | Expiring, owner/prompt-bound, one-shot approvals for server guardrails |
| Dataset engineering | `tools/monGARS_deep_scan/` | Swift prompt/tool extraction with stable provenance for SFT, agent, and RAG corpora |

The Swift kernel is authoritative for on-device execution. The TypeScript
mirror must remain fail-closed and must never broaden the native catalog.

## Structured execution contract

The model returns exactly one JSON decision per turn: an action with a
canonical tool ID and object arguments, or a final answer. The executor then:

1. Routes the user request to one of 22 intents and intersects the route with
   the host's currently available tool IDs.
2. Requests clarification before model generation when required information is
   absent.
3. Parses one JSON value, normalizes safe aliases, and rejects unknown tools,
   extra fields, missing fields, and incorrect JSON types.
4. Applies foreground/background, permission, risk, and approval policy outside
   the model.
5. Blocks duplicate tool/argument calls and consumes an approval only once for
   the exact bound call.
6. Sanitizes and bounds every observation before returning it to the model.
7. Allows one repair turn for malformed model output and stops at a bounded
   decision limit.

Before model generation, the native kernel can resolve a bounded clarification
answer or one unambiguous person reference from recent user-authored history.
It also recognizes four explicit two-step plans (memory save/recall, nearby
search, current-location weather, and calendar list/create). Every planned
call still crosses the same schema, permission, approval, duplicate, and
availability gates as a model-generated call. A failed read may try one
different read-only fulfillment tool; mutations, denials, repeated calls, and
invented success remain terminal.

Tool-oriented user requests are limited to 512 UTF-8 bytes. This keeps the
complete request, routed tool contract, bounded observations, and a useful
answer inside the on-device model's 2,048-token state and 192-token generation
budget. Oversized requests fail with an explicit validation result; they are
not silently truncated into a different action.

Tool observations are untrusted data, not instructions. Private reasoning is
not part of any public event, persisted record, or diagnostic payload.

## Capability manifest

The catalog contains exactly 53 canonical tools. Twenty-six tools require an
explicit foreground approval before the host executor can run them.

| Group | Count | Examples |
| --- | ---: | --- |
| Productivity | 17 | Calendar, reminders, triggers, and ten AlarmKit operations |
| Communication | 20 | Contacts, system drafts/calls, and sixteen Outlook Graph operations |
| Location, media, health | 8 | Location, weather, maps, photos, camera, HealthKit, motion |
| Knowledge | 8 | Web, files, memory, local RAG, and indexing |

Aliases are input-only and never become executable identities. In particular,
legacy `open.url` is not treated as approval-free `web.fetch`. The runtime
advertises only tools whose host implementation and required platform service
are actually available; an absent entitlement, permission, account token, or
OS API returns `unavailable` or a permission boundary rather than fabricated
success.

## Approval lifecycle

Native tool approvals are capped, expire after ten minutes by default, and
transition through `pending → approved → consumed`. Rejecting or expiring a
record permanently prevents execution. Consumption atomically verifies the
canonical tool ID and the lossless JSON arguments.

An approved resume must execute the exact approved record before it can return
a successful final answer. The model cannot replace an approved mutation with
a different read or mutation and still claim completion. A failed resume drops
the React Native binding so the record cannot be replayed; the native record
then expires normally.

Server security approvals use the same fail-closed shape. The public blocked
response includes only a high-entropy reference. Stored audit data contains a
hash of detected PII rather than its value, credential-like context fields are
redacted, and legacy proof tokens are stored only as SHA-256 digests. A retry
must match the authenticated user and prompt hash and consumes the approval so
it cannot be replayed.

## Model and Hugging Face provenance

The iOS runtime remains pinned to
`ales27pm/Dolphin3.0-CoreML@95671cf9a2f56d2a381816ae264cd9aae335d96f`.
Downloads are allow-listed and checked by exact byte size and SHA-256 before
Core ML compilation. Do not move this manifest to a mutable Hub branch without
regenerating and reviewing every expected hash.

Server inference preserves a tokenizer's native chat template. A role-aware
fallback is installed only when a tokenizer has none. Ollama receives
structured messages once, Transformer generation decodes completion tokens
only, and a streaming endpoint either yields genuine provider deltas or
reports that streaming is unavailable.

Semantic indexes do not accept synthetic hash vectors or silently resized
vectors. When the embedding identity or dimension changes, callers must rebuild
the affected index; lexical retrieval remains the honest degraded path.

Each persisted server vector now carries a SHA-256 storage identity derived
from its backend, model, revision, and dimension. Retrieval requires an exact
identity match, so rows produced by a previous model cannot enter a new model's
similarity search. Apply `20260721_01_add_embedding_identity` before deployment.
Legacy rows have no identity and are deliberately excluded until they are
re-embedded. The configured 3,072-dimensional vectors use exact pgvector
distance scans: PostgreSQL's 2,000-dimension IVFFlat limit means the ORM does
not declare an invalid approximate index.

Set `EMBEDDING_MODEL_REVISION` to a stable Hub commit, Ollama digest, or internal
release ID before creating a durable index. The compatibility default is
`unversioned`; it cannot detect mutable weights published under an unchanged
model name.

## iOS host lifecycle

Permissions are requested only after a user starts the relevant foreground
operation. Calendar creation accepts EventKit write-only authorization, while
calendar listing and event-relative triggers require full calendar access.
Reminders, contacts, location, photos, camera, HealthKit, motion, notifications,
WeatherKit, and AlarmKit retain their native denial and restricted states; the
agent cannot reinterpret those states as success.

Triggers support one relative notification, a daily `HH:mm` notification, a
repeating interval, and a notification scheduled before the next calendar
event. Their protected store is scoped by a SHA-256 owner identifier. A
notification carries only an opaque trigger ID; after a tap, the foreground UI
reveals the stored prompt and asks the user to **Run** or **Ignore** it. Returning
to an already-running app refreshes this handoff. monGARS never claims that iOS
will launch the model unattended in the background.

AlarmKit operations are exposed only on iOS 26 when the usage description and
runtime packaging checks pass. Countdown presentation is supplied by the
embedded `MonGARSAlarmWidget` Live Activity. Older systems keep the rest of the
iOS 18 app available and report AlarmKit as unavailable.

Microsoft Graph access uses OAuth 2.0 Authorization Code with PKCE in the
native app. Set `MONGARS_MICROSOFT_CLIENT_ID` as a local target build-setting
override or an `xcodebuild` command-line assignment (the checked-in value is
empty), register
`msauth.<PRODUCT_BUNDLE_IDENTIFIER>://auth` as a mobile/native public-client
redirect URI in Microsoft Entra, and never add a client secret. The app asks
only for `User.Read`, `Mail.ReadWrite`, `Mail.Send`, and `offline_access`.
Tokens are stored in the device-only, unlocked Keychain under a case-sensitive
owner scope. `outlook.status` is available without a session for local
diagnostics; the other Outlook operations are advertised only for that exact
owner when a usable session exists. Graph requests use fixed `v1.0` templates,
bounded projections, and at most one refresh-and-retry after HTTP 401. A 403 is
returned as an authorization failure and never triggers token refresh.

Local memories, trigger prompts, and RAG metadata use protected app storage and
opaque owner scopes. Imported documents are resolved only under
`Documents/ImportedDocuments`; traversal, symlink escape, unsupported types,
oversized input, and invalid UTF-8 fail closed. Local RAG is deterministic
lexical retrieval with provenance and checksums, not a synthetic embedding
claim.

Five App Intents expose Ask, Search Memory, Add Memory, Run Trigger, and
Diagnostics through Siri and Shortcuts. They create only a protected,
ten-minute, one-shot handoff; they never run the model, read private results,
write memory, or execute a trigger in the background. The foreground React
Native UI shows the exact pending action and requires **Execute/Open** or
**Ignore**. The app binds future handoffs to an opaque SHA-256 profile scope
explicitly during initialization and session changes. A handoff captured for a
different profile crosses the bridge only as an opaque identifier, timestamps,
and a `masked` placeholder: both its content and action kind remain hidden,
execution is blocked, and **Ignore** can discard only that exact identifier.
Warm-launch notifications likewise carry no action kind or input.

Search Memory and Add Memory never become model prompts. After confirmation,
the native bridge atomically compares and consumes the exact protected
identifier, owner, kind, and input, then derives only `memory.recall(query)` or
`memory.save(content, kind: fact)` from the consumed record. These operations
are one-shot and are never retried automatically. Their owner-scoped result
remains visible in the foreground even when the server chat backend is selected,
but it is excluded from later model conversation history. A successful
memory-add host status is reported as committed even if its output is empty; an
incoherent failure warns that the add may already have succeeded and asks the
user to verify before any manual retry.

Run Trigger resolves the owner-scoped trigger during preview and displays its
exact title, prompt, and repeat state before **Execute** is enabled. On Execute,
the app resolves the displayed UUID again and compares the full snapshot before
acknowledgement. Any drift launches no agent. Ask and unchanged trigger prompts
that require tools receive a deterministic native intent/tool allow-list; they
may use the local model or confirmed provider tools only after this foreground
boundary. iOS 26 uses immediate foreground intent mode, while the pre-iOS 26
compatibility path opens the app before the handoff is consumed. The app target
also packages `PrivacyInfo.xcprivacy` in its Resources build phase.

## Dataset and evaluation workflow

The deep scanner includes `.swift` by default. It extracts:

- documentation comments and prompt templates into provenance-backed embedding
  and SFT records;
- compact `AgentToolCatalog.tool(...)` declarations into structured agent
  records; and
- canonical Lumen-style `ToolDefinition(...)` declarations for compatibility.

The catalog regression test requires 53 unique tool records and 26 approval
flags. This prevents an iOS policy change from silently disappearing from the
engineering corpus.

`agent_manifest_pipeline` closes that source contract into a deterministic
engineering bundle. It fails unless the Swift sources expose exactly 53 tools,
26 approval-protected tools, and 22 routes; then it emits content-addressed
SFT/DPO train and validation lanes for Cortex, Executor, Mouth, Mimicry, and
REM, plus schemas, evaluation scenarios, provenance, and SHA-256 indexes.
Bounded runtime audit files can be ingested as repair samples. Static reports
cannot claim live E2E ownership, and the gap report stays open until genuine
runtime evidence is supplied.

## Validation

Run the portable checks from the repository root:

```bash
cd mobile-app
npm run typecheck
npm test -- --runInBand --no-cache
npm run lint
npm run doctor:native

cd ..
pytest -q tests/test_operator_approvals.py tests/test_extractors_unit.py \
  tests/test_swift_agent_catalog_scan.py
pytest -q tests/test_inference_utils.py tests/test_chat_templates.py \
  tests/test_llm_integration.py tests/test_unified_llm_runtime.py \
  tests/test_neuron_manager.py tests/test_mlops_artifacts.py
pytest -q tests/test_embeddings.py tests/test_persistence.py
```

Run the native package and app tests on macOS with the repository's pinned
Swift package dependencies, then perform a physical-device release pass for
model download/compilation, permissions, approval/replay, background denial,
tool UI presentation, cancellation, low-power mode, and serious/critical
thermal state. AlarmKit requires an iOS 26 SDK and device; its absence on an
older SDK is an explicit unavailable capability, not a skipped success.
