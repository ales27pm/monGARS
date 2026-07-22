# MonGARSAgentTools

`MonGARSAgentTools` is the iOS host boundary for the portable agent kernel in
`MonGARSCoreML`. `IOSAgentToolExecutor` has an explicit one-to-one dispatch
entry for every canonical tool ID; unknown IDs, unavailable frameworks,
missing entitlements, missing permissions, absent Microsoft credentials, and
background UI requests fail closed.

## Host API

```swift
import MonGARSAgentTools

let tools = IOSAgentToolExecutor.shared
let available = await tools.availableToolIDs()
let result = await tools.execute(invocation: invocation)

let permissions = IOSAgentPermissionProvider.shared
let state = await permissions.state(for: .calendar)

// Use this wrapper for account-aware agent runs.
let profileTools = ScopedIOSAgentToolExecutor(rawOwnerID: ownerID)
```

`ScopedIOSAgentToolExecutor` hashes the raw owner identifier into an opaque
profile scope for memory, RAG, triggers, and Outlook credentials. The default
singleton uses the Keychain-backed native OAuth 2.0 Authorization Code + PKCE
provider. `outlook.status` remains available to a scoped owner for safe local
diagnostics; the other 15 Outlook tools are advertised only when that exact
owner has an unexpired access token or a refreshable session.

## Microsoft OAuth setup

The checked-in `MONGARS_MICROSOFT_CLIENT_ID` Debug/Release build setting is
intentionally empty. Set that user-defined target build setting in a local
Xcode configuration, or pass it on the `xcodebuild` command line, for example
`MONGARS_MICROSOFT_CLIENT_ID=11111111-2222-4333-8444-555555555555`. A client ID
is public application metadata, not a client secret; this native flow must
never receive or store a Microsoft client secret.

When the build setting is empty, the iOS Settings surface can instead persist
the public client ID in `UserDefaults` as a runtime fallback. A valid build-time
value always wins. OAuth tokens remain owner-scoped in the Keychain and never
cross the React Native bridge.

In Microsoft Entra, register the application as a mobile/native public client
and add the exact redirect URI
`msauth.<PRODUCT_BUNDLE_IDENTIFIER>://auth` (for the checked-in app identifier:
`msauth.com.mongars.mobile://auth`). The app requests only `User.Read`,
`Mail.ReadWrite`, `Mail.Send`, and `offline_access`.

## Implemented boundaries

- EventKit calendar and reminders, Contacts, Core Location, MapKit local
  search/directions, rich Photos metadata filtering, system message/mail
  composers, direct AVCapture photo capture, phone handoff, HealthKit summaries
  (including de-duplicated sleep intervals), and Core Motion steps, distance,
  and floors.
- WeatherKit when the SDK, entitlement, signed provisioning profile, and
  service account support it.
- All ten AlarmKit operations behind `canImport(AlarmKit)`, a non-empty usage
  description, and iOS 26 runtime availability: authorization, fixed alarms,
  timers, listing, pause/resume/stop/snooze/cancel, and AlarmKit presentation.
  Scheduled alarms default to a five-minute snooze countdown and use the
  embedded `MonGARSAlarmWidget` Live Activity; a runtime packaging preflight
  fails those operations closed if the extension is missing. The extension has
  an iOS 18 availability widget and conditionally exposes its AlarmKit Live
  Activity on iOS 26 or newer, so it does not raise the app's iOS 18 deployment
  target.
- All 16 Outlook operations through fixed Microsoft Graph `v1.0` request
  templates. Model arguments never control the Graph origin, HTTP method, or
  arbitrary endpoint. The local status operation is token-free; all resource
  operations remain unadvertised while the scoped token provider is
  unavailable.
- HTTPS-only fixed-provider web search with DNS address classification,
  redirect revalidation, cancellation, timeouts, and an in-flight response byte
  cap. Arbitrary `web.fetch` is fail-closed and unadvertised because URLSession
  cannot pin its independently resolved address after DNS validation; hosts may
  inject a pinned or strict-allowlist service. Graph uses the bounded transport
  and same-origin-only redirects so bearer credentials cannot follow
  cross-origin redirects.
- Protected profile-scoped memory, deterministic lexical recall, local RAG
  chunks with source provenance/checksums, safe imported-file containment, and
  photo-metadata indexing. No synthetic embeddings are mixed into the index.
  File read/index tools are advertised only when the app's imported-document
  directory already contains at least one safe text document. Users can place
  allow-listed UTF-8 files in **On My iPhone > MonGARS > ImportedDocuments**
  (the app's `Documents/ImportedDocuments` directory) through Files or Finder.
- Notification-backed scheduled handoffs. iOS does not guarantee unattended
  model execution, so the result explicitly says the user must open MonGARS.
  Schedules support daily wall-clock time, a one-shot relative delay, repeating
  intervals, and a lead time before the next calendar event. The event-relative
  mode requires full calendar access in addition to notifications. Cancellation
  accepts either the UUID returned by `trigger.list` or one unambiguous exact
  title.
  `AppDelegate` records only the opaque notification ID and tap time;
  it also emits the same opaque metadata through the mounted React Native event
  bridge so a foreground notification tap is observed without waiting for an
  app-state transition.
  `pendingTrigger(rawOwnerID:)` performs a ten-minute, owner-scoped foreground
  lookup of the protected prompt. The UI calls
  `acknowledgePendingTrigger(rawOwnerID:id:)` only when the user chooses Run or
  Ignore; Run acknowledges immediately before explicit foreground execution.
  Neither operation claims or starts a background agent run.

## Validation boundary

The package contains deterministic XCTest coverage for catalog completeness,
background denial, unknown-tool denial, path and symlink traversal, profile
memory isolation, Graph method/path/body construction, missing-token behavior,
DNS/private-address classification, and redirect policy.

This checkout environment has neither Swift nor Xcode, so the package and app
could not be compiled here. Before release, validate on macOS with the intended
Xcode (Xcode 26 for AlarmKit), a signed iOS 18+ simulator/device build, and a
physical iOS 26 device for AlarmKit. Confirm HealthKit and WeatherKit
capabilities in the App ID/provisioning profile. AlarmKit uses the checked-in
`NSAlarmKitUsageDescription` and `AlarmManager` runtime authorization; Apple
does not define a separate AlarmKit entitlement. Exercise all permission states,
and run the system composer, camera, phone, WeatherKit, Microsoft Graph,
notification, trigger, and AlarmKit Live Activity paths on-device before
release.
