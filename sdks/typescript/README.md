# monGARS TypeScript SDK

> **Last updated:** 2025-10-06 _(auto-synced; run `python scripts/update_docs_metadata.py`)_

This package exposes a modern, Promise-based wrapper around the monGARS REST
API. It runs in both browser and Node.js environments by using
[`cross-fetch`](https://github.com/lquixada/cross-fetch).

## Installation

```bash
npm install @mongars/sdk
```

To build the package locally prior to publishing, use the shared release helper
from the repository root:

```bash
python -m scripts.sdk_release --output dist/sdk
```

The script installs dependencies, compiles the TypeScript sources, and writes an
npm tarball to `dist/sdk/typescript/`. Publish the artefact with
`npm publish dist/sdk/typescript/*.tgz`.

For day-to-day development you can still run the package scripts directly:

```bash
cd sdks/typescript
npm install
npm run build
```

## Usage

```ts
import { MonGARSClient } from "@mongars/sdk";

const client = new MonGARSClient({ baseUrl: "https://api.mongars.local" });
await client.login({ username: "alice", password: "secret" });

const reply = await client.chat({ message: "Hello" });
console.log(reply.response);
```

Check the [`examples/`](examples/) folder for an interactive CLI example that
covers login, chat, and streaming suggestions.

### Browser security guidance

When instantiating the client inside a browser, the SDK defaults to
`credentials: "omit"` so cross-site requests never forward cookies or other
ambient credentials. If your deployment relies on a CSRF token, pass it via
custom headers and keep bearer tokens in memory (or a hardened storage
mechanism) rather than mixing them with cookie-based sessions:

```ts
const client = new MonGARSClient({
  baseUrl: "https://api.mongars.local",
  credentials: "omit", // default, shown for clarity
  defaultHeaders: {
    "X-CSRF-Token": window.csrfToken,
  },
});
```

Avoid enabling `credentials: "include"` unless you have server-side CSRF
protections (such as same-site cookies and rotating anti-CSRF tokens) in place.

## API coverage

- Authentication: `login`, `registerUser`
- Conversation: `chat`, `history`
- Review: `fetchRagContext`
- UI: `suggestActions`
- Peer operations: `registerPeer`, `unregisterPeer`, `listPeers`, `peerLoad`,
  `publishPeerTelemetry`, `peerTelemetry`
- Model management: `modelConfiguration`, `provisionModels`

The SDK throws a typed `ApiError` when requests fail, making it easy to surface
actionable feedback to end users.
