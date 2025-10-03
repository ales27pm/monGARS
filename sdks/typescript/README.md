# monGARS TypeScript SDK

This package exposes a modern, Promise-based wrapper around the monGARS REST
API. It runs in both browser and Node.js environments by using
[`cross-fetch`](https://github.com/lquixada/cross-fetch).

## Installation

```bash
cd sdks/typescript
npm install
npm run build
```

To consume the SDK from another project:

```bash
npm install @mongars/sdk
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
