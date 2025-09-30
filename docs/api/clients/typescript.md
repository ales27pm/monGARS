# TypeScript Client Example

This sample demonstrates how to generate a typed SDK with
[`openapi-typescript-codegen`](https://www.npmjs.com/package/openapi-typescript-codegen)
and consume it in a modern application (Node.js or React Native).

## Prerequisites

```bash
npm install --save-dev openapi-typescript-codegen
```

## Generate the Client

```bash
npx openapi-typescript-codegen --input docs/api/openapi.json --output ./generated/mongars
```

The generator produces:

- A `Client` wrapper with configurable base URL and interceptors.
- Namespaced functions for each API group (e.g. `PeerService.registerPeer`).
- Type definitions mirroring the Pydantic models in `monGARS.api.schemas`.

## Example Usage

```ts
import { Client, ConversationService } from '../generated/mongars';

const client = new Client({
  BASE: 'https://mongars.example.com',
  TOKEN: async () => `Bearer ${process.env.MONGARS_TOKEN ?? ''}`,
});

async function run() {
  const chat = await ConversationService.postApiV1ConversationChat(
    {
      message: 'Draft a post-incident summary for ticket #42',
    },
    client,
  );

  console.log(chat.response, chat.confidence);
}

run().catch((error) => {
  console.error('monGARS client error', error);
});
```

## React Native Considerations

- Use a fetch-compatible polyfill such as `cross-fetch` or `whatwg-fetch` if the
target runtime does not provide `fetch` by default.
- Wrap network calls with platform-aware permission guards when integrating with
native diagnostics modules, as described in the repository guidelines.
