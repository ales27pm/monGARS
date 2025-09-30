# TypeScript Client Example

Generate a typed SDK for monGARS using
[`openapi-typescript-codegen`](https://www.npmjs.com/package/openapi-typescript-codegen).
The example below works for Node.js, React, or React Native projects.

## Prerequisites
```bash
npm install --save-dev openapi-typescript-codegen
```

## Generate the Client
```bash
npx openapi-typescript-codegen \
  --input docs/api/openapi.json \
  --output ./generated/mongars \
  --useUnionTypes
```

The generator produces:
- A configurable `Client` wrapper with base URL, auth token hook, and retry
  helpers.
- Namespaced services for each API group (e.g. `ConversationService`).
- Type definitions mirroring the Pydantic schemas.

## Example Usage
```ts
import { Client, ConversationService } from '../generated/mongars';

const client = new Client({
  BASE: process.env.MONGARS_URL ?? 'https://mongars.example.com',
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

## React Native Notes
- Provide a `fetch` polyfill (`whatwg-fetch`, `cross-fetch`) if the target
  runtime lacks a native implementation.
- Wrap network calls with platform-specific permission checks when integrating
  with native diagnostics modules.
