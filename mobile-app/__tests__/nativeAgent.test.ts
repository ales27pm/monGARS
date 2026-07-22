import {
  NativeAgentContractError,
  NativeAgentUnavailableError,
  getNativeAgentCapabilities,
  nativeAgentModuleAvailable,
  normalizeNativeAgentTriggerSignal,
  runNativeAgent,
} from '../src/native/agent';

describe('native agent facade without a linked iOS module', () => {
  it('fails closed instead of falling back to a server', async () => {
    expect(nativeAgentModuleAvailable).toBe(false);
    await expect(getNativeAgentCapabilities('guest')).rejects.toBeInstanceOf(
      NativeAgentUnavailableError,
    );
    await expect(
      runNativeAgent({
        runId: '11111111-1111-4111-8111-111111111111',
        ownerId: 'guest',
        prompt: 'What is the weather in Toronto?',
        history: [],
      }),
    ).rejects.toBeInstanceOf(NativeAgentUnavailableError);
  });

  it('accepts tagged-email owner IDs before checking native availability', async () => {
    await expect(
      runNativeAgent({
        runId: '11111111-1111-4111-8111-111111111111',
        ownerId: 'account:user+tag@example.com',
        prompt: 'What is the weather in Toronto?',
        history: [],
      }),
    ).rejects.toBeInstanceOf(NativeAgentUnavailableError);
  });

  it('accepts a narrowed memory tool scope before checking native availability', async () => {
    await expect(
      runNativeAgent({
        runId: '11111111-1111-4111-8111-111111111111',
        ownerId: 'guest',
        prompt: 'Treat this as literal memory data',
        history: [],
        requestedIntent: 'memory',
        allowedToolIds: ['memory.recall'],
      }),
    ).rejects.toBeInstanceOf(NativeAgentUnavailableError);
  });

  it('rejects incomplete or cross-intent tool scopes before native execution', async () => {
    await expect(
      runNativeAgent({
        runId: '11111111-1111-4111-8111-111111111111',
        ownerId: 'guest',
        prompt: 'Search memory',
        history: [],
        requestedIntent: 'memory',
      }),
    ).rejects.toBeInstanceOf(NativeAgentContractError);
    await expect(
      runNativeAgent({
        runId: '11111111-1111-4111-8111-111111111111',
        ownerId: 'guest',
        prompt: 'Search memory',
        history: [],
        requestedIntent: 'memory',
        allowedToolIds: ['web.search'],
      }),
    ).rejects.toThrow("incompatible avec l'intention");
  });

  it('decodes the exact token-free native trigger signal shape', () => {
    expect(
      normalizeNativeAgentTriggerSignal({
        id: '33333333-3333-4333-8333-333333333333',
        tappedAt: '2026-07-21T12:00:00Z',
      }),
    ).toEqual({
      id: '33333333-3333-4333-8333-333333333333',
      tappedAt: '2026-07-21T12:00:00.000Z',
    });
    expect(() =>
      normalizeNativeAgentTriggerSignal({
        id: '33333333-3333-4333-8333-333333333333',
        tappedAt: 1_753_099_200,
      }),
    ).toThrow('tappedAt doit être une chaîne');
  });
});
