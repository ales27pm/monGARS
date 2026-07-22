import {
  NativeAppIntentContractError,
  NativeAppIntentUnavailableError,
  getPendingNativeAppIntentHandoff,
  nativeAppIntentModuleAvailable,
  normalizeNativeAppIntentHandoff,
  normalizeNativeAppIntentHandoffSignal,
} from '../src/native/appIntents';

describe('App Intent native facade', () => {
  it('fails closed when the iOS handoff bridge is unavailable', async () => {
    expect(nativeAppIntentModuleAvailable).toBe(false);
    await expect(
      getPendingNativeAppIntentHandoff('guest'),
    ).rejects.toBeInstanceOf(NativeAppIntentUnavailableError);
  });

  it('normalizes a bounded foreground handoff', () => {
    expect(
      normalizeNativeAppIntentHandoff({
        id: '44444444-4444-4444-8444-444444444444',
        kind: 'memorySearch',
        input: '  atelier  ',
        createdAt: '2026-07-21T12:00:00Z',
        expiresAt: '2026-07-21T12:10:00Z',
        profileMatches: true,
      }),
    ).toEqual({
      id: '44444444-4444-4444-8444-444444444444',
      kind: 'memorySearch',
      input: 'atelier',
      createdAt: '2026-07-21T12:00:00.000Z',
      expiresAt: '2026-07-21T12:10:00.000Z',
      profileMatches: true,
    });
  });

  it('rejects a content-bearing handoff whose protected input is absent', () => {
    const base = {
      id: '44444444-4444-4444-8444-444444444444',
      kind: 'memoryAdd',
      createdAt: '2026-07-21T12:00:00Z',
      expiresAt: '2026-07-21T12:10:00Z',
      profileMatches: true,
    };
    expect(() => normalizeNativeAppIntentHandoff(base)).toThrow('doit être');
  });

  it('rejects unknown kinds, invalid diagnostics payloads and oversized input', () => {
    const base = {
      id: '44444444-4444-4444-8444-444444444444',
      createdAt: '2026-07-21T12:00:00Z',
      expiresAt: '2026-07-21T12:10:00Z',
      profileMatches: true,
    };
    expect(() =>
      normalizeNativeAppIntentHandoff({ ...base, kind: 'network', input: 'x' }),
    ).toThrow(NativeAppIntentContractError);
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'diagnostics',
        input: 'start capture',
      }),
    ).toThrow('ne doit pas contenir');
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'runTrigger',
        input: 'é'.repeat(257),
      }),
    ).toThrow('trop long');
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'memoryAdd',
        input: 'x'.repeat(187),
      }),
    ).toThrow('trop long');
  });

  it('accepts only masked metadata for a different profile', () => {
    const base = {
      id: '44444444-4444-4444-8444-444444444444',
      createdAt: '2026-07-21T12:00:00Z',
      expiresAt: '2026-07-21T12:10:00Z',
    };
    expect(
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'masked',
        profileMatches: false,
      }),
    ).toEqual({
      ...base,
      createdAt: '2026-07-21T12:00:00.000Z',
      expiresAt: '2026-07-21T12:10:00.000Z',
      kind: 'masked',
      profileMatches: false,
    });
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'memoryAdd',
        profileMatches: false,
      }),
    ).toThrow('doit être masqué');
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'masked',
        input: 'private',
        profileMatches: false,
      }),
    ).toThrow('ne doit pas être révélé');
    expect(() =>
      normalizeNativeAppIntentHandoff({
        ...base,
        kind: 'masked',
        profileMatches: true,
      }),
    ).toThrow('ne peut pas appartenir');
  });

  it('accepts only opaque metadata in warm-launch signals', () => {
    expect(
      normalizeNativeAppIntentHandoffSignal({
        id: '44444444-4444-4444-8444-444444444444',
        createdAt: '2026-07-21T12:00:00Z',
      }),
    ).toEqual({
      id: '44444444-4444-4444-8444-444444444444',
      createdAt: '2026-07-21T12:00:00.000Z',
    });
    expect(() =>
      normalizeNativeAppIntentHandoffSignal({
        id: '44444444-4444-4444-8444-444444444444',
        kind: 'memoryAdd',
        createdAt: '2026-07-21T12:00:00Z',
      }),
    ).toThrow('ni kind ni input');
  });
});
