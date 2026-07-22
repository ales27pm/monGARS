import {
  NativeOutlookContractError,
  NativeOutlookUnavailableError,
  OUTLOOK_REQUIRED_SCOPES,
  configureNativeOutlookClientId,
  getNativeOutlookConnectionStatus,
  nativeOutlookModuleAvailable,
  normalizeOutlookConnectionStatus,
} from '../src/native/outlook';

const validStatus = {
  configured: true,
  connected: true,
  account: 'alice@example.com',
  expiresAt: '2026-07-21T12:00:00.000Z',
  requiredScopes: [...OUTLOOK_REQUIRED_SCOPES],
  redirectUri: 'msauth.com.mongars.mobile://auth',
  detail: 'Compte Outlook connecté.',
};

describe('native Outlook facade', () => {
  it('strictly decodes a token-free connected status', () => {
    expect(normalizeOutlookConnectionStatus(validStatus)).toEqual(validStatus);
  });

  it('rejects missing scopes, impossible connection state, and secret fields', () => {
    expect(() =>
      normalizeOutlookConnectionStatus({
        ...validStatus,
        requiredScopes: ['User.Read'],
      }),
    ).toThrow(NativeOutlookContractError);
    expect(() =>
      normalizeOutlookConnectionStatus({
        ...validStatus,
        configured: false,
      }),
    ).toThrow(NativeOutlookContractError);
    expect(() =>
      normalizeOutlookConnectionStatus({
        ...validStatus,
        accessToken: 'must-never-cross-the-bridge',
      }),
    ).toThrow(NativeOutlookContractError);
  });

  it('fails closed when the native methods are not linked', async () => {
    expect(nativeOutlookModuleAvailable).toBe(false);
    await expect(
      getNativeOutlookConnectionStatus('account:alice'),
    ).rejects.toBeInstanceOf(NativeOutlookUnavailableError);
  });

  it('validates the monGARS owner before crossing the native boundary', async () => {
    await expect(
      getNativeOutlookConnectionStatus(' \n '),
    ).rejects.toBeInstanceOf(NativeOutlookContractError);
    await expect(
      getNativeOutlookConnectionStatus(`account:${'a'.repeat(260)}`),
    ).rejects.toBeInstanceOf(NativeOutlookContractError);
  });

  it('validates a runtime Microsoft client ID before crossing native code', async () => {
    await expect(
      configureNativeOutlookClientId('account:alice', 'not-a-uuid'),
    ).rejects.toBeInstanceOf(NativeOutlookContractError);
  });
});
