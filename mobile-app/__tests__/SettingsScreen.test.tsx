import React from 'react';
import { act, fireEvent, render, waitFor } from '@testing-library/react-native';
import { Alert, Linking } from 'react-native';
import SettingsScreen from '../src/screens/SettingsScreen';
import {
  configureNativeOutlookClientId,
  connectNativeOutlook,
  disconnectNativeOutlook,
  getNativeOutlookConnectionStatus,
  type NativeOutlookConnectionStatus,
} from '../src/native/outlook';
import { useChatStore } from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';

jest.mock('../src/native/outlook', () => ({
  nativeOutlookModuleAvailable: true,
  OUTLOOK_REQUIRED_SCOPES: [
    'User.Read',
    'Mail.ReadWrite',
    'Mail.Send',
    'offline_access',
  ],
  getNativeOutlookConnectionStatus: jest.fn(),
  configureNativeOutlookClientId: jest.fn(),
  connectNativeOutlook: jest.fn(),
  disconnectNativeOutlook: jest.fn(),
}));

const disconnectedOutlookStatus: NativeOutlookConnectionStatus = {
  configured: true,
  connected: false,
  account: null,
  expiresAt: null,
  requiredScopes: [
    'User.Read',
    'Mail.ReadWrite',
    'Mail.Send',
    'offline_access',
  ],
  redirectUri: 'msauth.com.mongars.mobile://auth',
  detail: 'Outlook est configuré, mais aucun compte Microsoft n’est connecté.',
};

describe('SettingsScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    useChatStore.setState({ session: null, loading: false });
    (getNativeOutlookConnectionStatus as jest.Mock).mockResolvedValue(
      disconnectedOutlookStatus,
    );
    (connectNativeOutlook as jest.Mock).mockResolvedValue({
      ...disconnectedOutlookStatus,
      connected: true,
      account: 'alice@example.com',
      detail: 'Compte Outlook connecté.',
    });
    (disconnectNativeOutlook as jest.Mock).mockResolvedValue(
      disconnectedOutlookStatus,
    );
    (configureNativeOutlookClientId as jest.Mock).mockResolvedValue(
      disconnectedOutlookStatus,
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('disables the on-device selector when Core ML is unavailable', () => {
    const status = {
      phase: 'unavailable' as const,
      modelId: null,
      displayName: null,
      revision: null,
      installedBytes: 0,
      contextLength: 0,
      minimumIOSVersion: 18,
      detail: 'Core ML indisponible',
    };
    useInferenceStore.setState({
      backend: 'server',
      status,
      activeRequestId: null,
      initialize: jest.fn().mockResolvedValue(status),
    });
    useChatStore.setState({ loading: false });

    const { getByRole } = render(<SettingsScreen />);

    expect(
      getByRole('button', { name: 'Sur l iPhone' }).props.accessibilityState,
    ).toMatchObject({ disabled: true });
  });

  it('shows the current download and free-space requirements', () => {
    const status = {
      phase: 'not-downloaded' as const,
      modelId: 'ales27pm/Dolphin3.0-CoreML',
      displayName: 'Dolphin 3.0',
      revision: '95671cf9a2f56d2a381816ae264cd9aae335d96f',
      installedBytes: 0,
      contextLength: 2048,
      minimumIOSVersion: 18,
      detail: null,
    };
    useInferenceStore.setState({
      backend: 'server',
      status,
      activeRequestId: null,
      initialize: jest.fn().mockResolvedValue(status),
    });
    useChatStore.setState({ loading: false });
    const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => {});

    const { getByText } = render(<SettingsScreen />);
    fireEvent.press(getByText('Telecharger et verifier'));

    expect(alert).toHaveBeenCalledWith(
      'Telecharger le modele local?',
      'Le telechargement fait environ 1.83 Go et exige au moins 5.00 Go libres. Utilisez de preference le Wi-Fi.',
      expect.any(Array),
    );
  });

  it('links to the pinned model revision and shows the required attribution', () => {
    const openURL = jest.spyOn(Linking, 'openURL').mockResolvedValue(undefined);
    const { getByText } = render(<SettingsScreen />);

    expect(getByText(/Built with Llama/)).toBeTruthy();
    fireEvent.press(getByText('Voir la provenance HF'));

    expect(openURL).toHaveBeenCalledWith(
      'https://huggingface.co/ales27pm/Dolphin3.0-CoreML/tree/95671cf9a2f56d2a381816ae264cd9aae335d96f/Dolphin3.0-Llama3.2-3B-stateful-int4.mlpackage',
    );
  });

  it('shows the active username without rendering any JWT prefix', async () => {
    useChatStore.setState({
      session: {
        username: 'Alice',
        token: 'secret-jwt-that-must-not-render',
      },
    });

    const { getByText, queryByText } = render(<SettingsScreen />);

    expect(getByText('Utilisateur: Alice')).toBeTruthy();
    expect(queryByText(/JWT:/)).toBeNull();
    expect(queryByText(/secret-jwt/)).toBeNull();
    await waitFor(() =>
      expect(getNativeOutlookConnectionStatus).toHaveBeenCalledWith(
        'account:Alice',
      ),
    );
  });

  it('requires a monGARS session before reading or connecting Outlook', () => {
    const { getByText, queryByRole } = render(<SettingsScreen />);

    expect(
      getByText(
        'Connectez-vous à monGARS avant de connecter un compte Outlook.',
      ),
    ).toBeTruthy();
    expect(queryByRole('button', { name: 'Connecter Outlook' })).toBeNull();
    expect(getNativeOutlookConnectionStatus).not.toHaveBeenCalled();
  });

  it('binds Outlook status and connect calls to the signed-in owner', async () => {
    useChatStore.setState({
      session: { username: 'Alice', token: 'session-token' },
    });
    const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => {});
    const { getByRole } = render(<SettingsScreen />);

    await waitFor(() =>
      expect(getNativeOutlookConnectionStatus).toHaveBeenCalledWith(
        'account:Alice',
      ),
    );
    fireEvent.press(getByRole('button', { name: 'Connecter Outlook' }));
    await waitFor(() =>
      expect(connectNativeOutlook).toHaveBeenCalledWith('account:Alice'),
    );
    expect(alert).toHaveBeenCalledWith(
      'Outlook connecté',
      'Compte actif: alice@example.com.',
    );
  });

  it('configures an unbranded build with a public Microsoft client ID', async () => {
    (getNativeOutlookConnectionStatus as jest.Mock).mockResolvedValue({
      ...disconnectedOutlookStatus,
      configured: false,
      detail: 'Microsoft Outlook n’est pas configuré.',
    });
    useChatStore.setState({
      session: { username: 'Alice', token: 'session-token' },
    });
    const { getByLabelText, getByRole } = render(<SettingsScreen />);
    await waitFor(() =>
      expect(getByLabelText('ID client Microsoft')).toBeTruthy(),
    );

    fireEvent.changeText(
      getByLabelText('ID client Microsoft'),
      '11111111-2222-4333-8444-555555555555',
    );
    fireEvent.press(
      getByRole('button', {
        name: 'Enregistrer l’ID client Microsoft',
      }),
    );

    await waitFor(() =>
      expect(configureNativeOutlookClientId).toHaveBeenCalledWith(
        'account:Alice',
        '11111111-2222-4333-8444-555555555555',
      ),
    );
  });

  it('drops a stale Outlook response after the monGARS owner changes', async () => {
    let resolveAlice:
      | ((status: typeof disconnectedOutlookStatus) => void)
      | undefined;
    (getNativeOutlookConnectionStatus as jest.Mock).mockImplementation(
      (ownerId: string) => {
        if (ownerId === 'account:Alice') {
          return new Promise<typeof disconnectedOutlookStatus>((resolve) => {
            resolveAlice = resolve;
          });
        }
        return Promise.resolve({
          ...disconnectedOutlookStatus,
          connected: true,
          account: 'bob@example.com',
          detail: 'Compte Bob connecté.',
        });
      },
    );
    useChatStore.setState({
      session: { username: 'Alice', token: 'alice-token' },
    });
    const { getByText, queryByText } = render(<SettingsScreen />);
    await waitFor(() =>
      expect(getNativeOutlookConnectionStatus).toHaveBeenCalledWith(
        'account:Alice',
      ),
    );

    act(() => {
      useChatStore.setState({
        session: { username: 'Bob', token: 'bob-token' },
      });
    });
    await waitFor(() =>
      expect(getNativeOutlookConnectionStatus).toHaveBeenCalledWith(
        'account:Bob',
      ),
    );
    await waitFor(() =>
      expect(getByText('Compte: bob@example.com')).toBeTruthy(),
    );

    await act(async () => {
      resolveAlice?.({
        ...disconnectedOutlookStatus,
        connected: true,
        account: 'alice@example.com',
      });
      await Promise.resolve();
    });
    expect(queryByText('Compte: alice@example.com')).toBeNull();
    expect(getByText('Compte: bob@example.com')).toBeTruthy();
  });
});
