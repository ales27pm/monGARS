import React from 'react';
import { fireEvent, render } from '@testing-library/react-native';
import { Alert } from 'react-native';
import SettingsScreen from '../src/screens/SettingsScreen';
import { useChatStore } from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';

describe('SettingsScreen', () => {
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
});
