import React from 'react';
import { render } from '@testing-library/react-native';
import SettingsScreen from '../src/screens/SettingsScreen';
import { useChatStore } from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';

describe('SettingsScreen', () => {
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
});
