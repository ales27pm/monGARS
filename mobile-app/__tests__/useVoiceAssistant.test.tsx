import React from 'react';
import { render, act, waitFor } from '@testing-library/react-native';
import type { NativeEventEmitter } from 'react-native';
import { createUseVoiceAssistant } from '../src/hooks/useVoiceAssistant';
import type { VoiceAssistantValue } from '../src/hooks/useVoiceAssistant';

jest.mock('../src/native/voice', () => ({
  configureAudioSession: jest.fn().mockResolvedValue(undefined),
  startListening: jest.fn().mockResolvedValue(undefined),
  stopListening: jest.fn().mockResolvedValue(undefined),
  setOnResultListener: jest.fn(),
  removeOnResultListener: jest.fn(),
}));

const VoiceControl = require('../src/native/voice');

type VoiceAssistantHook = ReturnType<typeof createUseVoiceAssistant>;
type VoiceAssistantHookValue = VoiceAssistantValue;

const buildHarness = () => {
  const handlers: Record<string, (...args: any[]) => void> = {};
  const emitter = {
    addListener: jest.fn((event: string, handler: (...args: any[]) => void) => {
      handlers[event] = handler;
      return { remove: jest.fn() };
    }),
    removeAllListeners: jest.fn(() => {
      Object.keys(handlers).forEach((key) => delete handlers[key]);
    }),
  };
  const useHook = createUseVoiceAssistant({
    platform: { OS: 'ios' } as any,
    createEmitter: () => emitter as unknown as NativeEventEmitter,
    voiceControl: VoiceControl,
  });
  return { useHook, handlers, emitter };
};

const createComponent = (
  hook: VoiceAssistantHook,
  onFinalText: (text: string) => void,
  onRender: (value: VoiceAssistantHookValue) => void,
) => {
  const Component: React.FC = () => {
    const value: VoiceAssistantHookValue = hook(onFinalText);
    React.useEffect(() => {
      onRender(value);
    }, [value]);
    return null;
  };
  return Component;
};

describe('useVoiceAssistant', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('starts and stops listening while updating state', async () => {
    const { useHook } = buildHarness();
    let latest: VoiceAssistantHookValue | null = null;
    const Harness = createComponent(useHook, jest.fn(), (value) => {
      latest = value;
    });

    render(<Harness />);

    await waitFor(() => {
      expect(VoiceControl.setOnResultListener).toHaveBeenCalled();
    });

    await act(async () => {
      await latest!.start();
    });

    expect(VoiceControl.startListening).toHaveBeenCalled();
    expect(latest!.listening).toBe(true);

    await act(async () => {
      await latest!.stop();
    });

    expect(VoiceControl.stopListening).toHaveBeenCalled();
    expect(latest!.listening).toBe(false);
  });

  it('invokes the final text callback when receiving transcript events', async () => {
    const { useHook, handlers } = buildHarness();
    const onFinal = jest.fn();
    let latest: VoiceAssistantHookValue | null = null;
    const ListenerHarness = createComponent(useHook, onFinal, (value) => {
      latest = value;
    });

    render(<ListenerHarness />);

    await waitFor(() => {
      expect(VoiceControl.setOnResultListener).toHaveBeenCalled();
    });

    const transcriptHandler = handlers['onTranscript'];
    expect(transcriptHandler).toBeDefined();

    act(() => {
      transcriptHandler?.({ text: 'bonjour', isFinal: false });
    });

    expect(latest!.transcript).toBe('bonjour');

    act(() => {
      transcriptHandler?.({ text: 'bonjour', isFinal: true });
    });

    expect(onFinal).toHaveBeenCalledWith('bonjour');
    expect(latest!.transcript).toBe('');
  });
});
