import { useEffect, useState } from 'react';
import { NativeEventEmitter, NativeModules, Platform } from 'react-native';
import VoiceControl from '../native/voice';
import { settings } from '../services/config';

type VoiceState = {
  listening: boolean;
  transcript: string;
};

export type VoiceAssistantValue = {
  listening: boolean;
  transcript: string;
  start: () => Promise<void>;
  stop: () => Promise<void>;
};

type VoiceAssistantDeps = {
  platform: typeof Platform;
  createEmitter: () => NativeEventEmitter;
  voiceControl: typeof VoiceControl;
};

const defaultDeps: VoiceAssistantDeps = {
  platform: Platform,
  createEmitter: () => new NativeEventEmitter(NativeModules.VoiceModule),
  voiceControl: VoiceControl,
};

export const createUseVoiceAssistant = (
  deps: VoiceAssistantDeps = defaultDeps,
) => {
  return function useVoiceAssistant(
    onFinalText: (text: string) => void,
  ): VoiceAssistantValue {
    const [state, setState] = useState<VoiceState>({
      listening: false,
      transcript: '',
    });
    const isSupportedPlatform =
      deps.platform.OS === 'ios' || deps.platform.OS === 'android';

    useEffect(() => {
      if (!isSupportedPlatform) {
        return;
      }
      const emitter = deps.createEmitter();

      const resultListener = emitter.addListener('onTranscript', (event) => {
        setState((current) => ({ ...current, transcript: event.text }));
        if (event.isFinal) {
          setState({ listening: false, transcript: '' });
          onFinalText(event.text);
        }
      });

      const errorListener = emitter.addListener(
        'onTranscriptError',
        (event) => {
          console.warn('[VoiceModule] error', event.message);
          setState({ listening: false, transcript: '' });
        },
      );

      deps.voiceControl.setOnResultListener();

      return () => {
        resultListener.remove();
        errorListener.remove();
        deps.voiceControl.removeOnResultListener();
      };
    }, [isSupportedPlatform, onFinalText]);

    const start = async () => {
      if (!isSupportedPlatform) {
        return;
      }
      await deps.voiceControl.startListening(settings.voiceLocale);
      setState({ listening: true, transcript: '' });
    };

    const stop = async () => {
      if (!isSupportedPlatform) {
        return;
      }
      await deps.voiceControl.stopListening();
      setState((current) => ({ ...current, listening: false }));
    };

    return {
      ...state,
      start,
      stop,
    };
  };
};

export const useVoiceAssistant = createUseVoiceAssistant();
