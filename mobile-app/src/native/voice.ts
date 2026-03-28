import { TurboModuleRegistry } from 'react-native';
import type { TurboModule } from 'react-native';

export type TranscriptResult = {
  text: string;
  isFinal: boolean;
};

export interface VoiceSpec extends TurboModule {
  configureAudioSession(): Promise<void>;
  startListening(locale?: string): Promise<void>;
  stopListening(): Promise<void>;
  setOnResultListener(): void;
  removeOnResultListener(): void;
}

function unavailableError() {
  return new Error('VoiceModule is unavailable in this native build.');
}

const nativeVoiceModule = TurboModuleRegistry.get<VoiceSpec>('VoiceModule');

export const voiceModuleAvailable = nativeVoiceModule != null;

const VoiceControl: VoiceSpec =
  nativeVoiceModule ??
  ({
    async configureAudioSession() {
      throw unavailableError();
    },
    async startListening() {
      throw unavailableError();
    },
    async stopListening() {
      throw unavailableError();
    },
    setOnResultListener() {
      throw unavailableError();
    },
    removeOnResultListener() {
      throw unavailableError();
    },
  } as VoiceSpec);

export default VoiceControl;
