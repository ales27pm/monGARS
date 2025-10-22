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

const VoiceControl = TurboModuleRegistry.getEnforcing<VoiceSpec>('VoiceModule');

export default VoiceControl;
