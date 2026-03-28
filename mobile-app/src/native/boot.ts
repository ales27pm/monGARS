import { AppState, Platform } from 'react-native';
import Diagnostics, { diagnosticsModuleAvailable } from './diagnostics';
import VoiceControl, { voiceModuleAvailable } from './voice';

diagnosticsWarmup();
voiceWarmup();

let appStateSubscription: { remove: () => void } | null = null;

function diagnosticsWarmup() {
  if (!diagnosticsModuleAvailable) {
    return;
  }

  Diagnostics.prepare().catch((error) => {
    console.warn('[Diagnostics] Failed to prepare module', error);
  });

  if (!appStateSubscription) {
    appStateSubscription = AppState.addEventListener('change', (state) => {
      if (state === 'active') {
        Diagnostics.refreshNetworkSnapshot().catch((err) => {
          console.warn('[Diagnostics] Unable to refresh snapshot', err);
        });
      }
    });
  }
}

function voiceWarmup() {
  if (Platform.OS === 'ios' && voiceModuleAvailable) {
    VoiceControl.configureAudioSession().catch((error) => {
      console.warn('[VoiceControl] Failed to prime audio session', error);
    });
  }
}
