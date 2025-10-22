import 'react-native-gesture-handler';
import { AppState, Platform } from 'react-native';
import Diagnostics from './diagnostics';
import VoiceControl from './voice';

diagnosticsWarmup();
voiceWarmup();

function diagnosticsWarmup() {
  Diagnostics.prepare().catch((error) => {
    console.warn('[Diagnostics] Failed to prepare module', error);
  });

  AppState.addEventListener('change', (state) => {
    if (state === 'active') {
      Diagnostics.refreshNetworkSnapshot().catch((err) => {
        console.warn('[Diagnostics] Unable to refresh snapshot', err);
      });
    }
  });
}

function voiceWarmup() {
  if (Platform.OS === 'ios') {
    VoiceControl.configureAudioSession().catch((error) => {
      console.warn('[VoiceControl] Failed to prime audio session', error);
    });
  }
}
