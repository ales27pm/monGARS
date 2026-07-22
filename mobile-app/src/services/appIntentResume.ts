import { AppState } from 'react-native';
import {
  nativeAppIntentModuleAvailable,
  subscribeNativeAppIntentHandoff,
} from '../native/appIntents';

export function subscribeToAppIntentResume(
  refresh: () => Promise<void>,
): () => void {
  if (!nativeAppIntentModuleAvailable) {
    return () => undefined;
  }

  const refreshSafely = () => {
    refresh().catch((error) =>
      console.debug('[appIntentResume] handoff refresh failed', error),
    );
  };
  const unsubscribeNative = subscribeNativeAppIntentHandoff(refreshSafely);
  const appStateSubscription = AppState.addEventListener('change', (state) => {
    if (state === 'active') {
      refreshSafely();
    }
  });
  return () => {
    unsubscribeNative();
    appStateSubscription.remove();
  };
}
