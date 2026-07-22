import { AppState, type AppStateStatus } from 'react-native';
import {
  nativeAgentModuleAvailable,
  subscribeNativeAgentTriggerHandoff,
} from '../native/agent';

type RefreshTrigger = () => Promise<void>;

/**
 * Refreshes the owner-scoped persisted handoff on warm resume and on the
 * deterministic native notification-tap signal. It never executes a trigger;
 * Run/Ignore remains an explicit user decision in ChatScreen.
 */
export function subscribeToAgentTriggerResume(
  refresh: RefreshTrigger,
): () => void {
  let previousState: AppStateStatus = AppState.currentState;
  let disposed = false;
  let inFlight: Promise<void> | null = null;
  let nativeRefreshQueued = false;
  let lastNativeSignal: string | null = null;

  const requestRefresh = (source: 'app-state' | 'native') => {
    if (disposed) {
      return;
    }
    if (inFlight) {
      // A native signal is posted only after the handoff was persisted. If an
      // earlier AppState fetch is racing it, one follow-up read is required.
      nativeRefreshQueued ||= source === 'native';
      return;
    }
    inFlight = refresh()
      .catch((error) => {
        console.debug('[ChatScreen] trigger refresh failed', error);
      })
      .finally(() => {
        inFlight = null;
        if (nativeRefreshQueued && !disposed) {
          nativeRefreshQueued = false;
          requestRefresh('native');
        }
      });
  };

  const appStateSubscription = AppState.addEventListener('change', (next) => {
    const becameActive = next === 'active' && previousState !== 'active';
    previousState = next;
    if (becameActive) {
      requestRefresh('app-state');
    }
  });

  let removeNativeListener: () => void = () => {};
  if (nativeAgentModuleAvailable) {
    try {
      removeNativeListener = subscribeNativeAgentTriggerHandoff((signal) => {
        const key = `${signal.id}:${signal.tappedAt}`;
        if (key === lastNativeSignal) {
          return;
        }
        lastNativeSignal = key;
        requestRefresh('native');
      });
    } catch (error) {
      console.debug('[ChatScreen] trigger signal unavailable', error);
    }
  }

  return () => {
    disposed = true;
    appStateSubscription.remove();
    removeNativeListener();
  };
}
