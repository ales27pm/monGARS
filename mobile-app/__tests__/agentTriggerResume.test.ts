import { AppState, type AppStateStatus } from 'react-native';
import { subscribeToAgentTriggerResume } from '../src/services/agentTriggerResume';
import { subscribeNativeAgentTriggerHandoff } from '../src/native/agent';

jest.mock('../src/native/agent', () => ({
  nativeAgentModuleAvailable: true,
  subscribeNativeAgentTriggerHandoff: jest.fn(),
}));

describe('agent trigger resume subscription', () => {
  let appStateHandler: ((state: AppStateStatus) => void) | undefined;
  let nativeHandler:
    | ((signal: { id: string; tappedAt: string }) => void)
    | undefined;
  const removeAppState = jest.fn();
  const removeNative = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    appStateHandler = undefined;
    nativeHandler = undefined;
    jest
      .spyOn(AppState, 'addEventListener')
      .mockImplementation((_event, handler) => {
        appStateHandler = handler;
        return { remove: removeAppState } as never;
      });
    (subscribeNativeAgentTriggerHandoff as jest.Mock).mockImplementation(
      (handler) => {
        nativeHandler = handler;
        return removeNative;
      },
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('refreshes on warm active and deduplicates the exact foreground signal', async () => {
    const refresh = jest.fn().mockResolvedValue(undefined);
    const unsubscribe = subscribeToAgentTriggerResume(refresh);

    appStateHandler?.('background');
    appStateHandler?.('active');
    await Promise.resolve();
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(1);

    const signal = {
      id: '33333333-3333-4333-8333-333333333333',
      tappedAt: '2026-07-21T12:00:00.000Z',
    };
    nativeHandler?.(signal);
    await Promise.resolve();
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(2);

    nativeHandler?.(signal);
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(2);

    unsubscribe();
    expect(removeAppState).toHaveBeenCalledTimes(1);
    expect(removeNative).toHaveBeenCalledTimes(1);
  });

  it('queues one persisted reread when a native tap races an AppState fetch', async () => {
    let finishFirst: (() => void) | undefined;
    const refresh = jest
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishFirst = resolve;
          }),
      )
      .mockResolvedValue(undefined);
    subscribeToAgentTriggerResume(refresh);

    appStateHandler?.('inactive');
    appStateHandler?.('active');
    nativeHandler?.({
      id: '33333333-3333-4333-8333-333333333333',
      tappedAt: '2026-07-21T12:00:00.000Z',
    });
    expect(refresh).toHaveBeenCalledTimes(1);

    finishFirst?.();
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(2);
  });
});
