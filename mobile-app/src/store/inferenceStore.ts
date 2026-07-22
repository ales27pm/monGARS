import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import {
  CoreMLBusyError,
  CoreMLUnavailableError,
  cancelCoreMLGeneration,
  deleteCoreMLModel,
  getCoreMLStatus,
  prepareCoreMLModel,
  startCoreMLGeneration,
  subscribeToCoreMLEvents,
  unavailableCoreMLStatus,
  unloadCoreMLModel,
} from '../native/coreml';
import type {
  CoreMLErrorEvent,
  CoreMLGenerationRequest,
  CoreMLGenerationResult,
  CoreMLGenerationUpdate,
  CoreMLPrepareOptions,
} from '../native/coreml';
import type {
  InferenceBackend,
  OnDeviceModelProgress,
  OnDeviceModelStatus,
} from '../types';

export type InferenceState = {
  backend: InferenceBackend;
  status: OnDeviceModelStatus;
  progress: OnDeviceModelProgress | null;
  initialized: boolean;
  activeRequestId: string | null;
  generation: CoreMLGenerationUpdate | null;
  lastResult: CoreMLGenerationResult | null;
  error: string | null;
  initialize: () => Promise<OnDeviceModelStatus>;
  setBackend: (backend: InferenceBackend) => void;
  prepareModel: (
    options?: CoreMLPrepareOptions,
  ) => Promise<OnDeviceModelStatus>;
  generate: (
    request: CoreMLGenerationRequest,
  ) => Promise<CoreMLGenerationResult>;
  cancelGeneration: (requestId?: string) => Promise<boolean>;
  unloadModel: () => Promise<OnDeviceModelStatus>;
  deleteModel: () => Promise<OnDeviceModelStatus>;
  clearError: () => void;
  dispose: () => Promise<void>;
};

type PendingGeneration = {
  requestId: string;
  resolve: (result: CoreMLGenerationResult) => void;
  reject: (error: Error) => void;
};

type LifecycleSignal<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

type GenerationLifecycle = {
  startSignal: LifecycleSignal<string | null>;
  terminalSignal: LifecycleSignal<void>;
};

type InferenceSetter = (
  partial:
    | Partial<InferenceState>
    | ((state: InferenceState) => Partial<InferenceState>),
) => void;

let unsubscribeNativeEvents: (() => void) | null = null;
let pendingGeneration: PendingGeneration | null = null;
let startingGeneration = false;
let cancelRequestedWhileStarting = false;
let statusRefreshVersion = 0;
let generationEpoch = 0;
let generationStartSignal: LifecycleSignal<string | null> | null = null;
let generationTerminalSignal: LifecycleSignal<void> | null = null;
const earlyUpdates = new Map<string, CoreMLGenerationUpdate>();
const earlyCompletions = new Map<string, CoreMLGenerationResult>();
const earlyErrors = new Map<string, Error>();
let earlyUnscopedError: Error | null = null;
let inferenceHydrationInFlight: Promise<void> | null = null;

const DISPOSE_WAIT_TIMEOUT_MS = 5_000;

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return "Erreur inconnue du moteur d'inférence locale.";
}

function nativeEventError(event: CoreMLErrorEvent): Error {
  const error = new Error(event.message);
  error.name = 'CoreMLNativeError';
  Object.assign(error, {
    ...(event.code ? { code: event.code } : {}),
    recoverable: event.recoverable,
  });
  return error;
}

function errorStatus(
  current: OnDeviceModelStatus,
  error: unknown,
): OnDeviceModelStatus {
  return {
    ...current,
    phase: error instanceof CoreMLUnavailableError ? 'unavailable' : 'error',
    detail: errorMessage(error),
  };
}

function invalidateStatusRefresh() {
  statusRefreshVersion += 1;
}

function createLifecycleSignal<T>(): LifecycleSignal<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function beginGenerationLifecycle(): GenerationLifecycle {
  const lifecycle = {
    startSignal: createLifecycleSignal<string | null>(),
    terminalSignal: createLifecycleSignal<void>(),
  };
  generationStartSignal = lifecycle.startSignal;
  generationTerminalSignal = lifecycle.terminalSignal;
  return lifecycle;
}

function markGenerationStarted(
  requestId: string | null,
  lifecycle?: GenerationLifecycle,
) {
  const startSignal = lifecycle?.startSignal ?? generationStartSignal;
  startSignal?.resolve(requestId);
  if (generationStartSignal === startSignal) {
    generationStartSignal = null;
  }
}

function finishGenerationLifecycle(lifecycle?: GenerationLifecycle) {
  const terminalSignal = lifecycle?.terminalSignal ?? generationTerminalSignal;
  terminalSignal?.resolve();
  if (generationTerminalSignal === terminalSignal) {
    generationTerminalSignal = null;
  }
  markGenerationStarted(null, lifecycle);
}

async function waitAtMost(
  promise: Promise<unknown>,
  timeoutMilliseconds: number,
): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  await Promise.race([
    promise,
    new Promise<void>((resolve) => {
      timeout = setTimeout(resolve, timeoutMilliseconds);
    }),
  ]);
  if (timeout) {
    clearTimeout(timeout);
  }
}

function isNewerGenerationUpdate(
  current: CoreMLGenerationUpdate | null,
  next: CoreMLGenerationUpdate,
): boolean {
  if (!current || current.requestId !== next.requestId) {
    return true;
  }

  if (current.sequence !== undefined && next.sequence !== undefined) {
    return next.sequence > current.sequence;
  }
  if (next.generatedTokens < current.generatedTokens) {
    return false;
  }
  if (
    next.generatedTokens === current.generatedTokens &&
    next.text.length < current.text.length
  ) {
    return false;
  }
  return (
    next.text !== current.text ||
    next.generatedTokens !== current.generatedTokens ||
    next.tokensPerSecond !== current.tokensPerSecond
  );
}

function clearEarlyEvents() {
  earlyUpdates.clear();
  earlyCompletions.clear();
  earlyErrors.clear();
  earlyUnscopedError = null;
}

function applyGenerationUpdate(
  set: InferenceSetter,
  generation: CoreMLGenerationUpdate,
) {
  invalidateStatusRefresh();
  set((state) => {
    if (!isNewerGenerationUpdate(state.generation, generation)) {
      return {};
    }
    return {
      activeRequestId: generation.requestId,
      generation,
      status: {
        ...state.status,
        phase: 'generating',
        detail: null,
      },
      error: null,
    };
  });
}

function applyGenerationCompletion(
  set: InferenceSetter,
  result: CoreMLGenerationResult,
) {
  invalidateStatusRefresh();
  set((state) => ({
    activeRequestId: null,
    generation: {
      requestId: result.requestId,
      text: result.text,
      generatedTokens: result.generatedTokens,
      tokensPerSecond: result.tokensPerSecond,
    },
    lastResult: result,
    status: {
      ...state.status,
      phase: 'ready',
      detail: null,
    },
    error: null,
  }));
}

function rejectPending(error: Error, requestId?: string | null) {
  if (
    pendingGeneration &&
    (!requestId || pendingGeneration.requestId === requestId)
  ) {
    const pending = pendingGeneration;
    pendingGeneration = null;
    pending.reject(error);
  }
}

function bindNativeEvents(set: InferenceSetter, get: () => InferenceState) {
  if (unsubscribeNativeEvents) {
    return;
  }

  unsubscribeNativeEvents = subscribeToCoreMLEvents({
    onStatus: (status) => {
      invalidateStatusRefresh();
      set({
        status,
        error: status.phase === 'error' ? status.detail : null,
      });
    },
    onDownloadProgress: (progress) => {
      invalidateStatusRefresh();
      set((state) => ({
        progress,
        status: {
          ...state.status,
          phase: progress.phase,
          detail: progress.detail,
        },
        error: null,
      }));
    },
    onGeneration: (generation) => {
      const activeRequestId = get().activeRequestId;
      if (!activeRequestId) {
        if (startingGeneration) {
          const current = earlyUpdates.get(generation.requestId) ?? null;
          if (isNewerGenerationUpdate(current, generation)) {
            earlyUpdates.set(generation.requestId, generation);
          }
        }
        return;
      }
      if (activeRequestId !== generation.requestId) {
        return;
      }
      applyGenerationUpdate(set, generation);
    },
    onComplete: (result) => {
      const activeRequestId = get().activeRequestId;
      if (!activeRequestId) {
        if (startingGeneration) {
          earlyCompletions.set(result.requestId, result);
        }
        return;
      }
      if (activeRequestId !== result.requestId) {
        return;
      }

      applyGenerationCompletion(set, result);

      if (
        pendingGeneration &&
        pendingGeneration.requestId === result.requestId
      ) {
        const pending = pendingGeneration;
        pendingGeneration = null;
        pending.resolve(result);
      }
      finishGenerationLifecycle();
    },
    onError: (event) => {
      const error = nativeEventError(event);
      const activeRequestId = get().activeRequestId;
      if (event.requestId && !activeRequestId) {
        if (startingGeneration) {
          earlyErrors.set(event.requestId, error);
        }
        return;
      }
      if (event.requestId && event.requestId !== activeRequestId) {
        return;
      }
      if (!event.requestId && startingGeneration && !activeRequestId) {
        earlyUnscopedError = error;
        return;
      }

      invalidateStatusRefresh();
      set((state) => ({
        activeRequestId: null,
        status: errorStatus(state.status, error),
        error: error.message,
      }));

      rejectPending(error, event.requestId);
      if (activeRequestId || startingGeneration) {
        finishGenerationLifecycle();
      }
    },
  });
}

function clearPendingGeneration(error: Error) {
  if (pendingGeneration) {
    const pending = pendingGeneration;
    pendingGeneration = null;
    pending.reject(error);
  }
  startingGeneration = false;
  cancelRequestedWhileStarting = false;
  clearEarlyEvents();
  finishGenerationLifecycle();
}

const INITIAL_STATUS = unavailableCoreMLStatus();

export const useInferenceStore = create<InferenceState>()(
  persist(
    (set, get) => ({
      backend: 'server',
      status: INITIAL_STATUS,
      progress: null,
      initialized: false,
      activeRequestId: null,
      generation: null,
      lastResult: null,
      error: null,
      initialize: async () => {
        bindNativeEvents(set, get);
        const refreshVersion = statusRefreshVersion;
        try {
          const status = await getCoreMLStatus();
          if (refreshVersion === statusRefreshVersion) {
            set({
              status,
              initialized: true,
              error: status.phase === 'error' ? status.detail : null,
            });
          }
          return status;
        } catch (error) {
          const status = errorStatus(get().status, error);
          if (refreshVersion === statusRefreshVersion) {
            set({
              status,
              initialized: true,
              error: status.detail,
            });
          }
          return status;
        }
      },
      setBackend: (backend) => {
        if (
          backend !== get().backend &&
          (startingGeneration ||
            pendingGeneration !== null ||
            get().activeRequestId !== null)
        ) {
          throw new CoreMLBusyError();
        }
        set({ backend, error: null });
      },
      prepareModel: async (options = {}) => {
        bindNativeEvents(set, get);
        invalidateStatusRefresh();
        set({ progress: null, error: null });
        try {
          const status = await prepareCoreMLModel(options);
          set({ status, progress: null, initialized: true, error: null });
          return status;
        } catch (error) {
          const status = errorStatus(get().status, error);
          set({ status, initialized: true, error: status.detail });
          throw error;
        }
      },
      generate: async (request) => {
        bindNativeEvents(set, get);
        if (startingGeneration || pendingGeneration || get().activeRequestId) {
          throw new CoreMLBusyError();
        }

        const operationEpoch = ++generationEpoch;
        startingGeneration = true;
        cancelRequestedWhileStarting = false;
        const lifecycle = beginGenerationLifecycle();
        invalidateStatusRefresh();
        clearEarlyEvents();
        set({ generation: null, lastResult: null, error: null });

        let requestId: string;
        try {
          const acknowledgement = await startCoreMLGeneration(request);
          requestId = acknowledgement.requestId;
          markGenerationStarted(requestId, lifecycle);
        } catch (error) {
          markGenerationStarted(null, lifecycle);
          if (operationEpoch !== generationEpoch) {
            finishGenerationLifecycle(lifecycle);
            throw new Error("Le moteur d'inférence locale a été arrêté.");
          }
          startingGeneration = false;
          cancelRequestedWhileStarting = false;
          clearEarlyEvents();
          const status = errorStatus(get().status, error);
          set({ status, activeRequestId: null, error: status.detail });
          finishGenerationLifecycle(lifecycle);
          throw error;
        }

        if (operationEpoch !== generationEpoch) {
          const stoppedError = new Error(
            "Le moteur d'inférence locale a été arrêté.",
          );
          try {
            await cancelCoreMLGeneration(requestId);
          } catch {
            // Teardown is already in progress; the native bridge will also
            // cancel its operation when React Native invalidates the module.
          }
          finishGenerationLifecycle(lifecycle);
          throw stoppedError;
        }

        const earlyError = earlyErrors.get(requestId) ?? earlyUnscopedError;
        if (earlyError) {
          startingGeneration = false;
          cancelRequestedWhileStarting = false;
          clearEarlyEvents();
          const status = errorStatus(get().status, earlyError);
          set({ status, activeRequestId: null, error: status.detail });
          finishGenerationLifecycle(lifecycle);
          throw earlyError;
        }
        const earlyResult = earlyCompletions.get(requestId);
        if (earlyResult) {
          startingGeneration = false;
          cancelRequestedWhileStarting = false;
          clearEarlyEvents();
          applyGenerationCompletion(set, earlyResult);
          finishGenerationLifecycle(lifecycle);
          return earlyResult;
        }
        const earlyUpdate = earlyUpdates.get(requestId) ?? null;
        const shouldCancel = cancelRequestedWhileStarting;
        cancelRequestedWhileStarting = false;
        clearEarlyEvents();

        set((state) => ({
          activeRequestId: requestId,
          ...(earlyUpdate ? { generation: earlyUpdate } : {}),
          status: {
            ...state.status,
            phase: 'generating',
            detail: null,
          },
        }));

        return new Promise<CoreMLGenerationResult>((resolve, reject) => {
          pendingGeneration = { requestId, resolve, reject };
          startingGeneration = false;

          if (shouldCancel) {
            cancelCoreMLGeneration(requestId).catch((error) => {
              if (get().activeRequestId !== requestId) {
                return;
              }
              const cancellationError =
                error instanceof Error ? error : new Error(errorMessage(error));
              invalidateStatusRefresh();
              set((state) => ({
                activeRequestId: null,
                status: errorStatus(state.status, cancellationError),
                error: cancellationError.message,
              }));
              rejectPending(cancellationError, requestId);
              finishGenerationLifecycle(lifecycle);
            });
          }
        });
      },
      cancelGeneration: async (requestId) => {
        const targetRequestId = requestId ?? get().activeRequestId;
        if (!targetRequestId) {
          if (startingGeneration) {
            cancelRequestedWhileStarting = true;
            set((state) => ({
              status: {
                ...state.status,
                phase: 'generating',
                detail: 'Annulation demandée…',
              },
            }));
            return true;
          }
          return false;
        }

        try {
          invalidateStatusRefresh();
          const result = await cancelCoreMLGeneration(targetRequestId);
          if (result.cancelled) {
            set((state) => ({
              status: {
                ...state.status,
                phase: 'generating',
                detail: 'Annulation de la génération locale…',
              },
            }));
          }
          return result.cancelled;
        } catch (error) {
          const status = errorStatus(get().status, error);
          set({ status, error: status.detail });
          throw error;
        }
      },
      unloadModel: async () => {
        invalidateStatusRefresh();
        if (get().activeRequestId) {
          await get().cancelGeneration();
        }
        try {
          const status = await unloadCoreMLModel();
          set({
            status,
            generation: null,
            lastResult: null,
            error: null,
          });
          return status;
        } catch (error) {
          const status = errorStatus(get().status, error);
          set({ status, error: status.detail });
          throw error;
        }
      },
      deleteModel: async () => {
        invalidateStatusRefresh();
        if (get().activeRequestId) {
          await get().cancelGeneration();
        }
        try {
          const status = await deleteCoreMLModel();
          set({
            status,
            progress: null,
            generation: null,
            lastResult: null,
            error: null,
          });
          return status;
        } catch (error) {
          const status = errorStatus(get().status, error);
          set({ status, error: status.detail });
          throw error;
        }
      },
      clearError: () => set({ error: null }),
      dispose: async () => {
        invalidateStatusRefresh();
        generationEpoch += 1;
        cancelRequestedWhileStarting = false;

        const startPromise = generationStartSignal?.promise ?? null;
        const terminalPromise = generationTerminalSignal?.promise ?? null;
        if (startingGeneration && startPromise) {
          await waitAtMost(startPromise, DISPOSE_WAIT_TIMEOUT_MS);
        }

        const requestId = get().activeRequestId;
        if (requestId) {
          try {
            await cancelCoreMLGeneration(requestId);
          } catch {
            // Continue teardown even if native cancellation cannot be
            // acknowledged; bridge invalidation is the final safety net.
          }
        }
        if (terminalPromise) {
          await waitAtMost(terminalPromise, DISPOSE_WAIT_TIMEOUT_MS);
        }

        unsubscribeNativeEvents?.();
        unsubscribeNativeEvents = null;
        clearPendingGeneration(
          new Error("Le moteur d'inférence locale a été arrêté."),
        );
        set({
          initialized: false,
          activeRequestId: null,
          generation: null,
        });
      },
    }),
    {
      name: 'mongars-inference',
      storage: createJSONStorage(() => AsyncStorage),
      version: 1,
      migrate: (persistedState) => {
        const state = persistedState as Partial<InferenceState> | undefined;
        return {
          backend: state?.backend === 'on-device' ? 'on-device' : 'server',
        } as InferenceState;
      },
      partialize: (state) => ({
        backend: state.backend,
      }),
    },
  ),
);

export async function ensureInferenceStoreHydrated(): Promise<void> {
  if (useInferenceStore.persist.hasHydrated()) {
    return;
  }
  if (!inferenceHydrationInFlight) {
    const hydration = Promise.resolve(
      useInferenceStore.persist.rehydrate(),
    ).finally(() => {
      if (inferenceHydrationInFlight === hydration) {
        inferenceHydrationInFlight = null;
      }
    });
    inferenceHydrationInFlight = hydration;
  }
  await inferenceHydrationInFlight;
}
