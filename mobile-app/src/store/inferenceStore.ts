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
  dispose: () => void;
};

type PendingGeneration = {
  requestId: string;
  resolve: (result: CoreMLGenerationResult) => void;
  reject: (error: Error) => void;
};

type InferenceSetter = (
  partial:
    | Partial<InferenceState>
    | ((state: InferenceState) => Partial<InferenceState>),
) => void;

let unsubscribeNativeEvents: (() => void) | null = null;
let pendingGeneration: PendingGeneration | null = null;
let startingGeneration = false;
const earlyCompletions = new Map<string, CoreMLGenerationResult>();
const earlyErrors = new Map<string, Error>();

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
  if (event.code) {
    Object.assign(error, { code: event.code });
  }
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
      set({
        status,
        error: status.phase === 'error' ? status.detail : null,
      });
    },
    onDownloadProgress: (progress) => {
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
      if (activeRequestId && activeRequestId !== generation.requestId) {
        return;
      }
      set((state) => ({
        activeRequestId: generation.requestId,
        generation,
        status: {
          ...state.status,
          phase: 'generating',
          detail: null,
        },
        error: null,
      }));
    },
    onComplete: (result) => {
      const activeRequestId = get().activeRequestId;
      if (activeRequestId && activeRequestId !== result.requestId) {
        earlyCompletions.set(result.requestId, result);
        return;
      }

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

      if (
        pendingGeneration &&
        pendingGeneration.requestId === result.requestId
      ) {
        const pending = pendingGeneration;
        pendingGeneration = null;
        pending.resolve(result);
      } else if (startingGeneration) {
        earlyCompletions.set(result.requestId, result);
      }
    },
    onError: (event) => {
      const error = nativeEventError(event);
      const activeRequestId = get().activeRequestId;
      if (
        event.requestId &&
        activeRequestId &&
        event.requestId !== activeRequestId
      ) {
        if (startingGeneration) {
          earlyErrors.set(event.requestId, error);
        }
        return;
      }

      set((state) => ({
        activeRequestId: null,
        status: errorStatus(state.status, error),
        error: error.message,
      }));

      if (
        startingGeneration &&
        event.requestId &&
        (!pendingGeneration || pendingGeneration.requestId !== event.requestId)
      ) {
        earlyErrors.set(event.requestId, error);
      }
      rejectPending(error, event.requestId);
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
  earlyCompletions.clear();
  earlyErrors.clear();
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
        try {
          const status = await getCoreMLStatus();
          set({
            status,
            initialized: true,
            error: status.phase === 'error' ? status.detail : null,
          });
          return status;
        } catch (error) {
          const status = errorStatus(get().status, error);
          set({
            status,
            initialized: true,
            error: status.detail,
          });
          return status;
        }
      },
      setBackend: (backend) => {
        set({ backend, error: null });
      },
      prepareModel: async (options = {}) => {
        bindNativeEvents(set, get);
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

        startingGeneration = true;
        set({ generation: null, lastResult: null, error: null });

        let requestId: string;
        try {
          const acknowledgement = await startCoreMLGeneration(request);
          requestId = acknowledgement.requestId;
        } catch (error) {
          startingGeneration = false;
          earlyCompletions.clear();
          earlyErrors.clear();
          const status = errorStatus(get().status, error);
          set({ status, activeRequestId: null, error: status.detail });
          throw error;
        }

        const earlyResult = earlyCompletions.get(requestId);
        if (earlyResult) {
          earlyCompletions.delete(requestId);
          startingGeneration = false;
          return earlyResult;
        }
        const earlyError = earlyErrors.get(requestId);
        if (earlyError) {
          earlyErrors.delete(requestId);
          startingGeneration = false;
          throw earlyError;
        }

        set((state) => ({
          activeRequestId: requestId,
          status: {
            ...state.status,
            phase: 'generating',
            detail: null,
          },
        }));

        return new Promise<CoreMLGenerationResult>((resolve, reject) => {
          pendingGeneration = { requestId, resolve, reject };
          startingGeneration = false;
        });
      },
      cancelGeneration: async (requestId) => {
        const targetRequestId = requestId ?? get().activeRequestId;
        if (!targetRequestId) {
          return false;
        }

        try {
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
      dispose: () => {
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
