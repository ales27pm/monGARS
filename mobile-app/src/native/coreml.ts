import { NativeEventEmitter, NativeModules, Platform } from 'react-native';
import type {
  OnDeviceInferencePhase,
  OnDeviceModelProgress,
  OnDeviceModelStatus,
} from '../types';

export type CoreMLChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type CoreMLGenerationOptions = {
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  doSample?: boolean;
};

export type CoreMLPrepareOptions = Record<string, never>;

export type CoreMLGenerationRequest = {
  messages: CoreMLChatMessage[];
  options?: CoreMLGenerationOptions;
};

export type CoreMLGenerationAcknowledgement = {
  requestId: string;
};

export type CoreMLGenerationUpdate = {
  requestId: string;
  text: string;
  generatedTokens: number;
  tokensPerSecond: number;
  sequence?: number;
};

export type CoreMLGenerationResult = {
  requestId: string;
  text: string;
  promptTokens: number;
  generatedTokens: number;
  duration: number;
  tokensPerSecond: number;
  finishReason: string;
  modelId: string | null;
};

export type CoreMLCancellationResult = {
  cancelled: boolean;
};

export type CoreMLErrorEvent = {
  requestId: string | null;
  code: string | null;
  message: string;
};

type NativeCoreMLInferenceModule = {
  getModelStatus(): Promise<unknown>;
  prepareModel(options: CoreMLPrepareOptions): Promise<unknown>;
  generate(request: CoreMLGenerationRequest): Promise<unknown>;
  cancelGeneration(requestId: string): Promise<unknown>;
  unloadModel(): Promise<unknown>;
  deleteModel(): Promise<unknown>;
  addListener(eventName: string): void;
  removeListeners(count: number): void;
};

export const COREML_EVENTS = {
  status: 'onCoreMLStatus',
  downloadProgress: 'onCoreMLDownloadProgress',
  generation: 'onCoreMLGeneration',
  complete: 'onCoreMLComplete',
  error: 'onCoreMLError',
} as const;

const INFERENCE_PHASES = new Set<OnDeviceInferencePhase>([
  'unavailable',
  'not-downloaded',
  'downloading',
  'verifying',
  'loading',
  'ready',
  'generating',
  'error',
]);

function isNativeModule(value: unknown): value is NativeCoreMLInferenceModule {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Partial<NativeCoreMLInferenceModule>;
  return (
    typeof candidate.getModelStatus === 'function' &&
    typeof candidate.prepareModel === 'function' &&
    typeof candidate.generate === 'function' &&
    typeof candidate.cancelGeneration === 'function' &&
    typeof candidate.unloadModel === 'function' &&
    typeof candidate.deleteModel === 'function'
  );
}

const linkedModule = NativeModules.CoreMLInferenceModule as unknown;
const nativeCoreMLModule =
  Platform.OS === 'ios' && isNativeModule(linkedModule) ? linkedModule : null;

export const coreMLModuleAvailable = nativeCoreMLModule !== null;

export class CoreMLUnavailableError extends Error {
  readonly code = 'COREML_MODULE_UNAVAILABLE';

  constructor() {
    super(
      Platform.OS === 'ios'
        ? "Le module d'inférence Core ML n'est pas lié à cette version de l'application."
        : "L'inférence Core ML locale est disponible uniquement sur iOS.",
    );
    this.name = 'CoreMLUnavailableError';
  }
}

export class CoreMLBusyError extends Error {
  readonly code = 'COREML_GENERATION_IN_PROGRESS';

  constructor() {
    super('Une génération locale est déjà en cours.');
    this.name = 'CoreMLBusyError';
  }
}

export function unavailableCoreMLStatus(): OnDeviceModelStatus {
  return {
    phase: 'unavailable',
    modelId: null,
    displayName: null,
    revision: null,
    installedBytes: 0,
    contextLength: 0,
    minimumIOSVersion: 18,
    detail: new CoreMLUnavailableError().message,
  };
}

function requireNativeModule(): NativeCoreMLInferenceModule {
  if (!nativeCoreMLModule) {
    throw new CoreMLUnavailableError();
  }
  return nativeCoreMLModule;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : {};
}

function nullableString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function inferencePhase(value: unknown): OnDeviceInferencePhase {
  return typeof value === 'string' &&
    INFERENCE_PHASES.has(value as OnDeviceInferencePhase)
    ? (value as OnDeviceInferencePhase)
    : 'error';
}

function requiredString(
  value: unknown,
  field: string,
  eventName: string,
): string {
  if (typeof value === 'string' && value.length > 0) {
    return value;
  }
  throw new Error(`${eventName}: champ ${field} absent.`);
}

function normalizeStatus(value: unknown): OnDeviceModelStatus {
  const record = asRecord(value);
  const phase = inferencePhase(record.phase);
  const nativeDetail = nullableString(record.detail);

  return {
    phase,
    modelId: nullableString(record.modelId ?? record.modelID),
    displayName: nullableString(record.displayName),
    revision: nullableString(record.revision),
    installedBytes: Math.max(0, finiteNumber(record.installedBytes)),
    contextLength: Math.max(0, finiteNumber(record.contextLength)),
    minimumIOSVersion: Math.max(0, finiteNumber(record.minimumIOSVersion, 18)),
    detail:
      nativeDetail ??
      (phase === 'error' ? 'Réponse de statut Core ML invalide.' : null),
  };
}

function normalizeProgress(value: unknown): OnDeviceModelProgress {
  const record = asRecord(value);
  const fractionCompleted = finiteNumber(record.fractionCompleted);
  const bytesPerSecond = finiteNumber(record.bytesPerSecond, Number.NaN);

  return {
    phase: inferencePhase(record.phase),
    fractionCompleted: Math.min(Math.max(fractionCompleted, 0), 1),
    bytesPerSecond: Number.isFinite(bytesPerSecond) ? bytesPerSecond : null,
    detail: nullableString(record.detail),
  };
}

function normalizeGenerationUpdate(value: unknown): CoreMLGenerationUpdate {
  const record = asRecord(value);
  const sequence = finiteNumber(record.sequence ?? record.seq, Number.NaN);
  return {
    requestId: requiredString(
      record.requestId,
      'requestId',
      COREML_EVENTS.generation,
    ),
    text: typeof record.text === 'string' ? record.text : '',
    generatedTokens: Math.max(0, finiteNumber(record.generatedTokens)),
    tokensPerSecond: Math.max(0, finiteNumber(record.tokensPerSecond)),
    ...(Number.isFinite(sequence) ? { sequence } : {}),
  };
}

function normalizeGenerationResult(value: unknown): CoreMLGenerationResult {
  const record = asRecord(value);
  return {
    requestId: requiredString(
      record.requestId,
      'requestId',
      COREML_EVENTS.complete,
    ),
    text: typeof record.text === 'string' ? record.text : '',
    promptTokens: Math.max(0, finiteNumber(record.promptTokens)),
    generatedTokens: Math.max(0, finiteNumber(record.generatedTokens)),
    duration: Math.max(0, finiteNumber(record.duration)),
    tokensPerSecond: Math.max(0, finiteNumber(record.tokensPerSecond)),
    finishReason:
      typeof record.finishReason === 'string' ? record.finishReason : 'unknown',
    modelId: nullableString(record.modelId ?? record.modelID),
  };
}

function normalizeErrorEvent(value: unknown): CoreMLErrorEvent {
  const record = asRecord(value);
  return {
    requestId: nullableString(record.requestId),
    code: nullableString(record.code),
    message:
      nullableString(record.message) ??
      nullableString(record.detail) ??
      "Erreur inconnue du moteur d'inférence Core ML.",
  };
}

function normalizeAcknowledgement(
  value: unknown,
): CoreMLGenerationAcknowledgement {
  const record = asRecord(value);
  return {
    requestId: requiredString(record.requestId, 'requestId', 'generate'),
  };
}

function normalizeCancellation(value: unknown): CoreMLCancellationResult {
  if (typeof value === 'boolean') {
    return { cancelled: value };
  }
  const record = asRecord(value);
  return { cancelled: record.cancelled === true };
}

export type CoreMLEventListeners = {
  onStatus?: (status: OnDeviceModelStatus) => void;
  onDownloadProgress?: (progress: OnDeviceModelProgress) => void;
  onGeneration?: (update: CoreMLGenerationUpdate) => void;
  onComplete?: (result: CoreMLGenerationResult) => void;
  onError?: (error: CoreMLErrorEvent) => void;
};

let eventEmitter: NativeEventEmitter | null = null;

function getEventEmitter(): NativeEventEmitter | null {
  if (!nativeCoreMLModule) {
    return null;
  }
  eventEmitter ??= new NativeEventEmitter(nativeCoreMLModule);
  return eventEmitter;
}

export function subscribeToCoreMLEvents(
  listeners: CoreMLEventListeners,
): () => void {
  const emitter = getEventEmitter();
  if (!emitter) {
    return () => undefined;
  }

  const reportMalformedEvent = (error: unknown) => {
    listeners.onError?.({
      requestId: null,
      code: 'COREML_INVALID_EVENT',
      message: error instanceof Error ? error.message : String(error),
    });
  };

  const subscriptions = [
    emitter.addListener(COREML_EVENTS.status, (value: unknown) => {
      listeners.onStatus?.(normalizeStatus(value));
    }),
    emitter.addListener(COREML_EVENTS.downloadProgress, (value: unknown) => {
      listeners.onDownloadProgress?.(normalizeProgress(value));
    }),
    emitter.addListener(COREML_EVENTS.generation, (value: unknown) => {
      try {
        listeners.onGeneration?.(normalizeGenerationUpdate(value));
      } catch (error) {
        reportMalformedEvent(error);
      }
    }),
    emitter.addListener(COREML_EVENTS.complete, (value: unknown) => {
      try {
        listeners.onComplete?.(normalizeGenerationResult(value));
      } catch (error) {
        reportMalformedEvent(error);
      }
    }),
    emitter.addListener(COREML_EVENTS.error, (value: unknown) => {
      listeners.onError?.(normalizeErrorEvent(value));
    }),
  ];

  return () => {
    subscriptions.forEach((subscription) => subscription.remove());
  };
}

export async function getCoreMLStatus(): Promise<OnDeviceModelStatus> {
  if (!nativeCoreMLModule) {
    return unavailableCoreMLStatus();
  }
  return normalizeStatus(await nativeCoreMLModule.getModelStatus());
}

export async function prepareCoreMLModel(
  options: CoreMLPrepareOptions = {},
): Promise<OnDeviceModelStatus> {
  return normalizeStatus(await requireNativeModule().prepareModel(options));
}

export async function startCoreMLGeneration(
  request: CoreMLGenerationRequest,
): Promise<CoreMLGenerationAcknowledgement> {
  return normalizeAcknowledgement(
    await requireNativeModule().generate(request),
  );
}

export async function cancelCoreMLGeneration(
  requestId: string,
): Promise<CoreMLCancellationResult> {
  return normalizeCancellation(
    await requireNativeModule().cancelGeneration(requestId),
  );
}

export async function unloadCoreMLModel(): Promise<OnDeviceModelStatus> {
  const status = await requireNativeModule().unloadModel();
  return status == null ? getCoreMLStatus() : normalizeStatus(status);
}

export async function deleteCoreMLModel(): Promise<OnDeviceModelStatus> {
  const status = await requireNativeModule().deleteModel();
  return status == null ? getCoreMLStatus() : normalizeStatus(status);
}

const CoreML = {
  getStatus: getCoreMLStatus,
  prepareModel: prepareCoreMLModel,
  generate: startCoreMLGeneration,
  cancelGeneration: cancelCoreMLGeneration,
  unloadModel: unloadCoreMLModel,
  deleteModel: deleteCoreMLModel,
  subscribe: subscribeToCoreMLEvents,
};

export default CoreML;
