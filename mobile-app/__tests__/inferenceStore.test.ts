import * as CoreML from '../src/native/coreml';
import type { CoreMLGenerationResult } from '../src/native/coreml';
import type { OnDeviceModelStatus } from '../src/types';
import { useInferenceStore } from '../src/store/inferenceStore';

let mockCoreMLEventListeners: CoreML.CoreMLEventListeners | null = null;
const mockUnsubscribe = jest.fn();
const mockGetCoreMLStatus = jest.spyOn(CoreML, 'getCoreMLStatus');
const mockPrepareCoreMLModel = jest.spyOn(CoreML, 'prepareCoreMLModel');
const mockStartCoreMLGeneration = jest.spyOn(CoreML, 'startCoreMLGeneration');
const mockCancelCoreMLGeneration = jest.spyOn(CoreML, 'cancelCoreMLGeneration');
const mockUnloadCoreMLModel = jest.spyOn(CoreML, 'unloadCoreMLModel');
const mockDeleteCoreMLModel = jest.spyOn(CoreML, 'deleteCoreMLModel');
const mockSubscribeToCoreMLEvents = jest.spyOn(
  CoreML,
  'subscribeToCoreMLEvents',
);

const unavailableStatus = (): OnDeviceModelStatus => ({
  phase: 'unavailable',
  modelId: null,
  displayName: null,
  revision: null,
  installedBytes: 0,
  contextLength: 0,
  minimumIOSVersion: 18,
  detail: 'Module indisponible',
});

const readyStatus = (): OnDeviceModelStatus => ({
  phase: 'ready',
  modelId: 'example/model',
  displayName: 'Example Core ML',
  revision: 'revision-1',
  installedBytes: 1024,
  contextLength: 512,
  minimumIOSVersion: 18,
  detail: null,
});

function listeners(): CoreML.CoreMLEventListeners {
  if (!mockCoreMLEventListeners) {
    throw new Error('Les événements Core ML ne sont pas abonnés.');
  }
  return mockCoreMLEventListeners;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

describe('inferenceStore', () => {
  beforeEach(async () => {
    await useInferenceStore.getState().dispose();
    jest.clearAllMocks();
    mockCoreMLEventListeners = null;
    mockSubscribeToCoreMLEvents.mockImplementation((nextListeners) => {
      mockCoreMLEventListeners = nextListeners;
      return mockUnsubscribe;
    });
    mockGetCoreMLStatus.mockResolvedValue(unavailableStatus());
    mockPrepareCoreMLModel.mockResolvedValue(readyStatus());
    mockStartCoreMLGeneration.mockResolvedValue({ requestId: 'request-1' });
    mockCancelCoreMLGeneration.mockResolvedValue({ cancelled: true });
    mockUnloadCoreMLModel.mockResolvedValue(readyStatus());
    mockDeleteCoreMLModel.mockResolvedValue({
      ...readyStatus(),
      phase: 'not-downloaded',
      installedBytes: 0,
    });

    useInferenceStore.setState({
      backend: 'server',
      status: unavailableStatus(),
      progress: null,
      initialized: false,
      activeRequestId: null,
      generation: null,
      lastResult: null,
      error: null,
    });
  });

  it('defaults to server mode and initializes safely when Core ML is unavailable', async () => {
    const status = await useInferenceStore.getState().initialize();

    expect(useInferenceStore.getState().backend).toBe('server');
    expect(status.phase).toBe('unavailable');
    expect(useInferenceStore.getState().initialized).toBe(true);
    expect(mockSubscribeToCoreMLEvents).toHaveBeenCalledTimes(1);
  });

  it('tracks download progress and the prepared model status', async () => {
    const preparation = useInferenceStore.getState().prepareModel();
    listeners().onDownloadProgress?.({
      phase: 'downloading',
      fractionCompleted: 0.25,
      bytesPerSecond: 2048,
      detail: 'Téléchargement',
    });

    expect(useInferenceStore.getState().progress?.fractionCompleted).toBe(0.25);
    await expect(preparation).resolves.toMatchObject({ phase: 'ready' });
    expect(useInferenceStore.getState().status.phase).toBe('ready');
    expect(useInferenceStore.getState().progress).toBeNull();
  });

  it('streams cumulative text and resolves generation on the completion event', async () => {
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Bonjour' }],
    });
    await Promise.resolve();

    listeners().onGeneration?.({
      requestId: 'request-1',
      text: 'Bon',
      generatedTokens: 1,
      tokensPerSecond: 4,
      sequence: 1,
    });
    expect(useInferenceStore.getState().generation?.text).toBe('Bon');

    const result: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: 'Bonjour !',
      promptTokens: 12,
      generatedTokens: 3,
      duration: 0.75,
      tokensPerSecond: 4,
      finishReason: 'eos',
      modelId: 'example/model',
    };
    listeners().onComplete?.(result);

    await expect(generation).resolves.toEqual(result);
    expect(useInferenceStore.getState()).toMatchObject({
      activeRequestId: null,
      lastResult: result,
      error: null,
    });
    expect(useInferenceStore.getState().status.phase).toBe('ready');
  });

  it('keeps the generation promise alive until native cancellation completes', async () => {
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Arrête-toi' }],
    });
    await Promise.resolve();

    await expect(useInferenceStore.getState().cancelGeneration()).resolves.toBe(
      true,
    );
    expect(useInferenceStore.getState().status.detail).toContain('Annulation');

    const cancelledResult: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: 'Réponse partielle',
      promptTokens: 0,
      generatedTokens: 2,
      duration: 0.5,
      tokensPerSecond: 4,
      finishReason: 'cancelled',
      modelId: 'example/model',
    };
    listeners().onComplete?.(cancelledResult);

    await expect(generation).resolves.toEqual(cancelledResult);
    expect(useInferenceStore.getState().activeRequestId).toBeNull();
  });

  it('cancels an active native request before disposing its event listeners', async () => {
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Continue' }],
    });
    await flushPromises();

    const disposal = useInferenceStore.getState().dispose();
    await flushPromises();
    expect(mockCancelCoreMLGeneration).toHaveBeenCalledWith('request-1');
    expect(mockUnsubscribe).not.toHaveBeenCalled();

    const cancelledResult: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: 'Réponse partielle',
      promptTokens: 3,
      generatedTokens: 2,
      duration: 0.5,
      tokensPerSecond: 4,
      finishReason: 'cancelled',
      modelId: 'example/model',
    };
    listeners().onComplete?.(cancelledResult);

    await expect(generation).resolves.toEqual(cancelledResult);
    await expect(disposal).resolves.toBeUndefined();
    expect(mockUnsubscribe).toHaveBeenCalledTimes(1);
    expect(useInferenceStore.getState().activeRequestId).toBeNull();
  });

  it('cancels a native request acknowledged during startup disposal', async () => {
    const acknowledgement = deferred<CoreML.CoreMLGenerationAcknowledgement>();
    mockStartCoreMLGeneration.mockReturnValueOnce(acknowledgement.promise);
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Commence' }],
    });
    const generationOutcome = generation.then(
      () => null,
      (error: unknown) => error,
    );

    const disposal = useInferenceStore.getState().dispose();
    acknowledgement.resolve({ requestId: 'request-1' });

    const generationError = await generationOutcome;
    await disposal;
    expect(generationError).toBeInstanceOf(Error);
    expect((generationError as Error).message).toBe(
      "Le moteur d'inférence locale a été arrêté.",
    );
    expect(mockCancelCoreMLGeneration).toHaveBeenCalledWith('request-1');
    expect(mockUnsubscribe).toHaveBeenCalledTimes(1);
  });

  it('ignores stale request events and out-of-order stream updates', async () => {
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Continue' }],
    });
    await flushPromises();

    listeners().onGeneration?.({
      requestId: 'request-1',
      text: 'Réponse complète',
      generatedTokens: 2,
      tokensPerSecond: 3,
      sequence: 2,
    });
    listeners().onGeneration?.({
      requestId: 'stale-request',
      text: 'Réponse étrangère',
      generatedTokens: 99,
      tokensPerSecond: 99,
      sequence: 99,
    });
    listeners().onGeneration?.({
      requestId: 'request-1',
      text: 'Réponse',
      generatedTokens: 1,
      tokensPerSecond: 2,
      sequence: 1,
    });

    expect(useInferenceStore.getState().generation?.text).toBe(
      'Réponse complète',
    );
    expect(useInferenceStore.getState().activeRequestId).toBe('request-1');

    const result: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: 'Réponse complète',
      promptTokens: 4,
      generatedTokens: 2,
      duration: 0.5,
      tokensPerSecond: 4,
      finishReason: 'eos',
      modelId: 'example/model',
    };
    listeners().onComplete?.({ ...result, requestId: 'stale-request' });
    expect(useInferenceStore.getState().lastResult).toBeNull();
    listeners().onComplete?.(result);

    await expect(generation).resolves.toEqual(result);
  });

  it('buffers only matching events that arrive before native acknowledgement', async () => {
    const acknowledgement = deferred<CoreML.CoreMLGenerationAcknowledgement>();
    mockStartCoreMLGeneration.mockReturnValueOnce(acknowledgement.promise);

    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Bonjour' }],
    });
    listeners().onGeneration?.({
      requestId: 'stale-request',
      text: 'À ignorer',
      generatedTokens: 20,
      tokensPerSecond: 20,
      sequence: 20,
    });
    listeners().onGeneration?.({
      requestId: 'request-1',
      text: 'Bon',
      generatedTokens: 1,
      tokensPerSecond: 2,
      sequence: 1,
    });
    acknowledgement.resolve({ requestId: 'request-1' });
    await flushPromises();

    expect(useInferenceStore.getState()).toMatchObject({
      activeRequestId: 'request-1',
      generation: {
        requestId: 'request-1',
        text: 'Bon',
      },
    });

    const result: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: 'Bonjour',
      promptTokens: 3,
      generatedTokens: 2,
      duration: 0.5,
      tokensPerSecond: 4,
      finishReason: 'eos',
      modelId: 'example/model',
    };
    listeners().onComplete?.(result);
    await expect(generation).resolves.toEqual(result);
  });

  it('honours cancellation requested before native acknowledgement', async () => {
    const acknowledgement = deferred<CoreML.CoreMLGenerationAcknowledgement>();
    mockStartCoreMLGeneration.mockReturnValueOnce(acknowledgement.promise);
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Une longue réponse' }],
    });

    await expect(useInferenceStore.getState().cancelGeneration()).resolves.toBe(
      true,
    );
    expect(mockCancelCoreMLGeneration).not.toHaveBeenCalled();

    acknowledgement.resolve({ requestId: 'request-1' });
    await flushPromises();
    expect(mockCancelCoreMLGeneration).toHaveBeenCalledWith('request-1');

    const result: CoreMLGenerationResult = {
      requestId: 'request-1',
      text: '',
      promptTokens: 0,
      generatedTokens: 0,
      duration: 0.1,
      tokensPerSecond: 0,
      finishReason: 'cancelled',
      modelId: 'example/model',
    };
    listeners().onComplete?.(result);
    await expect(generation).resolves.toEqual(result);
  });

  it('rejects an unscoped native error received before acknowledgement', async () => {
    const acknowledgement = deferred<CoreML.CoreMLGenerationAcknowledgement>();
    mockStartCoreMLGeneration.mockReturnValueOnce(acknowledgement.promise);
    const generation = useInferenceStore.getState().generate({
      messages: [{ role: 'user', content: 'Bonjour' }],
    });

    listeners().onError?.({
      requestId: null,
      code: 'COREML_INVALID_EVENT',
      message: 'Événement natif invalide.',
    });
    acknowledgement.resolve({ requestId: 'request-1' });

    await expect(generation).rejects.toThrow('Événement natif invalide.');
    expect(useInferenceStore.getState()).toMatchObject({
      activeRequestId: null,
      error: 'Événement natif invalide.',
    });
  });

  it('does not let a stale status refresh overwrite model preparation', async () => {
    const statusRefresh = deferred<OnDeviceModelStatus>();
    mockGetCoreMLStatus.mockReturnValueOnce(statusRefresh.promise);

    const initialization = useInferenceStore.getState().initialize();
    await expect(
      useInferenceStore.getState().prepareModel(),
    ).resolves.toMatchObject({ phase: 'ready' });
    statusRefresh.resolve(unavailableStatus());

    await expect(initialization).resolves.toMatchObject({
      phase: 'unavailable',
    });
    expect(useInferenceStore.getState().status.phase).toBe('ready');
  });
});
