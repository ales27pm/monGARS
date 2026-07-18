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

describe('inferenceStore', () => {
  beforeEach(() => {
    useInferenceStore.getState().dispose();
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
});
