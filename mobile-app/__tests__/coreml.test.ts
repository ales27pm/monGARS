import {
  CoreMLUnavailableError,
  coreMLModuleAvailable,
  getCoreMLStatus,
  normalizeCoreMLErrorEvent,
  prepareCoreMLModel,
  subscribeToCoreMLEvents,
} from '../src/native/coreml';

describe('Core ML native facade without a linked module', () => {
  it('reports an unavailable status without failing app startup', async () => {
    expect(coreMLModuleAvailable).toBe(false);

    await expect(getCoreMLStatus()).resolves.toMatchObject({
      phase: 'unavailable',
      minimumIOSVersion: 18,
    });
  });

  it('uses no-op subscriptions and rejects mutating operations explicitly', async () => {
    const unsubscribe = subscribeToCoreMLEvents({
      onStatus: jest.fn(),
    });

    expect(unsubscribe()).toBeUndefined();
    await expect(prepareCoreMLModel()).rejects.toBeInstanceOf(
      CoreMLUnavailableError,
    );
  });

  it('preserves the native recoverable error flag during normalization', () => {
    expect(
      normalizeCoreMLErrorEvent({
        requestId: 'request-1',
        code: 'coreml_integrity_failure',
        message: 'Le modele doit etre repare.',
        recoverable: true,
      }),
    ).toEqual({
      requestId: 'request-1',
      code: 'coreml_integrity_failure',
      message: 'Le modele doit etre repare.',
      recoverable: true,
    });
  });

  it('fails closed when a native error omits the recoverable flag', () => {
    expect(normalizeCoreMLErrorEvent({ message: 'Erreur native.' })).toEqual({
      requestId: null,
      code: null,
      message: 'Erreur native.',
      recoverable: false,
    });
  });
});
