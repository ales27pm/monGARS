import {
  CoreMLUnavailableError,
  coreMLModuleAvailable,
  getCoreMLStatus,
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
});
