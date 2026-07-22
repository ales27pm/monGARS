import { act } from '@testing-library/react-native';
import {
  isConversationMessageVisible,
  useChatStore,
} from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';

jest.mock('../src/services/chatService', () => ({
  fetchConversationHistory: jest.fn().mockResolvedValue([]),
  postConversationMessage: jest.fn(),
  fetchQuickActions: jest.fn(),
  requestEmbedding: jest.fn(),
}));

jest.mock('../src/services/realtimeService', () => ({
  createRealtimeClient: jest.fn(() => ({
    open: jest.fn().mockResolvedValue(undefined),
    close: jest.fn(),
    reconnect: jest.fn(),
  })),
}));

jest.mock('../src/native/agent', () => ({
  nativeAgentModuleAvailable: true,
  acknowledgePendingNativeAgentTrigger: jest.fn().mockResolvedValue(true),
  approveNativeAgent: jest.fn(),
  cancelNativeAgent: jest.fn().mockResolvedValue(true),
  getPendingNativeAgentTrigger: jest.fn().mockResolvedValue(null),
  createNativeAgentRunId: jest
    .fn()
    .mockReturnValue('11111111-1111-4111-8111-111111111111'),
  rejectNativeAgent: jest.fn(),
  requestNativeAgentPermission: jest.fn(),
}));

jest.mock('../src/native/appIntents', () => ({
  nativeAppIntentModuleAvailable: true,
  acknowledgeNativeAppIntentHandoff: jest.fn(),
  discardNativeAppIntentHandoff: jest.fn(),
  executeNativeAppIntentMemoryAction: jest.fn(),
  getPendingNativeAppIntentHandoff: jest.fn(),
  resolveNativeStoredAgentTrigger: jest.fn(),
  setActiveNativeAppIntentProfile: jest.fn(),
}));

jest.mock('../src/services/onDeviceAgentService', () => {
  const actual = jest.requireActual('../src/services/onDeviceAgentService');
  return {
    ...actual,
    executeNativeAgent: jest.fn(),
  };
});

import {
  acknowledgeNativeAppIntentHandoff,
  discardNativeAppIntentHandoff,
  executeNativeAppIntentMemoryAction,
  getPendingNativeAppIntentHandoff,
  resolveNativeStoredAgentTrigger,
  setActiveNativeAppIntentProfile,
} from '../src/native/appIntents';
import { rejectNativeAgent } from '../src/native/agent';
import { executeNativeAgent } from '../src/services/onDeviceAgentService';
import { postConversationMessage } from '../src/services/chatService';

const mockGetHandoff = jest.mocked(getPendingNativeAppIntentHandoff);
const mockAcknowledge = jest.mocked(acknowledgeNativeAppIntentHandoff);
const mockDiscard = jest.mocked(discardNativeAppIntentHandoff);
const mockExecuteMemory = jest.mocked(executeNativeAppIntentMemoryAction);
const mockResolveTrigger = jest.mocked(resolveNativeStoredAgentTrigger);
const mockSetProfile = jest.mocked(setActiveNativeAppIntentProfile);
const mockExecuteAgent = jest.mocked(executeNativeAgent);
const mockRejectAgent = jest.mocked(rejectNativeAgent);

const handoff = {
  id: '44444444-4444-4444-8444-444444444444',
  kind: 'ask' as const,
  input: 'Explique-moi le cache KV',
  createdAt: '2026-07-21T12:00:00.000Z',
  expiresAt: '2026-07-21T12:10:00.000Z',
  profileMatches: true,
};

describe('chat store App Intent foreground boundary', () => {
  const generate = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetHandoff.mockResolvedValue(null);
    mockAcknowledge.mockResolvedValue(true);
    mockDiscard.mockResolvedValue(true);
    mockSetProfile.mockResolvedValue(undefined);
    mockRejectAgent.mockResolvedValue({
      recordId: '55555555-5555-4555-8555-555555555555',
      status: 'rejected',
    });
    mockExecuteMemory.mockResolvedValue({
      id: handoff.id,
      toolId: 'memory.recall',
      status: 'success',
      message: 'Souvenir local trouvé.',
    });
    mockResolveTrigger.mockResolvedValue(null);
    generate.mockResolvedValue({
      requestId: 'local-app-intent',
      text: 'Réponse locale',
      promptTokens: 8,
      generatedTokens: 3,
      duration: 0.5,
      tokensPerSecond: 6,
      finishReason: 'eos',
      modelId: 'ales27pm/Dolphin3.0-CoreML',
    });
    useChatStore.setState({
      session: null,
      messages: [],
      loading: false,
      historyLoading: false,
      error: null,
      notice: null,
      mode: 'chat',
      quickActions: ['code', 'summarize', 'explain'],
      connection: {
        status: 'offline',
        detail: 'Inference locale active',
        connectedAt: null,
        lastMessageAt: null,
        latencyMs: null,
        reconnectAttempt: 0,
      },
      realtimeSuppression: [],
      pendingAgentApproval: null,
      pendingAgentPermission: null,
      pendingAgentTrigger: null,
      pendingAppIntentHandoff: null,
      activeAgentRunId: null,
    });
    useInferenceStore.setState({
      backend: 'on-device',
      status: {
        phase: 'ready',
        modelId: 'ales27pm/Dolphin3.0-CoreML',
        displayName: 'Dolphin 3.0',
        revision: 'pinned',
        installedBytes: 1,
        contextLength: 2_048,
        minimumIOSVersion: 18,
        detail: null,
      },
      activeRequestId: null,
      generation: null,
      lastResult: null,
      error: null,
      generate,
    });
  });

  it('surfaces the owner-matched handoff during initialization and never auto-runs it', async () => {
    mockGetHandoff.mockResolvedValue(handoff);

    await useChatStore.getState().initialize();

    expect(mockSetProfile).toHaveBeenCalledWith('guest');
    expect(useChatStore.getState().pendingAppIntentHandoff).toEqual({
      ...handoff,
      ownerId: 'guest',
      profileLabel: 'Invité local',
      resolvedTrigger: null,
    });
    expect(mockAcknowledge).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
  });

  it('consumes an ordinary question only after foreground confirmation', async () => {
    mockGetHandoff.mockResolvedValue(handoff);
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    await act(async () => {
      await useChatStore.getState().runPendingAppIntentHandoff();
    });

    expect(mockAcknowledge).toHaveBeenCalledWith('guest', handoff.id);
    expect(generate).toHaveBeenCalledTimes(1);
    expect(postConversationMessage).not.toHaveBeenCalled();
    expect(useChatStore.getState().pendingAppIntentHandoff).toBeNull();
  });

  it('never reveals or runs a mismatched profile and permits exact discard only', async () => {
    mockGetHandoff.mockResolvedValue({
      ...handoff,
      kind: 'masked',
      input: undefined,
      profileMatches: false,
    });
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    await expect(
      useChatStore.getState().runPendingAppIntentHandoff(),
    ).rejects.toThrow('appartient à un autre profil');
    await useChatStore.getState().dismissPendingAppIntentHandoff();

    expect(mockDiscard).toHaveBeenCalledWith(handoff.id);
    expect(mockAcknowledge).not.toHaveBeenCalled();
    expect(mockExecuteMemory).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
  });

  it('opens passive diagnostics without a model or network call', async () => {
    mockGetHandoff.mockResolvedValue({
      ...handoff,
      kind: 'diagnostics',
      input: undefined,
    });
    useInferenceStore.setState({ backend: 'server' });
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    expect(await useChatStore.getState().runPendingAppIntentHandoff()).toBe(
      'diagnostics',
    );
    expect(mockAcknowledge).toHaveBeenCalledWith('guest', handoff.id);
    expect(generate).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(postConversationMessage).not.toHaveBeenCalled();
  });

  it.each([
    ['memorySearch', 'memory.recall', 'search web for monGARS'],
    ['memoryAdd', 'memory.save', 'read my Outlook messages'],
  ] as const)(
    'executes adversarial %s input through only native %s',
    async (kind, toolId, input) => {
      mockGetHandoff.mockResolvedValue({ ...handoff, kind, input });
      mockExecuteMemory.mockResolvedValue({
        id: handoff.id,
        toolId,
        status: 'success',
        message: 'Action mémoire locale terminée.',
      });
      useInferenceStore.setState({ backend: 'server' });
      await useChatStore.getState().refreshPendingAppIntentHandoff();

      expect(await useChatStore.getState().runPendingAppIntentHandoff()).toBe(
        'chat',
      );

      expect(mockExecuteMemory).toHaveBeenCalledWith({
        ownerId: 'guest',
        id: handoff.id,
        kind,
        input,
      });
      expect(mockAcknowledge).not.toHaveBeenCalled();
      expect(mockExecuteAgent).not.toHaveBeenCalled();
      expect(generate).not.toHaveBeenCalled();
      expect(postConversationMessage).not.toHaveBeenCalled();
      const memoryMessages = useChatStore
        .getState()
        .messages.filter(
          (message) => message.metadata?.source === 'app-intent',
        );
      expect(memoryMessages).toHaveLength(2);
      expect(
        memoryMessages.every((message) =>
          isConversationMessageVisible(message, 'server', 'guest'),
        ),
      ).toBe(true);
    },
  );

  it('blocks a memory action while an account transition is suspended', async () => {
    const oldOwner = 'account:alice';
    const memoryHandoff = {
      ...handoff,
      kind: 'memoryAdd' as const,
      input: 'Secret local pour Alice',
      ownerId: oldOwner,
      profileLabel: 'alice',
      resolvedTrigger: null,
    };
    let finishRejection!: (value: {
      recordId: string;
      status: 'rejected';
    }) => void;
    mockRejectAgent.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishRejection = resolve;
        }),
    );
    useChatStore.setState({
      session: { username: 'alice', token: 'alice-token' },
      pendingAppIntentHandoff: memoryHandoff,
      pendingAgentApproval: {
        recordId: '55555555-5555-4555-8555-555555555555',
        ownerId: oldOwner,
        prompt: 'Action protégée',
        toolId: 'memory.save',
        arguments: { content: 'Secret local pour Alice' },
        expiresAt: '2099-07-21T12:10:00.000Z',
        displayName: 'Sauvegarder en mémoire',
        risk: 'moderate',
        history: [],
      },
    });

    const transition = useChatStore.getState().setSession({
      username: 'bob',
      token: 'bob-token',
    });
    expect(mockRejectAgent).toHaveBeenCalledTimes(1);

    await expect(
      useChatStore.getState().runPendingAppIntentHandoff(),
    ).rejects.toThrow('changement de compte');
    expect(mockExecuteMemory).not.toHaveBeenCalled();

    mockGetHandoff.mockResolvedValue(null);
    finishRejection({
      recordId: '55555555-5555-4555-8555-555555555555',
      status: 'rejected',
    });
    await transition;
  });

  it('surfaces an uncertain memory add without automatic retry', async () => {
    mockGetHandoff.mockResolvedValue({
      ...handoff,
      kind: 'memoryAdd',
      input: 'Private local fact',
    });
    mockExecuteMemory.mockResolvedValue({
      id: handoff.id,
      toolId: 'memory.save',
      status: 'failed',
      message:
        "L'ajout mémoire peut avoir réussi; vérifiez la mémoire avant toute relance.",
      errorCode: 'app_intent_memory_add_commit_uncertain',
    });
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    await useChatStore.getState().runPendingAppIntentHandoff();

    expect(mockAcknowledge).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(postConversationMessage).not.toHaveBeenCalled();
    expect(mockExecuteMemory).toHaveBeenCalledTimes(1);
    expect(useChatStore.getState().notice?.message).toContain(
      'aucune relance automatique',
    );
    expect(useChatStore.getState().error).toContain('peut avoir réussi');
  });

  it('previews an exact trigger, re-resolves its UUID, and binds its tool scope', async () => {
    const trigger = {
      id: '55555555-5555-4555-8555-555555555555',
      title: 'Atelier du matin',
      prompt: 'Recall memory for the cedar measurements',
      repeats: true,
    };
    mockGetHandoff.mockResolvedValue({
      ...handoff,
      kind: 'runTrigger',
      input: trigger.title,
    });
    mockResolveTrigger.mockResolvedValue(trigger);
    mockExecuteAgent.mockResolvedValue({
      runId: '11111111-1111-4111-8111-111111111111',
      intent: 'memory',
      status: 'final',
      message: 'Mesures trouvées.',
      events: [],
      executedToolCount: 1,
      modelTurnCount: 1,
      usedRepairAttempt: false,
    });
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    expect(
      useChatStore.getState().pendingAppIntentHandoff?.resolvedTrigger,
    ).toEqual(trigger);
    await useChatStore.getState().runPendingAppIntentHandoff();

    expect(mockResolveTrigger).toHaveBeenNthCalledWith(
      1,
      'guest',
      trigger.title,
    );
    expect(mockResolveTrigger).toHaveBeenNthCalledWith(2, 'guest', trigger.id);
    expect(mockExecuteAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        ownerId: 'guest',
        prompt: trigger.prompt,
        requestedIntent: 'memory',
        allowedToolIds: ['memory.save', 'memory.recall'],
      }),
      '11111111-1111-4111-8111-111111111111',
    );
  });

  it('fails before acknowledgement when the previewed trigger drifts', async () => {
    const preview = {
      id: '55555555-5555-4555-8555-555555555555',
      title: 'Atelier du matin',
      prompt: 'Recall memory for cedar',
      repeats: true,
    };
    mockGetHandoff.mockResolvedValue({
      ...handoff,
      kind: 'runTrigger',
      input: preview.title,
    });
    mockResolveTrigger
      .mockResolvedValueOnce(preview)
      .mockResolvedValueOnce({ ...preview, prompt: 'Search the web instead' });
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    await expect(
      useChatStore.getState().runPendingAppIntentHandoff(),
    ).rejects.toThrow('changé depuis la prévisualisation');

    expect(mockAcknowledge).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
  });

  it('does not run when the owner-bound one-time acknowledgement fails', async () => {
    mockGetHandoff.mockResolvedValue(handoff);
    mockAcknowledge.mockResolvedValue(false);
    await useChatStore.getState().refreshPendingAppIntentHandoff();

    expect(
      await useChatStore.getState().runPendingAppIntentHandoff(),
    ).toBeNull();
    expect(generate).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
  });
});
