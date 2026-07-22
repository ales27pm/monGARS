import { act } from '@testing-library/react-native';
import { useChatStore } from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';
import type { NativeAgentRunResult } from '../src/native/agent';

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
  acknowledgePendingNativeAgentTrigger: jest.fn(),
  approveNativeAgent: jest.fn(),
  cancelNativeAgent: jest.fn().mockResolvedValue(true),
  getPendingNativeAgentTrigger: jest.fn(),
  createNativeAgentRunId: jest
    .fn()
    .mockReturnValue('11111111-1111-4111-8111-111111111111'),
  rejectNativeAgent: jest.fn(),
  requestNativeAgentPermission: jest.fn(),
}));

jest.mock('../src/services/onDeviceAgentService', () => {
  const actual = jest.requireActual('../src/services/onDeviceAgentService');
  return {
    ...actual,
    executeNativeAgent: jest.fn(),
  };
});

import {
  approveNativeAgent,
  acknowledgePendingNativeAgentTrigger,
  getPendingNativeAgentTrigger,
  rejectNativeAgent,
  requestNativeAgentPermission,
} from '../src/native/agent';
import { executeNativeAgent } from '../src/services/onDeviceAgentService';
import { postConversationMessage } from '../src/services/chatService';

const mockExecuteAgent = executeNativeAgent as jest.MockedFunction<
  typeof executeNativeAgent
>;
const mockApproveAgent = approveNativeAgent as jest.MockedFunction<
  typeof approveNativeAgent
>;
const mockConsumeTrigger = getPendingNativeAgentTrigger as jest.MockedFunction<
  typeof getPendingNativeAgentTrigger
>;
const mockAcknowledgeTrigger =
  acknowledgePendingNativeAgentTrigger as jest.MockedFunction<
    typeof acknowledgePendingNativeAgentTrigger
  >;
const mockRejectAgent = rejectNativeAgent as jest.MockedFunction<
  typeof rejectNativeAgent
>;
const mockRequestPermission =
  requestNativeAgentPermission as jest.MockedFunction<
    typeof requestNativeAgentPermission
  >;

const baseResult = {
  runId: '11111111-1111-4111-8111-111111111111',
  intent: 'camera' as const,
  events: [],
  executedToolCount: 0,
  modelTurnCount: 1,
  usedRepairAttempt: false,
};

const approvalResult = (): NativeAgentRunResult => ({
  ...baseResult,
  status: 'approval_required',
  approval: {
    recordId: '22222222-2222-4222-8222-222222222222',
    toolId: 'camera.capture',
    arguments: {},
    displayName: 'Capture Image',
    risk: 'high',
    expiresAt: new Date(Date.now() + 600_000).toISOString(),
  },
});

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

describe('chat store native agent approval boundary', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApproveAgent.mockResolvedValue({
      recordId: '22222222-2222-4222-8222-222222222222',
      status: 'approved',
    });
    mockRejectAgent.mockResolvedValue({
      recordId: '22222222-2222-4222-8222-222222222222',
      status: 'rejected',
    });
    mockRequestPermission.mockResolvedValue({
      permission: 'camera',
      state: 'granted',
    });
    mockConsumeTrigger.mockResolvedValue(null);
    mockAcknowledgeTrigger.mockResolvedValue(true);
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
    });
  });

  it('fails honestly when the native agent is unavailable and never calls chat server', async () => {
    mockExecuteAgent.mockRejectedValueOnce(
      new Error("Le moteur d'outils local n'est pas lié."),
    );

    await expect(
      useChatStore.getState().sendMessage('Take a photo', 'chat'),
    ).rejects.toThrow("moteur d'outils local");

    expect(useChatStore.getState().error).toContain("moteur d'outils local");
    expect(postConversationMessage).not.toHaveBeenCalled();
  });

  it('stores a bound approval and executes nothing before approval', async () => {
    mockExecuteAgent.mockResolvedValueOnce(approvalResult());

    await useChatStore.getState().sendMessage('Take a photo', 'chat');

    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);
    expect(mockApproveAgent).not.toHaveBeenCalled();
    expect(useChatStore.getState().pendingAgentApproval).toMatchObject({
      ownerId: 'guest',
      prompt: 'Take a photo',
      toolId: 'camera.capture',
      arguments: {},
      recordId: '22222222-2222-4222-8222-222222222222',
    });
    expect(useChatStore.getState().messages).toHaveLength(1);
  });

  it('approves the exact binding and only then reruns the native agent', async () => {
    mockExecuteAgent
      .mockResolvedValueOnce(approvalResult())
      .mockResolvedValueOnce({
        ...baseResult,
        status: 'final',
        executedToolCount: 1,
        message: 'Camera opened after approval.',
      });

    await useChatStore.getState().sendMessage('Take a photo', 'chat');
    await useChatStore.getState().approvePendingAgent();

    expect(mockApproveAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        ownerId: 'guest',
        prompt: 'Take a photo',
        toolId: 'camera.capture',
        arguments: {},
      }),
    );
    expect(mockExecuteAgent).toHaveBeenCalledTimes(2);
    expect(mockExecuteAgent.mock.calls[1][0]).toMatchObject({
      ownerId: 'guest',
      prompt: 'Take a photo',
      approvalRecordId: '22222222-2222-4222-8222-222222222222',
    });
    expect(useChatStore.getState().pendingAgentApproval).toBeNull();
    expect(useChatStore.getState().messages.at(-1)?.content).toBe(
      'Camera opened after approval.',
    );
  });

  it('rejects a pending action without a second agent run', async () => {
    mockExecuteAgent.mockResolvedValueOnce(approvalResult());

    await useChatStore.getState().sendMessage('Take a photo', 'chat');
    await useChatStore.getState().rejectPendingAgent();

    expect(mockRejectAgent).toHaveBeenCalledTimes(1);
    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);
    expect(useChatStore.getState().pendingAgentApproval).toBeNull();
    expect(useChatStore.getState().messages.at(-1)?.content).toContain(
      'Aucun outil',
    );
  });

  it('requests a foreground iOS permission and retries only after it is granted', async () => {
    mockExecuteAgent
      .mockResolvedValueOnce({
        ...baseResult,
        status: 'permission_required',
        permission: 'camera',
        message: "L'autorisation camera est requise.",
      })
      .mockResolvedValueOnce({
        ...baseResult,
        status: 'final',
        executedToolCount: 1,
        message: 'Camera ready.',
      });

    await useChatStore.getState().sendMessage('Take a photo', 'chat');
    expect(useChatStore.getState().pendingAgentPermission).toMatchObject({
      ownerId: 'guest',
      prompt: 'Take a photo',
      permission: 'camera',
    });
    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);

    await useChatStore.getState().requestPendingAgentPermission();

    expect(mockRequestPermission).toHaveBeenCalledWith('camera');
    expect(mockExecuteAgent).toHaveBeenCalledTimes(2);
    expect(useChatStore.getState().pendingAgentPermission).toBeNull();
    expect(useChatStore.getState().messages.at(-1)?.content).toBe(
      'Camera ready.',
    );
  });

  it('does not retry a tool when the iOS permission is denied', async () => {
    mockExecuteAgent.mockResolvedValueOnce({
      ...baseResult,
      status: 'permission_required',
      permission: 'camera',
      message: "L'autorisation camera est requise.",
    });
    mockRequestPermission.mockResolvedValueOnce({
      permission: 'camera',
      state: 'denied',
    });

    await useChatStore.getState().sendMessage('Take a photo', 'chat');
    await useChatStore.getState().requestPendingAgentPermission();

    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);
    expect(useChatStore.getState().pendingAgentPermission).toBeNull();
    expect(useChatStore.getState().messages.at(-1)?.content).toContain(
      "aucun outil n'a été exécuté",
    );
  });

  it('shows a consumed trigger handoff and never auto-runs it', async () => {
    mockConsumeTrigger.mockResolvedValueOnce({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Météo du matin',
      prompt: 'What is the weather in Toronto?',
      repeats: false,
    });
    mockExecuteAgent.mockResolvedValueOnce({
      ...baseResult,
      intent: 'weather',
      status: 'final',
      executedToolCount: 1,
      message: 'Weather ready.',
    });

    await useChatStore.getState().initialize();

    expect(useChatStore.getState().pendingAgentTrigger).toMatchObject({
      ownerId: 'guest',
      title: 'Météo du matin',
    });
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(mockAcknowledgeTrigger).not.toHaveBeenCalled();

    await useChatStore.getState().runPendingAgentTrigger();

    expect(mockAcknowledgeTrigger).toHaveBeenCalledWith(
      'guest',
      '33333333-3333-4333-8333-333333333333',
    );
    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);
    expect(useChatStore.getState().pendingAgentTrigger).toBeNull();
    expect(useChatStore.getState().messages.at(-1)?.content).toBe(
      'Weather ready.',
    );
  });

  it('accepts and runs a trigger prompt at exactly 512 UTF-8 bytes', async () => {
    const prefix = 'Weather in Toronto? ';
    const prompt = prefix + 'a'.repeat(512 - prefix.length);
    mockConsumeTrigger.mockResolvedValueOnce({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Bounded weather',
      prompt,
      repeats: false,
    });
    mockExecuteAgent.mockResolvedValueOnce({
      ...baseResult,
      intent: 'weather',
      status: 'final',
      executedToolCount: 1,
      message: 'Weather ready.',
    });

    await useChatStore.getState().initialize();
    await useChatStore.getState().runPendingAgentTrigger();

    expect(mockAcknowledgeTrigger).toHaveBeenCalledTimes(1);
    expect(mockExecuteAgent).toHaveBeenCalledTimes(1);
    expect(mockExecuteAgent).toHaveBeenCalledWith(
      expect.objectContaining({ prompt }),
      expect.any(String),
    );
  });

  it('keeps a 513-byte legacy one-shot before acknowledgement', async () => {
    const prefix = 'Weather in Toronto? ';
    const prompt = prefix + 'a'.repeat(513 - prefix.length);
    useChatStore.setState({
      pendingAgentTrigger: {
        ownerId: 'guest',
        id: '33333333-3333-4333-8333-333333333333',
        title: 'Legacy oversized weather',
        prompt,
        repeats: false,
      },
    });

    await expect(
      useChatStore.getState().runPendingAgentTrigger(),
    ).rejects.toThrow('512 octets UTF-8');

    expect(mockAcknowledgeTrigger).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(useChatStore.getState().pendingAgentTrigger?.prompt).toBe(prompt);
    expect(useChatStore.getState().loading).toBe(false);
  });

  it('rejects an oversized native trigger handoff without consuming it', async () => {
    mockConsumeTrigger.mockResolvedValueOnce({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Legacy oversized weather',
      prompt: 'a'.repeat(513),
      repeats: false,
    });

    await useChatStore.getState().refreshPendingAgentTrigger();

    expect(useChatStore.getState().pendingAgentTrigger).toBeNull();
    expect(useChatStore.getState().notice?.message).toContain('512 octets');
    expect(mockAcknowledgeTrigger).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
  });

  it('deduplicates warm trigger refreshes without acknowledgement or execution', async () => {
    const handoff = {
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Météo du matin',
      prompt: 'What is the weather in Toronto?',
      repeats: false,
    };
    mockConsumeTrigger.mockResolvedValue(handoff);

    await useChatStore.getState().refreshPendingAgentTrigger();
    await useChatStore.getState().refreshPendingAgentTrigger();

    expect(mockConsumeTrigger).toHaveBeenNthCalledWith(1, 'guest');
    expect(mockConsumeTrigger).toHaveBeenNthCalledWith(2, 'guest');
    expect(useChatStore.getState().pendingAgentTrigger).toEqual({
      ...handoff,
      ownerId: 'guest',
    });
    expect(mockAcknowledgeTrigger).not.toHaveBeenCalled();
    expect(mockExecuteAgent).not.toHaveBeenCalled();
  });

  it('does not run a trigger whose native acknowledgement fails', async () => {
    mockConsumeTrigger.mockResolvedValueOnce({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Météo du matin',
      prompt: 'What is the weather in Toronto?',
      repeats: false,
    });
    mockAcknowledgeTrigger.mockResolvedValueOnce(false);

    await useChatStore.getState().initialize();
    await useChatStore.getState().runPendingAgentTrigger();

    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(useChatStore.getState().pendingAgentTrigger).toBeNull();
    expect(useChatStore.getState().notice?.message).toContain('aucun agent');
  });

  it('binds trigger acknowledgement to the exact owner and backend', async () => {
    const acknowledgement = deferred<boolean>();
    const secretPrompt = 'Email the private account summary to Alice';
    mockConsumeTrigger.mockResolvedValueOnce({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Private summary',
      prompt: secretPrompt,
      repeats: false,
    });
    mockAcknowledgeTrigger.mockReturnValueOnce(acknowledgement.promise);

    await useChatStore.getState().initialize();
    const run = useChatStore.getState().runPendingAgentTrigger();
    await Promise.resolve();

    await expect(
      useChatStore.getState().setSession({
        username: 'Bob',
        token: 'bob-token',
      }),
    ).rejects.toThrow('avant de changer de compte');
    await expect(
      useChatStore.getState().setInferenceBackend('server'),
    ).rejects.toThrow('avant de changer de backend');

    // Simulate an out-of-band state mutation despite the public switch guards.
    useChatStore.setState({
      session: { username: 'Bob', token: 'bob-token' },
    });
    acknowledgement.resolve(true);
    await run;

    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(postConversationMessage).not.toHaveBeenCalled();
    expect(
      useChatStore
        .getState()
        .messages.some((message) => message.content.includes(secretPrompt)),
    ).toBe(false);
    expect(useChatStore.getState().loading).toBe(false);
  });

  it('drops a stale trigger lookup when the local owner changes', async () => {
    const oldOwnerLookup = deferred<{
      id: string;
      title: string;
      prompt: string;
      repeats: boolean;
    } | null>();
    const secretPrompt = 'Private trigger for the guest profile';
    mockConsumeTrigger
      .mockReturnValueOnce(oldOwnerLookup.promise)
      .mockResolvedValueOnce(null);

    const initializeGuest = useChatStore.getState().initialize();
    for (
      let attempt = 0;
      attempt < 10 && mockConsumeTrigger.mock.calls.length === 0;
      attempt += 1
    ) {
      await Promise.resolve();
    }
    expect(mockConsumeTrigger).toHaveBeenCalledWith('guest');

    await useChatStore.getState().setSession({
      username: 'Bob',
      token: 'bob-token',
    });
    oldOwnerLookup.resolve({
      id: '33333333-3333-4333-8333-333333333333',
      title: 'Guest only',
      prompt: secretPrompt,
      repeats: false,
    });
    await initializeGuest;

    expect(useChatStore.getState().pendingAgentTrigger).toBeNull();
    expect(useChatStore.getState().notice?.message).not.toContain('Guest only');
    expect(
      useChatStore
        .getState()
        .messages.some((message) => message.content.includes(secretPrompt)),
    ).toBe(false);
  });

  it('rejects and removes an approval when the local account owner changes', async () => {
    useChatStore.setState({
      session: { username: 'Alice', token: 'alice-token' },
    });
    mockExecuteAgent.mockResolvedValueOnce(approvalResult());

    await useChatStore.getState().sendMessage('Take a photo', 'chat');
    expect(useChatStore.getState().pendingAgentApproval?.ownerId).toBe(
      'account:Alice',
    );
    await useChatStore.getState().setSession({
      username: 'Bob',
      token: 'bob-token',
    });

    expect(mockRejectAgent).toHaveBeenCalledWith(
      expect.objectContaining({ ownerId: 'account:Alice' }),
    );
    expect(useChatStore.getState().pendingAgentApproval).toBeNull();
    await expect(useChatStore.getState().approvePendingAgent()).rejects.toThrow(
      'Aucune action locale',
    );
  });

  it('blocks an account switch while an owner-scoped agent run is active', async () => {
    const pending = deferred<NativeAgentRunResult>();
    mockExecuteAgent.mockReturnValueOnce(pending.promise);

    const send = useChatStore
      .getState()
      .sendMessage('What is the weather in Toronto?', 'chat');
    await Promise.resolve();
    await Promise.resolve();

    await expect(
      useChatStore.getState().setSession({
        username: 'Bob',
        token: 'bob-token',
      }),
    ).rejects.toThrow('avant de changer de compte');
    expect(useChatStore.getState().session).toBeNull();

    pending.resolve({
      ...baseResult,
      intent: 'weather',
      status: 'final',
      message: 'Weather ready.',
    });
    await send;
  });

  it('keeps ordinary local chat on the existing streaming generator', async () => {
    const generate = jest.fn().mockResolvedValue({
      requestId: 'local-chat',
      text: 'Réponse en continu',
      promptTokens: 4,
      generatedTokens: 3,
      duration: 0.5,
      tokensPerSecond: 6,
      finishReason: 'eos',
      modelId: 'ales27pm/Dolphin3.0-CoreML',
    });
    useInferenceStore.setState({ generate });

    await act(async () => {
      await useChatStore.getState().sendMessage('Bonjour monGARS', 'chat');
    });

    expect(generate).toHaveBeenCalledTimes(1);
    expect(mockExecuteAgent).not.toHaveBeenCalled();
    expect(postConversationMessage).not.toHaveBeenCalled();
    expect(useChatStore.getState().messages.at(-1)?.content).toBe(
      'Réponse en continu',
    );
  });
});
