import { act } from '@testing-library/react-native';
import {
  getLocalConversationOwner,
  useChatStore,
} from '../src/store/chatStore';
import { useInferenceStore } from '../src/store/inferenceStore';
import type {
  CoreMLGenerationRequest,
  CoreMLGenerationResult,
} from '../src/native/coreml';

jest.mock('../src/services/chatService', () => ({
  fetchConversationHistory: jest.fn(),
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

import {
  fetchConversationHistory,
  postConversationMessage,
  requestEmbedding,
} from '../src/services/chatService';
import { createRealtimeClient } from '../src/services/realtimeService';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

describe('chatStore', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (createRealtimeClient as jest.Mock).mockReturnValue({
      open: jest.fn().mockResolvedValue(undefined),
      close: jest.fn(),
      reconnect: jest.fn(),
    });
    (fetchConversationHistory as jest.Mock).mockReset().mockResolvedValue([]);
    const storage = {
      getItem: jest.fn().mockResolvedValue(null),
      setItem: jest.fn().mockResolvedValue(undefined),
      removeItem: jest.fn().mockResolvedValue(undefined),
    };
    useChatStore.persist?.setOptions?.({ storage: storage as any });
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
        detail: 'Aucune session active',
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
      initialize: useChatStore.getState().initialize,
      setSession: useChatStore.getState().setSession,
      sendMessage: useChatStore.getState().sendMessage,
      refreshHistory: useChatStore.getState().refreshHistory,
      requestQuickActions: useChatStore.getState().requestQuickActions,
      setMode: useChatStore.getState().setMode,
      retryRealtime: useChatStore.getState().retryRealtime,
      clearError: useChatStore.getState().clearError,
      clearNotice: useChatStore.getState().clearNotice,
      logout: useChatStore.getState().logout,
    });
    useChatStore.persist?.clearStorage?.();
    useInferenceStore.setState({
      backend: 'server',
      status: {
        phase: 'unavailable',
        modelId: null,
        displayName: null,
        revision: null,
        installedBytes: 0,
        contextLength: 0,
        minimumIOSVersion: 18,
        detail: 'Module indisponible',
      },
      activeRequestId: null,
      generation: null,
      lastResult: null,
      error: null,
    });
  });

  it('keeps case-distinct authenticated owners isolated', () => {
    expect(getLocalConversationOwner({ username: 'Alice', token: 'one' })).toBe(
      'account:Alice',
    );
    expect(getLocalConversationOwner({ username: 'alice', token: 'two' })).toBe(
      'account:alice',
    );
    expect(
      getLocalConversationOwner({ username: 'Alice', token: 'one' }),
    ).not.toBe(getLocalConversationOwner({ username: 'alice', token: 'two' }));
  });

  it('loads conversation history when a session is present', async () => {
    (fetchConversationHistory as jest.Mock).mockResolvedValue([
      {
        query: 'Bonjour',
        response: 'Salut',
        timestamp: new Date().toISOString(),
      },
    ]);

    await act(async () => {
      await useChatStore.getState().setSession({
        username: 'u1',
        token: 'token',
      });
    });

    const state = useChatStore.getState();
    expect(state.messages).toHaveLength(2);
    expect(state.messages[0].role).toBe('user');
    expect(state.messages[0].content).toBe('Bonjour');
    expect(state.messages[1].content).toBe('Salut');
    expect(fetchConversationHistory).toHaveBeenCalledWith({
      username: 'u1',
      token: 'token',
    });
  });

  it('records an error when trying to send without a session', async () => {
    await act(async () => {
      await expect(
        useChatStore.getState().sendMessage('hello', 'chat'),
      ).rejects.toThrow('Session absente.');
    });

    expect(useChatStore.getState().error).toBe('Session absente.');
  });

  it('never serializes the in-memory session or bearer token', () => {
    const partialize = useChatStore.persist.getOptions().partialize;
    if (!partialize) {
      throw new Error('Partialisation chat manquante');
    }
    const secretToken = 'secret-jwt-that-must-never-reach-storage';
    useChatStore.setState({
      session: { username: 'Alice', token: secretToken },
    });

    const serialized = JSON.stringify({
      state: partialize(useChatStore.getState()),
      version: useChatStore.persist.getOptions().version,
    });

    expect(serialized).not.toContain(secretToken);
    expect(JSON.parse(serialized).state).not.toHaveProperty('session');
    expect(useChatStore.persist.getOptions().version).toBe(6);
  });

  it('stores a chat reply and suppresses the next realtime echo', async () => {
    (postConversationMessage as jest.Mock).mockResolvedValue({
      response: 'Salut!',
      confidence: 0.9,
      processingTime: 0.1,
      speechTurn: {
        turnId: 'turn-1',
        text: 'Salut!',
        createdAt: new Date().toISOString(),
        segments: [],
        averageWordsPerSecond: 2.0,
        tempo: 1,
      },
    });

    await act(async () => {
      await useChatStore.getState().setSession({
        username: 'u1',
        token: 'token',
      });
      await useChatStore.getState().sendMessage('hello', 'chat');
    });

    const { messages, loading, realtimeSuppression } = useChatStore.getState();
    expect(messages.some((message) => message.role === 'user')).toBe(true);
    expect(messages.some((message) => message.content === 'Salut!')).toBe(true);
    expect(loading).toBe(false);
    expect(realtimeSuppression).toContain('hello::Salut!');
    expect(postConversationMessage).toHaveBeenCalledWith(
      { username: 'u1', token: 'token' },
      'hello',
    );
  });

  it('handles embedding mode by appending an embedding summary', async () => {
    (requestEmbedding as jest.Mock).mockResolvedValue({
      vectors: [[0.1, 0.2, 0.3]],
      dims: 3,
      count: 1,
      normalised: false,
      backend: 'test-backend',
      model: 'test-model',
    });

    await act(async () => {
      await useChatStore.getState().setSession({
        username: 'u1',
        token: 'token',
      });
      await useChatStore.getState().sendMessage('embed me', 'embed');
    });

    const { messages } = useChatStore.getState();
    expect(messages).toHaveLength(2);
    expect(messages[1].content).toContain('Vecteurs: 1');
    expect(requestEmbedding).toHaveBeenCalledWith('embed me');
  });

  it('streams an on-device reply without a session or server fallback', async () => {
    const generate = jest.fn(async () => {
      useInferenceStore.setState({
        generation: {
          requestId: 'local-1',
          text: 'Salut depuis Core ML',
          generatedTokens: 5,
          tokensPerSecond: 3.5,
        },
      });
      return {
        requestId: 'local-1',
        text: 'Salut depuis Core ML',
        promptTokens: 24,
        generatedTokens: 5,
        duration: 1.4,
        tokensPerSecond: 3.5,
        finishReason: 'eos',
        modelId: 'example/qwen-coreml',
      };
    });
    useInferenceStore.setState({
      backend: 'on-device',
      generate,
    });

    await act(async () => {
      await useChatStore.getState().sendMessage('Bonjour local', 'chat');
    });

    const { messages, loading } = useChatStore.getState();
    expect(loading).toBe(false);
    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      role: 'user',
      metadata: { inferenceBackend: 'on-device' },
    });
    expect(messages[1]).toMatchObject({
      role: 'assistant',
      content: 'Salut depuis Core ML',
      metadata: {
        source: 'on-device',
        modelId: 'example/qwen-coreml',
        generatedTokens: 5,
        finishReason: 'eos',
      },
    });
    expect(generate).toHaveBeenCalledWith({
      messages: [{ role: 'user', content: 'Bonjour local' }],
    });
    expect(postConversationMessage).not.toHaveBeenCalled();
    expect(requestEmbedding).not.toHaveBeenCalled();
  });

  it('rejects a second local send without appending a phantom turn', async () => {
    const pending = deferred<{
      requestId: string;
      text: string;
      promptTokens: number;
      generatedTokens: number;
      duration: number;
      tokensPerSecond: number;
      finishReason: string;
      modelId: string;
    }>();
    const generate = jest.fn(() => pending.promise);
    useInferenceStore.setState({
      backend: 'on-device',
      generate,
    });

    const firstSend = useChatStore
      .getState()
      .sendMessage('Première demande', 'chat');
    await expect(
      useChatStore.getState().sendMessage('Deuxième demande', 'chat'),
    ).rejects.toThrow('Une requête est déjà en cours.');

    expect(useChatStore.getState().messages).toHaveLength(2);
    expect(
      useChatStore
        .getState()
        .messages.some((message) => message.content === 'Deuxième demande'),
    ).toBe(false);
    expect(generate).toHaveBeenCalledTimes(1);

    pending.resolve({
      requestId: 'local-1',
      text: 'Réponse finale',
      promptTokens: 4,
      generatedTokens: 2,
      duration: 0.5,
      tokensPerSecond: 4,
      finishReason: 'eos',
      modelId: 'example/model',
    });
    await expect(firstSend).resolves.toBeUndefined();
  });

  it('blocks backend changes while a local request is active', async () => {
    const pending = deferred<{
      requestId: string;
      text: string;
      promptTokens: number;
      generatedTokens: number;
      duration: number;
      tokensPerSecond: number;
      finishReason: string;
      modelId: string;
    }>();
    useInferenceStore.setState({
      backend: 'on-device',
      generate: jest.fn(() => pending.promise),
    });

    const send = useChatStore.getState().sendMessage('Continue', 'chat');
    await expect(
      useChatStore.getState().setInferenceBackend('server'),
    ).rejects.toThrow('annulez-la avant de changer de backend');
    expect(useInferenceStore.getState().backend).toBe('on-device');

    pending.resolve({
      requestId: 'local-2',
      text: 'Terminé',
      promptTokens: 4,
      generatedTokens: 1,
      duration: 0.25,
      tokensPerSecond: 4,
      finishReason: 'eos',
      modelId: 'example/model',
    });
    await send;
  });

  it('blocks backend changes when the native store still owns an active request', async () => {
    useInferenceStore.setState({
      backend: 'on-device',
      activeRequestId: 'active-native-request',
      status: {
        ...useInferenceStore.getState().status,
        phase: 'generating',
      },
    });

    await expect(
      useChatStore.getState().setInferenceBackend('server'),
    ).rejects.toThrow('annulez-la avant de changer de backend');
    expect(useInferenceStore.getState().backend).toBe('on-device');
  });

  it('discards server history that resolves after selecting on-device mode', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    expect(fetchConversationHistory).toHaveBeenCalledTimes(1);
    useInferenceStore.getState().setBackend('on-device');
    history.resolve([
      {
        query: 'Serveur',
        response: 'Réponse tardive',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    expect(useChatStore.getState().historyLoading).toBe(false);
    expect(useChatStore.getState().messages).toHaveLength(0);
  });

  it('hydrates the persisted backend before deciding whether to contact the server', async () => {
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
    });
    const hasHydrated = jest
      .spyOn(useInferenceStore.persist, 'hasHydrated')
      .mockReturnValue(false);
    const rehydrate = jest
      .spyOn(useInferenceStore.persist, 'rehydrate')
      .mockImplementation(async () => {
        useInferenceStore.setState({ backend: 'on-device' });
      });

    try {
      await useChatStore.getState().initialize();

      expect(rehydrate).toHaveBeenCalledTimes(1);
      expect(fetchConversationHistory).not.toHaveBeenCalled();
      expect(createRealtimeClient).not.toHaveBeenCalled();
      expect(useChatStore.getState().connection).toMatchObject({
        status: 'offline',
        detail: expect.stringContaining('Modèle et chat locaux'),
      });
    } finally {
      hasHydrated.mockRestore();
      rehydrate.mockRestore();
    }
  });

  it('keeps local prompt history isolated between signed-in accounts', async () => {
    let generationCount = 0;
    const generate = jest.fn(
      async (
        request: CoreMLGenerationRequest,
      ): Promise<CoreMLGenerationResult> => {
        generationCount += 1;
        return {
          requestId: `local-${generationCount}`,
          text: `Réponse ${generationCount}`,
          promptTokens: request.messages.length,
          generatedTokens: 2,
          duration: 0.5,
          tokensPerSecond: 4,
          finishReason: 'eos',
          modelId: 'example/model',
        };
      },
    );
    useInferenceStore.setState({ backend: 'on-device', generate });
    useChatStore.setState({
      session: { username: 'Alice', token: 'alice-token' },
    });

    await useChatStore.getState().sendMessage('Secret Alice', 'chat');
    useChatStore.setState({
      session: { username: 'Bob', token: 'bob-token' },
    });
    await useChatStore.getState().sendMessage('Question Bob', 'chat');

    expect(generate).toHaveBeenCalledTimes(2);
    expect(generate.mock.calls[1][0].messages).toEqual([
      { role: 'user', content: 'Question Bob' },
    ]);
    expect(
      useChatStore
        .getState()
        .messages.filter(
          (message) => message.metadata?.localOwnerId === 'account:Alice',
        ),
    ).toHaveLength(2);
    expect(
      useChatStore
        .getState()
        .messages.filter(
          (message) => message.metadata?.localOwnerId === 'account:Bob',
        ),
    ).toHaveLength(2);
  });

  it('preserves server messages received while history refresh is in flight', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
      messages: [
        {
          id: 'old-server-message',
          role: 'assistant',
          content: 'Ancienne copie serveur',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'history' },
        },
        {
          id: 'local-message',
          role: 'assistant',
          content: 'Conversation locale',
          createdAt: new Date(),
          metadata: {
            inferenceBackend: 'on-device',
            source: 'on-device',
            localOwnerId: 'account:u1',
          },
        },
      ],
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    expect(fetchConversationHistory).toHaveBeenCalledTimes(1);
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        {
          id: 'live-server-message',
          role: 'assistant',
          content: 'Message temps réel',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
      ],
    }));
    history.resolve([
      {
        query: 'Historique frais',
        response: 'Réponse fraîche',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    const contents = useChatStore
      .getState()
      .messages.map((message) => message.content);
    expect(contents).toEqual(
      expect.arrayContaining([
        'Conversation locale',
        'Historique frais',
        'Réponse fraîche',
        'Message temps réel',
      ]),
    );
    expect(contents).not.toContain('Ancienne copie serveur');
  });

  it('deduplicates a preserved in-flight turn that is also returned by history', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    expect(fetchConversationHistory).toHaveBeenCalledTimes(1);
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        {
          id: 'live-duplicate-user',
          role: 'user',
          content: 'Question synchronisée',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
        {
          id: 'live-duplicate-assistant',
          role: 'assistant',
          content: 'Réponse synchronisée',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
        {
          id: 'live-unrelated-user',
          role: 'user',
          content: 'Question indépendante',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
        {
          id: 'live-unrelated-assistant',
          role: 'assistant',
          content: 'Réponse indépendante',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
      ],
    }));
    history.resolve([
      {
        query: 'Question synchronisée',
        response: 'Réponse synchronisée',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    const messages = useChatStore.getState().messages;
    expect(
      messages.filter((message) => message.content === 'Question synchronisée'),
    ).toHaveLength(1);
    expect(
      messages.filter((message) => message.content === 'Réponse synchronisée'),
    ).toHaveLength(1);
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'live-unrelated-user' }),
        expect.objectContaining({ id: 'live-unrelated-assistant' }),
      ]),
    );
  });

  it('keeps both halves of a server turn split by the history snapshot', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
      messages: [
        {
          id: 'in-flight-user',
          role: 'user',
          content: 'Question en vol',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        {
          id: 'in-flight-assistant',
          role: 'assistant',
          content: 'Réponse en vol',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    }));
    history.resolve([]);
    await refresh;

    expect(useChatStore.getState().messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'in-flight-user' }),
        expect.objectContaining({ id: 'in-flight-assistant' }),
      ]),
    );
  });

  it('does not duplicate the assistant when history includes a split in-flight turn', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
      messages: [
        {
          id: 'in-flight-user',
          role: 'user',
          content: 'Question en vol',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        {
          id: 'in-flight-assistant',
          role: 'assistant',
          content: 'Réponse en vol',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    }));
    history.resolve([
      {
        query: 'Question en vol',
        response: 'Réponse en vol',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    const messages = useChatStore.getState().messages;
    expect(
      messages.filter((message) => message.content === 'Question en vol'),
    ).toHaveLength(1);
    expect(
      messages.filter((message) => message.content === 'Réponse en vol'),
    ).toHaveLength(1);
  });

  it('rejoins a split turn when an unrelated realtime turn arrives between its halves', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
      messages: [
        {
          id: 'split-user',
          role: 'user',
          content: 'Question principale',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        {
          id: 'realtime-user-between',
          role: 'user',
          content: 'Question temps réel',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
        {
          id: 'realtime-assistant-between',
          role: 'assistant',
          content: 'Réponse temps réel',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'realtime' },
        },
        {
          id: 'split-assistant',
          role: 'assistant',
          content: 'Réponse principale',
          createdAt: new Date(),
          metadata: { inferenceBackend: 'server', source: 'chat' },
        },
      ],
    }));
    history.resolve([
      {
        query: 'Question principale',
        response: 'Réponse principale',
        timestamp: new Date().toISOString(),
      },
      {
        query: 'Question temps réel',
        response: 'Réponse temps réel',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    const contents = useChatStore
      .getState()
      .messages.map((message) => message.content);
    expect(contents).toHaveLength(4);
    expect(
      contents.filter((content) => content === 'Question principale'),
    ).toHaveLength(1);
    expect(
      contents.filter((content) => content === 'Réponse principale'),
    ).toHaveLength(1);
    expect(
      contents.filter((content) => content === 'Question temps réel'),
    ).toHaveLength(1);
    expect(
      contents.filter((content) => content === 'Réponse temps réel'),
    ).toHaveLength(1);
  });

  it('uses history fingerprint counts without collapsing a repeated turn', async () => {
    const history =
      deferred<Array<{ query: string; response: string; timestamp: string }>>();
    (fetchConversationHistory as jest.Mock).mockReturnValueOnce(
      history.promise,
    );
    useChatStore.setState({
      session: { username: 'u1', token: 'token' },
    });

    const refresh = useChatStore.getState().refreshHistory();
    await flushPromises();
    useChatStore.setState((state) => ({
      messages: [
        ...state.messages,
        ...[1, 2].flatMap((turn) => [
          {
            id: `repeated-user-${turn}`,
            role: 'user' as const,
            content: 'Même question',
            createdAt: new Date(),
            metadata: {
              inferenceBackend: 'server' as const,
              source: 'realtime' as const,
            },
          },
          {
            id: `repeated-assistant-${turn}`,
            role: 'assistant' as const,
            content: 'Même réponse',
            createdAt: new Date(),
            metadata: {
              inferenceBackend: 'server' as const,
              source: 'realtime' as const,
            },
          },
        ]),
      ],
    }));
    history.resolve([
      {
        query: 'Même question',
        response: 'Même réponse',
        timestamp: new Date().toISOString(),
      },
    ]);
    await refresh;

    const messages = useChatStore.getState().messages;
    expect(
      messages.filter((message) => message.content === 'Même question'),
    ).toHaveLength(2);
    expect(
      messages.filter((message) => message.content === 'Même réponse'),
    ).toHaveLength(2);
    expect(messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'repeated-user-2' }),
        expect.objectContaining({ id: 'repeated-assistant-2' }),
      ]),
    );
  });

  it('strips a legacy persisted token while retaining its local owner attribution', async () => {
    const migrate = useChatStore.persist.getOptions().migrate;
    if (!migrate) {
      throw new Error('Migration chat manquante');
    }

    const migrated = (await migrate(
      {
        session: { username: 'Alice', token: 'alice-token' },
        messages: [
          {
            id: 'legacy-user',
            role: 'user',
            content: 'Question locale',
            createdAt: new Date().toISOString(),
            metadata: {
              inferenceBackend: 'on-device',
              source: 'on-device',
            },
          },
          {
            id: 'legacy-assistant',
            role: 'assistant',
            content: 'Réponse locale',
            createdAt: new Date().toISOString(),
            metadata: {
              inferenceBackend: 'on-device',
              source: 'on-device',
              finishReason: 'eos',
            },
          },
        ],
        mode: 'chat',
        quickActions: ['code', 'summarize', 'explain'],
      },
      5,
    )) as {
      session: unknown;
      messages: Array<{
        metadata?: { localOwnerId?: string };
      }>;
    };

    expect(migrated.session).toBeNull();
    expect(JSON.stringify(migrated)).not.toContain('alice-token');
    expect(migrated.messages).toHaveLength(2);
    expect(migrated.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          metadata: expect.objectContaining({
            localOwnerId: 'account:Alice',
          }),
        }),
      ]),
    );
    expect(
      migrated.messages.every(
        (message) => message.metadata?.localOwnerId === 'account:Alice',
      ),
    ).toBe(true);
  });
});
