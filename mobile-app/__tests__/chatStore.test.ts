import { act } from '@testing-library/react-native';
import { useChatStore } from '../src/store/chatStore';

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

describe('chatStore', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (createRealtimeClient as jest.Mock).mockReturnValue({
      open: jest.fn().mockResolvedValue(undefined),
      close: jest.fn(),
      reconnect: jest.fn(),
    });
    (fetchConversationHistory as jest.Mock).mockResolvedValue([]);
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
});
