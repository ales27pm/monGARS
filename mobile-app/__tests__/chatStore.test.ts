import { act } from '@testing-library/react-native';
import { useChatStore } from '../src/store/chatStore';
import type { Message } from '../src/types';

jest.mock('../src/services/chatService', () => ({
  fetchChatHistory: jest.fn(),
  sendChatMessage: jest.fn(),
  streamChatReply: jest.fn(),
}));

import {
  fetchChatHistory,
  sendChatMessage,
  streamChatReply,
} from '../src/services/chatService';

describe('chatStore', () => {
  beforeEach(() => {
    jest.resetAllMocks();
    (fetchChatHistory as jest.Mock).mockResolvedValue([]);
    const storage = {
      getItem: jest.fn().mockResolvedValue(null),
      setItem: jest.fn().mockResolvedValue(undefined),
      removeItem: jest.fn().mockResolvedValue(undefined),
    };
    useChatStore.persist?.setOptions?.({ storage: storage as any });
    useChatStore.setState({
      messages: [],
      loading: false,
      error: null,
      token: null,
      messageCounter: 0,
      initialize: useChatStore.getState().initialize,
      setToken: useChatStore.getState().setToken,
      sendMessage: useChatStore.getState().sendMessage,
      pushMessage: useChatStore.getState().pushMessage,
      clearError: useChatStore.getState().clearError,
      generateMessageId: useChatStore.getState().generateMessageId,
    });
    useChatStore.persist?.clearStorage?.();
  });

  it('loads chat history when a token is present', async () => {
    const history = [
      {
        id: '1',
        role: 'assistant',
        content: 'Bonjour',
        timestamp: new Date().toISOString(),
      },
    ];
    (fetchChatHistory as jest.Mock).mockResolvedValue(history);

    await act(async () => {
      useChatStore.getState().setToken('token');
      await useChatStore.getState().initialize();
    });

    const state = useChatStore.getState();
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0].content).toBe('Bonjour');
    expect(state.loading).toBe(false);
    expect(fetchChatHistory).toHaveBeenCalledWith('token');
  });

  it('records an error when trying to send without a token', async () => {
    await act(async () => {
      await expect(
        useChatStore.getState().sendMessage('hello', 'chat'),
      ).rejects.toThrow('Token missing');
    });

    expect(useChatStore.getState().error).toBe('Token missing');
    expect(sendChatMessage).not.toHaveBeenCalled();
    expect(streamChatReply).not.toHaveBeenCalled();
  });

  it('streams assistant updates during chat mode', async () => {
    const assistantMessage: Message = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'Salut!',
      createdAt: new Date(),
    };
    (streamChatReply as jest.Mock).mockImplementation(async ({ onToken }) => {
      onToken(assistantMessage);
    });

    await act(async () => {
      useChatStore.getState().setToken('token');
      await useChatStore.getState().initialize();
      await useChatStore.getState().sendMessage('hello', 'chat');
    });

    const { messages, loading } = useChatStore.getState();
    expect(messages.some((message) => message.role === 'user')).toBe(true);
    expect(messages.some((message) => message.id === assistantMessage.id)).toBe(true);
    expect(loading).toBe(false);
    expect(streamChatReply).toHaveBeenCalled();
  });

  it('handles embedding mode by awaiting the REST response', async () => {
    const response = {
      id: 'embedding-1',
      content: 'Vector generated',
      timestamp: new Date().toISOString(),
    };
    (sendChatMessage as jest.Mock).mockResolvedValue(response);

    await act(async () => {
      useChatStore.getState().setToken('token');
      await useChatStore.getState().initialize();
      await useChatStore.getState().sendMessage('embed me', 'embedding');
    });

    const { messages } = useChatStore.getState();
    expect(messages).toHaveLength(2);
    expect(messages[1].content).toBe('Vector generated');
    expect(sendChatMessage).toHaveBeenCalledWith('token', 'embed me', 'embedding');
  });
});
