import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { produce } from 'immer';
import { z } from 'zod';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  fetchChatHistory,
  sendChatMessage,
  streamChatReply,
} from '../services/chatService';
import type { Message } from '../types';

type ChatState = {
  messages: Message[];
  loading: boolean;
  error: string | null;
  token: string | null;
  initialize: () => Promise<void>;
  setToken: (token: string | null) => void;
  sendMessage: (content: string, mode: 'chat' | 'embedding') => Promise<void>;
  pushMessage: (message: Message) => void;
  clearError: () => void;
};

const historySchema = z.object({
  id: z.string(),
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
  timestamp: z.string(),
});

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      messages: [],
      loading: false,
      error: null,
      token: null,
      initialize: async () => {
        const token = get().token;
        if (!token) {
          return;
        }
        set({ loading: true, error: null });
        try {
          const payload = await fetchChatHistory(token);
          const history = z.array(historySchema).parse(payload);
          set({
            messages: history.map((item) => ({
              id: item.id,
              role: item.role,
              content: item.content,
              createdAt: new Date(item.timestamp),
            })),
            loading: false,
          });
        } catch (error) {
          console.error('[ChatStore] history failure', error);
          set({ error: 'Unable to load history', loading: false });
        }
      },
      setToken: (token) => {
        const previous = get().token;
        set({ token });

        if (token && token !== previous) {
          get()
            .initialize()
            .catch((error) => {
              console.error('[ChatStore] initialize failure', error);
            });
        }

        if (!token) {
          set({ messages: [] });
        }
      },
      sendMessage: async (content, mode) => {
        const { token } = get();
        if (!token) {
          set({ error: 'Token missing' });
          return;
        }
        const userMessage: Message = {
          id: `${Date.now()}`,
          role: 'user',
          content,
          createdAt: new Date(),
        };
        set(
          produce<ChatState>((draft) => {
            draft.messages.push(userMessage);
            draft.loading = true;
            draft.error = null;
          }),
        );
        try {
          if (mode === 'embedding') {
            const response = await sendChatMessage(token, content, mode);
            set(
              produce<ChatState>((draft) => {
                draft.messages.push({
                  id: response.id,
                  role: 'assistant',
                  content: response.content,
                  createdAt: new Date(response.timestamp),
                });
                draft.loading = false;
              }),
            );
            return;
          }
          await streamChatReply({
            token,
            content,
            onToken: (partial) => {
              set(
                produce<ChatState>((draft) => {
                  const existing = draft.messages.find(
                    (message) => message.id === partial.id,
                  );
                  if (existing) {
                    existing.content = partial.content;
                    existing.createdAt = partial.createdAt;
                  } else {
                    draft.messages.push({ ...partial });
                  }
                }),
              );
            },
          });
          set({ loading: false });
        } catch (error) {
          console.error('[ChatStore] send failure', error);
          set({ loading: false, error: 'Unable to send message' });
        }
      },
      pushMessage: (message) =>
        set(
          produce<ChatState>((draft) => {
            draft.messages.push(message);
          }),
        ),
      clearError: () => set({ error: null }),
    }),
    {
      name: 'mongars-chat',
      storage: createJSONStorage(() => AsyncStorage, {
        reviver: (key, value) => {
          if (key === 'createdAt' && typeof value === 'string') {
            const parsed = new Date(value);
            return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
          }
          return value;
        },
      }),
      version: 2,
      migrate: (persistedState) => {
        if (!persistedState) {
          return persistedState as unknown as {
            messages?: Message[];
          };
        }

        const state = persistedState as {
          messages?: Array<Partial<Message> & { createdAt?: string | Date }>;
        };

        if (!state.messages) {
          return { messages: [] };
        }

        return {
          messages: state.messages.map((message) => ({
            ...message,
            createdAt: message.createdAt
              ? new Date(message.createdAt)
              : new Date(),
          })) as Message[],
        };
      },
      partialize: (state) => ({ messages: state.messages }),
    },
  ),
);
