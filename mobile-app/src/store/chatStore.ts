import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { produce } from 'immer';
import {
  fetchConversationHistory,
  fetchQuickActions,
  postConversationMessage,
  requestEmbedding,
} from '../services/chatService';
import { createRealtimeClient } from '../services/realtimeService';
import type {
  ChatMode,
  ConnectionSnapshot,
  Message,
  QuickAction,
  UserSession,
} from '../types';
import {
  buildMessageId,
  buildRealtimeFingerprint,
  formatEmbeddingResult,
  mapHistoryToMessages,
  orderQuickActions,
} from '../utils/conversation';

type NoticeTone = 'info' | 'success' | 'warning' | 'danger';

type Notice = {
  tone: NoticeTone;
  message: string;
};

type ChatState = {
  session: UserSession | null;
  messages: Message[];
  loading: boolean;
  historyLoading: boolean;
  error: string | null;
  notice: Notice | null;
  mode: ChatMode;
  quickActions: QuickAction[];
  connection: ConnectionSnapshot;
  realtimeSuppression: string[];
  initialize: () => Promise<void>;
  setSession: (session: UserSession | null) => Promise<void>;
  sendMessage: (content: string, mode?: ChatMode) => Promise<void>;
  refreshHistory: () => Promise<void>;
  requestQuickActions: (prompt: string) => Promise<void>;
  setMode: (mode: ChatMode) => void;
  retryRealtime: () => void;
  clearError: () => void;
  clearNotice: () => void;
  logout: () => Promise<void>;
};

const DEFAULT_QUICK_ACTIONS: QuickAction[] = ['code', 'summarize', 'explain'];

const DEFAULT_CONNECTION: ConnectionSnapshot = {
  status: 'offline',
  detail: 'Aucune session active',
  connectedAt: null,
  lastMessageAt: null,
  latencyMs: null,
  reconnectAttempt: 0,
};

let realtimeClient: ReturnType<typeof createRealtimeClient> | null = null;

function isDuplicateRealtimePair(
  messages: Message[],
  query: string,
  response: string,
) {
  const lastUser = [...messages]
    .reverse()
    .find((message) => message.role === 'user');
  const lastAssistant = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant');
  return (
    lastUser?.content.trim() === query.trim() &&
    lastAssistant?.content.trim() === response.trim()
  );
}

function upsertRealtimeMessages(
  messages: Message[],
  item: {
    query: string;
    response: string;
    timestamp: string;
  },
): Message[] {
  if (isDuplicateRealtimePair(messages, item.query, item.response)) {
    return messages;
  }

  const createdAt = new Date(item.timestamp);
  const safeDate = Number.isNaN(createdAt.getTime()) ? new Date() : createdAt;

  return [
    ...messages,
    {
      id: buildMessageId('realtime-user'),
      role: 'user',
      content: item.query,
      createdAt: safeDate,
      metadata: {
        mode: 'chat',
        source: 'realtime',
      },
    },
    {
      id: buildMessageId('realtime-assistant'),
      role: 'assistant',
      content: item.response,
      createdAt: safeDate,
      metadata: {
        mode: 'chat',
        source: 'realtime',
      },
    },
  ];
}

function ensureRealtime(
  set: (
    partial: Partial<ChatState> | ((state: ChatState) => Partial<ChatState>),
  ) => void,
  get: () => ChatState,
) {
  if (realtimeClient) {
    return realtimeClient;
  }

  realtimeClient = createRealtimeClient({
    onStatus: (status) => {
      set((state) => ({
        connection: {
          ...state.connection,
          ...status,
          detail: status.detail ?? state.connection.detail,
          connectedAt:
            status.connectedAt === undefined
              ? state.connection.connectedAt
              : status.connectedAt,
          lastMessageAt:
            status.lastMessageAt === undefined
              ? state.connection.lastMessageAt
              : status.lastMessageAt,
          latencyMs:
            status.latencyMs === undefined
              ? state.connection.latencyMs
              : status.latencyMs,
          reconnectAttempt:
            status.reconnectAttempt === undefined
              ? state.connection.reconnectAttempt
              : status.reconnectAttempt,
        },
      }));
    },
    onHistory: (items) => {
      if (get().messages.length > 0) {
        return;
      }

      set({
        messages: mapHistoryToMessages(items),
      });
    },
    onMessage: (item) => {
      const fingerprint = buildRealtimeFingerprint(item);
      set(
        produce<ChatState>((draft) => {
          const suppressionIndex =
            draft.realtimeSuppression.indexOf(fingerprint);
          if (suppressionIndex !== -1) {
            draft.realtimeSuppression.splice(suppressionIndex, 1);
            draft.connection.lastMessageAt = new Date(item.timestamp);
            return;
          }

          draft.messages = upsertRealtimeMessages(draft.messages, item);
          draft.connection.lastMessageAt = new Date(item.timestamp);
        }),
      );
    },
    onError: (message) => {
      set({
        error: message,
        notice: {
          tone: 'warning',
          message,
        },
      });
    },
  });

  return realtimeClient;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      session: null,
      messages: [],
      loading: false,
      historyLoading: false,
      error: null,
      notice: null,
      mode: 'chat',
      quickActions: [...DEFAULT_QUICK_ACTIONS],
      connection: DEFAULT_CONNECTION,
      realtimeSuppression: [],
      initialize: async () => {
        const session = get().session;
        if (!session) {
          set({
            connection: DEFAULT_CONNECTION,
          });
          return;
        }

        await get().refreshHistory();
        await ensureRealtime(set, get).open(session);
      },
      setSession: async (session) => {
        if (!session) {
          realtimeClient?.close('logout');
          set({
            session: null,
            messages: [],
            error: null,
            notice: {
              tone: 'info',
              message: 'Session fermee.',
            },
            connection: DEFAULT_CONNECTION,
            realtimeSuppression: [],
          });
          return;
        }

        set({
          session,
          messages: [],
          error: null,
          notice: {
            tone: 'success',
            message: `Connecte en tant que ${session.username}.`,
          },
          connection: {
            ...DEFAULT_CONNECTION,
            status: 'connecting',
            detail: 'Initialisation de la session…',
          },
          realtimeSuppression: [],
        });

        await get().initialize();
      },
      sendMessage: async (content, mode) => {
        const trimmed = content.trim();
        if (!trimmed) {
          return;
        }

        const session = get().session;
        if (!session) {
          const error = 'Session absente.';
          set({
            error,
            notice: {
              tone: 'warning',
              message: 'Connectez-vous avant d envoyer un message.',
            },
          });
          throw new Error(error);
        }

        const activeMode = mode ?? get().mode;
        const userMessage: Message = {
          id: buildMessageId('user'),
          role: 'user',
          content: trimmed,
          createdAt: new Date(),
          metadata: {
            mode: activeMode,
            source: activeMode === 'embed' ? 'embedding' : 'chat',
          },
        };

        set(
          produce<ChatState>((draft) => {
            draft.messages.push(userMessage);
            draft.loading = true;
            draft.error = null;
            draft.notice = {
              tone: 'info',
              message:
                activeMode === 'embed'
                  ? 'Generation d embedding…'
                  : 'Generation de reponse…',
            };
          }),
        );

        try {
          if (activeMode === 'embed') {
            const embedding = await requestEmbedding(trimmed);
            set(
              produce<ChatState>((draft) => {
                draft.messages.push({
                  id: buildMessageId('embedding'),
                  role: 'assistant',
                  content: formatEmbeddingResult(embedding),
                  createdAt: new Date(),
                  metadata: {
                    mode: 'embed',
                    source: 'embedding',
                    embedding: {
                      backend: embedding.backend,
                      model: embedding.model,
                      dims: embedding.dims,
                      count: embedding.count,
                      normalised: embedding.normalised,
                    },
                  },
                });
                draft.loading = false;
                draft.notice = {
                  tone: 'success',
                  message: 'Embedding genere.',
                };
              }),
            );
            return;
          }

          const response = await postConversationMessage(session, trimmed);
          const fingerprint = buildRealtimeFingerprint({
            query: trimmed,
            response: response.response,
          });

          set(
            produce<ChatState>((draft) => {
              draft.realtimeSuppression.push(fingerprint);
              draft.messages.push({
                id: buildMessageId('assistant'),
                role: 'assistant',
                content: response.response,
                createdAt: new Date(),
                metadata: {
                  mode: 'chat',
                  source: 'chat',
                  confidence: response.confidence,
                  processingTime: response.processingTime,
                  speechTurn: response.speechTurn,
                },
              });
              draft.loading = false;
              draft.connection.lastMessageAt = new Date();
              draft.notice = {
                tone: 'success',
                message: 'Reponse recue.',
              };
            }),
          );
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Envoi impossible.';
          set({
            loading: false,
            error: message,
            notice: {
              tone: 'danger',
              message,
            },
          });
          throw error;
        }
      },
      refreshHistory: async () => {
        const session = get().session;
        if (!session) {
          return;
        }

        set({
          historyLoading: true,
          error: null,
        });

        try {
          const history = await fetchConversationHistory(session);
          set({
            messages: mapHistoryToMessages(history),
            historyLoading: false,
            notice: history.length
              ? {
                  tone: 'info',
                  message: 'Historique synchronise.',
                }
              : {
                  tone: 'info',
                  message: 'Aucun historique disponible.',
                },
          });
        } catch (error) {
          const message =
            error instanceof Error ? error.message : 'Historique indisponible.';
          set({
            historyLoading: false,
            error: message,
            notice: {
              tone: 'warning',
              message,
            },
          });
        }
      },
      requestQuickActions: async (prompt) => {
        const session = get().session;
        if (!session || prompt.trim().length < 3 || get().mode !== 'chat') {
          if (get().quickActions !== DEFAULT_QUICK_ACTIONS) {
            set({
              quickActions: [...DEFAULT_QUICK_ACTIONS],
            });
          }
          return;
        }

        try {
          const response = await fetchQuickActions(session, prompt.trim());
          set({
            quickActions: orderQuickActions(response.actions),
          });
        } catch (error) {
          console.debug('[chatStore] suggestions unavailable', error);
          set({
            quickActions: [...DEFAULT_QUICK_ACTIONS],
          });
        }
      },
      setMode: (mode) => {
        set({
          mode,
          notice: {
            tone: 'info',
            message:
              mode === 'embed' ? 'Mode embedding actif.' : 'Mode chat actif.',
          },
        });
      },
      retryRealtime: () => {
        const session = get().session;
        if (!session) {
          return;
        }
        ensureRealtime(set, get).reconnect();
      },
      clearError: () => set({ error: null }),
      clearNotice: () => set({ notice: null }),
      logout: async () => {
        await get().setSession(null);
      },
    }),
    {
      name: 'mongars-chat',
      storage: createJSONStorage(() => AsyncStorage, {
        reviver: (key, value) => {
          if (key === 'createdAt' && typeof value === 'string') {
            const parsed = new Date(value);
            return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
          }
          if (
            (key === 'connectedAt' || key === 'lastMessageAt') &&
            typeof value === 'string'
          ) {
            const parsed = new Date(value);
            return Number.isNaN(parsed.getTime()) ? null : parsed;
          }
          return value;
        },
      }),
      version: 3,
      migrate: (persistedState) => {
        if (!persistedState) {
          return persistedState as ChatState;
        }

        const state = persistedState as Partial<ChatState> & {
          messages?: Array<Partial<Message> & { createdAt?: string | Date }>;
          session?: { username?: string; token?: string };
        };

        return {
          session:
            state.session?.username && state.session?.token
              ? {
                  username: state.session.username,
                  token: state.session.token,
                }
              : null,
          messages: (state.messages ?? []).map((message) => ({
            ...message,
            createdAt: message.createdAt
              ? new Date(message.createdAt)
              : new Date(),
          })) as Message[],
          mode: state.mode ?? 'chat',
          quickActions: orderQuickActions(state.quickActions),
        };
      },
      partialize: (state) => ({
        session: state.session,
        messages: state.messages,
        mode: state.mode,
        quickActions: state.quickActions,
      }),
    },
  ),
);
