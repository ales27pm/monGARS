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
import {
  ensureInferenceStoreHydrated,
  useInferenceStore,
} from './inferenceStore';
import type {
  ChatMode,
  ConnectionSnapshot,
  InferenceBackend,
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
  setInferenceBackend: (backend: InferenceBackend) => Promise<void>;
  setSession: (session: UserSession | null) => Promise<void>;
  sendMessage: (content: string, mode?: ChatMode) => Promise<void>;
  cancelGeneration: () => Promise<void>;
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

const LOCAL_CONNECTION: ConnectionSnapshot = {
  ...DEFAULT_CONNECTION,
  detail: 'Inference locale active — aucun serveur contacte',
};

let realtimeClient: ReturnType<typeof createRealtimeClient> | null = null;
let historyRefreshVersion = 0;

export function getLocalConversationOwner(session: UserSession | null): string {
  const username = session?.username.trim().toLowerCase();
  return username ? `account:${username}` : 'guest';
}

function isServerMessage(message: Message): boolean {
  return (message.metadata?.inferenceBackend ?? 'server') === 'server';
}

function isLocalMessageForOwner(message: Message, ownerId: string): boolean {
  return (
    message.metadata?.inferenceBackend === 'on-device' &&
    message.metadata.localOwnerId === ownerId
  );
}

function serverTurnChannel(message: Message): string {
  return `${message.metadata?.source ?? 'server'}::${
    message.metadata?.mode ?? 'chat'
  }`;
}

function unmatchedServerUserMessageIDs(messages: Message[]): Set<string> {
  const pendingUsers = new Map<string, Message[]>();
  messages.filter(isServerMessage).forEach((message) => {
    const channel = serverTurnChannel(message);
    if (message.role === 'user') {
      const users = pendingUsers.get(channel) ?? [];
      users.push(message);
      pendingUsers.set(channel, users);
      return;
    }
    if (message.role === 'assistant') {
      pendingUsers.get(channel)?.pop();
    }
  });

  return new Set(
    [...pendingUsers.values()].flatMap((users) =>
      users.map((message) => message.id),
    ),
  );
}

function removeTurnsAlreadyInHistory(
  messages: Message[],
  history: ReadonlyArray<{ query: string; response: string }>,
): Message[] {
  const remainingHistoryTurns = new Map<string, number>();
  history.forEach((item) => {
    const fingerprint = buildRealtimeFingerprint(item);
    remainingHistoryTurns.set(
      fingerprint,
      (remainingHistoryTurns.get(fingerprint) ?? 0) + 1,
    );
  });

  const pendingUserIndexes = new Map<string, number[]>();
  const duplicateMessageIndexes = new Set<number>();
  messages.forEach((message, index) => {
    const channel = serverTurnChannel(message);
    if (message.role === 'user') {
      const indexes = pendingUserIndexes.get(channel) ?? [];
      indexes.push(index);
      pendingUserIndexes.set(channel, indexes);
      return;
    }
    if (message.role === 'assistant') {
      const userMessageIndex = pendingUserIndexes.get(channel)?.pop();
      if (userMessageIndex === undefined) {
        return;
      }
      const userMessage = messages[userMessageIndex];
      const fingerprint = buildRealtimeFingerprint({
        query: userMessage.content,
        response: message.content,
      });
      const remainingMatches = remainingHistoryTurns.get(fingerprint) ?? 0;
      if (remainingMatches > 0) {
        remainingHistoryTurns.set(fingerprint, remainingMatches - 1);
        duplicateMessageIndexes.add(userMessageIndex);
        duplicateMessageIndexes.add(index);
      }
    }
  });

  return messages.filter((_, index) => !duplicateMessageIndexes.has(index));
}

function serverMessagesPreservedAcrossSnapshot(
  messages: Message[],
  snapshotMessageIDs: ReadonlySet<string>,
): Message[] {
  return messages.filter(
    (message) =>
      isServerMessage(message) && !snapshotMessageIDs.has(message.id),
  );
}

function isDuplicateRealtimePair(
  messages: Message[],
  query: string,
  response: string,
) {
  const lastUser = [...messages]
    .reverse()
    .find((message) => message.role === 'user' && isServerMessage(message));
  const lastAssistant = [...messages]
    .reverse()
    .find(
      (message) => message.role === 'assistant' && isServerMessage(message),
    );
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
        inferenceBackend: 'server',
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
        inferenceBackend: 'server',
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
      if (useInferenceStore.getState().backend !== 'server') {
        return;
      }
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
      if (useInferenceStore.getState().backend !== 'server') {
        return;
      }
      if (get().messages.some(isServerMessage)) {
        return;
      }

      set((state) => ({
        messages: [
          ...state.messages.filter((message) => !isServerMessage(message)),
          ...mapHistoryToMessages(items),
        ],
      }));
    },
    onMessage: (item) => {
      if (useInferenceStore.getState().backend !== 'server') {
        return;
      }
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
      if (useInferenceStore.getState().backend !== 'server') {
        return;
      }
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
        await ensureInferenceStoreHydrated();
        const inference = useInferenceStore.getState();
        if (inference.backend === 'on-device') {
          realtimeClient?.close('on-device');
          set({
            connection: LOCAL_CONNECTION,
            historyLoading: false,
            mode: 'chat',
            quickActions: [...DEFAULT_QUICK_ACTIONS],
          });
          await inference.initialize();
          return;
        }

        const session = get().session;
        if (!session) {
          set({
            connection: DEFAULT_CONNECTION,
          });
          return;
        }

        await get().refreshHistory();
        const currentSession = get().session;
        if (
          useInferenceStore.getState().backend !== 'server' ||
          currentSession?.username !== session.username ||
          currentSession?.token !== session.token
        ) {
          return;
        }
        await ensureRealtime(set, get).open(session);
      },
      setInferenceBackend: async (backend) => {
        await ensureInferenceStoreHydrated();
        const inference = useInferenceStore.getState();
        if (
          backend !== inference.backend &&
          (get().loading ||
            inference.activeRequestId !== null ||
            inference.status.phase === 'generating')
        ) {
          throw new Error(
            'Attendez la fin de la requête active ou annulez-la avant de changer de backend.',
          );
        }
        inference.setBackend(backend);
        if (backend === 'on-device') {
          historyRefreshVersion += 1;
          realtimeClient?.close('on-device');
          set({
            mode: 'chat',
            connection: LOCAL_CONNECTION,
            historyLoading: false,
            quickActions: [...DEFAULT_QUICK_ACTIONS],
            error: null,
            notice: {
              tone: 'info',
              message: 'Inference Core ML sur l iPhone activee.',
            },
          });
        } else {
          set({
            connection: get().session
              ? {
                  ...DEFAULT_CONNECTION,
                  status: 'connecting',
                  detail: 'Initialisation de la session…',
                }
              : DEFAULT_CONNECTION,
            error: null,
            notice: {
              tone: 'info',
              message: 'Backend serveur actif.',
            },
          });
        }
        await get().initialize();
      },
      setSession: async (session) => {
        historyRefreshVersion += 1;
        if (!session) {
          realtimeClient?.close('logout');
          set((state) => ({
            session: null,
            historyLoading: false,
            messages: state.messages.filter(
              (message) => !isServerMessage(message),
            ),
            error: null,
            notice: {
              tone: 'info',
              message: 'Session fermee.',
            },
            connection:
              useInferenceStore.getState().backend === 'on-device'
                ? LOCAL_CONNECTION
                : DEFAULT_CONNECTION,
            realtimeSuppression: [],
          }));
          return;
        }

        set((state) => ({
          session,
          messages: state.messages.filter(
            (message) => !isServerMessage(message),
          ),
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
        }));

        await get().initialize();
      },
      sendMessage: async (content, mode) => {
        const trimmed = content.trim();
        if (!trimmed) {
          return;
        }
        await ensureInferenceStoreHydrated();
        if (get().loading) {
          throw new Error('Une requête est déjà en cours.');
        }

        const inference = useInferenceStore.getState();
        const backend = inference.backend;
        const activeMode = mode ?? get().mode;
        const session = get().session;
        const localOwnerId = getLocalConversationOwner(session);
        if (backend === 'on-device' && activeMode === 'embed') {
          const error = 'Les embeddings restent disponibles sur le serveur.';
          set({
            error,
            mode: 'chat',
            notice: {
              tone: 'warning',
              message: error,
            },
          });
          throw new Error(error);
        }

        if (backend === 'server' && !session) {
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

        const userMessage: Message = {
          id: buildMessageId('user'),
          role: 'user',
          content: trimmed,
          createdAt: new Date(),
          metadata: {
            mode: activeMode,
            source:
              backend === 'on-device'
                ? 'on-device'
                : activeMode === 'embed'
                  ? 'embedding'
                  : 'chat',
            inferenceBackend: backend,
            ...(backend === 'on-device' ? { localOwnerId } : {}),
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
                  : backend === 'on-device'
                    ? 'Generation privee sur l iPhone…'
                    : 'Generation de reponse…',
            };
          }),
        );

        try {
          if (backend === 'on-device') {
            const assistantID = buildMessageId('on-device-assistant');
            set(
              produce<ChatState>((draft) => {
                draft.messages.push({
                  id: assistantID,
                  role: 'assistant',
                  content: '',
                  createdAt: new Date(),
                  metadata: {
                    mode: 'chat',
                    source: 'on-device',
                    inferenceBackend: 'on-device',
                    localOwnerId,
                  },
                });
              }),
            );

            const unsubscribe = useInferenceStore.subscribe(
              (state, previousState) => {
                if (state.generation?.text === previousState.generation?.text) {
                  return;
                }
                set(
                  produce<ChatState>((draft) => {
                    const assistant = draft.messages.find(
                      (message) => message.id === assistantID,
                    );
                    if (assistant) {
                      assistant.content = state.generation?.text ?? '';
                    }
                  }),
                );
              },
            );

            try {
              const localMessages = get()
                .messages.filter(
                  (message) =>
                    isLocalMessageForOwner(message, localOwnerId) &&
                    (message.role === 'user' || message.role === 'assistant') &&
                    message.content.trim().length > 0,
                )
                .map((message) => ({
                  role: message.role as 'user' | 'assistant',
                  content: message.content,
                }));
              const result = await useInferenceStore.getState().generate({
                messages: localMessages,
              });

              set(
                produce<ChatState>((draft) => {
                  const assistantIndex = draft.messages.findIndex(
                    (message) => message.id === assistantID,
                  );
                  const assistant = draft.messages[assistantIndex];
                  const hasGeneratedText = result.text.trim().length > 0;
                  if (assistant && hasGeneratedText) {
                    assistant.content = result.text;
                    assistant.metadata = {
                      mode: 'chat',
                      source: 'on-device',
                      inferenceBackend: 'on-device',
                      localOwnerId,
                      modelId: result.modelId,
                      promptTokens: result.promptTokens ?? undefined,
                      generatedTokens: result.generatedTokens,
                      tokensPerSecond: result.tokensPerSecond,
                      finishReason: result.finishReason,
                      processingTime: result.duration,
                    };
                  } else if (assistantIndex !== -1) {
                    draft.messages.splice(assistantIndex, 1);
                  }
                  draft.loading = false;
                  draft.notice = {
                    tone:
                      result.finishReason === 'cancelled'
                        ? 'info'
                        : hasGeneratedText
                          ? 'success'
                          : 'warning',
                    message:
                      result.finishReason === 'cancelled'
                        ? 'Generation locale arretee.'
                        : hasGeneratedText
                          ? 'Reponse generee sur l iPhone.'
                          : 'La generation locale s est terminee sans texte.',
                  };
                }),
              );
            } catch (localError) {
              const partial =
                useInferenceStore.getState().generation?.text.trim() ?? '';
              set(
                produce<ChatState>((draft) => {
                  const index = draft.messages.findIndex(
                    (message) => message.id === assistantID,
                  );
                  if (index === -1) {
                    return;
                  }
                  if (partial) {
                    draft.messages[index].content = partial;
                    draft.messages[index].metadata = {
                      mode: 'chat',
                      source: 'on-device',
                      inferenceBackend: 'on-device',
                      localOwnerId,
                      finishReason: 'error',
                    };
                  } else {
                    draft.messages.splice(index, 1);
                  }
                }),
              );
              throw localError;
            } finally {
              unsubscribe();
            }
            return;
          }

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
                    inferenceBackend: 'server',
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

          if (!session) {
            throw new Error('Session absente.');
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
                  inferenceBackend: 'server',
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
      cancelGeneration: async () => {
        const cancelled = await useInferenceStore.getState().cancelGeneration();
        if (cancelled) {
          set({
            notice: {
              tone: 'info',
              message: 'Annulation de la generation locale…',
            },
          });
        }
      },
      refreshHistory: async () => {
        await ensureInferenceStoreHydrated();
        if (useInferenceStore.getState().backend !== 'server') {
          return;
        }
        const session = get().session;
        if (!session) {
          return;
        }

        const refreshVersion = ++historyRefreshVersion;
        const snapshotServerMessages = get().messages.filter(isServerMessage);
        const unmatchedServerUserIDs = unmatchedServerUserMessageIDs(
          snapshotServerMessages,
        );
        const replacedServerMessageIDs = new Set(
          snapshotServerMessages
            .filter((message) => !unmatchedServerUserIDs.has(message.id))
            .map((message) => message.id),
        );

        set({
          historyLoading: true,
          error: null,
        });

        try {
          const history = await fetchConversationHistory(session);
          const currentSession = get().session;
          if (refreshVersion !== historyRefreshVersion) {
            return;
          }
          if (
            useInferenceStore.getState().backend !== 'server' ||
            currentSession?.username !== session.username ||
            currentSession?.token !== session.token
          ) {
            set({ historyLoading: false });
            return;
          }
          const historyMessages = mapHistoryToMessages(history);
          set((state) => {
            const preservedServerMessages =
              serverMessagesPreservedAcrossSnapshot(
                state.messages,
                replacedServerMessageIDs,
              );
            return {
              messages: [
                ...state.messages.filter(
                  (message) => !isServerMessage(message),
                ),
                ...historyMessages,
                ...removeTurnsAlreadyInHistory(
                  preservedServerMessages,
                  history,
                ),
              ],
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
            };
          });
        } catch (error) {
          const currentSession = get().session;
          if (refreshVersion !== historyRefreshVersion) {
            return;
          }
          if (
            useInferenceStore.getState().backend !== 'server' ||
            currentSession?.username !== session.username ||
            currentSession?.token !== session.token
          ) {
            set({ historyLoading: false });
            return;
          }
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
        await ensureInferenceStoreHydrated();
        const session = get().session;
        if (
          useInferenceStore.getState().backend !== 'server' ||
          !session ||
          prompt.trim().length < 3 ||
          get().mode !== 'chat'
        ) {
          if (get().quickActions !== DEFAULT_QUICK_ACTIONS) {
            set({
              quickActions: [...DEFAULT_QUICK_ACTIONS],
            });
          }
          return;
        }

        try {
          const response = await fetchQuickActions(session, prompt.trim());
          const currentSession = get().session;
          if (
            useInferenceStore.getState().backend !== 'server' ||
            currentSession?.username !== session.username ||
            currentSession?.token !== session.token ||
            get().mode !== 'chat'
          ) {
            return;
          }
          set({
            quickActions: orderQuickActions(response.actions),
          });
        } catch (error) {
          console.debug('[chatStore] suggestions unavailable', error);
          if (useInferenceStore.getState().backend !== 'server') {
            return;
          }
          set({
            quickActions: [...DEFAULT_QUICK_ACTIONS],
          });
        }
      },
      setMode: (mode) => {
        if (
          mode === 'embed' &&
          useInferenceStore.getState().backend === 'on-device'
        ) {
          set({
            mode: 'chat',
            notice: {
              tone: 'warning',
              message: 'Les embeddings exigent le backend serveur.',
            },
          });
          return;
        }
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
        if (useInferenceStore.getState().backend !== 'server') {
          return;
        }
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
      version: 5,
      migrate: (persistedState) => {
        if (!persistedState) {
          return persistedState as ChatState;
        }

        const state = persistedState as Partial<ChatState> & {
          messages?: Array<Partial<Message> & { createdAt?: string | Date }>;
          session?: { username?: string; token?: string };
        };

        const session: UserSession | null =
          state.session?.username && state.session?.token
            ? {
                username: state.session.username,
                token: state.session.token,
              }
            : null;
        const legacyLocalOwnerId = getLocalConversationOwner(session);

        return {
          session,
          messages: (state.messages ?? []).map((message) => {
            const inferenceBackend =
              message.metadata?.inferenceBackend ?? 'server';
            return {
              ...message,
              createdAt: message.createdAt
                ? new Date(message.createdAt)
                : new Date(),
              metadata: {
                ...message.metadata,
                inferenceBackend,
                ...(inferenceBackend === 'on-device'
                  ? {
                      localOwnerId:
                        message.metadata?.localOwnerId ?? legacyLocalOwnerId,
                    }
                  : {}),
              },
            };
          }) as Message[],
          mode: state.mode ?? 'chat',
          quickActions: orderQuickActions(state.quickActions),
        };
      },
      partialize: (state) => ({
        session: state.session,
        messages: state.messages.filter(
          (message) =>
            message.metadata?.source !== 'on-device' ||
            message.role !== 'assistant' ||
            Boolean(message.metadata.finishReason),
        ),
        mode: state.mode,
        quickActions: state.quickActions,
      }),
    },
  ),
);
