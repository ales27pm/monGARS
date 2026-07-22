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
  executeNativeAgent,
  shouldUseNativeAgent,
} from '../services/onDeviceAgentService';
import {
  approveNativeAgent,
  acknowledgePendingNativeAgentTrigger,
  cancelNativeAgent,
  createNativeAgentRunId,
  getPendingNativeAgentTrigger,
  nativeAgentModuleAvailable,
  rejectNativeAgent,
  requestNativeAgentPermission,
  type NativeAgentApprovalBinding,
  type NativeAgentHistoryMessage,
  type NativeAgentRunResult,
  type NativeAgentTriggerHandoff,
} from '../native/agent';
import {
  acknowledgeNativeAppIntentHandoff,
  discardNativeAppIntentHandoff,
  executeNativeAppIntentMemoryAction,
  getPendingNativeAppIntentHandoff,
  nativeAppIntentModuleAvailable,
  resolveNativeStoredAgentTrigger,
  setActiveNativeAppIntentProfile,
  type NativeAppIntentHandoff,
  type NativeResolvedStoredTrigger,
} from '../native/appIntents';
import { appIntentHandoffPrompt } from '../agent/appIntentHandoff';
import { normalizedAgentTriggerPrompt } from '../agent/triggerHandoff';
import { routeAgentIntent } from '../agent/routing';
import type { AgentIntent } from '../agent/types';
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

export type PendingAgentApproval = NativeAgentApprovalBinding & {
  displayName: string;
  risk: 'low' | 'moderate' | 'high' | 'critical';
  history: NativeAgentHistoryMessage[];
};

export type PendingAgentPermission = {
  ownerId: string;
  prompt: string;
  permission: Extract<
    NativeAgentRunResult,
    { status: 'permission_required' }
  >['permission'];
  history: NativeAgentHistoryMessage[];
};

export type PendingAgentTrigger = NativeAgentTriggerHandoff & {
  ownerId: string;
};

export type PendingAppIntentHandoff = NativeAppIntentHandoff & {
  ownerId: string;
  profileLabel: string;
  resolvedTrigger: NativeResolvedStoredTrigger | null;
};

type AgentTriggerReservation = {
  ownerId: string;
  triggerId: string;
};

type AppIntentReservation = {
  ownerId: string;
  handoffId: string;
  prompt: string;
  requestedIntent?: AgentIntent;
  allowedToolIds?: string[];
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
  pendingAgentApproval: PendingAgentApproval | null;
  pendingAgentPermission: PendingAgentPermission | null;
  pendingAgentTrigger: PendingAgentTrigger | null;
  pendingAppIntentHandoff: PendingAppIntentHandoff | null;
  activeAgentRunId: string | null;
  initialize: () => Promise<void>;
  refreshPendingAgentTrigger: (alreadyHydrated?: boolean) => Promise<void>;
  refreshPendingAppIntentHandoff: () => Promise<void>;
  setInferenceBackend: (backend: InferenceBackend) => Promise<void>;
  setSession: (session: UserSession | null) => Promise<void>;
  sendMessage: (
    content: string,
    mode?: ChatMode,
    triggerReservation?: AgentTriggerReservation,
    appIntentReservation?: AppIntentReservation,
  ) => Promise<void>;
  approvePendingAgent: () => Promise<void>;
  rejectPendingAgent: () => Promise<void>;
  requestPendingAgentPermission: () => Promise<void>;
  dismissPendingAgentPermission: () => void;
  runPendingAgentTrigger: () => Promise<void>;
  dismissPendingAgentTrigger: () => Promise<void>;
  runPendingAppIntentHandoff: () => Promise<'chat' | 'diagnostics' | null>;
  dismissPendingAppIntentHandoff: () => Promise<void>;
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
  detail:
    'Modèle et chat locaux; les outils choisis peuvent contacter leur fournisseur.',
};

let realtimeClient: ReturnType<typeof createRealtimeClient> | null = null;
let historyRefreshVersion = 0;
let agentTriggerFetchVersion = 0;
let appIntentFetchVersion = 0;
let activeAgentTriggerReservation: AgentTriggerReservation | null = null;
let activeAppIntentReservation: AppIntentReservation | null = null;
let sessionTransitionInProgress = false;

function assertStableSession(): void {
  if (sessionTransitionInProgress) {
    throw new Error(
      'Attendez la fin du changement de compte avant de lancer une action.',
    );
  }
}

function isActiveTriggerReservation(
  reservation: AgentTriggerReservation | undefined,
): reservation is AgentTriggerReservation {
  return (
    reservation !== undefined && activeAgentTriggerReservation === reservation
  );
}

function matchesPendingTrigger(
  pending: PendingAgentTrigger | null,
  reservation: AgentTriggerReservation,
): boolean {
  return (
    pending?.ownerId === reservation.ownerId &&
    pending.id === reservation.triggerId
  );
}

function isActiveAppIntentReservation(
  reservation: AppIntentReservation | undefined,
): reservation is AppIntentReservation {
  return (
    reservation !== undefined && activeAppIntentReservation === reservation
  );
}

function matchesPendingAppIntent(
  pending: PendingAppIntentHandoff | null,
  reservation: AppIntentReservation,
): boolean {
  return (
    pending?.ownerId === reservation.ownerId &&
    pending.id === reservation.handoffId
  );
}

function appIntentProfileLabel(session: UserSession | null): string {
  const username = session?.username.trim();
  return username ? username : 'Invité local';
}

function sameResolvedTrigger(
  left: NativeResolvedStoredTrigger | null | undefined,
  right: NativeResolvedStoredTrigger | null | undefined,
): boolean {
  if (!left || !right) {
    return !left && !right;
  }
  return (
    left.id === right.id &&
    left.title === right.title &&
    left.prompt === right.prompt &&
    left.repeats === right.repeats
  );
}

function appIntentAgentScope(
  prompt: string,
): Pick<AppIntentReservation, 'requestedIntent' | 'allowedToolIds'> {
  const route = routeAgentIntent(prompt);
  if (!route.requiresTool || route.clarification) {
    return {};
  }
  return {
    requestedIntent: route.intent,
    allowedToolIds: [...route.allowedToolIds],
  };
}

export function getLocalConversationOwner(session: UserSession | null): string {
  // Preserve the authenticated canonical username exactly. Lowercasing here
  // can collapse distinct case-sensitive server accounts onto one local
  // message/trigger/approval/Outlook credential scope.
  const username = session?.username.trim();
  return username ? `account:${username}` : 'guest';
}

function isServerMessage(message: Message): boolean {
  return (message.metadata?.inferenceBackend ?? 'server') === 'server';
}

function isLocalMessageForOwner(message: Message, ownerId: string): boolean {
  return (
    message.metadata?.inferenceBackend === 'on-device' &&
    message.metadata.localOwnerId === ownerId &&
    message.metadata.source !== 'app-intent'
  );
}

export function isConversationMessageVisible(
  message: Message,
  backend: InferenceBackend,
  ownerId: string,
): boolean {
  const metadata = message.metadata;
  if (metadata?.source === 'app-intent') {
    return metadata.localOwnerId === ownerId;
  }
  const messageBackend = metadata?.inferenceBackend ?? 'server';
  return backend === 'server'
    ? messageBackend === 'server'
    : messageBackend === 'on-device' && metadata?.localOwnerId === ownerId;
}

function agentHistoryForOwner(
  messages: Message[],
  ownerId: string,
  excludingMessageId?: string,
): NativeAgentHistoryMessage[] {
  return messages
    .filter(
      (message) =>
        message.id !== excludingMessageId &&
        isLocalMessageForOwner(message, ownerId) &&
        (message.role === 'user' || message.role === 'assistant') &&
        message.content.trim().length > 0,
    )
    .slice(-40)
    .map((message) => ({
      role: message.role as NativeAgentHistoryMessage['role'],
      content: message.content,
    }));
}

function localAgentMessage(ownerId: string, content: string): Message {
  return {
    id: buildMessageId('on-device-agent'),
    role: 'assistant',
    content,
    createdAt: new Date(),
    metadata: {
      mode: 'chat',
      source: 'agent',
      inferenceBackend: 'on-device',
      localOwnerId: ownerId,
      modelId: 'ales27pm/Dolphin3.0-CoreML',
      finishReason: 'agent',
    },
  };
}

function localAppIntentUserMessage(ownerId: string, content: string): Message {
  return {
    id: buildMessageId('app-intent-memory'),
    role: 'user',
    content,
    createdAt: new Date(),
    metadata: {
      mode: 'chat',
      source: 'app-intent',
      inferenceBackend: 'on-device',
      localOwnerId: ownerId,
    },
  };
}

function localAppIntentResultMessage(
  ownerId: string,
  content: string,
): Message {
  return {
    id: buildMessageId('app-intent-memory-result'),
    role: 'assistant',
    content,
    createdAt: new Date(),
    metadata: {
      mode: 'chat',
      source: 'app-intent',
      inferenceBackend: 'on-device',
      localOwnerId: ownerId,
      finishReason: 'local-tool',
    },
  };
}

function pendingApprovalFromResult(
  result: Extract<NativeAgentRunResult, { status: 'approval_required' }>,
  ownerId: string,
  prompt: string,
  history: NativeAgentHistoryMessage[],
): PendingAgentApproval {
  return {
    recordId: result.approval.recordId,
    ownerId,
    prompt,
    toolId: result.approval.toolId,
    arguments: result.approval.arguments,
    expiresAt: result.approval.expiresAt,
    displayName: result.approval.displayName,
    risk: result.approval.risk,
    history,
  };
}

function approvalBinding(
  pending: PendingAgentApproval,
): NativeAgentApprovalBinding {
  return {
    recordId: pending.recordId,
    ownerId: pending.ownerId,
    prompt: pending.prompt,
    toolId: pending.toolId,
    arguments: pending.arguments,
    expiresAt: pending.expiresAt,
  };
}

function approvalExpired(pending: PendingAgentApproval): boolean {
  const expiresAt = new Date(pending.expiresAt).getTime();
  return !Number.isFinite(expiresAt) || expiresAt <= Date.now();
}

function agentResultPatch(
  state: ChatState,
  result: NativeAgentRunResult,
  context: {
    ownerId: string;
    prompt: string;
    history: NativeAgentHistoryMessage[];
  },
): Partial<ChatState> {
  const base = {
    loading: false,
    activeAgentRunId: null,
    pendingAgentApproval: null,
    pendingAgentPermission: null,
  };
  switch (result.status) {
    case 'final':
      return {
        ...base,
        messages: [
          ...state.messages,
          localAgentMessage(context.ownerId, result.message),
        ],
        notice: {
          tone: 'success',
          message: 'Action locale terminée sur cet iPhone.',
        },
      };
    case 'clarification':
      return {
        ...base,
        messages: [
          ...state.messages,
          localAgentMessage(context.ownerId, result.message),
        ],
        notice: {
          tone: 'info',
          message: "L'agent local a besoin d'une précision.",
        },
      };
    case 'approval_required':
      return {
        ...base,
        pendingAgentApproval: pendingApprovalFromResult(
          result,
          context.ownerId,
          context.prompt,
          context.history,
        ),
        notice: {
          tone: 'warning',
          message: `Approbation requise avant ${result.approval.displayName}.`,
        },
      };
    case 'permission_required':
      return {
        ...base,
        pendingAgentPermission: {
          ownerId: context.ownerId,
          prompt: context.prompt,
          permission: result.permission,
          history: context.history,
        },
        notice: {
          tone: 'warning',
          message: `Autorisation ${result.permission} requise.`,
        },
      };
    case 'unavailable':
      return {
        ...base,
        messages: [
          ...state.messages,
          localAgentMessage(context.ownerId, result.message),
        ],
        notice: { tone: 'warning', message: result.message },
      };
    case 'cancelled':
      return {
        ...base,
        notice: {
          tone: 'warning',
          message:
            'Annulation demandée. Si un outil externe avait déjà commencé, son état final peut être inconnu; vérifiez avant de réessayer.',
        },
      };
    case 'failed':
      return {
        ...base,
        messages: [
          ...state.messages,
          localAgentMessage(context.ownerId, result.message),
        ],
        error: result.message,
        notice: { tone: 'danger', message: result.message },
      };
  }
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
      pendingAgentApproval: null,
      pendingAgentPermission: null,
      pendingAgentTrigger: null,
      pendingAppIntentHandoff: null,
      activeAgentRunId: null,
      refreshPendingAgentTrigger: async (alreadyHydrated = false) => {
        const triggerFetchVersion = ++agentTriggerFetchVersion;
        if (!alreadyHydrated) {
          await ensureInferenceStoreHydrated();
        }
        if (
          useInferenceStore.getState().backend !== 'on-device' ||
          !nativeAgentModuleAvailable ||
          activeAgentTriggerReservation !== null
        ) {
          return;
        }
        const ownerId = getLocalConversationOwner(get().session);
        try {
          const handoff = await getPendingNativeAgentTrigger(ownerId);
          if (
            triggerFetchVersion !== agentTriggerFetchVersion ||
            useInferenceStore.getState().backend !== 'on-device' ||
            getLocalConversationOwner(get().session) !== ownerId ||
            activeAgentTriggerReservation !== null
          ) {
            return;
          }
          set((state) => {
            if (!handoff) {
              return state.pendingAgentTrigger?.ownerId === ownerId
                ? { pendingAgentTrigger: null }
                : {};
            }
            const prompt = normalizedAgentTriggerPrompt(handoff.prompt);
            if (!prompt) {
              return {
                ...(state.pendingAgentTrigger?.ownerId === ownerId
                  ? { pendingAgentTrigger: null }
                  : {}),
                notice: {
                  tone: 'warning' as const,
                  message:
                    'Cette requête planifiée dépasse 512 octets UTF-8 et reste enregistrée sans être exécutée.',
                },
              };
            }
            if (
              state.pendingAgentTrigger?.ownerId === ownerId &&
              state.pendingAgentTrigger.id === handoff.id &&
              state.pendingAgentTrigger.prompt === prompt
            ) {
              return {};
            }
            return {
              pendingAgentTrigger: { ...handoff, prompt, ownerId },
              notice: {
                tone: 'info' as const,
                message: `Requête planifiée prête : ${handoff.title}`,
              },
            };
          });
        } catch (triggerError) {
          console.debug(
            '[chatStore] scheduled agent handoff unavailable',
            triggerError,
          );
        }
      },
      refreshPendingAppIntentHandoff: async () => {
        const fetchVersion = ++appIntentFetchVersion;
        if (
          !nativeAppIntentModuleAvailable ||
          activeAppIntentReservation !== null
        ) {
          return;
        }
        const ownerId = getLocalConversationOwner(get().session);
        const profileLabel = appIntentProfileLabel(get().session);
        try {
          const handoff = await getPendingNativeAppIntentHandoff(ownerId);
          let resolvedTrigger: NativeResolvedStoredTrigger | null = null;
          if (
            handoff?.profileMatches &&
            handoff.kind === 'runTrigger' &&
            handoff.input
          ) {
            try {
              resolvedTrigger = await resolveNativeStoredAgentTrigger(
                ownerId,
                handoff.input,
              );
            } catch (triggerError) {
              console.debug(
                '[chatStore] App Intent trigger preview unavailable',
                triggerError,
              );
            }
          }
          if (
            fetchVersion !== appIntentFetchVersion ||
            getLocalConversationOwner(get().session) !== ownerId ||
            activeAppIntentReservation !== null
          ) {
            return;
          }
          set((state) => {
            if (!handoff) {
              return state.pendingAppIntentHandoff?.ownerId === ownerId
                ? { pendingAppIntentHandoff: null }
                : {};
            }
            if (
              state.pendingAppIntentHandoff?.ownerId === ownerId &&
              state.pendingAppIntentHandoff.id === handoff.id &&
              state.pendingAppIntentHandoff.profileMatches ===
                handoff.profileMatches &&
              sameResolvedTrigger(
                state.pendingAppIntentHandoff.resolvedTrigger,
                resolvedTrigger,
              )
            ) {
              return {};
            }
            return {
              pendingAppIntentHandoff: {
                ...handoff,
                ownerId,
                profileLabel,
                resolvedTrigger,
              },
              notice: {
                tone: handoff.profileMatches
                  ? resolvedTrigger || handoff.kind !== 'runTrigger'
                    ? ('info' as const)
                    : ('warning' as const)
                  : ('warning' as const),
                message: !handoff.profileMatches
                  ? 'Une action liée à un autre profil est masquée; seule sa suppression exacte est disponible.'
                  : handoff.kind === 'runTrigger' && !resolvedTrigger
                    ? "Le déclencheur n'a pas pu être prévisualisé; aucune exécution n'est autorisée."
                    : 'Une action Siri ou Raccourcis attend votre confirmation dans monGARS.',
              },
            };
          });
        } catch (handoffError) {
          console.debug(
            '[chatStore] App Intent handoff unavailable',
            handoffError,
          );
        }
      },
      initialize: async () => {
        await ensureInferenceStoreHydrated();
        if (nativeAppIntentModuleAvailable) {
          try {
            await setActiveNativeAppIntentProfile(
              getLocalConversationOwner(get().session),
            );
          } catch (profileError) {
            console.debug(
              '[chatStore] App Intent profile binding unavailable',
              profileError,
            );
            set({
              notice: {
                tone: 'warning',
                message:
                  "Le profil App Intent n'a pas pu être activé; les actions restent bloquées.",
              },
            });
          }
        }
        await get().refreshPendingAppIntentHandoff();
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
          await get().refreshPendingAgentTrigger(true);
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
        assertStableSession();
        await ensureInferenceStoreHydrated();
        const inference = useInferenceStore.getState();
        if (
          backend !== inference.backend &&
          (get().loading ||
            activeAgentTriggerReservation !== null ||
            activeAppIntentReservation !== null ||
            inference.activeRequestId !== null ||
            inference.status.phase === 'generating')
        ) {
          throw new Error(
            'Attendez la fin de la requête active ou annulez-la avant de changer de backend.',
          );
        }
        if (backend !== inference.backend) {
          agentTriggerFetchVersion += 1;
        }
        const pendingApproval = get().pendingAgentApproval;
        if (backend !== inference.backend && pendingApproval) {
          await rejectNativeAgent(approvalBinding(pendingApproval)).catch(
            () => {
              // The native process may already have expired the record. Clearing
              // the JS binding still prevents any later execution attempt.
            },
          );
          set({ pendingAgentApproval: null });
        }
        if (backend !== inference.backend && get().pendingAgentPermission) {
          set({ pendingAgentPermission: null });
        }
        if (backend !== inference.backend && get().pendingAgentTrigger) {
          set({ pendingAgentTrigger: null });
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
        const inference = useInferenceStore.getState();
        if (
          sessionTransitionInProgress ||
          get().loading ||
          get().activeAgentRunId !== null ||
          activeAgentTriggerReservation !== null ||
          activeAppIntentReservation !== null ||
          inference.activeRequestId !== null ||
          inference.status.phase === 'generating'
        ) {
          throw new Error(
            'Attendez la fin de la requête locale ou annulez-la avant de changer de compte.',
          );
        }
        sessionTransitionInProgress = true;
        try {
          agentTriggerFetchVersion += 1;
          appIntentFetchVersion += 1;
          const pendingApproval = get().pendingAgentApproval;
          const nextOwnerId = getLocalConversationOwner(session);
          if (pendingApproval && pendingApproval.ownerId !== nextOwnerId) {
            await rejectNativeAgent(approvalBinding(pendingApproval)).catch(
              () => {
                // A missing/expired native record is already non-executable.
              },
            );
            set({ pendingAgentApproval: null });
          }
          if (get().pendingAgentPermission?.ownerId !== nextOwnerId) {
            set({ pendingAgentPermission: null });
          }
          if (get().pendingAgentTrigger?.ownerId !== nextOwnerId) {
            set({ pendingAgentTrigger: null });
          }
          // Never acknowledge a handoff captured for the previous profile.
          // Explicit profile binding below governs future App Intents; reads do
          // not mutate ownership and can safely return masked mismatch metadata.
          set({ pendingAppIntentHandoff: null });
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
            if (nativeAppIntentModuleAvailable) {
              try {
                await setActiveNativeAppIntentProfile(nextOwnerId);
              } catch (profileError) {
                console.debug(
                  '[chatStore] App Intent profile binding unavailable',
                  profileError,
                );
              }
              await get().refreshPendingAppIntentHandoff();
            }
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

          if (nativeAppIntentModuleAvailable) {
            try {
              await setActiveNativeAppIntentProfile(nextOwnerId);
            } catch (profileError) {
              console.debug(
                '[chatStore] App Intent profile binding unavailable',
                profileError,
              );
            }
          }

          await get().initialize();
        } finally {
          sessionTransitionInProgress = false;
        }
      },
      sendMessage: async (
        content,
        mode,
        triggerReservation,
        appIntentReservation,
      ) => {
        assertStableSession();
        const trimmed = content.trim();
        if (!trimmed) {
          if (isActiveTriggerReservation(triggerReservation)) {
            activeAgentTriggerReservation = null;
            set({ loading: false });
            throw new Error('La requête planifiée est vide.');
          }
          if (isActiveAppIntentReservation(appIntentReservation)) {
            activeAppIntentReservation = null;
            set({ loading: false });
            throw new Error('La requête App Intent est vide.');
          }
          return;
        }
        await ensureInferenceStoreHydrated();
        const reservedTrigger = isActiveTriggerReservation(triggerReservation);
        const reservedAppIntent =
          isActiveAppIntentReservation(appIntentReservation);
        if (get().loading && !reservedTrigger && !reservedAppIntent) {
          throw new Error('Une requête est déjà en cours.');
        }

        const inference = useInferenceStore.getState();
        const backend = inference.backend;
        const activeMode = mode ?? get().mode;
        const session = get().session;
        const localOwnerId = getLocalConversationOwner(session);
        if (
          triggerReservation &&
          (!reservedTrigger ||
            backend !== 'on-device' ||
            localOwnerId !== triggerReservation.ownerId ||
            !matchesPendingTrigger(
              get().pendingAgentTrigger,
              triggerReservation,
            ) ||
            get().pendingAgentTrigger?.prompt !== content)
        ) {
          if (isActiveTriggerReservation(triggerReservation)) {
            activeAgentTriggerReservation = null;
            set({ loading: false });
          }
          throw new Error(
            'La requête planifiée ne correspond plus au profil local actif.',
          );
        }
        if (
          appIntentReservation &&
          (!reservedAppIntent ||
            backend !== 'on-device' ||
            localOwnerId !== appIntentReservation.ownerId ||
            !matchesPendingAppIntent(
              get().pendingAppIntentHandoff,
              appIntentReservation,
            ) ||
            appIntentReservation.prompt !== content)
        ) {
          if (isActiveAppIntentReservation(appIntentReservation)) {
            activeAppIntentReservation = null;
            set({ loading: false });
          }
          throw new Error(
            'La requête App Intent ne correspond plus au profil local actif.',
          );
        }
        const pendingApproval = get().pendingAgentApproval;
        if (backend === 'on-device' && pendingApproval) {
          if (approvalExpired(pendingApproval)) {
            await rejectNativeAgent(approvalBinding(pendingApproval)).catch(
              () => {
                // Expiry in either layer is terminal and cannot authorize work.
              },
            );
            set(
              produce<ChatState>((draft) => {
                draft.pendingAgentApproval = null;
                draft.messages.push(
                  localAgentMessage(
                    pendingApproval.ownerId,
                    "Cette approbation locale a expiré; l'action n'a pas été exécutée.",
                  ),
                );
              }),
            );
          } else if (pendingApproval.ownerId !== localOwnerId) {
            await rejectNativeAgent(approvalBinding(pendingApproval)).catch(
              () => undefined,
            );
            set({ pendingAgentApproval: null });
          } else {
            throw new Error(
              "Approuvez ou rejetez l'action locale en attente avant d'envoyer un autre message.",
            );
          }
        }
        const pendingPermission = get().pendingAgentPermission;
        if (backend === 'on-device' && pendingPermission) {
          if (pendingPermission.ownerId !== localOwnerId) {
            set({ pendingAgentPermission: null });
          } else {
            throw new Error(
              "Accordez ou refusez l'autorisation locale en attente avant d'envoyer un autre message.",
            );
          }
        }
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

        if (
          triggerReservation &&
          (!isActiveTriggerReservation(triggerReservation) ||
            useInferenceStore.getState().backend !== 'on-device' ||
            getLocalConversationOwner(get().session) !==
              triggerReservation.ownerId ||
            !matchesPendingTrigger(
              get().pendingAgentTrigger,
              triggerReservation,
            ) ||
            get().pendingAgentTrigger?.prompt !== content)
        ) {
          if (isActiveTriggerReservation(triggerReservation)) {
            activeAgentTriggerReservation = null;
            set({ loading: false });
          }
          throw new Error(
            'La requête planifiée a changé avant son exécution; aucun agent lancé.',
          );
        }
        if (
          appIntentReservation &&
          (!isActiveAppIntentReservation(appIntentReservation) ||
            useInferenceStore.getState().backend !== 'on-device' ||
            getLocalConversationOwner(get().session) !==
              appIntentReservation.ownerId ||
            !matchesPendingAppIntent(
              get().pendingAppIntentHandoff,
              appIntentReservation,
            ) ||
            appIntentReservation.prompt !== content)
        ) {
          if (isActiveAppIntentReservation(appIntentReservation)) {
            activeAppIntentReservation = null;
            set({ loading: false });
          }
          throw new Error(
            "La requête App Intent a changé avant l'exécution; aucun agent lancé.",
          );
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

        if (triggerReservation) {
          activeAgentTriggerReservation = null;
        }
        if (appIntentReservation) {
          activeAppIntentReservation = null;
        }

        set(
          produce<ChatState>((draft) => {
            if (
              triggerReservation &&
              matchesPendingTrigger(
                draft.pendingAgentTrigger,
                triggerReservation,
              )
            ) {
              draft.pendingAgentTrigger = null;
            }
            if (
              appIntentReservation &&
              matchesPendingAppIntent(
                draft.pendingAppIntentHandoff,
                appIntentReservation,
              )
            ) {
              draft.pendingAppIntentHandoff = null;
            }
            draft.messages.push(userMessage);
            draft.loading = true;
            draft.error = null;
            draft.notice = {
              tone: 'info',
              message:
                activeMode === 'embed'
                  ? 'Generation d embedding…'
                  : backend === 'on-device'
                    ? 'Modèle local actif; les outils peuvent contacter leur fournisseur…'
                    : 'Generation de reponse…',
            };
          }),
        );

        try {
          if (backend === 'on-device') {
            if (
              appIntentReservation?.requestedIntent ||
              shouldUseNativeAgent(trimmed)
            ) {
              const history = agentHistoryForOwner(
                get().messages,
                localOwnerId,
                userMessage.id,
              );
              const runId = createNativeAgentRunId();
              set({
                activeAgentRunId: runId,
                notice: {
                  tone: 'info',
                  message: "L'agent local prépare une action structurée…",
                },
              });
              const result = await executeNativeAgent(
                {
                  ownerId: localOwnerId,
                  prompt: trimmed,
                  history,
                  ...(appIntentReservation?.requestedIntent &&
                  appIntentReservation.allowedToolIds
                    ? {
                        requestedIntent: appIntentReservation.requestedIntent,
                        allowedToolIds: appIntentReservation.allowedToolIds,
                      }
                    : {}),
                },
                runId,
              );
              set((state) =>
                agentResultPatch(state, result, {
                  ownerId: localOwnerId,
                  prompt: trimmed,
                  history,
                }),
              );
              return;
            }

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
            activeAgentRunId: null,
            error: message,
            notice: {
              tone: 'danger',
              message,
            },
          });
          throw error;
        }
      },
      approvePendingAgent: async () => {
        assertStableSession();
        await ensureInferenceStoreHydrated();
        const pending = get().pendingAgentApproval;
        if (!pending) {
          throw new Error("Aucune action locale n'attend une approbation.");
        }
        if (get().loading) {
          throw new Error('Une requête est déjà en cours.');
        }
        const ownerId = getLocalConversationOwner(get().session);
        if (
          useInferenceStore.getState().backend !== 'on-device' ||
          pending.ownerId !== ownerId
        ) {
          set({ pendingAgentApproval: null });
          throw new Error(
            "L'approbation ne correspond pas au compte local actif.",
          );
        }
        if (approvalExpired(pending)) {
          await rejectNativeAgent(approvalBinding(pending)).catch(
            () => undefined,
          );
          set(
            produce<ChatState>((draft) => {
              draft.pendingAgentApproval = null;
              draft.messages.push(
                localAgentMessage(
                  ownerId,
                  "Cette approbation locale a expiré; l'action n'a pas été exécutée.",
                ),
              );
              draft.notice = {
                tone: 'warning',
                message: "L'approbation a expiré.",
              };
            }),
          );
          return;
        }

        const runId = createNativeAgentRunId();
        set({
          loading: true,
          activeAgentRunId: runId,
          error: null,
          notice: {
            tone: 'info',
            message: 'Approbation locale vérifiée; exécution structurée…',
          },
        });
        let approved = false;
        try {
          const approval = await approveNativeAgent(approvalBinding(pending));
          if (approval.status !== 'approved') {
            set(
              produce<ChatState>((draft) => {
                draft.loading = false;
                draft.activeAgentRunId = null;
                draft.pendingAgentApproval = null;
                draft.messages.push(
                  localAgentMessage(
                    ownerId,
                    "Cette approbation n'est plus valide; l'action n'a pas été exécutée.",
                  ),
                );
              }),
            );
            return;
          }
          approved = true;
          const result = await executeNativeAgent(
            {
              ownerId,
              prompt: pending.prompt,
              history: pending.history,
              approvalRecordId: pending.recordId,
            },
            runId,
          );
          set((state) =>
            agentResultPatch(state, result, {
              ownerId,
              prompt: pending.prompt,
              history: pending.history,
            }),
          );
        } catch (error) {
          if (approved) {
            await rejectNativeAgent(approvalBinding(pending)).catch(
              () => undefined,
            );
          }
          const message =
            error instanceof Error
              ? error.message
              : "L'action locale approuvée a échoué.";
          set({
            loading: false,
            activeAgentRunId: null,
            pendingAgentApproval: null,
            error: message,
            notice: { tone: 'danger', message },
          });
          throw error;
        }
      },
      rejectPendingAgent: async () => {
        assertStableSession();
        const pending = get().pendingAgentApproval;
        if (!pending) {
          return;
        }
        if (get().loading) {
          throw new Error('Une requête est déjà en cours.');
        }
        set({ loading: true, error: null });
        try {
          await rejectNativeAgent(approvalBinding(pending));
          set(
            produce<ChatState>((draft) => {
              draft.loading = false;
              draft.pendingAgentApproval = null;
              draft.messages.push(
                localAgentMessage(
                  pending.ownerId,
                  "Action rejetée. Aucun outil n'a été exécuté.",
                ),
              );
              draft.notice = {
                tone: 'info',
                message: 'Action locale rejetée.',
              };
            }),
          );
        } catch (error) {
          const message =
            error instanceof Error
              ? error.message
              : 'Impossible de vérifier le rejet local.';
          set({
            loading: false,
            pendingAgentApproval: null,
            error: message,
            notice: { tone: 'danger', message },
          });
          throw error;
        }
      },
      requestPendingAgentPermission: async () => {
        assertStableSession();
        await ensureInferenceStoreHydrated();
        const pending = get().pendingAgentPermission;
        if (!pending) {
          throw new Error("Aucune autorisation locale n'est en attente.");
        }
        if (get().loading) {
          throw new Error('Une requête est déjà en cours.');
        }
        const ownerId = getLocalConversationOwner(get().session);
        if (
          useInferenceStore.getState().backend !== 'on-device' ||
          pending.ownerId !== ownerId
        ) {
          set({ pendingAgentPermission: null });
          throw new Error(
            "L'autorisation ne correspond pas au compte local actif.",
          );
        }

        set({
          loading: true,
          error: null,
          notice: {
            tone: 'info',
            message: `Demande d'autorisation ${pending.permission}…`,
          },
        });
        try {
          const permission = await requestNativeAgentPermission(
            pending.permission,
          );
          if (
            permission.state !== 'granted' &&
            permission.state !== 'limited'
          ) {
            const message =
              permission.state === 'denied'
                ? `Autorisation ${pending.permission} refusée; aucun outil n'a été exécuté.`
                : `Autorisation ${pending.permission} indisponible (${permission.state}); aucun outil n'a été exécuté.`;
            set((state) => ({
              loading: false,
              pendingAgentPermission: null,
              messages: [
                ...state.messages,
                localAgentMessage(ownerId, message),
              ],
              notice: { tone: 'warning', message },
            }));
            return;
          }

          const runId = createNativeAgentRunId();
          set({ activeAgentRunId: runId });
          const result = await executeNativeAgent(
            {
              ownerId,
              prompt: pending.prompt,
              history: pending.history,
            },
            runId,
          );
          set((state) =>
            agentResultPatch(state, result, {
              ownerId,
              prompt: pending.prompt,
              history: pending.history,
            }),
          );
        } catch (error) {
          const message =
            error instanceof Error
              ? error.message
              : "Impossible de demander l'autorisation locale.";
          set({
            loading: false,
            activeAgentRunId: null,
            pendingAgentPermission: null,
            error: message,
            notice: { tone: 'danger', message },
          });
          throw error;
        }
      },
      dismissPendingAgentPermission: () => {
        const pending = get().pendingAgentPermission;
        if (!pending || get().loading) {
          return;
        }
        set((state) => ({
          pendingAgentPermission: null,
          messages: [
            ...state.messages,
            localAgentMessage(
              pending.ownerId,
              "Autorisation non accordée. Aucun outil n'a été exécuté.",
            ),
          ],
          notice: {
            tone: 'info',
            message: "Demande d'autorisation annulée.",
          },
        }));
      },
      runPendingAgentTrigger: async () => {
        assertStableSession();
        const pending = get().pendingAgentTrigger;
        if (!pending) {
          return;
        }
        const prompt = normalizedAgentTriggerPrompt(pending.prompt);
        if (!prompt || prompt !== pending.prompt) {
          const message =
            'Cette requête planifiée dépasse le budget de 512 octets UTF-8; elle reste enregistrée et aucun agent ne sera lancé.';
          set({
            error: message,
            notice: { tone: 'warning', message },
          });
          throw new Error(message);
        }
        if (
          get().loading ||
          get().pendingAgentApproval ||
          get().pendingAgentPermission
        ) {
          throw new Error(
            "Terminez l'action locale en attente avant la requête planifiée.",
          );
        }
        if (
          useInferenceStore.getState().backend !== 'on-device' ||
          pending.ownerId !== getLocalConversationOwner(get().session)
        ) {
          set({ pendingAgentTrigger: null });
          throw new Error(
            'La requête planifiée ne correspond pas au profil local actif.',
          );
        }
        const reservation: AgentTriggerReservation = {
          ownerId: pending.ownerId,
          triggerId: pending.id,
        };
        activeAgentTriggerReservation = reservation;
        set({
          loading: true,
          error: null,
          notice: {
            tone: 'info',
            message: 'Confirmation de la requête planifiée…',
          },
        });
        try {
          const acknowledged = await acknowledgePendingNativeAgentTrigger(
            pending.ownerId,
            pending.id,
          );
          if (!isActiveTriggerReservation(reservation)) {
            return;
          }
          if (!acknowledged) {
            activeAgentTriggerReservation = null;
            set((state) => ({
              loading: false,
              pendingAgentTrigger: matchesPendingTrigger(
                state.pendingAgentTrigger,
                reservation,
              )
                ? null
                : state.pendingAgentTrigger,
              notice: {
                tone: 'warning',
                message:
                  'La requête planifiée a expiré ou appartient à un autre profil; aucun agent lancé.',
              },
            }));
            return;
          }

          const currentPending = get().pendingAgentTrigger;
          if (
            useInferenceStore.getState().backend !== 'on-device' ||
            getLocalConversationOwner(get().session) !== reservation.ownerId ||
            !matchesPendingTrigger(currentPending, reservation) ||
            currentPending?.prompt !== pending.prompt
          ) {
            activeAgentTriggerReservation = null;
            set((state) => ({
              loading: false,
              pendingAgentTrigger: matchesPendingTrigger(
                state.pendingAgentTrigger,
                reservation,
              )
                ? null
                : state.pendingAgentTrigger,
              notice: {
                tone: 'warning',
                message:
                  'Le profil local a changé; la requête planifiée confirmée ne sera pas exécutée.',
              },
            }));
            return;
          }

          await get().sendMessage(prompt, 'chat', reservation);
        } catch (error) {
          if (isActiveTriggerReservation(reservation)) {
            activeAgentTriggerReservation = null;
            const message =
              error instanceof Error
                ? error.message
                : 'Impossible de confirmer la requête planifiée.';
            set({
              loading: false,
              error: message,
              notice: { tone: 'danger', message },
            });
          }
          throw error;
        }
      },
      dismissPendingAgentTrigger: async () => {
        assertStableSession();
        const pending = get().pendingAgentTrigger;
        if (!pending || get().loading) {
          return;
        }
        const reservation: AgentTriggerReservation = {
          ownerId: pending.ownerId,
          triggerId: pending.id,
        };
        activeAgentTriggerReservation = reservation;
        set({ loading: true });
        const acknowledged = await acknowledgePendingNativeAgentTrigger(
          pending.ownerId,
          pending.id,
        ).catch(() => false);
        if (!isActiveTriggerReservation(reservation)) {
          return;
        }
        activeAgentTriggerReservation = null;
        set((state) => ({
          loading: false,
          pendingAgentTrigger: matchesPendingTrigger(
            state.pendingAgentTrigger,
            reservation,
          )
            ? null
            : state.pendingAgentTrigger,
          notice: {
            tone: acknowledged ? 'info' : 'warning',
            message: acknowledged
              ? 'Requête planifiée ignorée; aucun agent lancé.'
              : 'Requête planifiée non confirmée; aucun agent lancé.',
          },
        }));
      },
      runPendingAppIntentHandoff: async () => {
        assertStableSession();
        const pending = get().pendingAppIntentHandoff;
        if (!pending) {
          return null;
        }
        if (
          get().loading ||
          get().pendingAgentApproval ||
          get().pendingAgentPermission ||
          activeAgentTriggerReservation !== null ||
          activeAppIntentReservation !== null
        ) {
          throw new Error(
            "Terminez l'action locale en attente avant l'App Intent.",
          );
        }
        const ownerId = getLocalConversationOwner(get().session);
        if (pending.ownerId !== ownerId || !pending.profileMatches) {
          throw new Error(
            "L'App Intent appartient à un autre profil; son contenu et son exécution restent bloqués.",
          );
        }
        if (
          (pending.kind === 'ask' || pending.kind === 'runTrigger') &&
          useInferenceStore.getState().backend !== 'on-device'
        ) {
          const message =
            "Activez le modèle local avant d'exécuter cette action Siri ou Raccourcis.";
          set({
            error: message,
            notice: { tone: 'warning', message },
          });
          throw new Error(message);
        }

        const reservation: AppIntentReservation = {
          ownerId,
          handoffId: pending.id,
          prompt: '',
        };
        activeAppIntentReservation = reservation;
        appIntentFetchVersion += 1;
        set({
          loading: true,
          error: null,
          notice: {
            tone: 'info',
            message: "Confirmation de l'action Siri ou Raccourcis…",
          },
        });

        try {
          if (pending.kind === 'memorySearch' || pending.kind === 'memoryAdd') {
            if (!pending.input) {
              throw new Error("L'action mémoire protégée est vide.");
            }
            const memoryInput = pending.input;
            const current = get().pendingAppIntentHandoff;
            if (
              getLocalConversationOwner(get().session) !== ownerId ||
              !matchesPendingAppIntent(current, reservation) ||
              current?.kind !== pending.kind ||
              current?.input !== memoryInput ||
              current?.profileMatches !== true
            ) {
              throw new Error(
                "Le profil ou l'action mémoire a changé; rien n'a été exécuté.",
              );
            }
            const result = await executeNativeAppIntentMemoryAction({
              ownerId,
              id: pending.id,
              kind: pending.kind,
              input: memoryInput,
            });
            if (!isActiveAppIntentReservation(reservation)) {
              return null;
            }
            activeAppIntentReservation = null;
            set(
              produce<ChatState>((draft) => {
                if (
                  matchesPendingAppIntent(
                    draft.pendingAppIntentHandoff,
                    reservation,
                  )
                ) {
                  draft.pendingAppIntentHandoff = null;
                }
                draft.loading = false;
                draft.messages.push(
                  localAppIntentUserMessage(ownerId, memoryInput),
                  localAppIntentResultMessage(ownerId, result.message),
                );
                if (result.status === 'success') {
                  draft.error = null;
                  draft.notice = {
                    tone: 'success',
                    message:
                      'Action mémoire locale terminée sans modèle. Action à usage unique; aucune relance automatique.',
                  };
                } else {
                  draft.error = result.message;
                  draft.notice = {
                    tone: 'danger',
                    message: `${result.message} Action à usage unique; aucune relance automatique.`,
                  };
                }
              }),
            );
            return 'chat';
          }

          const previewedTrigger =
            pending.kind === 'runTrigger' ? pending.resolvedTrigger : null;
          if (pending.kind === 'runTrigger' && !previewedTrigger) {
            throw new Error(
              "Aucun déclencheur unique n'a été prévisualisé; aucun agent lancé.",
            );
          }
          if (previewedTrigger) {
            const currentTrigger = await resolveNativeStoredAgentTrigger(
              ownerId,
              previewedTrigger.id,
            );
            if (!sameResolvedTrigger(previewedTrigger, currentTrigger)) {
              throw new Error(
                'Le déclencheur a changé depuis la prévisualisation; aucun agent lancé.',
              );
            }
          }
          const prompt = appIntentHandoffPrompt(
            pending,
            previewedTrigger ?? undefined,
          );
          if (pending.kind !== 'diagnostics' && !prompt) {
            throw new Error("L'App Intent ne contient aucune requête valide.");
          }
          reservation.prompt = prompt ?? '';
          if (prompt) {
            const route = routeAgentIntent(prompt);
            if (route.requiresTool && route.clarification) {
              throw new Error(
                `${route.clarification} La requête doit être précisée avant confirmation.`,
              );
            }
            Object.assign(reservation, appIntentAgentScope(prompt));
          }

          const acknowledged = await acknowledgeNativeAppIntentHandoff(
            ownerId,
            pending.id,
          );
          if (!isActiveAppIntentReservation(reservation)) {
            return null;
          }
          if (!acknowledged) {
            activeAppIntentReservation = null;
            set((state) => ({
              loading: false,
              pendingAppIntentHandoff: matchesPendingAppIntent(
                state.pendingAppIntentHandoff,
                reservation,
              )
                ? null
                : state.pendingAppIntentHandoff,
              notice: {
                tone: 'warning',
                message:
                  "L'App Intent a expiré ou a été remplacé; aucune action lancée.",
              },
            }));
            return null;
          }

          const current = get().pendingAppIntentHandoff;
          if (
            getLocalConversationOwner(get().session) !== ownerId ||
            !matchesPendingAppIntent(current, reservation) ||
            current?.kind !== pending.kind ||
            current?.input !== pending.input ||
            current?.profileMatches !== true ||
            !(pending.kind === 'runTrigger'
              ? sameResolvedTrigger(
                  current.resolvedTrigger,
                  pending.resolvedTrigger,
                )
              : true)
          ) {
            activeAppIntentReservation = null;
            set((state) => ({
              loading: false,
              pendingAppIntentHandoff: matchesPendingAppIntent(
                state.pendingAppIntentHandoff,
                reservation,
              )
                ? null
                : state.pendingAppIntentHandoff,
              notice: {
                tone: 'warning',
                message:
                  "Le profil ou l'App Intent a changé; aucune action lancée.",
              },
            }));
            return null;
          }

          if (pending.kind === 'diagnostics') {
            activeAppIntentReservation = null;
            set((state) => ({
              loading: false,
              pendingAppIntentHandoff: matchesPendingAppIntent(
                state.pendingAppIntentHandoff,
                reservation,
              )
                ? null
                : state.pendingAppIntentHandoff,
              notice: {
                tone: 'info',
                message:
                  'Ouverture des diagnostics passifs; aucune capture démarrée.',
              },
            }));
            return 'diagnostics';
          }

          await get().sendMessage(prompt!, 'chat', undefined, reservation);
          return 'chat';
        } catch (handoffError) {
          if (isActiveAppIntentReservation(reservation)) {
            activeAppIntentReservation = null;
            const message =
              handoffError instanceof Error
                ? handoffError.message
                : "Impossible de confirmer l'App Intent.";
            set({
              loading: false,
              error: message,
              notice: { tone: 'danger', message },
            });
            await get().refreshPendingAppIntentHandoff();
          }
          throw handoffError;
        }
      },
      dismissPendingAppIntentHandoff: async () => {
        assertStableSession();
        const pending = get().pendingAppIntentHandoff;
        if (!pending || get().loading) {
          return;
        }
        if (pending.ownerId !== getLocalConversationOwner(get().session)) {
          set({
            notice: {
              tone: 'warning',
              message:
                'Cette action ne correspond pas au profil actif et ne peut pas être modifiée ici.',
            },
          });
          return;
        }
        const reservation: AppIntentReservation = {
          ownerId: pending.ownerId,
          handoffId: pending.id,
          prompt: '',
        };
        activeAppIntentReservation = reservation;
        appIntentFetchVersion += 1;
        set({ loading: true });
        const acknowledged = await (
          pending.profileMatches
            ? acknowledgeNativeAppIntentHandoff(pending.ownerId, pending.id)
            : discardNativeAppIntentHandoff(pending.id)
        ).catch(() => false);
        if (!isActiveAppIntentReservation(reservation)) {
          return;
        }
        activeAppIntentReservation = null;
        set((state) => ({
          loading: false,
          pendingAppIntentHandoff: matchesPendingAppIntent(
            state.pendingAppIntentHandoff,
            reservation,
          )
            ? null
            : state.pendingAppIntentHandoff,
          notice: {
            tone: acknowledged ? 'info' : 'warning',
            message: acknowledged
              ? "Action Siri ou Raccourcis ignorée; rien n'a été exécuté."
              : "Action Siri ou Raccourcis non confirmée; rien n'a été exécuté.",
          },
        }));
      },
      cancelGeneration: async () => {
        const activeAgentRunId = get().activeAgentRunId;
        if (activeAgentRunId) {
          const cancelled = await cancelNativeAgent(activeAgentRunId);
          if (cancelled) {
            set({
              notice: {
                tone: 'warning',
                message:
                  "Annulation de l'agent local demandée; vérifiez tout effet externe avant de réessayer.",
              },
            });
          }
          return;
        }
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
      version: 6,
      migrate: (persistedState) => {
        if (!persistedState) {
          return persistedState as ChatState;
        }

        const state = persistedState as Partial<ChatState> & {
          messages?: Array<Partial<Message> & { createdAt?: string | Date }>;
          session?: { username?: string; token?: string };
        };

        const legacyUsername =
          typeof state.session?.username === 'string'
            ? state.session.username.trim()
            : '';
        const legacyLocalOwnerId = legacyUsername
          ? getLocalConversationOwner({ username: legacyUsername, token: '' })
          : 'guest';

        return {
          // Authentication is process-memory-only. Explicitly overwrite any
          // session restored from versions that persisted bearer tokens.
          session: null,
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
