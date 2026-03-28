import { fetchRealtimeTicket } from './chatService';
import { settings } from './config';
import type { ConversationHistoryRecord, UserSession } from '../types';

type RealtimeEvent =
  | {
      id: string;
      type: 'ws.connected';
      ts: number;
      user: string | null;
      data: Record<string, unknown>;
    }
  | {
      id: string;
      type: 'history.snapshot';
      ts: number;
      user: string | null;
      data: {
        items?: ConversationHistoryRecord[];
      };
    }
  | {
      id: string;
      type: 'chat.message';
      ts: number;
      user: string | null;
      data: ConversationHistoryRecord;
    }
  | {
      id: string;
      type: 'ping' | 'ack';
      ts?: number;
      user?: string | null;
      data?: Record<string, unknown>;
      payload?: Record<string, unknown> | null;
    };

type RealtimeStatus =
  | 'offline'
  | 'connecting'
  | 'online'
  | 'error'
  | 'auth-required';

type RealtimeCallbacks = {
  onStatus: (status: {
    status: RealtimeStatus;
    detail?: string | null;
    connectedAt?: Date | null;
    lastMessageAt?: Date | null;
    latencyMs?: number | null;
    reconnectAttempt?: number;
  }) => void;
  onHistory: (items: ConversationHistoryRecord[]) => void;
  onMessage: (item: ConversationHistoryRecord) => void;
  onError: (message: string) => void;
};

const BACKOFF_BASE_MS = 500;
const BACKOFF_MAX_MS = 15000;
const JITTER_RATIO = 0.25;
const CLIENT_PING_INTERVAL_MS = 15000;

function computeBackoffMs(attempt: number): number {
  const base = Math.min(
    BACKOFF_MAX_MS,
    BACKOFF_BASE_MS * 2 ** Math.max(0, attempt - 1),
  );
  const jitter = base * JITTER_RATIO;
  return Math.floor(base - jitter + Math.random() * jitter * 2);
}

function toSocketUrl(ticket: string): string {
  const url = new URL(settings.websocketUrl);
  url.searchParams.set('t', ticket);
  return url.toString();
}

export function createRealtimeClient(callbacks: RealtimeCallbacks) {
  let ws: WebSocket | null = null;
  let session: UserSession | null = null;
  let connectSeq = 0;
  let disposed = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let pingTimer: ReturnType<typeof setInterval> | null = null;
  let reconnectAttempt = 0;
  let pendingPingId: string | null = null;
  let pendingPingAt: number | null = null;

  const clearReconnect = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  const clearPingTimer = () => {
    if (pingTimer) {
      clearInterval(pingTimer);
      pingTimer = null;
    }
    pendingPingId = null;
    pendingPingAt = null;
  };

  const closeSocket = (reason?: string) => {
    if (!ws) {
      return;
    }
    try {
      ws.close(1000, reason);
    } catch (error) {
      console.warn('[realtime] close failed', error);
    }
    ws = null;
  };

  const scheduleReconnect = (detail: string) => {
    if (disposed || !session || reconnectTimer) {
      return;
    }

    reconnectAttempt += 1;
    const delay = computeBackoffMs(reconnectAttempt);
    callbacks.onStatus({
      status: 'error',
      detail,
      reconnectAttempt,
    });

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      void connect();
    }, delay);
  };

  const startClientPing = () => {
    clearPingTimer();
    pingTimer = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
      }

      pendingPingId = `client-ping-${Date.now()}`;
      pendingPingAt = Date.now();
      ws.send(
        JSON.stringify({
          id: pendingPingId,
          type: 'client.ping',
          t: pendingPingAt,
        }),
      );
    }, CLIENT_PING_INTERVAL_MS);
  };

  const handleAck = (event: RealtimeEvent) => {
    const payload = 'payload' in event ? event.payload ?? event.data ?? {} : {};
    if (
      typeof payload?.detail === 'string' &&
      payload.detail === 'client.ping' &&
      pendingPingId &&
      event.id === pendingPingId &&
      pendingPingAt
    ) {
      callbacks.onStatus({
        status: 'online',
        latencyMs: Date.now() - pendingPingAt,
      });
      pendingPingId = null;
      pendingPingAt = null;
    }
  };

  async function connect(): Promise<void> {
    if (disposed || !session) {
      return;
    }
    if (
      ws &&
      (ws.readyState === WebSocket.CONNECTING ||
        ws.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    const seq = ++connectSeq;
    callbacks.onStatus({
      status: 'connecting',
      detail: 'Obtention du ticket WebSocket…',
      reconnectAttempt,
    });

    try {
      const ticket = await fetchRealtimeTicket(session);
      if (disposed || !session || seq !== connectSeq) {
        return;
      }

      ws = new WebSocket(toSocketUrl(ticket));
      callbacks.onStatus({
        status: 'connecting',
        detail: 'Connexion temps réel…',
        reconnectAttempt,
      });

      ws.onopen = () => {
        reconnectAttempt = 0;
        startClientPing();
      };

      ws.onmessage = (messageEvent) => {
        try {
          const payload = JSON.parse(messageEvent.data) as RealtimeEvent;

          if (payload.type === 'ping') {
            ws?.send(JSON.stringify({ id: payload.id, type: 'pong' }));
            return;
          }

          if (payload.type === 'ack') {
            handleAck(payload);
            return;
          }

          if (payload.type === 'ws.connected') {
            callbacks.onStatus({
              status: 'online',
              detail: 'Connecté en temps réel',
              connectedAt: new Date((payload.ts || Date.now() / 1000) * 1000),
              reconnectAttempt: 0,
            });
            return;
          }

          if (payload.type === 'history.snapshot') {
            callbacks.onHistory(
              Array.isArray(payload.data.items) ? payload.data.items : [],
            );
            return;
          }

          if (payload.type === 'chat.message') {
            callbacks.onStatus({
              status: 'online',
              lastMessageAt: new Date(),
            });
            callbacks.onMessage(payload.data);
          }
        } catch (error) {
          callbacks.onError('Flux temps réel invalide.');
          console.warn('[realtime] invalid message', error);
        }
      };

      ws.onerror = () => {
        callbacks.onStatus({
          status: 'error',
          detail: 'Erreur WebSocket',
          reconnectAttempt,
        });
      };

      ws.onclose = (event) => {
        clearPingTimer();
        ws = null;

        if (disposed) {
          callbacks.onStatus({
            status: 'offline',
            detail: 'Temps réel désactivé',
            reconnectAttempt: 0,
          });
          return;
        }

        if (event.code === 4401 || event.code === 4403) {
          callbacks.onStatus({
            status: 'auth-required',
            detail: 'Jeton invalide ou expiré',
            reconnectAttempt: 0,
          });
          return;
        }

        scheduleReconnect(`Déconnecté (${event.code || 'réseau'})`);
      };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Connexion temps réel impossible.';
      if (/401|403|auth/i.test(message)) {
        callbacks.onStatus({
          status: 'auth-required',
          detail: 'Authentification temps réel requise',
          reconnectAttempt: 0,
        });
        return;
      }
      callbacks.onError(message);
      scheduleReconnect(message);
    }
  }

  return {
    async open(nextSession: UserSession) {
      disposed = false;
      session = nextSession;
      clearReconnect();
      await connect();
    },
    close(reason = 'client-close') {
      disposed = true;
      clearReconnect();
      clearPingTimer();
      closeSocket(reason);
      callbacks.onStatus({
        status: 'offline',
        detail: 'Temps réel déconnecté',
        reconnectAttempt: 0,
      });
    },
    reconnect() {
      if (!session) {
        return;
      }
      clearReconnect();
      clearPingTimer();
      closeSocket('reconnect');
      void connect();
    },
  };
}
