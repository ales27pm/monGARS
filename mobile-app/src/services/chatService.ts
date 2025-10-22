import axios from 'axios';
import { settings } from './config';
import type { Message } from '../types';

type StreamHandler = {
  token: string;
  content: string;
  onToken: (chunk: Message) => void;
};

type WebSocketCloseEventLike = {
  code: number;
  wasClean?: boolean;
};

export async function fetchChatHistory(token: string) {
  const response = await axios.get(`${settings.apiUrl}/chat/history`, {
    headers: { Authorization: `Bearer ${token}` },
    timeout: 10000,
  });
  return response.data;
}

export async function sendChatMessage(
  token: string,
  content: string,
  mode: 'chat' | 'embedding',
) {
  const response = await axios.post(
    `${settings.apiUrl}/chat/${mode}`,
    { content },
    {
      headers: { Authorization: `Bearer ${token}` },
      timeout: 10000,
    },
  );
  return response.data;
}

export async function streamChatReply({
  token,
  content,
  onToken,
}: StreamHandler) {
  const url = new URL(settings.websocketUrl);
  url.searchParams.set('token', encodeURIComponent(token));
  const ws = new WebSocket(url.toString());

  return new Promise<void>((resolve, reject) => {
    let currentId: string | null = null;
    let buffer = '';
    let settled = false;
    const connectTimeoutMs = 20000;
    const idleTimeoutMs = 60000;
    let connectTimer: ReturnType<typeof setTimeout> | null = null;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;

    const toDate = (value: unknown) => {
      if (value instanceof Date && !Number.isNaN(value.getTime())) {
        return value;
      }
      if (typeof value === 'string' || typeof value === 'number') {
        const parsed = new Date(value);
        if (!Number.isNaN(parsed.getTime())) {
          return parsed;
        }
      }
      return new Date();
    };

    const clearTimers = () => {
      if (connectTimer) {
        clearTimeout(connectTimer);
        connectTimer = null;
      }
      if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
    };

    const closeWith = (error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimers();
      try {
        ws.close();
      } catch (closeError) {
        console.warn('[chatService] ws close error', closeError);
      }
      error ? reject(error) : resolve();
    };

    const bumpIdleTimer = () => {
      if (idleTimer) {
        clearTimeout(idleTimer);
      }
      idleTimer = setTimeout(() => {
        closeWith(new Error('WebSocket idle timeout'));
      }, idleTimeoutMs);
    };

    connectTimer = setTimeout(() => {
      closeWith(new Error('WebSocket connect timeout'));
    }, connectTimeoutMs);

    ws.onerror = (event) => {
      const details =
        (event as any)?.message ??
        (event as any)?.reason ??
        JSON.stringify(event);
      closeWith(new Error(`WebSocket error: ${details}`));
    };

    ws.onopen = () => {
      if (connectTimer) {
        clearTimeout(connectTimer);
        connectTimer = null;
      }
      bumpIdleTimer();
      ws.send(
        JSON.stringify({
          type: 'prompt',
          payload: {
            content,
          },
        }),
      );
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        bumpIdleTimer();
        if (message.type === 'ticket') {
          ws.send(
            JSON.stringify({
              type: 'ack',
              payload: { ticket: message.payload.ticket },
            }),
          );
          return;
        }
        if (message.type === 'chunk') {
          const payload = message.payload ?? {};
          const chunkId = payload.id ?? currentId ?? `${Date.now()}`;
          const chunkContent =
            typeof payload.content === 'string' ? payload.content : '';
          currentId = chunkId;
          buffer += chunkContent;
          onToken({
            id: chunkId,
            role: 'assistant',
            content: buffer,
            createdAt: toDate(payload.timestamp),
          });
        }
        if (message.type === 'final') {
          const payload = message.payload ?? {};
          const finalId = payload.id ?? currentId ?? `${Date.now()}`;
          const finalContent =
            typeof payload.content === 'string' && payload.content.length > 0
              ? payload.content
              : buffer;
          onToken({
            id: finalId,
            role: 'assistant',
            content: finalContent,
            createdAt: toDate(payload.timestamp),
          });
          closeWith();
        }
        if (message.type === 'error') {
          closeWith(new Error(message.payload.detail));
        }
      } catch (error) {
        closeWith(error as Error);
      }
    };

    ws.onclose = (event) => {
      const closeEvent = event as WebSocketCloseEventLike;
      if (!settled) {
        const err =
          closeEvent?.wasClean === false
            ? new Error(`Connection closed: ${closeEvent.code}`)
            : undefined;
        closeWith(err);
      }
    };
  });
}
