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
    const timeoutMs = 20000;
    const timer = setTimeout(() => {
      closeWith(new Error('WebSocket timeout'));
    }, timeoutMs);

    const closeWith = (error?: Error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      try {
        ws.close();
      } catch (closeError) {
        console.warn('[chatService] ws close error', closeError);
      }
      error ? reject(error) : resolve();
    };

    ws.onerror = (event) => {
      const details =
        (event as any)?.message ??
        (event as any)?.reason ??
        JSON.stringify(event);
      closeWith(new Error(`WebSocket error: ${details}`));
    };

    ws.onopen = () => {
      clearTimeout(timer);
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
          currentId = message.payload.id;
          buffer += message.payload.content;
          onToken({
            id: currentId ?? `${Date.now()}`,
            role: 'assistant',
            content: buffer,
            createdAt: new Date(message.payload.timestamp),
          });
        }
        if (message.type === 'final') {
          onToken({
            id: message.payload.id ?? currentId ?? `${Date.now()}`,
            role: 'assistant',
            content: message.payload.content ?? buffer,
            createdAt: new Date(message.payload.timestamp),
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
