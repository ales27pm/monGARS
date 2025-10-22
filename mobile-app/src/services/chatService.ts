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
  const ws = new WebSocket(`${settings.websocketUrl}?token=${token}`);

  return new Promise<void>((resolve, reject) => {
    let currentId: string | null = null;
    let buffer = '';

    const closeWith = (error?: Error) => {
      ws.close();
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };

    ws.onerror = (event) => {
      closeWith(new Error(`WebSocket error: ${JSON.stringify(event)}`));
    };

    ws.onopen = () => {
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
      if (closeEvent.wasClean === false) {
        reject(new Error(`Connection closed: ${closeEvent.code}`));
      }
    };
  });
}
