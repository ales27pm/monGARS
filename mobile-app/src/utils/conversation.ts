import type {
  ConversationHistoryRecord,
  EmbeddingResponse,
  Message,
  QuickAction,
} from '../types';

const DEFAULT_QUICK_ACTIONS: QuickAction[] = ['code', 'summarize', 'explain'];

function toDate(value: string | Date): Date {
  const parsed = value instanceof Date ? value : new Date(value);
  return Number.isNaN(parsed.getTime()) ? new Date() : parsed;
}

export function buildMessageId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

export function mapHistoryToMessages(
  history: ConversationHistoryRecord[],
): Message[] {
  return [...history].reverse().flatMap((item, index) => {
    const createdAt = toDate(item.timestamp);
    const baseId = `${createdAt.getTime()}-${index}`;
    const userMessage: Message = {
      id: `history-user-${baseId}`,
      role: 'user',
      content: item.query,
      createdAt,
      metadata: {
        mode: 'chat',
        source: 'history',
      },
    };

    const assistantMessage: Message = {
      id: `history-assistant-${baseId}`,
      role: 'assistant',
      content: item.response,
      createdAt,
      metadata: {
        mode: 'chat',
        source: 'history',
      },
    };

    return item.response ? [userMessage, assistantMessage] : [userMessage];
  });
}

export function formatEmbeddingResult(result: EmbeddingResponse): string {
  const vectors = Array.isArray(result.vectors) ? result.vectors : [];
  const dims =
    typeof result.dims === 'number'
      ? result.dims
      : Array.isArray(vectors[0])
        ? vectors[0].length
        : 0;
  const count =
    typeof result.count === 'number' ? result.count : vectors.length;

  const lines = [
    'Resultat d embedding',
    `Vecteurs: ${count}`,
    `Dimensions: ${dims}`,
    `Normalise: ${result.normalised ? 'oui' : 'non'}`,
  ];

  if (result.backend) {
    lines.push(`Backend: ${result.backend}`);
  }
  if (result.model) {
    lines.push(`Modele: ${result.model}`);
  }
  if (vectors.length > 0) {
    lines.push(`Apercu: ${vectors[0].slice(0, 8).join(', ')}`);
  }

  return lines.join('\n');
}

export function buildRealtimeFingerprint(record: {
  query: string;
  response: string;
}): string {
  return `${record.query.trim()}::${record.response.trim()}`;
}

export function buildMarkdownExport(messages: Message[]): string {
  const lines = ['# Historique monGARS', ''];
  messages.forEach((message) => {
    lines.push(`## ${message.role.toUpperCase()}`);
    lines.push(`*Horodatage:* ${message.createdAt.toISOString()}`);
    if (message.metadata?.mode) {
      lines.push(`*Mode:* ${message.metadata.mode}`);
    }
    lines.push('');
    lines.push(message.content);
    lines.push('');
  });
  return lines.join('\n');
}

export function buildJsonExport(messages: Message[]): string {
  return JSON.stringify(
    {
      exported_at: new Date().toISOString(),
      count: messages.length,
      items: messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        created_at: message.createdAt.toISOString(),
        metadata: message.metadata ?? null,
      })),
    },
    null,
    2,
  );
}

export function orderQuickActions(actions?: string[]): QuickAction[] {
  if (!actions?.length) {
    return [...DEFAULT_QUICK_ACTIONS];
  }

  const ordered = actions.filter(
    (action, index): action is QuickAction =>
      DEFAULT_QUICK_ACTIONS.includes(action as QuickAction) &&
      actions.indexOf(action) === index,
  );

  DEFAULT_QUICK_ACTIONS.forEach((action) => {
    if (!ordered.includes(action)) {
      ordered.push(action);
    }
  });

  return ordered;
}
