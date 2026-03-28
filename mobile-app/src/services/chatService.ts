import axios from 'axios';
import { z } from 'zod';
import { settings } from './config';
import type {
  ChatReply,
  ConversationHistoryRecord,
  EmbeddingResponse,
  QuickAction,
  SuggestionResponse,
  UserSession,
} from '../types';

const speechSegmentSchema = z.object({
  text: z.string(),
  estimated_duration: z.number(),
  pause_after: z.number(),
});

const speechTurnSchema = z.object({
  turn_id: z.string(),
  text: z.string(),
  created_at: z.string(),
  segments: z.array(speechSegmentSchema),
  average_words_per_second: z.number(),
  tempo: z.number(),
});

const chatReplySchema = z.object({
  response: z.string(),
  confidence: z.number(),
  processing_time: z.number(),
  speech_turn: speechTurnSchema,
});

const historyRecordSchema = z.object({
  query: z.string(),
  response: z.string(),
  timestamp: z.string(),
});

const suggestionSchema = z.object({
  actions: z.array(z.enum(['code', 'summarize', 'explain'])),
});

const embeddingSchema = z.object({
  vectors: z.array(z.array(z.number())),
  dims: z.number().optional(),
  count: z.number().optional(),
  normalised: z.boolean().optional(),
  backend: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
});

const ticketSchema = z.object({
  ticket: z.string(),
  ttl: z.number(),
});

function authHeaders(token: string) {
  return {
    Authorization: `Bearer ${token}`,
  };
}

export async function fetchConversationHistory(
  session: UserSession,
): Promise<ConversationHistoryRecord[]> {
  const response = await axios.get(
    `${settings.apiBaseUrl}/conversation/history`,
    {
      params: {
        user_id: session.username,
        limit: settings.historyLimit,
      },
      headers: authHeaders(session.token),
      timeout: 10000,
    },
  );

  return z.array(historyRecordSchema).parse(response.data);
}

export async function postConversationMessage(
  session: UserSession,
  message: string,
): Promise<ChatReply> {
  const response = await axios.post(
    `${settings.apiBaseUrl}/conversation/chat`,
    { message },
    {
      headers: {
        ...authHeaders(session.token),
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    },
  );

  const parsed = chatReplySchema.parse(response.data);
  return {
    response: parsed.response,
    confidence: parsed.confidence,
    processingTime: parsed.processing_time,
    speechTurn: {
      turnId: parsed.speech_turn.turn_id,
      text: parsed.speech_turn.text,
      createdAt: parsed.speech_turn.created_at,
      segments: parsed.speech_turn.segments.map((segment) => ({
        text: segment.text,
        estimatedDuration: segment.estimated_duration,
        pauseAfter: segment.pause_after,
      })),
      averageWordsPerSecond: parsed.speech_turn.average_words_per_second,
      tempo: parsed.speech_turn.tempo,
    },
  };
}

export async function fetchQuickActions(
  session: UserSession,
  prompt: string,
): Promise<SuggestionResponse> {
  const response = await axios.post(
    `${settings.apiBaseUrl}/ui/suggestions`,
    {
      prompt,
      actions: ['code', 'summarize', 'explain'],
    },
    {
      headers: {
        ...authHeaders(session.token),
        'Content-Type': 'application/json',
      },
      timeout: 10000,
    },
  );

  const parsed = suggestionSchema.parse(response.data);
  return {
    actions: parsed.actions as QuickAction[],
  };
}

export async function requestEmbedding(
  text: string,
): Promise<EmbeddingResponse> {
  if (!settings.embedServiceUrl) {
    throw new Error("Service d'embedding indisponible.");
  }

  const response = await axios.post(
    settings.embedServiceUrl,
    {
      inputs: [text],
      normalise: false,
    },
    {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    },
  );

  return embeddingSchema.parse(response.data);
}

export async function fetchRealtimeTicket(
  session: UserSession,
): Promise<string> {
  const response = await axios.post(
    `${settings.apiBaseUrl}/auth/ws/ticket`,
    undefined,
    {
      headers: authHeaders(session.token),
      timeout: 10000,
    },
  );

  return ticketSchema.parse(response.data).ticket;
}
