export type ChatMode = 'chat' | 'embed';

export type QuickAction = 'code' | 'summarize' | 'explain';

export type SpeechSegment = {
  text: string;
  estimatedDuration: number;
  pauseAfter: number;
};

export type SpeechTurn = {
  turnId: string;
  text: string;
  createdAt: string;
  segments: SpeechSegment[];
  averageWordsPerSecond: number;
  tempo: number;
};

export type MessageMetadata = {
  mode?: ChatMode;
  source?: 'history' | 'chat' | 'embedding' | 'realtime' | 'system';
  confidence?: number;
  processingTime?: number;
  speechTurn?: SpeechTurn | null;
  embedding?: {
    backend?: string | null;
    model?: string | null;
    dims?: number;
    count?: number;
    normalised?: boolean;
  };
};

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: Date;
  metadata?: MessageMetadata;
};

export type AuthCredentials = {
  username: string;
  password: string;
};

export type AuthToken = {
  accessToken: string;
  tokenType: string;
  username: string;
};

export type UserSession = {
  username: string;
  token: string;
};

export type ConversationHistoryRecord = {
  query: string;
  response: string;
  timestamp: string;
};

export type ChatReply = {
  response: string;
  confidence: number;
  processingTime: number;
  speechTurn: SpeechTurn;
};

export type SuggestionResponse = {
  actions: QuickAction[];
};

export type EmbeddingResponse = {
  vectors: number[][];
  dims?: number;
  count?: number;
  normalised?: boolean;
  backend?: string | null;
  model?: string | null;
};

export type ConnectionStatus =
  | 'offline'
  | 'connecting'
  | 'online'
  | 'error'
  | 'auth-required';

export type ConnectionSnapshot = {
  status: ConnectionStatus;
  detail: string | null;
  connectedAt: Date | null;
  lastMessageAt: Date | null;
  latencyMs: number | null;
  reconnectAttempt: number;
};
