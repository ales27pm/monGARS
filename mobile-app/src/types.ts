export type ChatMode = 'chat' | 'embed';

export type InferenceBackend = 'server' | 'on-device';

export type OnDeviceInferencePhase =
  | 'unavailable'
  | 'not-downloaded'
  | 'downloading'
  | 'verifying'
  | 'compiling'
  | 'loading'
  | 'ready'
  | 'generating'
  | 'error';

export type OnDeviceModelStatus = {
  phase: OnDeviceInferencePhase;
  modelId: string | null;
  displayName: string | null;
  revision: string | null;
  installedBytes: number;
  contextLength: number;
  minimumIOSVersion: number;
  detail: string | null;
};

export type OnDeviceModelProgress = {
  phase: OnDeviceInferencePhase;
  fractionCompleted: number;
  bytesPerSecond: number | null;
  detail: string | null;
};

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
  source?:
    | 'history'
    | 'chat'
    | 'embedding'
    | 'realtime'
    | 'system'
    | 'on-device';
  inferenceBackend?: InferenceBackend;
  modelId?: string | null;
  promptTokens?: number;
  generatedTokens?: number;
  tokensPerSecond?: number;
  finishReason?: string | null;
  localOwnerId?: string;
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
