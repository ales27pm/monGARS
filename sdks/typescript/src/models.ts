export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  allowed_actions?: string[] | null;
  approval_token?: string | null;
  token_ref?: string | null;
}

export interface SpeechSegment {
  text: string;
  estimated_duration: number;
  pause_after: number;
}

export interface SpeechTurn {
  turn_id: string;
  text: string;
  created_at: string;
  segments: SpeechSegment[];
  average_words_per_second: number;
  tempo: number;
}

export interface ChatResponse {
  response: string;
  confidence: number;
  processing_time: number;
  speech_turn: SpeechTurn;
}

export interface MemoryItem {
  user_id: string;
  query: string;
  response: string;
  timestamp: string;
}

export interface RagContextRequest {
  query: string;
  repositories?: string[] | null;
  max_results?: number | null;
}

export interface RagReference {
  repository: string;
  file_path: string;
  summary: string;
  score?: number | null;
  url?: string | null;
}

export interface RagContextResponse {
  enabled: boolean;
  focus_areas: string[];
  references: RagReference[];
}

export interface PeerRegistration {
  url: string;
}

export interface PeerLoadSnapshot {
  scheduler_id?: string | null;
  queue_depth: number;
  active_workers: number;
  concurrency: number;
  load_factor: number;
}

export interface PeerTelemetryPayload extends PeerLoadSnapshot {
  worker_uptime_seconds: number;
  tasks_processed: number;
  tasks_failed: number;
  task_failure_rate: number;
  observed_at?: string | null;
  source?: string | null;
}

export interface PeerTelemetryEnvelope {
  telemetry: PeerTelemetryPayload[];
}

export interface SuggestRequest {
  prompt: string;
  actions?: string[] | null;
}

export interface SuggestResponse {
  actions: string[];
  scores: Record<string, number>;
  model: string;
}

export interface ProvisionRequest {
  roles?: string[] | null;
  force?: boolean;
}

export interface ProvisionStatus {
  role: string;
  name: string;
  provider: string;
  action: string;
  detail?: string | null;
}

export interface ProvisionReport {
  statuses: ProvisionStatus[];
}

export interface ModelDefinition {
  role: string;
  name: string;
  provider: string;
  parameters: Record<string, unknown>;
  auto_download: boolean;
  description?: string | null;
}

export interface ModelProfile {
  name: string;
  models: Record<string, ModelDefinition>;
}

export interface ModelConfiguration {
  active_profile: string;
  available_profiles: string[];
  profile: ModelProfile;
}
