import fetch, { HeadersInit, RequestInit, Response } from "cross-fetch";
import {
  ChatRequest,
  ChatResponse,
  MemoryItem,
  ModelConfiguration,
  PeerLoadSnapshot,
  PeerTelemetryEnvelope,
  PeerTelemetryPayload,
  PeerRegistration,
  ProvisionReport,
  ProvisionRequest,
  RagContextRequest,
  RagContextResponse,
  SuggestRequest,
  SuggestResponse,
  TokenResponse,
} from "./models.js";
import { ApiError, AuthenticationError } from "./errors.js";

export * from "./errors.js";
export * from "./models.js";

export interface ClientOptions {
  baseUrl: string;
  defaultHeaders?: HeadersInit;
  fetchImpl?: typeof fetch;
}

async function parseResponse(response: Response): Promise<any> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    return text;
  }
}

function raiseForStatus(status: number, payload: any): never {
  const detail =
    payload && typeof payload === "object" ? payload.detail : undefined;
  if (status === 401 || status === 403) {
    throw new AuthenticationError(
      status,
      detail ?? "Authentication failed",
      payload,
    );
  }
  throw new ApiError(
    status,
    detail ?? `Request failed with status ${status}`,
    payload,
  );
}

export class MonGARSClient {
  private readonly baseUrl: string;
  private token?: string;
  private readonly defaultHeaders: HeadersInit;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.defaultHeaders = options.defaultHeaders ?? {
      Accept: "application/json",
      "User-Agent": "monGARS-SDK/1.0",
    };
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  private buildHeaders(extra?: HeadersInit): HeadersInit {
    const headers: Record<string, string> = {
      ...this.defaultHeaders,
    };
    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }
    if (extra) {
      return { ...headers, ...(extra as Record<string, string>) };
    }
    return headers;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: this.buildHeaders(init.headers as HeadersInit),
    });
    if (!response.ok) {
      const payload = await parseResponse(response);
      raiseForStatus(response.status, payload);
    }
    const payload = await parseResponse(response);
    return payload as T;
  }

  async login(credentials: {
    username: string;
    password: string;
  }): Promise<TokenResponse> {
    const body = new URLSearchParams();
    body.append("username", credentials.username);
    body.append("password", credentials.password);
    const response = await this.fetchImpl(`${this.baseUrl}/token`, {
      method: "POST",
      headers: this.buildHeaders({
        "Content-Type": "application/x-www-form-urlencoded",
      }),
      body,
    });
    const payload = await parseResponse(response);
    if (!response.ok) {
      raiseForStatus(response.status, payload);
    }
    this.token = payload.access_token;
    return payload as TokenResponse;
  }

  async registerUser(payload: {
    username: string;
    password: string;
  }): Promise<Record<string, unknown>> {
    return this.request("/api/v1/user/register", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async chat(payload: ChatRequest): Promise<ChatResponse> {
    return this.request("/api/v1/conversation/chat", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async history(userId: string, limit = 10): Promise<MemoryItem[]> {
    const params = new URLSearchParams({
      user_id: userId,
      limit: String(limit),
    });
    return this.request(`/api/v1/conversation/history?${params.toString()}`);
  }

  async fetchRagContext(
    payload: RagContextRequest,
  ): Promise<RagContextResponse> {
    return this.request("/api/v1/review/rag-context", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async suggestActions(payload: SuggestRequest): Promise<SuggestResponse> {
    return this.request("/api/v1/ui/suggestions", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async sendPeerMessage(
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.request("/api/v1/peer/message", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async registerPeer(
    payload: PeerRegistration,
  ): Promise<Record<string, unknown>> {
    return this.request("/api/v1/peer/register", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async unregisterPeer(
    payload: PeerRegistration,
  ): Promise<Record<string, unknown>> {
    return this.request("/api/v1/peer/unregister", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async listPeers(): Promise<string[]> {
    return this.request("/api/v1/peer/list");
  }

  async peerLoad(): Promise<PeerLoadSnapshot> {
    return this.request("/api/v1/peer/load");
  }

  async publishPeerTelemetry(
    payload: PeerTelemetryPayload,
  ): Promise<Record<string, string>> {
    return this.request("/api/v1/peer/telemetry", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }

  async peerTelemetry(): Promise<PeerTelemetryEnvelope> {
    return this.request("/api/v1/peer/telemetry");
  }

  async modelConfiguration(): Promise<ModelConfiguration> {
    return this.request("/api/v1/models");
  }

  async provisionModels(payload: ProvisionRequest): Promise<ProvisionReport> {
    return this.request("/api/v1/models/provision", {
      method: "POST",
      headers: this.buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  }
}
