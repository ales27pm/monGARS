import {
  describe,
  expect,
  it,
  beforeEach,
  afterEach,
  jest,
} from "@jest/globals";
import { createHttpService } from "../http.js";

function createFetchResponse(overrides = {}) {
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    json: jest.fn().mockResolvedValue(undefined),
    text: jest.fn().mockResolvedValue(""),
    ...overrides,
  };
}

describe("createHttpService", () => {
  const baseUrl = new URL("https://api.example.test");
  let auth;
  let config;

  beforeEach(() => {
    auth = {
      getJwt: jest.fn().mockResolvedValue("jwt-token"),
    };
    config = {
      baseUrl,
      embedServiceUrl: "https://embed.example.test/vectors",
    };
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
    delete global.fetch;
  });

  it("adds an authorization header when posting chat messages", async () => {
    const response = createFetchResponse();
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    const result = await service.postChat("Hello world");

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/conversation/chat");
    expect(options.method).toBe("POST");
    const headers = options.headers;
    expect(headers.get("Authorization")).toBe("Bearer jwt-token");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(JSON.parse(options.body)).toEqual({ message: "Hello world" });
    expect(result).toBe(response);
  });

  it("throws when the JWT cannot be retrieved", async () => {
    auth.getJwt.mockRejectedValueOnce(new Error("boom"));
    const service = createHttpService({ config, auth });

    await expect(service.postChat("hi")).rejects.toThrow(
      "Authorization failed: missing or unreadable JWT",
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("retrieves a websocket ticket and returns the ticket value", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ ticket: "abc123" }),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.fetchTicket()).resolves.toBe("abc123");
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/auth/ws/ticket");
    expect(options.method).toBe("POST");
  });

  it("raises an error when the ticket request fails", async () => {
    const response = createFetchResponse({
      ok: false,
      status: 403,
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.fetchTicket()).rejects.toThrow("Ticket error: 403");
  });

  it("raises an error when the ticket payload is invalid", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({}),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.fetchTicket()).rejects.toThrow(
      "Ticket response invalide",
    );
  });

  it("throws when no embedding service URL is configured", async () => {
    const service = createHttpService({
      config: { baseUrl, embedServiceUrl: null },
      auth,
    });

    await expect(service.postEmbed("text"))
      .rejects.toThrow("Service d'embedding indisponible: aucune URL configurée.");
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("posts to the embedding service with normalise defaulting to false", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ vectors: [[1, 2, 3]] }),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    const payload = await service.postEmbed("hello");

    expect(payload).toEqual({ vectors: [[1, 2, 3]] });
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://embed.example.test/vectors");
    expect(options.method).toBe("POST");
    const body = JSON.parse(options.body);
    expect(body).toEqual({ inputs: ["hello"], normalise: false });
  });

  it("passes through the requested normalise flag for embeddings", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ vectors: [[4, 5]] }),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await service.postEmbed("hello", { normalise: true });
    const [, options] = global.fetch.mock.calls[0];
    expect(JSON.parse(options.body).normalise).toBe(true);
  });

  it("throws when the embedding call fails", async () => {
    const response = createFetchResponse({
      ok: false,
      status: 503,
      text: jest.fn().mockResolvedValue("offline"),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.postEmbed("hello")).rejects.toThrow(
      "HTTP 503: offline",
    );
  });

  it("throws when the embedding response payload is malformed", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ invalid: true }),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.postEmbed("hello")).rejects.toThrow(
      "Embedding response invalide: vecteurs manquants",
    );
  });

  it("propagates errors from postSuggestions when the response is not ok", async () => {
    const response = createFetchResponse({
      ok: false,
      status: 500,
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.postSuggestions("prompt"))
      .rejects.toThrow("Suggestion error: 500");
  });

  it("returns parsed JSON from the suggestion endpoint", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ actions: ["code"] }),
    });
    global.fetch.mockResolvedValueOnce(response);
    const service = createHttpService({ config, auth });

    await expect(service.postSuggestions("prompt"))
      .resolves.toEqual({ actions: ["code"] });
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/ui/suggestions");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      prompt: "prompt",
      actions: ["code", "summarize", "explain"],
    });
  });
});
