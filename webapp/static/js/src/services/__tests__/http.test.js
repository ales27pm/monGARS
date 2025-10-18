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
  let service;

  const makeService = (overrides = {}) => {
    const mergedConfig = {
      ...config,
      ...(overrides.config ?? {}),
    };
    const mergedAuth = overrides.auth ?? auth;

    return createHttpService({
      config: mergedConfig,
      auth: mergedAuth,
    });
  };

  beforeEach(() => {
    auth = {
      getJwt: jest.fn().mockResolvedValue("jwt-token"),
    };
    config = {
      baseUrl,
      embedServiceUrl: "https://embed.example.test/vectors",
    };
    global.fetch = jest.fn();
    service = makeService();
  });

  afterEach(() => {
    jest.resetAllMocks();
    delete global.fetch;
  });

  it("adds an authorization header when posting chat messages", async () => {
    const response = createFetchResponse();
    global.fetch.mockResolvedValueOnce(response);

    const result = await service.postChat("Hello world");

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/conversation/chat");
    expect(options.method).toBe("POST");
    const { headers } = options;
    expect(headers.get("Authorization")).toBe("Bearer jwt-token");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(JSON.parse(options.body)).toEqual({ message: "Hello world" });
    expect(result).toBe(response);
  });

  it("throws when the JWT cannot be retrieved", async () => {
    auth.getJwt.mockRejectedValueOnce(new Error("boom"));

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

    await expect(service.fetchTicket()).rejects.toThrow("Ticket error: 403");
  });

  it("raises an error when the ticket payload is invalid", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({}),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.fetchTicket()).rejects.toThrow(
      "Ticket response invalide",
    );
  });

  it("throws when no embedding service URL is configured", async () => {
    const serviceWithoutEmbed = makeService({
      config: { embedServiceUrl: null },
    });

    await expect(serviceWithoutEmbed.postEmbed("text")).rejects.toThrow(
      "Service d'embedding indisponible: aucune URL configurée.",
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("posts to the embedding service with normalise defaulting to false", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ vectors: [[1, 2, 3]] }),
    });
    global.fetch.mockResolvedValueOnce(response);

    const payload = await service.postEmbed("hello");

    expect(payload).toEqual({ vectors: [[1, 2, 3]] });
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://embed.example.test/vectors");
    expect(options.method).toBe("POST");
    expect(options.headers.get("Authorization")).toBeNull();
    const body = JSON.parse(options.body);
    expect(body).toEqual({ inputs: ["hello"], normalise: false });
  });

  it("posts multiple input strings to the embedding service", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({
        vectors: [
          [1, 2, 3],
          [4, 5, 6],
        ],
      }),
    });
    global.fetch.mockResolvedValueOnce(response);

    const payload = await service.postEmbed(["foo", "bar"]);

    expect(payload).toEqual({
      vectors: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    });
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://embed.example.test/vectors");
    expect(options.method).toBe("POST");
    const body = JSON.parse(options.body);
    expect(body).toEqual({ inputs: ["foo", "bar"], normalise: false });
  });

  it("passes through the requested normalise flag for embeddings", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ vectors: [[4, 5]] }),
    });
    global.fetch.mockResolvedValueOnce(response);

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

    await expect(service.postEmbed("hello")).rejects.toThrow(
      "HTTP 503: offline",
    );
  });

  it("throws when the embedding response payload is malformed", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ invalid: true }),
    });
    global.fetch.mockResolvedValueOnce(response);

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

    await expect(service.postSuggestions("prompt")).rejects.toThrow(
      "Suggestion error: 500",
    );
  });

  it("returns parsed JSON from the suggestion endpoint", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ actions: ["code"] }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.postSuggestions("prompt")).resolves.toEqual({
      actions: ["code"],
    });
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/ui/suggestions");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      prompt: "prompt",
      actions: ["code", "summarize", "explain"],
    });
  });

  it("throws an error when postSuggestions receives a malformed payload (missing actions)", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({}),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.postSuggestions("prompt")).rejects.toThrow(
      /Suggestion response invalid/i,
    );
  });

  it("throws an error when postSuggestions receives a malformed payload (actions is not an array)", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ actions: "nope" }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.postSuggestions("prompt")).rejects.toThrow(
      /Suggestion response invalid/i,
    );
  });

  it("retrieves the list of users", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ users: ["alice", "bob"] }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.listUsers()).resolves.toEqual(["alice", "bob"]);
    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/user/list");
    const headers = options.headers;
    expect(headers.get("Authorization")).toBe("Bearer jwt-token");
    expect(options.method ?? "GET").toBe("GET");
  });

  it("throws when the user list payload is missing", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({}),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.listUsers()).rejects.toThrow(
      "User list response invalid: users array missing",
    );
  });

  it("throws with detail when the user list request fails", async () => {
    const response = createFetchResponse({
      ok: false,
      status: 403,
      json: jest.fn().mockResolvedValue({ detail: "Forbidden" }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(service.listUsers()).rejects.toThrow("Forbidden");
  });

  it("changes the password and returns the payload", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ status: "changed" }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(
      service.changePassword({
        oldPassword: "oldpass1",
        newPassword: "newpass1",
      }),
    ).resolves.toEqual({ status: "changed" });

    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toBe("https://api.example.test/api/v1/user/change-password");
    expect(options.method).toBe("POST");
    const body = JSON.parse(options.body);
    expect(body).toEqual({
      old_password: "oldpass1",
      new_password: "newpass1",
    });
    const headers = options.headers;
    expect(headers.get("Authorization")).toBe("Bearer jwt-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("rejects password changes when inputs are blank", async () => {
    await expect(service.changePassword()).rejects.toThrow(TypeError);
    await expect(
      service.changePassword({ oldPassword: "", newPassword: "" }),
    ).rejects.toThrow("oldPassword and newPassword must be non-empty strings");
  });

  it("passes AbortSignal through when changing the password", async () => {
    const controller = new AbortController();
    const response = createFetchResponse({
      json: jest.fn().mockResolvedValue({ status: "changed" }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await service.changePassword({
      oldPassword: "oldpass1",
      newPassword: "newpass1",
      signal: controller.signal,
    });

    const [, options] = global.fetch.mock.calls[0];
    expect(options.signal).toBe(controller.signal);
  });

  it("throws with detail when the password change fails", async () => {
    const response = createFetchResponse({
      ok: false,
      status: 403,
      json: jest.fn().mockResolvedValue({ detail: "Incorrect password" }),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(
      service.changePassword({
        oldPassword: "oldpass1",
        newPassword: "newpass1",
      }),
    ).rejects.toThrow("Incorrect password");
  });

  it("returns a fallback payload when the password change response has no body", async () => {
    const response = createFetchResponse({
      json: jest.fn().mockRejectedValue(new Error("no body")),
    });
    global.fetch.mockResolvedValueOnce(response);

    await expect(
      service.changePassword({
        oldPassword: "oldpass1",
        newPassword: "newpass1",
      }),
    ).resolves.toEqual({ status: "changed" });
  });
});
