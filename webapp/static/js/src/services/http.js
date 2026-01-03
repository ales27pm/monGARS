import { apiUrl } from "../config.js";

export function createHttpService({ config, auth }) {
  class AuthError extends Error {
    constructor(message, { code, detail, status, cause } = {}) {
      super(message);
      this.name = "AuthError";
      this.code = code;
      this.detail = detail;
      this.status = status;
      this.cause = cause;
    }
  }

  class ApiError extends Error {
    constructor(message, { code, detail, status, cause } = {}) {
      super(message);
      this.name = "ApiError";
      this.code = code;
      this.detail = detail;
      this.status = status;
      this.cause = cause;
    }
  }

  function safeJsonParse(text) {
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  function normaliseJwt(raw) {
    if (raw == null) return null;
    let t = String(raw).trim();
    if (!t) return null;

    // Common footguns: storing the literal strings "null" / "undefined"
    if (t === "null" || t === "undefined") return null;

    // People often paste the whole header value.
    if (t.toLowerCase().startsWith("bearer ")) t = t.slice(7).trim();

    // People also paste tokens wrapped in quotes.
    if (
      (t.startsWith("\"") && t.endsWith("\"")) ||
      (t.startsWith("'") && t.endsWith("'"))
    ) {
      t = t.slice(1, -1).trim();
    }

    // Some SDKs store a JSON blob.
    if (t.startsWith("{") && t.endsWith("}")) {
      const obj = safeJsonParse(t);
      if (obj && typeof obj === "object") {
        const v = obj.access_token || obj.token || obj.jwt;
        if (typeof v === "string") return normaliseJwt(v);
      }
    }

    // Minimal sanity: JWTs have 2 dots.
    const dotCount = (t.match(/\./g) || []).length;
    if (dotCount !== 2) return null;

    return t;
  }

  function getJwtOrNull() {
    // Prefer the injected auth service, but fall back to localStorage keys
    // so debugging in dev tools is less painful.
    let raw = null;
    try {
      raw = typeof auth?.getJwt === "function" ? auth.getJwt() : null;
    } catch {
      raw = null;
    }

    const t = normaliseJwt(raw);
    if (t) return t;

    try {
      if (typeof localStorage === "undefined") return null;
      return (
        normaliseJwt(localStorage.getItem("mongars_jwt")) ||
        normaliseJwt(localStorage.getItem("jwt")) ||
        normaliseJwt(localStorage.getItem("access_token"))
      );
    } catch {
      return null;
    }
  }
  async function authorisedFetch(path, options = {}) {
    const { auth: useAuth = true, ...rest } = options;
    const headers = new Headers(rest.headers || {});
    if (useAuth) {
      const jwt = await getJwtOrNull(auth);
      if (!jwt) {
        throw new AuthError("Authorization failed: missing or unreadable JWT", {
          code: "AUTH_MISSING",
        });
      }
      if (!headers.has("Authorization")) headers.set("Authorization", `Bearer ${jwt}`);
    }
    try {
      return await fetch(apiUrl(config, path), { ...rest, headers });
    } catch (err) {
      if (err && typeof err === "object" && err.name === "AbortError") {
        throw err;
      }
      const message = err instanceof Error ? err.message : String(err);
      throw new ApiError(`Network request failed: ${message}`, {
        code: "NETWORK",
        cause: err,
      });
    }
  }

  async function fetchTicket({ signal } = {}) {
    const resp = await authorisedFetch(
      "/api/v1/auth/ws/ticket",
      {
        method: "POST",
        signal,
      },
    );

    if (!resp.ok) {
      const bodyText = await resp.text().catch(() => "");
      let detail = bodyText;
      try {
        const parsed = JSON.parse(bodyText);
        detail = parsed?.detail || detail;
      } catch {
        // non-JSON body
      }

      if (resp.status === 401 || resp.status === 403) {
        throw new AuthError("Ticket error: authentication failed", {
          code: "AUTH_INVALID",
          status: resp.status,
          detail,
        });
      }
      throw new ApiError(`Ticket error: HTTP ${resp.status}`, {
        code: "HTTP_ERROR",
        status: resp.status,
        detail,
      });
    }

    const body = await resp.json().catch(() => null);
    if (!body || !body.ticket) {
      throw new ApiError("Ticket response invalide", {
        code: "BAD_RESPONSE",
        status: resp.status,
      });
    }
    return body.ticket;
  }

  async function postChat(message) {
    const resp = await authorisedFetch("/api/v1/conversation/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      const parsed = parseMaybeJson(text);
      const detail = (parsed && (parsed.detail || parsed.error)) || text || null;
      throw new ApiError("Chat request failed", {
        status: resp.status,
        code: "HTTP_ERROR",
        detail,
      });
    }
    return resp;
  }

  async function postEmbed(text, options = {}) {
    if (!config.embedServiceUrl) {
      throw new Error(
        "Service d'embedding indisponible: aucune URL configurée.",
      );
    }
    const payload = {
      inputs: Array.isArray(text) ? text : [text],
    };
    if (Object.prototype.hasOwnProperty.call(options, "normalise")) {
      payload.normalise = Boolean(options.normalise);
    } else {
      payload.normalise = false;
    }
    const resp = await authorisedFetch(config.embedServiceUrl, {
      auth: false,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const bodyText = await resp.text().catch(() => "");
      const parsed = parseMaybeJson(bodyText);
      const detail = (parsed && (parsed.detail || parsed.error)) || bodyText || null;
      throw new ApiError("Embedding request failed", {
        status: resp.status,
        code: "HTTP_ERROR",
        detail,
      });
    }
    const data = await resp.json();
    if (!data || !Array.isArray(data.vectors)) {
      throw new Error("Embedding response invalide: vecteurs manquants");
    }
    return data;
  }

  async function postSuggestions(prompt) {
    const resp = await authorisedFetch("/api/v1/ui/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        actions: ["code", "summarize", "explain"],
      }),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      const parsed = parseMaybeJson(text);
      const detail = (parsed && (parsed.detail || parsed.error)) || text || null;
      throw new ApiError("Suggestions request failed", {
        status: resp.status,
        detail,
        code: "SUGGESTIONS_ERROR",
      });
    }
    const payload = await resp.json();
    if (!payload || !Array.isArray(payload.actions)) {
      throw new Error("Suggestion response invalid: actions array missing");
    }
    return payload;
  }

  async function listUsers() {
    const resp = await authorisedFetch("/api/v1/user/list");
    let payload;
    try {
      payload = await resp.json();
    } catch (err) {
      payload = null;
    }
    if (!resp.ok) {
      const detail =
        payload && (payload.detail || payload.error || payload.message);
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    if (!payload || !Array.isArray(payload.users)) {
      throw new Error("User list response invalid: users array missing");
    }
    return payload.users;
  }

  async function changePassword({ oldPassword, newPassword }) {
    const body = {
      old_password: oldPassword,
      new_password: newPassword,
    };
    const resp = await authorisedFetch("/api/v1/user/change-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let payload;
    try {
      payload = await resp.json();
    } catch (err) {
      payload = null;
    }
    if (!resp.ok) {
      const detail =
        payload && (payload.detail || payload.error || payload.message);
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    if (!payload || typeof payload !== "object") {
      return { status: "changed" };
    }
    return payload;
  }

  return {
    // Export error types / helpers so callers (e.g., socket.js) can classify
    // failures without string-matching.
    AuthError,
    ApiError,
    getJwt: () => getJwtOrNull(auth),
    isAuthError: (err) =>
      !!err &&
      (err.name === "AuthError" ||
        (typeof err.code === "string" && err.code.startsWith("AUTH_"))),
    fetchTicket,
    postChat,
    postEmbed,
    postSuggestions,
    listUsers,
    changePassword,
  };
}
