import { apiUrl } from "../config.js";

export function createHttpService({ config, auth }) {
  async function authorisedFetch(path, options = {}) {
    const { auth: useAuth = true, ...rest } = options;
    const headers = new Headers(rest.headers || {});
    if (useAuth) {
      let jwt;
      try {
        jwt = await auth.getJwt();
      } catch (err) {
        // Surface a consistent error and preserve abort semantics
        throw new Error("Authorization failed: missing or unreadable JWT");
      }
      if (!headers.has("Authorization")) {
        headers.set("Authorization", `Bearer ${jwt}`);
      }
    }
    try {
      return await fetch(apiUrl(config, path), { ...rest, headers });
    } catch (err) {
      if (err && typeof err === "object" && err.name === "AbortError") {
        throw err;
      }
      const message = err instanceof Error ? err.message : String(err);
      const wrappedError = new Error(
        `Network request failed: ${message}. Original error: ${String(err)}`
      );
      wrappedError.originalError = err;
      throw wrappedError;
    }
  }

  async function fetchTicket() {
    const resp = await authorisedFetch("/api/v1/auth/ws/ticket", {
      method: "POST",
    });
    if (!resp.ok) {
      throw new Error(`Ticket error: ${resp.status}`);
    }
    const body = await resp.json();
    if (!body || !body.ticket) {
      throw new Error("Ticket response invalide");
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
      const payload = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${payload}`);
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
      const bodyText = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${bodyText}`);
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
      throw new Error(`Suggestion error: ${resp.status}`);
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
    fetchTicket,
    postChat,
    postEmbed,
    postSuggestions,
    listUsers,
    changePassword,
  };
}
