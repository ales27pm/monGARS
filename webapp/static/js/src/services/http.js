import { apiUrl } from "../config.js";

export function createHttpService({ config, auth }) {
  async function authorisedFetch(path, options = {}) {
    let jwt;
    try {
      jwt = await auth.getJwt();
    } catch (err) {
      // Surface a consistent error and preserve abort semantics
      throw new Error("Authorization failed: missing or unreadable JWT");
    }
    const headers = new Headers(options.headers || {});
    if (!headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${jwt}`);
    }
    return fetch(apiUrl(config, path), { ...options, headers });
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
        "Service d'embedding indisponible: aucune URL configurée."
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

  return {
    fetchTicket,
    postChat,
    postEmbed,
    postSuggestions,
  };
}
