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
    return resp.json();
  }

  return {
    fetchTicket,
    postChat,
    postSuggestions,
  };
}
