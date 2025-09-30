/* monGARS chat frontend: event-driven UI wired to typed backend events */

(function () {
  const config = window.chatConfig || {};
  const els = {
    transcript: document.getElementById("transcript"),
    composer: document.getElementById("composer"),
    prompt: document.getElementById("prompt"),
    send: document.getElementById("send"),
    wsStatus: document.getElementById("ws-status"),
    quickActions: document.getElementById("quick-actions"),
    connection: document.getElementById("connection"),
    errorAlert: document.getElementById("error-alert"),
    errorMessage: document.getElementById("error-message"),
  };

  if (!els.transcript || !els.composer || !els.prompt) {
    return;
  }

  const baseUrl = (() => {
    const candidate = config.fastapiUrl || window.location.origin;
    try {
      return new URL(candidate);
    } catch (err) {
      console.error("Invalid FASTAPI URL", err, candidate);
      return new URL(window.location.origin);
    }
  })();

  const apiUrl = (path) => new URL(path, baseUrl).toString();

  try {
    if (config.token) {
      window.localStorage.setItem("jwt", config.token);
    }
  } catch (err) {
    console.warn("Unable to persist JWT in localStorage", err);
  }

  const historyElement = document.getElementById("chat-history");
  let chatHistory = [];
  if (historyElement) {
    try {
      const parsed = JSON.parse(historyElement.textContent || "null");
      if (Array.isArray(parsed)) {
        chatHistory = parsed;
      } else if (parsed && parsed.error) {
        showError(parsed.error);
      }
    } catch (err) {
      console.error("Unable to parse chat history", err);
    }
    historyElement.remove();
  }

  let historyBootstrapped = els.transcript.childElementCount > 0;

  // ---- UX helpers ---------------------------------------------------------
  const nowISO = () => new Date().toISOString();
  const statusLabels = {
    offline: "Hors ligne",
    connecting: "Connexion…",
    online: "En ligne",
    error: "Erreur",
  };

  function setBusy(busy) {
    els.transcript.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function hideError() {
    if (!els.errorAlert) return;
    els.errorAlert.classList.add("d-none");
    if (els.errorMessage) {
      els.errorMessage.textContent = "";
    }
  }

  function showError(message) {
    if (!els.errorAlert || !els.errorMessage) return;
    els.errorMessage.textContent = message;
    els.errorAlert.classList.remove("d-none");
  }

  function line(role, html) {
    const row = document.createElement("div");
    row.className = `chat-row chat-${role}`;
    row.innerHTML = html;
    els.transcript.appendChild(row);
    els.transcript.scrollTop = els.transcript.scrollHeight;
    return row;
  }

  function escapeHTML(str) {
    return String(str).replace(
      /[&<>"']/g,
      (ch) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[ch],
    );
  }

  function formatTimestamp(ts) {
    if (!ts) return "";
    try {
      return new Date(ts).toLocaleString("fr-CA");
    } catch (err) {
      return String(ts);
    }
  }

  function renderHistory(entries, options = {}) {
    const { replace = false } = options;
    if (!Array.isArray(entries) || entries.length === 0) {
      if (replace) {
        els.transcript.innerHTML = "";
        historyBootstrapped = false;
      }
      return;
    }
    if (replace) {
      els.transcript.innerHTML = "";
      historyBootstrapped = false;
      streamRow = null;
      streamBuf = "";
    }
    if (historyBootstrapped && !replace) {
      return;
    }
    entries
      .slice()
      .reverse()
      .forEach((item) => {
        if (item.query) {
          line(
            "user",
            `<div class="chat-bubble">${escapeHTML(item.query)}<div class="chat-meta">${escapeHTML(
              formatTimestamp(item.timestamp),
            )}</div></div>`,
          );
        }
        if (item.response) {
          line(
            "assistant",
            `<div class="chat-bubble">${escapeHTML(item.response)}<div class="chat-meta">${escapeHTML(
              formatTimestamp(item.timestamp),
            )}</div></div>`,
          );
        }
      });
    historyBootstrapped = true;
  }

  renderHistory(chatHistory);

  // Streaming buffer for the current assistant message
  let streamRow = null;
  let streamBuf = "";

  function startStream() {
    streamBuf = "";
    streamRow = line(
      "assistant",
      '<div class="chat-bubble"><span class="chat-cursor">▍</span></div>',
    );
  }

  function appendStream(delta) {
    if (!streamRow) {
      startStream();
    }
    streamBuf += delta || "";
    const bubble = streamRow.querySelector(".chat-bubble");
    if (bubble) {
      bubble.textContent = streamBuf;
      const cursor = document.createElement("span");
      cursor.className = "chat-cursor";
      cursor.textContent = "▍";
      bubble.appendChild(cursor);
    }
  }

  function endStream(data) {
    if (!streamRow) {
      return;
    }
    const bubble = streamRow.querySelector(".chat-bubble");
    if (bubble) {
      bubble.textContent = streamBuf;
      const meta = document.createElement("div");
      meta.className = "chat-meta";
      const ts = data && data.timestamp ? data.timestamp : nowISO();
      meta.textContent = formatTimestamp(ts);
      if (data && data.error) {
        meta.classList.add("text-danger");
        meta.textContent = `${meta.textContent} • ${data.error}`;
      }
      bubble.appendChild(meta);
    }
    streamRow = null;
    streamBuf = "";
  }

  function announceConnection(message, variant = "info") {
    if (!els.connection) {
      return;
    }
    const classList = els.connection.classList;
    Array.from(classList)
      .filter((cls) => cls.startsWith("alert-") && cls !== "alert")
      .forEach((cls) => classList.remove(cls));
    classList.add("alert");
    classList.add(`alert-${variant}`);
    els.connection.textContent = message;
    classList.remove("visually-hidden");
    window.setTimeout(() => {
      classList.add("visually-hidden");
    }, 4000);
  }

  // ---- WS ticket + socket -------------------------------------------------
  async function getJwt() {
    try {
      const stored = window.localStorage.getItem("jwt");
      if (stored) {
        return stored;
      }
    } catch (err) {
      console.warn("Unable to read JWT from localStorage", err);
    }
    if (config.token) {
      return config.token;
    }
    throw new Error("Missing JWT (store it in localStorage as 'jwt').");
  }

  async function fetchTicket() {
    const jwt = await getJwt();
    const resp = await fetch(apiUrl("/api/v1/auth/ws/ticket"), {
      method: "POST",
      headers: { Authorization: `Bearer ${jwt}` },
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

  let ws;
  let wsHBeat;
  let reconnectBackoff = 500; // ms
  const BACKOFF_MAX = 8000;

  function setWsStatus(state, title) {
    if (!els.wsStatus) return;
    const label = statusLabels[state] || state;
    els.wsStatus.textContent = label;
    els.wsStatus.className = `badge ws-badge ${state}`;
    if (title) {
      els.wsStatus.title = title;
    } else {
      els.wsStatus.removeAttribute("title");
    }
  }

  async function openSocket() {
    try {
      const ticket = await fetchTicket();
      const wsUrl = new URL("/ws/chat/", baseUrl);
      wsUrl.protocol = baseUrl.protocol === "https:" ? "wss:" : "ws:";
      wsUrl.searchParams.set("t", ticket);

      ws = new WebSocket(wsUrl.toString());
      setWsStatus("connecting");

      ws.onopen = () => {
        setWsStatus("online");
        hideError();
        wsHBeat = window.setInterval(() => {
          safeSend({ type: "client.ping", ts: nowISO() });
        }, 20000);
        reconnectBackoff = 500;
      };

      ws.onmessage = (evt) => {
        try {
          const ev = JSON.parse(evt.data);
          handleEvent(ev);
        } catch (err) {
          console.error("Bad event payload", err, evt.data);
        }
      };

      ws.onclose = () => {
        setWsStatus("offline");
        if (wsHBeat) {
          clearInterval(wsHBeat);
        }
        const delay = reconnectBackoff + Math.floor(Math.random() * 250);
        reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
        window.setTimeout(openSocket, delay);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        setWsStatus("error", "Erreur WebSocket");
      };
    } catch (err) {
      console.error(err);
      setWsStatus("error", String(err));
      const delay = Math.min(BACKOFF_MAX, reconnectBackoff);
      reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
      window.setTimeout(openSocket, delay);
    }
  }

  function safeSend(obj) {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
      }
    } catch (err) {
      console.warn("Unable to send over WebSocket", err);
    }
  }

  // ---- Typed event router -------------------------------------------------
  function handleEvent(ev) {
    const type = ev && ev.type ? ev.type : "";
    const data = ev && ev.data ? ev.data : {};
    switch (type) {
      case "ws.connected": {
        if (data && data.origin) {
          announceConnection(`Connecté via ${data.origin}`);
        } else {
          announceConnection("Connecté au serveur.");
        }
        break;
      }
      case "history.snapshot": {
        if (data && Array.isArray(data.items)) {
          renderHistory(data.items, { replace: true });
        }
        break;
      }
      case "ai_model.response_chunk": {
        const delta =
          typeof data.delta === "string" ? data.delta : data.text || "";
        appendStream(delta);
        break;
      }
      case "ai_model.response_complete": {
        if (data && data.text && !streamBuf) {
          appendStream(data.text);
        }
        endStream(data);
        setBusy(false);
        if (data && data.ok === false && data.error) {
          line(
            "system",
            `<div class="chat-bubble chat-bubble-error">${escapeHTML(data.error)}</div>`,
          );
        }
        break;
      }
      case "chat.message": {
        if (!streamRow) {
          startStream();
        }
        if (data && typeof data.response === "string" && !streamBuf) {
          appendStream(data.response);
        }
        endStream(data);
        setBusy(false);
        break;
      }
      case "evolution_engine.training_complete": {
        line(
          "system",
          `<div class="chat-bubble chat-bubble-ok">Évolution mise à jour ${escapeHTML(
            data.version || "",
          )}</div>`,
        );
        break;
      }
      case "evolution_engine.training_failed": {
        line(
          "system",
          `<div class="chat-bubble chat-bubble-error">Échec de l'évolution : ${escapeHTML(
            data.error || "inconnu",
          )}</div>`,
        );
        break;
      }
      case "sleep_time_compute.phase_start": {
        line(
          "system",
          '<div class="chat-bubble chat-bubble-hint">Optimisation en arrière-plan démarrée…</div>',
        );
        break;
      }
      case "sleep_time_compute.creative_phase": {
        line(
          "system",
          `<div class="chat-bubble chat-bubble-hint">Exploration de ${escapeHTML(
            Number(data.ideas || 1).toString(),
          )} idées…</div>`,
        );
        break;
      }
      case "performance.alert": {
        line(
          "system",
          `<div class="chat-bubble chat-bubble-warn">Perf : ${escapeHTML(formatPerf(data))}</div>`,
        );
        break;
      }
      case "ui.suggestions": {
        applyQuickActionOrdering(
          Array.isArray(data.actions) ? data.actions : [],
        );
        break;
      }
      default:
        if (type && type.startsWith("ws.")) {
          return;
        }
        console.debug("Unhandled event", ev);
    }
  }

  function formatPerf(d) {
    const bits = [];
    if (d && typeof d.cpu !== "undefined") {
      const cpu = Number(d.cpu);
      if (!Number.isNaN(cpu)) {
        bits.push(`CPU ${cpu.toFixed(0)}%`);
      }
    }
    if (d && typeof d.ttfb_ms !== "undefined") {
      const ttfb = Number(d.ttfb_ms);
      if (!Number.isNaN(ttfb)) {
        bits.push(`TTFB ${ttfb} ms`);
      }
    }
    return bits.join(" • ") || "mise à jour";
  }

  function applyQuickActionOrdering(suggestions) {
    if (!els.quickActions) return;
    if (!Array.isArray(suggestions) || suggestions.length === 0) return;
    const buttons = Array.from(els.quickActions.querySelectorAll("button.qa"));
    const lookup = new Map();
    buttons.forEach((btn) => lookup.set(btn.dataset.action, btn));
    const frag = document.createDocumentFragment();
    suggestions.forEach((key) => {
      if (lookup.has(key)) {
        frag.appendChild(lookup.get(key));
        lookup.delete(key);
      }
    });
    lookup.forEach((btn) => frag.appendChild(btn));
    els.quickActions.innerHTML = "";
    els.quickActions.appendChild(frag);
  }

  // ---- Debounced AUI suggestions -----------------------------------------
  let auiTimer = null;

  async function fetchSuggestionsDebounced() {
    if (auiTimer) {
      clearTimeout(auiTimer);
    }
    auiTimer = window.setTimeout(fetchSuggestions, 220);
  }

  async function fetchSuggestions() {
    const text = (els.prompt.value || "").trim();
    if (!text) {
      return;
    }
    try {
      const jwt = await getJwt();
      const resp = await fetch(apiUrl("/api/v1/ui/suggestions"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${jwt}`,
        },
        body: JSON.stringify({
          prompt: text,
          actions: ["code", "summarize", "explain"],
        }),
      });
      if (!resp.ok) {
        return;
      }
      const payload = await resp.json();
      if (payload && Array.isArray(payload.actions)) {
        applyQuickActionOrdering(payload.actions);
      }
    } catch (err) {
      console.debug("AUI suggestion fetch failed", err);
    }
  }

  // ---- Submit & quick actions --------------------------------------------
  els.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = (els.prompt.value || "").trim();
    if (!text) {
      return;
    }
    hideError();
    const submittedAt = nowISO();
    line(
      "user",
      `<div class=\"chat-bubble\">${escapeHTML(text)}<div class=\"chat-meta\">${escapeHTML(
        formatTimestamp(submittedAt),
      )}</div></div>`,
    );
    els.prompt.value = "";
    setBusy(true);
    applyQuickActionOrdering(["code", "summarize", "explain"]);

    try {
      const jwt = await getJwt();
      const resp = await fetch(apiUrl("/api/v1/conversation/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${jwt}`,
        },
        body: JSON.stringify({ message: text }),
      });
      if (!resp.ok) {
        const payload = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${payload}`);
      }
      startStream();
    } catch (err) {
      setBusy(false);
      showError(String(err));
      line(
        "system",
        `<div class="chat-bubble chat-bubble-error">${escapeHTML(String(err))}</div>`,
      );
    }
  });

  if (els.quickActions) {
    els.quickActions.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) {
        return;
      }
      const action = target.dataset.action;
      if (!action) {
        return;
      }
      const presets = {
        code: "Je souhaite écrire du code…",
        summarize: "Résume la dernière conversation.",
        explain: "Explique ta dernière réponse plus simplement.",
      };
      els.prompt.value = presets[action] || action;
      els.composer.dispatchEvent(new Event("submit"));
    });
  }

  if (els.prompt) {
    els.prompt.addEventListener("input", fetchSuggestionsDebounced);
  }

  // ---- Dark mode toggle ---------------------------------------------------
  const darkModeKey = "dark-mode";
  const toggleBtn = document.getElementById("toggle-dark-mode");

  function applyDarkMode(enabled) {
    document.body.classList.toggle("dark-mode", enabled);
    if (toggleBtn) {
      toggleBtn.textContent = enabled ? "Mode Clair" : "Mode Sombre";
    }
  }

  try {
    applyDarkMode(window.localStorage.getItem(darkModeKey) === "1");
  } catch (err) {
    console.warn("Unable to read dark mode preference", err);
  }

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const enabled = !document.body.classList.contains("dark-mode");
      applyDarkMode(enabled);
      try {
        window.localStorage.setItem(darkModeKey, enabled ? "1" : "0");
      } catch (err) {
        console.warn("Unable to persist dark mode preference", err);
      }
    });
  }

  // ---- Boot ---------------------------------------------------------------
  openSocket();
})();
