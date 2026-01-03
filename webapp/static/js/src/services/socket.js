// WebSocket client with sane reconnection semantics.
//
// Goals:
// - Never hammer the server with reconnect loops when auth is missing/invalid.
// - Exponential backoff with jitter for transient failures.
// - No overlapping connection attempts (single in-flight connect).
// - Ability to "wake up" when auth becomes available again.

export function createSocketClient({ config, http, ui, onEvent }) {
  let ws = null;
  let disposed = false;

  // Connect lifecycle
  let connectSeq = 0; // increments per open() request; lets us ignore stale async work
  let inFlight = false;
  let lastClose = null;

  // Retry state
  let retryTimer = null;
  let attempts = 0;

  // Heartbeat
  let heartbeatTimer = null;

  // Auth gate
  let waitingForAuth = false;
  let authStopReason = null;

  // Optional abort for the ticket fetch
  let ticketAbort = null;

  const BACKOFF_BASE_MS = 500;
  const BACKOFF_MAX_MS = 15_000;
  const JITTER_RATIO = 0.25; // +-25%

  function safeCallOnEvent(evt) {
    try {
      onEvent?.(evt);
    } catch (err) {
      // Avoid throwing inside ws callbacks.
      console.error("[socket] onEvent handler threw", err);
    }
  }

  function setUiStatus(text, level = "info") {
    try {
      ui?.updateConnectionMeta?.(text, level);
    } catch {
      // UI is optional
    }
  }

  function clearHeartbeat() {
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  function clearRetry() {
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
  }

  function abortTicketFetch() {
    if (ticketAbort) {
      try {
        ticketAbort.abort();
      } catch {
        // ignore
      }
      ticketAbort = null;
    }
  }

  function computeBackoffMs(n) {
    const exp = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * Math.pow(2, Math.max(0, n - 1)));
    const jitter = exp * JITTER_RATIO;
    const min = Math.max(0, exp - jitter);
    const max = exp + jitter;
    return Math.floor(min + Math.random() * (max - min));
  }

  function tokenPresent() {
    try {
      const t = http?.getJwt?.();
      return typeof t === "string" && t.length > 0;
    } catch {
      return false;
    }
  }

  function stopForAuth(reason, detail) {
    waitingForAuth = true;
    authStopReason = { reason, detail };
    attempts = 0;
    clearRetry();
    clearHeartbeat();
    abortTicketFetch();

    // Close any current socket.
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      try {
        ws.close(1000, "auth-required");
      } catch {
        // ignore
      }
    }

    ws = null;
    setUiStatus(detail ? `Authentification requise: ${detail}` : "Not authenticated", "warn");
    safeCallOnEvent({
      type: "auth",
      state: "required",
      reason,
      detail: detail || null,
    });
  }

  function shouldReconnectOnClose(evt) {
    // If the caller asked us to stop, do not reconnect.
    if (disposed) return false;
    if (waitingForAuth) return false;

    // Normal close: no reconnect.
    if (evt && evt.code === 1000) return false;

    // If browser is offline, wait for 'online' event.
    if (typeof navigator !== "undefined" && navigator && navigator.onLine === false) return false;

    return true;
  }

  function scheduleReconnect(reason) {
    if (disposed || waitingForAuth) return;
    if (retryTimer) return; // already scheduled

    attempts += 1;
    const delay = computeBackoffMs(attempts);
    const secs = Math.max(0, Math.round(delay / 100) / 10);
    setUiStatus(`Déconnecté. Reconnexion dans ${secs}s…`, "warn");
    safeCallOnEvent({ type: "reconnect", in: delay, attempt: attempts, reason: reason || null });

    retryTimer = setTimeout(() => {
      retryTimer = null;
      void open();
    }, delay);
  }

  function startHeartbeat() {
    clearHeartbeat();
    heartbeatTimer = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try {
        ws.send(JSON.stringify({ type: "ping", t: Date.now() }));
      } catch {
        // If sending fails, onclose will trigger reconnect.
      }
    }, 15_000);
  }

  function handleOpenError(err) {
    // Abort is not an error we want to surface.
    if (err && err.name === "AbortError") return true;

    const isAuth = http?.isAuthError?.(err) || err?.name === "AuthError" || (typeof err?.code === "string" && err.code.startsWith("AUTH_"));

    if (isAuth) {
      const code = err?.code || "AUTH";
      const detail = err?.detail || err?.message || null;

      // Missing token: don't retry.
      if (code === "AUTH_MISSING") {
        stopForAuth("missing_token", detail);
        return true;
      }

      // Invalid/expired token: also don't retry.
      if (code === "AUTH_INVALID") {
        stopForAuth("invalid_token", detail);
        return true;
      }

      stopForAuth("auth_error", detail);
      return true;
    }

    // Anything else: let the caller see it and retry.
    console.error("[socket] open error", err);
    safeCallOnEvent({ type: "error", stage: "open", error: String(err?.message || err) });
    return false;
  }

  async function open() {
    if (disposed) return;

    // If we're stopped on auth, only resume when a token exists.
    if (waitingForAuth) {
      if (!tokenPresent()) return;
      waitingForAuth = false;
      authStopReason = null;
      setUiStatus("Jeton détecté. Connexion…", "info");
    }

    // If socket is already open/connecting, do nothing.
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    if (inFlight) return;

    inFlight = true;
    clearRetry();
    clearHeartbeat();
    abortTicketFetch();

    const mySeq = ++connectSeq;

    try {
      // Quick auth pre-check to avoid a useless round-trip.
      if (!tokenPresent()) {
        stopForAuth("missing_token", "missing or unreadable JWT");
        return;
      }

      setUiStatus("Obtention d’un ticket…", "info");
      ticketAbort = new AbortController();
      const ticket = await http.fetchTicket({ signal: ticketAbort.signal });

      // Stale attempt?
      if (disposed || mySeq !== connectSeq) return;

      attempts = 0;
      lastClose = null;

      const wsUrl = new URL(config.baseUrl);
      wsUrl.protocol = wsUrl.protocol === "https:" ? "wss:" : "ws:";
      wsUrl.pathname = "/api/v1/ws";
      wsUrl.searchParams.set("ticket", ticket);

      setUiStatus("Connexion WebSocket…", "info");
      ws = new WebSocket(wsUrl.toString());

      ws.onopen = () => {
        if (disposed) {
          try {
            ws.close(1000, "disposed");
          } catch {
            // ignore
          }
          return;
        }
        setUiStatus("Connecté", "success");
        safeCallOnEvent({ type: "connection", state: "open" });
        startHeartbeat();
      };

      ws.onmessage = (ev) => {
        let msg = null;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          // Ignore non-JSON messages.
          return;
        }
        safeCallOnEvent({ type: "message", message: msg });
      };

      ws.onerror = () => {
        // Browser doesn't expose much here; we'll reconnect on close.
        safeCallOnEvent({ type: "error", stage: "ws" });
      };

      ws.onclose = (evt) => {
        lastClose = evt || null;
        clearHeartbeat();

        // If this close belongs to a stale attempt, ignore.
        if (disposed || mySeq !== connectSeq) return;

        ws = null;
        inFlight = false;

        const code = evt?.code;
        const reason = evt?.reason || "";
        safeCallOnEvent({ type: "connection", state: "closed", code, reason });

        if (shouldReconnectOnClose(evt)) {
          scheduleReconnect(`close:${code || "?"}`);
        } else {
          setUiStatus("Déconnecté", "warn");
        }
      };
    } catch (err) {
      // Stale attempt?
      if (disposed || mySeq !== connectSeq) {
        inFlight = false;
        return;
      }

      inFlight = false;
      clearHeartbeat();

      const handled = handleOpenError(err);
      if (!handled) scheduleReconnect("open_error");
    } finally {
      // Note: we keep `inFlight` true while the ws is connecting; the `onclose`
      // handler will clear it. However if we return before creating ws, we must clear.
      if (!ws) inFlight = false;
    }
  }

  function close({ reason = "client_close" } = {}) {
    disposed = true;
    waitingForAuth = false;
    authStopReason = null;
    clearRetry();
    clearHeartbeat();
    abortTicketFetch();
    connectSeq += 1; // invalidate in-flight open()

    if (ws) {
      try {
        ws.close(1000, reason);
      } catch {
        // ignore
      }
    }
    ws = null;
    inFlight = false;
    setUiStatus("Déconnecté", "info");
    safeCallOnEvent({ type: "connection", state: "closed", code: 1000, reason });
  }

  function send(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    try {
      ws.send(JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }

  // Public method for callers to notify that auth state changed in *this* tab.
  // (storage events don't fire in the same document that sets localStorage).
  function notifyAuthChanged() {
    if (disposed) return;
    if (waitingForAuth && tokenPresent()) {
      waitingForAuth = false;
      authStopReason = null;
      void open();
    }
  }

  // Wake up on network restore, focus, and cross-tab token updates.
  function onOnline() {
    if (disposed) return;
    if (waitingForAuth) {
      notifyAuthChanged();
      return;
    }
    if (!ws && !retryTimer) void open();
  }

  function onFocus() {
    if (disposed) return;
    notifyAuthChanged();
  }

  function onStorage(ev) {
    // If any auth-relevant localStorage value changes, we can try to resume.
    if (disposed) return;
    if (!waitingForAuth) return;
    if (!ev || !ev.key) return;
    if (tokenPresent()) notifyAuthChanged();
  }

  if (typeof window !== "undefined") {
    window.addEventListener("online", onOnline);
    window.addEventListener("focus", onFocus);
    window.addEventListener("storage", onStorage);
    // Optional custom event: dispatchEvent(new Event("mongars:auth-changed"))
    window.addEventListener("mongars:auth-changed", onFocus);
  }

  function dispose() {
    close({ reason: "dispose" });
    if (typeof window !== "undefined") {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("mongars:auth-changed", onFocus);
    }
  }

  return {
    open,
    close,
    dispose,
    send,
    notifyAuthChanged,
    getState: () => ({
      disposed,
      waitingForAuth,
      authStopReason,
      attempts,
      lastClose,
      readyState: ws ? ws.readyState : null,
    }),
  };
}
