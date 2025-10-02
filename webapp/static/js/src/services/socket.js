import { nowISO } from "../utils/time.js";

export function createSocketClient({ config, http, ui, onEvent }) {
  let ws;
  let wsHBeat;
  let reconnectBackoff = 500;
  const BACKOFF_MAX = 8000;
  let retryTimer = null;
  let disposed = false;

  function clearHeartbeat() {
    if (wsHBeat) {
      clearInterval(wsHBeat);
      wsHBeat = null;
    }
  }

  function scheduleReconnect(delayBase) {
    if (disposed) {
      return 0;
    }
    const jitter = Math.floor(Math.random() * 250);
    const delay = Math.min(BACKOFF_MAX, delayBase + jitter);
    if (retryTimer) {
      clearTimeout(retryTimer);
    }
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      void openSocket();
    }, delay);
    reconnectBackoff = Math.min(
      BACKOFF_MAX,
      Math.max(500, reconnectBackoff * 2),
    );
    return delay;
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

  async function openSocket() {
    if (disposed) {
      return;
    }

    try {
      ui.updateConnectionMeta("Obtention d’un ticket de connexion…", "info");
      const ticket = await http.fetchTicket();
      if (disposed) {
        return;
      }

      const wsUrl = new URL("/ws/chat/", config.baseUrl);
      wsUrl.protocol = config.baseUrl.protocol === "https:" ? "wss:" : "ws:";
      wsUrl.searchParams.set("t", ticket);

      if (ws) {
        try {
          ws.close();
        } catch (err) {
          console.warn("WebSocket close before reconnect failed", err);
        }
        ws = null;
      }

      ws = new WebSocket(wsUrl.toString());
      ui.setWsStatus("connecting");
      ui.updateConnectionMeta("Connexion au serveur…", "info");

      ws.onopen = () => {
        if (disposed) {
          return;
        }
        if (retryTimer) {
          clearTimeout(retryTimer);
          retryTimer = null;
        }
        reconnectBackoff = 500;
        const connectedAt = nowISO();
        ui.setWsStatus("online");
        ui.updateConnectionMeta(
          `Connecté le ${ui.formatTimestamp(connectedAt)}`,
          "success",
        );
        ui.setDiagnostics({ connectedAt, lastMessageAt: connectedAt });
        ui.hideError();
        clearHeartbeat();
        wsHBeat = window.setInterval(() => {
          safeSend({ type: "client.ping", ts: nowISO() });
        }, 20000);
        ui.setComposerStatus("Connecté. Vous pouvez échanger.", "success");
        ui.scheduleComposerIdle(4000);
      };

      ws.onmessage = (evt) => {
        const receivedAt = nowISO();
        ui.setDiagnostics({ lastMessageAt: receivedAt });
        try {
          const ev = JSON.parse(evt.data);
          onEvent(ev);
        } catch (err) {
          console.error("Bad event payload", err, evt.data);
        }
      };

      ws.onclose = () => {
        clearHeartbeat();
        ws = null;
        if (disposed) {
          return;
        }
        ui.setWsStatus("offline");
        ui.setDiagnostics({ latencyMs: undefined });
        const delay = scheduleReconnect(reconnectBackoff);
        const seconds = Math.max(1, Math.round(delay / 1000));
        ui.updateConnectionMeta(
          `Déconnecté. Nouvelle tentative dans ${seconds} s…`,
          "warning",
        );
        ui.setComposerStatus(
          "Connexion perdue. Reconnexion automatique…",
          "warning",
        );
        ui.scheduleComposerIdle(6000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        if (disposed) {
          return;
        }
        ui.setWsStatus("error", "Erreur WebSocket");
        ui.updateConnectionMeta("Erreur WebSocket détectée.", "danger");
        ui.setComposerStatus("Une erreur réseau est survenue.", "danger");
        ui.scheduleComposerIdle(6000);
      };
    } catch (err) {
      console.error(err);
      if (disposed) {
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      ui.setWsStatus("error", message);
      ui.updateConnectionMeta(message, "danger");
      ui.setComposerStatus(
        "Connexion indisponible. Nouvel essai bientôt.",
        "danger",
      );
      ui.scheduleComposerIdle(6000);
      scheduleReconnect(reconnectBackoff);
    }
  }

  function dispose() {
    disposed = true;
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    clearHeartbeat();
    if (ws) {
      try {
        ws.close();
      } catch (err) {
        console.warn("WebSocket close during dispose failed", err);
      }
      ws = null;
    }
  }

  return {
    open: openSocket,
    send: safeSend,
    dispose,
  };
}
