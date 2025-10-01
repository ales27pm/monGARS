import { nowISO } from "../utils/time.js";

export function createSocketClient({ config, http, ui, onEvent }) {
  let ws;
  let wsHBeat;
  let reconnectBackoff = 500;
  const BACKOFF_MAX = 8000;

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
    try {
      ui.updateConnectionMeta("Obtention d’un ticket de connexion…", "info");
      const ticket = await http.fetchTicket();
      const wsUrl = new URL("/ws/chat/", config.baseUrl);
      wsUrl.protocol = config.baseUrl.protocol === "https:" ? "wss:" : "ws:";
      wsUrl.searchParams.set("t", ticket);

      ws = new WebSocket(wsUrl.toString());
      ui.setWsStatus("connecting");
      ui.updateConnectionMeta("Connexion au serveur…", "info");

      ws.onopen = () => {
        ui.setWsStatus("online");
        const connectedAt = nowISO();
        ui.updateConnectionMeta(
          `Connecté le ${ui.formatTimestamp(connectedAt)}`,
          "success",
        );
        ui.setDiagnostics({ connectedAt, lastMessageAt: connectedAt });
        ui.hideError();
        wsHBeat = window.setInterval(() => {
          safeSend({ type: "client.ping", ts: nowISO() });
        }, 20000);
        reconnectBackoff = 500;
        ui.setComposerStatus("Connecté. Vous pouvez échanger.", "success");
        ui.scheduleComposerIdle(4000);
      };

      ws.onmessage = (evt) => {
        try {
          const ev = JSON.parse(evt.data);
          onEvent(ev);
        } catch (err) {
          console.error("Bad event payload", err, evt.data);
        }
      };

let ws;
let wsHBeat;
let reconnectBackoff = 500;
const BACKOFF_MAX = 8000;
let retryTimer = null;
let disposed = false;

async function openSocket() {
  if (disposed) {
    return;
  }
  try {
    // … (existing WebSocket initialization)
    ws.onclose = () => {
      if (wsHBeat) {
        clearInterval(wsHBeat);
        wsHBeat = null;
      }
      ws = null;
      if (disposed) {
        return;
      }
      ui.setWsStatus("offline");
      ui.setDiagnostics({ latencyMs: undefined });
      const delay = reconnectBackoff + Math.floor(Math.random() * 250);
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
      reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
      if (retryTimer) {
        clearTimeout(retryTimer);
      }
      retryTimer = window.setTimeout(() => {
        retryTimer = null;
        openSocket();
      }, delay);
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
    // … (rest of openSocket logic)
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
    const delay = Math.min(BACKOFF_MAX, reconnectBackoff);
    reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
    if (retryTimer) {
      clearTimeout(retryTimer);
    }
    retryTimer = window.setTimeout(() => {
      retryTimer = null;
      openSocket();
    }, delay);
  }
}

function dispose() {
  disposed = true;
  if (retryTimer) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  if (wsHBeat) {
    clearInterval(wsHBeat);
    wsHBeat = null;
  }
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
