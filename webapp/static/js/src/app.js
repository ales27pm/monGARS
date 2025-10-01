import { resolveConfig } from "./config.js";
import { createTimelineStore } from "./state/timelineStore.js";
import { createChatUi } from "./ui/chatUi.js";
import { createAuthService } from "./services/auth.js";
import { createHttpService } from "./services/http.js";
import { createExporter } from "./services/exporter.js";
import { createSocketClient } from "./services/socket.js";
import { createSuggestionService } from "./services/suggestions.js";
import { nowISO } from "./utils/time.js";

function queryElements(doc) {
  const byId = (id) => doc.getElementById(id);
  return {
    transcript: byId("transcript"),
    composer: byId("composer"),
    prompt: byId("prompt"),
    send: byId("send"),
    wsStatus: byId("ws-status"),
    quickActions: byId("quick-actions"),
    connection: byId("connection"),
    errorAlert: byId("error-alert"),
    errorMessage: byId("error-message"),
    scrollBottom: byId("scroll-bottom"),
    composerStatus: byId("composer-status"),
    promptCount: byId("prompt-count"),
    connectionMeta: byId("connection-meta"),
    filterInput: byId("chat-search"),
    filterClear: byId("chat-search-clear"),
    filterEmpty: byId("filter-empty"),
    filterHint: byId("chat-search-hint"),
    exportJson: byId("export-json"),
    exportMarkdown: byId("export-markdown"),
    exportCopy: byId("export-copy"),
    diagConnected: byId("diag-connected"),
    diagLastMessage: byId("diag-last-message"),
    diagLatency: byId("diag-latency"),
    diagNetwork: byId("diag-network"),
  };
}

function readHistory(doc) {
  const historyElement = doc.getElementById("chat-history");
  if (!historyElement) {
    return [];
  }
  const payload = historyElement.textContent || "null";
  historyElement.remove();
  try {
    const parsed = JSON.parse(payload);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (parsed && parsed.error) {
      return { error: parsed.error };
    }
  } catch (err) {
    console.error("Unable to parse chat history", err);
  }
  return [];
}

function ensureElements(elements) {
  return Boolean(elements.transcript && elements.composer && elements.prompt);
}

const QUICK_PRESETS = {
  code: "Je souhaite écrire du code…",
  summarize: "Résume la dernière conversation.",
  explain: "Explique ta dernière réponse plus simplement.",
};

export class ChatApp {
  constructor(doc = document, rawConfig = window.chatConfig || {}) {
    this.doc = doc;
    this.config = resolveConfig(rawConfig);
    this.elements = queryElements(doc);
    if (!ensureElements(this.elements)) {
      return;
    }
    if (window.marked && typeof window.marked.setOptions === "function") {
      window.marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false,
      });
    }
    this.timelineStore = createTimelineStore();
    this.ui = createChatUi({
      elements: this.elements,
      timelineStore: this.timelineStore,
    });
    this.auth = createAuthService(this.config);
    this.http = createHttpService({ config: this.config, auth: this.auth });
    this.exporter = createExporter({
      timelineStore: this.timelineStore,
      announce: (message, variant) =>
        this.ui.announceConnection(message, variant),
    });
    this.suggestions = createSuggestionService({
      http: this.http,
      ui: this.ui,
    });
    this.socket = createSocketClient({
      config: this.config,
      http: this.http,
      ui: this.ui,
      onEvent: (ev) => this.handleSocketEvent(ev),
    });

    const historyPayload = readHistory(doc);
    if (historyPayload && historyPayload.error) {
      this.ui.showError(historyPayload.error);
    } else if (Array.isArray(historyPayload)) {
      this.ui.renderHistory(historyPayload);
    }

    this.registerUiHandlers();
    this.ui.initialise();
    this.socket.open();
  }

  registerUiHandlers() {
    this.ui.on("submit", async ({ text }) => {
      const value = (text || "").trim();
      if (!value) {
        this.ui.setComposerStatus(
          "Saisissez un message avant d’envoyer.",
          "warning",
        );
        this.ui.scheduleComposerIdle(4000);
        return;
      }
      this.ui.hideError();
      const submittedAt = nowISO();
      this.ui.appendMessage("user", value, {
        timestamp: submittedAt,
        metadata: { submitted: true },
      });
      if (this.elements.prompt) {
        this.elements.prompt.value = "";
      }
      this.ui.updatePromptMetrics();
      this.ui.autosizePrompt();
      this.ui.setComposerStatus("Message envoyé…", "info");
      this.ui.scheduleComposerIdle(4000);
      this.ui.setBusy(true);
      this.ui.applyQuickActionOrdering(["code", "summarize", "explain"]);

      try {
        await this.http.postChat(value);
        if (this.elements.prompt) {
          this.elements.prompt.focus();
        }
        this.ui.startStream();
      } catch (err) {
        this.ui.setBusy(false);
        const message = err instanceof Error ? err.message : String(err);
        this.ui.showError(message);
        this.ui.appendMessage("system", message, {
          variant: "error",
          allowMarkdown: false,
          metadata: { stage: "submit" },
        });
        this.ui.setComposerStatus(
          "Envoi impossible. Vérifiez la connexion.",
          "danger",
        );
        this.ui.scheduleComposerIdle(6000);
      }
    });

    this.ui.on("quick-action", ({ action }) => {
      if (!action) return;
      const preset = QUICK_PRESETS[action] || action;
      if (this.elements.prompt) {
        this.elements.prompt.value = preset;
      }
      this.ui.updatePromptMetrics();
      this.ui.autosizePrompt();
      this.ui.setComposerStatus("Suggestion envoyée…", "info");
      this.ui.scheduleComposerIdle(4000);
      this.ui.emit("submit", { text: preset });
    });

    this.ui.on("filter-change", ({ value }) => {
      this.ui.applyTranscriptFilter(value, { preserveInput: true });
    });

    this.ui.on("filter-clear", () => {
      this.ui.clearTranscriptFilter();
    });

    this.ui.on("export", ({ format }) => {
      this.exporter.exportConversation(format);
    });

    this.ui.on("export-copy", () => {
      this.exporter.copyConversationToClipboard();
    });

    this.ui.on("prompt-input", ({ value }) => {
      if (!value || !value.trim()) {
        return;
      }
      if (this.elements.send && this.elements.send.disabled) {
        return;
      }
      this.suggestions.schedule(value);
    });
  }

  handleSocketEvent(ev) {
    const type = ev && ev.type ? ev.type : "";
    const data = ev && ev.data ? ev.data : {};
    switch (type) {
      case "ws.connected": {
        if (data && data.origin) {
          this.ui.announceConnection(`Connecté via ${data.origin}`);
          this.ui.updateConnectionMeta(
            `Connecté via ${data.origin}`,
            "success",
          );
        } else {
          this.ui.announceConnection("Connecté au serveur.");
          this.ui.updateConnectionMeta("Connecté au serveur.", "success");
        }
        this.ui.scheduleComposerIdle(4000);
        break;
      }
      case "history.snapshot": {
        if (data && Array.isArray(data.items)) {
          this.ui.renderHistory(data.items, { replace: true });
        }
        break;
      }
      case "ai_model.response_chunk": {
        const delta =
          typeof data.delta === "string" ? data.delta : data.text || "";
        this.ui.appendStream(delta);
        break;
      }
      case "ai_model.response_complete": {
        if (data && data.text && !this.ui.hasStreamBuffer()) {
          this.ui.appendStream(data.text);
        }
        this.ui.endStream(data);
        this.ui.setBusy(false);
        if (data && typeof data.latency_ms !== "undefined") {
          this.ui.setDiagnostics({ latencyMs: Number(data.latency_ms) });
        }
        if (data && data.ok === false && data.error) {
          this.ui.appendMessage("system", data.error, {
            variant: "error",
            allowMarkdown: false,
            metadata: { event: type },
          });
        }
        break;
      }
      case "chat.message": {
        if (!this.ui.isStreaming()) {
          this.ui.startStream();
        }
        if (
          data &&
          typeof data.response === "string" &&
          !this.ui.hasStreamBuffer()
        ) {
          this.ui.appendStream(data.response);
        }
        this.ui.endStream(data);
        this.ui.setBusy(false);
        break;
      }
      case "evolution_engine.training_complete": {
        this.ui.appendMessage(
          "system",
          `Évolution mise à jour ${data && data.version ? data.version : ""}`,
          {
            variant: "ok",
            allowMarkdown: false,
            metadata: { event: type },
          },
        );
        break;
      }
      case "evolution_engine.training_failed": {
        this.ui.appendMessage(
          "system",
          `Échec de l'évolution : ${data && data.error ? data.error : "inconnu"}`,
          {
            variant: "error",
            allowMarkdown: false,
            metadata: { event: type },
          },
        );
        break;
      }
      case "sleep_time_compute.phase_start": {
        this.ui.appendMessage(
          "system",
          "Optimisation en arrière-plan démarrée…",
          {
            variant: "hint",
            allowMarkdown: false,
            metadata: { event: type },
          },
        );
        break;
      }
      case "sleep_time_compute.creative_phase": {
        this.ui.appendMessage(
          "system",
          `Exploration de ${Number(data && data.ideas ? data.ideas : 1)} idées…`,
          {
            variant: "hint",
            allowMarkdown: false,
            metadata: { event: type },
          },
        );
        break;
      }
      case "performance.alert": {
        this.ui.appendMessage("system", `Perf : ${this.ui.formatPerf(data)}`, {
          variant: "warn",
          allowMarkdown: false,
          metadata: { event: type },
        });
        if (data && typeof data.ttfb_ms !== "undefined") {
          this.ui.setDiagnostics({ latencyMs: Number(data.ttfb_ms) });
        }
        break;
      }
      case "ui.suggestions": {
        this.ui.applyQuickActionOrdering(
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
}
