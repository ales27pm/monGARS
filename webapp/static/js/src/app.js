import { resolveConfig } from "./config.js";
import { createTimelineStore } from "./state/timelineStore.js";
import { createChatUi } from "./ui/chatUi.js";
import { createAuthService } from "./services/auth.js";
import { createHttpService } from "./services/http.js";
import { createExporter } from "./services/exporter.js";
import { createSocketClient } from "./services/socket.js";
import { createSuggestionService } from "./services/suggestions.js";
import { createSpeechService } from "./services/speech.js";
import { nowISO } from "./utils/time.js";

function queryElements(doc) {
  const byId = (id) => doc.getElementById(id);
  return {
    transcript: byId("transcript"),
    composer: byId("composer"),
    prompt: byId("prompt"),
    send: byId("send"),
    modeSelect: byId("chat-mode"),
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
    voiceControls: byId("voice-controls"),
    voiceRecognitionGroup: byId("voice-recognition-group"),
    voiceSynthesisGroup: byId("voice-synthesis-group"),
    voiceToggle: byId("voice-toggle"),
    voiceStatus: byId("voice-status"),
    voiceTranscript: byId("voice-transcript"),
    voiceAutoSend: byId("voice-auto-send"),
    voicePlayback: byId("voice-playback"),
    voiceStopPlayback: byId("voice-stop-playback"),
    voiceVoiceSelect: byId("voice-voice-select"),
    voiceSpeakingIndicator: byId("voice-speaking-indicator"),
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
    this.mode = this.ui.mode || "chat";
    this.auth = createAuthService(this.config);
    this.http = createHttpService({ config: this.config, auth: this.auth });
    this.embeddingAvailable = Boolean(this.config.embedServiceUrl);
    this.embedOptionLabel = null;
    this.configureModeAvailability();
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

    this.setupVoiceFeatures();

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

  configureModeAvailability() {
    const select = this.elements.modeSelect;
    if (!select) {
      return;
    }
    const option = select.querySelector('option[value="embed"]');
    if (!option) {
      return;
    }
    if (!this.embedOptionLabel) {
      this.embedOptionLabel = option.textContent.trim() || "Embedding";
    }
    if (this.embeddingAvailable) {
      option.disabled = false;
      option.removeAttribute("aria-disabled");
      option.textContent = this.embedOptionLabel;
    } else {
      option.disabled = true;
      option.setAttribute("aria-disabled", "true");
      option.textContent = `${this.embedOptionLabel} (indisponible)`;
      if (select.value === "embed") {
        select.value = "chat";
      }
      if (this.ui && typeof this.ui.setMode === "function") {
        this.ui.setMode("chat", { forceStatus: true });
      }
      this.mode = "chat";
    }
  }

  registerUiHandlers() {
    this.ui.on("submit", async ({ text }) => {
      const value = (text || "").trim();
      const requestMode = this.mode === "embed" ? "embed" : "chat";
      if (!value) {
        this.ui.setComposerStatus(
          "Saisissez un message avant d’envoyer.",
          "warning",
        );
        this.ui.scheduleComposerIdle(4000);
        return;
      }
      if (requestMode === "embed" && !this.embeddingAvailable) {
        this.ui.setMode("chat", { forceStatus: true });
        if (this.elements.modeSelect) {
          this.elements.modeSelect.value = "chat";
        }
        this.mode = "chat";
        this.ui.setComposerStatus(
          "Service d'embedding indisponible. Mode Chat rétabli.",
          "warning",
        );
        this.ui.scheduleComposerIdle(5000);
        return;
      }
      this.ui.hideError();
      const submittedAt = nowISO();
      this.ui.appendMessage("user", value, {
        timestamp: submittedAt,
        metadata: { submitted: true, mode: requestMode },
      });
      if (this.elements.prompt) {
        this.elements.prompt.value = "";
      }
      this.ui.updatePromptMetrics();
      this.ui.autosizePrompt();
      if (requestMode === "embed") {
        this.ui.setComposerStatus("Calcul de l'embedding…", "info");
      } else {
        this.ui.setComposerStatus("Message envoyé…", "info");
      }
      this.ui.scheduleComposerIdle(4000);
      this.ui.setBusy(true);
      if (requestMode === "chat") {
        this.ui.applyQuickActionOrdering(["code", "summarize", "explain"]);
      }

      try {
        if (requestMode === "embed") {
          const response = await this.http.postEmbed(value);
          if (this.elements.prompt) {
            this.elements.prompt.focus();
          }
          this.ui.setBusy(false);
          this.presentEmbeddingResult(response);
          this.ui.setComposerStatus("Vecteur généré.", "success");
          this.ui.scheduleComposerIdle(4000);
        } else {
          await this.http.postChat(value);
          if (this.elements.prompt) {
            this.elements.prompt.focus();
          }
          this.ui.startStream();
        }
      } catch (err) {
        this.ui.setBusy(false);
        this.ui.showError(err, {
          metadata: { stage: "submit", mode: requestMode },
        });
        if (requestMode === "embed") {
          this.ui.setComposerStatus(
            "Génération d'embedding impossible. Vérifiez la connexion.",
            "danger",
          );
        } else {
          this.ui.setComposerStatus(
            "Envoi impossible. Vérifiez la connexion.",
            "danger",
          );
        }
        this.ui.scheduleComposerIdle(6000);
      }
    });

    this.ui.on("mode-change", ({ mode }) => {
      const requestedMode = mode === "embed" ? "embed" : "chat";
      if (requestedMode === "embed" && !this.embeddingAvailable) {
        this.configureModeAvailability();
        this.ui.setComposerStatus(
          "Service d'embedding indisponible. Mode Chat rétabli.",
          "warning",
        );
        this.ui.scheduleComposerIdle(5000);
        return;
      }
      if (this.mode === requestedMode) {
        return;
      }
      this.mode = requestedMode;
      this.ui.setMode(requestedMode);
      if (requestedMode === "embed") {
        this.ui.setComposerStatus(
          "Mode Embedding activé. Les requêtes renvoient des vecteurs.",
          "info",
        );
        this.ui.scheduleComposerIdle(5000);
      } else {
        this.ui.setComposerStatus(
          "Mode Chat activé. Les réponses seront générées par le LLM.",
          "info",
        );
        this.ui.scheduleComposerIdle(4000);
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

    this.ui.on("voice-toggle", () => {
      this.toggleVoiceListening().catch((err) => {
        console.error("Voice toggle failed", err);
      });
    });

    this.ui.on("voice-autosend-change", ({ enabled }) => {
      this.handleVoiceAutoSendChange(Boolean(enabled));
    });

    this.ui.on("voice-playback-change", ({ enabled }) => {
      this.handleVoicePlaybackChange(Boolean(enabled));
    });

    this.ui.on("voice-stop-playback", () => {
      this.stopVoicePlayback();
    });

    this.ui.on("voice-voice-change", ({ voiceURI }) => {
      this.handleVoiceVoiceChange(voiceURI || null);
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

  loadVoicePreferences(defaultLanguage) {
    const fallback = {
      autoSend: true,
      playback: true,
      voiceURI: null,
      language: defaultLanguage,
    };
    try {
      const raw = window.localStorage.getItem("chat.voice");
      if (!raw) {
        return fallback;
      }
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") {
        return fallback;
      }
      return {
        autoSend:
          typeof parsed.autoSend === "boolean"
            ? parsed.autoSend
            : fallback.autoSend,
        playback:
          typeof parsed.playback === "boolean"
            ? parsed.playback
            : fallback.playback,
        voiceURI:
          typeof parsed.voiceURI === "string" && parsed.voiceURI.length > 0
            ? parsed.voiceURI
            : null,
        language:
          typeof parsed.language === "string" && parsed.language
            ? parsed.language
            : fallback.language,
      };
    } catch (err) {
      console.warn("Unable to read voice preferences", err);
      return fallback;
    }
  }

  persistVoicePreferences() {
    if (!this.voicePrefs) {
      return;
    }
    try {
      window.localStorage.setItem(
        "chat.voice",
        JSON.stringify({
          autoSend: Boolean(this.voicePrefs.autoSend),
          playback: Boolean(this.voicePrefs.playback),
          voiceURI: this.voicePrefs.voiceURI || null,
          language: this.voicePrefs.language || null,
        }),
      );
    } catch (err) {
      console.warn("Unable to persist voice preferences", err);
    }
  }

  setupVoiceFeatures() {
    const docLang = (
      this.doc?.documentElement?.getAttribute("lang") || ""
    ).trim();
    const navigatorLang =
      typeof navigator !== "undefined" && navigator.language
        ? navigator.language
        : null;
    const defaultLanguage = docLang || navigatorLang || "fr-CA";
    this.voicePrefs = this.loadVoicePreferences(defaultLanguage);
    if (!this.voicePrefs.language) {
      this.voicePrefs.language = defaultLanguage;
      this.persistVoicePreferences();
    }
    this.voiceState = {
      enabled: false,
      listening: false,
      awaitingResponse: false,
      manualStop: false,
      restartTimer: null,
      lastTranscript: "",
    };
    this.speech = createSpeechService({
      defaultLanguage: this.voicePrefs.language,
    });
    if (this.voicePrefs.voiceURI) {
      this.speech.setPreferredVoice(this.voicePrefs.voiceURI);
    }
    if (this.voicePrefs.language) {
      this.speech.setLanguage(this.voicePrefs.language);
    }
    const recognitionSupported = this.speech.isRecognitionSupported();
    const synthesisSupported = this.speech.isSynthesisSupported();
    this.ui.setVoiceAvailability({
      recognition: recognitionSupported,
      synthesis: synthesisSupported,
    });
    this.ui.setVoicePreferences(this.voicePrefs);
    if (recognitionSupported) {
      this.ui.setVoiceStatus(
        "Activez le micro pour dicter votre message.",
        "muted",
      );
    } else if (synthesisSupported) {
      this.ui.setVoiceStatus(
        "Lecture vocale disponible. La dictée nécessite un navigateur compatible.",
        "warning",
      );
    } else {
      this.ui.setVoiceStatus(
        "Les fonctionnalités vocales ne sont pas disponibles dans ce navigateur.",
        "danger",
      );
    }
    this.ui.scheduleVoiceStatusIdle(recognitionSupported ? 5000 : 7000);
    this.speech.on("listening-change", (payload) =>
      this.handleVoiceListeningChange(payload),
    );
    this.speech.on("transcript", (payload) =>
      this.handleVoiceTranscript(payload),
    );
    this.speech.on("error", (payload) => this.handleVoiceError(payload));
    this.speech.on("speaking-change", (payload) =>
      this.handleVoiceSpeakingChange(payload),
    );
    this.speech.on("voices", ({ voices }) =>
      this.handleVoiceVoices(Array.isArray(voices) ? voices : []),
    );
  }

  async toggleVoiceListening() {
    if (!this.speech || !this.speech.isRecognitionSupported()) {
      this.ui.setVoiceStatus(
        "La dictée vocale n'est pas disponible dans ce navigateur.",
        "danger",
      );
      return;
    }
    if (this.voiceState.listening || this.voiceState.awaitingResponse) {
      this.voiceState.enabled = false;
      this.voiceState.manualStop = true;
      this.voiceState.awaitingResponse = false;
      if (this.voiceState.restartTimer) {
        window.clearTimeout(this.voiceState.restartTimer);
        this.voiceState.restartTimer = null;
      }
      this.speech.stopListening();
      this.ui.setVoiceStatus("Dictée interrompue.", "muted");
      this.ui.scheduleVoiceStatusIdle(3500);
      return;
    }
    this.voiceState.manualStop = false;
    this.voiceState.enabled = true;
    this.voiceState.awaitingResponse = false;
    if (this.voiceState.restartTimer) {
      window.clearTimeout(this.voiceState.restartTimer);
      this.voiceState.restartTimer = null;
    }
    const started = await this.speech.startListening({
      language: this.voicePrefs.language,
      interimResults: true,
      continuous: false,
    });
    if (!started) {
      this.voiceState.enabled = false;
      this.ui.setVoiceStatus(
        "Impossible de démarrer la dictée. Vérifiez le micro.",
        "danger",
      );
    }
  }

  handleVoiceListeningChange(payload = {}) {
    const listening = Boolean(payload.listening);
    this.voiceState.listening = listening;
    if (listening) {
      this.ui.setVoiceListening(true);
      this.ui.setVoiceTranscript("", { state: "idle" });
      this.ui.setVoiceStatus(
        "En écoute… Parlez lorsque vous êtes prêt.",
        "info",
      );
      return;
    }
    this.ui.setVoiceListening(false);
    if (payload.reason === "manual") {
      this.voiceState.manualStop = false;
      this.voiceState.enabled = false;
      this.ui.setVoiceStatus("Dictée interrompue.", "muted");
      this.ui.scheduleVoiceStatusIdle(3500);
      return;
    }
    if (payload.reason === "error") {
      this.voiceState.enabled = false;
      this.voiceState.awaitingResponse = false;
      const message =
        payload.code === "not-allowed"
          ? "Autorisez l'accès au microphone pour continuer."
          : "La dictée vocale s'est interrompue. Réessayez.";
      const tone = payload.code === "not-allowed" ? "danger" : "warning";
      this.ui.setVoiceStatus(message, tone);
      return;
    }
    if (!this.voicePrefs.autoSend) {
      this.voiceState.enabled = false;
      this.ui.scheduleVoiceStatusIdle(3500);
      return;
    }
    if (this.voiceState.enabled && !this.voiceState.awaitingResponse) {
      this.maybeRestartVoiceListening(650);
    }
  }

  handleVoiceTranscript(payload = {}) {
    const transcript =
      typeof payload.transcript === "string" ? payload.transcript : "";
    const isFinal = Boolean(payload.isFinal);
    const confidence =
      typeof payload.confidence === "number" ? payload.confidence : null;
    if (transcript) {
      this.voiceState.lastTranscript = transcript;
      this.ui.setVoiceTranscript(transcript, {
        state: isFinal ? "final" : "interim",
      });
    }
    if (!isFinal) {
      if (transcript) {
        this.ui.setVoiceStatus("Transcription en cours…", "info");
      }
      return;
    }
    if (!transcript) {
      this.ui.setVoiceStatus("Aucun texte n'a été reconnu.", "warning");
      this.ui.scheduleVoiceStatusIdle(3000);
      this.voiceState.awaitingResponse = false;
      if (!this.voicePrefs.autoSend) {
        this.voiceState.enabled = false;
      }
      return;
    }
    if (this.voicePrefs.autoSend) {
      this.voiceState.awaitingResponse = true;
      const confidencePct =
        confidence !== null
          ? Math.round(Math.max(0, Math.min(1, confidence)) * 100)
          : null;
      if (confidencePct !== null) {
        this.ui.setVoiceStatus(
          `Envoi du message dicté (${confidencePct}% de confiance)…`,
          "info",
        );
      } else {
        this.ui.setVoiceStatus("Envoi du message dicté…", "info");
      }
      this.submitVoicePrompt(transcript);
    } else {
      if (this.elements.prompt) {
        this.elements.prompt.value = transcript;
      }
      this.ui.updatePromptMetrics();
      this.ui.autosizePrompt();
      this.ui.setVoiceStatus("Message dicté. Vérifiez avant l'envoi.", "info");
      this.ui.scheduleVoiceStatusIdle(4500);
      this.voiceState.enabled = false;
    }
  }

  handleVoiceError(payload = {}) {
    const message =
      typeof payload.message === "string" && payload.message.length > 0
        ? payload.message
        : "Une erreur vocale est survenue.";
    this.ui.setVoiceStatus(message, "danger");
    this.voiceState.enabled = false;
    this.voiceState.awaitingResponse = false;
    if (this.voiceState.restartTimer) {
      window.clearTimeout(this.voiceState.restartTimer);
      this.voiceState.restartTimer = null;
    }
    this.ui.scheduleVoiceStatusIdle(6000);
  }

  handleVoiceSpeakingChange(payload = {}) {
    const speaking = Boolean(payload.speaking);
    this.ui.setVoiceSpeaking(speaking);
    if (speaking) {
      this.ui.setVoiceStatus("Lecture de la réponse…", "info");
      return;
    }
    if (
      this.voicePrefs.autoSend &&
      this.voiceState.enabled &&
      !this.voiceState.awaitingResponse
    ) {
      this.maybeRestartVoiceListening(800);
    }
    this.ui.scheduleVoiceStatusIdle(3500);
  }

  handleVoiceVoices(voices = []) {
    if (!Array.isArray(voices)) {
      return;
    }
    let selectedUri = this.voicePrefs.voiceURI;
    if (!selectedUri && voices.length > 0) {
      const preferred = voices.find((voice) => {
        if (!voice || !voice.lang) {
          return false;
        }
        const lang = String(voice.lang).toLowerCase();
        const target = (this.voicePrefs.language || "").toLowerCase();
        return target && lang.startsWith(target.slice(0, 2));
      });
      if (preferred) {
        selectedUri = preferred.voiceURI || null;
        this.voicePrefs.voiceURI = selectedUri;
        this.persistVoicePreferences();
      }
    }
    this.ui.setVoiceVoiceOptions(voices, selectedUri || null);
    if (selectedUri) {
      this.speech.setPreferredVoice(selectedUri);
    }
  }

  handleVoiceAutoSendChange(enabled) {
    if (!this.voicePrefs) {
      return;
    }
    this.voicePrefs.autoSend = Boolean(enabled);
    this.persistVoicePreferences();
    if (!this.voicePrefs.autoSend) {
      this.voiceState.enabled = false;
      if (this.voiceState.listening) {
        this.speech.stopListening();
      }
      this.ui.setVoiceStatus(
        "Mode manuel activé. Utilisez le micro pour remplir le champ.",
        "muted",
      );
      this.ui.scheduleVoiceStatusIdle(4000);
    } else {
      this.ui.setVoiceStatus(
        "Les messages dictés seront envoyés automatiquement.",
        "info",
      );
      this.ui.scheduleVoiceStatusIdle(3500);
    }
  }

  handleVoicePlaybackChange(enabled) {
    if (!this.voicePrefs) {
      return;
    }
    const next = Boolean(enabled);
    this.voicePrefs.playback = next;
    this.persistVoicePreferences();
    if (!next) {
      this.stopVoicePlayback();
      this.ui.setVoiceStatus("Lecture vocale désactivée.", "muted");
    } else {
      this.ui.setVoiceStatus("Lecture vocale activée.", "info");
    }
    this.ui.scheduleVoiceStatusIdle(3500);
  }

  handleVoiceVoiceChange(voiceURI) {
    if (!this.voicePrefs) {
      return;
    }
    const value = voiceURI && voiceURI.length > 0 ? voiceURI : null;
    this.voicePrefs.voiceURI = value;
    this.speech.setPreferredVoice(value);
    this.persistVoicePreferences();
    if (value) {
      this.ui.setVoiceStatus("Voix sélectionnée mise à jour.", "success");
    } else {
      this.ui.setVoiceStatus("Voix par défaut du système utilisée.", "muted");
    }
    this.ui.scheduleVoiceStatusIdle(3000);
  }

  stopVoicePlayback() {
    if (!this.speech || !this.speech.isSynthesisSupported()) {
      return;
    }
    this.speech.stopSpeaking();
    this.ui.setVoiceSpeaking(false);
    this.ui.setVoiceStatus("Lecture vocale interrompue.", "muted");
    this.ui.scheduleVoiceStatusIdle(3000);
  }

  maybeRestartVoiceListening(delay = 650) {
    if (!this.speech || !this.speech.isRecognitionSupported()) {
      return;
    }
    if (!this.voicePrefs.autoSend || !this.voiceState.enabled) {
      return;
    }
    if (this.voiceState.listening || this.voiceState.awaitingResponse) {
      return;
    }
    if (this.voiceState.restartTimer) {
      window.clearTimeout(this.voiceState.restartTimer);
    }
    this.voiceState.restartTimer = window.setTimeout(() => {
      this.voiceState.restartTimer = null;
      if (!this.voicePrefs.autoSend || !this.voiceState.enabled) {
        return;
      }
      if (this.voiceState.listening || this.voiceState.awaitingResponse) {
        return;
      }
      const attempt = this.speech.startListening({
        language: this.voicePrefs.language,
        interimResults: true,
        continuous: false,
      });
      Promise.resolve(attempt)
        .then((started) => {
          if (started) {
            return;
          }
          this.voiceState.enabled = false;
          this.ui.setVoiceStatus(
            "Impossible de relancer la dictée vocale.",
            "danger",
          );
        })
        .catch((err) => {
          this.voiceState.enabled = false;
          console.error("Automatic voice restart failed", err);
          this.ui.setVoiceStatus(
            "Impossible de relancer la dictée vocale.",
            "danger",
          );
        });
    }, delay);
  }

  submitVoicePrompt(text) {
    if (this.elements.prompt) {
      this.elements.prompt.value = text;
    }
    this.ui.updatePromptMetrics();
    this.ui.autosizePrompt();
    this.ui.emit("submit", { text });
  }

  getLatestAssistantText() {
    if (!this.timelineStore || !this.timelineStore.order) {
      return "";
    }
    for (let i = this.timelineStore.order.length - 1; i >= 0; i -= 1) {
      const id = this.timelineStore.order[i];
      const entry = this.timelineStore.map.get(id);
      if (entry && entry.role === "assistant" && entry.text) {
        return entry.text;
      }
    }
    return "";
  }

  formatEmbeddingResponse(result) {
    const safeInline = (value) => {
      if (value === null || typeof value === "undefined" || value === "") {
        return "—";
      }
      return `\`${String(value).replace(/`/g, "\\`")}\``;
    };
    const vectors = Array.isArray(result?.vectors) ? result.vectors : [];
    const dimsCandidate =
      typeof result?.dims === "number" ? result.dims : Number(result?.dims);
    const dims = Number.isFinite(dimsCandidate)
      ? Number(dimsCandidate)
      : Array.isArray(vectors[0])
        ? vectors[0].length
        : 0;
    const countCandidate =
      typeof result?.count === "number" ? result.count : Number(result?.count);
    const count = Number.isFinite(countCandidate)
      ? Number(countCandidate)
      : vectors.length;
    const normalised = Boolean(result?.normalised);
    const summaryLines = [
      `- **Backend :** ${safeInline(result?.backend ?? "inconnu")}`,
      `- **Modèle :** ${safeInline(result?.model ?? "inconnu")}`,
      `- **Dimensions :** ${dims || 0}`,
      `- **Vecteurs générés :** ${count}`,
      `- **Normalisation appliquée :** ${normalised ? "Oui" : "Non"}`,
    ];

    const vectorSections = [];
    vectors.forEach((vector, index) => {
      if (!Array.isArray(vector)) {
        return;
      }
      const previewLength = Math.min(12, vector.length);
      const previewValues = vector.slice(0, previewLength).map((value) => {
        const numeric = typeof value === "number" ? value : Number(value);
        if (Number.isFinite(numeric)) {
          return Number.parseFloat(numeric.toFixed(6));
        }
        return value;
      });
      const previewJson = JSON.stringify(previewValues, null, 2);
      let section = [
        `**${vectors.length > 1 ? `Vecteur ${index + 1}` : "Vecteur"}**`,
        "```json",
        `${previewJson}${vector.length > previewLength ? "\n// …" : ""}`,
        "```",
      ].join("\n");
      if (vector.length > previewLength) {
        const fullVector = vector.map((value) => {
          const numeric = typeof value === "number" ? value : Number(value);
          return Number.isFinite(numeric) ? numeric : value;
        });
        section += `\n\n<details><summary>Vecteur complet ${index + 1}</summary>\n\n\`\`\`json\n${JSON.stringify(
          fullVector,
          null,
          2,
        )}\n\`\`\`\n\n</details>`;
      }
      vectorSections.push(section);
    });

    const sections = ["### Résultat d'embedding", summaryLines.join("\n")];
    if (vectorSections.length > 0) {
      sections.push(vectorSections.join("\n\n"));
    } else {
      sections.push("**Aucune composante d'embedding n'a été renvoyée.**");
    }
    return sections.join("\n\n");
  }

  presentEmbeddingResult(result) {
    const vectors = Array.isArray(result?.vectors) ? result.vectors : [];
    const dimsCandidate =
      typeof result?.dims === "number" ? result.dims : Number(result?.dims);
    const dims = Number.isFinite(dimsCandidate)
      ? Number(dimsCandidate)
      : Array.isArray(vectors[0])
        ? vectors[0].length
        : 0;
    const countCandidate =
      typeof result?.count === "number" ? result.count : Number(result?.count);
    const count = Number.isFinite(countCandidate)
      ? Number(countCandidate)
      : vectors.length;
    const normalised = Boolean(result?.normalised);
    const metaBits = ["Embedding"];
    if (dims) {
      metaBits.push(`${dims} dims`);
    }
    if (count) {
      metaBits.push(`${count} vecteur${count > 1 ? "s" : ""}`);
    }
    if (normalised) {
      metaBits.push("Normalisé");
    }
    const message = this.formatEmbeddingResponse(result);
    this.ui.appendMessage("assistant", message, {
      timestamp: nowISO(),
      metaSuffix: metaBits.join(" • "),
      metadata: {
        mode: "embed",
        dims,
        backend:
          typeof result?.backend === "string" && result.backend
            ? result.backend
            : null,
        model:
          typeof result?.model === "string" && result.model
            ? result.model
            : null,
        count,
        normalised,
      },
      embeddingData: {
        backend:
          typeof result?.backend === "string" && result.backend
            ? result.backend
            : null,
        model:
          typeof result?.model === "string" && result.model
            ? result.model
            : null,
        dims,
        count,
        normalised,
        vectors,
        raw: result,
      },
    });
  }

  handleVoiceAssistantCompletion() {
    if (!this.voicePrefs) {
      return;
    }
    const latest = this.getLatestAssistantText();
    this.voiceState.awaitingResponse = false;
    if (!latest) {
      this.ui.scheduleVoiceStatusIdle(3500);
      this.maybeRestartVoiceListening(800);
      return;
    }
    if (
      this.voicePrefs.playback &&
      this.speech &&
      this.speech.isSynthesisSupported()
    ) {
      this.ui.setVoiceStatus("Lecture de la réponse…", "info");
      const utterance = this.speech.speak(latest, {
        lang: this.voicePrefs.language,
        voiceURI: this.voicePrefs.voiceURI,
      });
      if (!utterance) {
        this.ui.scheduleVoiceStatusIdle(3500);
        this.maybeRestartVoiceListening(800);
      }
    } else {
      this.ui.scheduleVoiceStatusIdle(3500);
      this.maybeRestartVoiceListening(800);
    }
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
        this.handleVoiceAssistantCompletion();
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
        this.handleVoiceAssistantCompletion();
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
