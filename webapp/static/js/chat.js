(() => {
  // src/config.js
  function resolveConfig(raw = {}) {
    const config = { ...raw };
    const candidate = config.fastapiUrl || window.location.origin;
    try {
      config.baseUrl = new URL(candidate);
    } catch (err) {
      console.error("Invalid FASTAPI URL", err, candidate);
      config.baseUrl = new URL(window.location.origin);
    }
    return config;
  }
  function apiUrl(config, path) {
    return new URL(path, config.baseUrl).toString();
  }

  // src/utils/time.js
  function nowISO() {
    return (/* @__PURE__ */ new Date()).toISOString();
  }
  function formatTimestamp(ts) {
    if (!ts) return "";
    try {
      return new Date(ts).toLocaleString("fr-CA");
    } catch (err) {
      return String(ts);
    }
  }

  // src/utils/errorUtils.js
  var DEFAULT_ERROR_MESSAGE =
    "Une erreur inattendue est survenue. Veuillez réessayer.";
  var ERROR_PREFIX_REGEX = /^\s*[\W_]*\s*(erreur|error)/i;
  function normaliseErrorText(error) {
    if (error instanceof Error) {
      const message = error.message ? error.message.trim() : "";
      if (message) {
        return message;
      }
    }
    if (typeof error === "string") {
      const trimmed = error.trim();
      return trimmed || DEFAULT_ERROR_MESSAGE;
    }
    if (error && typeof error === "object") {
      const candidate =
        (typeof error.message === "string" && error.message.trim()) ||
        (typeof error.error === "string" && error.error.trim());
      if (candidate) {
        return candidate;
      }
      try {
        const serialised = JSON.stringify(error);
        if (serialised && serialised !== "{}") {
          return serialised;
        }
      } catch (serialiseErr) {
        console.debug("Unable to serialise error payload", serialiseErr);
      }
    }
    if (typeof error === "undefined" || error === null) {
      return DEFAULT_ERROR_MESSAGE;
    }
    const fallback = String(error).trim();
    return fallback || DEFAULT_ERROR_MESSAGE;
  }
  function resolveErrorText(errorOrText) {
    if (typeof errorOrText === "string") {
      const trimmed = errorOrText.trim();
      return trimmed || DEFAULT_ERROR_MESSAGE;
    }
    return normaliseErrorText(errorOrText);
  }
  function computeErrorBubbleText(errorOrText, options = {}) {
    const { prefix } = options;
    const text = resolveErrorText(errorOrText);
    const basePrefix =
      options.prefix === null
        ? ""
        : typeof prefix === "string"
          ? prefix
          : "Erreur : ";
    const trimmedPrefix = basePrefix.trim().toLowerCase();
    const shouldPrefix =
      Boolean(basePrefix) &&
      !ERROR_PREFIX_REGEX.test(text) &&
      !(trimmedPrefix && text.toLowerCase().startsWith(trimmedPrefix));
    const bubbleText = shouldPrefix ? `${basePrefix}${text}` : text;
    return { text, bubbleText };
  }

  // src/state/timelineStore.js
  function makeMessageId() {
    return `msg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }
  function createTimelineStore() {
    const order = [];
    const map = /* @__PURE__ */ new Map();
    function register({
      id,
      role,
      text = "",
      timestamp = nowISO(),
      row,
      metadata = {}
    }) {
      const messageId = id || makeMessageId();
      if (!map.has(messageId)) {
        order.push(messageId);
      }
      map.set(messageId, {
        id: messageId,
        role,
        text,
        timestamp,
        row,
        metadata: { ...metadata }
      });
      if (row) {
        row.dataset.messageId = messageId;
        row.dataset.role = role;
        row.dataset.rawText = text;
        row.dataset.timestamp = timestamp;
      }
      return messageId;
    }
    function update(id, patch = {}) {
      if (!map.has(id)) {
        return null;
      }
      const entry = map.get(id);
      const next = { ...entry, ...patch };
      if (patch && typeof patch.metadata === "object" && patch.metadata !== null) {
        const merged = { ...entry.metadata };
        Object.entries(patch.metadata).forEach(([key, value]) => {
          if (value === void 0 || value === null) {
            delete merged[key];
          } else {
            merged[key] = value;
          }
        });
        next.metadata = merged;
      }
      map.set(id, next);
      const { row } = next;
      if (row && row.isConnected) {
        if (next.text !== entry.text) {
          row.dataset.rawText = next.text || "";
        }
        if (next.timestamp !== entry.timestamp) {
          row.dataset.timestamp = next.timestamp || "";
        }
        if (next.role && next.role !== entry.role) {
          row.dataset.role = next.role;
        }
      }
      return next;
    }
    function collect() {
      return order.map((id) => {
        const entry = map.get(id);
        if (!entry) {
          return null;
        }
        return {
          role: entry.role,
          text: entry.text,
          timestamp: entry.timestamp,
          ...entry.metadata && Object.keys(entry.metadata).length > 0 && {
            metadata: { ...entry.metadata }
          }
        };
      }).filter(Boolean);
    }
    function clear() {
      order.length = 0;
      map.clear();
    }
    return {
      register,
      update,
      collect,
      clear,
      order,
      map,
      makeMessageId
    };
  }

  // src/utils/emitter.js
  function createEmitter() {
    const listeners = /* @__PURE__ */ new Map();
    function on(event, handler) {
      if (!listeners.has(event)) {
        listeners.set(event, /* @__PURE__ */ new Set());
      }
      listeners.get(event).add(handler);
      return () => off(event, handler);
    }
    function off(event, handler) {
      if (!listeners.has(event)) return;
      const bucket = listeners.get(event);
      bucket.delete(handler);
      if (bucket.size === 0) {
        listeners.delete(event);
      }
    }
    function emit(event, payload) {
      if (!listeners.has(event)) return;
      listeners.get(event).forEach((handler) => {
        try {
          handler(payload);
        } catch (err) {
          console.error("Emitter handler error", err);
        }
      });
    }
    return { on, off, emit };
  }

  // src/utils/dom.js
  function escapeHTML(str) {
    return String(str).replace(
      /[&<>"']/g,
      (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[ch]
    );
  }
  function htmlToText(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    return doc.body.textContent || "";
  }
  function extractBubbleText(bubble) {
    const clone = bubble.cloneNode(true);
    clone.querySelectorAll(".copy-btn, .chat-meta").forEach((node) => node.remove());
    return clone.textContent.trim();
  }

  // src/services/markdown.js
  function renderMarkdown(text) {
    if (text == null) {
      return "";
    }
    const value = String(text);
    const fallback = () => {
      const escaped = escapeHTML(value);
      return escaped.replace(/\n/g, "<br>");
    };
    try {
      if (window.marked && typeof window.marked.parse === "function") {
        const rendered = window.marked.parse(value);
        if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
          return window.DOMPurify.sanitize(rendered, {
            ALLOW_UNKNOWN_PROTOCOLS: false,
            USE_PROFILES: { html: true }
          });
        }
        const escaped = escapeHTML(value);
        return escaped.replace(/\n/g, "<br>");
      }
    } catch (err) {
      console.warn("Markdown rendering failed", err);
    }
    return fallback();
  }

  // src/ui/chatUi.js
  function createChatUi({ elements, timelineStore }) {
    var _a;
    const emitter = createEmitter();
    const sendIdleMarkup = elements.send ? elements.send.innerHTML : "";
    const sendIdleLabel = elements.send && elements.send.getAttribute("data-idle-label") || (elements.send ? elements.send.textContent.trim() : "Envoyer");
    const sendBusyMarkup = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Envoi\u2026';
    const SUPPORTED_TONES = ["muted", "info", "success", "danger", "warning"];
    const composerStatusDefault = elements.composerStatus && elements.composerStatus.textContent.trim() || "Appuyez sur Ctrl+Entr\xE9e pour envoyer rapidement.";
    const filterHintDefault = elements.filterHint && elements.filterHint.textContent.trim() || "Utilisez le filtre pour limiter l'historique. Appuyez sur \xC9chap pour effacer.";
    const voiceStatusDefault = elements.voiceStatus && elements.voiceStatus.textContent.trim() || "V\xE9rification des capacit\xE9s vocales\u2026";
    const promptMax = Number((_a = elements.prompt) == null ? void 0 : _a.getAttribute("maxlength")) || null;
    const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const SCROLL_THRESHOLD = 140;
    const PROMPT_MAX_HEIGHT = 320;
    const diagnostics = {
      connectedAt: null,
      lastMessageAt: null,
      latencyMs: null
    };
    const state = {
      resetStatusTimer: null,
      hideScrollTimer: null,
      voiceStatusTimer: null,
      activeFilter: "",
      historyBootstrapped: elements.transcript.childElementCount > 0,
      bootstrapping: false,
      streamRow: null,
      streamBuf: "",
      streamMessageId: null
    };
    const statusLabels = {
      offline: "Hors ligne",
      connecting: "Connexion\u2026",
      online: "En ligne",
      error: "Erreur"
    };
    function on(event, handler) {
      return emitter.on(event, handler);
    }
    function emit(event, payload) {
      emitter.emit(event, payload);
    }
    function setBusy(busy) {
      elements.transcript.setAttribute("aria-busy", busy ? "true" : "false");
      if (elements.send) {
        elements.send.disabled = Boolean(busy);
        elements.send.setAttribute("aria-busy", busy ? "true" : "false");
        if (busy) {
          elements.send.innerHTML = sendBusyMarkup;
        } else if (sendIdleMarkup) {
          elements.send.innerHTML = sendIdleMarkup;
        } else {
          elements.send.textContent = sendIdleLabel;
        }
      }
    }
    const hideError = () => {
      if (!elements.errorAlert) return;
      elements.errorAlert.classList.add("d-none");
      if (elements.errorMessage) {
        elements.errorMessage.textContent = "";
      }
    };
    const appendErrorBubble = (error, options = {}) => {
      const {
        metadata = {},
        timestamp = nowISO(),
        role = "system",
        prefix,
        register,
        messageId,
        resolvedText
      } = options;
      const { text, bubbleText } = computeErrorBubbleText(
        typeof resolvedText === "string" ? resolvedText : error,
        { prefix }
      );
      return appendMessage(role, bubbleText, {
        variant: "error",
        allowMarkdown: false,
        timestamp,
        metadata: { ...metadata, error: text },
        register,
        messageId
      });
    };
    const showError = (error, options = {}) => {
      const { text } = computeErrorBubbleText(error, options);
      if (elements.errorAlert && elements.errorMessage) {
        elements.errorMessage.textContent = text;
        elements.errorAlert.classList.remove("d-none");
      }
      if (options.bubble === false) {
        return null;
      }
      const { bubble, ...bubbleOptions } = options;
      return appendErrorBubble(error, { ...bubbleOptions, resolvedText: text });
    };
    const setComposerStatus = (message, tone = "muted") => {
      if (!elements.composerStatus) return;
      elements.composerStatus.textContent = message;
      SUPPORTED_TONES.forEach(
        (t) => elements.composerStatus.classList.remove(`text-${t}`)
      );
      elements.composerStatus.classList.add(`text-${tone}`);
    };
    const setComposerStatusIdle = () => {
      setComposerStatus(composerStatusDefault, "muted");
    };
    const scheduleComposerIdle = (delay = 3500) => {
      if (state.resetStatusTimer) {
        clearTimeout(state.resetStatusTimer);
      }
      state.resetStatusTimer = window.setTimeout(() => {
        setComposerStatusIdle();
      }, delay);
    };
    const setVoiceStatus = (message, tone = "muted") => {
      if (!elements.voiceStatus) return;
      if (state.voiceStatusTimer) {
        clearTimeout(state.voiceStatusTimer);
        state.voiceStatusTimer = null;
      }
      elements.voiceStatus.textContent = message;
      SUPPORTED_TONES.forEach(
        (t) => elements.voiceStatus.classList.remove(`text-${t}`)
      );
      elements.voiceStatus.classList.add(`text-${tone}`);
    };
    const scheduleVoiceStatusIdle = (delay = 4e3) => {
      if (!elements.voiceStatus) return;
      if (state.voiceStatusTimer) {
        clearTimeout(state.voiceStatusTimer);
      }
      state.voiceStatusTimer = window.setTimeout(() => {
        setVoiceStatus(voiceStatusDefault, "muted");
        state.voiceStatusTimer = null;
      }, delay);
    };
    const setVoiceAvailability = (
      { recognition = false, synthesis = false } = {}
    ) => {
      if (elements.voiceControls) {
        elements.voiceControls.classList.toggle(
          "d-none",
          !recognition && !synthesis
        );
      }
      if (elements.voiceRecognitionGroup) {
        elements.voiceRecognitionGroup.classList.toggle(
          "d-none",
          !recognition
        );
      }
      if (elements.voiceToggle) {
        elements.voiceToggle.disabled = !recognition;
        elements.voiceToggle.setAttribute(
          "title",
          recognition ? "Activer ou d\xE9sactiver la dict\xE9e vocale." : "Dict\xE9e vocale indisponible dans ce navigateur."
        );
        elements.voiceToggle.setAttribute("aria-pressed", "false");
        elements.voiceToggle.classList.remove("btn-danger");
        elements.voiceToggle.classList.add("btn-outline-secondary");
        elements.voiceToggle.textContent = "\u{1F399}\uFE0F Activer la dict\xE9e";
      }
      if (elements.voiceAutoSend) {
        elements.voiceAutoSend.disabled = !recognition;
      }
      if (!recognition) {
        setVoiceTranscript("", { state: "idle" });
      }
      if (elements.voiceSynthesisGroup) {
        elements.voiceSynthesisGroup.classList.toggle("d-none", !synthesis);
      }
      if (elements.voicePlayback) {
        elements.voicePlayback.disabled = !synthesis;
      }
      if (elements.voiceStopPlayback) {
        elements.voiceStopPlayback.disabled = !synthesis;
      }
      if (elements.voiceVoiceSelect) {
        elements.voiceVoiceSelect.disabled = !synthesis;
        if (!synthesis) {
          elements.voiceVoiceSelect.innerHTML = "";
        }
      }
    };
    function setVoiceListening(listening) {
      if (!elements.voiceToggle) return;
      elements.voiceToggle.setAttribute("aria-pressed", listening ? "true" : "false");
      elements.voiceToggle.classList.toggle("btn-danger", listening);
      elements.voiceToggle.classList.toggle("btn-outline-secondary", !listening);
      elements.voiceToggle.textContent = listening ? "\u{1F6D1} Arr\xEAter l'\xE9coute" : "\u{1F399}\uFE0F Activer la dict\xE9e";
    }
    function setVoiceTranscript(text, options = {}) {
      if (!elements.voiceTranscript) return;
      const value = text || "";
      const stateValue = options.state || (value ? "final" : "idle");
      elements.voiceTranscript.textContent = value;
      elements.voiceTranscript.dataset.state = stateValue;
      if (!value && options.placeholder) {
        elements.voiceTranscript.textContent = options.placeholder;
      }
    }
    function setVoicePreferences(prefs = {}) {
      if (elements.voiceAutoSend) {
        elements.voiceAutoSend.checked = Boolean(prefs.autoSend);
      }
      if (elements.voicePlayback) {
        elements.voicePlayback.checked = Boolean(prefs.playback);
      }
    }
    function setVoiceSpeaking(active) {
      if (elements.voiceSpeakingIndicator) {
        elements.voiceSpeakingIndicator.classList.toggle("d-none", !active);
      }
      if (elements.voiceStopPlayback) {
        elements.voiceStopPlayback.disabled = !active;
      }
    }
    function setVoiceVoiceOptions(voices = [], selectedUri = null) {
      if (!elements.voiceVoiceSelect) return;
      const select = elements.voiceVoiceSelect;
      const frag = document.createDocumentFragment();
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = voices.length ? "Voix par d\xE9faut du syst\xE8me" : "Aucune voix disponible";
      frag.appendChild(placeholder);
      voices.forEach((voice) => {
        const option = document.createElement("option");
        option.value = voice.voiceURI || voice.name || "";
        const bits = [voice.name || voice.voiceURI || "Voix"];
        if (voice.lang) {
          bits.push(`(${voice.lang})`);
        }
        if (voice.default) {
          bits.push("\u2022 d\xE9faut");
        }
        option.textContent = bits.join(" ");
        frag.appendChild(option);
      });
      select.innerHTML = "";
      select.appendChild(frag);
      if (selectedUri) {
        let matched = false;
        Array.from(select.options).forEach((option) => {
          if (!matched && option.value === selectedUri) {
            matched = true;
          }
        });
        select.value = matched ? selectedUri : "";
      } else {
        select.value = "";
      }
    }
    function updatePromptMetrics() {
      if (!elements.promptCount || !elements.prompt) return;
      const value = elements.prompt.value || "";
      if (promptMax) {
        elements.promptCount.textContent = `${value.length} / ${promptMax}`;
      } else {
        elements.promptCount.textContent = `${value.length}`;
      }
      elements.promptCount.classList.remove("text-warning", "text-danger");
      if (promptMax) {
        const remaining = promptMax - value.length;
        if (remaining <= 5) {
          elements.promptCount.classList.add("text-danger");
        } else if (remaining <= 20) {
          elements.promptCount.classList.add("text-warning");
        }
      }
    }
    function autosizePrompt() {
      if (!elements.prompt) return;
      elements.prompt.style.height = "auto";
      const nextHeight = Math.min(
        elements.prompt.scrollHeight,
        PROMPT_MAX_HEIGHT
      );
      elements.prompt.style.height = `${nextHeight}px`;
    }
    function isAtBottom() {
      if (!elements.transcript) return true;
      const distance = elements.transcript.scrollHeight - (elements.transcript.scrollTop + elements.transcript.clientHeight);
      return distance <= SCROLL_THRESHOLD;
    }
    function scrollToBottom(options = {}) {
      if (!elements.transcript) return;
      const smooth = options.smooth !== false && !prefersReducedMotion;
      elements.transcript.scrollTo({
        top: elements.transcript.scrollHeight,
        behavior: smooth ? "smooth" : "auto"
      });
      hideScrollButton();
    }
    function showScrollButton() {
      if (!elements.scrollBottom) return;
      if (state.hideScrollTimer) {
        clearTimeout(state.hideScrollTimer);
        state.hideScrollTimer = null;
      }
      elements.scrollBottom.classList.remove("d-none");
      elements.scrollBottom.classList.add("is-visible");
      elements.scrollBottom.setAttribute("aria-hidden", "false");
    }
    function hideScrollButton() {
      if (!elements.scrollBottom) return;
      elements.scrollBottom.classList.remove("is-visible");
      elements.scrollBottom.setAttribute("aria-hidden", "true");
      state.hideScrollTimer = window.setTimeout(() => {
        if (elements.scrollBottom) {
          elements.scrollBottom.classList.add("d-none");
        }
      }, 200);
    }
    async function handleCopy(bubble) {
      const text = extractBubbleText(bubble);
      if (!text) {
        return;
      }
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          textarea.setAttribute("readonly", "readonly");
          textarea.style.position = "absolute";
          textarea.style.left = "-9999px";
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          document.body.removeChild(textarea);
        }
        announceConnection("Contenu copi\xE9 dans le presse-papiers.", "success");
      } catch (err) {
        console.warn("Copy failed", err);
        announceConnection("Impossible de copier le message.", "danger");
      }
    }
    function decorateRow(row, role) {
      const bubble = row.querySelector(".chat-bubble");
      if (!bubble) return;
      if (role === "assistant" || role === "user") {
        bubble.classList.add("has-tools");
        bubble.querySelectorAll(".copy-btn").forEach((btn) => btn.remove());
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "copy-btn";
        copyBtn.innerHTML = '<span aria-hidden="true">\u29C9</span><span class="visually-hidden">Copier le message</span>';
        copyBtn.addEventListener("click", () => handleCopy(bubble));
        bubble.appendChild(copyBtn);
      }
    }
    function highlightRow(row, role) {
      if (!row || state.bootstrapping || role === "system") {
        return;
      }
      row.classList.add("chat-row-highlight");
      window.setTimeout(() => {
        row.classList.remove("chat-row-highlight");
      }, 600);
    }
    function line(role, html, options = {}) {
      const shouldStick = isAtBottom();
      const row = document.createElement("div");
      row.className = `chat-row chat-${role}`;
      row.innerHTML = html;
      row.dataset.role = role;
      row.dataset.rawText = options.rawText || "";
      row.dataset.timestamp = options.timestamp || "";
      elements.transcript.appendChild(row);
      decorateRow(row, role);
      if (options.register !== false) {
        const ts = options.timestamp || nowISO();
        const text = options.rawText && options.rawText.length > 0 ? options.rawText : htmlToText(html);
        const id = timelineStore.register({
          id: options.messageId,
          role,
          text,
          timestamp: ts,
          row,
          metadata: options.metadata || {}
        });
        row.dataset.messageId = id;
      } else if (options.messageId) {
        row.dataset.messageId = options.messageId;
      } else if (!row.dataset.messageId) {
        row.dataset.messageId = timelineStore.makeMessageId();
      }
      if (shouldStick) {
        scrollToBottom({ smooth: !state.bootstrapping });
      } else {
        showScrollButton();
      }
      highlightRow(row, role);
      if (state.activeFilter) {
        applyTranscriptFilter(state.activeFilter, { preserveInput: true });
      }
      return row;
    }
    function buildBubble({
      text,
      timestamp,
      variant,
      metaSuffix,
      allowMarkdown = true
    }) {
      const classes = ["chat-bubble"];
      if (variant) {
        classes.push(`chat-bubble-${variant}`);
      }
      const content = allowMarkdown ? renderMarkdown(text) : escapeHTML(String(text));
      const metaBits = [];
      if (timestamp) {
        metaBits.push(formatTimestamp(timestamp));
      }
      if (metaSuffix) {
        metaBits.push(metaSuffix);
      }
      const metaHtml = metaBits.length > 0 ? `<div class="chat-meta">${escapeHTML(metaBits.join(" \u2022 "))}</div>` : "";
      return `<div class="${classes.join(" ")}">${content}${metaHtml}</div>`;
    }
    function appendMessage(role, text, options = {}) {
      const {
        timestamp,
        variant,
        metaSuffix,
        allowMarkdown = true,
        messageId,
        register = true,
        metadata
      } = options;
      const bubble = buildBubble({
        text,
        timestamp,
        variant,
        metaSuffix,
        allowMarkdown
      });
      const row = line(role, bubble, {
        rawText: text,
        timestamp,
        messageId,
        register,
        metadata
      });
      setDiagnostics({ lastMessageAt: timestamp || nowISO() });
      return row;
    }
    function updateDiagnosticField(el, value) {
      if (!el) return;
      el.textContent = value || "\u2014";
    }
    function setDiagnostics(patch) {
      Object.assign(diagnostics, patch);
      if (Object.prototype.hasOwnProperty.call(patch, "connectedAt")) {
        updateDiagnosticField(
          elements.diagConnected,
          diagnostics.connectedAt ? formatTimestamp(diagnostics.connectedAt) : "\u2014"
        );
      }
      if (Object.prototype.hasOwnProperty.call(patch, "lastMessageAt")) {
        updateDiagnosticField(
          elements.diagLastMessage,
          diagnostics.lastMessageAt ? formatTimestamp(diagnostics.lastMessageAt) : "\u2014"
        );
      }
      if (Object.prototype.hasOwnProperty.call(patch, "latencyMs")) {
        if (typeof diagnostics.latencyMs === "number") {
          updateDiagnosticField(
            elements.diagLatency,
            `${Math.max(0, Math.round(diagnostics.latencyMs))} ms`
          );
        } else {
          updateDiagnosticField(elements.diagLatency, "\u2014");
        }
      }
    }
    function updateNetworkStatus() {
      if (!elements.diagNetwork) return;
      const online = navigator.onLine;
      elements.diagNetwork.textContent = online ? "En ligne" : "Hors ligne";
      elements.diagNetwork.classList.toggle("text-danger", !online);
      elements.diagNetwork.classList.toggle("text-success", online);
    }
    function announceConnection(message, variant = "info") {
      if (!elements.connection) {
        return;
      }
      const classList = elements.connection.classList;
      Array.from(classList).filter((cls) => cls.startsWith("alert-") && cls !== "alert").forEach((cls) => classList.remove(cls));
      classList.add("alert");
      classList.add(`alert-${variant}`);
      elements.connection.textContent = message;
      classList.remove("visually-hidden");
      window.setTimeout(() => {
        classList.add("visually-hidden");
      }, 4e3);
    }
    function updateConnectionMeta(message, tone = "muted") {
      if (!elements.connectionMeta) return;
      const tones = ["muted", "info", "success", "danger", "warning"];
      elements.connectionMeta.textContent = message;
      tones.forEach((t) => elements.connectionMeta.classList.remove(`text-${t}`));
      elements.connectionMeta.classList.add(`text-${tone}`);
    }
    function setWsStatus(state2, title) {
      if (!elements.wsStatus) return;
      const label = statusLabels[state2] || state2;
      elements.wsStatus.textContent = label;
      elements.wsStatus.className = `badge ws-badge ${state2}`;
      if (title) {
        elements.wsStatus.title = title;
      } else {
        elements.wsStatus.removeAttribute("title");
      }
    }
    function normalizeString(str) {
      const value = String(str || "");
      try {
        return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
      } catch (err) {
        return value.toLowerCase();
      }
    }
    function applyTranscriptFilter(query, options = {}) {
      if (!elements.transcript) return 0;
      const { preserveInput = false } = options;
      const rawQuery = typeof query === "string" ? query : "";
      if (!preserveInput && elements.filterInput) {
        elements.filterInput.value = rawQuery;
      }
      const trimmed = rawQuery.trim();
      state.activeFilter = trimmed;
      const normalized = normalizeString(trimmed);
      let matches = 0;
      const rows = Array.from(elements.transcript.querySelectorAll(".chat-row"));
      rows.forEach((row) => {
        row.classList.remove("chat-hidden", "chat-filter-match");
        if (!normalized) {
          return;
        }
        const raw = row.dataset.rawText || "";
        const normalizedRow = normalizeString(raw);
        if (normalizedRow.includes(normalized)) {
          row.classList.add("chat-filter-match");
          matches += 1;
        } else {
          row.classList.add("chat-hidden");
        }
      });
      elements.transcript.classList.toggle("filtered", Boolean(trimmed));
      if (elements.filterEmpty) {
        if (trimmed && matches === 0) {
          elements.filterEmpty.classList.remove("d-none");
          elements.filterEmpty.setAttribute(
            "aria-live",
            elements.filterEmpty.getAttribute("aria-live") || "polite"
          );
        } else {
          elements.filterEmpty.classList.add("d-none");
        }
      }
      if (elements.filterHint) {
        if (trimmed) {
          let summary = "Aucun message ne correspond.";
          if (matches === 1) {
            summary = "1 message correspond.";
          } else if (matches > 1) {
            summary = `${matches} messages correspondent.`;
          }
          elements.filterHint.textContent = summary;
        } else {
          elements.filterHint.textContent = filterHintDefault;
        }
      }
      return matches;
    }
    function reapplyTranscriptFilter() {
      if (state.activeFilter) {
        applyTranscriptFilter(state.activeFilter, { preserveInput: true });
      } else if (elements.transcript) {
        elements.transcript.classList.remove("filtered");
        const rows = Array.from(
          elements.transcript.querySelectorAll(".chat-row")
        );
        rows.forEach((row) => {
          row.classList.remove("chat-hidden", "chat-filter-match");
        });
        if (elements.filterEmpty) {
          elements.filterEmpty.classList.add("d-none");
        }
        if (elements.filterHint) {
          elements.filterHint.textContent = filterHintDefault;
        }
      }
    }
    function clearTranscriptFilter(focus = true) {
      state.activeFilter = "";
      if (elements.filterInput) {
        elements.filterInput.value = "";
      }
      reapplyTranscriptFilter();
      if (focus && elements.filterInput) {
        elements.filterInput.focus();
      }
    }
    function renderHistory(entries, options = {}) {
      const { replace = false } = options;
      if (!Array.isArray(entries) || entries.length === 0) {
        if (replace) {
          elements.transcript.innerHTML = "";
          state.historyBootstrapped = false;
          hideScrollButton();
          timelineStore.clear();
        }
        return;
      }
      if (replace) {
        elements.transcript.innerHTML = "";
        state.historyBootstrapped = false;
        state.streamRow = null;
        state.streamBuf = "";
        timelineStore.clear();
      }
      if (state.historyBootstrapped && !replace) {
        state.bootstrapping = true;
        const rows = Array.from(
          elements.transcript.querySelectorAll(".chat-row")
        );
        rows.forEach((row) => {
          const existingId = row.dataset.messageId;
          if (existingId && timelineStore.map.has(existingId)) {
            const currentRole = row.dataset.role || "";
            if (currentRole) {
              decorateRow(row, currentRole);
            }
            return;
          }
          const bubble = row.querySelector(".chat-bubble");
          const meta = (bubble == null ? void 0 : bubble.querySelector(".chat-meta")) || null;
          const role = row.dataset.role || (row.classList.contains("chat-user") ? "user" : row.classList.contains("chat-assistant") ? "assistant" : "system");
          const text = row.dataset.rawText && row.dataset.rawText.length > 0 ? row.dataset.rawText : bubble ? extractBubbleText(bubble) : row.textContent.trim();
          const timestamp = row.dataset.timestamp && row.dataset.timestamp.length > 0 ? row.dataset.timestamp : meta ? meta.textContent.trim() : nowISO();
          const messageId = timelineStore.register({
            id: existingId,
            role,
            text,
            timestamp,
            row
          });
          row.dataset.messageId = messageId;
          row.dataset.role = role;
          row.dataset.rawText = text;
          row.dataset.timestamp = timestamp;
          decorateRow(row, role);
        });
        state.bootstrapping = false;
        reapplyTranscriptFilter();
        return;
      }
      state.bootstrapping = true;
      entries.slice().reverse().forEach((item) => {
        if (item.query) {
          appendMessage("user", item.query, {
            timestamp: item.timestamp
          });
        }
        if (item.response) {
          appendMessage("assistant", item.response, {
            timestamp: item.timestamp
          });
        }
      });
      state.bootstrapping = false;
      state.historyBootstrapped = true;
      scrollToBottom({ smooth: false });
      hideScrollButton();
    }
    function startStream() {
      state.streamBuf = "";
      const ts = nowISO();
      state.streamMessageId = timelineStore.makeMessageId();
      state.streamRow = line(
        "assistant",
        '<div class="chat-bubble"><span class="chat-cursor">\u258D</span></div>',
        {
          rawText: "",
          timestamp: ts,
          messageId: state.streamMessageId,
          metadata: { streaming: true }
        }
      );
      setDiagnostics({ lastMessageAt: ts });
      if (state.resetStatusTimer) {
        clearTimeout(state.resetStatusTimer);
      }
      setComposerStatus("R\xE9ponse en cours\u2026", "info");
    }
    function isStreaming() {
      return Boolean(state.streamRow);
    }
    function hasStreamBuffer() {
      return Boolean(state.streamBuf);
    }
    function appendStream(delta) {
      if (!state.streamRow) {
        startStream();
      }
      const shouldStick = isAtBottom();
      state.streamBuf += delta || "";
      const bubble = state.streamRow.querySelector(".chat-bubble");
      if (bubble) {
        bubble.innerHTML = `${renderMarkdown(state.streamBuf)}<span class="chat-cursor">\u258D</span>`;
      }
      if (state.streamMessageId) {
        timelineStore.update(state.streamMessageId, {
          text: state.streamBuf,
          metadata: { streaming: true }
        });
      }
      setDiagnostics({ lastMessageAt: nowISO() });
      if (shouldStick) {
        scrollToBottom({ smooth: false });
      }
    }
    function endStream(data) {
      if (!state.streamRow) {
        return;
      }
      const bubble = state.streamRow.querySelector(".chat-bubble");
      if (bubble) {
        bubble.innerHTML = renderMarkdown(state.streamBuf);
        const meta = document.createElement("div");
        meta.className = "chat-meta";
        const ts = data && data.timestamp ? data.timestamp : nowISO();
        meta.textContent = formatTimestamp(ts);
        if (data && data.error) {
          meta.classList.add("text-danger");
          meta.textContent = `${meta.textContent} \u2022 ${data.error}`;
        }
        bubble.appendChild(meta);
        decorateRow(state.streamRow, "assistant");
        highlightRow(state.streamRow, "assistant");
        if (isAtBottom()) {
          scrollToBottom({ smooth: true });
        } else {
          showScrollButton();
        }
        if (state.streamMessageId) {
          timelineStore.update(state.streamMessageId, {
            text: state.streamBuf,
            timestamp: ts,
            metadata: {
              streaming: null,
              ...data && data.error ? { error: data.error } : { error: null }
            }
          });
        }
        setDiagnostics({ lastMessageAt: ts });
      }
      const hasError = Boolean(data && data.error);
      setComposerStatus(
        hasError ? "R\xE9ponse indisponible. Consultez les journaux." : "R\xE9ponse re\xE7ue.",
        hasError ? "danger" : "success"
      );
      scheduleComposerIdle(hasError ? 6e3 : 3500);
      state.streamRow = null;
      state.streamBuf = "";
      state.streamMessageId = null;
    }
    function applyQuickActionOrdering(suggestions) {
      if (!elements.quickActions) return;
      if (!Array.isArray(suggestions) || suggestions.length === 0) return;
      const buttons = Array.from(
        elements.quickActions.querySelectorAll("button.qa")
      );
      const lookup = /* @__PURE__ */ new Map();
      buttons.forEach((btn) => lookup.set(btn.dataset.action, btn));
      const frag = document.createDocumentFragment();
      suggestions.forEach((key) => {
        if (lookup.has(key)) {
          frag.appendChild(lookup.get(key));
          lookup.delete(key);
        }
      });
      lookup.forEach((btn) => frag.appendChild(btn));
      elements.quickActions.innerHTML = "";
      elements.quickActions.appendChild(frag);
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
      return bits.join(" \u2022 ") || "mise \xE0 jour";
    }
    function attachEvents() {
      if (elements.composer) {
        elements.composer.addEventListener("submit", (event) => {
          event.preventDefault();
          const text = (elements.prompt.value || "").trim();
          emit("submit", { text });
        });
      }
      if (elements.quickActions) {
        elements.quickActions.addEventListener("click", (event) => {
          const target = event.target;
          if (!(target instanceof HTMLButtonElement)) {
            return;
          }
          const action = target.dataset.action;
          if (!action) {
            return;
          }
          emit("quick-action", { action });
        });
      }
      if (elements.filterInput) {
        elements.filterInput.addEventListener("input", (event) => {
          emit("filter-change", { value: event.target.value || "" });
        });
        elements.filterInput.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            emit("filter-clear");
          }
        });
      }
      if (elements.filterClear) {
        elements.filterClear.addEventListener("click", () => {
          emit("filter-clear");
        });
      }
      if (elements.exportJson) {
        elements.exportJson.addEventListener(
          "click",
          () => emit("export", { format: "json" })
        );
      }
      if (elements.exportMarkdown) {
        elements.exportMarkdown.addEventListener(
          "click",
          () => emit("export", { format: "markdown" })
        );
      }
      if (elements.exportCopy) {
        elements.exportCopy.addEventListener("click", () => emit("export-copy"));
      }
      if (elements.prompt) {
        elements.prompt.addEventListener("input", (event) => {
          updatePromptMetrics();
          autosizePrompt();
          const value = event.target.value || "";
          if (!value.trim()) {
            setComposerStatusIdle();
          }
          emit("prompt-input", { value });
        });
        elements.prompt.addEventListener("paste", () => {
          window.setTimeout(() => {
            updatePromptMetrics();
            autosizePrompt();
            emit("prompt-input", { value: elements.prompt.value || "" });
          }, 0);
        });
        elements.prompt.addEventListener("keydown", (event) => {
          if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            event.preventDefault();
            emit("submit", { text: (elements.prompt.value || "").trim() });
          }
        });
        elements.prompt.addEventListener("focus", () => {
          setComposerStatus(
            "R\xE9digez votre message, puis Ctrl+Entr\xE9e pour l'envoyer.",
            "info"
          );
          scheduleComposerIdle(4e3);
        });
      }
      if (elements.transcript) {
        elements.transcript.addEventListener("scroll", () => {
          if (isAtBottom()) {
            hideScrollButton();
          } else {
            showScrollButton();
          }
        });
      }
      if (elements.scrollBottom) {
        elements.scrollBottom.addEventListener("click", () => {
          scrollToBottom({ smooth: true });
          if (elements.prompt) {
            elements.prompt.focus();
          }
        });
      }
      window.addEventListener("resize", () => {
        if (isAtBottom()) {
          scrollToBottom({ smooth: false });
        }
      });
      updateNetworkStatus();
      window.addEventListener("online", () => {
        updateNetworkStatus();
        announceConnection("Connexion r\xE9seau restaur\xE9e.", "info");
      });
      window.addEventListener("offline", () => {
        updateNetworkStatus();
        announceConnection("Connexion r\xE9seau perdue.", "danger");
      });
      const toggleBtn = document.getElementById("toggle-dark-mode");
      const darkModeKey = "dark-mode";
      function applyDarkMode(enabled) {
        document.body.classList.toggle("dark-mode", enabled);
        if (toggleBtn) {
          toggleBtn.textContent = enabled ? "Mode Clair" : "Mode Sombre";
          toggleBtn.setAttribute("aria-pressed", enabled ? "true" : "false");
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
      if (elements.voiceToggle) {
        elements.voiceToggle.addEventListener("click", () => {
          emit("voice-toggle");
        });
      }
      if (elements.voiceAutoSend) {
        elements.voiceAutoSend.addEventListener("change", (event) => {
          emit("voice-autosend-change", { enabled: event.target.checked });
        });
      }
      if (elements.voicePlayback) {
        elements.voicePlayback.addEventListener("change", (event) => {
          emit("voice-playback-change", { enabled: event.target.checked });
        });
      }
      if (elements.voiceStopPlayback) {
        elements.voiceStopPlayback.addEventListener("click", () => {
          emit("voice-stop-playback");
        });
      }
      if (elements.voiceVoiceSelect) {
        elements.voiceVoiceSelect.addEventListener("change", (event) => {
          emit("voice-voice-change", { voiceURI: event.target.value || null });
        });
      }
    }
    function initialise() {
      setDiagnostics({ connectedAt: null, lastMessageAt: null, latencyMs: null });
      updatePromptMetrics();
      autosizePrompt();
      setComposerStatusIdle();
      setVoiceTranscript("", { state: "idle", placeholder: "" });
      attachEvents();
    }
    return {
      elements,
      on,
      emit,
      initialise,
      renderHistory,
      appendMessage,
      setBusy,
      showError,
      hideError,
      setComposerStatus,
      setComposerStatusIdle,
      scheduleComposerIdle,
      updatePromptMetrics,
      autosizePrompt,
      startStream,
      appendStream,
      endStream,
      announceConnection,
      updateConnectionMeta,
      setDiagnostics,
      applyQuickActionOrdering,
      applyTranscriptFilter,
      reapplyTranscriptFilter,
      clearTranscriptFilter,
      setWsStatus,
      updateNetworkStatus,
      scrollToBottom,
      setVoiceStatus,
      scheduleVoiceStatusIdle,
      setVoiceAvailability,
      setVoiceListening,
      setVoiceTranscript,
      setVoicePreferences,
      setVoiceSpeaking,
      setVoiceVoiceOptions,
      set diagnostics(value) {
        Object.assign(diagnostics, value);
      },
      get diagnostics() {
        return { ...diagnostics };
      },
      formatTimestamp,
      nowISO,
      formatPerf,
      isStreaming,
      hasStreamBuffer,
      get voiceStatusDefault() {
        return voiceStatusDefault;
      }
    };
  }

  // src/services/auth.js
  var DEFAULT_STORAGE_KEY = "mongars_jwt";
  function hasLocalStorage() {
    try {
      return typeof window !== "undefined" && Boolean(window.localStorage);
    } catch (err) {
      console.warn("Accessing localStorage failed", err);
      return false;
    }
  }
  function createAuthService(config = {}) {
    const storageKey = config.storageKey || DEFAULT_STORAGE_KEY;
    let fallbackToken = typeof config.token === "string" && config.token.trim() !== "" ? config.token.trim() : void 0;
    function persistToken(token) {
      if (typeof token === "string") {
        token = token.trim();
      }
      fallbackToken = token || void 0;
      if (!token) {
        clearToken();
        return;
      }
      if (!hasLocalStorage()) {
        return;
      }
      try {
        window.localStorage.setItem(storageKey, token);
      } catch (err) {
        console.warn("Unable to persist JWT in localStorage", err);
      }
    }
    function readStoredToken() {
      if (!hasLocalStorage()) {
        return void 0;
      }
      try {
        const stored = window.localStorage.getItem(storageKey);
        return stored || void 0;
      } catch (err) {
        console.warn("Unable to read JWT from localStorage", err);
        return void 0;
      }
    }
    function clearToken() {
      fallbackToken = void 0;
      if (!hasLocalStorage()) {
        return;
      }
      try {
        window.localStorage.removeItem(storageKey);
      } catch (err) {
        console.warn("Unable to clear JWT from localStorage", err);
      }
    }
    if (fallbackToken) {
      persistToken(fallbackToken);
    }
    async function getJwt() {
      const stored = readStoredToken();
      if (stored) {
        return stored;
      }
      if (fallbackToken) {
        return fallbackToken;
      }
      throw new Error("Missing JWT for chat authentication.");
    }
    return {
      getJwt,
      persistToken,
      clearToken,
      storageKey
    };
  }

  // src/services/http.js
  function createHttpService({ config, auth }) {
    async function authorisedFetch(path, options = {}) {
      let jwt;
      try {
        jwt = await auth.getJwt();
      } catch (err) {
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
        method: "POST"
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
        body: JSON.stringify({ message })
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
          actions: ["code", "summarize", "explain"]
        })
      });
      if (!resp.ok) {
        throw new Error(`Suggestion error: ${resp.status}`);
      }
      return resp.json();
    }
    return {
      fetchTicket,
      postChat,
      postSuggestions
    };
  }

  // src/services/exporter.js
  function buildExportFilename(extension) {
    const stamp = nowISO().replace(/[:.]/g, "-");
    return `mongars-chat-${stamp}.${extension}`;
  }
  function buildMarkdownExport(items) {
    const lines = ["# Historique de conversation monGARS", ""];
    items.forEach((item) => {
      const role = item.role ? item.role.toUpperCase() : "MESSAGE";
      lines.push(`## ${role}`);
      if (item.timestamp) {
        lines.push(`*Horodatage\xA0:* ${item.timestamp}`);
      }
      if (item.metadata && Object.keys(item.metadata).length > 0) {
        Object.entries(item.metadata).forEach(([key, value]) => {
          lines.push(`*${key}\xA0:* ${value}`);
        });
      }
      lines.push("");
      lines.push(item.text || "");
      lines.push("");
    });
    return lines.join("\n");
  }
  function downloadBlob(filename, text, type) {
    if (!window.URL || typeof window.URL.createObjectURL !== "function") {
      console.warn("Blob export unsupported in this environment");
      return false;
    }
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    return true;
  }
  async function copyToClipboard(text) {
    if (!text) return false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "readonly");
        textarea.style.position = "absolute";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      return true;
    } catch (err) {
      console.warn("Copy conversation failed", err);
      return false;
    }
  }
  function createExporter({ timelineStore, announce }) {
    function collectTranscript() {
      return timelineStore.collect();
    }
    async function exportConversation(format) {
      const items = collectTranscript();
      if (!items.length) {
        announce("Aucun message \xE0 exporter.", "warning");
        return;
      }
      if (format === "json") {
        const payload = {
          exported_at: nowISO(),
          count: items.length,
          items
        };
        if (downloadBlob(
          buildExportFilename("json"),
          JSON.stringify(payload, null, 2),
          "application/json"
        )) {
          announce("Export JSON g\xE9n\xE9r\xE9.", "success");
        } else {
          announce("Export non support\xE9 dans ce navigateur.", "danger");
        }
        return;
      }
      if (format === "markdown") {
        if (downloadBlob(
          buildExportFilename("md"),
          buildMarkdownExport(items),
          "text/markdown"
        )) {
          announce("Export Markdown g\xE9n\xE9r\xE9.", "success");
        } else {
          announce("Export non support\xE9 dans ce navigateur.", "danger");
        }
      }
    }
    async function copyConversationToClipboard() {
      const items = collectTranscript();
      if (!items.length) {
        announce("Aucun message \xE0 copier.", "warning");
        return;
      }
      const text = buildMarkdownExport(items);
      if (await copyToClipboard(text)) {
        announce("Conversation copi\xE9e au presse-papiers.", "success");
      } else {
        announce("Impossible de copier la conversation.", "danger");
      }
    }
    return {
      exportConversation,
      copyConversationToClipboard
    };
  }

  // src/services/socket.js
  function createSocketClient({ config, http, ui, onEvent }) {
    let ws;
    let wsHBeat;
    let reconnectBackoff = 500;
    const BACKOFF_MAX = 8e3;
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
        reconnectBackoff = Math.min(
          BACKOFF_MAX,
          Math.max(500, reconnectBackoff * 2)
        );
        void openSocket();
      }, delay);
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
        ui.updateConnectionMeta("Obtention d\u2019un ticket de connexion\u2026", "info");
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
        ui.updateConnectionMeta("Connexion au serveur\u2026", "info");
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
            `Connect\xE9 le ${ui.formatTimestamp(connectedAt)}`,
            "success"
          );
          ui.setDiagnostics({ connectedAt, lastMessageAt: connectedAt });
          ui.hideError();
          clearHeartbeat();
          wsHBeat = window.setInterval(() => {
            safeSend({ type: "client.ping", ts: nowISO() });
          }, 2e4);
          ui.setComposerStatus("Connect\xE9. Vous pouvez \xE9changer.", "success");
          ui.scheduleComposerIdle(4e3);
        };
        ws.onmessage = (evt) => {
          const receivedAt = nowISO();
          try {
            const ev = JSON.parse(evt.data);
            ui.setDiagnostics({ lastMessageAt: receivedAt });
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
          ui.setDiagnostics({ latencyMs: void 0 });
          const delay = scheduleReconnect(reconnectBackoff);
          const seconds = Math.max(1, Math.round(delay / 1e3));
          ui.updateConnectionMeta(
            `D\xE9connect\xE9. Nouvelle tentative dans ${seconds} s\u2026`,
            "warning"
          );
          ui.setComposerStatus(
            "Connexion perdue. Reconnexion automatique\u2026",
            "warning"
          );
          ui.scheduleComposerIdle(6e3);
        };
        ws.onerror = (err) => {
          console.error("WebSocket error", err);
          if (disposed) {
            return;
          }
          ui.setWsStatus("error", "Erreur WebSocket");
          ui.updateConnectionMeta("Erreur WebSocket d\xE9tect\xE9e.", "danger");
          ui.setComposerStatus("Une erreur r\xE9seau est survenue.", "danger");
          ui.scheduleComposerIdle(6e3);
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
          "Connexion indisponible. Nouvel essai bient\xF4t.",
          "danger"
        );
        ui.scheduleComposerIdle(6e3);
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
      dispose
    };
  }

  // src/services/suggestions.js
  function createSuggestionService({ http, ui }) {
    let timer = null;
    function schedule(prompt) {
      if (timer) {
        clearTimeout(timer);
      }
      timer = window.setTimeout(() => fetchSuggestions(prompt), 220);
    }
    async function fetchSuggestions(prompt) {
      if (!prompt || prompt.trim().length < 3) {
        return;
      }
      try {
        const payload = await http.postSuggestions(prompt.trim());
        if (payload && Array.isArray(payload.actions)) {
          ui.applyQuickActionOrdering(payload.actions);
        }
      } catch (err) {
        console.debug("AUI suggestion fetch failed", err);
      }
    }
    return {
      schedule
    };
  }

  // src/services/speech.js
  function normalizeText(value) {
    if (!value) {
      return "";
    }
    return String(value).replace(/\s+/g, " ").trim();
  }
  function describeRecognitionError(code, fallback = "") {
    switch (code) {
      case "not-allowed":
      case "service-not-allowed":
        return "Acc\xE8s au microphone refus\xE9. Autorisez la dict\xE9e vocale dans votre navigateur.";
      case "network":
        return "La reconnaissance vocale a \xE9t\xE9 interrompue par un probl\xE8me r\xE9seau.";
      case "no-speech":
        return "Aucune voix d\xE9tect\xE9e. Essayez de parler plus pr\xE8s du micro.";
      case "aborted":
        return "La dict\xE9e a \xE9t\xE9 interrompue.";
      case "audio-capture":
        return "Aucun microphone disponible. V\xE9rifiez votre mat\xE9riel.";
      case "bad-grammar":
        return "Le service de dict\xE9e a rencontr\xE9 une erreur de traitement.";
      default:
        return fallback || "La reconnaissance vocale a rencontr\xE9 une erreur inattendue.";
    }
  }
  function mapVoice(voice) {
    return {
      name: voice.name,
      lang: voice.lang,
      voiceURI: voice.voiceURI,
      default: Boolean(voice.default),
      localService: Boolean(voice.localService)
    };
  }
  function createSpeechService({ defaultLanguage } = {}) {
    const emitter = createEmitter();
    const globalScope = typeof window !== "undefined" ? window : {};
    const RecognitionCtor = globalScope.SpeechRecognition || globalScope.webkitSpeechRecognition || null;
    const recognitionSupported = Boolean(RecognitionCtor);
    const synthesisSupported = Boolean(globalScope.speechSynthesis);
    const synth = synthesisSupported ? globalScope.speechSynthesis : null;
    let recognition = null;
    const navigatorLanguage = typeof navigator !== "undefined" && navigator.language ? navigator.language : null;
    let recognitionLang = defaultLanguage || navigatorLanguage || "fr-CA";
    let manualStop = false;
    let listening = false;
    let speaking = false;
    let preferredVoiceURI = null;
    let voicesCache = [];
    let microphonePrimed = false;
    let microphonePriming = null;
    const userAgent = typeof navigator !== "undefined" && navigator.userAgent ? navigator.userAgent.toLowerCase() : "";
    const isAppleMobile = /iphone|ipad|ipod/.test(userAgent);
    const isSafari = /safari/.test(userAgent) && !/crios|fxios|chrome|android|edge|edg|opr|opera/.test(userAgent);
    function requiresMicrophonePriming() {
      if (!recognitionSupported) {
        return false;
      }
      if (microphonePrimed) {
        return false;
      }
      if (typeof navigator === "undefined") {
        return false;
      }
      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
        return false;
      }
      return isAppleMobile && isSafari;
    }
    function releaseStream(stream) {
      if (!stream) {
        return;
      }
      const tracks = typeof stream.getTracks === "function" ? stream.getTracks() : [];
      tracks.forEach((track) => {
        try {
          track.stop();
        } catch (err) {
          console.debug("Unable to stop track", err);
        }
      });
    }
    async function ensureMicrophoneAccess() {
      if (!requiresMicrophonePriming()) {
        return true;
      }
      if (microphonePrimed) {
        return true;
      }
      if (microphonePriming) {
        try {
          return await microphonePriming;
        } catch (err) {
          return false;
        }
      }
      microphonePriming = navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        microphonePrimed = true;
        releaseStream(stream);
        return true;
      }).catch((err) => {
        emitError({
          source: "recognition",
          code: "permission-denied",
          message: "Autorisation du microphone refus\xE9e. Activez l'acc\xE8s dans les r\xE9glages de Safari.",
          details: err
        });
        return false;
      }).finally(() => {
        microphonePriming = null;
      });
      return microphonePriming;
    }
    function isPermissionError(error) {
      if (!error) {
        return false;
      }
      const code = typeof error === "string" ? error : error.name || error.code || error.message || "";
      const normalised = String(code).toLowerCase();
      return [
        "notallowederror",
        "not-allowed",
        "service-not-allowed",
        "securityerror",
        "permissiondeniederror",
        "aborterror"
      ].some((candidate) => normalised.includes(candidate));
    }
    function emitError(payload) {
      const enriched = {
        timestamp: nowISO(),
        ...payload
      };
      console.error("Speech service error", enriched);
      emitter.emit("error", enriched);
    }
    function ensureRecognition() {
      if (!recognitionSupported) {
        return null;
      }
      if (recognition) {
        return recognition;
      }
      recognition = new RecognitionCtor();
      recognition.lang = recognitionLang;
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.onstart = () => {
        listening = true;
        emitter.emit("listening-change", {
          listening: true,
          reason: "start",
          timestamp: nowISO()
        });
      };
      recognition.onend = () => {
        const reason = manualStop ? "manual" : "ended";
        listening = false;
        emitter.emit("listening-change", {
          listening: false,
          reason,
          timestamp: nowISO()
        });
        manualStop = false;
      };
      recognition.onerror = (event) => {
        listening = false;
        const code = event.error || "unknown";
        emitError({
          source: "recognition",
          code,
          message: describeRecognitionError(code, event.message),
          event
        });
        emitter.emit("listening-change", {
          listening: false,
          reason: "error",
          code,
          timestamp: nowISO()
        });
      };
      recognition.onresult = (event) => {
        if (!event.results) {
          return;
        }
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
          const result = event.results[i];
          if (!result || result.length === 0) {
            continue;
          }
          const alternative = result[0];
          const transcript = normalizeText((alternative == null ? void 0 : alternative.transcript) || "");
          if (!transcript) {
            continue;
          }
          emitter.emit("transcript", {
            transcript,
            isFinal: Boolean(result.isFinal),
            confidence: typeof alternative.confidence === "number" ? alternative.confidence : null,
            timestamp: nowISO()
          });
        }
      };
      recognition.onaudioend = () => {
        emitter.emit("audio-end", { timestamp: nowISO() });
      };
      recognition.onspeechend = () => {
        emitter.emit("speech-end", { timestamp: nowISO() });
      };
      return recognition;
    }
    async function startListening(options = {}) {
      if (!recognitionSupported) {
        emitError({
          source: "recognition",
          code: "unsupported",
          message: "La dict\xE9e vocale n'est pas disponible sur cet appareil."
        });
        return false;
      }
      const instance = ensureRecognition();
      if (!instance) {
        return false;
      }
      if (listening) {
        return true;
      }
      manualStop = false;
      recognitionLang = normalizeText(options.language) || recognitionLang;
      instance.lang = recognitionLang;
      instance.interimResults = options.interimResults !== false;
      instance.continuous = Boolean(options.continuous);
      instance.maxAlternatives = options.maxAlternatives || 1;
      if (requiresMicrophonePriming()) {
        const granted = await ensureMicrophoneAccess();
        if (!granted) {
          return false;
        }
      }
      try {
        instance.start();
        return true;
      } catch (err) {
        if (requiresMicrophonePriming() && !microphonePrimed && isPermissionError(err)) {
          const granted = await ensureMicrophoneAccess();
          if (granted) {
            try {
              instance.start();
              return true;
            } catch (retryErr) {
              emitError({
                source: "recognition",
                code: "start-failed",
                message: retryErr && retryErr.message ? retryErr.message : "Impossible de d\xE9marrer la reconnaissance vocale.",
                details: retryErr
              });
              return false;
            }
          }
        }
        emitError({
          source: "recognition",
          code: "start-failed",
          message: err && err.message ? err.message : "Impossible de d\xE9marrer la reconnaissance vocale.",
          details: err
        });
        return false;
      }
    }
    function stopListening(options = {}) {
      if (!recognition) {
        return;
      }
      manualStop = true;
      try {
        if (options && options.abort && typeof recognition.abort === "function") {
          recognition.abort();
        } else {
          recognition.stop();
        }
      } catch (err) {
        emitError({
          source: "recognition",
          code: "stop-failed",
          message: "Arr\xEAt de la dict\xE9e impossible.",
          details: err
        });
      }
    }
    function findVoice(uri) {
      if (!uri || !synth) {
        return null;
      }
      const voices = synth.getVoices();
      return voices.find((voice) => voice.voiceURI === uri) || null;
    }
    function refreshVoices() {
      if (!synth) {
        return [];
      }
      try {
        voicesCache = synth.getVoices();
        const payload = voicesCache.map(mapVoice);
        emitter.emit("voices", { voices: payload });
        return payload;
      } catch (err) {
        emitError({
          source: "synthesis",
          code: "voices-failed",
          message: "Impossible de r\xE9cup\xE9rer la liste des voix disponibles.",
          details: err
        });
        return [];
      }
    }
    function speak(text, options = {}) {
      if (!synthesisSupported) {
        emitError({
          source: "synthesis",
          code: "unsupported",
          message: "La synth\xE8se vocale n'est pas disponible sur cet appareil."
        });
        return null;
      }
      const content = normalizeText(text);
      if (!content) {
        return null;
      }
      if (listening) {
        stopListening({ abort: true });
      }
      stopSpeaking();
      const utterance = new SpeechSynthesisUtterance(content);
      utterance.lang = normalizeText(options.lang) || recognitionLang;
      const rate = Number(options.rate);
      if (!Number.isNaN(rate) && rate > 0) {
        utterance.rate = Math.min(rate, 2);
      }
      const pitch = Number(options.pitch);
      if (!Number.isNaN(pitch) && pitch > 0) {
        utterance.pitch = Math.min(pitch, 2);
      }
      const voice = findVoice(options.voiceURI) || findVoice(preferredVoiceURI) || null;
      if (voice) {
        utterance.voice = voice;
      }
      utterance.onstart = () => {
        speaking = true;
        emitter.emit("speaking-change", {
          speaking: true,
          utterance,
          timestamp: nowISO()
        });
      };
      utterance.onend = () => {
        speaking = false;
        emitter.emit("speaking-change", {
          speaking: false,
          utterance,
          timestamp: nowISO()
        });
      };
      utterance.onerror = (event) => {
        speaking = false;
        emitError({
          source: "synthesis",
          code: event.error || "unknown",
          message: event && event.message ? event.message : "La synth\xE8se vocale a rencontr\xE9 une erreur.",
          event
        });
        emitter.emit("speaking-change", {
          speaking: false,
          utterance,
          reason: "error",
          timestamp: nowISO()
        });
      };
      synth.speak(utterance);
      return utterance;
    }
    function stopSpeaking() {
      if (!synthesisSupported) {
        return;
      }
      if (synth.speaking || synth.pending) {
        synth.cancel();
      }
      if (speaking) {
        speaking = false;
        emitter.emit("speaking-change", {
          speaking: false,
          reason: "cancel",
          timestamp: nowISO()
        });
      }
    }
    function setPreferredVoice(uri) {
      preferredVoiceURI = uri || null;
    }
    function setLanguage(lang) {
      const next = normalizeText(lang);
      if (next) {
        recognitionLang = next;
        if (recognition) {
          recognition.lang = recognitionLang;
        }
      }
    }
    if (synthesisSupported) {
      refreshVoices();
      if (synth.addEventListener) {
        synth.addEventListener("voiceschanged", refreshVoices);
      } else if ("onvoiceschanged" in synth) {
        synth.onvoiceschanged = refreshVoices;
      }
    }
    return {
      on: emitter.on,
      off: emitter.off,
      startListening,
      stopListening,
      speak,
      stopSpeaking,
      setPreferredVoice,
      setLanguage,
      refreshVoices,
      getVoices: () => voicesCache.map(mapVoice),
      getPreferredVoice: () => preferredVoiceURI,
      isRecognitionSupported: () => recognitionSupported,
      isSynthesisSupported: () => synthesisSupported
    };
  }

  // src/app.js
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
      voiceSpeakingIndicator: byId("voice-speaking-indicator")
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
  var QUICK_PRESETS = {
    code: "Je souhaite \xE9crire du code\u2026",
    summarize: "R\xE9sume la derni\xE8re conversation.",
    explain: "Explique ta derni\xE8re r\xE9ponse plus simplement."
  };
  var ChatApp = class {
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
          mangle: false
        });
      }
      this.timelineStore = createTimelineStore();
      this.ui = createChatUi({
        elements: this.elements,
        timelineStore: this.timelineStore
      });
      this.auth = createAuthService(this.config);
      this.http = createHttpService({ config: this.config, auth: this.auth });
      this.exporter = createExporter({
        timelineStore: this.timelineStore,
        announce: (message, variant) => this.ui.announceConnection(message, variant)
      });
      this.suggestions = createSuggestionService({
        http: this.http,
        ui: this.ui
      });
      this.socket = createSocketClient({
        config: this.config,
        http: this.http,
        ui: this.ui,
        onEvent: (ev) => this.handleSocketEvent(ev)
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
    registerUiHandlers() {
      this.ui.on("submit", async ({ text }) => {
        const value = (text || "").trim();
        if (!value) {
          this.ui.setComposerStatus(
            "Saisissez un message avant d\u2019envoyer.",
            "warning"
          );
          this.ui.scheduleComposerIdle(4e3);
          return;
        }
        this.ui.hideError();
        const submittedAt = nowISO();
        this.ui.appendMessage("user", value, {
          timestamp: submittedAt,
          metadata: { submitted: true, mode: requestMode }
        });
        if (this.elements.prompt) {
          this.elements.prompt.value = "";
        }
        this.ui.updatePromptMetrics();
        this.ui.autosizePrompt();
        this.ui.setComposerStatus("Message envoy\xE9\u2026", "info");
        this.ui.scheduleComposerIdle(4e3);
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
          this.ui.showError(err, {
            metadata: { stage: "submit", mode: requestMode }
          });
          this.ui.setComposerStatus(
            "Envoi impossible. V\xE9rifiez la connexion.",
            "danger"
          );
          this.ui.scheduleComposerIdle(6e3);
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
        this.ui.setComposerStatus("Suggestion envoy\xE9e\u2026", "info");
        this.ui.scheduleComposerIdle(4e3);
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
        language: defaultLanguage
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
          autoSend: typeof parsed.autoSend === "boolean" ? parsed.autoSend : fallback.autoSend,
          playback: typeof parsed.playback === "boolean" ? parsed.playback : fallback.playback,
          voiceURI: typeof parsed.voiceURI === "string" && parsed.voiceURI.length > 0 ? parsed.voiceURI : null,
          language: typeof parsed.language === "string" && parsed.language ? parsed.language : fallback.language
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
            language: this.voicePrefs.language || null
          })
        );
      } catch (err) {
        console.warn("Unable to persist voice preferences", err);
      }
    }
    setupVoiceFeatures() {
      var _a, _b;
      const docLang = (((_b = (_a = this.doc) == null ? void 0 : _a.documentElement) == null ? void 0 : _b.getAttribute("lang")) || "").trim();
      const navigatorLang = typeof navigator !== "undefined" && navigator.language ? navigator.language : null;
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
        lastTranscript: ""
      };
      this.speech = createSpeechService({
        defaultLanguage: this.voicePrefs.language
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
        synthesis: synthesisSupported
      });
      this.ui.setVoicePreferences(this.voicePrefs);
      if (recognitionSupported) {
        this.ui.setVoiceStatus(
          "Activez le micro pour dicter votre message.",
          "muted"
        );
      } else if (synthesisSupported) {
        this.ui.setVoiceStatus(
          "Lecture vocale disponible. La dict\xE9e n\xE9cessite un navigateur compatible.",
          "warning"
        );
      } else {
        this.ui.setVoiceStatus(
          "Les fonctionnalit\xE9s vocales ne sont pas disponibles dans ce navigateur.",
          "danger"
        );
      }
      this.ui.scheduleVoiceStatusIdle(recognitionSupported ? 5e3 : 7e3);
      this.speech.on(
        "listening-change",
        (payload) => this.handleVoiceListeningChange(payload)
      );
      this.speech.on(
        "transcript",
        (payload) => this.handleVoiceTranscript(payload)
      );
      this.speech.on("error", (payload) => this.handleVoiceError(payload));
      this.speech.on(
        "speaking-change",
        (payload) => this.handleVoiceSpeakingChange(payload)
      );
      this.speech.on(
        "voices",
        ({ voices }) => this.handleVoiceVoices(Array.isArray(voices) ? voices : [])
      );
    }
    async toggleVoiceListening() {
      if (!this.speech || !this.speech.isRecognitionSupported()) {
        this.ui.setVoiceStatus(
          "La dict\xE9e vocale n'est pas disponible dans ce navigateur.",
          "danger"
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
        this.ui.setVoiceStatus("Dict\xE9e interrompue.", "muted");
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
        continuous: false
      });
      if (!started) {
        this.voiceState.enabled = false;
        this.ui.setVoiceStatus(
          "Impossible de d\xE9marrer la dict\xE9e. V\xE9rifiez le micro.",
          "danger"
        );
      }
    }
    handleVoiceListeningChange(payload = {}) {
      const listening = Boolean(payload.listening);
      this.voiceState.listening = listening;
      if (listening) {
        this.ui.setVoiceListening(true);
        this.ui.setVoiceTranscript("", { state: "idle" });
        this.ui.setVoiceStatus("En \xE9coute\u2026 Parlez lorsque vous \xEAtes pr\xEAt.", "info");
        return;
      }
      this.ui.setVoiceListening(false);
      if (payload.reason === "manual") {
        this.voiceState.manualStop = false;
        this.voiceState.enabled = false;
        this.ui.setVoiceStatus("Dict\xE9e interrompue.", "muted");
        this.ui.scheduleVoiceStatusIdle(3500);
        return;
      }
      if (payload.reason === "error") {
        this.voiceState.enabled = false;
        this.voiceState.awaitingResponse = false;
        const message = payload.code === "not-allowed" ? "Autorisez l'acc\xE8s au microphone pour continuer." : "La dict\xE9e vocale s'est interrompue. R\xE9essayez.";
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
      const transcript = typeof payload.transcript === "string" ? payload.transcript : "";
      const isFinal = Boolean(payload.isFinal);
      const confidence = typeof payload.confidence === "number" ? payload.confidence : null;
      if (transcript) {
        this.voiceState.lastTranscript = transcript;
        this.ui.setVoiceTranscript(transcript, {
          state: isFinal ? "final" : "interim"
        });
      }
      if (!isFinal) {
        if (transcript) {
          this.ui.setVoiceStatus("Transcription en cours\u2026", "info");
        }
        return;
      }
      if (!transcript) {
        this.ui.setVoiceStatus("Aucun texte n'a \xE9t\xE9 reconnu.", "warning");
        this.ui.scheduleVoiceStatusIdle(3e3);
        this.voiceState.awaitingResponse = false;
        if (!this.voicePrefs.autoSend) {
          this.voiceState.enabled = false;
        }
        return;
      }
      if (this.voicePrefs.autoSend) {
        this.voiceState.awaitingResponse = true;
        const confidencePct = confidence !== null ? Math.round(Math.max(0, Math.min(1, confidence)) * 100) : null;
        if (confidencePct !== null) {
          this.ui.setVoiceStatus(
            `Envoi du message dict\xE9 (${confidencePct}% de confiance)\u2026`,
            "info"
          );
        } else {
          this.ui.setVoiceStatus("Envoi du message dict\xE9\u2026", "info");
        }
        this.submitVoicePrompt(transcript);
      } else {
        if (this.elements.prompt) {
          this.elements.prompt.value = transcript;
        }
        this.ui.updatePromptMetrics();
        this.ui.autosizePrompt();
        this.ui.setVoiceStatus(
          "Message dict\xE9. V\xE9rifiez avant l'envoi.",
          "info"
        );
        this.ui.scheduleVoiceStatusIdle(4500);
        this.voiceState.enabled = false;
      }
    }
    handleVoiceError(payload = {}) {
      const message = typeof payload.message === "string" && payload.message.length > 0 ? payload.message : "Une erreur vocale est survenue.";
      this.ui.setVoiceStatus(message, "danger");
      this.voiceState.enabled = false;
      this.voiceState.awaitingResponse = false;
      if (this.voiceState.restartTimer) {
        window.clearTimeout(this.voiceState.restartTimer);
        this.voiceState.restartTimer = null;
      }
      this.ui.scheduleVoiceStatusIdle(6e3);
    }
    handleVoiceSpeakingChange(payload = {}) {
      const speaking = Boolean(payload.speaking);
      this.ui.setVoiceSpeaking(speaking);
      if (speaking) {
        this.ui.setVoiceStatus("Lecture de la r\xE9ponse\u2026", "info");
        return;
      }
      if (this.voicePrefs.autoSend && this.voiceState.enabled && !this.voiceState.awaitingResponse) {
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
          "Mode manuel activ\xE9. Utilisez le micro pour remplir le champ.",
          "muted"
        );
        this.ui.scheduleVoiceStatusIdle(4e3);
      } else {
        this.ui.setVoiceStatus(
          "Les messages dict\xE9s seront envoy\xE9s automatiquement.",
          "info"
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
        this.ui.setVoiceStatus("Lecture vocale d\xE9sactiv\xE9e.", "muted");
      } else {
        this.ui.setVoiceStatus("Lecture vocale activ\xE9e.", "info");
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
        this.ui.setVoiceStatus("Voix s\xE9lectionn\xE9e mise \xE0 jour.", "success");
      } else {
        this.ui.setVoiceStatus("Voix par d\xE9faut du syst\xE8me utilis\xE9e.", "muted");
      }
      this.ui.scheduleVoiceStatusIdle(3e3);
    }
    stopVoicePlayback() {
      if (!this.speech || !this.speech.isSynthesisSupported()) {
        return;
      }
      this.speech.stopSpeaking();
      this.ui.setVoiceSpeaking(false);
      this.ui.setVoiceStatus("Lecture vocale interrompue.", "muted");
      this.ui.scheduleVoiceStatusIdle(3e3);
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
          continuous: false
        });
        Promise.resolve(attempt).then((started) => {
          if (started) {
            return;
          }
          this.voiceState.enabled = false;
          this.ui.setVoiceStatus(
            "Impossible de relancer la dict\xE9e vocale.",
            "danger"
          );
        }).catch((err) => {
          this.voiceState.enabled = false;
          console.error("Automatic voice restart failed", err);
          this.ui.setVoiceStatus(
            "Impossible de relancer la dict\xE9e vocale.",
            "danger"
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
      if (this.voicePrefs.playback && this.speech && this.speech.isSynthesisSupported()) {
        this.ui.setVoiceStatus("Lecture de la r\xE9ponse\u2026", "info");
        const utterance = this.speech.speak(latest, {
          lang: this.voicePrefs.language,
          voiceURI: this.voicePrefs.voiceURI
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
            this.ui.announceConnection(`Connect\xE9 via ${data.origin}`);
            this.ui.updateConnectionMeta(
              `Connect\xE9 via ${data.origin}`,
              "success"
            );
          } else {
            this.ui.announceConnection("Connect\xE9 au serveur.");
            this.ui.updateConnectionMeta("Connect\xE9 au serveur.", "success");
          }
          this.ui.scheduleComposerIdle(4e3);
          break;
        }
        case "history.snapshot": {
          if (data && Array.isArray(data.items)) {
            this.ui.renderHistory(data.items, { replace: true });
          }
          break;
        }
        case "ai_model.response_chunk": {
          const delta = typeof data.delta === "string" ? data.delta : data.text || "";
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
              metadata: { event: type }
            });
          }
          this.handleVoiceAssistantCompletion();
          break;
        }
        case "chat.message": {
          if (!this.ui.isStreaming()) {
            this.ui.startStream();
          }
          if (data && typeof data.response === "string" && !this.ui.hasStreamBuffer()) {
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
            `\xC9volution mise \xE0 jour ${data && data.version ? data.version : ""}`,
            {
              variant: "ok",
              allowMarkdown: false,
              metadata: { event: type }
            }
          );
          break;
        }
        case "evolution_engine.training_failed": {
          this.ui.appendMessage(
            "system",
            `\xC9chec de l'\xE9volution : ${data && data.error ? data.error : "inconnu"}`,
            {
              variant: "error",
              allowMarkdown: false,
              metadata: { event: type }
            }
          );
          break;
        }
        case "sleep_time_compute.phase_start": {
          this.ui.appendMessage(
            "system",
            "Optimisation en arri\xE8re-plan d\xE9marr\xE9e\u2026",
            {
              variant: "hint",
              allowMarkdown: false,
              metadata: { event: type }
            }
          );
          break;
        }
        case "sleep_time_compute.creative_phase": {
          this.ui.appendMessage(
            "system",
            `Exploration de ${Number(data && data.ideas ? data.ideas : 1)} id\xE9es\u2026`,
            {
              variant: "hint",
              allowMarkdown: false,
              metadata: { event: type }
            }
          );
          break;
        }
        case "performance.alert": {
          this.ui.appendMessage("system", `Perf : ${this.ui.formatPerf(data)}`, {
            variant: "warn",
            allowMarkdown: false,
            metadata: { event: type }
          });
          if (data && typeof data.ttfb_ms !== "undefined") {
            this.ui.setDiagnostics({ latencyMs: Number(data.ttfb_ms) });
          }
          break;
        }
        case "ui.suggestions": {
          this.ui.applyQuickActionOrdering(
            Array.isArray(data.actions) ? data.actions : []
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
  };

  // src/index.js
  new ChatApp(document, window.chatConfig || {});
})();
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsic3JjL2NvbmZpZy5qcyIsICJzcmMvdXRpbHMvdGltZS5qcyIsICJzcmMvc3RhdGUvdGltZWxpbmVTdG9yZS5qcyIsICJzcmMvdXRpbHMvZW1pdHRlci5qcyIsICJzcmMvdXRpbHMvZG9tLmpzIiwgInNyYy9zZXJ2aWNlcy9tYXJrZG93bi5qcyIsICJzcmMvdWkvY2hhdFVpLmpzIiwgInNyYy9zZXJ2aWNlcy9hdXRoLmpzIiwgInNyYy9zZXJ2aWNlcy9odHRwLmpzIiwgInNyYy9zZXJ2aWNlcy9leHBvcnRlci5qcyIsICJzcmMvc2VydmljZXMvc29ja2V0LmpzIiwgInNyYy9zZXJ2aWNlcy9zdWdnZXN0aW9ucy5qcyIsICJzcmMvc2VydmljZXMvc3BlZWNoLmpzIiwgInNyYy9hcHAuanMiLCAic3JjL2luZGV4LmpzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJleHBvcnQgZnVuY3Rpb24gcmVzb2x2ZUNvbmZpZyhyYXcgPSB7fSkge1xuICBjb25zdCBjb25maWcgPSB7IC4uLnJhdyB9O1xuICBjb25zdCBjYW5kaWRhdGUgPSBjb25maWcuZmFzdGFwaVVybCB8fCB3aW5kb3cubG9jYXRpb24ub3JpZ2luO1xuICB0cnkge1xuICAgIGNvbmZpZy5iYXNlVXJsID0gbmV3IFVSTChjYW5kaWRhdGUpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLmVycm9yKFwiSW52YWxpZCBGQVNUQVBJIFVSTFwiLCBlcnIsIGNhbmRpZGF0ZSk7XG4gICAgY29uZmlnLmJhc2VVcmwgPSBuZXcgVVJMKHdpbmRvdy5sb2NhdGlvbi5vcmlnaW4pO1xuICB9XG4gIHJldHVybiBjb25maWc7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhcGlVcmwoY29uZmlnLCBwYXRoKSB7XG4gIHJldHVybiBuZXcgVVJMKHBhdGgsIGNvbmZpZy5iYXNlVXJsKS50b1N0cmluZygpO1xufVxuIiwgImV4cG9ydCBmdW5jdGlvbiBub3dJU08oKSB7XG4gIHJldHVybiBuZXcgRGF0ZSgpLnRvSVNPU3RyaW5nKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBmb3JtYXRUaW1lc3RhbXAodHMpIHtcbiAgaWYgKCF0cykgcmV0dXJuIFwiXCI7XG4gIHRyeSB7XG4gICAgcmV0dXJuIG5ldyBEYXRlKHRzKS50b0xvY2FsZVN0cmluZyhcImZyLUNBXCIpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICByZXR1cm4gU3RyaW5nKHRzKTtcbiAgfVxufVxuIiwgImltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuLi91dGlscy90aW1lLmpzXCI7XG5cbmZ1bmN0aW9uIG1ha2VNZXNzYWdlSWQoKSB7XG4gIHJldHVybiBgbXNnLSR7RGF0ZS5ub3coKS50b1N0cmluZygzNil9LSR7TWF0aC5yYW5kb20oKS50b1N0cmluZygzNikuc2xpY2UoMiwgOCl9YDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVRpbWVsaW5lU3RvcmUoKSB7XG4gIGNvbnN0IG9yZGVyID0gW107XG4gIGNvbnN0IG1hcCA9IG5ldyBNYXAoKTtcblxuICBmdW5jdGlvbiByZWdpc3Rlcih7XG4gICAgaWQsXG4gICAgcm9sZSxcbiAgICB0ZXh0ID0gXCJcIixcbiAgICB0aW1lc3RhbXAgPSBub3dJU08oKSxcbiAgICByb3csXG4gICAgbWV0YWRhdGEgPSB7fSxcbiAgfSkge1xuICAgIGNvbnN0IG1lc3NhZ2VJZCA9IGlkIHx8IG1ha2VNZXNzYWdlSWQoKTtcbiAgICBpZiAoIW1hcC5oYXMobWVzc2FnZUlkKSkge1xuICAgICAgb3JkZXIucHVzaChtZXNzYWdlSWQpO1xuICAgIH1cbiAgICBtYXAuc2V0KG1lc3NhZ2VJZCwge1xuICAgICAgaWQ6IG1lc3NhZ2VJZCxcbiAgICAgIHJvbGUsXG4gICAgICB0ZXh0LFxuICAgICAgdGltZXN0YW1wLFxuICAgICAgcm93LFxuICAgICAgbWV0YWRhdGE6IHsgLi4ubWV0YWRhdGEgfSxcbiAgICB9KTtcbiAgICBpZiAocm93KSB7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBtZXNzYWdlSWQ7XG4gICAgICByb3cuZGF0YXNldC5yb2xlID0gcm9sZTtcbiAgICAgIHJvdy5kYXRhc2V0LnJhd1RleHQgPSB0ZXh0O1xuICAgICAgcm93LmRhdGFzZXQudGltZXN0YW1wID0gdGltZXN0YW1wO1xuICAgIH1cbiAgICByZXR1cm4gbWVzc2FnZUlkO1xuICB9XG5cbiAgZnVuY3Rpb24gdXBkYXRlKGlkLCBwYXRjaCA9IHt9KSB7XG4gICAgaWYgKCFtYXAuaGFzKGlkKSkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGNvbnN0IGVudHJ5ID0gbWFwLmdldChpZCk7XG4gICAgY29uc3QgbmV4dCA9IHsgLi4uZW50cnksIC4uLnBhdGNoIH07XG4gICAgaWYgKHBhdGNoICYmIHR5cGVvZiBwYXRjaC5tZXRhZGF0YSA9PT0gXCJvYmplY3RcIiAmJiBwYXRjaC5tZXRhZGF0YSAhPT0gbnVsbCkge1xuICAgICAgY29uc3QgbWVyZ2VkID0geyAuLi5lbnRyeS5tZXRhZGF0YSB9O1xuICAgICAgT2JqZWN0LmVudHJpZXMocGF0Y2gubWV0YWRhdGEpLmZvckVhY2goKFtrZXksIHZhbHVlXSkgPT4ge1xuICAgICAgICBpZiAodmFsdWUgPT09IHVuZGVmaW5lZCB8fCB2YWx1ZSA9PT0gbnVsbCkge1xuICAgICAgICAgIGRlbGV0ZSBtZXJnZWRba2V5XTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBtZXJnZWRba2V5XSA9IHZhbHVlO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIG5leHQubWV0YWRhdGEgPSBtZXJnZWQ7XG4gICAgfVxuICAgIG1hcC5zZXQoaWQsIG5leHQpO1xuICAgIGNvbnN0IHsgcm93IH0gPSBuZXh0O1xuICAgIGlmIChyb3cgJiYgcm93LmlzQ29ubmVjdGVkKSB7XG4gICAgICBpZiAobmV4dC50ZXh0ICE9PSBlbnRyeS50ZXh0KSB7XG4gICAgICAgIHJvdy5kYXRhc2V0LnJhd1RleHQgPSBuZXh0LnRleHQgfHwgXCJcIjtcbiAgICAgIH1cbiAgICAgIGlmIChuZXh0LnRpbWVzdGFtcCAhPT0gZW50cnkudGltZXN0YW1wKSB7XG4gICAgICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IG5leHQudGltZXN0YW1wIHx8IFwiXCI7XG4gICAgICB9XG4gICAgICBpZiAobmV4dC5yb2xlICYmIG5leHQucm9sZSAhPT0gZW50cnkucm9sZSkge1xuICAgICAgICByb3cuZGF0YXNldC5yb2xlID0gbmV4dC5yb2xlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbmV4dDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGNvbGxlY3QoKSB7XG4gICAgcmV0dXJuIG9yZGVyXG4gICAgICAubWFwKChpZCkgPT4ge1xuICAgICAgICBjb25zdCBlbnRyeSA9IG1hcC5nZXQoaWQpO1xuICAgICAgICBpZiAoIWVudHJ5KSB7XG4gICAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICByb2xlOiBlbnRyeS5yb2xlLFxuICAgICAgICAgIHRleHQ6IGVudHJ5LnRleHQsXG4gICAgICAgICAgdGltZXN0YW1wOiBlbnRyeS50aW1lc3RhbXAsXG4gICAgICAgICAgLi4uKGVudHJ5Lm1ldGFkYXRhICYmXG4gICAgICAgICAgICBPYmplY3Qua2V5cyhlbnRyeS5tZXRhZGF0YSkubGVuZ3RoID4gMCAmJiB7XG4gICAgICAgICAgICAgIG1ldGFkYXRhOiB7IC4uLmVudHJ5Lm1ldGFkYXRhIH0sXG4gICAgICAgICAgICB9KSxcbiAgICAgICAgfTtcbiAgICAgIH0pXG4gICAgICAuZmlsdGVyKEJvb2xlYW4pO1xuICB9XG5cbiAgZnVuY3Rpb24gY2xlYXIoKSB7XG4gICAgb3JkZXIubGVuZ3RoID0gMDtcbiAgICBtYXAuY2xlYXIoKTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgcmVnaXN0ZXIsXG4gICAgdXBkYXRlLFxuICAgIGNvbGxlY3QsXG4gICAgY2xlYXIsXG4gICAgb3JkZXIsXG4gICAgbWFwLFxuICAgIG1ha2VNZXNzYWdlSWQsXG4gIH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUVtaXR0ZXIoKSB7XG4gIGNvbnN0IGxpc3RlbmVycyA9IG5ldyBNYXAoKTtcblxuICBmdW5jdGlvbiBvbihldmVudCwgaGFuZGxlcikge1xuICAgIGlmICghbGlzdGVuZXJzLmhhcyhldmVudCkpIHtcbiAgICAgIGxpc3RlbmVycy5zZXQoZXZlbnQsIG5ldyBTZXQoKSk7XG4gICAgfVxuICAgIGxpc3RlbmVycy5nZXQoZXZlbnQpLmFkZChoYW5kbGVyKTtcbiAgICByZXR1cm4gKCkgPT4gb2ZmKGV2ZW50LCBoYW5kbGVyKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIG9mZihldmVudCwgaGFuZGxlcikge1xuICAgIGlmICghbGlzdGVuZXJzLmhhcyhldmVudCkpIHJldHVybjtcbiAgICBjb25zdCBidWNrZXQgPSBsaXN0ZW5lcnMuZ2V0KGV2ZW50KTtcbiAgICBidWNrZXQuZGVsZXRlKGhhbmRsZXIpO1xuICAgIGlmIChidWNrZXQuc2l6ZSA9PT0gMCkge1xuICAgICAgbGlzdGVuZXJzLmRlbGV0ZShldmVudCk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZW1pdChldmVudCwgcGF5bG9hZCkge1xuICAgIGlmICghbGlzdGVuZXJzLmhhcyhldmVudCkpIHJldHVybjtcbiAgICBsaXN0ZW5lcnMuZ2V0KGV2ZW50KS5mb3JFYWNoKChoYW5kbGVyKSA9PiB7XG4gICAgICB0cnkge1xuICAgICAgICBoYW5kbGVyKHBheWxvYWQpO1xuICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgIGNvbnNvbGUuZXJyb3IoXCJFbWl0dGVyIGhhbmRsZXIgZXJyb3JcIiwgZXJyKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIHJldHVybiB7IG9uLCBvZmYsIGVtaXQgfTtcbn1cbiIsICJleHBvcnQgZnVuY3Rpb24gZXNjYXBlSFRNTChzdHIpIHtcbiAgcmV0dXJuIFN0cmluZyhzdHIpLnJlcGxhY2UoXG4gICAgL1smPD5cIiddL2csXG4gICAgKGNoKSA9PlxuICAgICAgKHtcbiAgICAgICAgXCImXCI6IFwiJmFtcDtcIixcbiAgICAgICAgXCI8XCI6IFwiJmx0O1wiLFxuICAgICAgICBcIj5cIjogXCImZ3Q7XCIsXG4gICAgICAgICdcIic6IFwiJnF1b3Q7XCIsXG4gICAgICAgIFwiJ1wiOiBcIiYjMzk7XCIsXG4gICAgICB9KVtjaF0sXG4gICk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBodG1sVG9UZXh0KGh0bWwpIHtcbiAgY29uc3QgcGFyc2VyID0gbmV3IERPTVBhcnNlcigpO1xuICBjb25zdCBkb2MgPSBwYXJzZXIucGFyc2VGcm9tU3RyaW5nKGh0bWwsIFwidGV4dC9odG1sXCIpO1xuICByZXR1cm4gZG9jLmJvZHkudGV4dENvbnRlbnQgfHwgXCJcIjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGV4dHJhY3RCdWJibGVUZXh0KGJ1YmJsZSkge1xuICBjb25zdCBjbG9uZSA9IGJ1YmJsZS5jbG9uZU5vZGUodHJ1ZSk7XG4gIGNsb25lXG4gICAgLnF1ZXJ5U2VsZWN0b3JBbGwoXCIuY29weS1idG4sIC5jaGF0LW1ldGFcIilcbiAgICAuZm9yRWFjaCgobm9kZSkgPT4gbm9kZS5yZW1vdmUoKSk7XG4gIHJldHVybiBjbG9uZS50ZXh0Q29udGVudC50cmltKCk7XG59XG4iLCAiaW1wb3J0IHsgZXNjYXBlSFRNTCB9IGZyb20gXCIuLi91dGlscy9kb20uanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIHJlbmRlck1hcmtkb3duKHRleHQpIHtcbiAgaWYgKHRleHQgPT0gbnVsbCkge1xuICAgIHJldHVybiBcIlwiO1xuICB9XG4gIGNvbnN0IHZhbHVlID0gU3RyaW5nKHRleHQpO1xuICBjb25zdCBmYWxsYmFjayA9ICgpID0+IHtcbiAgICBjb25zdCBlc2NhcGVkID0gZXNjYXBlSFRNTCh2YWx1ZSk7XG4gICAgcmV0dXJuIGVzY2FwZWQucmVwbGFjZSgvXFxuL2csIFwiPGJyPlwiKTtcbiAgfTtcbiAgdHJ5IHtcbiAgICBpZiAod2luZG93Lm1hcmtlZCAmJiB0eXBlb2Ygd2luZG93Lm1hcmtlZC5wYXJzZSA9PT0gXCJmdW5jdGlvblwiKSB7XG4gICAgICBjb25zdCByZW5kZXJlZCA9IHdpbmRvdy5tYXJrZWQucGFyc2UodmFsdWUpO1xuICAgICAgaWYgKHdpbmRvdy5ET01QdXJpZnkgJiYgdHlwZW9mIHdpbmRvdy5ET01QdXJpZnkuc2FuaXRpemUgPT09IFwiZnVuY3Rpb25cIikge1xuICAgICAgICByZXR1cm4gd2luZG93LkRPTVB1cmlmeS5zYW5pdGl6ZShyZW5kZXJlZCwge1xuICAgICAgICAgIEFMTE9XX1VOS05PV05fUFJPVE9DT0xTOiBmYWxzZSxcbiAgICAgICAgICBVU0VfUFJPRklMRVM6IHsgaHRtbDogdHJ1ZSB9LFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICAgIC8vIEZhbGxiYWNrOiBlc2NhcGUgcmF3IHRleHQgYW5kIGRvIG1pbmltYWwgZm9ybWF0dGluZyB0byBhdm9pZCBYU1NcbiAgICAgIGNvbnN0IGVzY2FwZWQgPSBlc2NhcGVIVE1MKHZhbHVlKTtcbiAgICAgIHJldHVybiBlc2NhcGVkLnJlcGxhY2UoL1xcbi9nLCBcIjxicj5cIik7XG4gICAgfVxuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLndhcm4oXCJNYXJrZG93biByZW5kZXJpbmcgZmFpbGVkXCIsIGVycik7XG4gIH1cbiAgcmV0dXJuIGZhbGxiYWNrKCk7XG59XG4iLCAiaW1wb3J0IHsgY3JlYXRlRW1pdHRlciB9IGZyb20gXCIuLi91dGlscy9lbWl0dGVyLmpzXCI7XG5pbXBvcnQgeyBodG1sVG9UZXh0LCBleHRyYWN0QnViYmxlVGV4dCwgZXNjYXBlSFRNTCB9IGZyb20gXCIuLi91dGlscy9kb20uanNcIjtcbmltcG9ydCB7IHJlbmRlck1hcmtkb3duIH0gZnJvbSBcIi4uL3NlcnZpY2VzL21hcmtkb3duLmpzXCI7XG5pbXBvcnQgeyBmb3JtYXRUaW1lc3RhbXAsIG5vd0lTTyB9IGZyb20gXCIuLi91dGlscy90aW1lLmpzXCI7XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVDaGF0VWkoeyBlbGVtZW50cywgdGltZWxpbmVTdG9yZSB9KSB7XG4gIGNvbnN0IGVtaXR0ZXIgPSBjcmVhdGVFbWl0dGVyKCk7XG5cbiAgY29uc3Qgc2VuZElkbGVNYXJrdXAgPSBlbGVtZW50cy5zZW5kID8gZWxlbWVudHMuc2VuZC5pbm5lckhUTUwgOiBcIlwiO1xuICBjb25zdCBzZW5kSWRsZUxhYmVsID1cbiAgICAoZWxlbWVudHMuc2VuZCAmJiBlbGVtZW50cy5zZW5kLmdldEF0dHJpYnV0ZShcImRhdGEtaWRsZS1sYWJlbFwiKSkgfHxcbiAgICAoZWxlbWVudHMuc2VuZCA/IGVsZW1lbnRzLnNlbmQudGV4dENvbnRlbnQudHJpbSgpIDogXCJFbnZveWVyXCIpO1xuICBjb25zdCBzZW5kQnVzeU1hcmt1cCA9XG4gICAgJzxzcGFuIGNsYXNzPVwic3Bpbm5lci1ib3JkZXIgc3Bpbm5lci1ib3JkZXItc20gbWUtMVwiIHJvbGU9XCJzdGF0dXNcIiBhcmlhLWhpZGRlbj1cInRydWVcIj48L3NwYW4+RW52b2lcdTIwMjYnO1xuICBjb25zdCBTVVBQT1JURURfVE9ORVMgPSBbXCJtdXRlZFwiLCBcImluZm9cIiwgXCJzdWNjZXNzXCIsIFwiZGFuZ2VyXCIsIFwid2FybmluZ1wiXTtcbiAgY29uc3QgY29tcG9zZXJTdGF0dXNEZWZhdWx0ID1cbiAgICAoZWxlbWVudHMuY29tcG9zZXJTdGF0dXMgJiYgZWxlbWVudHMuY29tcG9zZXJTdGF0dXMudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiQXBwdXlleiBzdXIgQ3RybCtFbnRyXHUwMEU5ZSBwb3VyIGVudm95ZXIgcmFwaWRlbWVudC5cIjtcbiAgY29uc3QgZmlsdGVySGludERlZmF1bHQgPVxuICAgIChlbGVtZW50cy5maWx0ZXJIaW50ICYmIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiVXRpbGlzZXogbGUgZmlsdHJlIHBvdXIgbGltaXRlciBsJ2hpc3RvcmlxdWUuIEFwcHV5ZXogc3VyIFx1MDBDOWNoYXAgcG91ciBlZmZhY2VyLlwiO1xuICBjb25zdCB2b2ljZVN0YXR1c0RlZmF1bHQgPVxuICAgIChlbGVtZW50cy52b2ljZVN0YXR1cyAmJiBlbGVtZW50cy52b2ljZVN0YXR1cy50ZXh0Q29udGVudC50cmltKCkpIHx8XG4gICAgXCJWXHUwMEU5cmlmaWNhdGlvbiBkZXMgY2FwYWNpdFx1MDBFOXMgdm9jYWxlc1x1MjAyNlwiO1xuICBjb25zdCBwcm9tcHRNYXggPSBOdW1iZXIoZWxlbWVudHMucHJvbXB0Py5nZXRBdHRyaWJ1dGUoXCJtYXhsZW5ndGhcIikpIHx8IG51bGw7XG4gIGNvbnN0IHByZWZlcnNSZWR1Y2VkTW90aW9uID1cbiAgICB3aW5kb3cubWF0Y2hNZWRpYSAmJlxuICAgIHdpbmRvdy5tYXRjaE1lZGlhKFwiKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246IHJlZHVjZSlcIikubWF0Y2hlcztcbiAgY29uc3QgU0NST0xMX1RIUkVTSE9MRCA9IDE0MDtcbiAgY29uc3QgUFJPTVBUX01BWF9IRUlHSFQgPSAzMjA7XG5cbiAgY29uc3QgZGlhZ25vc3RpY3MgPSB7XG4gICAgY29ubmVjdGVkQXQ6IG51bGwsXG4gICAgbGFzdE1lc3NhZ2VBdDogbnVsbCxcbiAgICBsYXRlbmN5TXM6IG51bGwsXG4gIH07XG5cbiAgY29uc3Qgc3RhdGUgPSB7XG4gICAgcmVzZXRTdGF0dXNUaW1lcjogbnVsbCxcbiAgICBoaWRlU2Nyb2xsVGltZXI6IG51bGwsXG4gICAgdm9pY2VTdGF0dXNUaW1lcjogbnVsbCxcbiAgICBhY3RpdmVGaWx0ZXI6IFwiXCIsXG4gICAgaGlzdG9yeUJvb3RzdHJhcHBlZDogZWxlbWVudHMudHJhbnNjcmlwdC5jaGlsZEVsZW1lbnRDb3VudCA+IDAsXG4gICAgYm9vdHN0cmFwcGluZzogZmFsc2UsXG4gICAgc3RyZWFtUm93OiBudWxsLFxuICAgIHN0cmVhbUJ1ZjogXCJcIixcbiAgICBzdHJlYW1NZXNzYWdlSWQ6IG51bGwsXG4gIH07XG5cbiAgY29uc3Qgc3RhdHVzTGFiZWxzID0ge1xuICAgIG9mZmxpbmU6IFwiSG9ycyBsaWduZVwiLFxuICAgIGNvbm5lY3Rpbmc6IFwiQ29ubmV4aW9uXHUyMDI2XCIsXG4gICAgb25saW5lOiBcIkVuIGxpZ25lXCIsXG4gICAgZXJyb3I6IFwiRXJyZXVyXCIsXG4gIH07XG5cbiAgZnVuY3Rpb24gb24oZXZlbnQsIGhhbmRsZXIpIHtcbiAgICByZXR1cm4gZW1pdHRlci5vbihldmVudCwgaGFuZGxlcik7XG4gIH1cblxuICBmdW5jdGlvbiBlbWl0KGV2ZW50LCBwYXlsb2FkKSB7XG4gICAgZW1pdHRlci5lbWl0KGV2ZW50LCBwYXlsb2FkKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldEJ1c3koYnVzeSkge1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuc2V0QXR0cmlidXRlKFwiYXJpYS1idXN5XCIsIGJ1c3kgPyBcInRydWVcIiA6IFwiZmFsc2VcIik7XG4gICAgaWYgKGVsZW1lbnRzLnNlbmQpIHtcbiAgICAgIGVsZW1lbnRzLnNlbmQuZGlzYWJsZWQgPSBCb29sZWFuKGJ1c3kpO1xuICAgICAgZWxlbWVudHMuc2VuZC5zZXRBdHRyaWJ1dGUoXCJhcmlhLWJ1c3lcIiwgYnVzeSA/IFwidHJ1ZVwiIDogXCJmYWxzZVwiKTtcbiAgICAgIGlmIChidXN5KSB7XG4gICAgICAgIGVsZW1lbnRzLnNlbmQuaW5uZXJIVE1MID0gc2VuZEJ1c3lNYXJrdXA7XG4gICAgICB9IGVsc2UgaWYgKHNlbmRJZGxlTWFya3VwKSB7XG4gICAgICAgIGVsZW1lbnRzLnNlbmQuaW5uZXJIVE1MID0gc2VuZElkbGVNYXJrdXA7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5zZW5kLnRleHRDb250ZW50ID0gc2VuZElkbGVMYWJlbDtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBoaWRlRXJyb3IoKSB7XG4gICAgaWYgKCFlbGVtZW50cy5lcnJvckFsZXJ0KSByZXR1cm47XG4gICAgZWxlbWVudHMuZXJyb3JBbGVydC5jbGFzc0xpc3QuYWRkKFwiZC1ub25lXCIpO1xuICAgIGlmIChlbGVtZW50cy5lcnJvck1lc3NhZ2UpIHtcbiAgICAgIGVsZW1lbnRzLmVycm9yTWVzc2FnZS50ZXh0Q29udGVudCA9IFwiXCI7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2hvd0Vycm9yKG1lc3NhZ2UpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmVycm9yQWxlcnQgfHwgIWVsZW1lbnRzLmVycm9yTWVzc2FnZSkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLmVycm9yTWVzc2FnZS50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgZWxlbWVudHMuZXJyb3JBbGVydC5jbGFzc0xpc3QucmVtb3ZlKFwiZC1ub25lXCIpO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Q29tcG9zZXJTdGF0dXMobWVzc2FnZSwgdG9uZSA9IFwibXV0ZWRcIikge1xuICAgIGlmICghZWxlbWVudHMuY29tcG9zZXJTdGF0dXMpIHJldHVybjtcbiAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgU1VQUE9SVEVEX1RPTkVTLmZvckVhY2goKHQpID0+XG4gICAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy5jbGFzc0xpc3QucmVtb3ZlKGB0ZXh0LSR7dH1gKSxcbiAgICApO1xuICAgIGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzLmNsYXNzTGlzdC5hZGQoYHRleHQtJHt0b25lfWApO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCkge1xuICAgIHNldENvbXBvc2VyU3RhdHVzKGNvbXBvc2VyU3RhdHVzRGVmYXVsdCwgXCJtdXRlZFwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlQ29tcG9zZXJJZGxlKGRlbGF5ID0gMzUwMCkge1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHN0YXRlLnJlc2V0U3RhdHVzVGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICB9LCBkZWxheSk7XG4gIH1cblxuICBmdW5jdGlvbiBzZXRWb2ljZVN0YXR1cyhtZXNzYWdlLCB0b25lID0gXCJtdXRlZFwiKSB7XG4gICAgaWYgKCFlbGVtZW50cy52b2ljZVN0YXR1cykgcmV0dXJuO1xuICAgIGlmIChzdGF0ZS52b2ljZVN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUudm9pY2VTdGF0dXNUaW1lcik7XG4gICAgICBzdGF0ZS52b2ljZVN0YXR1c1RpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgZWxlbWVudHMudm9pY2VTdGF0dXMudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIFNVUFBPUlRFRF9UT05FUy5mb3JFYWNoKCh0KSA9PlxuICAgICAgZWxlbWVudHMudm9pY2VTdGF0dXMuY2xhc3NMaXN0LnJlbW92ZShgdGV4dC0ke3R9YCksXG4gICAgKTtcbiAgICBlbGVtZW50cy52b2ljZVN0YXR1cy5jbGFzc0xpc3QuYWRkKGB0ZXh0LSR7dG9uZX1gKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKGRlbGF5ID0gNDAwMCkge1xuICAgIGlmICghZWxlbWVudHMudm9pY2VTdGF0dXMpIHJldHVybjtcbiAgICBpZiAoc3RhdGUudm9pY2VTdGF0dXNUaW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHN0YXRlLnZvaWNlU3RhdHVzVGltZXIpO1xuICAgIH1cbiAgICBzdGF0ZS52b2ljZVN0YXR1c1RpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgc2V0Vm9pY2VTdGF0dXModm9pY2VTdGF0dXNEZWZhdWx0LCBcIm11dGVkXCIpO1xuICAgICAgc3RhdGUudm9pY2VTdGF0dXNUaW1lciA9IG51bGw7XG4gICAgfSwgZGVsYXkpO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VBdmFpbGFiaWxpdHkoeyByZWNvZ25pdGlvbiA9IGZhbHNlLCBzeW50aGVzaXMgPSBmYWxzZSB9ID0ge30pIHtcbiAgICBpZiAoZWxlbWVudHMudm9pY2VDb250cm9scykge1xuICAgICAgZWxlbWVudHMudm9pY2VDb250cm9scy5jbGFzc0xpc3QudG9nZ2xlKFxuICAgICAgICBcImQtbm9uZVwiLFxuICAgICAgICAhcmVjb2duaXRpb24gJiYgIXN5bnRoZXNpcyxcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy52b2ljZVJlY29nbml0aW9uR3JvdXApIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlUmVjb2duaXRpb25Hcm91cC5jbGFzc0xpc3QudG9nZ2xlKFxuICAgICAgICBcImQtbm9uZVwiLFxuICAgICAgICAhcmVjb2duaXRpb24sXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VUb2dnbGUpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLmRpc2FibGVkID0gIXJlY29nbml0aW9uO1xuICAgICAgZWxlbWVudHMudm9pY2VUb2dnbGUuc2V0QXR0cmlidXRlKFxuICAgICAgICBcInRpdGxlXCIsXG4gICAgICAgIHJlY29nbml0aW9uXG4gICAgICAgICAgPyBcIkFjdGl2ZXIgb3UgZFx1MDBFOXNhY3RpdmVyIGxhIGRpY3RcdTAwRTllIHZvY2FsZS5cIlxuICAgICAgICAgIDogXCJEaWN0XHUwMEU5ZSB2b2NhbGUgaW5kaXNwb25pYmxlIGRhbnMgY2UgbmF2aWdhdGV1ci5cIixcbiAgICAgICk7XG4gICAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS5zZXRBdHRyaWJ1dGUoXCJhcmlhLXByZXNzZWRcIiwgXCJmYWxzZVwiKTtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLmNsYXNzTGlzdC5yZW1vdmUoXCJidG4tZGFuZ2VyXCIpO1xuICAgICAgZWxlbWVudHMudm9pY2VUb2dnbGUuY2xhc3NMaXN0LmFkZChcImJ0bi1vdXRsaW5lLXNlY29uZGFyeVwiKTtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLnRleHRDb250ZW50ID0gXCJcdUQ4M0NcdURGOTlcdUZFMEYgQWN0aXZlciBsYSBkaWN0XHUwMEU5ZVwiO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VBdXRvU2VuZCkge1xuICAgICAgZWxlbWVudHMudm9pY2VBdXRvU2VuZC5kaXNhYmxlZCA9ICFyZWNvZ25pdGlvbjtcbiAgICB9XG4gICAgaWYgKCFyZWNvZ25pdGlvbikge1xuICAgICAgc2V0Vm9pY2VUcmFuc2NyaXB0KFwiXCIsIHsgc3RhdGU6IFwiaWRsZVwiIH0pO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VTeW50aGVzaXNHcm91cCkge1xuICAgICAgZWxlbWVudHMudm9pY2VTeW50aGVzaXNHcm91cC5jbGFzc0xpc3QudG9nZ2xlKFwiZC1ub25lXCIsICFzeW50aGVzaXMpO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VQbGF5YmFjay5kaXNhYmxlZCA9ICFzeW50aGVzaXM7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy52b2ljZVN0b3BQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2suZGlzYWJsZWQgPSAhc3ludGhlc2lzO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdCkge1xuICAgICAgZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdC5kaXNhYmxlZCA9ICFzeW50aGVzaXM7XG4gICAgICBpZiAoIXN5bnRoZXNpcykge1xuICAgICAgICBlbGVtZW50cy52b2ljZVZvaWNlU2VsZWN0LmlubmVySFRNTCA9IFwiXCI7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VMaXN0ZW5pbmcobGlzdGVuaW5nKSB7XG4gICAgaWYgKCFlbGVtZW50cy52b2ljZVRvZ2dsZSkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLnNldEF0dHJpYnV0ZShcImFyaWEtcHJlc3NlZFwiLCBsaXN0ZW5pbmcgPyBcInRydWVcIiA6IFwiZmFsc2VcIik7XG4gICAgZWxlbWVudHMudm9pY2VUb2dnbGUuY2xhc3NMaXN0LnRvZ2dsZShcImJ0bi1kYW5nZXJcIiwgbGlzdGVuaW5nKTtcbiAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS5jbGFzc0xpc3QudG9nZ2xlKFwiYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCIsICFsaXN0ZW5pbmcpO1xuICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLnRleHRDb250ZW50ID0gbGlzdGVuaW5nXG4gICAgICA/IFwiXHVEODNEXHVERUQxIEFyclx1MDBFQXRlciBsJ1x1MDBFOWNvdXRlXCJcbiAgICAgIDogXCJcdUQ4M0NcdURGOTlcdUZFMEYgQWN0aXZlciBsYSBkaWN0XHUwMEU5ZVwiO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VUcmFuc2NyaXB0KHRleHQsIG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghZWxlbWVudHMudm9pY2VUcmFuc2NyaXB0KSByZXR1cm47XG4gICAgY29uc3QgdmFsdWUgPSB0ZXh0IHx8IFwiXCI7XG4gICAgY29uc3Qgc3RhdGVWYWx1ZSA9IG9wdGlvbnMuc3RhdGUgfHwgKHZhbHVlID8gXCJmaW5hbFwiIDogXCJpZGxlXCIpO1xuICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC50ZXh0Q29udGVudCA9IHZhbHVlO1xuICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC5kYXRhc2V0LnN0YXRlID0gc3RhdGVWYWx1ZTtcbiAgICBpZiAoIXZhbHVlICYmIG9wdGlvbnMucGxhY2Vob2xkZXIpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC50ZXh0Q29udGVudCA9IG9wdGlvbnMucGxhY2Vob2xkZXI7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VQcmVmZXJlbmNlcyhwcmVmcyA9IHt9KSB7XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQuY2hlY2tlZCA9IEJvb2xlYW4ocHJlZnMuYXV0b1NlbmQpO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VQbGF5YmFjay5jaGVja2VkID0gQm9vbGVhbihwcmVmcy5wbGF5YmFjayk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VTcGVha2luZyhhY3RpdmUpIHtcbiAgICBpZiAoZWxlbWVudHMudm9pY2VTcGVha2luZ0luZGljYXRvcikge1xuICAgICAgZWxlbWVudHMudm9pY2VTcGVha2luZ0luZGljYXRvci5jbGFzc0xpc3QudG9nZ2xlKFwiZC1ub25lXCIsICFhY3RpdmUpO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2spIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlU3RvcFBsYXliYWNrLmRpc2FibGVkID0gIWFjdGl2ZTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBzZXRWb2ljZVZvaWNlT3B0aW9ucyh2b2ljZXMgPSBbXSwgc2VsZWN0ZWRVcmkgPSBudWxsKSB7XG4gICAgaWYgKCFlbGVtZW50cy52b2ljZVZvaWNlU2VsZWN0KSByZXR1cm47XG4gICAgY29uc3Qgc2VsZWN0ID0gZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdDtcbiAgICBjb25zdCBmcmFnID0gZG9jdW1lbnQuY3JlYXRlRG9jdW1lbnRGcmFnbWVudCgpO1xuICAgIGNvbnN0IHBsYWNlaG9sZGVyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcIm9wdGlvblwiKTtcbiAgICBwbGFjZWhvbGRlci52YWx1ZSA9IFwiXCI7XG4gICAgcGxhY2Vob2xkZXIudGV4dENvbnRlbnQgPSB2b2ljZXMubGVuZ3RoXG4gICAgICA/IFwiVm9peCBwYXIgZFx1MDBFOWZhdXQgZHUgc3lzdFx1MDBFOG1lXCJcbiAgICAgIDogXCJBdWN1bmUgdm9peCBkaXNwb25pYmxlXCI7XG4gICAgZnJhZy5hcHBlbmRDaGlsZChwbGFjZWhvbGRlcik7XG4gICAgdm9pY2VzLmZvckVhY2goKHZvaWNlKSA9PiB7XG4gICAgICBjb25zdCBvcHRpb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwib3B0aW9uXCIpO1xuICAgICAgb3B0aW9uLnZhbHVlID0gdm9pY2Uudm9pY2VVUkkgfHwgdm9pY2UubmFtZSB8fCBcIlwiO1xuICAgICAgY29uc3QgYml0cyA9IFt2b2ljZS5uYW1lIHx8IHZvaWNlLnZvaWNlVVJJIHx8IFwiVm9peFwiXTtcbiAgICAgIGlmICh2b2ljZS5sYW5nKSB7XG4gICAgICAgIGJpdHMucHVzaChgKCR7dm9pY2UubGFuZ30pYCk7XG4gICAgICB9XG4gICAgICBpZiAodm9pY2UuZGVmYXVsdCkge1xuICAgICAgICBiaXRzLnB1c2goXCJcdTIwMjIgZFx1MDBFOWZhdXRcIik7XG4gICAgICB9XG4gICAgICBvcHRpb24udGV4dENvbnRlbnQgPSBiaXRzLmpvaW4oXCIgXCIpO1xuICAgICAgZnJhZy5hcHBlbmRDaGlsZChvcHRpb24pO1xuICAgIH0pO1xuICAgIHNlbGVjdC5pbm5lckhUTUwgPSBcIlwiO1xuICAgIHNlbGVjdC5hcHBlbmRDaGlsZChmcmFnKTtcbiAgICBpZiAoc2VsZWN0ZWRVcmkpIHtcbiAgICAgIGxldCBtYXRjaGVkID0gZmFsc2U7XG4gICAgICBBcnJheS5mcm9tKHNlbGVjdC5vcHRpb25zKS5mb3JFYWNoKChvcHRpb24pID0+IHtcbiAgICAgICAgaWYgKCFtYXRjaGVkICYmIG9wdGlvbi52YWx1ZSA9PT0gc2VsZWN0ZWRVcmkpIHtcbiAgICAgICAgICBtYXRjaGVkID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBzZWxlY3QudmFsdWUgPSBtYXRjaGVkID8gc2VsZWN0ZWRVcmkgOiBcIlwiO1xuICAgIH0gZWxzZSB7XG4gICAgICBzZWxlY3QudmFsdWUgPSBcIlwiO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZVByb21wdE1ldHJpY3MoKSB7XG4gICAgaWYgKCFlbGVtZW50cy5wcm9tcHRDb3VudCB8fCAhZWxlbWVudHMucHJvbXB0KSByZXR1cm47XG4gICAgY29uc3QgdmFsdWUgPSBlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIjtcbiAgICBpZiAocHJvbXB0TWF4KSB7XG4gICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC50ZXh0Q29udGVudCA9IGAke3ZhbHVlLmxlbmd0aH0gLyAke3Byb21wdE1heH1gO1xuICAgIH0gZWxzZSB7XG4gICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC50ZXh0Q29udGVudCA9IGAke3ZhbHVlLmxlbmd0aH1gO1xuICAgIH1cbiAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QucmVtb3ZlKFwidGV4dC13YXJuaW5nXCIsIFwidGV4dC1kYW5nZXJcIik7XG4gICAgaWYgKHByb21wdE1heCkge1xuICAgICAgY29uc3QgcmVtYWluaW5nID0gcHJvbXB0TWF4IC0gdmFsdWUubGVuZ3RoO1xuICAgICAgaWYgKHJlbWFpbmluZyA8PSA1KSB7XG4gICAgICAgIGVsZW1lbnRzLnByb21wdENvdW50LmNsYXNzTGlzdC5hZGQoXCJ0ZXh0LWRhbmdlclwiKTtcbiAgICAgIH0gZWxzZSBpZiAocmVtYWluaW5nIDw9IDIwKSB7XG4gICAgICAgIGVsZW1lbnRzLnByb21wdENvdW50LmNsYXNzTGlzdC5hZGQoXCJ0ZXh0LXdhcm5pbmdcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gYXV0b3NpemVQcm9tcHQoKSB7XG4gICAgaWYgKCFlbGVtZW50cy5wcm9tcHQpIHJldHVybjtcbiAgICBlbGVtZW50cy5wcm9tcHQuc3R5bGUuaGVpZ2h0ID0gXCJhdXRvXCI7XG4gICAgY29uc3QgbmV4dEhlaWdodCA9IE1hdGgubWluKFxuICAgICAgZWxlbWVudHMucHJvbXB0LnNjcm9sbEhlaWdodCxcbiAgICAgIFBST01QVF9NQVhfSEVJR0hULFxuICAgICk7XG4gICAgZWxlbWVudHMucHJvbXB0LnN0eWxlLmhlaWdodCA9IGAke25leHRIZWlnaHR9cHhgO1xuICB9XG5cbiAgZnVuY3Rpb24gaXNBdEJvdHRvbSgpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybiB0cnVlO1xuICAgIGNvbnN0IGRpc3RhbmNlID1cbiAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuc2Nyb2xsSGVpZ2h0IC1cbiAgICAgIChlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbFRvcCArIGVsZW1lbnRzLnRyYW5zY3JpcHQuY2xpZW50SGVpZ2h0KTtcbiAgICByZXR1cm4gZGlzdGFuY2UgPD0gU0NST0xMX1RIUkVTSE9MRDtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjcm9sbFRvQm90dG9tKG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghZWxlbWVudHMudHJhbnNjcmlwdCkgcmV0dXJuO1xuICAgIGNvbnN0IHNtb290aCA9IG9wdGlvbnMuc21vb3RoICE9PSBmYWxzZSAmJiAhcHJlZmVyc1JlZHVjZWRNb3Rpb247XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5zY3JvbGxUbyh7XG4gICAgICB0b3A6IGVsZW1lbnRzLnRyYW5zY3JpcHQuc2Nyb2xsSGVpZ2h0LFxuICAgICAgYmVoYXZpb3I6IHNtb290aCA/IFwic21vb3RoXCIgOiBcImF1dG9cIixcbiAgICB9KTtcbiAgICBoaWRlU2Nyb2xsQnV0dG9uKCk7XG4gIH1cblxuICBmdW5jdGlvbiBzaG93U2Nyb2xsQnV0dG9uKCkge1xuICAgIGlmICghZWxlbWVudHMuc2Nyb2xsQm90dG9tKSByZXR1cm47XG4gICAgaWYgKHN0YXRlLmhpZGVTY3JvbGxUaW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHN0YXRlLmhpZGVTY3JvbGxUaW1lcik7XG4gICAgICBzdGF0ZS5oaWRlU2Nyb2xsVGltZXIgPSBudWxsO1xuICAgIH1cbiAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uY2xhc3NMaXN0LnJlbW92ZShcImQtbm9uZVwiKTtcbiAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uY2xhc3NMaXN0LmFkZChcImlzLXZpc2libGVcIik7XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLnNldEF0dHJpYnV0ZShcImFyaWEtaGlkZGVuXCIsIFwiZmFsc2VcIik7XG4gIH1cblxuICBmdW5jdGlvbiBoaWRlU2Nyb2xsQnV0dG9uKCkge1xuICAgIGlmICghZWxlbWVudHMuc2Nyb2xsQm90dG9tKSByZXR1cm47XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5yZW1vdmUoXCJpcy12aXNpYmxlXCIpO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5zZXRBdHRyaWJ1dGUoXCJhcmlhLWhpZGRlblwiLCBcInRydWVcIik7XG4gICAgc3RhdGUuaGlkZVNjcm9sbFRpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgaWYgKGVsZW1lbnRzLnNjcm9sbEJvdHRvbSkge1xuICAgICAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICAgIH1cbiAgICB9LCAyMDApO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gaGFuZGxlQ29weShidWJibGUpIHtcbiAgICBjb25zdCB0ZXh0ID0gZXh0cmFjdEJ1YmJsZVRleHQoYnViYmxlKTtcbiAgICBpZiAoIXRleHQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgdHJ5IHtcbiAgICAgIGlmIChuYXZpZ2F0b3IuY2xpcGJvYXJkICYmIG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KSB7XG4gICAgICAgIGF3YWl0IG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KHRleHQpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgY29uc3QgdGV4dGFyZWEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidGV4dGFyZWFcIik7XG4gICAgICAgIHRleHRhcmVhLnZhbHVlID0gdGV4dDtcbiAgICAgICAgdGV4dGFyZWEuc2V0QXR0cmlidXRlKFwicmVhZG9ubHlcIiwgXCJyZWFkb25seVwiKTtcbiAgICAgICAgdGV4dGFyZWEuc3R5bGUucG9zaXRpb24gPSBcImFic29sdXRlXCI7XG4gICAgICAgIHRleHRhcmVhLnN0eWxlLmxlZnQgPSBcIi05OTk5cHhcIjtcbiAgICAgICAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZCh0ZXh0YXJlYSk7XG4gICAgICAgIHRleHRhcmVhLnNlbGVjdCgpO1xuICAgICAgICBkb2N1bWVudC5leGVjQ29tbWFuZChcImNvcHlcIik7XG4gICAgICAgIGRvY3VtZW50LmJvZHkucmVtb3ZlQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgfVxuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29udGVudSBjb3BpXHUwMEU5IGRhbnMgbGUgcHJlc3NlLXBhcGllcnMuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIkNvcHkgZmFpbGVkXCIsIGVycik7XG4gICAgICBhbm5vdW5jZUNvbm5lY3Rpb24oXCJJbXBvc3NpYmxlIGRlIGNvcGllciBsZSBtZXNzYWdlLlwiLCBcImRhbmdlclwiKTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBkZWNvcmF0ZVJvdyhyb3csIHJvbGUpIHtcbiAgICBjb25zdCBidWJibGUgPSByb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICBpZiAoIWJ1YmJsZSkgcmV0dXJuO1xuICAgIGlmIChyb2xlID09PSBcImFzc2lzdGFudFwiIHx8IHJvbGUgPT09IFwidXNlclwiKSB7XG4gICAgICBidWJibGUuY2xhc3NMaXN0LmFkZChcImhhcy10b29sc1wiKTtcbiAgICAgIGJ1YmJsZS5xdWVyeVNlbGVjdG9yQWxsKFwiLmNvcHktYnRuXCIpLmZvckVhY2goKGJ0bikgPT4gYnRuLnJlbW92ZSgpKTtcbiAgICAgIGNvbnN0IGNvcHlCdG4gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiYnV0dG9uXCIpO1xuICAgICAgY29weUJ0bi50eXBlID0gXCJidXR0b25cIjtcbiAgICAgIGNvcHlCdG4uY2xhc3NOYW1lID0gXCJjb3B5LWJ0blwiO1xuICAgICAgY29weUJ0bi5pbm5lckhUTUwgPVxuICAgICAgICAnPHNwYW4gYXJpYS1oaWRkZW49XCJ0cnVlXCI+XHUyOUM5PC9zcGFuPjxzcGFuIGNsYXNzPVwidmlzdWFsbHktaGlkZGVuXCI+Q29waWVyIGxlIG1lc3NhZ2U8L3NwYW4+JztcbiAgICAgIGNvcHlCdG4uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IGhhbmRsZUNvcHkoYnViYmxlKSk7XG4gICAgICBidWJibGUuYXBwZW5kQ2hpbGQoY29weUJ0bik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gaGlnaGxpZ2h0Um93KHJvdywgcm9sZSkge1xuICAgIGlmICghcm93IHx8IHN0YXRlLmJvb3RzdHJhcHBpbmcgfHwgcm9sZSA9PT0gXCJzeXN0ZW1cIikge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICByb3cuY2xhc3NMaXN0LmFkZChcImNoYXQtcm93LWhpZ2hsaWdodFwiKTtcbiAgICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICByb3cuY2xhc3NMaXN0LnJlbW92ZShcImNoYXQtcm93LWhpZ2hsaWdodFwiKTtcbiAgICB9LCA2MDApO1xuICB9XG5cbiAgZnVuY3Rpb24gbGluZShyb2xlLCBodG1sLCBvcHRpb25zID0ge30pIHtcbiAgICBjb25zdCBzaG91bGRTdGljayA9IGlzQXRCb3R0b20oKTtcbiAgICBjb25zdCByb3cgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiZGl2XCIpO1xuICAgIHJvdy5jbGFzc05hbWUgPSBgY2hhdC1yb3cgY2hhdC0ke3JvbGV9YDtcbiAgICByb3cuaW5uZXJIVE1MID0gaHRtbDtcbiAgICByb3cuZGF0YXNldC5yb2xlID0gcm9sZTtcbiAgICByb3cuZGF0YXNldC5yYXdUZXh0ID0gb3B0aW9ucy5yYXdUZXh0IHx8IFwiXCI7XG4gICAgcm93LmRhdGFzZXQudGltZXN0YW1wID0gb3B0aW9ucy50aW1lc3RhbXAgfHwgXCJcIjtcbiAgICBlbGVtZW50cy50cmFuc2NyaXB0LmFwcGVuZENoaWxkKHJvdyk7XG4gICAgZGVjb3JhdGVSb3cocm93LCByb2xlKTtcbiAgICBpZiAob3B0aW9ucy5yZWdpc3RlciAhPT0gZmFsc2UpIHtcbiAgICAgIGNvbnN0IHRzID0gb3B0aW9ucy50aW1lc3RhbXAgfHwgbm93SVNPKCk7XG4gICAgICBjb25zdCB0ZXh0ID1cbiAgICAgICAgb3B0aW9ucy5yYXdUZXh0ICYmIG9wdGlvbnMucmF3VGV4dC5sZW5ndGggPiAwXG4gICAgICAgICAgPyBvcHRpb25zLnJhd1RleHRcbiAgICAgICAgICA6IGh0bWxUb1RleHQoaHRtbCk7XG4gICAgICBjb25zdCBpZCA9IHRpbWVsaW5lU3RvcmUucmVnaXN0ZXIoe1xuICAgICAgICBpZDogb3B0aW9ucy5tZXNzYWdlSWQsXG4gICAgICAgIHJvbGUsXG4gICAgICAgIHRleHQsXG4gICAgICAgIHRpbWVzdGFtcDogdHMsXG4gICAgICAgIHJvdyxcbiAgICAgICAgbWV0YWRhdGE6IG9wdGlvbnMubWV0YWRhdGEgfHwge30sXG4gICAgICB9KTtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IGlkO1xuICAgIH0gZWxzZSBpZiAob3B0aW9ucy5tZXNzYWdlSWQpIHtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IG9wdGlvbnMubWVzc2FnZUlkO1xuICAgIH0gZWxzZSBpZiAoIXJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCkge1xuICAgICAgcm93LmRhdGFzZXQubWVzc2FnZUlkID0gdGltZWxpbmVTdG9yZS5tYWtlTWVzc2FnZUlkKCk7XG4gICAgfVxuICAgIGlmIChzaG91bGRTdGljaykge1xuICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6ICFzdGF0ZS5ib290c3RyYXBwaW5nIH0pO1xuICAgIH0gZWxzZSB7XG4gICAgICBzaG93U2Nyb2xsQnV0dG9uKCk7XG4gICAgfVxuICAgIGhpZ2hsaWdodFJvdyhyb3csIHJvbGUpO1xuICAgIGlmIChzdGF0ZS5hY3RpdmVGaWx0ZXIpIHtcbiAgICAgIGFwcGx5VHJhbnNjcmlwdEZpbHRlcihzdGF0ZS5hY3RpdmVGaWx0ZXIsIHsgcHJlc2VydmVJbnB1dDogdHJ1ZSB9KTtcbiAgICB9XG4gICAgcmV0dXJuIHJvdztcbiAgfVxuXG4gIGZ1bmN0aW9uIGJ1aWxkQnViYmxlKHtcbiAgICB0ZXh0LFxuICAgIHRpbWVzdGFtcCxcbiAgICB2YXJpYW50LFxuICAgIG1ldGFTdWZmaXgsXG4gICAgYWxsb3dNYXJrZG93biA9IHRydWUsXG4gIH0pIHtcbiAgICBjb25zdCBjbGFzc2VzID0gW1wiY2hhdC1idWJibGVcIl07XG4gICAgaWYgKHZhcmlhbnQpIHtcbiAgICAgIGNsYXNzZXMucHVzaChgY2hhdC1idWJibGUtJHt2YXJpYW50fWApO1xuICAgIH1cbiAgICBjb25zdCBjb250ZW50ID0gYWxsb3dNYXJrZG93blxuICAgICAgPyByZW5kZXJNYXJrZG93bih0ZXh0KVxuICAgICAgOiBlc2NhcGVIVE1MKFN0cmluZyh0ZXh0KSk7XG4gICAgY29uc3QgbWV0YUJpdHMgPSBbXTtcbiAgICBpZiAodGltZXN0YW1wKSB7XG4gICAgICBtZXRhQml0cy5wdXNoKGZvcm1hdFRpbWVzdGFtcCh0aW1lc3RhbXApKTtcbiAgICB9XG4gICAgaWYgKG1ldGFTdWZmaXgpIHtcbiAgICAgIG1ldGFCaXRzLnB1c2gobWV0YVN1ZmZpeCk7XG4gICAgfVxuICAgIGNvbnN0IG1ldGFIdG1sID1cbiAgICAgIG1ldGFCaXRzLmxlbmd0aCA+IDBcbiAgICAgICAgPyBgPGRpdiBjbGFzcz1cImNoYXQtbWV0YVwiPiR7ZXNjYXBlSFRNTChtZXRhQml0cy5qb2luKFwiIFx1MjAyMiBcIikpfTwvZGl2PmBcbiAgICAgICAgOiBcIlwiO1xuICAgIHJldHVybiBgPGRpdiBjbGFzcz1cIiR7Y2xhc3Nlcy5qb2luKFwiIFwiKX1cIj4ke2NvbnRlbnR9JHttZXRhSHRtbH08L2Rpdj5gO1xuICB9XG5cbiAgZnVuY3Rpb24gYXBwZW5kTWVzc2FnZShyb2xlLCB0ZXh0LCBvcHRpb25zID0ge30pIHtcbiAgICBjb25zdCB7XG4gICAgICB0aW1lc3RhbXAsXG4gICAgICB2YXJpYW50LFxuICAgICAgbWV0YVN1ZmZpeCxcbiAgICAgIGFsbG93TWFya2Rvd24gPSB0cnVlLFxuICAgICAgbWVzc2FnZUlkLFxuICAgICAgcmVnaXN0ZXIgPSB0cnVlLFxuICAgICAgbWV0YWRhdGEsXG4gICAgfSA9IG9wdGlvbnM7XG4gICAgY29uc3QgYnViYmxlID0gYnVpbGRCdWJibGUoe1xuICAgICAgdGV4dCxcbiAgICAgIHRpbWVzdGFtcCxcbiAgICAgIHZhcmlhbnQsXG4gICAgICBtZXRhU3VmZml4LFxuICAgICAgYWxsb3dNYXJrZG93bixcbiAgICB9KTtcbiAgICBjb25zdCByb3cgPSBsaW5lKHJvbGUsIGJ1YmJsZSwge1xuICAgICAgcmF3VGV4dDogdGV4dCxcbiAgICAgIHRpbWVzdGFtcCxcbiAgICAgIG1lc3NhZ2VJZCxcbiAgICAgIHJlZ2lzdGVyLFxuICAgICAgbWV0YWRhdGEsXG4gICAgfSk7XG4gICAgc2V0RGlhZ25vc3RpY3MoeyBsYXN0TWVzc2FnZUF0OiB0aW1lc3RhbXAgfHwgbm93SVNPKCkgfSk7XG4gICAgcmV0dXJuIHJvdztcbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZURpYWdub3N0aWNGaWVsZChlbCwgdmFsdWUpIHtcbiAgICBpZiAoIWVsKSByZXR1cm47XG4gICAgZWwudGV4dENvbnRlbnQgPSB2YWx1ZSB8fCBcIlx1MjAxNFwiO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0RGlhZ25vc3RpY3MocGF0Y2gpIHtcbiAgICBPYmplY3QuYXNzaWduKGRpYWdub3N0aWNzLCBwYXRjaCk7XG4gICAgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChwYXRjaCwgXCJjb25uZWN0ZWRBdFwiKSkge1xuICAgICAgdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKFxuICAgICAgICBlbGVtZW50cy5kaWFnQ29ubmVjdGVkLFxuICAgICAgICBkaWFnbm9zdGljcy5jb25uZWN0ZWRBdFxuICAgICAgICAgID8gZm9ybWF0VGltZXN0YW1wKGRpYWdub3N0aWNzLmNvbm5lY3RlZEF0KVxuICAgICAgICAgIDogXCJcdTIwMTRcIixcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocGF0Y2gsIFwibGFzdE1lc3NhZ2VBdFwiKSkge1xuICAgICAgdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKFxuICAgICAgICBlbGVtZW50cy5kaWFnTGFzdE1lc3NhZ2UsXG4gICAgICAgIGRpYWdub3N0aWNzLmxhc3RNZXNzYWdlQXRcbiAgICAgICAgICA/IGZvcm1hdFRpbWVzdGFtcChkaWFnbm9zdGljcy5sYXN0TWVzc2FnZUF0KVxuICAgICAgICAgIDogXCJcdTIwMTRcIixcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocGF0Y2gsIFwibGF0ZW5jeU1zXCIpKSB7XG4gICAgICBpZiAodHlwZW9mIGRpYWdub3N0aWNzLmxhdGVuY3lNcyA9PT0gXCJudW1iZXJcIikge1xuICAgICAgICB1cGRhdGVEaWFnbm9zdGljRmllbGQoXG4gICAgICAgICAgZWxlbWVudHMuZGlhZ0xhdGVuY3ksXG4gICAgICAgICAgYCR7TWF0aC5tYXgoMCwgTWF0aC5yb3VuZChkaWFnbm9zdGljcy5sYXRlbmN5TXMpKX0gbXNgLFxuICAgICAgICApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKGVsZW1lbnRzLmRpYWdMYXRlbmN5LCBcIlx1MjAxNFwiKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVOZXR3b3JrU3RhdHVzKCkge1xuICAgIGlmICghZWxlbWVudHMuZGlhZ05ldHdvcmspIHJldHVybjtcbiAgICBjb25zdCBvbmxpbmUgPSBuYXZpZ2F0b3Iub25MaW5lO1xuICAgIGVsZW1lbnRzLmRpYWdOZXR3b3JrLnRleHRDb250ZW50ID0gb25saW5lID8gXCJFbiBsaWduZVwiIDogXCJIb3JzIGxpZ25lXCI7XG4gICAgZWxlbWVudHMuZGlhZ05ldHdvcmsuY2xhc3NMaXN0LnRvZ2dsZShcInRleHQtZGFuZ2VyXCIsICFvbmxpbmUpO1xuICAgIGVsZW1lbnRzLmRpYWdOZXR3b3JrLmNsYXNzTGlzdC50b2dnbGUoXCJ0ZXh0LXN1Y2Nlc3NcIiwgb25saW5lKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFubm91bmNlQ29ubmVjdGlvbihtZXNzYWdlLCB2YXJpYW50ID0gXCJpbmZvXCIpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmNvbm5lY3Rpb24pIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgY2xhc3NMaXN0ID0gZWxlbWVudHMuY29ubmVjdGlvbi5jbGFzc0xpc3Q7XG4gICAgQXJyYXkuZnJvbShjbGFzc0xpc3QpXG4gICAgICAuZmlsdGVyKChjbHMpID0+IGNscy5zdGFydHNXaXRoKFwiYWxlcnQtXCIpICYmIGNscyAhPT0gXCJhbGVydFwiKVxuICAgICAgLmZvckVhY2goKGNscykgPT4gY2xhc3NMaXN0LnJlbW92ZShjbHMpKTtcbiAgICBjbGFzc0xpc3QuYWRkKFwiYWxlcnRcIik7XG4gICAgY2xhc3NMaXN0LmFkZChgYWxlcnQtJHt2YXJpYW50fWApO1xuICAgIGVsZW1lbnRzLmNvbm5lY3Rpb24udGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIGNsYXNzTGlzdC5yZW1vdmUoXCJ2aXN1YWxseS1oaWRkZW5cIik7XG4gICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgY2xhc3NMaXN0LmFkZChcInZpc3VhbGx5LWhpZGRlblwiKTtcbiAgICB9LCA0MDAwKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZUNvbm5lY3Rpb25NZXRhKG1lc3NhZ2UsIHRvbmUgPSBcIm11dGVkXCIpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmNvbm5lY3Rpb25NZXRhKSByZXR1cm47XG4gICAgY29uc3QgdG9uZXMgPSBbXCJtdXRlZFwiLCBcImluZm9cIiwgXCJzdWNjZXNzXCIsIFwiZGFuZ2VyXCIsIFwid2FybmluZ1wiXTtcbiAgICBlbGVtZW50cy5jb25uZWN0aW9uTWV0YS50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgdG9uZXMuZm9yRWFjaCgodCkgPT4gZWxlbWVudHMuY29ubmVjdGlvbk1ldGEuY2xhc3NMaXN0LnJlbW92ZShgdGV4dC0ke3R9YCkpO1xuICAgIGVsZW1lbnRzLmNvbm5lY3Rpb25NZXRhLmNsYXNzTGlzdC5hZGQoYHRleHQtJHt0b25lfWApO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0V3NTdGF0dXMoc3RhdGUsIHRpdGxlKSB7XG4gICAgaWYgKCFlbGVtZW50cy53c1N0YXR1cykgcmV0dXJuO1xuICAgIGNvbnN0IGxhYmVsID0gc3RhdHVzTGFiZWxzW3N0YXRlXSB8fCBzdGF0ZTtcbiAgICBlbGVtZW50cy53c1N0YXR1cy50ZXh0Q29udGVudCA9IGxhYmVsO1xuICAgIGVsZW1lbnRzLndzU3RhdHVzLmNsYXNzTmFtZSA9IGBiYWRnZSB3cy1iYWRnZSAke3N0YXRlfWA7XG4gICAgaWYgKHRpdGxlKSB7XG4gICAgICBlbGVtZW50cy53c1N0YXR1cy50aXRsZSA9IHRpdGxlO1xuICAgIH0gZWxzZSB7XG4gICAgICBlbGVtZW50cy53c1N0YXR1cy5yZW1vdmVBdHRyaWJ1dGUoXCJ0aXRsZVwiKTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBub3JtYWxpemVTdHJpbmcoc3RyKSB7XG4gICAgY29uc3QgdmFsdWUgPSBTdHJpbmcoc3RyIHx8IFwiXCIpO1xuICAgIHRyeSB7XG4gICAgICByZXR1cm4gdmFsdWVcbiAgICAgICAgLm5vcm1hbGl6ZShcIk5GRFwiKVxuICAgICAgICAucmVwbGFjZSgvW1xcdTAzMDAtXFx1MDM2Zl0vZywgXCJcIilcbiAgICAgICAgLnRvTG93ZXJDYXNlKCk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICByZXR1cm4gdmFsdWUudG9Mb3dlckNhc2UoKTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBhcHBseVRyYW5zY3JpcHRGaWx0ZXIocXVlcnksIG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghZWxlbWVudHMudHJhbnNjcmlwdCkgcmV0dXJuIDA7XG4gICAgY29uc3QgeyBwcmVzZXJ2ZUlucHV0ID0gZmFsc2UgfSA9IG9wdGlvbnM7XG4gICAgY29uc3QgcmF3UXVlcnkgPSB0eXBlb2YgcXVlcnkgPT09IFwic3RyaW5nXCIgPyBxdWVyeSA6IFwiXCI7XG4gICAgaWYgKCFwcmVzZXJ2ZUlucHV0ICYmIGVsZW1lbnRzLmZpbHRlcklucHV0KSB7XG4gICAgICBlbGVtZW50cy5maWx0ZXJJbnB1dC52YWx1ZSA9IHJhd1F1ZXJ5O1xuICAgIH1cbiAgICBjb25zdCB0cmltbWVkID0gcmF3UXVlcnkudHJpbSgpO1xuICAgIHN0YXRlLmFjdGl2ZUZpbHRlciA9IHRyaW1tZWQ7XG4gICAgY29uc3Qgbm9ybWFsaXplZCA9IG5vcm1hbGl6ZVN0cmluZyh0cmltbWVkKTtcbiAgICBsZXQgbWF0Y2hlcyA9IDA7XG4gICAgY29uc3Qgcm93cyA9IEFycmF5LmZyb20oZWxlbWVudHMudHJhbnNjcmlwdC5xdWVyeVNlbGVjdG9yQWxsKFwiLmNoYXQtcm93XCIpKTtcbiAgICByb3dzLmZvckVhY2goKHJvdykgPT4ge1xuICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LWhpZGRlblwiLCBcImNoYXQtZmlsdGVyLW1hdGNoXCIpO1xuICAgICAgaWYgKCFub3JtYWxpemVkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHJhdyA9IHJvdy5kYXRhc2V0LnJhd1RleHQgfHwgXCJcIjtcbiAgICAgIGNvbnN0IG5vcm1hbGl6ZWRSb3cgPSBub3JtYWxpemVTdHJpbmcocmF3KTtcbiAgICAgIGlmIChub3JtYWxpemVkUm93LmluY2x1ZGVzKG5vcm1hbGl6ZWQpKSB7XG4gICAgICAgIHJvdy5jbGFzc0xpc3QuYWRkKFwiY2hhdC1maWx0ZXItbWF0Y2hcIik7XG4gICAgICAgIG1hdGNoZXMgKz0gMTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJvdy5jbGFzc0xpc3QuYWRkKFwiY2hhdC1oaWRkZW5cIik7XG4gICAgICB9XG4gICAgfSk7XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5jbGFzc0xpc3QudG9nZ2xlKFwiZmlsdGVyZWRcIiwgQm9vbGVhbih0cmltbWVkKSk7XG4gICAgaWYgKGVsZW1lbnRzLmZpbHRlckVtcHR5KSB7XG4gICAgICBpZiAodHJpbW1lZCAmJiBtYXRjaGVzID09PSAwKSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LmNsYXNzTGlzdC5yZW1vdmUoXCJkLW5vbmVcIik7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LnNldEF0dHJpYnV0ZShcbiAgICAgICAgICBcImFyaWEtbGl2ZVwiLFxuICAgICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LmdldEF0dHJpYnV0ZShcImFyaWEtbGl2ZVwiKSB8fCBcInBvbGl0ZVwiLFxuICAgICAgICApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLmZpbHRlckhpbnQpIHtcbiAgICAgIGlmICh0cmltbWVkKSB7XG4gICAgICAgIGxldCBzdW1tYXJ5ID0gXCJBdWN1biBtZXNzYWdlIG5lIGNvcnJlc3BvbmQuXCI7XG4gICAgICAgIGlmIChtYXRjaGVzID09PSAxKSB7XG4gICAgICAgICAgc3VtbWFyeSA9IFwiMSBtZXNzYWdlIGNvcnJlc3BvbmQuXCI7XG4gICAgICAgIH0gZWxzZSBpZiAobWF0Y2hlcyA+IDEpIHtcbiAgICAgICAgICBzdW1tYXJ5ID0gYCR7bWF0Y2hlc30gbWVzc2FnZXMgY29ycmVzcG9uZGVudC5gO1xuICAgICAgICB9XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQgPSBzdW1tYXJ5O1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVySGludC50ZXh0Q29udGVudCA9IGZpbHRlckhpbnREZWZhdWx0O1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbWF0Y2hlcztcbiAgfVxuXG4gIGZ1bmN0aW9uIHJlYXBwbHlUcmFuc2NyaXB0RmlsdGVyKCkge1xuICAgIGlmIChzdGF0ZS5hY3RpdmVGaWx0ZXIpIHtcbiAgICAgIGFwcGx5VHJhbnNjcmlwdEZpbHRlcihzdGF0ZS5hY3RpdmVGaWx0ZXIsIHsgcHJlc2VydmVJbnB1dDogdHJ1ZSB9KTtcbiAgICB9IGVsc2UgaWYgKGVsZW1lbnRzLnRyYW5zY3JpcHQpIHtcbiAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuY2xhc3NMaXN0LnJlbW92ZShcImZpbHRlcmVkXCIpO1xuICAgICAgY29uc3Qgcm93cyA9IEFycmF5LmZyb20oXG4gICAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQucXVlcnlTZWxlY3RvckFsbChcIi5jaGF0LXJvd1wiKSxcbiAgICAgICk7XG4gICAgICByb3dzLmZvckVhY2goKHJvdykgPT4ge1xuICAgICAgICByb3cuY2xhc3NMaXN0LnJlbW92ZShcImNoYXQtaGlkZGVuXCIsIFwiY2hhdC1maWx0ZXItbWF0Y2hcIik7XG4gICAgICB9KTtcbiAgICAgIGlmIChlbGVtZW50cy5maWx0ZXJFbXB0eSkge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5jbGFzc0xpc3QuYWRkKFwiZC1ub25lXCIpO1xuICAgICAgfVxuICAgICAgaWYgKGVsZW1lbnRzLmZpbHRlckhpbnQpIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVySGludC50ZXh0Q29udGVudCA9IGZpbHRlckhpbnREZWZhdWx0O1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGNsZWFyVHJhbnNjcmlwdEZpbHRlcihmb2N1cyA9IHRydWUpIHtcbiAgICBzdGF0ZS5hY3RpdmVGaWx0ZXIgPSBcIlwiO1xuICAgIGlmIChlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQudmFsdWUgPSBcIlwiO1xuICAgIH1cbiAgICByZWFwcGx5VHJhbnNjcmlwdEZpbHRlcigpO1xuICAgIGlmIChmb2N1cyAmJiBlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQuZm9jdXMoKTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiByZW5kZXJIaXN0b3J5KGVudHJpZXMsIG9wdGlvbnMgPSB7fSkge1xuICAgIGNvbnN0IHsgcmVwbGFjZSA9IGZhbHNlIH0gPSBvcHRpb25zO1xuICAgIGlmICghQXJyYXkuaXNBcnJheShlbnRyaWVzKSB8fCBlbnRyaWVzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgaWYgKHJlcGxhY2UpIHtcbiAgICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5pbm5lckhUTUwgPSBcIlwiO1xuICAgICAgICBzdGF0ZS5oaXN0b3J5Qm9vdHN0cmFwcGVkID0gZmFsc2U7XG4gICAgICAgIGhpZGVTY3JvbGxCdXR0b24oKTtcbiAgICAgICAgdGltZWxpbmVTdG9yZS5jbGVhcigpO1xuICAgICAgfVxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAocmVwbGFjZSkge1xuICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5pbm5lckhUTUwgPSBcIlwiO1xuICAgICAgc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCA9IGZhbHNlO1xuICAgICAgc3RhdGUuc3RyZWFtUm93ID0gbnVsbDtcbiAgICAgIHN0YXRlLnN0cmVhbUJ1ZiA9IFwiXCI7XG4gICAgICB0aW1lbGluZVN0b3JlLmNsZWFyKCk7XG4gICAgfVxuICAgIGlmIChzdGF0ZS5oaXN0b3J5Qm9vdHN0cmFwcGVkICYmICFyZXBsYWNlKSB7XG4gICAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gdHJ1ZTtcbiAgICAgIGNvbnN0IHJvd3MgPSBBcnJheS5mcm9tKFxuICAgICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnF1ZXJ5U2VsZWN0b3JBbGwoXCIuY2hhdC1yb3dcIiksXG4gICAgICApO1xuICAgICAgcm93cy5mb3JFYWNoKChyb3cpID0+IHtcbiAgICAgICAgY29uc3QgZXhpc3RpbmdJZCA9IHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZDtcbiAgICAgICAgaWYgKGV4aXN0aW5nSWQgJiYgdGltZWxpbmVTdG9yZS5tYXAuaGFzKGV4aXN0aW5nSWQpKSB7XG4gICAgICAgICAgY29uc3QgY3VycmVudFJvbGUgPSByb3cuZGF0YXNldC5yb2xlIHx8IFwiXCI7XG4gICAgICAgICAgaWYgKGN1cnJlbnRSb2xlKSB7XG4gICAgICAgICAgICBkZWNvcmF0ZVJvdyhyb3csIGN1cnJlbnRSb2xlKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGJ1YmJsZSA9IHJvdy5xdWVyeVNlbGVjdG9yKFwiLmNoYXQtYnViYmxlXCIpO1xuICAgICAgICBjb25zdCBtZXRhID0gYnViYmxlPy5xdWVyeVNlbGVjdG9yKFwiLmNoYXQtbWV0YVwiKSB8fCBudWxsO1xuICAgICAgICBjb25zdCByb2xlID1cbiAgICAgICAgICByb3cuZGF0YXNldC5yb2xlIHx8XG4gICAgICAgICAgKHJvdy5jbGFzc0xpc3QuY29udGFpbnMoXCJjaGF0LXVzZXJcIilcbiAgICAgICAgICAgID8gXCJ1c2VyXCJcbiAgICAgICAgICAgIDogcm93LmNsYXNzTGlzdC5jb250YWlucyhcImNoYXQtYXNzaXN0YW50XCIpXG4gICAgICAgICAgICA/IFwiYXNzaXN0YW50XCJcbiAgICAgICAgICAgIDogXCJzeXN0ZW1cIik7XG4gICAgICAgIGNvbnN0IHRleHQgPVxuICAgICAgICAgIHJvdy5kYXRhc2V0LnJhd1RleHQgJiYgcm93LmRhdGFzZXQucmF3VGV4dC5sZW5ndGggPiAwXG4gICAgICAgICAgICA/IHJvdy5kYXRhc2V0LnJhd1RleHRcbiAgICAgICAgICAgIDogYnViYmxlXG4gICAgICAgICAgICA/IGV4dHJhY3RCdWJibGVUZXh0KGJ1YmJsZSlcbiAgICAgICAgICAgIDogcm93LnRleHRDb250ZW50LnRyaW0oKTtcbiAgICAgICAgY29uc3QgdGltZXN0YW1wID1cbiAgICAgICAgICByb3cuZGF0YXNldC50aW1lc3RhbXAgJiYgcm93LmRhdGFzZXQudGltZXN0YW1wLmxlbmd0aCA+IDBcbiAgICAgICAgICAgID8gcm93LmRhdGFzZXQudGltZXN0YW1wXG4gICAgICAgICAgICA6IG1ldGFcbiAgICAgICAgICAgID8gbWV0YS50ZXh0Q29udGVudC50cmltKClcbiAgICAgICAgICAgIDogbm93SVNPKCk7XG4gICAgICAgIGNvbnN0IG1lc3NhZ2VJZCA9IHRpbWVsaW5lU3RvcmUucmVnaXN0ZXIoe1xuICAgICAgICAgIGlkOiBleGlzdGluZ0lkLFxuICAgICAgICAgIHJvbGUsXG4gICAgICAgICAgdGV4dCxcbiAgICAgICAgICB0aW1lc3RhbXAsXG4gICAgICAgICAgcm93LFxuICAgICAgICB9KTtcbiAgICAgICAgcm93LmRhdGFzZXQubWVzc2FnZUlkID0gbWVzc2FnZUlkO1xuICAgICAgICByb3cuZGF0YXNldC5yb2xlID0gcm9sZTtcbiAgICAgICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IHRleHQ7XG4gICAgICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IHRpbWVzdGFtcDtcbiAgICAgICAgZGVjb3JhdGVSb3cocm93LCByb2xlKTtcbiAgICAgIH0pO1xuICAgICAgc3RhdGUuYm9vdHN0cmFwcGluZyA9IGZhbHNlO1xuICAgICAgcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgc3RhdGUuYm9vdHN0cmFwcGluZyA9IHRydWU7XG4gICAgZW50cmllc1xuICAgICAgLnNsaWNlKClcbiAgICAgIC5yZXZlcnNlKClcbiAgICAgIC5mb3JFYWNoKChpdGVtKSA9PiB7XG4gICAgICAgIGlmIChpdGVtLnF1ZXJ5KSB7XG4gICAgICAgICAgYXBwZW5kTWVzc2FnZShcInVzZXJcIiwgaXRlbS5xdWVyeSwge1xuICAgICAgICAgICAgdGltZXN0YW1wOiBpdGVtLnRpbWVzdGFtcCxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoaXRlbS5yZXNwb25zZSkge1xuICAgICAgICAgIGFwcGVuZE1lc3NhZ2UoXCJhc3Npc3RhbnRcIiwgaXRlbS5yZXNwb25zZSwge1xuICAgICAgICAgICAgdGltZXN0YW1wOiBpdGVtLnRpbWVzdGFtcCxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgc3RhdGUuYm9vdHN0cmFwcGluZyA9IGZhbHNlO1xuICAgIHN0YXRlLmhpc3RvcnlCb290c3RyYXBwZWQgPSB0cnVlO1xuICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiBmYWxzZSB9KTtcbiAgICBoaWRlU2Nyb2xsQnV0dG9uKCk7XG4gIH1cblxuICBmdW5jdGlvbiBzdGFydFN0cmVhbSgpIHtcbiAgICBzdGF0ZS5zdHJlYW1CdWYgPSBcIlwiO1xuICAgIGNvbnN0IHRzID0gbm93SVNPKCk7XG4gICAgc3RhdGUuc3RyZWFtTWVzc2FnZUlkID0gdGltZWxpbmVTdG9yZS5tYWtlTWVzc2FnZUlkKCk7XG4gICAgc3RhdGUuc3RyZWFtUm93ID0gbGluZShcbiAgICAgIFwiYXNzaXN0YW50XCIsXG4gICAgICAnPGRpdiBjbGFzcz1cImNoYXQtYnViYmxlXCI+PHNwYW4gY2xhc3M9XCJjaGF0LWN1cnNvclwiPlx1MjU4RDwvc3Bhbj48L2Rpdj4nLFxuICAgICAge1xuICAgICAgICByYXdUZXh0OiBcIlwiLFxuICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICBtZXNzYWdlSWQ6IHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCxcbiAgICAgICAgbWV0YWRhdGE6IHsgc3RyZWFtaW5nOiB0cnVlIH0sXG4gICAgICB9LFxuICAgICk7XG4gICAgc2V0RGlhZ25vc3RpY3MoeyBsYXN0TWVzc2FnZUF0OiB0cyB9KTtcbiAgICBpZiAoc3RhdGUucmVzZXRTdGF0dXNUaW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHN0YXRlLnJlc2V0U3RhdHVzVGltZXIpO1xuICAgIH1cbiAgICBzZXRDb21wb3NlclN0YXR1cyhcIlJcdTAwRTlwb25zZSBlbiBjb3Vyc1x1MjAyNlwiLCBcImluZm9cIik7XG4gIH1cblxuICBmdW5jdGlvbiBpc1N0cmVhbWluZygpIHtcbiAgICByZXR1cm4gQm9vbGVhbihzdGF0ZS5zdHJlYW1Sb3cpO1xuICB9XG5cbiAgZnVuY3Rpb24gaGFzU3RyZWFtQnVmZmVyKCkge1xuICAgIHJldHVybiBCb29sZWFuKHN0YXRlLnN0cmVhbUJ1Zik7XG4gIH1cblxuICBmdW5jdGlvbiBhcHBlbmRTdHJlYW0oZGVsdGEpIHtcbiAgICBpZiAoIXN0YXRlLnN0cmVhbVJvdykge1xuICAgICAgc3RhcnRTdHJlYW0oKTtcbiAgICB9XG4gICAgY29uc3Qgc2hvdWxkU3RpY2sgPSBpc0F0Qm90dG9tKCk7XG4gICAgc3RhdGUuc3RyZWFtQnVmICs9IGRlbHRhIHx8IFwiXCI7XG4gICAgY29uc3QgYnViYmxlID0gc3RhdGUuc3RyZWFtUm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKGJ1YmJsZSkge1xuICAgICAgYnViYmxlLmlubmVySFRNTCA9IGAke3JlbmRlck1hcmtkb3duKHN0YXRlLnN0cmVhbUJ1Zil9PHNwYW4gY2xhc3M9XCJjaGF0LWN1cnNvclwiPlx1MjU4RDwvc3Bhbj5gO1xuICAgIH1cbiAgICBpZiAoc3RhdGUuc3RyZWFtTWVzc2FnZUlkKSB7XG4gICAgICB0aW1lbGluZVN0b3JlLnVwZGF0ZShzdGF0ZS5zdHJlYW1NZXNzYWdlSWQsIHtcbiAgICAgICAgdGV4dDogc3RhdGUuc3RyZWFtQnVmLFxuICAgICAgICBtZXRhZGF0YTogeyBzdHJlYW1pbmc6IHRydWUgfSxcbiAgICAgIH0pO1xuICAgIH1cbiAgICBzZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IG5vd0lTTygpIH0pO1xuICAgIGlmIChzaG91bGRTdGljaykge1xuICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IGZhbHNlIH0pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGVuZFN0cmVhbShkYXRhKSB7XG4gICAgaWYgKCFzdGF0ZS5zdHJlYW1Sb3cpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgYnViYmxlID0gc3RhdGUuc3RyZWFtUm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKGJ1YmJsZSkge1xuICAgICAgYnViYmxlLmlubmVySFRNTCA9IHJlbmRlck1hcmtkb3duKHN0YXRlLnN0cmVhbUJ1Zik7XG4gICAgICBjb25zdCBtZXRhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRpdlwiKTtcbiAgICAgIG1ldGEuY2xhc3NOYW1lID0gXCJjaGF0LW1ldGFcIjtcbiAgICAgIGNvbnN0IHRzID0gZGF0YSAmJiBkYXRhLnRpbWVzdGFtcCA/IGRhdGEudGltZXN0YW1wIDogbm93SVNPKCk7XG4gICAgICBtZXRhLnRleHRDb250ZW50ID0gZm9ybWF0VGltZXN0YW1wKHRzKTtcbiAgICAgIGlmIChkYXRhICYmIGRhdGEuZXJyb3IpIHtcbiAgICAgICAgbWV0YS5jbGFzc0xpc3QuYWRkKFwidGV4dC1kYW5nZXJcIik7XG4gICAgICAgIG1ldGEudGV4dENvbnRlbnQgPSBgJHttZXRhLnRleHRDb250ZW50fSBcdTIwMjIgJHtkYXRhLmVycm9yfWA7XG4gICAgICB9XG4gICAgICBidWJibGUuYXBwZW5kQ2hpbGQobWV0YSk7XG4gICAgICBkZWNvcmF0ZVJvdyhzdGF0ZS5zdHJlYW1Sb3csIFwiYXNzaXN0YW50XCIpO1xuICAgICAgaGlnaGxpZ2h0Um93KHN0YXRlLnN0cmVhbVJvdywgXCJhc3Npc3RhbnRcIik7XG4gICAgICBpZiAoaXNBdEJvdHRvbSgpKSB7XG4gICAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiB0cnVlIH0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgc2hvd1Njcm9sbEJ1dHRvbigpO1xuICAgICAgfVxuICAgICAgaWYgKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCkge1xuICAgICAgICB0aW1lbGluZVN0b3JlLnVwZGF0ZShzdGF0ZS5zdHJlYW1NZXNzYWdlSWQsIHtcbiAgICAgICAgICB0ZXh0OiBzdGF0ZS5zdHJlYW1CdWYsXG4gICAgICAgICAgdGltZXN0YW1wOiB0cyxcbiAgICAgICAgICBtZXRhZGF0YToge1xuICAgICAgICAgICAgc3RyZWFtaW5nOiBudWxsLFxuICAgICAgICAgICAgLi4uKGRhdGEgJiYgZGF0YS5lcnJvciA/IHsgZXJyb3I6IGRhdGEuZXJyb3IgfSA6IHsgZXJyb3I6IG51bGwgfSksXG4gICAgICAgICAgfSxcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgICBzZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHRzIH0pO1xuICAgIH1cbiAgICBjb25zdCBoYXNFcnJvciA9IEJvb2xlYW4oZGF0YSAmJiBkYXRhLmVycm9yKTtcbiAgICBzZXRDb21wb3NlclN0YXR1cyhcbiAgICAgIGhhc0Vycm9yXG4gICAgICAgID8gXCJSXHUwMEU5cG9uc2UgaW5kaXNwb25pYmxlLiBDb25zdWx0ZXogbGVzIGpvdXJuYXV4LlwiXG4gICAgICAgIDogXCJSXHUwMEU5cG9uc2UgcmVcdTAwRTd1ZS5cIixcbiAgICAgIGhhc0Vycm9yID8gXCJkYW5nZXJcIiA6IFwic3VjY2Vzc1wiLFxuICAgICk7XG4gICAgc2NoZWR1bGVDb21wb3NlcklkbGUoaGFzRXJyb3IgPyA2MDAwIDogMzUwMCk7XG4gICAgc3RhdGUuc3RyZWFtUm93ID0gbnVsbDtcbiAgICBzdGF0ZS5zdHJlYW1CdWYgPSBcIlwiO1xuICAgIHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCA9IG51bGw7XG4gIH1cblxuICBmdW5jdGlvbiBhcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcoc3VnZ2VzdGlvbnMpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnF1aWNrQWN0aW9ucykgcmV0dXJuO1xuICAgIGlmICghQXJyYXkuaXNBcnJheShzdWdnZXN0aW9ucykgfHwgc3VnZ2VzdGlvbnMubGVuZ3RoID09PSAwKSByZXR1cm47XG4gICAgY29uc3QgYnV0dG9ucyA9IEFycmF5LmZyb20oXG4gICAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMucXVlcnlTZWxlY3RvckFsbChcImJ1dHRvbi5xYVwiKSxcbiAgICApO1xuICAgIGNvbnN0IGxvb2t1cCA9IG5ldyBNYXAoKTtcbiAgICBidXR0b25zLmZvckVhY2goKGJ0bikgPT4gbG9va3VwLnNldChidG4uZGF0YXNldC5hY3Rpb24sIGJ0bikpO1xuICAgIGNvbnN0IGZyYWcgPSBkb2N1bWVudC5jcmVhdGVEb2N1bWVudEZyYWdtZW50KCk7XG4gICAgc3VnZ2VzdGlvbnMuZm9yRWFjaCgoa2V5KSA9PiB7XG4gICAgICBpZiAobG9va3VwLmhhcyhrZXkpKSB7XG4gICAgICAgIGZyYWcuYXBwZW5kQ2hpbGQobG9va3VwLmdldChrZXkpKTtcbiAgICAgICAgbG9va3VwLmRlbGV0ZShrZXkpO1xuICAgICAgfVxuICAgIH0pO1xuICAgIGxvb2t1cC5mb3JFYWNoKChidG4pID0+IGZyYWcuYXBwZW5kQ2hpbGQoYnRuKSk7XG4gICAgZWxlbWVudHMucXVpY2tBY3Rpb25zLmlubmVySFRNTCA9IFwiXCI7XG4gICAgZWxlbWVudHMucXVpY2tBY3Rpb25zLmFwcGVuZENoaWxkKGZyYWcpO1xuICB9XG5cbiAgZnVuY3Rpb24gZm9ybWF0UGVyZihkKSB7XG4gICAgY29uc3QgYml0cyA9IFtdO1xuICAgIGlmIChkICYmIHR5cGVvZiBkLmNwdSAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgY29uc3QgY3B1ID0gTnVtYmVyKGQuY3B1KTtcbiAgICAgIGlmICghTnVtYmVyLmlzTmFOKGNwdSkpIHtcbiAgICAgICAgYml0cy5wdXNoKGBDUFUgJHtjcHUudG9GaXhlZCgwKX0lYCk7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChkICYmIHR5cGVvZiBkLnR0ZmJfbXMgIT09IFwidW5kZWZpbmVkXCIpIHtcbiAgICAgIGNvbnN0IHR0ZmIgPSBOdW1iZXIoZC50dGZiX21zKTtcbiAgICAgIGlmICghTnVtYmVyLmlzTmFOKHR0ZmIpKSB7XG4gICAgICAgIGJpdHMucHVzaChgVFRGQiAke3R0ZmJ9IG1zYCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBiaXRzLmpvaW4oXCIgXHUyMDIyIFwiKSB8fCBcIm1pc2UgXHUwMEUwIGpvdXJcIjtcbiAgfVxuXG4gIGZ1bmN0aW9uIGF0dGFjaEV2ZW50cygpIHtcbiAgICBpZiAoZWxlbWVudHMuY29tcG9zZXIpIHtcbiAgICAgIGVsZW1lbnRzLmNvbXBvc2VyLmFkZEV2ZW50TGlzdGVuZXIoXCJzdWJtaXRcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XG4gICAgICAgIGNvbnN0IHRleHQgPSAoZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCIpLnRyaW0oKTtcbiAgICAgICAgZW1pdChcInN1Ym1pdFwiLCB7IHRleHQgfSk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMucXVpY2tBY3Rpb25zKSB7XG4gICAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsIChldmVudCkgPT4ge1xuICAgICAgICBjb25zdCB0YXJnZXQgPSBldmVudC50YXJnZXQ7XG4gICAgICAgIGlmICghKHRhcmdldCBpbnN0YW5jZW9mIEhUTUxCdXR0b25FbGVtZW50KSkge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBhY3Rpb24gPSB0YXJnZXQuZGF0YXNldC5hY3Rpb247XG4gICAgICAgIGlmICghYWN0aW9uKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGVtaXQoXCJxdWljay1hY3Rpb25cIiwgeyBhY3Rpb24gfSk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LmFkZEV2ZW50TGlzdGVuZXIoXCJpbnB1dFwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgZW1pdChcImZpbHRlci1jaGFuZ2VcIiwgeyB2YWx1ZTogZXZlbnQudGFyZ2V0LnZhbHVlIHx8IFwiXCIgfSk7XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LmFkZEV2ZW50TGlzdGVuZXIoXCJrZXlkb3duXCIsIChldmVudCkgPT4ge1xuICAgICAgICBpZiAoZXZlbnQua2V5ID09PSBcIkVzY2FwZVwiKSB7XG4gICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICBlbWl0KFwiZmlsdGVyLWNsZWFyXCIpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMuZmlsdGVyQ2xlYXIpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlckNsZWFyLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIGVtaXQoXCJmaWx0ZXItY2xlYXJcIik7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMuZXhwb3J0SnNvbikge1xuICAgICAgZWxlbWVudHMuZXhwb3J0SnNvbi5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT5cbiAgICAgICAgZW1pdChcImV4cG9ydFwiLCB7IGZvcm1hdDogXCJqc29uXCIgfSksXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMuZXhwb3J0TWFya2Rvd24pIHtcbiAgICAgIGVsZW1lbnRzLmV4cG9ydE1hcmtkb3duLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PlxuICAgICAgICBlbWl0KFwiZXhwb3J0XCIsIHsgZm9ybWF0OiBcIm1hcmtkb3duXCIgfSksXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMuZXhwb3J0Q29weSkge1xuICAgICAgZWxlbWVudHMuZXhwb3J0Q29weS5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4gZW1pdChcImV4cG9ydC1jb3B5XCIpKTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICBlbGVtZW50cy5wcm9tcHQuYWRkRXZlbnRMaXN0ZW5lcihcImlucHV0XCIsIChldmVudCkgPT4ge1xuICAgICAgICB1cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICAgIGF1dG9zaXplUHJvbXB0KCk7XG4gICAgICAgIGNvbnN0IHZhbHVlID0gZXZlbnQudGFyZ2V0LnZhbHVlIHx8IFwiXCI7XG4gICAgICAgIGlmICghdmFsdWUudHJpbSgpKSB7XG4gICAgICAgICAgc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCk7XG4gICAgICAgIH1cbiAgICAgICAgZW1pdChcInByb21wdC1pbnB1dFwiLCB7IHZhbHVlIH0pO1xuICAgICAgfSk7XG4gICAgICBlbGVtZW50cy5wcm9tcHQuYWRkRXZlbnRMaXN0ZW5lcihcInBhc3RlXCIsICgpID0+IHtcbiAgICAgICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgIHVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgICAgICBhdXRvc2l6ZVByb21wdCgpO1xuICAgICAgICAgIGVtaXQoXCJwcm9tcHQtaW5wdXRcIiwgeyB2YWx1ZTogZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCIgfSk7XG4gICAgICAgIH0sIDApO1xuICAgICAgfSk7XG4gICAgICBlbGVtZW50cy5wcm9tcHQuYWRkRXZlbnRMaXN0ZW5lcihcImtleWRvd25cIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGlmICgoZXZlbnQuY3RybEtleSB8fCBldmVudC5tZXRhS2V5KSAmJiBldmVudC5rZXkgPT09IFwiRW50ZXJcIikge1xuICAgICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XG4gICAgICAgICAgZW1pdChcInN1Ym1pdFwiLCB7IHRleHQ6IChlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIikudHJpbSgpIH0pO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwiZm9jdXNcIiwgKCkgPT4ge1xuICAgICAgICBzZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgICBcIlJcdTAwRTlkaWdleiB2b3RyZSBtZXNzYWdlLCBwdWlzIEN0cmwrRW50clx1MDBFOWUgcG91ciBsJ2Vudm95ZXIuXCIsXG4gICAgICAgICAgXCJpbmZvXCIsXG4gICAgICAgICk7XG4gICAgICAgIHNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLnRyYW5zY3JpcHQpIHtcbiAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuYWRkRXZlbnRMaXN0ZW5lcihcInNjcm9sbFwiLCAoKSA9PiB7XG4gICAgICAgIGlmIChpc0F0Qm90dG9tKCkpIHtcbiAgICAgICAgICBoaWRlU2Nyb2xsQnV0dG9uKCk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgc2hvd1Njcm9sbEJ1dHRvbigpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMuc2Nyb2xsQm90dG9tKSB7XG4gICAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IHRydWUgfSk7XG4gICAgICAgIGlmIChlbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgICBlbGVtZW50cy5wcm9tcHQuZm9jdXMoKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoXCJyZXNpemVcIiwgKCkgPT4ge1xuICAgICAgaWYgKGlzQXRCb3R0b20oKSkge1xuICAgICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogZmFsc2UgfSk7XG4gICAgICB9XG4gICAgfSk7XG5cbiAgICB1cGRhdGVOZXR3b3JrU3RhdHVzKCk7XG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoXCJvbmxpbmVcIiwgKCkgPT4ge1xuICAgICAgdXBkYXRlTmV0d29ya1N0YXR1cygpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29ubmV4aW9uIHJcdTAwRTlzZWF1IHJlc3RhdXJcdTAwRTllLlwiLCBcImluZm9cIik7XG4gICAgfSk7XG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoXCJvZmZsaW5lXCIsICgpID0+IHtcbiAgICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMoKTtcbiAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcIkNvbm5leGlvbiByXHUwMEU5c2VhdSBwZXJkdWUuXCIsIFwiZGFuZ2VyXCIpO1xuICAgIH0pO1xuXG4gICAgY29uc3QgdG9nZ2xlQnRuID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJ0b2dnbGUtZGFyay1tb2RlXCIpO1xuICAgIGNvbnN0IGRhcmtNb2RlS2V5ID0gXCJkYXJrLW1vZGVcIjtcblxuICAgIGZ1bmN0aW9uIGFwcGx5RGFya01vZGUoZW5hYmxlZCkge1xuICAgICAgZG9jdW1lbnQuYm9keS5jbGFzc0xpc3QudG9nZ2xlKFwiZGFyay1tb2RlXCIsIGVuYWJsZWQpO1xuICAgICAgaWYgKHRvZ2dsZUJ0bikge1xuICAgICAgICB0b2dnbGVCdG4udGV4dENvbnRlbnQgPSBlbmFibGVkID8gXCJNb2RlIENsYWlyXCIgOiBcIk1vZGUgU29tYnJlXCI7XG4gICAgICAgIHRvZ2dsZUJ0bi5zZXRBdHRyaWJ1dGUoXCJhcmlhLXByZXNzZWRcIiwgZW5hYmxlZCA/IFwidHJ1ZVwiIDogXCJmYWxzZVwiKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgYXBwbHlEYXJrTW9kZSh3aW5kb3cubG9jYWxTdG9yYWdlLmdldEl0ZW0oZGFya01vZGVLZXkpID09PSBcIjFcIik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcmVhZCBkYXJrIG1vZGUgcHJlZmVyZW5jZVwiLCBlcnIpO1xuICAgIH1cblxuICAgIGlmICh0b2dnbGVCdG4pIHtcbiAgICAgIHRvZ2dsZUJ0bi5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgICAgICBjb25zdCBlbmFibGVkID0gIWRvY3VtZW50LmJvZHkuY2xhc3NMaXN0LmNvbnRhaW5zKFwiZGFyay1tb2RlXCIpO1xuICAgICAgICBhcHBseURhcmtNb2RlKGVuYWJsZWQpO1xuICAgICAgICB0cnkge1xuICAgICAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2Uuc2V0SXRlbShkYXJrTW9kZUtleSwgZW5hYmxlZCA/IFwiMVwiIDogXCIwXCIpO1xuICAgICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcGVyc2lzdCBkYXJrIG1vZGUgcHJlZmVyZW5jZVwiLCBlcnIpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudm9pY2VUb2dnbGUpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIGVtaXQoXCJ2b2ljZS10b2dnbGVcIik7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudm9pY2VBdXRvU2VuZCkge1xuICAgICAgZWxlbWVudHMudm9pY2VBdXRvU2VuZC5hZGRFdmVudExpc3RlbmVyKFwiY2hhbmdlXCIsIChldmVudCkgPT4ge1xuICAgICAgICBlbWl0KFwidm9pY2UtYXV0b3NlbmQtY2hhbmdlXCIsIHsgZW5hYmxlZDogZXZlbnQudGFyZ2V0LmNoZWNrZWQgfSk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudm9pY2VQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VQbGF5YmFjay5hZGRFdmVudExpc3RlbmVyKFwiY2hhbmdlXCIsIChldmVudCkgPT4ge1xuICAgICAgICBlbWl0KFwidm9pY2UtcGxheWJhY2stY2hhbmdlXCIsIHsgZW5hYmxlZDogZXZlbnQudGFyZ2V0LmNoZWNrZWQgfSk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2spIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlU3RvcFBsYXliYWNrLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIGVtaXQoXCJ2b2ljZS1zdG9wLXBsYXliYWNrXCIpO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlVm9pY2VTZWxlY3QpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVm9pY2VTZWxlY3QuYWRkRXZlbnRMaXN0ZW5lcihcImNoYW5nZVwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgZW1pdChcInZvaWNlLXZvaWNlLWNoYW5nZVwiLCB7IHZvaWNlVVJJOiBldmVudC50YXJnZXQudmFsdWUgfHwgbnVsbCB9KTtcbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGluaXRpYWxpc2UoKSB7XG4gICAgc2V0RGlhZ25vc3RpY3MoeyBjb25uZWN0ZWRBdDogbnVsbCwgbGFzdE1lc3NhZ2VBdDogbnVsbCwgbGF0ZW5jeU1zOiBudWxsIH0pO1xuICAgIHVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICBhdXRvc2l6ZVByb21wdCgpO1xuICAgIHNldENvbXBvc2VyU3RhdHVzSWRsZSgpO1xuICAgIHNldFZvaWNlVHJhbnNjcmlwdChcIlwiLCB7IHN0YXRlOiBcImlkbGVcIiwgcGxhY2Vob2xkZXI6IFwiXCIgfSk7XG4gICAgYXR0YWNoRXZlbnRzKCk7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGVsZW1lbnRzLFxuICAgIG9uLFxuICAgIGVtaXQsXG4gICAgaW5pdGlhbGlzZSxcbiAgICByZW5kZXJIaXN0b3J5LFxuICAgIGFwcGVuZE1lc3NhZ2UsXG4gICAgc2V0QnVzeSxcbiAgICBzaG93RXJyb3IsXG4gICAgaGlkZUVycm9yLFxuICAgIHNldENvbXBvc2VyU3RhdHVzLFxuICAgIHNldENvbXBvc2VyU3RhdHVzSWRsZSxcbiAgICBzY2hlZHVsZUNvbXBvc2VySWRsZSxcbiAgICB1cGRhdGVQcm9tcHRNZXRyaWNzLFxuICAgIGF1dG9zaXplUHJvbXB0LFxuICAgIHN0YXJ0U3RyZWFtLFxuICAgIGFwcGVuZFN0cmVhbSxcbiAgICBlbmRTdHJlYW0sXG4gICAgYW5ub3VuY2VDb25uZWN0aW9uLFxuICAgIHVwZGF0ZUNvbm5lY3Rpb25NZXRhLFxuICAgIHNldERpYWdub3N0aWNzLFxuICAgIGFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyxcbiAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIsXG4gICAgcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIsXG4gICAgY2xlYXJUcmFuc2NyaXB0RmlsdGVyLFxuICAgIHNldFdzU3RhdHVzLFxuICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMsXG4gICAgc2Nyb2xsVG9Cb3R0b20sXG4gICAgc2V0Vm9pY2VTdGF0dXMsXG4gICAgc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUsXG4gICAgc2V0Vm9pY2VBdmFpbGFiaWxpdHksXG4gICAgc2V0Vm9pY2VMaXN0ZW5pbmcsXG4gICAgc2V0Vm9pY2VUcmFuc2NyaXB0LFxuICAgIHNldFZvaWNlUHJlZmVyZW5jZXMsXG4gICAgc2V0Vm9pY2VTcGVha2luZyxcbiAgICBzZXRWb2ljZVZvaWNlT3B0aW9ucyxcbiAgICBzZXQgZGlhZ25vc3RpY3ModmFsdWUpIHtcbiAgICAgIE9iamVjdC5hc3NpZ24oZGlhZ25vc3RpY3MsIHZhbHVlKTtcbiAgICB9LFxuICAgIGdldCBkaWFnbm9zdGljcygpIHtcbiAgICAgIHJldHVybiB7IC4uLmRpYWdub3N0aWNzIH07XG4gICAgfSxcbiAgICBmb3JtYXRUaW1lc3RhbXAsXG4gICAgbm93SVNPLFxuICAgIGZvcm1hdFBlcmYsXG4gICAgaXNTdHJlYW1pbmcsXG4gICAgaGFzU3RyZWFtQnVmZmVyLFxuICAgIGdldCB2b2ljZVN0YXR1c0RlZmF1bHQoKSB7XG4gICAgICByZXR1cm4gdm9pY2VTdGF0dXNEZWZhdWx0O1xuICAgIH0sXG4gIH07XG59XG4iLCAiY29uc3QgREVGQVVMVF9TVE9SQUdFX0tFWSA9IFwibW9uZ2Fyc19qd3RcIjtcblxuZnVuY3Rpb24gaGFzTG9jYWxTdG9yYWdlKCkge1xuICB0cnkge1xuICAgIHJldHVybiB0eXBlb2Ygd2luZG93ICE9PSBcInVuZGVmaW5lZFwiICYmIEJvb2xlYW4od2luZG93LmxvY2FsU3RvcmFnZSk7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIGNvbnNvbGUud2FybihcIkFjY2Vzc2luZyBsb2NhbFN0b3JhZ2UgZmFpbGVkXCIsIGVycik7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVBdXRoU2VydmljZShjb25maWcgPSB7fSkge1xuICBjb25zdCBzdG9yYWdlS2V5ID0gY29uZmlnLnN0b3JhZ2VLZXkgfHwgREVGQVVMVF9TVE9SQUdFX0tFWTtcbiAgbGV0IGZhbGxiYWNrVG9rZW4gPVxuICAgIHR5cGVvZiBjb25maWcudG9rZW4gPT09IFwic3RyaW5nXCIgJiYgY29uZmlnLnRva2VuLnRyaW0oKSAhPT0gXCJcIlxuICAgICAgPyBjb25maWcudG9rZW4udHJpbSgpXG4gICAgICA6IHVuZGVmaW5lZDtcblxuICBmdW5jdGlvbiBwZXJzaXN0VG9rZW4odG9rZW4pIHtcbiAgICBpZiAodHlwZW9mIHRva2VuID09PSBcInN0cmluZ1wiKSB7XG4gICAgICB0b2tlbiA9IHRva2VuLnRyaW0oKTtcbiAgICB9XG4gICAgZmFsbGJhY2tUb2tlbiA9IHRva2VuIHx8IHVuZGVmaW5lZDtcbiAgICBpZiAoIXRva2VuKSB7XG4gICAgICBjbGVhclRva2VuKCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKCFoYXNMb2NhbFN0b3JhZ2UoKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICB3aW5kb3cubG9jYWxTdG9yYWdlLnNldEl0ZW0oc3RvcmFnZUtleSwgdG9rZW4pO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHBlcnNpc3QgSldUIGluIGxvY2FsU3RvcmFnZVwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHJlYWRTdG9yZWRUb2tlbigpIHtcbiAgICBpZiAoIWhhc0xvY2FsU3RvcmFnZSgpKSB7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICBjb25zdCBzdG9yZWQgPSB3aW5kb3cubG9jYWxTdG9yYWdlLmdldEl0ZW0oc3RvcmFnZUtleSk7XG4gICAgICByZXR1cm4gc3RvcmVkIHx8IHVuZGVmaW5lZDtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byByZWFkIEpXVCBmcm9tIGxvY2FsU3RvcmFnZVwiLCBlcnIpO1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBjbGVhclRva2VuKCkge1xuICAgIGZhbGxiYWNrVG9rZW4gPSB1bmRlZmluZWQ7XG5cbiAgICBpZiAoIWhhc0xvY2FsU3RvcmFnZSgpKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2UucmVtb3ZlSXRlbShzdG9yYWdlS2V5KTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byBjbGVhciBKV1QgZnJvbSBsb2NhbFN0b3JhZ2VcIiwgZXJyKTtcbiAgICB9XG4gIH1cblxuICBpZiAoZmFsbGJhY2tUb2tlbikge1xuICAgIHBlcnNpc3RUb2tlbihmYWxsYmFja1Rva2VuKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGdldEp3dCgpIHtcbiAgICBjb25zdCBzdG9yZWQgPSByZWFkU3RvcmVkVG9rZW4oKTtcbiAgICBpZiAoc3RvcmVkKSB7XG4gICAgICByZXR1cm4gc3RvcmVkO1xuICAgIH1cbiAgICBpZiAoZmFsbGJhY2tUb2tlbikge1xuICAgICAgcmV0dXJuIGZhbGxiYWNrVG9rZW47XG4gICAgfVxuICAgIHRocm93IG5ldyBFcnJvcihcIk1pc3NpbmcgSldUIGZvciBjaGF0IGF1dGhlbnRpY2F0aW9uLlwiKTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgZ2V0Snd0LFxuICAgIHBlcnNpc3RUb2tlbixcbiAgICBjbGVhclRva2VuLFxuICAgIHN0b3JhZ2VLZXksXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgYXBpVXJsIH0gZnJvbSBcIi4uL2NvbmZpZy5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlSHR0cFNlcnZpY2UoeyBjb25maWcsIGF1dGggfSkge1xuICBhc3luYyBmdW5jdGlvbiBhdXRob3Jpc2VkRmV0Y2gocGF0aCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgbGV0IGp3dDtcbiAgICB0cnkge1xuICAgICAgand0ID0gYXdhaXQgYXV0aC5nZXRKd3QoKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIC8vIFN1cmZhY2UgYSBjb25zaXN0ZW50IGVycm9yIGFuZCBwcmVzZXJ2ZSBhYm9ydCBzZW1hbnRpY3NcbiAgICAgIHRocm93IG5ldyBFcnJvcihcIkF1dGhvcml6YXRpb24gZmFpbGVkOiBtaXNzaW5nIG9yIHVucmVhZGFibGUgSldUXCIpO1xuICAgIH1cbiAgICBjb25zdCBoZWFkZXJzID0gbmV3IEhlYWRlcnMob3B0aW9ucy5oZWFkZXJzIHx8IHt9KTtcbiAgICBpZiAoIWhlYWRlcnMuaGFzKFwiQXV0aG9yaXphdGlvblwiKSkge1xuICAgICAgaGVhZGVycy5zZXQoXCJBdXRob3JpemF0aW9uXCIsIGBCZWFyZXIgJHtqd3R9YCk7XG4gICAgfVxuICAgIHJldHVybiBmZXRjaChhcGlVcmwoY29uZmlnLCBwYXRoKSwgeyAuLi5vcHRpb25zLCBoZWFkZXJzIH0pO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZmV0Y2hUaWNrZXQoKSB7XG4gICAgY29uc3QgcmVzcCA9IGF3YWl0IGF1dGhvcmlzZWRGZXRjaChcIi9hcGkvdjEvYXV0aC93cy90aWNrZXRcIiwge1xuICAgICAgbWV0aG9kOiBcIlBPU1RcIixcbiAgICB9KTtcbiAgICBpZiAoIXJlc3Aub2spIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVGlja2V0IGVycm9yOiAke3Jlc3Auc3RhdHVzfWApO1xuICAgIH1cbiAgICBjb25zdCBib2R5ID0gYXdhaXQgcmVzcC5qc29uKCk7XG4gICAgaWYgKCFib2R5IHx8ICFib2R5LnRpY2tldCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFwiVGlja2V0IHJlc3BvbnNlIGludmFsaWRlXCIpO1xuICAgIH1cbiAgICByZXR1cm4gYm9keS50aWNrZXQ7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBwb3N0Q2hhdChtZXNzYWdlKSB7XG4gICAgY29uc3QgcmVzcCA9IGF3YWl0IGF1dGhvcmlzZWRGZXRjaChcIi9hcGkvdjEvY29udmVyc2F0aW9uL2NoYXRcIiwge1xuICAgICAgbWV0aG9kOiBcIlBPU1RcIixcbiAgICAgIGhlYWRlcnM6IHsgXCJDb250ZW50LVR5cGVcIjogXCJhcHBsaWNhdGlvbi9qc29uXCIgfSxcbiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHsgbWVzc2FnZSB9KSxcbiAgICB9KTtcbiAgICBpZiAoIXJlc3Aub2spIHtcbiAgICAgIGNvbnN0IHBheWxvYWQgPSBhd2FpdCByZXNwLnRleHQoKTtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgSFRUUCAke3Jlc3Auc3RhdHVzfTogJHtwYXlsb2FkfWApO1xuICAgIH1cbiAgICByZXR1cm4gcmVzcDtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIHBvc3RTdWdnZXN0aW9ucyhwcm9tcHQpIHtcbiAgICBjb25zdCByZXNwID0gYXdhaXQgYXV0aG9yaXNlZEZldGNoKFwiL2FwaS92MS91aS9zdWdnZXN0aW9uc1wiLCB7XG4gICAgICBtZXRob2Q6IFwiUE9TVFwiLFxuICAgICAgaGVhZGVyczogeyBcIkNvbnRlbnQtVHlwZVwiOiBcImFwcGxpY2F0aW9uL2pzb25cIiB9LFxuICAgICAgYm9keTogSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICBwcm9tcHQsXG4gICAgICAgIGFjdGlvbnM6IFtcImNvZGVcIiwgXCJzdW1tYXJpemVcIiwgXCJleHBsYWluXCJdLFxuICAgICAgfSksXG4gICAgfSk7XG4gICAgaWYgKCFyZXNwLm9rKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFN1Z2dlc3Rpb24gZXJyb3I6ICR7cmVzcC5zdGF0dXN9YCk7XG4gICAgfVxuICAgIHJldHVybiByZXNwLmpzb24oKTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgZmV0Y2hUaWNrZXQsXG4gICAgcG9zdENoYXQsXG4gICAgcG9zdFN1Z2dlc3Rpb25zLFxuICB9O1xufVxuIiwgImltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuLi91dGlscy90aW1lLmpzXCI7XG5cbmZ1bmN0aW9uIGJ1aWxkRXhwb3J0RmlsZW5hbWUoZXh0ZW5zaW9uKSB7XG4gIGNvbnN0IHN0YW1wID0gbm93SVNPKCkucmVwbGFjZSgvWzouXS9nLCBcIi1cIik7XG4gIHJldHVybiBgbW9uZ2Fycy1jaGF0LSR7c3RhbXB9LiR7ZXh0ZW5zaW9ufWA7XG59XG5cbmZ1bmN0aW9uIGJ1aWxkTWFya2Rvd25FeHBvcnQoaXRlbXMpIHtcbiAgY29uc3QgbGluZXMgPSBbXCIjIEhpc3RvcmlxdWUgZGUgY29udmVyc2F0aW9uIG1vbkdBUlNcIiwgXCJcIl07XG4gIGl0ZW1zLmZvckVhY2goKGl0ZW0pID0+IHtcbiAgICBjb25zdCByb2xlID0gaXRlbS5yb2xlID8gaXRlbS5yb2xlLnRvVXBwZXJDYXNlKCkgOiBcIk1FU1NBR0VcIjtcbiAgICBsaW5lcy5wdXNoKGAjIyAke3JvbGV9YCk7XG4gICAgaWYgKGl0ZW0udGltZXN0YW1wKSB7XG4gICAgICBsaW5lcy5wdXNoKGAqSG9yb2RhdGFnZVx1MDBBMDoqICR7aXRlbS50aW1lc3RhbXB9YCk7XG4gICAgfVxuICAgIGlmIChpdGVtLm1ldGFkYXRhICYmIE9iamVjdC5rZXlzKGl0ZW0ubWV0YWRhdGEpLmxlbmd0aCA+IDApIHtcbiAgICAgIE9iamVjdC5lbnRyaWVzKGl0ZW0ubWV0YWRhdGEpLmZvckVhY2goKFtrZXksIHZhbHVlXSkgPT4ge1xuICAgICAgICBsaW5lcy5wdXNoKGAqJHtrZXl9XHUwMEEwOiogJHt2YWx1ZX1gKTtcbiAgICAgIH0pO1xuICAgIH1cbiAgICBsaW5lcy5wdXNoKFwiXCIpO1xuICAgIGxpbmVzLnB1c2goaXRlbS50ZXh0IHx8IFwiXCIpO1xuICAgIGxpbmVzLnB1c2goXCJcIik7XG4gIH0pO1xuICByZXR1cm4gbGluZXMuam9pbihcIlxcblwiKTtcbn1cblxuZnVuY3Rpb24gZG93bmxvYWRCbG9iKGZpbGVuYW1lLCB0ZXh0LCB0eXBlKSB7XG4gIGlmICghd2luZG93LlVSTCB8fCB0eXBlb2Ygd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwgIT09IFwiZnVuY3Rpb25cIikge1xuICAgIGNvbnNvbGUud2FybihcIkJsb2IgZXhwb3J0IHVuc3VwcG9ydGVkIGluIHRoaXMgZW52aXJvbm1lbnRcIik7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbdGV4dF0sIHsgdHlwZSB9KTtcbiAgY29uc3QgdXJsID0gVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTtcbiAgY29uc3QgYW5jaG9yID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImFcIik7XG4gIGFuY2hvci5ocmVmID0gdXJsO1xuICBhbmNob3IuZG93bmxvYWQgPSBmaWxlbmFtZTtcbiAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChhbmNob3IpO1xuICBhbmNob3IuY2xpY2soKTtcbiAgZG9jdW1lbnQuYm9keS5yZW1vdmVDaGlsZChhbmNob3IpO1xuICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiBVUkwucmV2b2tlT2JqZWN0VVJMKHVybCksIDApO1xuICByZXR1cm4gdHJ1ZTtcbn1cblxuYXN5bmMgZnVuY3Rpb24gY29weVRvQ2xpcGJvYXJkKHRleHQpIHtcbiAgaWYgKCF0ZXh0KSByZXR1cm4gZmFsc2U7XG4gIHRyeSB7XG4gICAgaWYgKG5hdmlnYXRvci5jbGlwYm9hcmQgJiYgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQpIHtcbiAgICAgIGF3YWl0IG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KHRleHQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB0ZXh0YXJlYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJ0ZXh0YXJlYVwiKTtcbiAgICAgIHRleHRhcmVhLnZhbHVlID0gdGV4dDtcbiAgICAgIHRleHRhcmVhLnNldEF0dHJpYnV0ZShcInJlYWRvbmx5XCIsIFwicmVhZG9ubHlcIik7XG4gICAgICB0ZXh0YXJlYS5zdHlsZS5wb3NpdGlvbiA9IFwiYWJzb2x1dGVcIjtcbiAgICAgIHRleHRhcmVhLnN0eWxlLmxlZnQgPSBcIi05OTk5cHhcIjtcbiAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgdGV4dGFyZWEuc2VsZWN0KCk7XG4gICAgICBkb2N1bWVudC5leGVjQ29tbWFuZChcImNvcHlcIik7XG4gICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKHRleHRhcmVhKTtcbiAgICB9XG4gICAgcmV0dXJuIHRydWU7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIGNvbnNvbGUud2FybihcIkNvcHkgY29udmVyc2F0aW9uIGZhaWxlZFwiLCBlcnIpO1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRXhwb3J0ZXIoeyB0aW1lbGluZVN0b3JlLCBhbm5vdW5jZSB9KSB7XG4gIGZ1bmN0aW9uIGNvbGxlY3RUcmFuc2NyaXB0KCkge1xuICAgIHJldHVybiB0aW1lbGluZVN0b3JlLmNvbGxlY3QoKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGV4cG9ydENvbnZlcnNhdGlvbihmb3JtYXQpIHtcbiAgICBjb25zdCBpdGVtcyA9IGNvbGxlY3RUcmFuc2NyaXB0KCk7XG4gICAgaWYgKCFpdGVtcy5sZW5ndGgpIHtcbiAgICAgIGFubm91bmNlKFwiQXVjdW4gbWVzc2FnZSBcdTAwRTAgZXhwb3J0ZXIuXCIsIFwid2FybmluZ1wiKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKGZvcm1hdCA9PT0gXCJqc29uXCIpIHtcbiAgICAgIGNvbnN0IHBheWxvYWQgPSB7XG4gICAgICAgIGV4cG9ydGVkX2F0OiBub3dJU08oKSxcbiAgICAgICAgY291bnQ6IGl0ZW1zLmxlbmd0aCxcbiAgICAgICAgaXRlbXMsXG4gICAgICB9O1xuICAgICAgaWYgKFxuICAgICAgICBkb3dubG9hZEJsb2IoXG4gICAgICAgICAgYnVpbGRFeHBvcnRGaWxlbmFtZShcImpzb25cIiksXG4gICAgICAgICAgSlNPTi5zdHJpbmdpZnkocGF5bG9hZCwgbnVsbCwgMiksXG4gICAgICAgICAgXCJhcHBsaWNhdGlvbi9qc29uXCIsXG4gICAgICAgIClcbiAgICAgICkge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBKU09OIGdcdTAwRTluXHUwMEU5clx1MDBFOS5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYW5ub3VuY2UoXCJFeHBvcnQgbm9uIHN1cHBvcnRcdTAwRTkgZGFucyBjZSBuYXZpZ2F0ZXVyLlwiLCBcImRhbmdlclwiKTtcbiAgICAgIH1cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKGZvcm1hdCA9PT0gXCJtYXJrZG93blwiKSB7XG4gICAgICBpZiAoXG4gICAgICAgIGRvd25sb2FkQmxvYihcbiAgICAgICAgICBidWlsZEV4cG9ydEZpbGVuYW1lKFwibWRcIiksXG4gICAgICAgICAgYnVpbGRNYXJrZG93bkV4cG9ydChpdGVtcyksXG4gICAgICAgICAgXCJ0ZXh0L21hcmtkb3duXCIsXG4gICAgICAgIClcbiAgICAgICkge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBNYXJrZG93biBnXHUwMEU5blx1MDBFOXJcdTAwRTkuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGFubm91bmNlKFwiRXhwb3J0IG5vbiBzdXBwb3J0XHUwMEU5IGRhbnMgY2UgbmF2aWdhdGV1ci5cIiwgXCJkYW5nZXJcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gY29weUNvbnZlcnNhdGlvblRvQ2xpcGJvYXJkKCkge1xuICAgIGNvbnN0IGl0ZW1zID0gY29sbGVjdFRyYW5zY3JpcHQoKTtcbiAgICBpZiAoIWl0ZW1zLmxlbmd0aCkge1xuICAgICAgYW5ub3VuY2UoXCJBdWN1biBtZXNzYWdlIFx1MDBFMCBjb3BpZXIuXCIsIFwid2FybmluZ1wiKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdGV4dCA9IGJ1aWxkTWFya2Rvd25FeHBvcnQoaXRlbXMpO1xuICAgIGlmIChhd2FpdCBjb3B5VG9DbGlwYm9hcmQodGV4dCkpIHtcbiAgICAgIGFubm91bmNlKFwiQ29udmVyc2F0aW9uIGNvcGlcdTAwRTllIGF1IHByZXNzZS1wYXBpZXJzLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgfSBlbHNlIHtcbiAgICAgIGFubm91bmNlKFwiSW1wb3NzaWJsZSBkZSBjb3BpZXIgbGEgY29udmVyc2F0aW9uLlwiLCBcImRhbmdlclwiKTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGV4cG9ydENvbnZlcnNhdGlvbixcbiAgICBjb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgbm93SVNPIH0gZnJvbSBcIi4uL3V0aWxzL3RpbWUuanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVNvY2tldENsaWVudCh7IGNvbmZpZywgaHR0cCwgdWksIG9uRXZlbnQgfSkge1xuICBsZXQgd3M7XG4gIGxldCB3c0hCZWF0O1xuICBsZXQgcmVjb25uZWN0QmFja29mZiA9IDUwMDtcbiAgY29uc3QgQkFDS09GRl9NQVggPSA4MDAwO1xuICBsZXQgcmV0cnlUaW1lciA9IG51bGw7XG4gIGxldCBkaXNwb3NlZCA9IGZhbHNlO1xuXG4gIGZ1bmN0aW9uIGNsZWFySGVhcnRiZWF0KCkge1xuICAgIGlmICh3c0hCZWF0KSB7XG4gICAgICBjbGVhckludGVydmFsKHdzSEJlYXQpO1xuICAgICAgd3NIQmVhdCA9IG51bGw7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2NoZWR1bGVSZWNvbm5lY3QoZGVsYXlCYXNlKSB7XG4gICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICByZXR1cm4gMDtcbiAgICB9XG4gICAgY29uc3Qgaml0dGVyID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICogMjUwKTtcbiAgICBjb25zdCBkZWxheSA9IE1hdGgubWluKEJBQ0tPRkZfTUFYLCBkZWxheUJhc2UgKyBqaXR0ZXIpO1xuICAgIGlmIChyZXRyeVRpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQocmV0cnlUaW1lcik7XG4gICAgfVxuICAgIHJldHJ5VGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICByZXRyeVRpbWVyID0gbnVsbDtcbiAgICAgIHJlY29ubmVjdEJhY2tvZmYgPSBNYXRoLm1pbihcbiAgICAgICAgQkFDS09GRl9NQVgsXG4gICAgICAgIE1hdGgubWF4KDUwMCwgcmVjb25uZWN0QmFja29mZiAqIDIpLFxuICAgICAgKTtcbiAgICAgIHZvaWQgb3BlblNvY2tldCgpO1xuICAgIH0sIGRlbGF5KTtcbiAgICByZXR1cm4gZGVsYXk7XG4gIH1cblxuICBmdW5jdGlvbiBzYWZlU2VuZChvYmopIHtcbiAgICB0cnkge1xuICAgICAgaWYgKHdzICYmIHdzLnJlYWR5U3RhdGUgPT09IFdlYlNvY2tldC5PUEVOKSB7XG4gICAgICAgIHdzLnNlbmQoSlNPTi5zdHJpbmdpZnkob2JqKSk7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gc2VuZCBvdmVyIFdlYlNvY2tldFwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIG9wZW5Tb2NrZXQoKSB7XG4gICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiT2J0ZW50aW9uIGRcdTIwMTl1biB0aWNrZXQgZGUgY29ubmV4aW9uXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgICAgIGNvbnN0IHRpY2tldCA9IGF3YWl0IGh0dHAuZmV0Y2hUaWNrZXQoKTtcbiAgICAgIGlmIChkaXNwb3NlZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IHdzVXJsID0gbmV3IFVSTChcIi93cy9jaGF0L1wiLCBjb25maWcuYmFzZVVybCk7XG4gICAgICB3c1VybC5wcm90b2NvbCA9IGNvbmZpZy5iYXNlVXJsLnByb3RvY29sID09PSBcImh0dHBzOlwiID8gXCJ3c3M6XCIgOiBcIndzOlwiO1xuICAgICAgd3NVcmwuc2VhcmNoUGFyYW1zLnNldChcInRcIiwgdGlja2V0KTtcblxuICAgICAgaWYgKHdzKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgd3MuY2xvc2UoKTtcbiAgICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFwiV2ViU29ja2V0IGNsb3NlIGJlZm9yZSByZWNvbm5lY3QgZmFpbGVkXCIsIGVycik7XG4gICAgICAgIH1cbiAgICAgICAgd3MgPSBudWxsO1xuICAgICAgfVxuXG4gICAgICB3cyA9IG5ldyBXZWJTb2NrZXQod3NVcmwudG9TdHJpbmcoKSk7XG4gICAgICB1aS5zZXRXc1N0YXR1cyhcImNvbm5lY3RpbmdcIik7XG4gICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIkNvbm5leGlvbiBhdSBzZXJ2ZXVyXHUyMDI2XCIsIFwiaW5mb1wiKTtcblxuICAgICAgd3Mub25vcGVuID0gKCkgPT4ge1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgaWYgKHJldHJ5VGltZXIpIHtcbiAgICAgICAgICBjbGVhclRpbWVvdXQocmV0cnlUaW1lcik7XG4gICAgICAgICAgcmV0cnlUaW1lciA9IG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmVjb25uZWN0QmFja29mZiA9IDUwMDtcbiAgICAgICAgY29uc3QgY29ubmVjdGVkQXQgPSBub3dJU08oKTtcbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJvbmxpbmVcIik7XG4gICAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFxuICAgICAgICAgIGBDb25uZWN0XHUwMEU5IGxlICR7dWkuZm9ybWF0VGltZXN0YW1wKGNvbm5lY3RlZEF0KX1gLFxuICAgICAgICAgIFwic3VjY2Vzc1wiLFxuICAgICAgICApO1xuICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGNvbm5lY3RlZEF0LCBsYXN0TWVzc2FnZUF0OiBjb25uZWN0ZWRBdCB9KTtcbiAgICAgICAgdWkuaGlkZUVycm9yKCk7XG4gICAgICAgIGNsZWFySGVhcnRiZWF0KCk7XG4gICAgICAgIHdzSEJlYXQgPSB3aW5kb3cuc2V0SW50ZXJ2YWwoKCkgPT4ge1xuICAgICAgICAgIHNhZmVTZW5kKHsgdHlwZTogXCJjbGllbnQucGluZ1wiLCB0czogbm93SVNPKCkgfSk7XG4gICAgICAgIH0sIDIwMDAwKTtcbiAgICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJDb25uZWN0XHUwMEU5LiBWb3VzIHBvdXZleiBcdTAwRTljaGFuZ2VyLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgfTtcblxuICAgICAgd3Mub25tZXNzYWdlID0gKGV2dCkgPT4ge1xuICAgICAgICBjb25zdCByZWNlaXZlZEF0ID0gbm93SVNPKCk7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgZXYgPSBKU09OLnBhcnNlKGV2dC5kYXRhKTtcbiAgICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHJlY2VpdmVkQXQgfSk7XG4gICAgICAgICAgb25FdmVudChldik7XG4gICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgIGNvbnNvbGUuZXJyb3IoXCJCYWQgZXZlbnQgcGF5bG9hZFwiLCBlcnIsIGV2dC5kYXRhKTtcbiAgICAgICAgfVxuICAgICAgfTtcblxuICAgICAgd3Mub25jbG9zZSA9ICgpID0+IHtcbiAgICAgICAgY2xlYXJIZWFydGJlYXQoKTtcbiAgICAgICAgd3MgPSBudWxsO1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJvZmZsaW5lXCIpO1xuICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGxhdGVuY3lNczogdW5kZWZpbmVkIH0pO1xuICAgICAgICBjb25zdCBkZWxheSA9IHNjaGVkdWxlUmVjb25uZWN0KHJlY29ubmVjdEJhY2tvZmYpO1xuICAgICAgICBjb25zdCBzZWNvbmRzID0gTWF0aC5tYXgoMSwgTWF0aC5yb3VuZChkZWxheSAvIDEwMDApKTtcbiAgICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEoXG4gICAgICAgICAgYERcdTAwRTljb25uZWN0XHUwMEU5LiBOb3V2ZWxsZSB0ZW50YXRpdmUgZGFucyAke3NlY29uZHN9IHNcdTIwMjZgLFxuICAgICAgICAgIFwid2FybmluZ1wiLFxuICAgICAgICApO1xuICAgICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgICBcIkNvbm5leGlvbiBwZXJkdWUuIFJlY29ubmV4aW9uIGF1dG9tYXRpcXVlXHUyMDI2XCIsXG4gICAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICAgICk7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfTtcblxuICAgICAgd3Mub25lcnJvciA9IChlcnIpID0+IHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIldlYlNvY2tldCBlcnJvclwiLCBlcnIpO1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJlcnJvclwiLCBcIkVycmV1ciBXZWJTb2NrZXRcIik7XG4gICAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiRXJyZXVyIFdlYlNvY2tldCBkXHUwMEU5dGVjdFx1MDBFOWUuXCIsIFwiZGFuZ2VyXCIpO1xuICAgICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcIlVuZSBlcnJldXIgclx1MDBFOXNlYXUgZXN0IHN1cnZlbnVlLlwiLCBcImRhbmdlclwiKTtcbiAgICAgICAgdWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNjAwMCk7XG4gICAgICB9O1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS5lcnJvcihlcnIpO1xuICAgICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGNvbnN0IG1lc3NhZ2UgPSBlcnIgaW5zdGFuY2VvZiBFcnJvciA/IGVyci5tZXNzYWdlIDogU3RyaW5nKGVycik7XG4gICAgICB1aS5zZXRXc1N0YXR1cyhcImVycm9yXCIsIG1lc3NhZ2UpO1xuICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEobWVzc2FnZSwgXCJkYW5nZXJcIik7XG4gICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgXCJDb25uZXhpb24gaW5kaXNwb25pYmxlLiBOb3V2ZWwgZXNzYWkgYmllbnRcdTAwRjR0LlwiLFxuICAgICAgICBcImRhbmdlclwiLFxuICAgICAgKTtcbiAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgc2NoZWR1bGVSZWNvbm5lY3QocmVjb25uZWN0QmFja29mZik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZGlzcG9zZSgpIHtcbiAgICBkaXNwb3NlZCA9IHRydWU7XG4gICAgaWYgKHJldHJ5VGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChyZXRyeVRpbWVyKTtcbiAgICAgIHJldHJ5VGltZXIgPSBudWxsO1xuICAgIH1cbiAgICBjbGVhckhlYXJ0YmVhdCgpO1xuICAgIGlmICh3cykge1xuICAgICAgdHJ5IHtcbiAgICAgICAgd3MuY2xvc2UoKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICBjb25zb2xlLndhcm4oXCJXZWJTb2NrZXQgY2xvc2UgZHVyaW5nIGRpc3Bvc2UgZmFpbGVkXCIsIGVycik7XG4gICAgICB9XG4gICAgICB3cyA9IG51bGw7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBvcGVuOiBvcGVuU29ja2V0LFxuICAgIHNlbmQ6IHNhZmVTZW5kLFxuICAgIGRpc3Bvc2UsXG4gIH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlKHsgaHR0cCwgdWkgfSkge1xuICBsZXQgdGltZXIgPSBudWxsO1xuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlKHByb21wdCkge1xuICAgIGlmICh0aW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHRpbWVyKTtcbiAgICB9XG4gICAgdGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCksIDIyMCk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCkge1xuICAgIGlmICghcHJvbXB0IHx8IHByb21wdC50cmltKCkubGVuZ3RoIDwgMykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0cnkge1xuICAgICAgY29uc3QgcGF5bG9hZCA9IGF3YWl0IGh0dHAucG9zdFN1Z2dlc3Rpb25zKHByb21wdC50cmltKCkpO1xuICAgICAgaWYgKHBheWxvYWQgJiYgQXJyYXkuaXNBcnJheShwYXlsb2FkLmFjdGlvbnMpKSB7XG4gICAgICAgIHVpLmFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhwYXlsb2FkLmFjdGlvbnMpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcIkFVSSBzdWdnZXN0aW9uIGZldGNoIGZhaWxlZFwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgc2NoZWR1bGUsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgY3JlYXRlRW1pdHRlciB9IGZyb20gXCIuLi91dGlscy9lbWl0dGVyLmpzXCI7XG5pbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBub3JtYWxpemVUZXh0KHZhbHVlKSB7XG4gIGlmICghdmFsdWUpIHtcbiAgICByZXR1cm4gXCJcIjtcbiAgfVxuICByZXR1cm4gU3RyaW5nKHZhbHVlKS5yZXBsYWNlKC9cXHMrL2csIFwiIFwiKS50cmltKCk7XG59XG5cbmZ1bmN0aW9uIGRlc2NyaWJlUmVjb2duaXRpb25FcnJvcihjb2RlLCBmYWxsYmFjayA9IFwiXCIpIHtcbiAgc3dpdGNoIChjb2RlKSB7XG4gICAgY2FzZSBcIm5vdC1hbGxvd2VkXCI6XG4gICAgY2FzZSBcInNlcnZpY2Utbm90LWFsbG93ZWRcIjpcbiAgICAgIHJldHVybiAoXG4gICAgICAgIFwiQWNjXHUwMEU4cyBhdSBtaWNyb3Bob25lIHJlZnVzXHUwMEU5LiBBdXRvcmlzZXogbGEgZGljdFx1MDBFOWUgdm9jYWxlIGRhbnMgdm90cmUgbmF2aWdhdGV1ci5cIlxuICAgICAgKTtcbiAgICBjYXNlIFwibmV0d29ya1wiOlxuICAgICAgcmV0dXJuIFwiTGEgcmVjb25uYWlzc2FuY2Ugdm9jYWxlIGEgXHUwMEU5dFx1MDBFOSBpbnRlcnJvbXB1ZSBwYXIgdW4gcHJvYmxcdTAwRThtZSByXHUwMEU5c2VhdS5cIjtcbiAgICBjYXNlIFwibm8tc3BlZWNoXCI6XG4gICAgICByZXR1cm4gXCJBdWN1bmUgdm9peCBkXHUwMEU5dGVjdFx1MDBFOWUuIEVzc2F5ZXogZGUgcGFybGVyIHBsdXMgcHJcdTAwRThzIGR1IG1pY3JvLlwiO1xuICAgIGNhc2UgXCJhYm9ydGVkXCI6XG4gICAgICByZXR1cm4gXCJMYSBkaWN0XHUwMEU5ZSBhIFx1MDBFOXRcdTAwRTkgaW50ZXJyb21wdWUuXCI7XG4gICAgY2FzZSBcImF1ZGlvLWNhcHR1cmVcIjpcbiAgICAgIHJldHVybiBcIkF1Y3VuIG1pY3JvcGhvbmUgZGlzcG9uaWJsZS4gVlx1MDBFOXJpZmlleiB2b3RyZSBtYXRcdTAwRTlyaWVsLlwiO1xuICAgIGNhc2UgXCJiYWQtZ3JhbW1hclwiOlxuICAgICAgcmV0dXJuIFwiTGUgc2VydmljZSBkZSBkaWN0XHUwMEU5ZSBhIHJlbmNvbnRyXHUwMEU5IHVuZSBlcnJldXIgZGUgdHJhaXRlbWVudC5cIjtcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGZhbGxiYWNrIHx8IFwiTGEgcmVjb25uYWlzc2FuY2Ugdm9jYWxlIGEgcmVuY29udHJcdTAwRTkgdW5lIGVycmV1ciBpbmF0dGVuZHVlLlwiO1xuICB9XG59XG5cbmZ1bmN0aW9uIG1hcFZvaWNlKHZvaWNlKSB7XG4gIHJldHVybiB7XG4gICAgbmFtZTogdm9pY2UubmFtZSxcbiAgICBsYW5nOiB2b2ljZS5sYW5nLFxuICAgIHZvaWNlVVJJOiB2b2ljZS52b2ljZVVSSSxcbiAgICBkZWZhdWx0OiBCb29sZWFuKHZvaWNlLmRlZmF1bHQpLFxuICAgIGxvY2FsU2VydmljZTogQm9vbGVhbih2b2ljZS5sb2NhbFNlcnZpY2UpLFxuICB9O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3BlZWNoU2VydmljZSh7IGRlZmF1bHRMYW5ndWFnZSB9ID0ge30pIHtcbiAgY29uc3QgZW1pdHRlciA9IGNyZWF0ZUVtaXR0ZXIoKTtcbiAgY29uc3QgZ2xvYmFsU2NvcGUgPSB0eXBlb2Ygd2luZG93ICE9PSBcInVuZGVmaW5lZFwiID8gd2luZG93IDoge307XG4gIGNvbnN0IFJlY29nbml0aW9uQ3RvciA9XG4gICAgZ2xvYmFsU2NvcGUuU3BlZWNoUmVjb2duaXRpb24gfHwgZ2xvYmFsU2NvcGUud2Via2l0U3BlZWNoUmVjb2duaXRpb24gfHwgbnVsbDtcbiAgY29uc3QgcmVjb2duaXRpb25TdXBwb3J0ZWQgPSBCb29sZWFuKFJlY29nbml0aW9uQ3Rvcik7XG4gIGNvbnN0IHN5bnRoZXNpc1N1cHBvcnRlZCA9IEJvb2xlYW4oZ2xvYmFsU2NvcGUuc3BlZWNoU3ludGhlc2lzKTtcbiAgY29uc3Qgc3ludGggPSBzeW50aGVzaXNTdXBwb3J0ZWQgPyBnbG9iYWxTY29wZS5zcGVlY2hTeW50aGVzaXMgOiBudWxsO1xuXG4gIGxldCByZWNvZ25pdGlvbiA9IG51bGw7XG4gIGNvbnN0IG5hdmlnYXRvckxhbmd1YWdlID1cbiAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgPyBuYXZpZ2F0b3IubGFuZ3VhZ2VcbiAgICAgIDogbnVsbDtcbiAgbGV0IHJlY29nbml0aW9uTGFuZyA9XG4gICAgZGVmYXVsdExhbmd1YWdlIHx8IG5hdmlnYXRvckxhbmd1YWdlIHx8IFwiZnItQ0FcIjtcbiAgbGV0IG1hbnVhbFN0b3AgPSBmYWxzZTtcbiAgbGV0IGxpc3RlbmluZyA9IGZhbHNlO1xuICBsZXQgc3BlYWtpbmcgPSBmYWxzZTtcbiAgbGV0IHByZWZlcnJlZFZvaWNlVVJJID0gbnVsbDtcbiAgbGV0IHZvaWNlc0NhY2hlID0gW107XG4gIGxldCBtaWNyb3Bob25lUHJpbWVkID0gZmFsc2U7XG4gIGxldCBtaWNyb3Bob25lUHJpbWluZyA9IG51bGw7XG5cbiAgY29uc3QgdXNlckFnZW50ID1cbiAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci51c2VyQWdlbnRcbiAgICAgID8gbmF2aWdhdG9yLnVzZXJBZ2VudC50b0xvd2VyQ2FzZSgpXG4gICAgICA6IFwiXCI7XG4gIGNvbnN0IGlzQXBwbGVNb2JpbGUgPSAvaXBob25lfGlwYWR8aXBvZC8udGVzdCh1c2VyQWdlbnQpO1xuICBjb25zdCBpc1NhZmFyaSA9XG4gICAgL3NhZmFyaS8udGVzdCh1c2VyQWdlbnQpICYmXG4gICAgIS9jcmlvc3xmeGlvc3xjaHJvbWV8YW5kcm9pZHxlZGdlfGVkZ3xvcHJ8b3BlcmEvLnRlc3QodXNlckFnZW50KTtcblxuICBmdW5jdGlvbiByZXF1aXJlc01pY3JvcGhvbmVQcmltaW5nKCkge1xuICAgIGlmICghcmVjb2duaXRpb25TdXBwb3J0ZWQpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgaWYgKG1pY3JvcGhvbmVQcmltZWQpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgaWYgKHR5cGVvZiBuYXZpZ2F0b3IgPT09IFwidW5kZWZpbmVkXCIpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgaWYgKFxuICAgICAgIW5hdmlnYXRvci5tZWRpYURldmljZXMgfHxcbiAgICAgIHR5cGVvZiBuYXZpZ2F0b3IubWVkaWFEZXZpY2VzLmdldFVzZXJNZWRpYSAhPT0gXCJmdW5jdGlvblwiXG4gICAgKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIHJldHVybiBpc0FwcGxlTW9iaWxlICYmIGlzU2FmYXJpO1xuICB9XG5cbiAgZnVuY3Rpb24gcmVsZWFzZVN0cmVhbShzdHJlYW0pIHtcbiAgICBpZiAoIXN0cmVhbSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCB0cmFja3MgPSB0eXBlb2Ygc3RyZWFtLmdldFRyYWNrcyA9PT0gXCJmdW5jdGlvblwiID8gc3RyZWFtLmdldFRyYWNrcygpIDogW107XG4gICAgdHJhY2tzLmZvckVhY2goKHRyYWNrKSA9PiB7XG4gICAgICB0cnkge1xuICAgICAgICB0cmFjay5zdG9wKCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcIlVuYWJsZSB0byBzdG9wIHRyYWNrXCIsIGVycik7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBlbnN1cmVNaWNyb3Bob25lQWNjZXNzKCkge1xuICAgIGlmICghcmVxdWlyZXNNaWNyb3Bob25lUHJpbWluZygpKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgaWYgKG1pY3JvcGhvbmVQcmltZWQpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICBpZiAobWljcm9waG9uZVByaW1pbmcpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIHJldHVybiBhd2FpdCBtaWNyb3Bob25lUHJpbWluZztcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG4gICAgfVxuICAgIG1pY3JvcGhvbmVQcmltaW5nID0gbmF2aWdhdG9yLm1lZGlhRGV2aWNlc1xuICAgICAgLmdldFVzZXJNZWRpYSh7IGF1ZGlvOiB0cnVlIH0pXG4gICAgICAudGhlbigoc3RyZWFtKSA9PiB7XG4gICAgICAgIG1pY3JvcGhvbmVQcmltZWQgPSB0cnVlO1xuICAgICAgICByZWxlYXNlU3RyZWFtKHN0cmVhbSk7XG4gICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgfSlcbiAgICAgIC5jYXRjaCgoZXJyKSA9PiB7XG4gICAgICAgIGVtaXRFcnJvcih7XG4gICAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgICAgY29kZTogXCJwZXJtaXNzaW9uLWRlbmllZFwiLFxuICAgICAgICAgIG1lc3NhZ2U6XG4gICAgICAgICAgICBcIkF1dG9yaXNhdGlvbiBkdSBtaWNyb3Bob25lIHJlZnVzXHUwMEU5ZS4gQWN0aXZleiBsJ2FjY1x1MDBFOHMgZGFucyBsZXMgclx1MDBFOWdsYWdlcyBkZSBTYWZhcmkuXCIsXG4gICAgICAgICAgZGV0YWlsczogZXJyLFxuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfSlcbiAgICAgIC5maW5hbGx5KCgpID0+IHtcbiAgICAgICAgbWljcm9waG9uZVByaW1pbmcgPSBudWxsO1xuICAgICAgfSk7XG4gICAgcmV0dXJuIG1pY3JvcGhvbmVQcmltaW5nO1xuICB9XG5cbiAgZnVuY3Rpb24gaXNQZXJtaXNzaW9uRXJyb3IoZXJyb3IpIHtcbiAgICBpZiAoIWVycm9yKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGNvbnN0IGNvZGUgPVxuICAgICAgdHlwZW9mIGVycm9yID09PSBcInN0cmluZ1wiXG4gICAgICAgID8gZXJyb3JcbiAgICAgICAgOiBlcnJvci5uYW1lIHx8IGVycm9yLmNvZGUgfHwgZXJyb3IubWVzc2FnZSB8fCBcIlwiO1xuICAgIGNvbnN0IG5vcm1hbGlzZWQgPSBTdHJpbmcoY29kZSkudG9Mb3dlckNhc2UoKTtcbiAgICByZXR1cm4gW1xuICAgICAgXCJub3RhbGxvd2VkZXJyb3JcIixcbiAgICAgIFwibm90LWFsbG93ZWRcIixcbiAgICAgIFwic2VydmljZS1ub3QtYWxsb3dlZFwiLFxuICAgICAgXCJzZWN1cml0eWVycm9yXCIsXG4gICAgICBcInBlcm1pc3Npb25kZW5pZWRlcnJvclwiLFxuICAgICAgXCJhYm9ydGVycm9yXCIsXG4gICAgXS5zb21lKChjYW5kaWRhdGUpID0+IG5vcm1hbGlzZWQuaW5jbHVkZXMoY2FuZGlkYXRlKSk7XG4gIH1cblxuICBmdW5jdGlvbiBlbWl0RXJyb3IocGF5bG9hZCkge1xuICAgIGNvbnN0IGVucmljaGVkID0ge1xuICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIC4uLnBheWxvYWQsXG4gICAgfTtcbiAgICBjb25zb2xlLmVycm9yKFwiU3BlZWNoIHNlcnZpY2UgZXJyb3JcIiwgZW5yaWNoZWQpO1xuICAgIGVtaXR0ZXIuZW1pdChcImVycm9yXCIsIGVucmljaGVkKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGVuc3VyZVJlY29nbml0aW9uKCkge1xuICAgIGlmICghcmVjb2duaXRpb25TdXBwb3J0ZWQpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBpZiAocmVjb2duaXRpb24pIHtcbiAgICAgIHJldHVybiByZWNvZ25pdGlvbjtcbiAgICB9XG4gICAgcmVjb2duaXRpb24gPSBuZXcgUmVjb2duaXRpb25DdG9yKCk7XG4gICAgcmVjb2duaXRpb24ubGFuZyA9IHJlY29nbml0aW9uTGFuZztcbiAgICByZWNvZ25pdGlvbi5jb250aW51b3VzID0gZmFsc2U7XG4gICAgcmVjb2duaXRpb24uaW50ZXJpbVJlc3VsdHMgPSB0cnVlO1xuICAgIHJlY29nbml0aW9uLm1heEFsdGVybmF0aXZlcyA9IDE7XG5cbiAgICByZWNvZ25pdGlvbi5vbnN0YXJ0ID0gKCkgPT4ge1xuICAgICAgbGlzdGVuaW5nID0gdHJ1ZTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcImxpc3RlbmluZy1jaGFuZ2VcIiwge1xuICAgICAgICBsaXN0ZW5pbmc6IHRydWUsXG4gICAgICAgIHJlYXNvbjogXCJzdGFydFwiLFxuICAgICAgICB0aW1lc3RhbXA6IG5vd0lTTygpLFxuICAgICAgfSk7XG4gICAgfTtcblxuICAgIHJlY29nbml0aW9uLm9uZW5kID0gKCkgPT4ge1xuICAgICAgY29uc3QgcmVhc29uID0gbWFudWFsU3RvcCA/IFwibWFudWFsXCIgOiBcImVuZGVkXCI7XG4gICAgICBsaXN0ZW5pbmcgPSBmYWxzZTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcImxpc3RlbmluZy1jaGFuZ2VcIiwge1xuICAgICAgICBsaXN0ZW5pbmc6IGZhbHNlLFxuICAgICAgICByZWFzb24sXG4gICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICB9KTtcbiAgICAgIG1hbnVhbFN0b3AgPSBmYWxzZTtcbiAgICB9O1xuXG4gICAgcmVjb2duaXRpb24ub25lcnJvciA9IChldmVudCkgPT4ge1xuICAgICAgbGlzdGVuaW5nID0gZmFsc2U7XG4gICAgICBjb25zdCBjb2RlID0gZXZlbnQuZXJyb3IgfHwgXCJ1bmtub3duXCI7XG4gICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICBzb3VyY2U6IFwicmVjb2duaXRpb25cIixcbiAgICAgICAgY29kZSxcbiAgICAgICAgbWVzc2FnZTogZGVzY3JpYmVSZWNvZ25pdGlvbkVycm9yKGNvZGUsIGV2ZW50Lm1lc3NhZ2UpLFxuICAgICAgICBldmVudCxcbiAgICAgIH0pO1xuICAgICAgZW1pdHRlci5lbWl0KFwibGlzdGVuaW5nLWNoYW5nZVwiLCB7XG4gICAgICAgIGxpc3RlbmluZzogZmFsc2UsXG4gICAgICAgIHJlYXNvbjogXCJlcnJvclwiLFxuICAgICAgICBjb2RlLFxuICAgICAgICB0aW1lc3RhbXA6IG5vd0lTTygpLFxuICAgICAgfSk7XG4gICAgfTtcblxuICAgIHJlY29nbml0aW9uLm9ucmVzdWx0ID0gKGV2ZW50KSA9PiB7XG4gICAgICBpZiAoIWV2ZW50LnJlc3VsdHMpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgZm9yIChsZXQgaSA9IGV2ZW50LnJlc3VsdEluZGV4OyBpIDwgZXZlbnQucmVzdWx0cy5sZW5ndGg7IGkgKz0gMSkge1xuICAgICAgICBjb25zdCByZXN1bHQgPSBldmVudC5yZXN1bHRzW2ldO1xuICAgICAgICBpZiAoIXJlc3VsdCB8fCByZXN1bHQubGVuZ3RoID09PSAwKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgYWx0ZXJuYXRpdmUgPSByZXN1bHRbMF07XG4gICAgICAgIGNvbnN0IHRyYW5zY3JpcHQgPSBub3JtYWxpemVUZXh0KGFsdGVybmF0aXZlPy50cmFuc2NyaXB0IHx8IFwiXCIpO1xuICAgICAgICBpZiAoIXRyYW5zY3JpcHQpIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICBlbWl0dGVyLmVtaXQoXCJ0cmFuc2NyaXB0XCIsIHtcbiAgICAgICAgICB0cmFuc2NyaXB0LFxuICAgICAgICAgIGlzRmluYWw6IEJvb2xlYW4ocmVzdWx0LmlzRmluYWwpLFxuICAgICAgICAgIGNvbmZpZGVuY2U6XG4gICAgICAgICAgICB0eXBlb2YgYWx0ZXJuYXRpdmUuY29uZmlkZW5jZSA9PT0gXCJudW1iZXJcIlxuICAgICAgICAgICAgICA/IGFsdGVybmF0aXZlLmNvbmZpZGVuY2VcbiAgICAgICAgICAgICAgOiBudWxsLFxuICAgICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH07XG5cbiAgICByZWNvZ25pdGlvbi5vbmF1ZGlvZW5kID0gKCkgPT4ge1xuICAgICAgZW1pdHRlci5lbWl0KFwiYXVkaW8tZW5kXCIsIHsgdGltZXN0YW1wOiBub3dJU08oKSB9KTtcbiAgICB9O1xuXG4gICAgcmVjb2duaXRpb24ub25zcGVlY2hlbmQgPSAoKSA9PiB7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJzcGVlY2gtZW5kXCIsIHsgdGltZXN0YW1wOiBub3dJU08oKSB9KTtcbiAgICB9O1xuXG4gICAgcmV0dXJuIHJlY29nbml0aW9uO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gc3RhcnRMaXN0ZW5pbmcob3B0aW9ucyA9IHt9KSB7XG4gICAgaWYgKCFyZWNvZ25pdGlvblN1cHBvcnRlZCkge1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgIGNvZGU6IFwidW5zdXBwb3J0ZWRcIixcbiAgICAgICAgbWVzc2FnZTogXCJMYSBkaWN0XHUwMEU5ZSB2b2NhbGUgbidlc3QgcGFzIGRpc3BvbmlibGUgc3VyIGNldCBhcHBhcmVpbC5cIixcbiAgICAgIH0pO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgICBjb25zdCBpbnN0YW5jZSA9IGVuc3VyZVJlY29nbml0aW9uKCk7XG4gICAgaWYgKCFpbnN0YW5jZSkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgICBpZiAobGlzdGVuaW5nKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgbWFudWFsU3RvcCA9IGZhbHNlO1xuICAgIHJlY29nbml0aW9uTGFuZyA9IG5vcm1hbGl6ZVRleHQob3B0aW9ucy5sYW5ndWFnZSkgfHwgcmVjb2duaXRpb25MYW5nO1xuICAgIGluc3RhbmNlLmxhbmcgPSByZWNvZ25pdGlvbkxhbmc7XG4gICAgaW5zdGFuY2UuaW50ZXJpbVJlc3VsdHMgPSBvcHRpb25zLmludGVyaW1SZXN1bHRzICE9PSBmYWxzZTtcbiAgICBpbnN0YW5jZS5jb250aW51b3VzID0gQm9vbGVhbihvcHRpb25zLmNvbnRpbnVvdXMpO1xuICAgIGluc3RhbmNlLm1heEFsdGVybmF0aXZlcyA9IG9wdGlvbnMubWF4QWx0ZXJuYXRpdmVzIHx8IDE7XG4gICAgaWYgKHJlcXVpcmVzTWljcm9waG9uZVByaW1pbmcoKSkge1xuICAgICAgY29uc3QgZ3JhbnRlZCA9IGF3YWl0IGVuc3VyZU1pY3JvcGhvbmVBY2Nlc3MoKTtcbiAgICAgIGlmICghZ3JhbnRlZCkge1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICBpbnN0YW5jZS5zdGFydCgpO1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBpZiAocmVxdWlyZXNNaWNyb3Bob25lUHJpbWluZygpICYmICFtaWNyb3Bob25lUHJpbWVkICYmIGlzUGVybWlzc2lvbkVycm9yKGVycikpIHtcbiAgICAgICAgY29uc3QgZ3JhbnRlZCA9IGF3YWl0IGVuc3VyZU1pY3JvcGhvbmVBY2Nlc3MoKTtcbiAgICAgICAgaWYgKGdyYW50ZWQpIHtcbiAgICAgICAgICB0cnkge1xuICAgICAgICAgICAgaW5zdGFuY2Uuc3RhcnQoKTtcbiAgICAgICAgICAgIHJldHVybiB0cnVlO1xuICAgICAgICAgIH0gY2F0Y2ggKHJldHJ5RXJyKSB7XG4gICAgICAgICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICAgICAgICBzb3VyY2U6IFwicmVjb2duaXRpb25cIixcbiAgICAgICAgICAgICAgY29kZTogXCJzdGFydC1mYWlsZWRcIixcbiAgICAgICAgICAgICAgbWVzc2FnZTpcbiAgICAgICAgICAgICAgICByZXRyeUVyciAmJiByZXRyeUVyci5tZXNzYWdlXG4gICAgICAgICAgICAgICAgICA/IHJldHJ5RXJyLm1lc3NhZ2VcbiAgICAgICAgICAgICAgICAgIDogXCJJbXBvc3NpYmxlIGRlIGRcdTAwRTltYXJyZXIgbGEgcmVjb25uYWlzc2FuY2Ugdm9jYWxlLlwiLFxuICAgICAgICAgICAgICBkZXRhaWxzOiByZXRyeUVycixcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgIGNvZGU6IFwic3RhcnQtZmFpbGVkXCIsXG4gICAgICAgIG1lc3NhZ2U6XG4gICAgICAgICAgZXJyICYmIGVyci5tZXNzYWdlXG4gICAgICAgICAgICA/IGVyci5tZXNzYWdlXG4gICAgICAgICAgICA6IFwiSW1wb3NzaWJsZSBkZSBkXHUwMEU5bWFycmVyIGxhIHJlY29ubmFpc3NhbmNlIHZvY2FsZS5cIixcbiAgICAgICAgZGV0YWlsczogZXJyLFxuICAgICAgfSk7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc3RvcExpc3RlbmluZyhvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIXJlY29nbml0aW9uKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIG1hbnVhbFN0b3AgPSB0cnVlO1xuICAgIHRyeSB7XG4gICAgICBpZiAob3B0aW9ucyAmJiBvcHRpb25zLmFib3J0ICYmIHR5cGVvZiByZWNvZ25pdGlvbi5hYm9ydCA9PT0gXCJmdW5jdGlvblwiKSB7XG4gICAgICAgIHJlY29nbml0aW9uLmFib3J0KCk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZWNvZ25pdGlvbi5zdG9wKCk7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICBzb3VyY2U6IFwicmVjb2duaXRpb25cIixcbiAgICAgICAgY29kZTogXCJzdG9wLWZhaWxlZFwiLFxuICAgICAgICBtZXNzYWdlOiBcIkFyclx1MDBFQXQgZGUgbGEgZGljdFx1MDBFOWUgaW1wb3NzaWJsZS5cIixcbiAgICAgICAgZGV0YWlsczogZXJyLFxuICAgICAgfSk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZmluZFZvaWNlKHVyaSkge1xuICAgIGlmICghdXJpIHx8ICFzeW50aCkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGNvbnN0IHZvaWNlcyA9IHN5bnRoLmdldFZvaWNlcygpO1xuICAgIHJldHVybiB2b2ljZXMuZmluZCgodm9pY2UpID0+IHZvaWNlLnZvaWNlVVJJID09PSB1cmkpIHx8IG51bGw7XG4gIH1cblxuICBmdW5jdGlvbiByZWZyZXNoVm9pY2VzKCkge1xuICAgIGlmICghc3ludGgpIHtcbiAgICAgIHJldHVybiBbXTtcbiAgICB9XG4gICAgdHJ5IHtcbiAgICAgIHZvaWNlc0NhY2hlID0gc3ludGguZ2V0Vm9pY2VzKCk7XG4gICAgICBjb25zdCBwYXlsb2FkID0gdm9pY2VzQ2FjaGUubWFwKG1hcFZvaWNlKTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcInZvaWNlc1wiLCB7IHZvaWNlczogcGF5bG9hZCB9KTtcbiAgICAgIHJldHVybiBwYXlsb2FkO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInN5bnRoZXNpc1wiLFxuICAgICAgICBjb2RlOiBcInZvaWNlcy1mYWlsZWRcIixcbiAgICAgICAgbWVzc2FnZTogXCJJbXBvc3NpYmxlIGRlIHJcdTAwRTljdXBcdTAwRTlyZXIgbGEgbGlzdGUgZGVzIHZvaXggZGlzcG9uaWJsZXMuXCIsXG4gICAgICAgIGRldGFpbHM6IGVycixcbiAgICAgIH0pO1xuICAgICAgcmV0dXJuIFtdO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNwZWFrKHRleHQsIG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghc3ludGhlc2lzU3VwcG9ydGVkKSB7XG4gICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICBzb3VyY2U6IFwic3ludGhlc2lzXCIsXG4gICAgICAgIGNvZGU6IFwidW5zdXBwb3J0ZWRcIixcbiAgICAgICAgbWVzc2FnZTogXCJMYSBzeW50aFx1MDBFOHNlIHZvY2FsZSBuJ2VzdCBwYXMgZGlzcG9uaWJsZSBzdXIgY2V0IGFwcGFyZWlsLlwiLFxuICAgICAgfSk7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgY29uc3QgY29udGVudCA9IG5vcm1hbGl6ZVRleHQodGV4dCk7XG4gICAgaWYgKCFjb250ZW50KSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgaWYgKGxpc3RlbmluZykge1xuICAgICAgc3RvcExpc3RlbmluZyh7IGFib3J0OiB0cnVlIH0pO1xuICAgIH1cbiAgICBzdG9wU3BlYWtpbmcoKTtcbiAgICBjb25zdCB1dHRlcmFuY2UgPSBuZXcgU3BlZWNoU3ludGhlc2lzVXR0ZXJhbmNlKGNvbnRlbnQpO1xuICAgIHV0dGVyYW5jZS5sYW5nID0gbm9ybWFsaXplVGV4dChvcHRpb25zLmxhbmcpIHx8IHJlY29nbml0aW9uTGFuZztcbiAgICBjb25zdCByYXRlID0gTnVtYmVyKG9wdGlvbnMucmF0ZSk7XG4gICAgaWYgKCFOdW1iZXIuaXNOYU4ocmF0ZSkgJiYgcmF0ZSA+IDApIHtcbiAgICAgIHV0dGVyYW5jZS5yYXRlID0gTWF0aC5taW4ocmF0ZSwgMik7XG4gICAgfVxuICAgIGNvbnN0IHBpdGNoID0gTnVtYmVyKG9wdGlvbnMucGl0Y2gpO1xuICAgIGlmICghTnVtYmVyLmlzTmFOKHBpdGNoKSAmJiBwaXRjaCA+IDApIHtcbiAgICAgIHV0dGVyYW5jZS5waXRjaCA9IE1hdGgubWluKHBpdGNoLCAyKTtcbiAgICB9XG4gICAgY29uc3Qgdm9pY2UgPVxuICAgICAgZmluZFZvaWNlKG9wdGlvbnMudm9pY2VVUkkpIHx8IGZpbmRWb2ljZShwcmVmZXJyZWRWb2ljZVVSSSkgfHwgbnVsbDtcbiAgICBpZiAodm9pY2UpIHtcbiAgICAgIHV0dGVyYW5jZS52b2ljZSA9IHZvaWNlO1xuICAgIH1cblxuICAgIHV0dGVyYW5jZS5vbnN0YXJ0ID0gKCkgPT4ge1xuICAgICAgc3BlYWtpbmcgPSB0cnVlO1xuICAgICAgZW1pdHRlci5lbWl0KFwic3BlYWtpbmctY2hhbmdlXCIsIHtcbiAgICAgICAgc3BlYWtpbmc6IHRydWUsXG4gICAgICAgIHV0dGVyYW5jZSxcbiAgICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIH0pO1xuICAgIH07XG5cbiAgICB1dHRlcmFuY2Uub25lbmQgPSAoKSA9PiB7XG4gICAgICBzcGVha2luZyA9IGZhbHNlO1xuICAgICAgZW1pdHRlci5lbWl0KFwic3BlYWtpbmctY2hhbmdlXCIsIHtcbiAgICAgICAgc3BlYWtpbmc6IGZhbHNlLFxuICAgICAgICB1dHRlcmFuY2UsXG4gICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICB9KTtcbiAgICB9O1xuXG4gICAgdXR0ZXJhbmNlLm9uZXJyb3IgPSAoZXZlbnQpID0+IHtcbiAgICAgIHNwZWFraW5nID0gZmFsc2U7XG4gICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICBzb3VyY2U6IFwic3ludGhlc2lzXCIsXG4gICAgICAgIGNvZGU6IGV2ZW50LmVycm9yIHx8IFwidW5rbm93blwiLFxuICAgICAgICBtZXNzYWdlOlxuICAgICAgICAgIGV2ZW50ICYmIGV2ZW50Lm1lc3NhZ2VcbiAgICAgICAgICAgID8gZXZlbnQubWVzc2FnZVxuICAgICAgICAgICAgOiBcIkxhIHN5bnRoXHUwMEU4c2Ugdm9jYWxlIGEgcmVuY29udHJcdTAwRTkgdW5lIGVycmV1ci5cIixcbiAgICAgICAgZXZlbnQsXG4gICAgICB9KTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcInNwZWFraW5nLWNoYW5nZVwiLCB7XG4gICAgICAgIHNwZWFraW5nOiBmYWxzZSxcbiAgICAgICAgdXR0ZXJhbmNlLFxuICAgICAgICByZWFzb246IFwiZXJyb3JcIixcbiAgICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIH0pO1xuICAgIH07XG5cbiAgICBzeW50aC5zcGVhayh1dHRlcmFuY2UpO1xuICAgIHJldHVybiB1dHRlcmFuY2U7XG4gIH1cblxuICBmdW5jdGlvbiBzdG9wU3BlYWtpbmcoKSB7XG4gICAgaWYgKCFzeW50aGVzaXNTdXBwb3J0ZWQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHN5bnRoLnNwZWFraW5nIHx8IHN5bnRoLnBlbmRpbmcpIHtcbiAgICAgIHN5bnRoLmNhbmNlbCgpO1xuICAgIH1cbiAgICBpZiAoc3BlYWtpbmcpIHtcbiAgICAgIHNwZWFraW5nID0gZmFsc2U7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJzcGVha2luZy1jaGFuZ2VcIiwge1xuICAgICAgICBzcGVha2luZzogZmFsc2UsXG4gICAgICAgIHJlYXNvbjogXCJjYW5jZWxcIixcbiAgICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNldFByZWZlcnJlZFZvaWNlKHVyaSkge1xuICAgIHByZWZlcnJlZFZvaWNlVVJJID0gdXJpIHx8IG51bGw7XG4gIH1cblxuICBmdW5jdGlvbiBzZXRMYW5ndWFnZShsYW5nKSB7XG4gICAgY29uc3QgbmV4dCA9IG5vcm1hbGl6ZVRleHQobGFuZyk7XG4gICAgaWYgKG5leHQpIHtcbiAgICAgIHJlY29nbml0aW9uTGFuZyA9IG5leHQ7XG4gICAgICBpZiAocmVjb2duaXRpb24pIHtcbiAgICAgICAgcmVjb2duaXRpb24ubGFuZyA9IHJlY29nbml0aW9uTGFuZztcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBpZiAoc3ludGhlc2lzU3VwcG9ydGVkKSB7XG4gICAgcmVmcmVzaFZvaWNlcygpO1xuICAgIGlmIChzeW50aC5hZGRFdmVudExpc3RlbmVyKSB7XG4gICAgICBzeW50aC5hZGRFdmVudExpc3RlbmVyKFwidm9pY2VzY2hhbmdlZFwiLCByZWZyZXNoVm9pY2VzKTtcbiAgICB9IGVsc2UgaWYgKFwib252b2ljZXNjaGFuZ2VkXCIgaW4gc3ludGgpIHtcbiAgICAgIHN5bnRoLm9udm9pY2VzY2hhbmdlZCA9IHJlZnJlc2hWb2ljZXM7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBvbjogZW1pdHRlci5vbixcbiAgICBvZmY6IGVtaXR0ZXIub2ZmLFxuICAgIHN0YXJ0TGlzdGVuaW5nLFxuICAgIHN0b3BMaXN0ZW5pbmcsXG4gICAgc3BlYWssXG4gICAgc3RvcFNwZWFraW5nLFxuICAgIHNldFByZWZlcnJlZFZvaWNlLFxuICAgIHNldExhbmd1YWdlLFxuICAgIHJlZnJlc2hWb2ljZXMsXG4gICAgZ2V0Vm9pY2VzOiAoKSA9PiB2b2ljZXNDYWNoZS5tYXAobWFwVm9pY2UpLFxuICAgIGdldFByZWZlcnJlZFZvaWNlOiAoKSA9PiBwcmVmZXJyZWRWb2ljZVVSSSxcbiAgICBpc1JlY29nbml0aW9uU3VwcG9ydGVkOiAoKSA9PiByZWNvZ25pdGlvblN1cHBvcnRlZCxcbiAgICBpc1N5bnRoZXNpc1N1cHBvcnRlZDogKCkgPT4gc3ludGhlc2lzU3VwcG9ydGVkLFxuICB9O1xufVxuIiwgImltcG9ydCB7IHJlc29sdmVDb25maWcgfSBmcm9tIFwiLi9jb25maWcuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVRpbWVsaW5lU3RvcmUgfSBmcm9tIFwiLi9zdGF0ZS90aW1lbGluZVN0b3JlLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVDaGF0VWkgfSBmcm9tIFwiLi91aS9jaGF0VWkuanNcIjtcbmltcG9ydCB7IGNyZWF0ZUF1dGhTZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvYXV0aC5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlSHR0cFNlcnZpY2UgfSBmcm9tIFwiLi9zZXJ2aWNlcy9odHRwLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVFeHBvcnRlciB9IGZyb20gXCIuL3NlcnZpY2VzL2V4cG9ydGVyLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVTb2NrZXRDbGllbnQgfSBmcm9tIFwiLi9zZXJ2aWNlcy9zb2NrZXQuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvc3VnZ2VzdGlvbnMuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVNwZWVjaFNlcnZpY2UgfSBmcm9tIFwiLi9zZXJ2aWNlcy9zcGVlY2guanNcIjtcbmltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuL3V0aWxzL3RpbWUuanNcIjtcblxuZnVuY3Rpb24gcXVlcnlFbGVtZW50cyhkb2MpIHtcbiAgY29uc3QgYnlJZCA9IChpZCkgPT4gZG9jLmdldEVsZW1lbnRCeUlkKGlkKTtcbiAgcmV0dXJuIHtcbiAgICB0cmFuc2NyaXB0OiBieUlkKFwidHJhbnNjcmlwdFwiKSxcbiAgICBjb21wb3NlcjogYnlJZChcImNvbXBvc2VyXCIpLFxuICAgIHByb21wdDogYnlJZChcInByb21wdFwiKSxcbiAgICBzZW5kOiBieUlkKFwic2VuZFwiKSxcbiAgICB3c1N0YXR1czogYnlJZChcIndzLXN0YXR1c1wiKSxcbiAgICBxdWlja0FjdGlvbnM6IGJ5SWQoXCJxdWljay1hY3Rpb25zXCIpLFxuICAgIGNvbm5lY3Rpb246IGJ5SWQoXCJjb25uZWN0aW9uXCIpLFxuICAgIGVycm9yQWxlcnQ6IGJ5SWQoXCJlcnJvci1hbGVydFwiKSxcbiAgICBlcnJvck1lc3NhZ2U6IGJ5SWQoXCJlcnJvci1tZXNzYWdlXCIpLFxuICAgIHNjcm9sbEJvdHRvbTogYnlJZChcInNjcm9sbC1ib3R0b21cIiksXG4gICAgY29tcG9zZXJTdGF0dXM6IGJ5SWQoXCJjb21wb3Nlci1zdGF0dXNcIiksXG4gICAgcHJvbXB0Q291bnQ6IGJ5SWQoXCJwcm9tcHQtY291bnRcIiksXG4gICAgY29ubmVjdGlvbk1ldGE6IGJ5SWQoXCJjb25uZWN0aW9uLW1ldGFcIiksXG4gICAgZmlsdGVySW5wdXQ6IGJ5SWQoXCJjaGF0LXNlYXJjaFwiKSxcbiAgICBmaWx0ZXJDbGVhcjogYnlJZChcImNoYXQtc2VhcmNoLWNsZWFyXCIpLFxuICAgIGZpbHRlckVtcHR5OiBieUlkKFwiZmlsdGVyLWVtcHR5XCIpLFxuICAgIGZpbHRlckhpbnQ6IGJ5SWQoXCJjaGF0LXNlYXJjaC1oaW50XCIpLFxuICAgIGV4cG9ydEpzb246IGJ5SWQoXCJleHBvcnQtanNvblwiKSxcbiAgICBleHBvcnRNYXJrZG93bjogYnlJZChcImV4cG9ydC1tYXJrZG93blwiKSxcbiAgICBleHBvcnRDb3B5OiBieUlkKFwiZXhwb3J0LWNvcHlcIiksXG4gICAgZGlhZ0Nvbm5lY3RlZDogYnlJZChcImRpYWctY29ubmVjdGVkXCIpLFxuICAgIGRpYWdMYXN0TWVzc2FnZTogYnlJZChcImRpYWctbGFzdC1tZXNzYWdlXCIpLFxuICAgIGRpYWdMYXRlbmN5OiBieUlkKFwiZGlhZy1sYXRlbmN5XCIpLFxuICAgIGRpYWdOZXR3b3JrOiBieUlkKFwiZGlhZy1uZXR3b3JrXCIpLFxuICAgIHZvaWNlQ29udHJvbHM6IGJ5SWQoXCJ2b2ljZS1jb250cm9sc1wiKSxcbiAgICB2b2ljZVJlY29nbml0aW9uR3JvdXA6IGJ5SWQoXCJ2b2ljZS1yZWNvZ25pdGlvbi1ncm91cFwiKSxcbiAgICB2b2ljZVN5bnRoZXNpc0dyb3VwOiBieUlkKFwidm9pY2Utc3ludGhlc2lzLWdyb3VwXCIpLFxuICAgIHZvaWNlVG9nZ2xlOiBieUlkKFwidm9pY2UtdG9nZ2xlXCIpLFxuICAgIHZvaWNlU3RhdHVzOiBieUlkKFwidm9pY2Utc3RhdHVzXCIpLFxuICAgIHZvaWNlVHJhbnNjcmlwdDogYnlJZChcInZvaWNlLXRyYW5zY3JpcHRcIiksXG4gICAgdm9pY2VBdXRvU2VuZDogYnlJZChcInZvaWNlLWF1dG8tc2VuZFwiKSxcbiAgICB2b2ljZVBsYXliYWNrOiBieUlkKFwidm9pY2UtcGxheWJhY2tcIiksXG4gICAgdm9pY2VTdG9wUGxheWJhY2s6IGJ5SWQoXCJ2b2ljZS1zdG9wLXBsYXliYWNrXCIpLFxuICAgIHZvaWNlVm9pY2VTZWxlY3Q6IGJ5SWQoXCJ2b2ljZS12b2ljZS1zZWxlY3RcIiksXG4gICAgdm9pY2VTcGVha2luZ0luZGljYXRvcjogYnlJZChcInZvaWNlLXNwZWFraW5nLWluZGljYXRvclwiKSxcbiAgfTtcbn1cblxuZnVuY3Rpb24gcmVhZEhpc3RvcnkoZG9jKSB7XG4gIGNvbnN0IGhpc3RvcnlFbGVtZW50ID0gZG9jLmdldEVsZW1lbnRCeUlkKFwiY2hhdC1oaXN0b3J5XCIpO1xuICBpZiAoIWhpc3RvcnlFbGVtZW50KSB7XG4gICAgcmV0dXJuIFtdO1xuICB9XG4gIGNvbnN0IHBheWxvYWQgPSBoaXN0b3J5RWxlbWVudC50ZXh0Q29udGVudCB8fCBcIm51bGxcIjtcbiAgaGlzdG9yeUVsZW1lbnQucmVtb3ZlKCk7XG4gIHRyeSB7XG4gICAgY29uc3QgcGFyc2VkID0gSlNPTi5wYXJzZShwYXlsb2FkKTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheShwYXJzZWQpKSB7XG4gICAgICByZXR1cm4gcGFyc2VkO1xuICAgIH1cbiAgICBpZiAocGFyc2VkICYmIHBhcnNlZC5lcnJvcikge1xuICAgICAgcmV0dXJuIHsgZXJyb3I6IHBhcnNlZC5lcnJvciB9O1xuICAgIH1cbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS5lcnJvcihcIlVuYWJsZSB0byBwYXJzZSBjaGF0IGhpc3RvcnlcIiwgZXJyKTtcbiAgfVxuICByZXR1cm4gW107XG59XG5cbmZ1bmN0aW9uIGVuc3VyZUVsZW1lbnRzKGVsZW1lbnRzKSB7XG4gIHJldHVybiBCb29sZWFuKGVsZW1lbnRzLnRyYW5zY3JpcHQgJiYgZWxlbWVudHMuY29tcG9zZXIgJiYgZWxlbWVudHMucHJvbXB0KTtcbn1cblxuY29uc3QgUVVJQ0tfUFJFU0VUUyA9IHtcbiAgY29kZTogXCJKZSBzb3VoYWl0ZSBcdTAwRTljcmlyZSBkdSBjb2RlXHUyMDI2XCIsXG4gIHN1bW1hcml6ZTogXCJSXHUwMEU5c3VtZSBsYSBkZXJuaVx1MDBFOHJlIGNvbnZlcnNhdGlvbi5cIixcbiAgZXhwbGFpbjogXCJFeHBsaXF1ZSB0YSBkZXJuaVx1MDBFOHJlIHJcdTAwRTlwb25zZSBwbHVzIHNpbXBsZW1lbnQuXCIsXG59O1xuXG5leHBvcnQgY2xhc3MgQ2hhdEFwcCB7XG4gIGNvbnN0cnVjdG9yKGRvYyA9IGRvY3VtZW50LCByYXdDb25maWcgPSB3aW5kb3cuY2hhdENvbmZpZyB8fCB7fSkge1xuICAgIHRoaXMuZG9jID0gZG9jO1xuICAgIHRoaXMuY29uZmlnID0gcmVzb2x2ZUNvbmZpZyhyYXdDb25maWcpO1xuICAgIHRoaXMuZWxlbWVudHMgPSBxdWVyeUVsZW1lbnRzKGRvYyk7XG4gICAgaWYgKCFlbnN1cmVFbGVtZW50cyh0aGlzLmVsZW1lbnRzKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAod2luZG93Lm1hcmtlZCAmJiB0eXBlb2Ygd2luZG93Lm1hcmtlZC5zZXRPcHRpb25zID09PSBcImZ1bmN0aW9uXCIpIHtcbiAgICAgIHdpbmRvdy5tYXJrZWQuc2V0T3B0aW9ucyh7XG4gICAgICAgIGJyZWFrczogdHJ1ZSxcbiAgICAgICAgZ2ZtOiB0cnVlLFxuICAgICAgICBoZWFkZXJJZHM6IGZhbHNlLFxuICAgICAgICBtYW5nbGU6IGZhbHNlLFxuICAgICAgfSk7XG4gICAgfVxuICAgIHRoaXMudGltZWxpbmVTdG9yZSA9IGNyZWF0ZVRpbWVsaW5lU3RvcmUoKTtcbiAgICB0aGlzLnVpID0gY3JlYXRlQ2hhdFVpKHtcbiAgICAgIGVsZW1lbnRzOiB0aGlzLmVsZW1lbnRzLFxuICAgICAgdGltZWxpbmVTdG9yZTogdGhpcy50aW1lbGluZVN0b3JlLFxuICAgIH0pO1xuICAgIHRoaXMuYXV0aCA9IGNyZWF0ZUF1dGhTZXJ2aWNlKHRoaXMuY29uZmlnKTtcbiAgICB0aGlzLmh0dHAgPSBjcmVhdGVIdHRwU2VydmljZSh7IGNvbmZpZzogdGhpcy5jb25maWcsIGF1dGg6IHRoaXMuYXV0aCB9KTtcbiAgICB0aGlzLmV4cG9ydGVyID0gY3JlYXRlRXhwb3J0ZXIoe1xuICAgICAgdGltZWxpbmVTdG9yZTogdGhpcy50aW1lbGluZVN0b3JlLFxuICAgICAgYW5ub3VuY2U6IChtZXNzYWdlLCB2YXJpYW50KSA9PlxuICAgICAgICB0aGlzLnVpLmFubm91bmNlQ29ubmVjdGlvbihtZXNzYWdlLCB2YXJpYW50KSxcbiAgICB9KTtcbiAgICB0aGlzLnN1Z2dlc3Rpb25zID0gY3JlYXRlU3VnZ2VzdGlvblNlcnZpY2Uoe1xuICAgICAgaHR0cDogdGhpcy5odHRwLFxuICAgICAgdWk6IHRoaXMudWksXG4gICAgfSk7XG4gICAgdGhpcy5zb2NrZXQgPSBjcmVhdGVTb2NrZXRDbGllbnQoe1xuICAgICAgY29uZmlnOiB0aGlzLmNvbmZpZyxcbiAgICAgIGh0dHA6IHRoaXMuaHR0cCxcbiAgICAgIHVpOiB0aGlzLnVpLFxuICAgICAgb25FdmVudDogKGV2KSA9PiB0aGlzLmhhbmRsZVNvY2tldEV2ZW50KGV2KSxcbiAgICB9KTtcblxuICAgIHRoaXMuc2V0dXBWb2ljZUZlYXR1cmVzKCk7XG5cbiAgICBjb25zdCBoaXN0b3J5UGF5bG9hZCA9IHJlYWRIaXN0b3J5KGRvYyk7XG4gICAgaWYgKGhpc3RvcnlQYXlsb2FkICYmIGhpc3RvcnlQYXlsb2FkLmVycm9yKSB7XG4gICAgICB0aGlzLnVpLnNob3dFcnJvcihoaXN0b3J5UGF5bG9hZC5lcnJvcik7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGhpc3RvcnlQYXlsb2FkKSkge1xuICAgICAgdGhpcy51aS5yZW5kZXJIaXN0b3J5KGhpc3RvcnlQYXlsb2FkKTtcbiAgICB9XG5cbiAgICB0aGlzLnJlZ2lzdGVyVWlIYW5kbGVycygpO1xuICAgIHRoaXMudWkuaW5pdGlhbGlzZSgpO1xuICAgIHRoaXMuc29ja2V0Lm9wZW4oKTtcbiAgfVxuXG4gIHJlZ2lzdGVyVWlIYW5kbGVycygpIHtcbiAgICB0aGlzLnVpLm9uKFwic3VibWl0XCIsIGFzeW5jICh7IHRleHQgfSkgPT4ge1xuICAgICAgY29uc3QgdmFsdWUgPSAodGV4dCB8fCBcIlwiKS50cmltKCk7XG4gICAgICBpZiAoIXZhbHVlKSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJTYWlzaXNzZXogdW4gbWVzc2FnZSBhdmFudCBkXHUyMDE5ZW52b3llci5cIixcbiAgICAgICAgICBcIndhcm5pbmdcIixcbiAgICAgICAgKTtcbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgdGhpcy51aS5oaWRlRXJyb3IoKTtcbiAgICAgIGNvbnN0IHN1Ym1pdHRlZEF0ID0gbm93SVNPKCk7XG4gICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJ1c2VyXCIsIHZhbHVlLCB7XG4gICAgICAgIHRpbWVzdGFtcDogc3VibWl0dGVkQXQsXG4gICAgICAgIG1ldGFkYXRhOiB7IHN1Ym1pdHRlZDogdHJ1ZSB9LFxuICAgICAgfSk7XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSBcIlwiO1xuICAgICAgfVxuICAgICAgdGhpcy51aS51cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICB0aGlzLnVpLmF1dG9zaXplUHJvbXB0KCk7XG4gICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFwiTWVzc2FnZSBlbnZveVx1MDBFOVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgdGhpcy51aS5zZXRCdXN5KHRydWUpO1xuICAgICAgdGhpcy51aS5hcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcoW1wiY29kZVwiLCBcInN1bW1hcml6ZVwiLCBcImV4cGxhaW5cIl0pO1xuXG4gICAgICB0cnkge1xuICAgICAgICBhd2FpdCB0aGlzLmh0dHAucG9zdENoYXQodmFsdWUpO1xuICAgICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC5mb2N1cygpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuc3RhcnRTdHJlYW0oKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICB0aGlzLnVpLnNldEJ1c3koZmFsc2UpO1xuICAgICAgICBjb25zdCBtZXNzYWdlID0gZXJyIGluc3RhbmNlb2YgRXJyb3IgPyBlcnIubWVzc2FnZSA6IFN0cmluZyhlcnIpO1xuICAgICAgICB0aGlzLnVpLnNob3dFcnJvcihtZXNzYWdlKTtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwic3lzdGVtXCIsIG1lc3NhZ2UsIHtcbiAgICAgICAgICB2YXJpYW50OiBcImVycm9yXCIsXG4gICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgbWV0YWRhdGE6IHsgc3RhZ2U6IFwic3VibWl0XCIgfSxcbiAgICAgICAgfSk7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJFbnZvaSBpbXBvc3NpYmxlLiBWXHUwMEU5cmlmaWV6IGxhIGNvbm5leGlvbi5cIixcbiAgICAgICAgICBcImRhbmdlclwiLFxuICAgICAgICApO1xuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfVxuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcInF1aWNrLWFjdGlvblwiLCAoeyBhY3Rpb24gfSkgPT4ge1xuICAgICAgaWYgKCFhY3Rpb24pIHJldHVybjtcbiAgICAgIGNvbnN0IHByZXNldCA9IFFVSUNLX1BSRVNFVFNbYWN0aW9uXSB8fCBhY3Rpb247XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSBwcmVzZXQ7XG4gICAgICB9XG4gICAgICB0aGlzLnVpLnVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJTdWdnZXN0aW9uIGVudm95XHUwMEU5ZVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgdGhpcy51aS5lbWl0KFwic3VibWl0XCIsIHsgdGV4dDogcHJlc2V0IH0pO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcImZpbHRlci1jaGFuZ2VcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgdGhpcy51aS5hcHBseVRyYW5zY3JpcHRGaWx0ZXIodmFsdWUsIHsgcHJlc2VydmVJbnB1dDogdHJ1ZSB9KTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJmaWx0ZXItY2xlYXJcIiwgKCkgPT4ge1xuICAgICAgdGhpcy51aS5jbGVhclRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJleHBvcnRcIiwgKHsgZm9ybWF0IH0pID0+IHtcbiAgICAgIHRoaXMuZXhwb3J0ZXIuZXhwb3J0Q29udmVyc2F0aW9uKGZvcm1hdCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwiZXhwb3J0LWNvcHlcIiwgKCkgPT4ge1xuICAgICAgdGhpcy5leHBvcnRlci5jb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJ2b2ljZS10b2dnbGVcIiwgKCkgPT4ge1xuICAgICAgdGhpcy50b2dnbGVWb2ljZUxpc3RlbmluZygpLmNhdGNoKChlcnIpID0+IHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIlZvaWNlIHRvZ2dsZSBmYWlsZWRcIiwgZXJyKTtcbiAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcInZvaWNlLWF1dG9zZW5kLWNoYW5nZVwiLCAoeyBlbmFibGVkIH0pID0+IHtcbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VBdXRvU2VuZENoYW5nZShCb29sZWFuKGVuYWJsZWQpKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJ2b2ljZS1wbGF5YmFjay1jaGFuZ2VcIiwgKHsgZW5hYmxlZCB9KSA9PiB7XG4gICAgICB0aGlzLmhhbmRsZVZvaWNlUGxheWJhY2tDaGFuZ2UoQm9vbGVhbihlbmFibGVkKSk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwidm9pY2Utc3RvcC1wbGF5YmFja1wiLCAoKSA9PiB7XG4gICAgICB0aGlzLnN0b3BWb2ljZVBsYXliYWNrKCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwidm9pY2Utdm9pY2UtY2hhbmdlXCIsICh7IHZvaWNlVVJJIH0pID0+IHtcbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VWb2ljZUNoYW5nZSh2b2ljZVVSSSB8fCBudWxsKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJwcm9tcHQtaW5wdXRcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgaWYgKCF2YWx1ZSB8fCAhdmFsdWUudHJpbSgpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmVsZW1lbnRzLnNlbmQgJiYgdGhpcy5lbGVtZW50cy5zZW5kLmRpc2FibGVkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMuc3VnZ2VzdGlvbnMuc2NoZWR1bGUodmFsdWUpO1xuICAgIH0pO1xuICB9XG5cbiAgbG9hZFZvaWNlUHJlZmVyZW5jZXMoZGVmYXVsdExhbmd1YWdlKSB7XG4gICAgY29uc3QgZmFsbGJhY2sgPSB7XG4gICAgICBhdXRvU2VuZDogdHJ1ZSxcbiAgICAgIHBsYXliYWNrOiB0cnVlLFxuICAgICAgdm9pY2VVUkk6IG51bGwsXG4gICAgICBsYW5ndWFnZTogZGVmYXVsdExhbmd1YWdlLFxuICAgIH07XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHJhdyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2UuZ2V0SXRlbShcImNoYXQudm9pY2VcIik7XG4gICAgICBpZiAoIXJhdykge1xuICAgICAgICByZXR1cm4gZmFsbGJhY2s7XG4gICAgICB9XG4gICAgICBjb25zdCBwYXJzZWQgPSBKU09OLnBhcnNlKHJhdyk7XG4gICAgICBpZiAoIXBhcnNlZCB8fCB0eXBlb2YgcGFyc2VkICE9PSBcIm9iamVjdFwiKSB7XG4gICAgICAgIHJldHVybiBmYWxsYmFjaztcbiAgICAgIH1cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGF1dG9TZW5kOlxuICAgICAgICAgIHR5cGVvZiBwYXJzZWQuYXV0b1NlbmQgPT09IFwiYm9vbGVhblwiID8gcGFyc2VkLmF1dG9TZW5kIDogZmFsbGJhY2suYXV0b1NlbmQsXG4gICAgICAgIHBsYXliYWNrOlxuICAgICAgICAgIHR5cGVvZiBwYXJzZWQucGxheWJhY2sgPT09IFwiYm9vbGVhblwiID8gcGFyc2VkLnBsYXliYWNrIDogZmFsbGJhY2sucGxheWJhY2ssXG4gICAgICAgIHZvaWNlVVJJOlxuICAgICAgICAgIHR5cGVvZiBwYXJzZWQudm9pY2VVUkkgPT09IFwic3RyaW5nXCIgJiYgcGFyc2VkLnZvaWNlVVJJLmxlbmd0aCA+IDBcbiAgICAgICAgICAgID8gcGFyc2VkLnZvaWNlVVJJXG4gICAgICAgICAgICA6IG51bGwsXG4gICAgICAgIGxhbmd1YWdlOlxuICAgICAgICAgIHR5cGVvZiBwYXJzZWQubGFuZ3VhZ2UgPT09IFwic3RyaW5nXCIgJiYgcGFyc2VkLmxhbmd1YWdlXG4gICAgICAgICAgICA/IHBhcnNlZC5sYW5ndWFnZVxuICAgICAgICAgICAgOiBmYWxsYmFjay5sYW5ndWFnZSxcbiAgICAgIH07XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcmVhZCB2b2ljZSBwcmVmZXJlbmNlc1wiLCBlcnIpO1xuICAgICAgcmV0dXJuIGZhbGxiYWNrO1xuICAgIH1cbiAgfVxuXG4gIHBlcnNpc3RWb2ljZVByZWZlcmVuY2VzKCkge1xuICAgIGlmICghdGhpcy52b2ljZVByZWZzKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICB3aW5kb3cubG9jYWxTdG9yYWdlLnNldEl0ZW0oXG4gICAgICAgIFwiY2hhdC52b2ljZVwiLFxuICAgICAgICBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgYXV0b1NlbmQ6IEJvb2xlYW4odGhpcy52b2ljZVByZWZzLmF1dG9TZW5kKSxcbiAgICAgICAgICBwbGF5YmFjazogQm9vbGVhbih0aGlzLnZvaWNlUHJlZnMucGxheWJhY2spLFxuICAgICAgICAgIHZvaWNlVVJJOiB0aGlzLnZvaWNlUHJlZnMudm9pY2VVUkkgfHwgbnVsbCxcbiAgICAgICAgICBsYW5ndWFnZTogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlIHx8IG51bGwsXG4gICAgICAgIH0pLFxuICAgICAgKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byBwZXJzaXN0IHZvaWNlIHByZWZlcmVuY2VzXCIsIGVycik7XG4gICAgfVxuICB9XG5cbiAgc2V0dXBWb2ljZUZlYXR1cmVzKCkge1xuICAgIGNvbnN0IGRvY0xhbmcgPSAodGhpcy5kb2M/LmRvY3VtZW50RWxlbWVudD8uZ2V0QXR0cmlidXRlKFwibGFuZ1wiKSB8fCBcIlwiKS50cmltKCk7XG4gICAgY29uc3QgbmF2aWdhdG9yTGFuZyA9XG4gICAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgICA/IG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgICA6IG51bGw7XG4gICAgY29uc3QgZGVmYXVsdExhbmd1YWdlID0gZG9jTGFuZyB8fCBuYXZpZ2F0b3JMYW5nIHx8IFwiZnItQ0FcIjtcbiAgICB0aGlzLnZvaWNlUHJlZnMgPSB0aGlzLmxvYWRWb2ljZVByZWZlcmVuY2VzKGRlZmF1bHRMYW5ndWFnZSk7XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMubGFuZ3VhZ2UpIHtcbiAgICAgIHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSA9IGRlZmF1bHRMYW5ndWFnZTtcbiAgICAgIHRoaXMucGVyc2lzdFZvaWNlUHJlZmVyZW5jZXMoKTtcbiAgICB9XG4gICAgdGhpcy52b2ljZVN0YXRlID0ge1xuICAgICAgZW5hYmxlZDogZmFsc2UsXG4gICAgICBsaXN0ZW5pbmc6IGZhbHNlLFxuICAgICAgYXdhaXRpbmdSZXNwb25zZTogZmFsc2UsXG4gICAgICBtYW51YWxTdG9wOiBmYWxzZSxcbiAgICAgIHJlc3RhcnRUaW1lcjogbnVsbCxcbiAgICAgIGxhc3RUcmFuc2NyaXB0OiBcIlwiLFxuICAgIH07XG4gICAgdGhpcy5zcGVlY2ggPSBjcmVhdGVTcGVlY2hTZXJ2aWNlKHtcbiAgICAgIGRlZmF1bHRMYW5ndWFnZTogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlLFxuICAgIH0pO1xuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMudm9pY2VVUkkpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSk7XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMubGFuZ3VhZ2UpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldExhbmd1YWdlKHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSk7XG4gICAgfVxuICAgIGNvbnN0IHJlY29nbml0aW9uU3VwcG9ydGVkID0gdGhpcy5zcGVlY2guaXNSZWNvZ25pdGlvblN1cHBvcnRlZCgpO1xuICAgIGNvbnN0IHN5bnRoZXNpc1N1cHBvcnRlZCA9IHRoaXMuc3BlZWNoLmlzU3ludGhlc2lzU3VwcG9ydGVkKCk7XG4gICAgdGhpcy51aS5zZXRWb2ljZUF2YWlsYWJpbGl0eSh7XG4gICAgICByZWNvZ25pdGlvbjogcmVjb2duaXRpb25TdXBwb3J0ZWQsXG4gICAgICBzeW50aGVzaXM6IHN5bnRoZXNpc1N1cHBvcnRlZCxcbiAgICB9KTtcbiAgICB0aGlzLnVpLnNldFZvaWNlUHJlZmVyZW5jZXModGhpcy52b2ljZVByZWZzKTtcbiAgICBpZiAocmVjb2duaXRpb25TdXBwb3J0ZWQpIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiQWN0aXZleiBsZSBtaWNybyBwb3VyIGRpY3RlciB2b3RyZSBtZXNzYWdlLlwiLFxuICAgICAgICBcIm11dGVkXCIsXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoc3ludGhlc2lzU3VwcG9ydGVkKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkxlY3R1cmUgdm9jYWxlIGRpc3BvbmlibGUuIExhIGRpY3RcdTAwRTllIG5cdTAwRTljZXNzaXRlIHVuIG5hdmlnYXRldXIgY29tcGF0aWJsZS5cIixcbiAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICApO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkxlcyBmb25jdGlvbm5hbGl0XHUwMEU5cyB2b2NhbGVzIG5lIHNvbnQgcGFzIGRpc3BvbmlibGVzIGRhbnMgY2UgbmF2aWdhdGV1ci5cIixcbiAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICk7XG4gICAgfVxuICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUocmVjb2duaXRpb25TdXBwb3J0ZWQgPyA1MDAwIDogNzAwMCk7XG4gICAgdGhpcy5zcGVlY2gub24oXCJsaXN0ZW5pbmctY2hhbmdlXCIsIChwYXlsb2FkKSA9PlxuICAgICAgdGhpcy5oYW5kbGVWb2ljZUxpc3RlbmluZ0NoYW5nZShwYXlsb2FkKSxcbiAgICApO1xuICAgIHRoaXMuc3BlZWNoLm9uKFwidHJhbnNjcmlwdFwiLCAocGF5bG9hZCkgPT5cbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VUcmFuc2NyaXB0KHBheWxvYWQpLFxuICAgICk7XG4gICAgdGhpcy5zcGVlY2gub24oXCJlcnJvclwiLCAocGF5bG9hZCkgPT4gdGhpcy5oYW5kbGVWb2ljZUVycm9yKHBheWxvYWQpKTtcbiAgICB0aGlzLnNwZWVjaC5vbihcInNwZWFraW5nLWNoYW5nZVwiLCAocGF5bG9hZCkgPT5cbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VTcGVha2luZ0NoYW5nZShwYXlsb2FkKSxcbiAgICApO1xuICAgIHRoaXMuc3BlZWNoLm9uKFwidm9pY2VzXCIsICh7IHZvaWNlcyB9KSA9PlxuICAgICAgdGhpcy5oYW5kbGVWb2ljZVZvaWNlcyhBcnJheS5pc0FycmF5KHZvaWNlcykgPyB2b2ljZXMgOiBbXSksXG4gICAgKTtcbiAgfVxuXG4gIGFzeW5jIHRvZ2dsZVZvaWNlTGlzdGVuaW5nKCkge1xuICAgIGlmICghdGhpcy5zcGVlY2ggfHwgIXRoaXMuc3BlZWNoLmlzUmVjb2duaXRpb25TdXBwb3J0ZWQoKSkge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcbiAgICAgICAgXCJMYSBkaWN0XHUwMEU5ZSB2b2NhbGUgbidlc3QgcGFzIGRpc3BvbmlibGUgZGFucyBjZSBuYXZpZ2F0ZXVyLlwiLFxuICAgICAgICBcImRhbmdlclwiLFxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5saXN0ZW5pbmcgfHwgdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUubWFudWFsU3RvcCA9IHRydWU7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpIHtcbiAgICAgICAgd2luZG93LmNsZWFyVGltZW91dCh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKTtcbiAgICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgICB9XG4gICAgICB0aGlzLnNwZWVjaC5zdG9wTGlzdGVuaW5nKCk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiRGljdFx1MDBFOWUgaW50ZXJyb21wdWUuXCIsIFwibXV0ZWRcIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlU3RhdGUubWFudWFsU3RvcCA9IGZhbHNlO1xuICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gdHJ1ZTtcbiAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKSB7XG4gICAgICB3aW5kb3cuY2xlYXJUaW1lb3V0KHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpO1xuICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgfVxuICAgIGNvbnN0IHN0YXJ0ZWQgPSBhd2FpdCB0aGlzLnNwZWVjaC5zdGFydExpc3RlbmluZyh7XG4gICAgICBsYW5ndWFnZTogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlLFxuICAgICAgaW50ZXJpbVJlc3VsdHM6IHRydWUsXG4gICAgICBjb250aW51b3VzOiBmYWxzZSxcbiAgICB9KTtcbiAgICBpZiAoIXN0YXJ0ZWQpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkltcG9zc2libGUgZGUgZFx1MDBFOW1hcnJlciBsYSBkaWN0XHUwMEU5ZS4gVlx1MDBFOXJpZmlleiBsZSBtaWNyby5cIixcbiAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICk7XG4gICAgfVxuICB9XG5cbiAgaGFuZGxlVm9pY2VMaXN0ZW5pbmdDaGFuZ2UocGF5bG9hZCA9IHt9KSB7XG4gICAgY29uc3QgbGlzdGVuaW5nID0gQm9vbGVhbihwYXlsb2FkLmxpc3RlbmluZyk7XG4gICAgdGhpcy52b2ljZVN0YXRlLmxpc3RlbmluZyA9IGxpc3RlbmluZztcbiAgICBpZiAobGlzdGVuaW5nKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlTGlzdGVuaW5nKHRydWUpO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVRyYW5zY3JpcHQoXCJcIiwgeyBzdGF0ZTogXCJpZGxlXCIgfSk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiRW4gXHUwMEU5Y291dGVcdTIwMjYgUGFybGV6IGxvcnNxdWUgdm91cyBcdTAwRUF0ZXMgcHJcdTAwRUF0LlwiLCBcImluZm9cIik7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRoaXMudWkuc2V0Vm9pY2VMaXN0ZW5pbmcoZmFsc2UpO1xuICAgIGlmIChwYXlsb2FkLnJlYXNvbiA9PT0gXCJtYW51YWxcIikge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLm1hbnVhbFN0b3AgPSBmYWxzZTtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiRGljdFx1MDBFOWUgaW50ZXJyb21wdWUuXCIsIFwibXV0ZWRcIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAocGF5bG9hZC5yZWFzb24gPT09IFwiZXJyb3JcIikge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gZmFsc2U7XG4gICAgICBjb25zdCBtZXNzYWdlID1cbiAgICAgICAgcGF5bG9hZC5jb2RlID09PSBcIm5vdC1hbGxvd2VkXCJcbiAgICAgICAgICA/IFwiQXV0b3Jpc2V6IGwnYWNjXHUwMEU4cyBhdSBtaWNyb3Bob25lIHBvdXIgY29udGludWVyLlwiXG4gICAgICAgICAgOiBcIkxhIGRpY3RcdTAwRTllIHZvY2FsZSBzJ2VzdCBpbnRlcnJvbXB1ZS4gUlx1MDBFOWVzc2F5ZXouXCI7XG4gICAgICBjb25zdCB0b25lID0gcGF5bG9hZC5jb2RlID09PSBcIm5vdC1hbGxvd2VkXCIgPyBcImRhbmdlclwiIDogXCJ3YXJuaW5nXCI7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKG1lc3NhZ2UsIHRvbmUpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCkge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCAmJiAhdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgIHRoaXMubWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoNjUwKTtcbiAgICB9XG4gIH1cblxuICBoYW5kbGVWb2ljZVRyYW5zY3JpcHQocGF5bG9hZCA9IHt9KSB7XG4gICAgY29uc3QgdHJhbnNjcmlwdCA9IHR5cGVvZiBwYXlsb2FkLnRyYW5zY3JpcHQgPT09IFwic3RyaW5nXCIgPyBwYXlsb2FkLnRyYW5zY3JpcHQgOiBcIlwiO1xuICAgIGNvbnN0IGlzRmluYWwgPSBCb29sZWFuKHBheWxvYWQuaXNGaW5hbCk7XG4gICAgY29uc3QgY29uZmlkZW5jZSA9XG4gICAgICB0eXBlb2YgcGF5bG9hZC5jb25maWRlbmNlID09PSBcIm51bWJlclwiID8gcGF5bG9hZC5jb25maWRlbmNlIDogbnVsbDtcbiAgICBpZiAodHJhbnNjcmlwdCkge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmxhc3RUcmFuc2NyaXB0ID0gdHJhbnNjcmlwdDtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VUcmFuc2NyaXB0KHRyYW5zY3JpcHQsIHtcbiAgICAgICAgc3RhdGU6IGlzRmluYWwgPyBcImZpbmFsXCIgOiBcImludGVyaW1cIixcbiAgICAgIH0pO1xuICAgIH1cbiAgICBpZiAoIWlzRmluYWwpIHtcbiAgICAgIGlmICh0cmFuc2NyaXB0KSB7XG4gICAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXCJUcmFuc2NyaXB0aW9uIGVuIGNvdXJzXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgICAgIH1cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKCF0cmFuc2NyaXB0KSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiQXVjdW4gdGV4dGUgbidhIFx1MDBFOXRcdTAwRTkgcmVjb25udS5cIiwgXCJ3YXJuaW5nXCIpO1xuICAgICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSgzMDAwKTtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gZmFsc2U7XG4gICAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCkge1xuICAgICAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgICAgfVxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAodGhpcy52b2ljZVByZWZzLmF1dG9TZW5kKSB7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IHRydWU7XG4gICAgICBjb25zdCBjb25maWRlbmNlUGN0ID1cbiAgICAgICAgY29uZmlkZW5jZSAhPT0gbnVsbCA/IE1hdGgucm91bmQoTWF0aC5tYXgoMCwgTWF0aC5taW4oMSwgY29uZmlkZW5jZSkpICogMTAwKSA6IG51bGw7XG4gICAgICBpZiAoY29uZmlkZW5jZVBjdCAhPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICAgIGBFbnZvaSBkdSBtZXNzYWdlIGRpY3RcdTAwRTkgKCR7Y29uZmlkZW5jZVBjdH0lIGRlIGNvbmZpYW5jZSlcdTIwMjZgLFxuICAgICAgICAgIFwiaW5mb1wiLFxuICAgICAgICApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkVudm9pIGR1IG1lc3NhZ2UgZGljdFx1MDBFOVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB9XG4gICAgICB0aGlzLnN1Ym1pdFZvaWNlUHJvbXB0KHRyYW5zY3JpcHQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSB0cmFuc2NyaXB0O1xuICAgICAgfVxuICAgICAgdGhpcy51aS51cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICB0aGlzLnVpLmF1dG9zaXplUHJvbXB0KCk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIk1lc3NhZ2UgZGljdFx1MDBFOS4gVlx1MDBFOXJpZmlleiBhdmFudCBsJ2Vudm9pLlwiLFxuICAgICAgICBcImluZm9cIixcbiAgICAgICk7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDQ1MDApO1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICB9XG4gIH1cblxuICBoYW5kbGVWb2ljZUVycm9yKHBheWxvYWQgPSB7fSkge1xuICAgIGNvbnN0IG1lc3NhZ2UgPVxuICAgICAgdHlwZW9mIHBheWxvYWQubWVzc2FnZSA9PT0gXCJzdHJpbmdcIiAmJiBwYXlsb2FkLm1lc3NhZ2UubGVuZ3RoID4gMFxuICAgICAgICA/IHBheWxvYWQubWVzc2FnZVxuICAgICAgICA6IFwiVW5lIGVycmV1ciB2b2NhbGUgZXN0IHN1cnZlbnVlLlwiO1xuICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMobWVzc2FnZSwgXCJkYW5nZXJcIik7XG4gICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKSB7XG4gICAgICB3aW5kb3cuY2xlYXJUaW1lb3V0KHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpO1xuICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgfVxuICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoNjAwMCk7XG4gIH1cblxuICBoYW5kbGVWb2ljZVNwZWFraW5nQ2hhbmdlKHBheWxvYWQgPSB7fSkge1xuICAgIGNvbnN0IHNwZWFraW5nID0gQm9vbGVhbihwYXlsb2FkLnNwZWFraW5nKTtcbiAgICB0aGlzLnVpLnNldFZvaWNlU3BlYWtpbmcoc3BlYWtpbmcpO1xuICAgIGlmIChzcGVha2luZykge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkxlY3R1cmUgZGUgbGEgclx1MDBFOXBvbnNlXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCAmJiB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCAmJiAhdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgIHRoaXMubWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoODAwKTtcbiAgICB9XG4gICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSgzNTAwKTtcbiAgfVxuXG4gIGhhbmRsZVZvaWNlVm9pY2VzKHZvaWNlcyA9IFtdKSB7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHZvaWNlcykpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgbGV0IHNlbGVjdGVkVXJpID0gdGhpcy52b2ljZVByZWZzLnZvaWNlVVJJO1xuICAgIGlmICghc2VsZWN0ZWRVcmkgJiYgdm9pY2VzLmxlbmd0aCA+IDApIHtcbiAgICAgIGNvbnN0IHByZWZlcnJlZCA9IHZvaWNlcy5maW5kKCh2b2ljZSkgPT4ge1xuICAgICAgICBpZiAoIXZvaWNlIHx8ICF2b2ljZS5sYW5nKSB7XG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGxhbmcgPSBTdHJpbmcodm9pY2UubGFuZykudG9Mb3dlckNhc2UoKTtcbiAgICAgICAgY29uc3QgdGFyZ2V0ID0gKHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSB8fCBcIlwiKS50b0xvd2VyQ2FzZSgpO1xuICAgICAgICByZXR1cm4gdGFyZ2V0ICYmIGxhbmcuc3RhcnRzV2l0aCh0YXJnZXQuc2xpY2UoMCwgMikpO1xuICAgICAgfSk7XG4gICAgICBpZiAocHJlZmVycmVkKSB7XG4gICAgICAgIHNlbGVjdGVkVXJpID0gcHJlZmVycmVkLnZvaWNlVVJJIHx8IG51bGw7XG4gICAgICAgIHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSA9IHNlbGVjdGVkVXJpO1xuICAgICAgICB0aGlzLnBlcnNpc3RWb2ljZVByZWZlcmVuY2VzKCk7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMudWkuc2V0Vm9pY2VWb2ljZU9wdGlvbnModm9pY2VzLCBzZWxlY3RlZFVyaSB8fCBudWxsKTtcbiAgICBpZiAoc2VsZWN0ZWRVcmkpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHNlbGVjdGVkVXJpKTtcbiAgICB9XG4gIH1cblxuICBoYW5kbGVWb2ljZUF1dG9TZW5kQ2hhbmdlKGVuYWJsZWQpIHtcbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQgPSBCb29sZWFuKGVuYWJsZWQpO1xuICAgIHRoaXMucGVyc2lzdFZvaWNlUHJlZmVyZW5jZXMoKTtcbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCkge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICAgIGlmICh0aGlzLnZvaWNlU3RhdGUubGlzdGVuaW5nKSB7XG4gICAgICAgIHRoaXMuc3BlZWNoLnN0b3BMaXN0ZW5pbmcoKTtcbiAgICAgIH1cbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiTW9kZSBtYW51ZWwgYWN0aXZcdTAwRTkuIFV0aWxpc2V6IGxlIG1pY3JvIHBvdXIgcmVtcGxpciBsZSBjaGFtcC5cIixcbiAgICAgICAgXCJtdXRlZFwiLFxuICAgICAgKTtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoNDAwMCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiTGVzIG1lc3NhZ2VzIGRpY3RcdTAwRTlzIHNlcm9udCBlbnZveVx1MDBFOXMgYXV0b21hdGlxdWVtZW50LlwiLFxuICAgICAgICBcImluZm9cIixcbiAgICAgICk7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgIH1cbiAgfVxuXG4gIGhhbmRsZVZvaWNlUGxheWJhY2tDaGFuZ2UoZW5hYmxlZCkge1xuICAgIGlmICghdGhpcy52b2ljZVByZWZzKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IG5leHQgPSBCb29sZWFuKGVuYWJsZWQpO1xuICAgIHRoaXMudm9pY2VQcmVmcy5wbGF5YmFjayA9IG5leHQ7XG4gICAgdGhpcy5wZXJzaXN0Vm9pY2VQcmVmZXJlbmNlcygpO1xuICAgIGlmICghbmV4dCkge1xuICAgICAgdGhpcy5zdG9wVm9pY2VQbGF5YmFjaygpO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkxlY3R1cmUgdm9jYWxlIGRcdTAwRTlzYWN0aXZcdTAwRTllLlwiLCBcIm11dGVkXCIpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiTGVjdHVyZSB2b2NhbGUgYWN0aXZcdTAwRTllLlwiLCBcImluZm9cIik7XG4gICAgfVxuICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gIH1cblxuICBoYW5kbGVWb2ljZVZvaWNlQ2hhbmdlKHZvaWNlVVJJKSB7XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdmFsdWUgPSB2b2ljZVVSSSAmJiB2b2ljZVVSSS5sZW5ndGggPiAwID8gdm9pY2VVUkkgOiBudWxsO1xuICAgIHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSA9IHZhbHVlO1xuICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHZhbHVlKTtcbiAgICB0aGlzLnBlcnNpc3RWb2ljZVByZWZlcmVuY2VzKCk7XG4gICAgaWYgKHZhbHVlKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiVm9peCBzXHUwMEU5bGVjdGlvbm5cdTAwRTllIG1pc2UgXHUwMEUwIGpvdXIuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIlZvaXggcGFyIGRcdTAwRTlmYXV0IGR1IHN5c3RcdTAwRThtZSB1dGlsaXNcdTAwRTllLlwiLCBcIm11dGVkXCIpO1xuICAgIH1cbiAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDMwMDApO1xuICB9XG5cbiAgc3RvcFZvaWNlUGxheWJhY2soKSB7XG4gICAgaWYgKCF0aGlzLnNwZWVjaCB8fCAhdGhpcy5zcGVlY2guaXNTeW50aGVzaXNTdXBwb3J0ZWQoKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnNwZWVjaC5zdG9wU3BlYWtpbmcoKTtcbiAgICB0aGlzLnVpLnNldFZvaWNlU3BlYWtpbmcoZmFsc2UpO1xuICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXCJMZWN0dXJlIHZvY2FsZSBpbnRlcnJvbXB1ZS5cIiwgXCJtdXRlZFwiKTtcbiAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDMwMDApO1xuICB9XG5cbiAgbWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoZGVsYXkgPSA2NTApIHtcbiAgICBpZiAoIXRoaXMuc3BlZWNoIHx8ICF0aGlzLnNwZWVjaC5pc1JlY29nbml0aW9uU3VwcG9ydGVkKCkpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQgfHwgIXRoaXMudm9pY2VTdGF0ZS5lbmFibGVkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUubGlzdGVuaW5nIHx8IHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKSB7XG4gICAgICB3aW5kb3cuY2xlYXJUaW1lb3V0KHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCB8fCAhdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5saXN0ZW5pbmcgfHwgdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgY29uc3QgYXR0ZW1wdCA9IHRoaXMuc3BlZWNoLnN0YXJ0TGlzdGVuaW5nKHtcbiAgICAgICAgbGFuZ3VhZ2U6IHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSxcbiAgICAgICAgaW50ZXJpbVJlc3VsdHM6IHRydWUsXG4gICAgICAgIGNvbnRpbnVvdXM6IGZhbHNlLFxuICAgICAgfSk7XG4gICAgICBQcm9taXNlLnJlc29sdmUoYXR0ZW1wdClcbiAgICAgICAgLnRoZW4oKHN0YXJ0ZWQpID0+IHtcbiAgICAgICAgICBpZiAoc3RhcnRlZCkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgIH1cbiAgICAgICAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgICAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgICAgICBcIkltcG9zc2libGUgZGUgcmVsYW5jZXIgbGEgZGljdFx1MDBFOWUgdm9jYWxlLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9KVxuICAgICAgICAuY2F0Y2goKGVycikgPT4ge1xuICAgICAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICAgICAgY29uc29sZS5lcnJvcihcIkF1dG9tYXRpYyB2b2ljZSByZXN0YXJ0IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgICAgICBcIkltcG9zc2libGUgZGUgcmVsYW5jZXIgbGEgZGljdFx1MDBFOWUgdm9jYWxlLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9LCBkZWxheSk7XG4gIH1cblxuICBzdWJtaXRWb2ljZVByb21wdCh0ZXh0KSB7XG4gICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC52YWx1ZSA9IHRleHQ7XG4gICAgfVxuICAgIHRoaXMudWkudXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICB0aGlzLnVpLmVtaXQoXCJzdWJtaXRcIiwgeyB0ZXh0IH0pO1xuICB9XG5cbiAgZ2V0TGF0ZXN0QXNzaXN0YW50VGV4dCgpIHtcbiAgICBpZiAoIXRoaXMudGltZWxpbmVTdG9yZSB8fCAhdGhpcy50aW1lbGluZVN0b3JlLm9yZGVyKSB7XG4gICAgICByZXR1cm4gXCJcIjtcbiAgICB9XG4gICAgZm9yIChsZXQgaSA9IHRoaXMudGltZWxpbmVTdG9yZS5vcmRlci5sZW5ndGggLSAxOyBpID49IDA7IGkgLT0gMSkge1xuICAgICAgY29uc3QgaWQgPSB0aGlzLnRpbWVsaW5lU3RvcmUub3JkZXJbaV07XG4gICAgICBjb25zdCBlbnRyeSA9IHRoaXMudGltZWxpbmVTdG9yZS5tYXAuZ2V0KGlkKTtcbiAgICAgIGlmIChlbnRyeSAmJiBlbnRyeS5yb2xlID09PSBcImFzc2lzdGFudFwiICYmIGVudHJ5LnRleHQpIHtcbiAgICAgICAgcmV0dXJuIGVudHJ5LnRleHQ7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBcIlwiO1xuICB9XG5cbiAgaGFuZGxlVm9pY2VBc3Npc3RhbnRDb21wbGV0aW9uKCkge1xuICAgIGlmICghdGhpcy52b2ljZVByZWZzKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGxhdGVzdCA9IHRoaXMuZ2V0TGF0ZXN0QXNzaXN0YW50VGV4dCgpO1xuICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gZmFsc2U7XG4gICAgaWYgKCFsYXRlc3QpIHtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gICAgICB0aGlzLm1heWJlUmVzdGFydFZvaWNlTGlzdGVuaW5nKDgwMCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMucGxheWJhY2sgJiYgdGhpcy5zcGVlY2ggJiYgdGhpcy5zcGVlY2guaXNTeW50aGVzaXNTdXBwb3J0ZWQoKSkge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkxlY3R1cmUgZGUgbGEgclx1MDBFOXBvbnNlXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgICAgIGNvbnN0IHV0dGVyYW5jZSA9IHRoaXMuc3BlZWNoLnNwZWFrKGxhdGVzdCwge1xuICAgICAgICBsYW5nOiB0aGlzLnZvaWNlUHJlZnMubGFuZ3VhZ2UsXG4gICAgICAgIHZvaWNlVVJJOiB0aGlzLnZvaWNlUHJlZnMudm9pY2VVUkksXG4gICAgICB9KTtcbiAgICAgIGlmICghdXR0ZXJhbmNlKSB7XG4gICAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gICAgICAgIHRoaXMubWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoODAwKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSgzNTAwKTtcbiAgICAgIHRoaXMubWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoODAwKTtcbiAgICB9XG4gIH1cblxuICBoYW5kbGVTb2NrZXRFdmVudChldikge1xuICAgIGNvbnN0IHR5cGUgPSBldiAmJiBldi50eXBlID8gZXYudHlwZSA6IFwiXCI7XG4gICAgY29uc3QgZGF0YSA9IGV2ICYmIGV2LmRhdGEgPyBldi5kYXRhIDoge307XG4gICAgc3dpdGNoICh0eXBlKSB7XG4gICAgICBjYXNlIFwid3MuY29ubmVjdGVkXCI6IHtcbiAgICAgICAgaWYgKGRhdGEgJiYgZGF0YS5vcmlnaW4pIHtcbiAgICAgICAgICB0aGlzLnVpLmFubm91bmNlQ29ubmVjdGlvbihgQ29ubmVjdFx1MDBFOSB2aWEgJHtkYXRhLm9yaWdpbn1gKTtcbiAgICAgICAgICB0aGlzLnVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFxuICAgICAgICAgICAgYENvbm5lY3RcdTAwRTkgdmlhICR7ZGF0YS5vcmlnaW59YCxcbiAgICAgICAgICAgIFwic3VjY2Vzc1wiLFxuICAgICAgICAgICk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgdGhpcy51aS5hbm5vdW5jZUNvbm5lY3Rpb24oXCJDb25uZWN0XHUwMEU5IGF1IHNlcnZldXIuXCIpO1xuICAgICAgICAgIHRoaXMudWkudXBkYXRlQ29ubmVjdGlvbk1ldGEoXCJDb25uZWN0XHUwMEU5IGF1IHNlcnZldXIuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJoaXN0b3J5LnNuYXBzaG90XCI6IHtcbiAgICAgICAgaWYgKGRhdGEgJiYgQXJyYXkuaXNBcnJheShkYXRhLml0ZW1zKSkge1xuICAgICAgICAgIHRoaXMudWkucmVuZGVySGlzdG9yeShkYXRhLml0ZW1zLCB7IHJlcGxhY2U6IHRydWUgfSk7XG4gICAgICAgIH1cbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiYWlfbW9kZWwucmVzcG9uc2VfY2h1bmtcIjoge1xuICAgICAgICBjb25zdCBkZWx0YSA9XG4gICAgICAgICAgdHlwZW9mIGRhdGEuZGVsdGEgPT09IFwic3RyaW5nXCIgPyBkYXRhLmRlbHRhIDogZGF0YS50ZXh0IHx8IFwiXCI7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kU3RyZWFtKGRlbHRhKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiYWlfbW9kZWwucmVzcG9uc2VfY29tcGxldGVcIjoge1xuICAgICAgICBpZiAoZGF0YSAmJiBkYXRhLnRleHQgJiYgIXRoaXMudWkuaGFzU3RyZWFtQnVmZmVyKCkpIHtcbiAgICAgICAgICB0aGlzLnVpLmFwcGVuZFN0cmVhbShkYXRhLnRleHQpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuZW5kU3RyZWFtKGRhdGEpO1xuICAgICAgICB0aGlzLnVpLnNldEJ1c3koZmFsc2UpO1xuICAgICAgICBpZiAoZGF0YSAmJiB0eXBlb2YgZGF0YS5sYXRlbmN5X21zICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICAgICAgdGhpcy51aS5zZXREaWFnbm9zdGljcyh7IGxhdGVuY3lNczogTnVtYmVyKGRhdGEubGF0ZW5jeV9tcykgfSk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGRhdGEgJiYgZGF0YS5vayA9PT0gZmFsc2UgJiYgZGF0YS5lcnJvcikge1xuICAgICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcInN5c3RlbVwiLCBkYXRhLmVycm9yLCB7XG4gICAgICAgICAgICB2YXJpYW50OiBcImVycm9yXCIsXG4gICAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy5oYW5kbGVWb2ljZUFzc2lzdGFudENvbXBsZXRpb24oKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiY2hhdC5tZXNzYWdlXCI6IHtcbiAgICAgICAgaWYgKCF0aGlzLnVpLmlzU3RyZWFtaW5nKCkpIHtcbiAgICAgICAgICB0aGlzLnVpLnN0YXJ0U3RyZWFtKCk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKFxuICAgICAgICAgIGRhdGEgJiZcbiAgICAgICAgICB0eXBlb2YgZGF0YS5yZXNwb25zZSA9PT0gXCJzdHJpbmdcIiAmJlxuICAgICAgICAgICF0aGlzLnVpLmhhc1N0cmVhbUJ1ZmZlcigpXG4gICAgICAgICkge1xuICAgICAgICAgIHRoaXMudWkuYXBwZW5kU3RyZWFtKGRhdGEucmVzcG9uc2UpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuZW5kU3RyZWFtKGRhdGEpO1xuICAgICAgICB0aGlzLnVpLnNldEJ1c3koZmFsc2UpO1xuICAgICAgICB0aGlzLmhhbmRsZVZvaWNlQXNzaXN0YW50Q29tcGxldGlvbigpO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJldm9sdXRpb25fZW5naW5lLnRyYWluaW5nX2NvbXBsZXRlXCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFxuICAgICAgICAgIFwic3lzdGVtXCIsXG4gICAgICAgICAgYFx1MDBDOXZvbHV0aW9uIG1pc2UgXHUwMEUwIGpvdXIgJHtkYXRhICYmIGRhdGEudmVyc2lvbiA/IGRhdGEudmVyc2lvbiA6IFwiXCJ9YCxcbiAgICAgICAgICB7XG4gICAgICAgICAgICB2YXJpYW50OiBcIm9rXCIsXG4gICAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiZXZvbHV0aW9uX2VuZ2luZS50cmFpbmluZ19mYWlsZWRcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXG4gICAgICAgICAgXCJzeXN0ZW1cIixcbiAgICAgICAgICBgXHUwMEM5Y2hlYyBkZSBsJ1x1MDBFOXZvbHV0aW9uIDogJHtkYXRhICYmIGRhdGEuZXJyb3IgPyBkYXRhLmVycm9yIDogXCJpbmNvbm51XCJ9YCxcbiAgICAgICAgICB7XG4gICAgICAgICAgICB2YXJpYW50OiBcImVycm9yXCIsXG4gICAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwic2xlZXBfdGltZV9jb21wdXRlLnBoYXNlX3N0YXJ0XCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFxuICAgICAgICAgIFwic3lzdGVtXCIsXG4gICAgICAgICAgXCJPcHRpbWlzYXRpb24gZW4gYXJyaVx1MDBFOHJlLXBsYW4gZFx1MDBFOW1hcnJcdTAwRTllXHUyMDI2XCIsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJoaW50XCIsXG4gICAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwic2xlZXBfdGltZV9jb21wdXRlLmNyZWF0aXZlX3BoYXNlXCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFxuICAgICAgICAgIFwic3lzdGVtXCIsXG4gICAgICAgICAgYEV4cGxvcmF0aW9uIGRlICR7TnVtYmVyKGRhdGEgJiYgZGF0YS5pZGVhcyA/IGRhdGEuaWRlYXMgOiAxKX0gaWRcdTAwRTllc1x1MjAyNmAsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJoaW50XCIsXG4gICAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwicGVyZm9ybWFuY2UuYWxlcnRcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJzeXN0ZW1cIiwgYFBlcmYgOiAke3RoaXMudWkuZm9ybWF0UGVyZihkYXRhKX1gLCB7XG4gICAgICAgICAgdmFyaWFudDogXCJ3YXJuXCIsXG4gICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgfSk7XG4gICAgICAgIGlmIChkYXRhICYmIHR5cGVvZiBkYXRhLnR0ZmJfbXMgIT09IFwidW5kZWZpbmVkXCIpIHtcbiAgICAgICAgICB0aGlzLnVpLnNldERpYWdub3N0aWNzKHsgbGF0ZW5jeU1zOiBOdW1iZXIoZGF0YS50dGZiX21zKSB9KTtcbiAgICAgICAgfVxuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJ1aS5zdWdnZXN0aW9uc1wiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwbHlRdWlja0FjdGlvbk9yZGVyaW5nKFxuICAgICAgICAgIEFycmF5LmlzQXJyYXkoZGF0YS5hY3Rpb25zKSA/IGRhdGEuYWN0aW9ucyA6IFtdLFxuICAgICAgICApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIGlmICh0eXBlICYmIHR5cGUuc3RhcnRzV2l0aChcIndzLlwiKSkge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBjb25zb2xlLmRlYnVnKFwiVW5oYW5kbGVkIGV2ZW50XCIsIGV2KTtcbiAgICB9XG4gIH1cbn1cbiIsICIvKipcbiAqIEVudHJ5IHBvaW50IGZvciB0aGUgY2hhdCBhcHBsaWNhdGlvbi5cbiAqIEV4cGVjdHMgd2luZG93LmNoYXRDb25maWcgdG8gYmUgZGVmaW5lZCBieSB0aGUgc2VydmVyLXJlbmRlcmVkIHRlbXBsYXRlLlxuICogRmFsbHMgYmFjayB0byBhbiBlbXB0eSBjb25maWcgaWYgbm90IHByZXNlbnQuXG4gKiBSZXF1aXJlZCBjb25maWcgc2hhcGU6IHsgYXBpVXJsPywgd3NVcmw/LCB0b2tlbj8sIC4uLiB9XG4gKi9cbmltcG9ydCB7IENoYXRBcHAgfSBmcm9tIFwiLi9hcHAuanNcIjtcblxubmV3IENoYXRBcHAoZG9j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
