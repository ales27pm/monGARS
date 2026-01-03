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
    const embedCandidate = typeof config.embedServiceUrl === "string" ? config.embedServiceUrl.trim() : "";
    if (embedCandidate) {
      try {
        const url = new URL(embedCandidate);
        if (url.protocol === "http:" || url.protocol === "https:") {
          config.embedServiceUrl = url.toString();
        } else {
          console.warn("Unsupported embedding service protocol", url.protocol);
          config.embedServiceUrl = null;
        }
      } catch (err) {
        console.warn("Invalid embedding service URL", err, embedCandidate);
        config.embedServiceUrl = null;
      }
    } else {
      config.embedServiceUrl = null;
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

  // src/utils/errorUtils.js
  var DEFAULT_ERROR_MESSAGE = "Une erreur inattendue est survenue. Veuillez r\xE9essayer.";
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
      const candidate = typeof error.message === "string" && error.message.trim() || typeof error.error === "string" && error.error.trim();
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
    const basePrefix = options.prefix === null ? "" : typeof prefix === "string" ? prefix : "Erreur : ";
    const trimmedPrefix = basePrefix.trim().toLowerCase();
    const shouldPrefix = Boolean(basePrefix) && !ERROR_PREFIX_REGEX.test(text) && !(trimmedPrefix && text.toLowerCase().startsWith(trimmedPrefix));
    const bubbleText = shouldPrefix ? `${basePrefix}${text}` : text;
    return { text, bubbleText };
  }

  // src/ui/chatUi.js
  function createChatUi({ elements, timelineStore }) {
    var _a, _b, _c;
    const emitter = createEmitter();
    const sendLabelElement = ((_a = elements.send) == null ? void 0 : _a.querySelector("[data-role='send-label']")) || null;
    const sendSpinnerElement = ((_b = elements.send) == null ? void 0 : _b.querySelector("[data-role='send-spinner']")) || null;
    const sendIdleMarkup = elements.send ? elements.send.innerHTML : "";
    const sendIdleLabel = elements.send && elements.send.getAttribute("data-idle-label") || sendLabelElement && sendLabelElement.textContent.trim() || (elements.send ? elements.send.textContent.trim() : "Envoyer");
    const sendBusyLabel = elements.send && elements.send.getAttribute("data-busy-label") || "Envoi\u2026";
    const sendBusyAriaLabel = elements.send && elements.send.getAttribute("data-busy-aria-label") || sendBusyLabel;
    const sendBusyMarkup = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Envoi\u2026';
    const SUPPORTED_TONES = ["muted", "info", "success", "danger", "warning"];
    const composerStatusDefault = elements.composerStatus && elements.composerStatus.textContent.trim() || "Appuyez sur Ctrl+Entr\xE9e pour envoyer rapidement.";
    const filterHintDefault = elements.filterHint && elements.filterHint.textContent.trim() || "Utilisez le filtre pour limiter l'historique. Appuyez sur \xC9chap pour effacer.";
    const voiceStatusDefault = elements.voiceStatus && elements.voiceStatus.textContent.trim() || "V\xE9rification des capacit\xE9s vocales\u2026";
    const promptMax = Number((_c = elements.prompt) == null ? void 0 : _c.getAttribute("maxlength")) || null;
    const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const SCROLL_THRESHOLD = 140;
    const PROMPT_MAX_HEIGHT = 320;
    const composerStatusEmbedding = elements.composerStatus && elements.composerStatus.getAttribute("data-embed-label") || "Mode Embedding : g\xE9n\xE9rez des vecteurs pour vos textes.";
    const promptPlaceholderDefault = elements.prompt && elements.prompt.getAttribute("placeholder") || "";
    const promptPlaceholderEmbedding = elements.prompt && elements.prompt.getAttribute("data-embed-placeholder") || "Entrez le texte \xE0 encoder\u2026";
    const promptAriaDefault = elements.prompt && elements.prompt.getAttribute("aria-label") || "";
    const promptAriaEmbedding = elements.prompt && elements.prompt.getAttribute("data-embed-aria-label") || "Texte \xE0 encoder";
    const sendAriaDefault = elements.send && elements.send.getAttribute("aria-label") || sendIdleLabel;
    const sendAriaEmbedding = elements.send && elements.send.getAttribute("data-embed-aria-label") || "G\xE9n\xE9rer un embedding";
    const diagnostics = {
      connectedAt: null,
      lastMessageAt: null,
      latencyMs: null
    };
    const state = {
      resetStatusTimer: null,
      hideScrollTimer: null,
      voiceStatusTimer: null,
      mode: "chat",
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
    function normaliseMode(mode) {
      return mode === "embed" ? "embed" : "chat";
    }
    function setBusy(busy) {
      elements.transcript.setAttribute("aria-busy", busy ? "true" : "false");
      if (!elements.send) {
        return;
      }
      elements.send.disabled = Boolean(busy);
      elements.send.setAttribute("aria-busy", busy ? "true" : "false");
      elements.send.classList.toggle("is-loading", Boolean(busy));
      if (sendSpinnerElement) {
        if (busy) {
          sendSpinnerElement.classList.remove("d-none");
          sendSpinnerElement.setAttribute("aria-hidden", "false");
        } else {
          sendSpinnerElement.classList.add("d-none");
          sendSpinnerElement.setAttribute("aria-hidden", "true");
        }
      }
      if (sendLabelElement) {
        sendLabelElement.textContent = busy ? sendBusyLabel : sendIdleLabel;
      } else if (busy) {
        elements.send.innerHTML = sendBusyMarkup;
      } else if (sendIdleMarkup) {
        elements.send.innerHTML = sendIdleMarkup;
      } else {
        elements.send.textContent = sendIdleLabel;
      }
      if (busy) {
        if (sendBusyAriaLabel) {
          elements.send.setAttribute("aria-label", sendBusyAriaLabel);
        }
      } else {
        const ariaLabel = state.mode === "embed" ? sendAriaEmbedding : sendAriaDefault;
        if (ariaLabel) {
          elements.send.setAttribute("aria-label", ariaLabel);
        } else {
          elements.send.removeAttribute("aria-label");
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
      const message = state.mode === "embed" ? composerStatusEmbedding : composerStatusDefault;
      setComposerStatus(message, "muted");
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
    const setVoiceAvailability = ({
      recognition = false,
      synthesis = false
    } = {}) => {
      if (elements.voiceControls) {
        elements.voiceControls.classList.toggle(
          "d-none",
          !recognition && !synthesis
        );
      }
      if (elements.voiceRecognitionGroup) {
        elements.voiceRecognitionGroup.classList.toggle("d-none", !recognition);
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
      elements.voiceToggle.setAttribute(
        "aria-pressed",
        listening ? "true" : "false"
      );
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
    function setMode(mode, options = {}) {
      const next = normaliseMode(mode);
      const previous = state.mode;
      state.mode = next;
      if (elements.modeSelect && elements.modeSelect.value !== next) {
        elements.modeSelect.value = next;
      }
      if (elements.composer) {
        elements.composer.dataset.mode = next;
      }
      if (elements.prompt) {
        const placeholder = next === "embed" ? promptPlaceholderEmbedding : promptPlaceholderDefault;
        if (placeholder) {
          elements.prompt.setAttribute("placeholder", placeholder);
        } else {
          elements.prompt.removeAttribute("placeholder");
        }
        const ariaLabel = next === "embed" ? promptAriaEmbedding : promptAriaDefault;
        if (ariaLabel) {
          elements.prompt.setAttribute("aria-label", ariaLabel);
        } else {
          elements.prompt.removeAttribute("aria-label");
        }
      }
      if (elements.send) {
        const ariaLabel = next === "embed" ? sendAriaEmbedding : sendAriaDefault;
        if (ariaLabel) {
          elements.send.setAttribute("aria-label", ariaLabel);
        } else {
          elements.send.removeAttribute("aria-label");
        }
      }
      if (!options.skipStatus && (previous !== next || options.forceStatus)) {
        setComposerStatusIdle();
      }
      return next;
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
    function normaliseNumeric(value) {
      if (typeof value === "number" && Number.isFinite(value)) {
        return value;
      }
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function formatNumeric(value) {
      if (!Number.isFinite(value)) {
        return "\u2014";
      }
      if (value === 0) {
        return "0";
      }
      const abs = Math.abs(value);
      if (abs >= 1e3 || abs < 1e-3) {
        return value.toExponential(4);
      }
      const fixed = value.toFixed(6);
      return fixed.replace(/\.0+$/, "").replace(/0+$/, "");
    }
    function summariseVector(vector, index) {
      if (!Array.isArray(vector)) {
        return null;
      }
      const preview = [];
      let count = 0;
      let sum = 0;
      let squares = 0;
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < vector.length; i += 1) {
        const value = normaliseNumeric(vector[i]);
        if (value === null) {
          continue;
        }
        if (preview.length < 8) {
          preview.push(value);
        }
        count += 1;
        sum += value;
        squares += value * value;
        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
      const magnitude = count > 0 ? Math.sqrt(squares) : null;
      const mean = count > 0 ? sum / count : null;
      return {
        index,
        count,
        sum,
        squares,
        magnitude,
        mean,
        min: count > 0 ? min : null,
        max: count > 0 ? max : null,
        preview
      };
    }
    function createVectorStatsTable(stats) {
      const table = document.createElement("table");
      table.className = "table table-sm table-striped embedding-details-table mb-0";
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      [
        "Vecteur",
        "Composantes",
        "Magnitude",
        "Moyenne",
        "Min",
        "Max",
        "Aper\xE7u"
      ].forEach((label) => {
        const th = document.createElement("th");
        th.scope = "col";
        th.textContent = label;
        headerRow.appendChild(th);
      });
      thead.appendChild(headerRow);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      stats.forEach((stat) => {
        const row = document.createElement("tr");
        const cells = [
          stat.index + 1,
          stat.count,
          formatNumeric(stat.magnitude),
          formatNumeric(stat.mean),
          formatNumeric(stat.min),
          formatNumeric(stat.max),
          stat.preview.length ? stat.preview.map((value) => formatNumeric(value)).join(", ") : "\u2014"
        ];
        cells.forEach((value) => {
          const td = document.createElement("td");
          td.textContent = String(value);
          row.appendChild(td);
        });
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      return table;
    }
    function attachEmbeddingDetails(row, embeddingData = {}, metadata = {}) {
      var _a2;
      if (!row) {
        return;
      }
      const bubble = row.querySelector(".chat-bubble");
      if (!bubble) {
        return;
      }
      bubble.querySelectorAll(".embedding-details").forEach((node) => node.remove());
      const vectors = Array.isArray(embeddingData.vectors) ? embeddingData.vectors.filter((vector) => Array.isArray(vector)) : [];
      if (vectors.length === 0) {
        return;
      }
      const stats = vectors.map((vector, index) => summariseVector(vector, index)).filter((entry) => entry && entry.count >= 0);
      if (stats.length === 0) {
        return;
      }
      const details = document.createElement("div");
      details.className = "embedding-details card mt-3";
      const cardBody = document.createElement("div");
      cardBody.className = "card-body p-3";
      details.appendChild(cardBody);
      const header = document.createElement("div");
      header.className = "d-flex flex-wrap align-items-center gap-2 mb-3";
      const title = document.createElement("h5");
      title.className = "card-title mb-0";
      title.textContent = "Analyse des embeddings";
      header.appendChild(title);
      const downloadBtn = document.createElement("button");
      downloadBtn.type = "button";
      downloadBtn.className = "btn btn-sm btn-outline-primary ms-auto";
      downloadBtn.textContent = "T\xE9l\xE9charger le JSON";
      downloadBtn.addEventListener("click", () => {
        var _a3, _b2, _c2, _d, _e, _f, _g, _h;
        try {
          const payload = typeof embeddingData.raw === "object" && embeddingData.raw !== null ? embeddingData.raw : {
            backend: (_b2 = (_a3 = embeddingData.backend) != null ? _a3 : metadata.backend) != null ? _b2 : null,
            model: (_d = (_c2 = embeddingData.model) != null ? _c2 : metadata.model) != null ? _d : null,
            dims: (_h = (_g = (_e = embeddingData.dims) != null ? _e : metadata.dims) != null ? _g : (_f = stats[0]) == null ? void 0 : _f.count) != null ? _h : null,
            normalised: typeof embeddingData.normalised !== "undefined" ? Boolean(embeddingData.normalised) : Boolean(metadata.normalised),
            count: vectors.length,
            vectors
          };
          const blob = new Blob([JSON.stringify(payload, null, 2)], {
            type: "application/json"
          });
          const url = window.URL.createObjectURL(blob);
          const link = document.createElement("a");
          const slugSource = (embeddingData.model || metadata.model || "embedding").toString().toLowerCase();
          const slug = slugSource.replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
          link.href = url;
          link.download = `embedding-${slug || "result"}-${Date.now()}.json`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          window.setTimeout(() => {
            window.URL.revokeObjectURL(url);
          }, 1e3);
        } catch (err) {
          console.warn("Unable to download embedding payload", err);
          announceConnection(
            "Impossible de t\xE9l\xE9charger le r\xE9sultat d'embedding.",
            "danger"
          );
        }
      });
      header.appendChild(downloadBtn);
      cardBody.appendChild(header);
      const dimsCandidate = Number((_a2 = embeddingData.dims) != null ? _a2 : metadata.dims);
      const dims = Number.isFinite(dimsCandidate) ? Number(dimsCandidate) : Array.isArray(vectors[0]) ? vectors[0].length : null;
      const validMagnitudeStats = stats.filter(
        (stat) => typeof stat.magnitude === "number" && !Number.isNaN(stat.magnitude)
      );
      const totalMagnitude = validMagnitudeStats.reduce(
        (acc, stat) => acc + stat.magnitude,
        0
      );
      const avgMagnitude = validMagnitudeStats.length > 0 ? totalMagnitude / validMagnitudeStats.length : null;
      let componentCount = 0;
      let componentSum = 0;
      let componentSquares = 0;
      let globalMin = null;
      let globalMax = null;
      stats.forEach((stat) => {
        componentCount += stat.count;
        componentSum += stat.sum;
        componentSquares += stat.squares;
        if (stat.count > 0) {
          globalMin = globalMin === null ? stat.min : Math.min(globalMin, stat.min);
          globalMax = globalMax === null ? stat.max : Math.max(globalMax, stat.max);
        }
      });
      const aggregateMagnitude = componentCount > 0 ? Math.sqrt(componentSquares) : null;
      const aggregateMean = componentCount > 0 ? componentSum / componentCount : null;
      const metaList = document.createElement("dl");
      metaList.className = "row g-2 mb-0";
      const pushMeta = (label, value) => {
        const dt = document.createElement("dt");
        dt.className = "col-sm-4";
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.className = "col-sm-8";
        dd.textContent = value;
        metaList.appendChild(dt);
        metaList.appendChild(dd);
      };
      if (embeddingData.backend || metadata.backend) {
        pushMeta("Backend", String(embeddingData.backend || metadata.backend));
      }
      if (embeddingData.model || metadata.model) {
        pushMeta("Mod\xE8le", String(embeddingData.model || metadata.model));
      }
      if (dims) {
        pushMeta("Dimensions", String(dims));
      }
      pushMeta("Vecteurs", `${vectors.length}`);
      if (componentCount) {
        pushMeta("Composantes", `${componentCount}`);
      }
      pushMeta(
        "Normalisation",
        Boolean(
          typeof embeddingData.normalised !== "undefined" ? embeddingData.normalised : metadata.normalised
        ) ? "Oui" : "Non"
      );
      pushMeta("Magnitude moyenne", formatNumeric(avgMagnitude));
      pushMeta("Magnitude agr\xE9g\xE9e", formatNumeric(aggregateMagnitude));
      pushMeta("Moyenne globale", formatNumeric(aggregateMean));
      pushMeta("Minimum global", formatNumeric(globalMin));
      pushMeta("Maximum global", formatNumeric(globalMax));
      cardBody.appendChild(metaList);
      const table = createVectorStatsTable(stats);
      table.classList.add("mt-3");
      cardBody.appendChild(table);
      const detailsWrapper = document.createElement("details");
      detailsWrapper.className = "mt-3";
      const summary = document.createElement("summary");
      summary.textContent = "Afficher le JSON brut";
      detailsWrapper.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "mt-2 mb-0 overflow-auto";
      pre.style.maxHeight = "240px";
      pre.textContent = JSON.stringify(vectors, null, 2);
      detailsWrapper.appendChild(pre);
      cardBody.appendChild(detailsWrapper);
      bubble.appendChild(details);
    }
    function appendMessage(role, text, options = {}) {
      const {
        timestamp,
        variant,
        metaSuffix,
        allowMarkdown = true,
        messageId,
        register = true,
        metadata,
        embeddingData
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
      if (role === "assistant" && embeddingData) {
        attachEmbeddingDetails(row, embeddingData, metadata || {});
      }
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
      if (elements.modeSelect) {
        elements.modeSelect.addEventListener("change", (event) => {
          const value = event.target.value || "";
          const nextMode = normaliseMode(value);
          setMode(nextMode);
          emit("mode-change", { mode: nextMode });
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
      setMode(state.mode, { skipStatus: true });
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
      setMode,
      renderEmbeddingDetails(row, embeddingData, metadata = {}) {
        attachEmbeddingDetails(row, embeddingData, metadata);
      },
      set diagnostics(value) {
        Object.assign(diagnostics, value);
      },
      get diagnostics() {
        return { ...diagnostics };
      },
      get mode() {
        return state.mode;
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
  var PLACEHOLDER_TOKENS = /* @__PURE__ */ new Set([
    "YOUR.JWT.HERE",
    "REPLACE_ME",
    "REPLACE_THIS",
    "INSERT_TOKEN_HERE"
  ]);
  function isValidJwtToken(t) {
    if (typeof t !== "string") return false;
    const s = t.trim();
    if (!s) return false;
    if (s === "null" || s === "undefined") return false;
    if (PLACEHOLDER_TOKENS.has(s)) return false;
    const parts = s.split(".");
    return parts.length === 3 && parts[0] && parts[1] && parts[2];
  }
  function seedTokenFromUrlOnce(storageKey) {
    try {
      if (typeof window === "undefined") return void 0;
      const u = new URL(window.location.href);
      const t = u.searchParams.get("jwt") || u.searchParams.get("token");
      if (!isValidJwtToken(t || "")) return void 0;
      if (hasLocalStorage()) {
        window.localStorage.setItem(storageKey, (t || "").trim());
      }
      u.searchParams.delete("jwt");
      u.searchParams.delete("token");
      window.history.replaceState({}, "", u.toString());
      return (t || "").trim();
    } catch (err) {
      return void 0;
    }
  }
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
        return isValidJwtToken(stored) ? stored : void 0;
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
      const seeded = seedTokenFromUrlOnce(storageKey);
      if (seeded) {
        fallbackToken = seeded;
        return seeded;
      }

      const stored = readStoredToken();
      if (stored) {
        return stored;
      }
      if (fallbackToken && isValidJwtToken(fallbackToken)) {
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
    async function postEmbed(text, options = {}) {
      if (!config.embedServiceUrl) {
        throw new Error(
          "Service d'embedding indisponible: aucune URL configur\xE9e."
        );
      }
      const payload = {
        inputs: [text]
      };
      if (Object.prototype.hasOwnProperty.call(options, "normalise")) {
        payload.normalise = Boolean(options.normalise);
      } else {
        payload.normalise = false;
      }
      const resp = await authorisedFetch(config.embedServiceUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
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
      postEmbed,
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
    const platform = typeof navigator !== "undefined" && navigator.platform ? navigator.platform.toLowerCase() : "";
    const maxTouchPoints = typeof navigator !== "undefined" && typeof navigator.maxTouchPoints === "number" ? navigator.maxTouchPoints : 0;
    const isAppleMobile = /iphone|ipad|ipod/.test(userAgent) || platform === "macintel" && maxTouchPoints > 1;
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
      this.mode = this.ui.mode || "chat";
      this.auth = createAuthService(this.config);
      this.http = createHttpService({ config: this.config, auth: this.auth });
      this.embeddingAvailable = Boolean(this.config.embedServiceUrl);
      this.embedOptionLabel = null;
      this.configureModeAvailability();
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
            "Saisissez un message avant d\u2019envoyer.",
            "warning"
          );
          this.ui.scheduleComposerIdle(4e3);
          return;
        }
        if (requestMode === "embed" && !this.embeddingAvailable) {
          this.ui.setMode("chat", { forceStatus: true });
          if (this.elements.modeSelect) {
            this.elements.modeSelect.value = "chat";
          }
          this.mode = "chat";
          this.ui.setComposerStatus(
            "Service d'embedding indisponible. Mode Chat r\xE9tabli.",
            "warning"
          );
          this.ui.scheduleComposerIdle(5e3);
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
        if (requestMode === "embed") {
          this.ui.setComposerStatus("Calcul de l'embedding\u2026", "info");
        } else {
          this.ui.setComposerStatus("Message envoy\xE9\u2026", "info");
        }
        this.ui.scheduleComposerIdle(4e3);
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
            this.ui.setComposerStatus("Vecteur g\xE9n\xE9r\xE9.", "success");
            this.ui.scheduleComposerIdle(4e3);
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
            metadata: { stage: "submit", mode: requestMode }
          });
          if (requestMode === "embed") {
            this.ui.setComposerStatus(
              "G\xE9n\xE9ration d'embedding impossible. V\xE9rifiez la connexion.",
              "danger"
            );
          } else {
            this.ui.setComposerStatus(
              "Envoi impossible. V\xE9rifiez la connexion.",
              "danger"
            );
          }
          this.ui.scheduleComposerIdle(6e3);
        }
      });
      this.ui.on("mode-change", ({ mode }) => {
        const requestedMode = mode === "embed" ? "embed" : "chat";
        if (requestedMode === "embed" && !this.embeddingAvailable) {
          this.configureModeAvailability();
          this.ui.setComposerStatus(
            "Service d'embedding indisponible. Mode Chat r\xE9tabli.",
            "warning"
          );
          this.ui.scheduleComposerIdle(5e3);
          return;
        }
        if (this.mode === requestedMode) {
          return;
        }
        this.mode = requestedMode;
        this.ui.setMode(requestedMode);
        if (requestedMode === "embed") {
          this.ui.setComposerStatus(
            "Mode Embedding activ\xE9. Les requ\xEAtes renvoient des vecteurs.",
            "info"
          );
          this.ui.scheduleComposerIdle(5e3);
        } else {
          this.ui.setComposerStatus(
            "Mode Chat activ\xE9. Les r\xE9ponses seront g\xE9n\xE9r\xE9es par le LLM.",
            "info"
          );
          this.ui.scheduleComposerIdle(4e3);
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
        this.ui.setVoiceStatus(
          "En \xE9coute\u2026 Parlez lorsque vous \xEAtes pr\xEAt.",
          "info"
        );
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
        this.ui.setVoiceStatus("Message dict\xE9. V\xE9rifiez avant l'envoi.", "info");
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
    formatEmbeddingResponse(result) {
      var _a, _b;
      const safeInline = (value) => {
        if (value === null || typeof value === "undefined" || value === "") {
          return "\u2014";
        }
        return `\`${String(value).replace(/`/g, "\\`")}\``;
      };
      const vectors = Array.isArray(result == null ? void 0 : result.vectors) ? result.vectors : [];
      const dimsCandidate = typeof (result == null ? void 0 : result.dims) === "number" ? result.dims : Number(result == null ? void 0 : result.dims);
      const dims = Number.isFinite(dimsCandidate) ? Number(dimsCandidate) : Array.isArray(vectors[0]) ? vectors[0].length : 0;
      const countCandidate = typeof (result == null ? void 0 : result.count) === "number" ? result.count : Number(result == null ? void 0 : result.count);
      const count = Number.isFinite(countCandidate) ? Number(countCandidate) : vectors.length;
      const normalised = Boolean(result == null ? void 0 : result.normalised);
      const summaryLines = [
        `- **Backend :** ${safeInline((_a = result == null ? void 0 : result.backend) != null ? _a : "inconnu")}`,
        `- **Mod\xE8le :** ${safeInline((_b = result == null ? void 0 : result.model) != null ? _b : "inconnu")}`,
        `- **Dimensions :** ${dims || 0}`,
        `- **Vecteurs g\xE9n\xE9r\xE9s :** ${count}`,
        `- **Normalisation appliqu\xE9e :** ${normalised ? "Oui" : "Non"}`
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
          `${previewJson}${vector.length > previewLength ? "\n// \u2026" : ""}`,
          "```"
        ].join("\n");
        if (vector.length > previewLength) {
          const fullVector = vector.map((value) => {
            const numeric = typeof value === "number" ? value : Number(value);
            return Number.isFinite(numeric) ? numeric : value;
          });
          section += `

<details><summary>Vecteur complet ${index + 1}</summary>

\`\`\`json
${JSON.stringify(
            fullVector,
            null,
            2
          )}
\`\`\`

</details>`;
        }
        vectorSections.push(section);
      });
      const sections = ["### R\xE9sultat d'embedding", summaryLines.join("\n")];
      if (vectorSections.length > 0) {
        sections.push(vectorSections.join("\n\n"));
      } else {
        sections.push("**Aucune composante d'embedding n'a \xE9t\xE9 renvoy\xE9e.**");
      }
      return sections.join("\n\n");
    }
    presentEmbeddingResult(result) {
      const vectors = Array.isArray(result == null ? void 0 : result.vectors) ? result.vectors : [];
      const dimsCandidate = typeof (result == null ? void 0 : result.dims) === "number" ? result.dims : Number(result == null ? void 0 : result.dims);
      const dims = Number.isFinite(dimsCandidate) ? Number(dimsCandidate) : Array.isArray(vectors[0]) ? vectors[0].length : 0;
      const countCandidate = typeof (result == null ? void 0 : result.count) === "number" ? result.count : Number(result == null ? void 0 : result.count);
      const count = Number.isFinite(countCandidate) ? Number(countCandidate) : vectors.length;
      const normalised = Boolean(result == null ? void 0 : result.normalised);
      const metaBits = ["Embedding"];
      if (dims) {
        metaBits.push(`${dims} dims`);
      }
      if (count) {
        metaBits.push(`${count} vecteur${count > 1 ? "s" : ""}`);
      }
      if (normalised) {
        metaBits.push("Normalis\xE9");
      }
      const message = this.formatEmbeddingResponse(result);
      this.ui.appendMessage("assistant", message, {
        timestamp: nowISO(),
        metaSuffix: metaBits.join(" \u2022 "),
        metadata: {
          mode: "embed",
          dims,
          backend: typeof (result == null ? void 0 : result.backend) === "string" && result.backend ? result.backend : null,
          model: typeof (result == null ? void 0 : result.model) === "string" && result.model ? result.model : null,
          count,
          normalised
        },
        embeddingData: {
          backend: typeof (result == null ? void 0 : result.backend) === "string" && result.backend ? result.backend : null,
          model: typeof (result == null ? void 0 : result.model) === "string" && result.model ? result.model : null,
          dims,
          count,
          normalised,
          vectors,
          raw: result
        }
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
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsic3JjL2NvbmZpZy5qcyIsICJzcmMvdXRpbHMvdGltZS5qcyIsICJzcmMvc3RhdGUvdGltZWxpbmVTdG9yZS5qcyIsICJzcmMvdXRpbHMvZW1pdHRlci5qcyIsICJzcmMvdXRpbHMvZG9tLmpzIiwgInNyYy9zZXJ2aWNlcy9tYXJrZG93bi5qcyIsICJzcmMvdXRpbHMvZXJyb3JVdGlscy5qcyIsICJzcmMvdWkvY2hhdFVpLmpzIiwgInNyYy9zZXJ2aWNlcy9hdXRoLmpzIiwgInNyYy9zZXJ2aWNlcy9odHRwLmpzIiwgInNyYy9zZXJ2aWNlcy9leHBvcnRlci5qcyIsICJzcmMvc2VydmljZXMvc29ja2V0LmpzIiwgInNyYy9zZXJ2aWNlcy9zdWdnZXN0aW9ucy5qcyIsICJzcmMvc2VydmljZXMvc3BlZWNoLmpzIiwgInNyYy9hcHAuanMiLCAic3JjL2luZGV4LmpzIl0sCiAgInNvdXJjZXNDb250ZW50IjogWyJleHBvcnQgZnVuY3Rpb24gcmVzb2x2ZUNvbmZpZyhyYXcgPSB7fSkge1xuICBjb25zdCBjb25maWcgPSB7IC4uLnJhdyB9O1xuICBjb25zdCBjYW5kaWRhdGUgPSBjb25maWcuZmFzdGFwaVVybCB8fCB3aW5kb3cubG9jYXRpb24ub3JpZ2luO1xuICB0cnkge1xuICAgIGNvbmZpZy5iYXNlVXJsID0gbmV3IFVSTChjYW5kaWRhdGUpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLmVycm9yKFwiSW52YWxpZCBGQVNUQVBJIFVSTFwiLCBlcnIsIGNhbmRpZGF0ZSk7XG4gICAgY29uZmlnLmJhc2VVcmwgPSBuZXcgVVJMKHdpbmRvdy5sb2NhdGlvbi5vcmlnaW4pO1xuICB9XG4gIGNvbnN0IGVtYmVkQ2FuZGlkYXRlID1cbiAgICB0eXBlb2YgY29uZmlnLmVtYmVkU2VydmljZVVybCA9PT0gXCJzdHJpbmdcIlxuICAgICAgPyBjb25maWcuZW1iZWRTZXJ2aWNlVXJsLnRyaW0oKVxuICAgICAgOiBcIlwiO1xuICBpZiAoZW1iZWRDYW5kaWRhdGUpIHtcbiAgICB0cnkge1xuICAgICAgY29uc3QgdXJsID0gbmV3IFVSTChlbWJlZENhbmRpZGF0ZSk7XG4gICAgICBpZiAodXJsLnByb3RvY29sID09PSBcImh0dHA6XCIgfHwgdXJsLnByb3RvY29sID09PSBcImh0dHBzOlwiKSB7XG4gICAgICAgIGNvbmZpZy5lbWJlZFNlcnZpY2VVcmwgPSB1cmwudG9TdHJpbmcoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcIlVuc3VwcG9ydGVkIGVtYmVkZGluZyBzZXJ2aWNlIHByb3RvY29sXCIsIHVybC5wcm90b2NvbCk7XG4gICAgICAgIGNvbmZpZy5lbWJlZFNlcnZpY2VVcmwgPSBudWxsO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiSW52YWxpZCBlbWJlZGRpbmcgc2VydmljZSBVUkxcIiwgZXJyLCBlbWJlZENhbmRpZGF0ZSk7XG4gICAgICBjb25maWcuZW1iZWRTZXJ2aWNlVXJsID0gbnVsbDtcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgY29uZmlnLmVtYmVkU2VydmljZVVybCA9IG51bGw7XG4gIH1cbiAgcmV0dXJuIGNvbmZpZztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFwaVVybChjb25maWcsIHBhdGgpIHtcbiAgcmV0dXJuIG5ldyBVUkwocGF0aCwgY29uZmlnLmJhc2VVcmwpLnRvU3RyaW5nKCk7XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIG5vd0lTTygpIHtcbiAgcmV0dXJuIG5ldyBEYXRlKCkudG9JU09TdHJpbmcoKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGZvcm1hdFRpbWVzdGFtcCh0cykge1xuICBpZiAoIXRzKSByZXR1cm4gXCJcIjtcbiAgdHJ5IHtcbiAgICByZXR1cm4gbmV3IERhdGUodHMpLnRvTG9jYWxlU3RyaW5nKFwiZnItQ0FcIik7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIHJldHVybiBTdHJpbmcodHMpO1xuICB9XG59XG4iLCAiaW1wb3J0IHsgbm93SVNPIH0gZnJvbSBcIi4uL3V0aWxzL3RpbWUuanNcIjtcblxuZnVuY3Rpb24gbWFrZU1lc3NhZ2VJZCgpIHtcbiAgcmV0dXJuIGBtc2ctJHtEYXRlLm5vdygpLnRvU3RyaW5nKDM2KX0tJHtNYXRoLnJhbmRvbSgpLnRvU3RyaW5nKDM2KS5zbGljZSgyLCA4KX1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVGltZWxpbmVTdG9yZSgpIHtcbiAgY29uc3Qgb3JkZXIgPSBbXTtcbiAgY29uc3QgbWFwID0gbmV3IE1hcCgpO1xuXG4gIGZ1bmN0aW9uIHJlZ2lzdGVyKHtcbiAgICBpZCxcbiAgICByb2xlLFxuICAgIHRleHQgPSBcIlwiLFxuICAgIHRpbWVzdGFtcCA9IG5vd0lTTygpLFxuICAgIHJvdyxcbiAgICBtZXRhZGF0YSA9IHt9LFxuICB9KSB7XG4gICAgY29uc3QgbWVzc2FnZUlkID0gaWQgfHwgbWFrZU1lc3NhZ2VJZCgpO1xuICAgIGlmICghbWFwLmhhcyhtZXNzYWdlSWQpKSB7XG4gICAgICBvcmRlci5wdXNoKG1lc3NhZ2VJZCk7XG4gICAgfVxuICAgIG1hcC5zZXQobWVzc2FnZUlkLCB7XG4gICAgICBpZDogbWVzc2FnZUlkLFxuICAgICAgcm9sZSxcbiAgICAgIHRleHQsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICByb3csXG4gICAgICBtZXRhZGF0YTogeyAuLi5tZXRhZGF0YSB9LFxuICAgIH0pO1xuICAgIGlmIChyb3cpIHtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IG1lc3NhZ2VJZDtcbiAgICAgIHJvdy5kYXRhc2V0LnJvbGUgPSByb2xlO1xuICAgICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IHRleHQ7XG4gICAgICByb3cuZGF0YXNldC50aW1lc3RhbXAgPSB0aW1lc3RhbXA7XG4gICAgfVxuICAgIHJldHVybiBtZXNzYWdlSWQ7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGUoaWQsIHBhdGNoID0ge30pIHtcbiAgICBpZiAoIW1hcC5oYXMoaWQpKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgY29uc3QgZW50cnkgPSBtYXAuZ2V0KGlkKTtcbiAgICBjb25zdCBuZXh0ID0geyAuLi5lbnRyeSwgLi4ucGF0Y2ggfTtcbiAgICBpZiAocGF0Y2ggJiYgdHlwZW9mIHBhdGNoLm1ldGFkYXRhID09PSBcIm9iamVjdFwiICYmIHBhdGNoLm1ldGFkYXRhICE9PSBudWxsKSB7XG4gICAgICBjb25zdCBtZXJnZWQgPSB7IC4uLmVudHJ5Lm1ldGFkYXRhIH07XG4gICAgICBPYmplY3QuZW50cmllcyhwYXRjaC5tZXRhZGF0YSkuZm9yRWFjaCgoW2tleSwgdmFsdWVdKSA9PiB7XG4gICAgICAgIGlmICh2YWx1ZSA9PT0gdW5kZWZpbmVkIHx8IHZhbHVlID09PSBudWxsKSB7XG4gICAgICAgICAgZGVsZXRlIG1lcmdlZFtrZXldO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIG1lcmdlZFtrZXldID0gdmFsdWU7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgICAgbmV4dC5tZXRhZGF0YSA9IG1lcmdlZDtcbiAgICB9XG4gICAgbWFwLnNldChpZCwgbmV4dCk7XG4gICAgY29uc3QgeyByb3cgfSA9IG5leHQ7XG4gICAgaWYgKHJvdyAmJiByb3cuaXNDb25uZWN0ZWQpIHtcbiAgICAgIGlmIChuZXh0LnRleHQgIT09IGVudHJ5LnRleHQpIHtcbiAgICAgICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IG5leHQudGV4dCB8fCBcIlwiO1xuICAgICAgfVxuICAgICAgaWYgKG5leHQudGltZXN0YW1wICE9PSBlbnRyeS50aW1lc3RhbXApIHtcbiAgICAgICAgcm93LmRhdGFzZXQudGltZXN0YW1wID0gbmV4dC50aW1lc3RhbXAgfHwgXCJcIjtcbiAgICAgIH1cbiAgICAgIGlmIChuZXh0LnJvbGUgJiYgbmV4dC5yb2xlICE9PSBlbnRyeS5yb2xlKSB7XG4gICAgICAgIHJvdy5kYXRhc2V0LnJvbGUgPSBuZXh0LnJvbGU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBuZXh0O1xuICB9XG5cbiAgZnVuY3Rpb24gY29sbGVjdCgpIHtcbiAgICByZXR1cm4gb3JkZXJcbiAgICAgIC5tYXAoKGlkKSA9PiB7XG4gICAgICAgIGNvbnN0IGVudHJ5ID0gbWFwLmdldChpZCk7XG4gICAgICAgIGlmICghZW50cnkpIHtcbiAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIHJvbGU6IGVudHJ5LnJvbGUsXG4gICAgICAgICAgdGV4dDogZW50cnkudGV4dCxcbiAgICAgICAgICB0aW1lc3RhbXA6IGVudHJ5LnRpbWVzdGFtcCxcbiAgICAgICAgICAuLi4oZW50cnkubWV0YWRhdGEgJiZcbiAgICAgICAgICAgIE9iamVjdC5rZXlzKGVudHJ5Lm1ldGFkYXRhKS5sZW5ndGggPiAwICYmIHtcbiAgICAgICAgICAgICAgbWV0YWRhdGE6IHsgLi4uZW50cnkubWV0YWRhdGEgfSxcbiAgICAgICAgICAgIH0pLFxuICAgICAgICB9O1xuICAgICAgfSlcbiAgICAgIC5maWx0ZXIoQm9vbGVhbik7XG4gIH1cblxuICBmdW5jdGlvbiBjbGVhcigpIHtcbiAgICBvcmRlci5sZW5ndGggPSAwO1xuICAgIG1hcC5jbGVhcigpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICByZWdpc3RlcixcbiAgICB1cGRhdGUsXG4gICAgY29sbGVjdCxcbiAgICBjbGVhcixcbiAgICBvcmRlcixcbiAgICBtYXAsXG4gICAgbWFrZU1lc3NhZ2VJZCxcbiAgfTtcbn1cbiIsICJleHBvcnQgZnVuY3Rpb24gY3JlYXRlRW1pdHRlcigpIHtcbiAgY29uc3QgbGlzdGVuZXJzID0gbmV3IE1hcCgpO1xuXG4gIGZ1bmN0aW9uIG9uKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkge1xuICAgICAgbGlzdGVuZXJzLnNldChldmVudCwgbmV3IFNldCgpKTtcbiAgICB9XG4gICAgbGlzdGVuZXJzLmdldChldmVudCkuYWRkKGhhbmRsZXIpO1xuICAgIHJldHVybiAoKSA9PiBvZmYoZXZlbnQsIGhhbmRsZXIpO1xuICB9XG5cbiAgZnVuY3Rpb24gb2ZmKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkgcmV0dXJuO1xuICAgIGNvbnN0IGJ1Y2tldCA9IGxpc3RlbmVycy5nZXQoZXZlbnQpO1xuICAgIGJ1Y2tldC5kZWxldGUoaGFuZGxlcik7XG4gICAgaWYgKGJ1Y2tldC5zaXplID09PSAwKSB7XG4gICAgICBsaXN0ZW5lcnMuZGVsZXRlKGV2ZW50KTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBlbWl0KGV2ZW50LCBwYXlsb2FkKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkgcmV0dXJuO1xuICAgIGxpc3RlbmVycy5nZXQoZXZlbnQpLmZvckVhY2goKGhhbmRsZXIpID0+IHtcbiAgICAgIHRyeSB7XG4gICAgICAgIGhhbmRsZXIocGF5bG9hZCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIkVtaXR0ZXIgaGFuZGxlciBlcnJvclwiLCBlcnIpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgcmV0dXJuIHsgb24sIG9mZiwgZW1pdCB9O1xufVxuIiwgImV4cG9ydCBmdW5jdGlvbiBlc2NhcGVIVE1MKHN0cikge1xuICByZXR1cm4gU3RyaW5nKHN0cikucmVwbGFjZShcbiAgICAvWyY8PlwiJ10vZyxcbiAgICAoY2gpID0+XG4gICAgICAoe1xuICAgICAgICBcIiZcIjogXCImYW1wO1wiLFxuICAgICAgICBcIjxcIjogXCImbHQ7XCIsXG4gICAgICAgIFwiPlwiOiBcIiZndDtcIixcbiAgICAgICAgJ1wiJzogXCImcXVvdDtcIixcbiAgICAgICAgXCInXCI6IFwiJiMzOTtcIixcbiAgICAgIH0pW2NoXSxcbiAgKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGh0bWxUb1RleHQoaHRtbCkge1xuICBjb25zdCBwYXJzZXIgPSBuZXcgRE9NUGFyc2VyKCk7XG4gIGNvbnN0IGRvYyA9IHBhcnNlci5wYXJzZUZyb21TdHJpbmcoaHRtbCwgXCJ0ZXh0L2h0bWxcIik7XG4gIHJldHVybiBkb2MuYm9keS50ZXh0Q29udGVudCB8fCBcIlwiO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZXh0cmFjdEJ1YmJsZVRleHQoYnViYmxlKSB7XG4gIGNvbnN0IGNsb25lID0gYnViYmxlLmNsb25lTm9kZSh0cnVlKTtcbiAgY2xvbmVcbiAgICAucXVlcnlTZWxlY3RvckFsbChcIi5jb3B5LWJ0biwgLmNoYXQtbWV0YVwiKVxuICAgIC5mb3JFYWNoKChub2RlKSA9PiBub2RlLnJlbW92ZSgpKTtcbiAgcmV0dXJuIGNsb25lLnRleHRDb250ZW50LnRyaW0oKTtcbn1cbiIsICJpbXBvcnQgeyBlc2NhcGVIVE1MIH0gZnJvbSBcIi4uL3V0aWxzL2RvbS5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gcmVuZGVyTWFya2Rvd24odGV4dCkge1xuICBpZiAodGV4dCA9PSBudWxsKSB7XG4gICAgcmV0dXJuIFwiXCI7XG4gIH1cbiAgY29uc3QgdmFsdWUgPSBTdHJpbmcodGV4dCk7XG4gIGNvbnN0IGZhbGxiYWNrID0gKCkgPT4ge1xuICAgIGNvbnN0IGVzY2FwZWQgPSBlc2NhcGVIVE1MKHZhbHVlKTtcbiAgICByZXR1cm4gZXNjYXBlZC5yZXBsYWNlKC9cXG4vZywgXCI8YnI+XCIpO1xuICB9O1xuICB0cnkge1xuICAgIGlmICh3aW5kb3cubWFya2VkICYmIHR5cGVvZiB3aW5kb3cubWFya2VkLnBhcnNlID09PSBcImZ1bmN0aW9uXCIpIHtcbiAgICAgIGNvbnN0IHJlbmRlcmVkID0gd2luZG93Lm1hcmtlZC5wYXJzZSh2YWx1ZSk7XG4gICAgICBpZiAod2luZG93LkRPTVB1cmlmeSAmJiB0eXBlb2Ygd2luZG93LkRPTVB1cmlmeS5zYW5pdGl6ZSA9PT0gXCJmdW5jdGlvblwiKSB7XG4gICAgICAgIHJldHVybiB3aW5kb3cuRE9NUHVyaWZ5LnNhbml0aXplKHJlbmRlcmVkLCB7XG4gICAgICAgICAgQUxMT1dfVU5LTk9XTl9QUk9UT0NPTFM6IGZhbHNlLFxuICAgICAgICAgIFVTRV9QUk9GSUxFUzogeyBodG1sOiB0cnVlIH0sXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgICAgLy8gRmFsbGJhY2s6IGVzY2FwZSByYXcgdGV4dCBhbmQgZG8gbWluaW1hbCBmb3JtYXR0aW5nIHRvIGF2b2lkIFhTU1xuICAgICAgY29uc3QgZXNjYXBlZCA9IGVzY2FwZUhUTUwodmFsdWUpO1xuICAgICAgcmV0dXJuIGVzY2FwZWQucmVwbGFjZSgvXFxuL2csIFwiPGJyPlwiKTtcbiAgICB9XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIGNvbnNvbGUud2FybihcIk1hcmtkb3duIHJlbmRlcmluZyBmYWlsZWRcIiwgZXJyKTtcbiAgfVxuICByZXR1cm4gZmFsbGJhY2soKTtcbn1cbiIsICJleHBvcnQgY29uc3QgREVGQVVMVF9FUlJPUl9NRVNTQUdFID1cbiAgXCJVbmUgZXJyZXVyIGluYXR0ZW5kdWUgZXN0IHN1cnZlbnVlLiBWZXVpbGxleiByXHUwMEU5ZXNzYXllci5cIjtcblxuY29uc3QgRVJST1JfUFJFRklYX1JFR0VYID0gL15cXHMqW1xcV19dKlxccyooZXJyZXVyfGVycm9yKS9pO1xuXG5leHBvcnQgZnVuY3Rpb24gbm9ybWFsaXNlRXJyb3JUZXh0KGVycm9yKSB7XG4gIGlmIChlcnJvciBpbnN0YW5jZW9mIEVycm9yKSB7XG4gICAgY29uc3QgbWVzc2FnZSA9IGVycm9yLm1lc3NhZ2UgPyBlcnJvci5tZXNzYWdlLnRyaW0oKSA6IFwiXCI7XG4gICAgaWYgKG1lc3NhZ2UpIHtcbiAgICAgIHJldHVybiBtZXNzYWdlO1xuICAgIH1cbiAgfVxuXG4gIGlmICh0eXBlb2YgZXJyb3IgPT09IFwic3RyaW5nXCIpIHtcbiAgICBjb25zdCB0cmltbWVkID0gZXJyb3IudHJpbSgpO1xuICAgIHJldHVybiB0cmltbWVkIHx8IERFRkFVTFRfRVJST1JfTUVTU0FHRTtcbiAgfVxuXG4gIGlmIChlcnJvciAmJiB0eXBlb2YgZXJyb3IgPT09IFwib2JqZWN0XCIpIHtcbiAgICBjb25zdCBjYW5kaWRhdGUgPVxuICAgICAgKHR5cGVvZiBlcnJvci5tZXNzYWdlID09PSBcInN0cmluZ1wiICYmIGVycm9yLm1lc3NhZ2UudHJpbSgpKSB8fFxuICAgICAgKHR5cGVvZiBlcnJvci5lcnJvciA9PT0gXCJzdHJpbmdcIiAmJiBlcnJvci5lcnJvci50cmltKCkpO1xuICAgIGlmIChjYW5kaWRhdGUpIHtcbiAgICAgIHJldHVybiBjYW5kaWRhdGU7XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHNlcmlhbGlzZWQgPSBKU09OLnN0cmluZ2lmeShlcnJvcik7XG4gICAgICBpZiAoc2VyaWFsaXNlZCAmJiBzZXJpYWxpc2VkICE9PSBcInt9XCIpIHtcbiAgICAgICAgcmV0dXJuIHNlcmlhbGlzZWQ7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoc2VyaWFsaXNlRXJyKSB7XG4gICAgICBjb25zb2xlLmRlYnVnKFwiVW5hYmxlIHRvIHNlcmlhbGlzZSBlcnJvciBwYXlsb2FkXCIsIHNlcmlhbGlzZUVycik7XG4gICAgfVxuICB9XG5cbiAgaWYgKHR5cGVvZiBlcnJvciA9PT0gXCJ1bmRlZmluZWRcIiB8fCBlcnJvciA9PT0gbnVsbCkge1xuICAgIHJldHVybiBERUZBVUxUX0VSUk9SX01FU1NBR0U7XG4gIH1cblxuICBjb25zdCBmYWxsYmFjayA9IFN0cmluZyhlcnJvcikudHJpbSgpO1xuICByZXR1cm4gZmFsbGJhY2sgfHwgREVGQVVMVF9FUlJPUl9NRVNTQUdFO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcmVzb2x2ZUVycm9yVGV4dChlcnJvck9yVGV4dCkge1xuICBpZiAodHlwZW9mIGVycm9yT3JUZXh0ID09PSBcInN0cmluZ1wiKSB7XG4gICAgY29uc3QgdHJpbW1lZCA9IGVycm9yT3JUZXh0LnRyaW0oKTtcbiAgICByZXR1cm4gdHJpbW1lZCB8fCBERUZBVUxUX0VSUk9SX01FU1NBR0U7XG4gIH1cbiAgcmV0dXJuIG5vcm1hbGlzZUVycm9yVGV4dChlcnJvck9yVGV4dCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRXJyb3JCdWJibGVUZXh0KGVycm9yT3JUZXh0LCBvcHRpb25zID0ge30pIHtcbiAgY29uc3QgeyBwcmVmaXggfSA9IG9wdGlvbnM7XG4gIGNvbnN0IHRleHQgPSByZXNvbHZlRXJyb3JUZXh0KGVycm9yT3JUZXh0KTtcbiAgY29uc3QgYmFzZVByZWZpeCA9XG4gICAgb3B0aW9ucy5wcmVmaXggPT09IG51bGxcbiAgICAgID8gXCJcIlxuICAgICAgOiB0eXBlb2YgcHJlZml4ID09PSBcInN0cmluZ1wiXG4gICAgICAgID8gcHJlZml4XG4gICAgICAgIDogXCJFcnJldXIgOiBcIjtcbiAgY29uc3QgdHJpbW1lZFByZWZpeCA9IGJhc2VQcmVmaXgudHJpbSgpLnRvTG93ZXJDYXNlKCk7XG4gIGNvbnN0IHNob3VsZFByZWZpeCA9XG4gICAgQm9vbGVhbihiYXNlUHJlZml4KSAmJlxuICAgICFFUlJPUl9QUkVGSVhfUkVHRVgudGVzdCh0ZXh0KSAmJlxuICAgICEodHJpbW1lZFByZWZpeCAmJiB0ZXh0LnRvTG93ZXJDYXNlKCkuc3RhcnRzV2l0aCh0cmltbWVkUHJlZml4KSk7XG4gIGNvbnN0IGJ1YmJsZVRleHQgPSBzaG91bGRQcmVmaXggPyBgJHtiYXNlUHJlZml4fSR7dGV4dH1gIDogdGV4dDtcbiAgcmV0dXJuIHsgdGV4dCwgYnViYmxlVGV4dCB9O1xufVxuIiwgImltcG9ydCB7IGNyZWF0ZUVtaXR0ZXIgfSBmcm9tIFwiLi4vdXRpbHMvZW1pdHRlci5qc1wiO1xuaW1wb3J0IHsgaHRtbFRvVGV4dCwgZXh0cmFjdEJ1YmJsZVRleHQsIGVzY2FwZUhUTUwgfSBmcm9tIFwiLi4vdXRpbHMvZG9tLmpzXCI7XG5pbXBvcnQgeyByZW5kZXJNYXJrZG93biB9IGZyb20gXCIuLi9zZXJ2aWNlcy9tYXJrZG93bi5qc1wiO1xuaW1wb3J0IHsgZm9ybWF0VGltZXN0YW1wLCBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuaW1wb3J0IHsgY29tcHV0ZUVycm9yQnViYmxlVGV4dCB9IGZyb20gXCIuLi91dGlscy9lcnJvclV0aWxzLmpzXCI7XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVDaGF0VWkoeyBlbGVtZW50cywgdGltZWxpbmVTdG9yZSB9KSB7XG4gIGNvbnN0IGVtaXR0ZXIgPSBjcmVhdGVFbWl0dGVyKCk7XG5cbiAgY29uc3Qgc2VuZExhYmVsRWxlbWVudCA9XG4gICAgZWxlbWVudHMuc2VuZD8ucXVlcnlTZWxlY3RvcihcIltkYXRhLXJvbGU9J3NlbmQtbGFiZWwnXVwiKSB8fCBudWxsO1xuICBjb25zdCBzZW5kU3Bpbm5lckVsZW1lbnQgPVxuICAgIGVsZW1lbnRzLnNlbmQ/LnF1ZXJ5U2VsZWN0b3IoXCJbZGF0YS1yb2xlPSdzZW5kLXNwaW5uZXInXVwiKSB8fCBudWxsO1xuICBjb25zdCBzZW5kSWRsZU1hcmt1cCA9IGVsZW1lbnRzLnNlbmQgPyBlbGVtZW50cy5zZW5kLmlubmVySFRNTCA6IFwiXCI7XG4gIGNvbnN0IHNlbmRJZGxlTGFiZWwgPVxuICAgIChlbGVtZW50cy5zZW5kICYmIGVsZW1lbnRzLnNlbmQuZ2V0QXR0cmlidXRlKFwiZGF0YS1pZGxlLWxhYmVsXCIpKSB8fFxuICAgIChzZW5kTGFiZWxFbGVtZW50ICYmIHNlbmRMYWJlbEVsZW1lbnQudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIChlbGVtZW50cy5zZW5kID8gZWxlbWVudHMuc2VuZC50ZXh0Q29udGVudC50cmltKCkgOiBcIkVudm95ZXJcIik7XG4gIGNvbnN0IHNlbmRCdXN5TGFiZWwgPVxuICAgIChlbGVtZW50cy5zZW5kICYmIGVsZW1lbnRzLnNlbmQuZ2V0QXR0cmlidXRlKFwiZGF0YS1idXN5LWxhYmVsXCIpKSB8fFxuICAgIFwiRW52b2lcdTIwMjZcIjtcbiAgY29uc3Qgc2VuZEJ1c3lBcmlhTGFiZWwgPVxuICAgIChlbGVtZW50cy5zZW5kICYmIGVsZW1lbnRzLnNlbmQuZ2V0QXR0cmlidXRlKFwiZGF0YS1idXN5LWFyaWEtbGFiZWxcIikpIHx8XG4gICAgc2VuZEJ1c3lMYWJlbDtcbiAgY29uc3Qgc2VuZEJ1c3lNYXJrdXAgPVxuICAgICc8c3BhbiBjbGFzcz1cInNwaW5uZXItYm9yZGVyIHNwaW5uZXItYm9yZGVyLXNtIG1lLTFcIiByb2xlPVwic3RhdHVzXCIgYXJpYS1oaWRkZW49XCJ0cnVlXCI+PC9zcGFuPkVudm9pXHUyMDI2JztcbiAgY29uc3QgU1VQUE9SVEVEX1RPTkVTID0gW1wibXV0ZWRcIiwgXCJpbmZvXCIsIFwic3VjY2Vzc1wiLCBcImRhbmdlclwiLCBcIndhcm5pbmdcIl07XG4gIGNvbnN0IGNvbXBvc2VyU3RhdHVzRGVmYXVsdCA9XG4gICAgKGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzICYmIGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzLnRleHRDb250ZW50LnRyaW0oKSkgfHxcbiAgICBcIkFwcHV5ZXogc3VyIEN0cmwrRW50clx1MDBFOWUgcG91ciBlbnZveWVyIHJhcGlkZW1lbnQuXCI7XG4gIGNvbnN0IGZpbHRlckhpbnREZWZhdWx0ID1cbiAgICAoZWxlbWVudHMuZmlsdGVySGludCAmJiBlbGVtZW50cy5maWx0ZXJIaW50LnRleHRDb250ZW50LnRyaW0oKSkgfHxcbiAgICBcIlV0aWxpc2V6IGxlIGZpbHRyZSBwb3VyIGxpbWl0ZXIgbCdoaXN0b3JpcXVlLiBBcHB1eWV6IHN1ciBcdTAwQzljaGFwIHBvdXIgZWZmYWNlci5cIjtcbiAgY29uc3Qgdm9pY2VTdGF0dXNEZWZhdWx0ID1cbiAgICAoZWxlbWVudHMudm9pY2VTdGF0dXMgJiYgZWxlbWVudHMudm9pY2VTdGF0dXMudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiVlx1MDBFOXJpZmljYXRpb24gZGVzIGNhcGFjaXRcdTAwRTlzIHZvY2FsZXNcdTIwMjZcIjtcbiAgY29uc3QgcHJvbXB0TWF4ID0gTnVtYmVyKGVsZW1lbnRzLnByb21wdD8uZ2V0QXR0cmlidXRlKFwibWF4bGVuZ3RoXCIpKSB8fCBudWxsO1xuICBjb25zdCBwcmVmZXJzUmVkdWNlZE1vdGlvbiA9XG4gICAgd2luZG93Lm1hdGNoTWVkaWEgJiZcbiAgICB3aW5kb3cubWF0Y2hNZWRpYShcIihwcmVmZXJzLXJlZHVjZWQtbW90aW9uOiByZWR1Y2UpXCIpLm1hdGNoZXM7XG4gIGNvbnN0IFNDUk9MTF9USFJFU0hPTEQgPSAxNDA7XG4gIGNvbnN0IFBST01QVF9NQVhfSEVJR0hUID0gMzIwO1xuICBjb25zdCBjb21wb3NlclN0YXR1c0VtYmVkZGluZyA9XG4gICAgKGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzICYmXG4gICAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy5nZXRBdHRyaWJ1dGUoXCJkYXRhLWVtYmVkLWxhYmVsXCIpKSB8fFxuICAgIFwiTW9kZSBFbWJlZGRpbmcgOiBnXHUwMEU5blx1MDBFOXJleiBkZXMgdmVjdGV1cnMgcG91ciB2b3MgdGV4dGVzLlwiO1xuICBjb25zdCBwcm9tcHRQbGFjZWhvbGRlckRlZmF1bHQgPVxuICAgIChlbGVtZW50cy5wcm9tcHQgJiYgZWxlbWVudHMucHJvbXB0LmdldEF0dHJpYnV0ZShcInBsYWNlaG9sZGVyXCIpKSB8fCBcIlwiO1xuICBjb25zdCBwcm9tcHRQbGFjZWhvbGRlckVtYmVkZGluZyA9XG4gICAgKGVsZW1lbnRzLnByb21wdCAmJlxuICAgICAgZWxlbWVudHMucHJvbXB0LmdldEF0dHJpYnV0ZShcImRhdGEtZW1iZWQtcGxhY2Vob2xkZXJcIikpIHx8XG4gICAgXCJFbnRyZXogbGUgdGV4dGUgXHUwMEUwIGVuY29kZXJcdTIwMjZcIjtcbiAgY29uc3QgcHJvbXB0QXJpYURlZmF1bHQgPVxuICAgIChlbGVtZW50cy5wcm9tcHQgJiYgZWxlbWVudHMucHJvbXB0LmdldEF0dHJpYnV0ZShcImFyaWEtbGFiZWxcIikpIHx8IFwiXCI7XG4gIGNvbnN0IHByb21wdEFyaWFFbWJlZGRpbmcgPVxuICAgIChlbGVtZW50cy5wcm9tcHQgJiZcbiAgICAgIGVsZW1lbnRzLnByb21wdC5nZXRBdHRyaWJ1dGUoXCJkYXRhLWVtYmVkLWFyaWEtbGFiZWxcIikpIHx8XG4gICAgXCJUZXh0ZSBcdTAwRTAgZW5jb2RlclwiO1xuICBjb25zdCBzZW5kQXJpYURlZmF1bHQgPVxuICAgIChlbGVtZW50cy5zZW5kICYmIGVsZW1lbnRzLnNlbmQuZ2V0QXR0cmlidXRlKFwiYXJpYS1sYWJlbFwiKSkgfHxcbiAgICBzZW5kSWRsZUxhYmVsO1xuICBjb25zdCBzZW5kQXJpYUVtYmVkZGluZyA9XG4gICAgKGVsZW1lbnRzLnNlbmQgJiYgZWxlbWVudHMuc2VuZC5nZXRBdHRyaWJ1dGUoXCJkYXRhLWVtYmVkLWFyaWEtbGFiZWxcIikpIHx8XG4gICAgXCJHXHUwMEU5blx1MDBFOXJlciB1biBlbWJlZGRpbmdcIjtcblxuICBjb25zdCBkaWFnbm9zdGljcyA9IHtcbiAgICBjb25uZWN0ZWRBdDogbnVsbCxcbiAgICBsYXN0TWVzc2FnZUF0OiBudWxsLFxuICAgIGxhdGVuY3lNczogbnVsbCxcbiAgfTtcblxuICBjb25zdCBzdGF0ZSA9IHtcbiAgICByZXNldFN0YXR1c1RpbWVyOiBudWxsLFxuICAgIGhpZGVTY3JvbGxUaW1lcjogbnVsbCxcbiAgICB2b2ljZVN0YXR1c1RpbWVyOiBudWxsLFxuICAgIG1vZGU6IFwiY2hhdFwiLFxuICAgIGFjdGl2ZUZpbHRlcjogXCJcIixcbiAgICBoaXN0b3J5Qm9vdHN0cmFwcGVkOiBlbGVtZW50cy50cmFuc2NyaXB0LmNoaWxkRWxlbWVudENvdW50ID4gMCxcbiAgICBib290c3RyYXBwaW5nOiBmYWxzZSxcbiAgICBzdHJlYW1Sb3c6IG51bGwsXG4gICAgc3RyZWFtQnVmOiBcIlwiLFxuICAgIHN0cmVhbU1lc3NhZ2VJZDogbnVsbCxcbiAgfTtcblxuICBjb25zdCBzdGF0dXNMYWJlbHMgPSB7XG4gICAgb2ZmbGluZTogXCJIb3JzIGxpZ25lXCIsXG4gICAgY29ubmVjdGluZzogXCJDb25uZXhpb25cdTIwMjZcIixcbiAgICBvbmxpbmU6IFwiRW4gbGlnbmVcIixcbiAgICBlcnJvcjogXCJFcnJldXJcIixcbiAgfTtcblxuICBmdW5jdGlvbiBvbihldmVudCwgaGFuZGxlcikge1xuICAgIHJldHVybiBlbWl0dGVyLm9uKGV2ZW50LCBoYW5kbGVyKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGVtaXQoZXZlbnQsIHBheWxvYWQpIHtcbiAgICBlbWl0dGVyLmVtaXQoZXZlbnQsIHBheWxvYWQpO1xuICB9XG5cbiAgZnVuY3Rpb24gbm9ybWFsaXNlTW9kZShtb2RlKSB7XG4gICAgcmV0dXJuIG1vZGUgPT09IFwiZW1iZWRcIiA/IFwiZW1iZWRcIiA6IFwiY2hhdFwiO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0QnVzeShidXN5KSB7XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5zZXRBdHRyaWJ1dGUoXCJhcmlhLWJ1c3lcIiwgYnVzeSA/IFwidHJ1ZVwiIDogXCJmYWxzZVwiKTtcbiAgICBpZiAoIWVsZW1lbnRzLnNlbmQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBlbGVtZW50cy5zZW5kLmRpc2FibGVkID0gQm9vbGVhbihidXN5KTtcbiAgICBlbGVtZW50cy5zZW5kLnNldEF0dHJpYnV0ZShcImFyaWEtYnVzeVwiLCBidXN5ID8gXCJ0cnVlXCIgOiBcImZhbHNlXCIpO1xuICAgIGVsZW1lbnRzLnNlbmQuY2xhc3NMaXN0LnRvZ2dsZShcImlzLWxvYWRpbmdcIiwgQm9vbGVhbihidXN5KSk7XG5cbiAgICBpZiAoc2VuZFNwaW5uZXJFbGVtZW50KSB7XG4gICAgICBpZiAoYnVzeSkge1xuICAgICAgICBzZW5kU3Bpbm5lckVsZW1lbnQuY2xhc3NMaXN0LnJlbW92ZShcImQtbm9uZVwiKTtcbiAgICAgICAgc2VuZFNwaW5uZXJFbGVtZW50LnNldEF0dHJpYnV0ZShcImFyaWEtaGlkZGVuXCIsIFwiZmFsc2VcIik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBzZW5kU3Bpbm5lckVsZW1lbnQuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICAgICAgc2VuZFNwaW5uZXJFbGVtZW50LnNldEF0dHJpYnV0ZShcImFyaWEtaGlkZGVuXCIsIFwidHJ1ZVwiKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAoc2VuZExhYmVsRWxlbWVudCkge1xuICAgICAgc2VuZExhYmVsRWxlbWVudC50ZXh0Q29udGVudCA9IGJ1c3kgPyBzZW5kQnVzeUxhYmVsIDogc2VuZElkbGVMYWJlbDtcbiAgICB9IGVsc2UgaWYgKGJ1c3kpIHtcbiAgICAgIGVsZW1lbnRzLnNlbmQuaW5uZXJIVE1MID0gc2VuZEJ1c3lNYXJrdXA7XG4gICAgfSBlbHNlIGlmIChzZW5kSWRsZU1hcmt1cCkge1xuICAgICAgZWxlbWVudHMuc2VuZC5pbm5lckhUTUwgPSBzZW5kSWRsZU1hcmt1cDtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMuc2VuZC50ZXh0Q29udGVudCA9IHNlbmRJZGxlTGFiZWw7XG4gICAgfVxuXG4gICAgaWYgKGJ1c3kpIHtcbiAgICAgIGlmIChzZW5kQnVzeUFyaWFMYWJlbCkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLnNldEF0dHJpYnV0ZShcImFyaWEtbGFiZWxcIiwgc2VuZEJ1c3lBcmlhTGFiZWwpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBhcmlhTGFiZWwgPSBzdGF0ZS5tb2RlID09PSBcImVtYmVkXCIgPyBzZW5kQXJpYUVtYmVkZGluZyA6IHNlbmRBcmlhRGVmYXVsdDtcbiAgICAgIGlmIChhcmlhTGFiZWwpIHtcbiAgICAgICAgZWxlbWVudHMuc2VuZC5zZXRBdHRyaWJ1dGUoXCJhcmlhLWxhYmVsXCIsIGFyaWFMYWJlbCk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5zZW5kLnJlbW92ZUF0dHJpYnV0ZShcImFyaWEtbGFiZWxcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgY29uc3QgaGlkZUVycm9yID0gKCkgPT4ge1xuICAgIGlmICghZWxlbWVudHMuZXJyb3JBbGVydCkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLmVycm9yQWxlcnQuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICBpZiAoZWxlbWVudHMuZXJyb3JNZXNzYWdlKSB7XG4gICAgICBlbGVtZW50cy5lcnJvck1lc3NhZ2UudGV4dENvbnRlbnQgPSBcIlwiO1xuICAgIH1cbiAgfTtcblxuICBjb25zdCBhcHBlbmRFcnJvckJ1YmJsZSA9IChlcnJvciwgb3B0aW9ucyA9IHt9KSA9PiB7XG4gICAgY29uc3Qge1xuICAgICAgbWV0YWRhdGEgPSB7fSxcbiAgICAgIHRpbWVzdGFtcCA9IG5vd0lTTygpLFxuICAgICAgcm9sZSA9IFwic3lzdGVtXCIsXG4gICAgICBwcmVmaXgsXG4gICAgICByZWdpc3RlcixcbiAgICAgIG1lc3NhZ2VJZCxcbiAgICAgIHJlc29sdmVkVGV4dCxcbiAgICB9ID0gb3B0aW9ucztcbiAgICBjb25zdCB7IHRleHQsIGJ1YmJsZVRleHQgfSA9IGNvbXB1dGVFcnJvckJ1YmJsZVRleHQoXG4gICAgICB0eXBlb2YgcmVzb2x2ZWRUZXh0ID09PSBcInN0cmluZ1wiID8gcmVzb2x2ZWRUZXh0IDogZXJyb3IsXG4gICAgICB7IHByZWZpeCB9LFxuICAgICk7XG4gICAgcmV0dXJuIGFwcGVuZE1lc3NhZ2Uocm9sZSwgYnViYmxlVGV4dCwge1xuICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICBtZXRhZGF0YTogeyAuLi5tZXRhZGF0YSwgZXJyb3I6IHRleHQgfSxcbiAgICAgIHJlZ2lzdGVyLFxuICAgICAgbWVzc2FnZUlkLFxuICAgIH0pO1xuICB9O1xuXG4gIGNvbnN0IHNob3dFcnJvciA9IChlcnJvciwgb3B0aW9ucyA9IHt9KSA9PiB7XG4gICAgY29uc3QgeyB0ZXh0IH0gPSBjb21wdXRlRXJyb3JCdWJibGVUZXh0KGVycm9yLCBvcHRpb25zKTtcbiAgICBpZiAoZWxlbWVudHMuZXJyb3JBbGVydCAmJiBlbGVtZW50cy5lcnJvck1lc3NhZ2UpIHtcbiAgICAgIGVsZW1lbnRzLmVycm9yTWVzc2FnZS50ZXh0Q29udGVudCA9IHRleHQ7XG4gICAgICBlbGVtZW50cy5lcnJvckFsZXJ0LmNsYXNzTGlzdC5yZW1vdmUoXCJkLW5vbmVcIik7XG4gICAgfVxuICAgIGlmIChvcHRpb25zLmJ1YmJsZSA9PT0gZmFsc2UpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCB7IGJ1YmJsZSwgLi4uYnViYmxlT3B0aW9ucyB9ID0gb3B0aW9ucztcbiAgICByZXR1cm4gYXBwZW5kRXJyb3JCdWJibGUoZXJyb3IsIHsgLi4uYnViYmxlT3B0aW9ucywgcmVzb2x2ZWRUZXh0OiB0ZXh0IH0pO1xuICB9O1xuXG4gIGNvbnN0IHNldENvbXBvc2VyU3RhdHVzID0gKG1lc3NhZ2UsIHRvbmUgPSBcIm11dGVkXCIpID0+IHtcbiAgICBpZiAoIWVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzKSByZXR1cm47XG4gICAgZWxlbWVudHMuY29tcG9zZXJTdGF0dXMudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIFNVUFBPUlRFRF9UT05FUy5mb3JFYWNoKCh0KSA9PlxuICAgICAgZWxlbWVudHMuY29tcG9zZXJTdGF0dXMuY2xhc3NMaXN0LnJlbW92ZShgdGV4dC0ke3R9YCksXG4gICAgKTtcbiAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy5jbGFzc0xpc3QuYWRkKGB0ZXh0LSR7dG9uZX1gKTtcbiAgfTtcblxuICBjb25zdCBzZXRDb21wb3NlclN0YXR1c0lkbGUgPSAoKSA9PiB7XG4gICAgY29uc3QgbWVzc2FnZSA9XG4gICAgICBzdGF0ZS5tb2RlID09PSBcImVtYmVkXCIgPyBjb21wb3NlclN0YXR1c0VtYmVkZGluZyA6IGNvbXBvc2VyU3RhdHVzRGVmYXVsdDtcbiAgICBzZXRDb21wb3NlclN0YXR1cyhtZXNzYWdlLCBcIm11dGVkXCIpO1xuICB9O1xuXG4gIGNvbnN0IHNjaGVkdWxlQ29tcG9zZXJJZGxlID0gKGRlbGF5ID0gMzUwMCkgPT4ge1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHN0YXRlLnJlc2V0U3RhdHVzVGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICB9LCBkZWxheSk7XG4gIH07XG5cbiAgY29uc3Qgc2V0Vm9pY2VTdGF0dXMgPSAobWVzc2FnZSwgdG9uZSA9IFwibXV0ZWRcIikgPT4ge1xuICAgIGlmICghZWxlbWVudHMudm9pY2VTdGF0dXMpIHJldHVybjtcbiAgICBpZiAoc3RhdGUudm9pY2VTdGF0dXNUaW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHN0YXRlLnZvaWNlU3RhdHVzVGltZXIpO1xuICAgICAgc3RhdGUudm9pY2VTdGF0dXNUaW1lciA9IG51bGw7XG4gICAgfVxuICAgIGVsZW1lbnRzLnZvaWNlU3RhdHVzLnRleHRDb250ZW50ID0gbWVzc2FnZTtcbiAgICBTVVBQT1JURURfVE9ORVMuZm9yRWFjaCgodCkgPT5cbiAgICAgIGVsZW1lbnRzLnZvaWNlU3RhdHVzLmNsYXNzTGlzdC5yZW1vdmUoYHRleHQtJHt0fWApLFxuICAgICk7XG4gICAgZWxlbWVudHMudm9pY2VTdGF0dXMuY2xhc3NMaXN0LmFkZChgdGV4dC0ke3RvbmV9YCk7XG4gIH07XG5cbiAgY29uc3Qgc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUgPSAoZGVsYXkgPSA0MDAwKSA9PiB7XG4gICAgaWYgKCFlbGVtZW50cy52b2ljZVN0YXR1cykgcmV0dXJuO1xuICAgIGlmIChzdGF0ZS52b2ljZVN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUudm9pY2VTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHN0YXRlLnZvaWNlU3RhdHVzVGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBzZXRWb2ljZVN0YXR1cyh2b2ljZVN0YXR1c0RlZmF1bHQsIFwibXV0ZWRcIik7XG4gICAgICBzdGF0ZS52b2ljZVN0YXR1c1RpbWVyID0gbnVsbDtcbiAgICB9LCBkZWxheSk7XG4gIH07XG5cbiAgY29uc3Qgc2V0Vm9pY2VBdmFpbGFiaWxpdHkgPSAoe1xuICAgIHJlY29nbml0aW9uID0gZmFsc2UsXG4gICAgc3ludGhlc2lzID0gZmFsc2UsXG4gIH0gPSB7fSkgPT4ge1xuICAgIGlmIChlbGVtZW50cy52b2ljZUNvbnRyb2xzKSB7XG4gICAgICBlbGVtZW50cy52b2ljZUNvbnRyb2xzLmNsYXNzTGlzdC50b2dnbGUoXG4gICAgICAgIFwiZC1ub25lXCIsXG4gICAgICAgICFyZWNvZ25pdGlvbiAmJiAhc3ludGhlc2lzLFxuICAgICAgKTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlUmVjb2duaXRpb25Hcm91cCkge1xuICAgICAgZWxlbWVudHMudm9pY2VSZWNvZ25pdGlvbkdyb3VwLmNsYXNzTGlzdC50b2dnbGUoXCJkLW5vbmVcIiwgIXJlY29nbml0aW9uKTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlVG9nZ2xlKSB7XG4gICAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS5kaXNhYmxlZCA9ICFyZWNvZ25pdGlvbjtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLnNldEF0dHJpYnV0ZShcbiAgICAgICAgXCJ0aXRsZVwiLFxuICAgICAgICByZWNvZ25pdGlvblxuICAgICAgICAgID8gXCJBY3RpdmVyIG91IGRcdTAwRTlzYWN0aXZlciBsYSBkaWN0XHUwMEU5ZSB2b2NhbGUuXCJcbiAgICAgICAgICA6IFwiRGljdFx1MDBFOWUgdm9jYWxlIGluZGlzcG9uaWJsZSBkYW5zIGNlIG5hdmlnYXRldXIuXCIsXG4gICAgICApO1xuICAgICAgZWxlbWVudHMudm9pY2VUb2dnbGUuc2V0QXR0cmlidXRlKFwiYXJpYS1wcmVzc2VkXCIsIFwiZmFsc2VcIik7XG4gICAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS5jbGFzc0xpc3QucmVtb3ZlKFwiYnRuLWRhbmdlclwiKTtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLmNsYXNzTGlzdC5hZGQoXCJidG4tb3V0bGluZS1zZWNvbmRhcnlcIik7XG4gICAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS50ZXh0Q29udGVudCA9IFwiXHVEODNDXHVERjk5XHVGRTBGIEFjdGl2ZXIgbGEgZGljdFx1MDBFOWVcIjtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQuZGlzYWJsZWQgPSAhcmVjb2duaXRpb247XG4gICAgfVxuICAgIGlmICghcmVjb2duaXRpb24pIHtcbiAgICAgIHNldFZvaWNlVHJhbnNjcmlwdChcIlwiLCB7IHN0YXRlOiBcImlkbGVcIiB9KTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlU3ludGhlc2lzR3JvdXApIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlU3ludGhlc2lzR3JvdXAuY2xhc3NMaXN0LnRvZ2dsZShcImQtbm9uZVwiLCAhc3ludGhlc2lzKTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlUGxheWJhY2spIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlUGxheWJhY2suZGlzYWJsZWQgPSAhc3ludGhlc2lzO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2spIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlU3RvcFBsYXliYWNrLmRpc2FibGVkID0gIXN5bnRoZXNpcztcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlVm9pY2VTZWxlY3QpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVm9pY2VTZWxlY3QuZGlzYWJsZWQgPSAhc3ludGhlc2lzO1xuICAgICAgaWYgKCFzeW50aGVzaXMpIHtcbiAgICAgICAgZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdC5pbm5lckhUTUwgPSBcIlwiO1xuICAgICAgfVxuICAgIH1cbiAgfTtcblxuICBmdW5jdGlvbiBzZXRWb2ljZUxpc3RlbmluZyhsaXN0ZW5pbmcpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnZvaWNlVG9nZ2xlKSByZXR1cm47XG4gICAgZWxlbWVudHMudm9pY2VUb2dnbGUuc2V0QXR0cmlidXRlKFxuICAgICAgXCJhcmlhLXByZXNzZWRcIixcbiAgICAgIGxpc3RlbmluZyA/IFwidHJ1ZVwiIDogXCJmYWxzZVwiLFxuICAgICk7XG4gICAgZWxlbWVudHMudm9pY2VUb2dnbGUuY2xhc3NMaXN0LnRvZ2dsZShcImJ0bi1kYW5nZXJcIiwgbGlzdGVuaW5nKTtcbiAgICBlbGVtZW50cy52b2ljZVRvZ2dsZS5jbGFzc0xpc3QudG9nZ2xlKFwiYnRuLW91dGxpbmUtc2Vjb25kYXJ5XCIsICFsaXN0ZW5pbmcpO1xuICAgIGVsZW1lbnRzLnZvaWNlVG9nZ2xlLnRleHRDb250ZW50ID0gbGlzdGVuaW5nXG4gICAgICA/IFwiXHVEODNEXHVERUQxIEFyclx1MDBFQXRlciBsJ1x1MDBFOWNvdXRlXCJcbiAgICAgIDogXCJcdUQ4M0NcdURGOTlcdUZFMEYgQWN0aXZlciBsYSBkaWN0XHUwMEU5ZVwiO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VUcmFuc2NyaXB0KHRleHQsIG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghZWxlbWVudHMudm9pY2VUcmFuc2NyaXB0KSByZXR1cm47XG4gICAgY29uc3QgdmFsdWUgPSB0ZXh0IHx8IFwiXCI7XG4gICAgY29uc3Qgc3RhdGVWYWx1ZSA9IG9wdGlvbnMuc3RhdGUgfHwgKHZhbHVlID8gXCJmaW5hbFwiIDogXCJpZGxlXCIpO1xuICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC50ZXh0Q29udGVudCA9IHZhbHVlO1xuICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC5kYXRhc2V0LnN0YXRlID0gc3RhdGVWYWx1ZTtcbiAgICBpZiAoIXZhbHVlICYmIG9wdGlvbnMucGxhY2Vob2xkZXIpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlVHJhbnNjcmlwdC50ZXh0Q29udGVudCA9IG9wdGlvbnMucGxhY2Vob2xkZXI7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VQcmVmZXJlbmNlcyhwcmVmcyA9IHt9KSB7XG4gICAgaWYgKGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQpIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlQXV0b1NlbmQuY2hlY2tlZCA9IEJvb2xlYW4ocHJlZnMuYXV0b1NlbmQpO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VQbGF5YmFjay5jaGVja2VkID0gQm9vbGVhbihwcmVmcy5wbGF5YmFjayk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2V0Vm9pY2VTcGVha2luZyhhY3RpdmUpIHtcbiAgICBpZiAoZWxlbWVudHMudm9pY2VTcGVha2luZ0luZGljYXRvcikge1xuICAgICAgZWxlbWVudHMudm9pY2VTcGVha2luZ0luZGljYXRvci5jbGFzc0xpc3QudG9nZ2xlKFwiZC1ub25lXCIsICFhY3RpdmUpO1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2spIHtcbiAgICAgIGVsZW1lbnRzLnZvaWNlU3RvcFBsYXliYWNrLmRpc2FibGVkID0gIWFjdGl2ZTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBzZXRWb2ljZVZvaWNlT3B0aW9ucyh2b2ljZXMgPSBbXSwgc2VsZWN0ZWRVcmkgPSBudWxsKSB7XG4gICAgaWYgKCFlbGVtZW50cy52b2ljZVZvaWNlU2VsZWN0KSByZXR1cm47XG4gICAgY29uc3Qgc2VsZWN0ID0gZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdDtcbiAgICBjb25zdCBmcmFnID0gZG9jdW1lbnQuY3JlYXRlRG9jdW1lbnRGcmFnbWVudCgpO1xuICAgIGNvbnN0IHBsYWNlaG9sZGVyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcIm9wdGlvblwiKTtcbiAgICBwbGFjZWhvbGRlci52YWx1ZSA9IFwiXCI7XG4gICAgcGxhY2Vob2xkZXIudGV4dENvbnRlbnQgPSB2b2ljZXMubGVuZ3RoXG4gICAgICA/IFwiVm9peCBwYXIgZFx1MDBFOWZhdXQgZHUgc3lzdFx1MDBFOG1lXCJcbiAgICAgIDogXCJBdWN1bmUgdm9peCBkaXNwb25pYmxlXCI7XG4gICAgZnJhZy5hcHBlbmRDaGlsZChwbGFjZWhvbGRlcik7XG4gICAgdm9pY2VzLmZvckVhY2goKHZvaWNlKSA9PiB7XG4gICAgICBjb25zdCBvcHRpb24gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwib3B0aW9uXCIpO1xuICAgICAgb3B0aW9uLnZhbHVlID0gdm9pY2Uudm9pY2VVUkkgfHwgdm9pY2UubmFtZSB8fCBcIlwiO1xuICAgICAgY29uc3QgYml0cyA9IFt2b2ljZS5uYW1lIHx8IHZvaWNlLnZvaWNlVVJJIHx8IFwiVm9peFwiXTtcbiAgICAgIGlmICh2b2ljZS5sYW5nKSB7XG4gICAgICAgIGJpdHMucHVzaChgKCR7dm9pY2UubGFuZ30pYCk7XG4gICAgICB9XG4gICAgICBpZiAodm9pY2UuZGVmYXVsdCkge1xuICAgICAgICBiaXRzLnB1c2goXCJcdTIwMjIgZFx1MDBFOWZhdXRcIik7XG4gICAgICB9XG4gICAgICBvcHRpb24udGV4dENvbnRlbnQgPSBiaXRzLmpvaW4oXCIgXCIpO1xuICAgICAgZnJhZy5hcHBlbmRDaGlsZChvcHRpb24pO1xuICAgIH0pO1xuICAgIHNlbGVjdC5pbm5lckhUTUwgPSBcIlwiO1xuICAgIHNlbGVjdC5hcHBlbmRDaGlsZChmcmFnKTtcbiAgICBpZiAoc2VsZWN0ZWRVcmkpIHtcbiAgICAgIGxldCBtYXRjaGVkID0gZmFsc2U7XG4gICAgICBBcnJheS5mcm9tKHNlbGVjdC5vcHRpb25zKS5mb3JFYWNoKChvcHRpb24pID0+IHtcbiAgICAgICAgaWYgKCFtYXRjaGVkICYmIG9wdGlvbi52YWx1ZSA9PT0gc2VsZWN0ZWRVcmkpIHtcbiAgICAgICAgICBtYXRjaGVkID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBzZWxlY3QudmFsdWUgPSBtYXRjaGVkID8gc2VsZWN0ZWRVcmkgOiBcIlwiO1xuICAgIH0gZWxzZSB7XG4gICAgICBzZWxlY3QudmFsdWUgPSBcIlwiO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNldE1vZGUobW9kZSwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3QgbmV4dCA9IG5vcm1hbGlzZU1vZGUobW9kZSk7XG4gICAgY29uc3QgcHJldmlvdXMgPSBzdGF0ZS5tb2RlO1xuICAgIHN0YXRlLm1vZGUgPSBuZXh0O1xuICAgIGlmIChlbGVtZW50cy5tb2RlU2VsZWN0ICYmIGVsZW1lbnRzLm1vZGVTZWxlY3QudmFsdWUgIT09IG5leHQpIHtcbiAgICAgIGVsZW1lbnRzLm1vZGVTZWxlY3QudmFsdWUgPSBuZXh0O1xuICAgIH1cbiAgICBpZiAoZWxlbWVudHMuY29tcG9zZXIpIHtcbiAgICAgIGVsZW1lbnRzLmNvbXBvc2VyLmRhdGFzZXQubW9kZSA9IG5leHQ7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgIGNvbnN0IHBsYWNlaG9sZGVyID1cbiAgICAgICAgbmV4dCA9PT0gXCJlbWJlZFwiXG4gICAgICAgICAgPyBwcm9tcHRQbGFjZWhvbGRlckVtYmVkZGluZ1xuICAgICAgICAgIDogcHJvbXB0UGxhY2Vob2xkZXJEZWZhdWx0O1xuICAgICAgaWYgKHBsYWNlaG9sZGVyKSB7XG4gICAgICAgIGVsZW1lbnRzLnByb21wdC5zZXRBdHRyaWJ1dGUoXCJwbGFjZWhvbGRlclwiLCBwbGFjZWhvbGRlcik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHQucmVtb3ZlQXR0cmlidXRlKFwicGxhY2Vob2xkZXJcIik7XG4gICAgICB9XG4gICAgICBjb25zdCBhcmlhTGFiZWwgPVxuICAgICAgICBuZXh0ID09PSBcImVtYmVkXCIgPyBwcm9tcHRBcmlhRW1iZWRkaW5nIDogcHJvbXB0QXJpYURlZmF1bHQ7XG4gICAgICBpZiAoYXJpYUxhYmVsKSB7XG4gICAgICAgIGVsZW1lbnRzLnByb21wdC5zZXRBdHRyaWJ1dGUoXCJhcmlhLWxhYmVsXCIsIGFyaWFMYWJlbCk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHQucmVtb3ZlQXR0cmlidXRlKFwiYXJpYS1sYWJlbFwiKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLnNlbmQpIHtcbiAgICAgIGNvbnN0IGFyaWFMYWJlbCA9IG5leHQgPT09IFwiZW1iZWRcIiA/IHNlbmRBcmlhRW1iZWRkaW5nIDogc2VuZEFyaWFEZWZhdWx0O1xuICAgICAgaWYgKGFyaWFMYWJlbCkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLnNldEF0dHJpYnV0ZShcImFyaWEtbGFiZWxcIiwgYXJpYUxhYmVsKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGVsZW1lbnRzLnNlbmQucmVtb3ZlQXR0cmlidXRlKFwiYXJpYS1sYWJlbFwiKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKCFvcHRpb25zLnNraXBTdGF0dXMgJiYgKHByZXZpb3VzICE9PSBuZXh0IHx8IG9wdGlvbnMuZm9yY2VTdGF0dXMpKSB7XG4gICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICB9XG4gICAgcmV0dXJuIG5leHQ7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVQcm9tcHRNZXRyaWNzKCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0Q291bnQgfHwgIWVsZW1lbnRzLnByb21wdCkgcmV0dXJuO1xuICAgIGNvbnN0IHZhbHVlID0gZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCI7XG4gICAgaWYgKHByb21wdE1heCkge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9IC8gJHtwcm9tcHRNYXh9YDtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9YDtcbiAgICB9XG4gICAgZWxlbWVudHMucHJvbXB0Q291bnQuY2xhc3NMaXN0LnJlbW92ZShcInRleHQtd2FybmluZ1wiLCBcInRleHQtZGFuZ2VyXCIpO1xuICAgIGlmIChwcm9tcHRNYXgpIHtcbiAgICAgIGNvbnN0IHJlbWFpbmluZyA9IHByb21wdE1heCAtIHZhbHVlLmxlbmd0aDtcbiAgICAgIGlmIChyZW1haW5pbmcgPD0gNSkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC1kYW5nZXJcIik7XG4gICAgICB9IGVsc2UgaWYgKHJlbWFpbmluZyA8PSAyMCkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC13YXJuaW5nXCIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGF1dG9zaXplUHJvbXB0KCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0KSByZXR1cm47XG4gICAgZWxlbWVudHMucHJvbXB0LnN0eWxlLmhlaWdodCA9IFwiYXV0b1wiO1xuICAgIGNvbnN0IG5leHRIZWlnaHQgPSBNYXRoLm1pbihcbiAgICAgIGVsZW1lbnRzLnByb21wdC5zY3JvbGxIZWlnaHQsXG4gICAgICBQUk9NUFRfTUFYX0hFSUdIVCxcbiAgICApO1xuICAgIGVsZW1lbnRzLnByb21wdC5zdHlsZS5oZWlnaHQgPSBgJHtuZXh0SGVpZ2h0fXB4YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzQXRCb3R0b20oKSB7XG4gICAgaWYgKCFlbGVtZW50cy50cmFuc2NyaXB0KSByZXR1cm4gdHJ1ZTtcbiAgICBjb25zdCBkaXN0YW5jZSA9XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCAtXG4gICAgICAoZWxlbWVudHMudHJhbnNjcmlwdC5zY3JvbGxUb3AgKyBlbGVtZW50cy50cmFuc2NyaXB0LmNsaWVudEhlaWdodCk7XG4gICAgcmV0dXJuIGRpc3RhbmNlIDw9IFNDUk9MTF9USFJFU0hPTEQ7XG4gIH1cblxuICBmdW5jdGlvbiBzY3JvbGxUb0JvdHRvbShvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybjtcbiAgICBjb25zdCBzbW9vdGggPSBvcHRpb25zLnNtb290aCAhPT0gZmFsc2UgJiYgIXByZWZlcnNSZWR1Y2VkTW90aW9uO1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuc2Nyb2xsVG8oe1xuICAgICAgdG9wOiBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCxcbiAgICAgIGJlaGF2aW9yOiBzbW9vdGggPyBcInNtb290aFwiIDogXCJhdXRvXCIsXG4gICAgfSk7XG4gICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICB9XG5cbiAgZnVuY3Rpb24gc2hvd1Njcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGlmIChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpO1xuICAgICAgc3RhdGUuaGlkZVNjcm9sbFRpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5yZW1vdmUoXCJkLW5vbmVcIik7XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJpcy12aXNpYmxlXCIpO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5zZXRBdHRyaWJ1dGUoXCJhcmlhLWhpZGRlblwiLCBcImZhbHNlXCIpO1xuICB9XG5cbiAgZnVuY3Rpb24gaGlkZVNjcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKFwiaXMtdmlzaWJsZVwiKTtcbiAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uc2V0QXR0cmlidXRlKFwiYXJpYS1oaWRkZW5cIiwgXCJ0cnVlXCIpO1xuICAgIHN0YXRlLmhpZGVTY3JvbGxUaW1lciA9IHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGlmIChlbGVtZW50cy5zY3JvbGxCb3R0b20pIHtcbiAgICAgICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgfSwgMjAwKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGhhbmRsZUNvcHkoYnViYmxlKSB7XG4gICAgY29uc3QgdGV4dCA9IGV4dHJhY3RCdWJibGVUZXh0KGJ1YmJsZSk7XG4gICAgaWYgKCF0ZXh0KSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICBpZiAobmF2aWdhdG9yLmNsaXBib2FyZCAmJiBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCkge1xuICAgICAgICBhd2FpdCBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCh0ZXh0KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IHRleHRhcmVhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRleHRhcmVhXCIpO1xuICAgICAgICB0ZXh0YXJlYS52YWx1ZSA9IHRleHQ7XG4gICAgICAgIHRleHRhcmVhLnNldEF0dHJpYnV0ZShcInJlYWRvbmx5XCIsIFwicmVhZG9ubHlcIik7XG4gICAgICAgIHRleHRhcmVhLnN0eWxlLnBvc2l0aW9uID0gXCJhYnNvbHV0ZVwiO1xuICAgICAgICB0ZXh0YXJlYS5zdHlsZS5sZWZ0ID0gXCItOTk5OXB4XCI7XG4gICAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgICB0ZXh0YXJlYS5zZWxlY3QoKTtcbiAgICAgICAgZG9jdW1lbnQuZXhlY0NvbW1hbmQoXCJjb3B5XCIpO1xuICAgICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKHRleHRhcmVhKTtcbiAgICAgIH1cbiAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcIkNvbnRlbnUgY29waVx1MDBFOSBkYW5zIGxlIHByZXNzZS1wYXBpZXJzLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJDb3B5IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiSW1wb3NzaWJsZSBkZSBjb3BpZXIgbGUgbWVzc2FnZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZGVjb3JhdGVSb3cocm93LCByb2xlKSB7XG4gICAgY29uc3QgYnViYmxlID0gcm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKCFidWJibGUpIHJldHVybjtcbiAgICBpZiAocm9sZSA9PT0gXCJhc3Npc3RhbnRcIiB8fCByb2xlID09PSBcInVzZXJcIikge1xuICAgICAgYnViYmxlLmNsYXNzTGlzdC5hZGQoXCJoYXMtdG9vbHNcIik7XG4gICAgICBidWJibGUucXVlcnlTZWxlY3RvckFsbChcIi5jb3B5LWJ0blwiKS5mb3JFYWNoKChidG4pID0+IGJ0bi5yZW1vdmUoKSk7XG4gICAgICBjb25zdCBjb3B5QnRuID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImJ1dHRvblwiKTtcbiAgICAgIGNvcHlCdG4udHlwZSA9IFwiYnV0dG9uXCI7XG4gICAgICBjb3B5QnRuLmNsYXNzTmFtZSA9IFwiY29weS1idG5cIjtcbiAgICAgIGNvcHlCdG4uaW5uZXJIVE1MID1cbiAgICAgICAgJzxzcGFuIGFyaWEtaGlkZGVuPVwidHJ1ZVwiPlx1MjlDOTwvc3Bhbj48c3BhbiBjbGFzcz1cInZpc3VhbGx5LWhpZGRlblwiPkNvcGllciBsZSBtZXNzYWdlPC9zcGFuPic7XG4gICAgICBjb3B5QnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiBoYW5kbGVDb3B5KGJ1YmJsZSkpO1xuICAgICAgYnViYmxlLmFwcGVuZENoaWxkKGNvcHlCdG4pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGhpZ2hsaWdodFJvdyhyb3csIHJvbGUpIHtcbiAgICBpZiAoIXJvdyB8fCBzdGF0ZS5ib290c3RyYXBwaW5nIHx8IHJvbGUgPT09IFwic3lzdGVtXCIpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgcm93LmNsYXNzTGlzdC5hZGQoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgfSwgNjAwKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGxpbmUocm9sZSwgaHRtbCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qgc2hvdWxkU3RpY2sgPSBpc0F0Qm90dG9tKCk7XG4gICAgY29uc3Qgcm93ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRpdlwiKTtcbiAgICByb3cuY2xhc3NOYW1lID0gYGNoYXQtcm93IGNoYXQtJHtyb2xlfWA7XG4gICAgcm93LmlubmVySFRNTCA9IGh0bWw7XG4gICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IG9wdGlvbnMucmF3VGV4dCB8fCBcIlwiO1xuICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IG9wdGlvbnMudGltZXN0YW1wIHx8IFwiXCI7XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5hcHBlbmRDaGlsZChyb3cpO1xuICAgIGRlY29yYXRlUm93KHJvdywgcm9sZSk7XG4gICAgaWYgKG9wdGlvbnMucmVnaXN0ZXIgIT09IGZhbHNlKSB7XG4gICAgICBjb25zdCB0cyA9IG9wdGlvbnMudGltZXN0YW1wIHx8IG5vd0lTTygpO1xuICAgICAgY29uc3QgdGV4dCA9XG4gICAgICAgIG9wdGlvbnMucmF3VGV4dCAmJiBvcHRpb25zLnJhd1RleHQubGVuZ3RoID4gMFxuICAgICAgICAgID8gb3B0aW9ucy5yYXdUZXh0XG4gICAgICAgICAgOiBodG1sVG9UZXh0KGh0bWwpO1xuICAgICAgY29uc3QgaWQgPSB0aW1lbGluZVN0b3JlLnJlZ2lzdGVyKHtcbiAgICAgICAgaWQ6IG9wdGlvbnMubWVzc2FnZUlkLFxuICAgICAgICByb2xlLFxuICAgICAgICB0ZXh0LFxuICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICByb3csXG4gICAgICAgIG1ldGFkYXRhOiBvcHRpb25zLm1ldGFkYXRhIHx8IHt9LFxuICAgICAgfSk7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBpZDtcbiAgICB9IGVsc2UgaWYgKG9wdGlvbnMubWVzc2FnZUlkKSB7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBvcHRpb25zLm1lc3NhZ2VJZDtcbiAgICB9IGVsc2UgaWYgKCFyb3cuZGF0YXNldC5tZXNzYWdlSWQpIHtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IHRpbWVsaW5lU3RvcmUubWFrZU1lc3NhZ2VJZCgpO1xuICAgIH1cbiAgICBpZiAoc2hvdWxkU3RpY2spIHtcbiAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiAhc3RhdGUuYm9vdHN0cmFwcGluZyB9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgc2hvd1Njcm9sbEJ1dHRvbigpO1xuICAgIH1cbiAgICBoaWdobGlnaHRSb3cocm93LCByb2xlKTtcbiAgICBpZiAoc3RhdGUuYWN0aXZlRmlsdGVyKSB7XG4gICAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIoc3RhdGUuYWN0aXZlRmlsdGVyLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfVxuICAgIHJldHVybiByb3c7XG4gIH1cblxuICBmdW5jdGlvbiBidWlsZEJ1YmJsZSh7XG4gICAgdGV4dCxcbiAgICB0aW1lc3RhbXAsXG4gICAgdmFyaWFudCxcbiAgICBtZXRhU3VmZml4LFxuICAgIGFsbG93TWFya2Rvd24gPSB0cnVlLFxuICB9KSB7XG4gICAgY29uc3QgY2xhc3NlcyA9IFtcImNoYXQtYnViYmxlXCJdO1xuICAgIGlmICh2YXJpYW50KSB7XG4gICAgICBjbGFzc2VzLnB1c2goYGNoYXQtYnViYmxlLSR7dmFyaWFudH1gKTtcbiAgICB9XG4gICAgY29uc3QgY29udGVudCA9IGFsbG93TWFya2Rvd25cbiAgICAgID8gcmVuZGVyTWFya2Rvd24odGV4dClcbiAgICAgIDogZXNjYXBlSFRNTChTdHJpbmcodGV4dCkpO1xuICAgIGNvbnN0IG1ldGFCaXRzID0gW107XG4gICAgaWYgKHRpbWVzdGFtcCkge1xuICAgICAgbWV0YUJpdHMucHVzaChmb3JtYXRUaW1lc3RhbXAodGltZXN0YW1wKSk7XG4gICAgfVxuICAgIGlmIChtZXRhU3VmZml4KSB7XG4gICAgICBtZXRhQml0cy5wdXNoKG1ldGFTdWZmaXgpO1xuICAgIH1cbiAgICBjb25zdCBtZXRhSHRtbCA9XG4gICAgICBtZXRhQml0cy5sZW5ndGggPiAwXG4gICAgICAgID8gYDxkaXYgY2xhc3M9XCJjaGF0LW1ldGFcIj4ke2VzY2FwZUhUTUwobWV0YUJpdHMuam9pbihcIiBcdTIwMjIgXCIpKX08L2Rpdj5gXG4gICAgICAgIDogXCJcIjtcbiAgICByZXR1cm4gYDxkaXYgY2xhc3M9XCIke2NsYXNzZXMuam9pbihcIiBcIil9XCI+JHtjb250ZW50fSR7bWV0YUh0bWx9PC9kaXY+YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIG5vcm1hbGlzZU51bWVyaWModmFsdWUpIHtcbiAgICBpZiAodHlwZW9mIHZhbHVlID09PSBcIm51bWJlclwiICYmIE51bWJlci5pc0Zpbml0ZSh2YWx1ZSkpIHtcbiAgICAgIHJldHVybiB2YWx1ZTtcbiAgICB9XG4gICAgY29uc3QgcGFyc2VkID0gTnVtYmVyKHZhbHVlKTtcbiAgICByZXR1cm4gTnVtYmVyLmlzRmluaXRlKHBhcnNlZCkgPyBwYXJzZWQgOiBudWxsO1xuICB9XG5cbiAgZnVuY3Rpb24gZm9ybWF0TnVtZXJpYyh2YWx1ZSkge1xuICAgIGlmICghTnVtYmVyLmlzRmluaXRlKHZhbHVlKSkge1xuICAgICAgcmV0dXJuIFwiXHUyMDE0XCI7XG4gICAgfVxuICAgIGlmICh2YWx1ZSA9PT0gMCkge1xuICAgICAgcmV0dXJuIFwiMFwiO1xuICAgIH1cbiAgICBjb25zdCBhYnMgPSBNYXRoLmFicyh2YWx1ZSk7XG4gICAgaWYgKGFicyA+PSAxMDAwIHx8IGFicyA8IDAuMDAxKSB7XG4gICAgICByZXR1cm4gdmFsdWUudG9FeHBvbmVudGlhbCg0KTtcbiAgICB9XG4gICAgY29uc3QgZml4ZWQgPSB2YWx1ZS50b0ZpeGVkKDYpO1xuICAgIHJldHVybiBmaXhlZC5yZXBsYWNlKC9cXC4wKyQvLCBcIlwiKS5yZXBsYWNlKC8wKyQvLCBcIlwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHN1bW1hcmlzZVZlY3Rvcih2ZWN0b3IsIGluZGV4KSB7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHZlY3RvcikpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCBwcmV2aWV3ID0gW107XG4gICAgbGV0IGNvdW50ID0gMDtcbiAgICBsZXQgc3VtID0gMDtcbiAgICBsZXQgc3F1YXJlcyA9IDA7XG4gICAgbGV0IG1pbiA9IEluZmluaXR5O1xuICAgIGxldCBtYXggPSAtSW5maW5pdHk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2ZWN0b3IubGVuZ3RoOyBpICs9IDEpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gbm9ybWFsaXNlTnVtZXJpYyh2ZWN0b3JbaV0pO1xuICAgICAgaWYgKHZhbHVlID09PSBudWxsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgaWYgKHByZXZpZXcubGVuZ3RoIDwgOCkge1xuICAgICAgICBwcmV2aWV3LnB1c2godmFsdWUpO1xuICAgICAgfVxuICAgICAgY291bnQgKz0gMTtcbiAgICAgIHN1bSArPSB2YWx1ZTtcbiAgICAgIHNxdWFyZXMgKz0gdmFsdWUgKiB2YWx1ZTtcbiAgICAgIGlmICh2YWx1ZSA8IG1pbikge1xuICAgICAgICBtaW4gPSB2YWx1ZTtcbiAgICAgIH1cbiAgICAgIGlmICh2YWx1ZSA+IG1heCkge1xuICAgICAgICBtYXggPSB2YWx1ZTtcbiAgICAgIH1cbiAgICB9XG4gICAgY29uc3QgbWFnbml0dWRlID0gY291bnQgPiAwID8gTWF0aC5zcXJ0KHNxdWFyZXMpIDogbnVsbDtcbiAgICBjb25zdCBtZWFuID0gY291bnQgPiAwID8gc3VtIC8gY291bnQgOiBudWxsO1xuICAgIHJldHVybiB7XG4gICAgICBpbmRleCxcbiAgICAgIGNvdW50LFxuICAgICAgc3VtLFxuICAgICAgc3F1YXJlcyxcbiAgICAgIG1hZ25pdHVkZSxcbiAgICAgIG1lYW4sXG4gICAgICBtaW46IGNvdW50ID4gMCA/IG1pbiA6IG51bGwsXG4gICAgICBtYXg6IGNvdW50ID4gMCA/IG1heCA6IG51bGwsXG4gICAgICBwcmV2aWV3LFxuICAgIH07XG4gIH1cblxuICBmdW5jdGlvbiBjcmVhdGVWZWN0b3JTdGF0c1RhYmxlKHN0YXRzKSB7XG4gICAgY29uc3QgdGFibGUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidGFibGVcIik7XG4gICAgdGFibGUuY2xhc3NOYW1lID1cbiAgICAgIFwidGFibGUgdGFibGUtc20gdGFibGUtc3RyaXBlZCBlbWJlZGRpbmctZGV0YWlscy10YWJsZSBtYi0wXCI7XG4gICAgY29uc3QgdGhlYWQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidGhlYWRcIik7XG4gICAgY29uc3QgaGVhZGVyUm93ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRyXCIpO1xuICAgIFtcbiAgICAgIFwiVmVjdGV1clwiLFxuICAgICAgXCJDb21wb3NhbnRlc1wiLFxuICAgICAgXCJNYWduaXR1ZGVcIixcbiAgICAgIFwiTW95ZW5uZVwiLFxuICAgICAgXCJNaW5cIixcbiAgICAgIFwiTWF4XCIsXG4gICAgICBcIkFwZXJcdTAwRTd1XCIsXG4gICAgXS5mb3JFYWNoKChsYWJlbCkgPT4ge1xuICAgICAgY29uc3QgdGggPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidGhcIik7XG4gICAgICB0aC5zY29wZSA9IFwiY29sXCI7XG4gICAgICB0aC50ZXh0Q29udGVudCA9IGxhYmVsO1xuICAgICAgaGVhZGVyUm93LmFwcGVuZENoaWxkKHRoKTtcbiAgICB9KTtcbiAgICB0aGVhZC5hcHBlbmRDaGlsZChoZWFkZXJSb3cpO1xuICAgIHRhYmxlLmFwcGVuZENoaWxkKHRoZWFkKTtcblxuICAgIGNvbnN0IHRib2R5ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRib2R5XCIpO1xuICAgIHN0YXRzLmZvckVhY2goKHN0YXQpID0+IHtcbiAgICAgIGNvbnN0IHJvdyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJ0clwiKTtcbiAgICAgIGNvbnN0IGNlbGxzID0gW1xuICAgICAgICBzdGF0LmluZGV4ICsgMSxcbiAgICAgICAgc3RhdC5jb3VudCxcbiAgICAgICAgZm9ybWF0TnVtZXJpYyhzdGF0Lm1hZ25pdHVkZSksXG4gICAgICAgIGZvcm1hdE51bWVyaWMoc3RhdC5tZWFuKSxcbiAgICAgICAgZm9ybWF0TnVtZXJpYyhzdGF0Lm1pbiksXG4gICAgICAgIGZvcm1hdE51bWVyaWMoc3RhdC5tYXgpLFxuICAgICAgICBzdGF0LnByZXZpZXcubGVuZ3RoXG4gICAgICAgICAgPyBzdGF0LnByZXZpZXcubWFwKCh2YWx1ZSkgPT4gZm9ybWF0TnVtZXJpYyh2YWx1ZSkpLmpvaW4oXCIsIFwiKVxuICAgICAgICAgIDogXCJcdTIwMTRcIixcbiAgICAgIF07XG4gICAgICBjZWxscy5mb3JFYWNoKCh2YWx1ZSkgPT4ge1xuICAgICAgICBjb25zdCB0ZCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJ0ZFwiKTtcbiAgICAgICAgdGQudGV4dENvbnRlbnQgPSBTdHJpbmcodmFsdWUpO1xuICAgICAgICByb3cuYXBwZW5kQ2hpbGQodGQpO1xuICAgICAgfSk7XG4gICAgICB0Ym9keS5hcHBlbmRDaGlsZChyb3cpO1xuICAgIH0pO1xuICAgIHRhYmxlLmFwcGVuZENoaWxkKHRib2R5KTtcbiAgICByZXR1cm4gdGFibGU7XG4gIH1cblxuICBmdW5jdGlvbiBhdHRhY2hFbWJlZGRpbmdEZXRhaWxzKHJvdywgZW1iZWRkaW5nRGF0YSA9IHt9LCBtZXRhZGF0YSA9IHt9KSB7XG4gICAgaWYgKCFyb3cpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgYnViYmxlID0gcm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKCFidWJibGUpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgYnViYmxlXG4gICAgICAucXVlcnlTZWxlY3RvckFsbChcIi5lbWJlZGRpbmctZGV0YWlsc1wiKVxuICAgICAgLmZvckVhY2goKG5vZGUpID0+IG5vZGUucmVtb3ZlKCkpO1xuXG4gICAgY29uc3QgdmVjdG9ycyA9IEFycmF5LmlzQXJyYXkoZW1iZWRkaW5nRGF0YS52ZWN0b3JzKVxuICAgICAgPyBlbWJlZGRpbmdEYXRhLnZlY3RvcnMuZmlsdGVyKCh2ZWN0b3IpID0+IEFycmF5LmlzQXJyYXkodmVjdG9yKSlcbiAgICAgIDogW107XG4gICAgaWYgKHZlY3RvcnMubGVuZ3RoID09PSAwKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgc3RhdHMgPSB2ZWN0b3JzXG4gICAgICAubWFwKCh2ZWN0b3IsIGluZGV4KSA9PiBzdW1tYXJpc2VWZWN0b3IodmVjdG9yLCBpbmRleCkpXG4gICAgICAuZmlsdGVyKChlbnRyeSkgPT4gZW50cnkgJiYgZW50cnkuY291bnQgPj0gMCk7XG4gICAgaWYgKHN0YXRzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IGRldGFpbHMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiZGl2XCIpO1xuICAgIGRldGFpbHMuY2xhc3NOYW1lID0gXCJlbWJlZGRpbmctZGV0YWlscyBjYXJkIG10LTNcIjtcblxuICAgIGNvbnN0IGNhcmRCb2R5ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRpdlwiKTtcbiAgICBjYXJkQm9keS5jbGFzc05hbWUgPSBcImNhcmQtYm9keSBwLTNcIjtcbiAgICBkZXRhaWxzLmFwcGVuZENoaWxkKGNhcmRCb2R5KTtcblxuICAgIGNvbnN0IGhlYWRlciA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJkaXZcIik7XG4gICAgaGVhZGVyLmNsYXNzTmFtZSA9IFwiZC1mbGV4IGZsZXgtd3JhcCBhbGlnbi1pdGVtcy1jZW50ZXIgZ2FwLTIgbWItM1wiO1xuXG4gICAgY29uc3QgdGl0bGUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiaDVcIik7XG4gICAgdGl0bGUuY2xhc3NOYW1lID0gXCJjYXJkLXRpdGxlIG1iLTBcIjtcbiAgICB0aXRsZS50ZXh0Q29udGVudCA9IFwiQW5hbHlzZSBkZXMgZW1iZWRkaW5nc1wiO1xuICAgIGhlYWRlci5hcHBlbmRDaGlsZCh0aXRsZSk7XG5cbiAgICBjb25zdCBkb3dubG9hZEJ0biA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJidXR0b25cIik7XG4gICAgZG93bmxvYWRCdG4udHlwZSA9IFwiYnV0dG9uXCI7XG4gICAgZG93bmxvYWRCdG4uY2xhc3NOYW1lID0gXCJidG4gYnRuLXNtIGJ0bi1vdXRsaW5lLXByaW1hcnkgbXMtYXV0b1wiO1xuICAgIGRvd25sb2FkQnRuLnRleHRDb250ZW50ID0gXCJUXHUwMEU5bFx1MDBFOWNoYXJnZXIgbGUgSlNPTlwiO1xuICAgIGRvd25sb2FkQnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICB0cnkge1xuICAgICAgICBjb25zdCBwYXlsb2FkID1cbiAgICAgICAgICB0eXBlb2YgZW1iZWRkaW5nRGF0YS5yYXcgPT09IFwib2JqZWN0XCIgJiYgZW1iZWRkaW5nRGF0YS5yYXcgIT09IG51bGxcbiAgICAgICAgICAgID8gZW1iZWRkaW5nRGF0YS5yYXdcbiAgICAgICAgICAgIDoge1xuICAgICAgICAgICAgICAgIGJhY2tlbmQ6IGVtYmVkZGluZ0RhdGEuYmFja2VuZCA/PyBtZXRhZGF0YS5iYWNrZW5kID8/IG51bGwsXG4gICAgICAgICAgICAgICAgbW9kZWw6IGVtYmVkZGluZ0RhdGEubW9kZWwgPz8gbWV0YWRhdGEubW9kZWwgPz8gbnVsbCxcbiAgICAgICAgICAgICAgICBkaW1zOlxuICAgICAgICAgICAgICAgICAgZW1iZWRkaW5nRGF0YS5kaW1zID8/XG4gICAgICAgICAgICAgICAgICBtZXRhZGF0YS5kaW1zID8/XG4gICAgICAgICAgICAgICAgICBzdGF0c1swXT8uY291bnQgPz9cbiAgICAgICAgICAgICAgICAgIG51bGwsXG4gICAgICAgICAgICAgICAgbm9ybWFsaXNlZDpcbiAgICAgICAgICAgICAgICAgIHR5cGVvZiBlbWJlZGRpbmdEYXRhLm5vcm1hbGlzZWQgIT09IFwidW5kZWZpbmVkXCJcbiAgICAgICAgICAgICAgICAgICAgPyBCb29sZWFuKGVtYmVkZGluZ0RhdGEubm9ybWFsaXNlZClcbiAgICAgICAgICAgICAgICAgICAgOiBCb29sZWFuKG1ldGFkYXRhLm5vcm1hbGlzZWQpLFxuICAgICAgICAgICAgICAgIGNvdW50OiB2ZWN0b3JzLmxlbmd0aCxcbiAgICAgICAgICAgICAgICB2ZWN0b3JzLFxuICAgICAgICAgICAgICB9O1xuICAgICAgICBjb25zdCBibG9iID0gbmV3IEJsb2IoW0pTT04uc3RyaW5naWZ5KHBheWxvYWQsIG51bGwsIDIpXSwge1xuICAgICAgICAgIHR5cGU6IFwiYXBwbGljYXRpb24vanNvblwiLFxuICAgICAgICB9KTtcbiAgICAgICAgY29uc3QgdXJsID0gd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG4gICAgICAgIGNvbnN0IGxpbmsgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiYVwiKTtcbiAgICAgICAgY29uc3Qgc2x1Z1NvdXJjZSA9IChcbiAgICAgICAgICBlbWJlZGRpbmdEYXRhLm1vZGVsIHx8XG4gICAgICAgICAgbWV0YWRhdGEubW9kZWwgfHxcbiAgICAgICAgICBcImVtYmVkZGluZ1wiXG4gICAgICAgIClcbiAgICAgICAgICAudG9TdHJpbmcoKVxuICAgICAgICAgIC50b0xvd2VyQ2FzZSgpO1xuICAgICAgICBjb25zdCBzbHVnID0gc2x1Z1NvdXJjZVxuICAgICAgICAgIC5yZXBsYWNlKC9bXmEtejAtOS5fLV0rL2csIFwiLVwiKVxuICAgICAgICAgIC5yZXBsYWNlKC9eLSt8LSskL2csIFwiXCIpXG4gICAgICAgICAgLnNsaWNlKDAsIDYwKTtcbiAgICAgICAgbGluay5ocmVmID0gdXJsO1xuICAgICAgICBsaW5rLmRvd25sb2FkID0gYGVtYmVkZGluZy0ke3NsdWcgfHwgXCJyZXN1bHRcIn0tJHtEYXRlLm5vdygpfS5qc29uYDtcbiAgICAgICAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChsaW5rKTtcbiAgICAgICAgbGluay5jbGljaygpO1xuICAgICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKGxpbmspO1xuICAgICAgICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgd2luZG93LlVSTC5yZXZva2VPYmplY3RVUkwodXJsKTtcbiAgICAgICAgfSwgMTAwMCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIGRvd25sb2FkIGVtYmVkZGluZyBwYXlsb2FkXCIsIGVycik7XG4gICAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcbiAgICAgICAgICBcIkltcG9zc2libGUgZGUgdFx1MDBFOWxcdTAwRTljaGFyZ2VyIGxlIHJcdTAwRTlzdWx0YXQgZCdlbWJlZGRpbmcuXCIsXG4gICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBoZWFkZXIuYXBwZW5kQ2hpbGQoZG93bmxvYWRCdG4pO1xuICAgIGNhcmRCb2R5LmFwcGVuZENoaWxkKGhlYWRlcik7XG5cbiAgICBjb25zdCBkaW1zQ2FuZGlkYXRlID0gTnVtYmVyKGVtYmVkZGluZ0RhdGEuZGltcyA/PyBtZXRhZGF0YS5kaW1zKTtcbiAgICBjb25zdCBkaW1zID0gTnVtYmVyLmlzRmluaXRlKGRpbXNDYW5kaWRhdGUpXG4gICAgICA/IE51bWJlcihkaW1zQ2FuZGlkYXRlKVxuICAgICAgOiBBcnJheS5pc0FycmF5KHZlY3RvcnNbMF0pXG4gICAgICAgID8gdmVjdG9yc1swXS5sZW5ndGhcbiAgICAgICAgOiBudWxsO1xuICAgIGNvbnN0IHZhbGlkTWFnbml0dWRlU3RhdHMgPSBzdGF0cy5maWx0ZXIoXG4gICAgICAoc3RhdCkgPT5cbiAgICAgICAgdHlwZW9mIHN0YXQubWFnbml0dWRlID09PSBcIm51bWJlclwiICYmICFOdW1iZXIuaXNOYU4oc3RhdC5tYWduaXR1ZGUpLFxuICAgICk7XG4gICAgY29uc3QgdG90YWxNYWduaXR1ZGUgPSB2YWxpZE1hZ25pdHVkZVN0YXRzLnJlZHVjZShcbiAgICAgIChhY2MsIHN0YXQpID0+IGFjYyArIHN0YXQubWFnbml0dWRlLFxuICAgICAgMCxcbiAgICApO1xuICAgIGNvbnN0IGF2Z01hZ25pdHVkZSA9XG4gICAgICB2YWxpZE1hZ25pdHVkZVN0YXRzLmxlbmd0aCA+IDBcbiAgICAgICAgPyB0b3RhbE1hZ25pdHVkZSAvIHZhbGlkTWFnbml0dWRlU3RhdHMubGVuZ3RoXG4gICAgICAgIDogbnVsbDtcblxuICAgIGxldCBjb21wb25lbnRDb3VudCA9IDA7XG4gICAgbGV0IGNvbXBvbmVudFN1bSA9IDA7XG4gICAgbGV0IGNvbXBvbmVudFNxdWFyZXMgPSAwO1xuICAgIGxldCBnbG9iYWxNaW4gPSBudWxsO1xuICAgIGxldCBnbG9iYWxNYXggPSBudWxsO1xuICAgIHN0YXRzLmZvckVhY2goKHN0YXQpID0+IHtcbiAgICAgIGNvbXBvbmVudENvdW50ICs9IHN0YXQuY291bnQ7XG4gICAgICBjb21wb25lbnRTdW0gKz0gc3RhdC5zdW07XG4gICAgICBjb21wb25lbnRTcXVhcmVzICs9IHN0YXQuc3F1YXJlcztcbiAgICAgIGlmIChzdGF0LmNvdW50ID4gMCkge1xuICAgICAgICBnbG9iYWxNaW4gPVxuICAgICAgICAgIGdsb2JhbE1pbiA9PT0gbnVsbCA/IHN0YXQubWluIDogTWF0aC5taW4oZ2xvYmFsTWluLCBzdGF0Lm1pbik7XG4gICAgICAgIGdsb2JhbE1heCA9XG4gICAgICAgICAgZ2xvYmFsTWF4ID09PSBudWxsID8gc3RhdC5tYXggOiBNYXRoLm1heChnbG9iYWxNYXgsIHN0YXQubWF4KTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBjb25zdCBhZ2dyZWdhdGVNYWduaXR1ZGUgPVxuICAgICAgY29tcG9uZW50Q291bnQgPiAwID8gTWF0aC5zcXJ0KGNvbXBvbmVudFNxdWFyZXMpIDogbnVsbDtcbiAgICBjb25zdCBhZ2dyZWdhdGVNZWFuID1cbiAgICAgIGNvbXBvbmVudENvdW50ID4gMCA/IGNvbXBvbmVudFN1bSAvIGNvbXBvbmVudENvdW50IDogbnVsbDtcblxuICAgIGNvbnN0IG1ldGFMaXN0ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRsXCIpO1xuICAgIG1ldGFMaXN0LmNsYXNzTmFtZSA9IFwicm93IGctMiBtYi0wXCI7XG5cbiAgICBjb25zdCBwdXNoTWV0YSA9IChsYWJlbCwgdmFsdWUpID0+IHtcbiAgICAgIGNvbnN0IGR0ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImR0XCIpO1xuICAgICAgZHQuY2xhc3NOYW1lID0gXCJjb2wtc20tNFwiO1xuICAgICAgZHQudGV4dENvbnRlbnQgPSBsYWJlbDtcbiAgICAgIGNvbnN0IGRkID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRkXCIpO1xuICAgICAgZGQuY2xhc3NOYW1lID0gXCJjb2wtc20tOFwiO1xuICAgICAgZGQudGV4dENvbnRlbnQgPSB2YWx1ZTtcbiAgICAgIG1ldGFMaXN0LmFwcGVuZENoaWxkKGR0KTtcbiAgICAgIG1ldGFMaXN0LmFwcGVuZENoaWxkKGRkKTtcbiAgICB9O1xuXG4gICAgaWYgKGVtYmVkZGluZ0RhdGEuYmFja2VuZCB8fCBtZXRhZGF0YS5iYWNrZW5kKSB7XG4gICAgICBwdXNoTWV0YShcIkJhY2tlbmRcIiwgU3RyaW5nKGVtYmVkZGluZ0RhdGEuYmFja2VuZCB8fCBtZXRhZGF0YS5iYWNrZW5kKSk7XG4gICAgfVxuICAgIGlmIChlbWJlZGRpbmdEYXRhLm1vZGVsIHx8IG1ldGFkYXRhLm1vZGVsKSB7XG4gICAgICBwdXNoTWV0YShcIk1vZFx1MDBFOGxlXCIsIFN0cmluZyhlbWJlZGRpbmdEYXRhLm1vZGVsIHx8IG1ldGFkYXRhLm1vZGVsKSk7XG4gICAgfVxuICAgIGlmIChkaW1zKSB7XG4gICAgICBwdXNoTWV0YShcIkRpbWVuc2lvbnNcIiwgU3RyaW5nKGRpbXMpKTtcbiAgICB9XG4gICAgcHVzaE1ldGEoXCJWZWN0ZXVyc1wiLCBgJHt2ZWN0b3JzLmxlbmd0aH1gKTtcbiAgICBpZiAoY29tcG9uZW50Q291bnQpIHtcbiAgICAgIHB1c2hNZXRhKFwiQ29tcG9zYW50ZXNcIiwgYCR7Y29tcG9uZW50Q291bnR9YCk7XG4gICAgfVxuICAgIHB1c2hNZXRhKFxuICAgICAgXCJOb3JtYWxpc2F0aW9uXCIsXG4gICAgICBCb29sZWFuKFxuICAgICAgICB0eXBlb2YgZW1iZWRkaW5nRGF0YS5ub3JtYWxpc2VkICE9PSBcInVuZGVmaW5lZFwiXG4gICAgICAgICAgPyBlbWJlZGRpbmdEYXRhLm5vcm1hbGlzZWRcbiAgICAgICAgICA6IG1ldGFkYXRhLm5vcm1hbGlzZWQsXG4gICAgICApXG4gICAgICAgID8gXCJPdWlcIlxuICAgICAgICA6IFwiTm9uXCIsXG4gICAgKTtcbiAgICBwdXNoTWV0YShcIk1hZ25pdHVkZSBtb3llbm5lXCIsIGZvcm1hdE51bWVyaWMoYXZnTWFnbml0dWRlKSk7XG4gICAgcHVzaE1ldGEoXCJNYWduaXR1ZGUgYWdyXHUwMEU5Z1x1MDBFOWVcIiwgZm9ybWF0TnVtZXJpYyhhZ2dyZWdhdGVNYWduaXR1ZGUpKTtcbiAgICBwdXNoTWV0YShcIk1veWVubmUgZ2xvYmFsZVwiLCBmb3JtYXROdW1lcmljKGFnZ3JlZ2F0ZU1lYW4pKTtcbiAgICBwdXNoTWV0YShcIk1pbmltdW0gZ2xvYmFsXCIsIGZvcm1hdE51bWVyaWMoZ2xvYmFsTWluKSk7XG4gICAgcHVzaE1ldGEoXCJNYXhpbXVtIGdsb2JhbFwiLCBmb3JtYXROdW1lcmljKGdsb2JhbE1heCkpO1xuXG4gICAgY2FyZEJvZHkuYXBwZW5kQ2hpbGQobWV0YUxpc3QpO1xuXG4gICAgY29uc3QgdGFibGUgPSBjcmVhdGVWZWN0b3JTdGF0c1RhYmxlKHN0YXRzKTtcbiAgICB0YWJsZS5jbGFzc0xpc3QuYWRkKFwibXQtM1wiKTtcbiAgICBjYXJkQm9keS5hcHBlbmRDaGlsZCh0YWJsZSk7XG5cbiAgICBjb25zdCBkZXRhaWxzV3JhcHBlciA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJkZXRhaWxzXCIpO1xuICAgIGRldGFpbHNXcmFwcGVyLmNsYXNzTmFtZSA9IFwibXQtM1wiO1xuICAgIGNvbnN0IHN1bW1hcnkgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwic3VtbWFyeVwiKTtcbiAgICBzdW1tYXJ5LnRleHRDb250ZW50ID0gXCJBZmZpY2hlciBsZSBKU09OIGJydXRcIjtcbiAgICBkZXRhaWxzV3JhcHBlci5hcHBlbmRDaGlsZChzdW1tYXJ5KTtcbiAgICBjb25zdCBwcmUgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwicHJlXCIpO1xuICAgIHByZS5jbGFzc05hbWUgPSBcIm10LTIgbWItMCBvdmVyZmxvdy1hdXRvXCI7XG4gICAgcHJlLnN0eWxlLm1heEhlaWdodCA9IFwiMjQwcHhcIjtcbiAgICBwcmUudGV4dENvbnRlbnQgPSBKU09OLnN0cmluZ2lmeSh2ZWN0b3JzLCBudWxsLCAyKTtcbiAgICBkZXRhaWxzV3JhcHBlci5hcHBlbmRDaGlsZChwcmUpO1xuICAgIGNhcmRCb2R5LmFwcGVuZENoaWxkKGRldGFpbHNXcmFwcGVyKTtcblxuICAgIGJ1YmJsZS5hcHBlbmRDaGlsZChkZXRhaWxzKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGVuZE1lc3NhZ2Uocm9sZSwgdGV4dCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qge1xuICAgICAgdGltZXN0YW1wLFxuICAgICAgdmFyaWFudCxcbiAgICAgIG1ldGFTdWZmaXgsXG4gICAgICBhbGxvd01hcmtkb3duID0gdHJ1ZSxcbiAgICAgIG1lc3NhZ2VJZCxcbiAgICAgIHJlZ2lzdGVyID0gdHJ1ZSxcbiAgICAgIG1ldGFkYXRhLFxuICAgICAgZW1iZWRkaW5nRGF0YSxcbiAgICB9ID0gb3B0aW9ucztcbiAgICBjb25zdCBidWJibGUgPSBidWlsZEJ1YmJsZSh7XG4gICAgICB0ZXh0LFxuICAgICAgdGltZXN0YW1wLFxuICAgICAgdmFyaWFudCxcbiAgICAgIG1ldGFTdWZmaXgsXG4gICAgICBhbGxvd01hcmtkb3duLFxuICAgIH0pO1xuICAgIGNvbnN0IHJvdyA9IGxpbmUocm9sZSwgYnViYmxlLCB7XG4gICAgICByYXdUZXh0OiB0ZXh0LFxuICAgICAgdGltZXN0YW1wLFxuICAgICAgbWVzc2FnZUlkLFxuICAgICAgcmVnaXN0ZXIsXG4gICAgICBtZXRhZGF0YSxcbiAgICB9KTtcbiAgICBpZiAocm9sZSA9PT0gXCJhc3Npc3RhbnRcIiAmJiBlbWJlZGRpbmdEYXRhKSB7XG4gICAgICBhdHRhY2hFbWJlZGRpbmdEZXRhaWxzKHJvdywgZW1iZWRkaW5nRGF0YSwgbWV0YWRhdGEgfHwge30pO1xuICAgIH1cbiAgICBzZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHRpbWVzdGFtcCB8fCBub3dJU08oKSB9KTtcbiAgICByZXR1cm4gcm93O1xuICB9XG5cbiAgZnVuY3Rpb24gdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKGVsLCB2YWx1ZSkge1xuICAgIGlmICghZWwpIHJldHVybjtcbiAgICBlbC50ZXh0Q29udGVudCA9IHZhbHVlIHx8IFwiXHUyMDE0XCI7XG4gIH1cblxuICBmdW5jdGlvbiBzZXREaWFnbm9zdGljcyhwYXRjaCkge1xuICAgIE9iamVjdC5hc3NpZ24oZGlhZ25vc3RpY3MsIHBhdGNoKTtcbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHBhdGNoLCBcImNvbm5lY3RlZEF0XCIpKSB7XG4gICAgICB1cGRhdGVEaWFnbm9zdGljRmllbGQoXG4gICAgICAgIGVsZW1lbnRzLmRpYWdDb25uZWN0ZWQsXG4gICAgICAgIGRpYWdub3N0aWNzLmNvbm5lY3RlZEF0XG4gICAgICAgICAgPyBmb3JtYXRUaW1lc3RhbXAoZGlhZ25vc3RpY3MuY29ubmVjdGVkQXQpXG4gICAgICAgICAgOiBcIlx1MjAxNFwiLFxuICAgICAgKTtcbiAgICB9XG4gICAgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChwYXRjaCwgXCJsYXN0TWVzc2FnZUF0XCIpKSB7XG4gICAgICB1cGRhdGVEaWFnbm9zdGljRmllbGQoXG4gICAgICAgIGVsZW1lbnRzLmRpYWdMYXN0TWVzc2FnZSxcbiAgICAgICAgZGlhZ25vc3RpY3MubGFzdE1lc3NhZ2VBdFxuICAgICAgICAgID8gZm9ybWF0VGltZXN0YW1wKGRpYWdub3N0aWNzLmxhc3RNZXNzYWdlQXQpXG4gICAgICAgICAgOiBcIlx1MjAxNFwiLFxuICAgICAgKTtcbiAgICB9XG4gICAgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChwYXRjaCwgXCJsYXRlbmN5TXNcIikpIHtcbiAgICAgIGlmICh0eXBlb2YgZGlhZ25vc3RpY3MubGF0ZW5jeU1zID09PSBcIm51bWJlclwiKSB7XG4gICAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChcbiAgICAgICAgICBlbGVtZW50cy5kaWFnTGF0ZW5jeSxcbiAgICAgICAgICBgJHtNYXRoLm1heCgwLCBNYXRoLnJvdW5kKGRpYWdub3N0aWNzLmxhdGVuY3lNcykpfSBtc2AsXG4gICAgICAgICk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB1cGRhdGVEaWFnbm9zdGljRmllbGQoZWxlbWVudHMuZGlhZ0xhdGVuY3ksIFwiXHUyMDE0XCIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZU5ldHdvcmtTdGF0dXMoKSB7XG4gICAgaWYgKCFlbGVtZW50cy5kaWFnTmV0d29yaykgcmV0dXJuO1xuICAgIGNvbnN0IG9ubGluZSA9IG5hdmlnYXRvci5vbkxpbmU7XG4gICAgZWxlbWVudHMuZGlhZ05ldHdvcmsudGV4dENvbnRlbnQgPSBvbmxpbmUgPyBcIkVuIGxpZ25lXCIgOiBcIkhvcnMgbGlnbmVcIjtcbiAgICBlbGVtZW50cy5kaWFnTmV0d29yay5jbGFzc0xpc3QudG9nZ2xlKFwidGV4dC1kYW5nZXJcIiwgIW9ubGluZSk7XG4gICAgZWxlbWVudHMuZGlhZ05ldHdvcmsuY2xhc3NMaXN0LnRvZ2dsZShcInRleHQtc3VjY2Vzc1wiLCBvbmxpbmUpO1xuICB9XG5cbiAgZnVuY3Rpb24gYW5ub3VuY2VDb25uZWN0aW9uKG1lc3NhZ2UsIHZhcmlhbnQgPSBcImluZm9cIikge1xuICAgIGlmICghZWxlbWVudHMuY29ubmVjdGlvbikge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBjbGFzc0xpc3QgPSBlbGVtZW50cy5jb25uZWN0aW9uLmNsYXNzTGlzdDtcbiAgICBBcnJheS5mcm9tKGNsYXNzTGlzdClcbiAgICAgIC5maWx0ZXIoKGNscykgPT4gY2xzLnN0YXJ0c1dpdGgoXCJhbGVydC1cIikgJiYgY2xzICE9PSBcImFsZXJ0XCIpXG4gICAgICAuZm9yRWFjaCgoY2xzKSA9PiBjbGFzc0xpc3QucmVtb3ZlKGNscykpO1xuICAgIGNsYXNzTGlzdC5hZGQoXCJhbGVydFwiKTtcbiAgICBjbGFzc0xpc3QuYWRkKGBhbGVydC0ke3ZhcmlhbnR9YCk7XG4gICAgZWxlbWVudHMuY29ubmVjdGlvbi50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgY2xhc3NMaXN0LnJlbW92ZShcInZpc3VhbGx5LWhpZGRlblwiKTtcbiAgICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBjbGFzc0xpc3QuYWRkKFwidmlzdWFsbHktaGlkZGVuXCIpO1xuICAgIH0sIDQwMDApO1xuICB9XG5cbiAgZnVuY3Rpb24gdXBkYXRlQ29ubmVjdGlvbk1ldGEobWVzc2FnZSwgdG9uZSA9IFwibXV0ZWRcIikge1xuICAgIGlmICghZWxlbWVudHMuY29ubmVjdGlvbk1ldGEpIHJldHVybjtcbiAgICBjb25zdCB0b25lcyA9IFtcIm11dGVkXCIsIFwiaW5mb1wiLCBcInN1Y2Nlc3NcIiwgXCJkYW5nZXJcIiwgXCJ3YXJuaW5nXCJdO1xuICAgIGVsZW1lbnRzLmNvbm5lY3Rpb25NZXRhLnRleHRDb250ZW50ID0gbWVzc2FnZTtcbiAgICB0b25lcy5mb3JFYWNoKCh0KSA9PiBlbGVtZW50cy5jb25uZWN0aW9uTWV0YS5jbGFzc0xpc3QucmVtb3ZlKGB0ZXh0LSR7dH1gKSk7XG4gICAgZWxlbWVudHMuY29ubmVjdGlvbk1ldGEuY2xhc3NMaXN0LmFkZChgdGV4dC0ke3RvbmV9YCk7XG4gIH1cblxuICBmdW5jdGlvbiBzZXRXc1N0YXR1cyhzdGF0ZSwgdGl0bGUpIHtcbiAgICBpZiAoIWVsZW1lbnRzLndzU3RhdHVzKSByZXR1cm47XG4gICAgY29uc3QgbGFiZWwgPSBzdGF0dXNMYWJlbHNbc3RhdGVdIHx8IHN0YXRlO1xuICAgIGVsZW1lbnRzLndzU3RhdHVzLnRleHRDb250ZW50ID0gbGFiZWw7XG4gICAgZWxlbWVudHMud3NTdGF0dXMuY2xhc3NOYW1lID0gYGJhZGdlIHdzLWJhZGdlICR7c3RhdGV9YDtcbiAgICBpZiAodGl0bGUpIHtcbiAgICAgIGVsZW1lbnRzLndzU3RhdHVzLnRpdGxlID0gdGl0bGU7XG4gICAgfSBlbHNlIHtcbiAgICAgIGVsZW1lbnRzLndzU3RhdHVzLnJlbW92ZUF0dHJpYnV0ZShcInRpdGxlXCIpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIG5vcm1hbGl6ZVN0cmluZyhzdHIpIHtcbiAgICBjb25zdCB2YWx1ZSA9IFN0cmluZyhzdHIgfHwgXCJcIik7XG4gICAgdHJ5IHtcbiAgICAgIHJldHVybiB2YWx1ZVxuICAgICAgICAubm9ybWFsaXplKFwiTkZEXCIpXG4gICAgICAgIC5yZXBsYWNlKC9bXFx1MDMwMC1cXHUwMzZmXS9nLCBcIlwiKVxuICAgICAgICAudG9Mb3dlckNhc2UoKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIHJldHVybiB2YWx1ZS50b0xvd2VyQ2FzZSgpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGx5VHJhbnNjcmlwdEZpbHRlcihxdWVyeSwgb3B0aW9ucyA9IHt9KSB7XG4gICAgaWYgKCFlbGVtZW50cy50cmFuc2NyaXB0KSByZXR1cm4gMDtcbiAgICBjb25zdCB7IHByZXNlcnZlSW5wdXQgPSBmYWxzZSB9ID0gb3B0aW9ucztcbiAgICBjb25zdCByYXdRdWVyeSA9IHR5cGVvZiBxdWVyeSA9PT0gXCJzdHJpbmdcIiA/IHF1ZXJ5IDogXCJcIjtcbiAgICBpZiAoIXByZXNlcnZlSW5wdXQgJiYgZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LnZhbHVlID0gcmF3UXVlcnk7XG4gICAgfVxuICAgIGNvbnN0IHRyaW1tZWQgPSByYXdRdWVyeS50cmltKCk7XG4gICAgc3RhdGUuYWN0aXZlRmlsdGVyID0gdHJpbW1lZDtcbiAgICBjb25zdCBub3JtYWxpemVkID0gbm9ybWFsaXplU3RyaW5nKHRyaW1tZWQpO1xuICAgIGxldCBtYXRjaGVzID0gMDtcbiAgICBjb25zdCByb3dzID0gQXJyYXkuZnJvbShlbGVtZW50cy50cmFuc2NyaXB0LnF1ZXJ5U2VsZWN0b3JBbGwoXCIuY2hhdC1yb3dcIikpO1xuICAgIHJvd3MuZm9yRWFjaCgocm93KSA9PiB7XG4gICAgICByb3cuY2xhc3NMaXN0LnJlbW92ZShcImNoYXQtaGlkZGVuXCIsIFwiY2hhdC1maWx0ZXItbWF0Y2hcIik7XG4gICAgICBpZiAoIW5vcm1hbGl6ZWQpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgY29uc3QgcmF3ID0gcm93LmRhdGFzZXQucmF3VGV4dCB8fCBcIlwiO1xuICAgICAgY29uc3Qgbm9ybWFsaXplZFJvdyA9IG5vcm1hbGl6ZVN0cmluZyhyYXcpO1xuICAgICAgaWYgKG5vcm1hbGl6ZWRSb3cuaW5jbHVkZXMobm9ybWFsaXplZCkpIHtcbiAgICAgICAgcm93LmNsYXNzTGlzdC5hZGQoXCJjaGF0LWZpbHRlci1tYXRjaFwiKTtcbiAgICAgICAgbWF0Y2hlcyArPSAxO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcm93LmNsYXNzTGlzdC5hZGQoXCJjaGF0LWhpZGRlblwiKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBlbGVtZW50cy50cmFuc2NyaXB0LmNsYXNzTGlzdC50b2dnbGUoXCJmaWx0ZXJlZFwiLCBCb29sZWFuKHRyaW1tZWQpKTtcbiAgICBpZiAoZWxlbWVudHMuZmlsdGVyRW1wdHkpIHtcbiAgICAgIGlmICh0cmltbWVkICYmIG1hdGNoZXMgPT09IDApIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuY2xhc3NMaXN0LnJlbW92ZShcImQtbm9uZVwiKTtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuc2V0QXR0cmlidXRlKFxuICAgICAgICAgIFwiYXJpYS1saXZlXCIsXG4gICAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuZ2V0QXR0cmlidXRlKFwiYXJpYS1saXZlXCIpIHx8IFwicG9saXRlXCIsXG4gICAgICAgICk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5jbGFzc0xpc3QuYWRkKFwiZC1ub25lXCIpO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAoZWxlbWVudHMuZmlsdGVySGludCkge1xuICAgICAgaWYgKHRyaW1tZWQpIHtcbiAgICAgICAgbGV0IHN1bW1hcnkgPSBcIkF1Y3VuIG1lc3NhZ2UgbmUgY29ycmVzcG9uZC5cIjtcbiAgICAgICAgaWYgKG1hdGNoZXMgPT09IDEpIHtcbiAgICAgICAgICBzdW1tYXJ5ID0gXCIxIG1lc3NhZ2UgY29ycmVzcG9uZC5cIjtcbiAgICAgICAgfSBlbHNlIGlmIChtYXRjaGVzID4gMSkge1xuICAgICAgICAgIHN1bW1hcnkgPSBgJHttYXRjaGVzfSBtZXNzYWdlcyBjb3JyZXNwb25kZW50LmA7XG4gICAgICAgIH1cbiAgICAgICAgZWxlbWVudHMuZmlsdGVySGludC50ZXh0Q29udGVudCA9IHN1bW1hcnk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJIaW50LnRleHRDb250ZW50ID0gZmlsdGVySGludERlZmF1bHQ7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBtYXRjaGVzO1xuICB9XG5cbiAgZnVuY3Rpb24gcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIoKSB7XG4gICAgaWYgKHN0YXRlLmFjdGl2ZUZpbHRlcikge1xuICAgICAgYXBwbHlUcmFuc2NyaXB0RmlsdGVyKHN0YXRlLmFjdGl2ZUZpbHRlciwgeyBwcmVzZXJ2ZUlucHV0OiB0cnVlIH0pO1xuICAgIH0gZWxzZSBpZiAoZWxlbWVudHMudHJhbnNjcmlwdCkge1xuICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5jbGFzc0xpc3QucmVtb3ZlKFwiZmlsdGVyZWRcIik7XG4gICAgICBjb25zdCByb3dzID0gQXJyYXkuZnJvbShcbiAgICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5xdWVyeVNlbGVjdG9yQWxsKFwiLmNoYXQtcm93XCIpLFxuICAgICAgKTtcbiAgICAgIHJvd3MuZm9yRWFjaCgocm93KSA9PiB7XG4gICAgICAgIHJvdy5jbGFzc0xpc3QucmVtb3ZlKFwiY2hhdC1oaWRkZW5cIiwgXCJjaGF0LWZpbHRlci1tYXRjaFwiKTtcbiAgICAgIH0pO1xuICAgICAgaWYgKGVsZW1lbnRzLmZpbHRlckVtcHR5KSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgICBpZiAoZWxlbWVudHMuZmlsdGVySGludCkge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJIaW50LnRleHRDb250ZW50ID0gZmlsdGVySGludERlZmF1bHQ7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gY2xlYXJUcmFuc2NyaXB0RmlsdGVyKGZvY3VzID0gdHJ1ZSkge1xuICAgIHN0YXRlLmFjdGl2ZUZpbHRlciA9IFwiXCI7XG4gICAgaWYgKGVsZW1lbnRzLmZpbHRlcklucHV0KSB7XG4gICAgICBlbGVtZW50cy5maWx0ZXJJbnB1dC52YWx1ZSA9IFwiXCI7XG4gICAgfVxuICAgIHJlYXBwbHlUcmFuc2NyaXB0RmlsdGVyKCk7XG4gICAgaWYgKGZvY3VzICYmIGVsZW1lbnRzLmZpbHRlcklucHV0KSB7XG4gICAgICBlbGVtZW50cy5maWx0ZXJJbnB1dC5mb2N1cygpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHJlbmRlckhpc3RvcnkoZW50cmllcywgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3QgeyByZXBsYWNlID0gZmFsc2UgfSA9IG9wdGlvbnM7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KGVudHJpZXMpIHx8IGVudHJpZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICBpZiAocmVwbGFjZSkge1xuICAgICAgICBlbGVtZW50cy50cmFuc2NyaXB0LmlubmVySFRNTCA9IFwiXCI7XG4gICAgICAgIHN0YXRlLmhpc3RvcnlCb290c3RyYXBwZWQgPSBmYWxzZTtcbiAgICAgICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICAgICAgICB0aW1lbGluZVN0b3JlLmNsZWFyKCk7XG4gICAgICB9XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmIChyZXBsYWNlKSB7XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LmlubmVySFRNTCA9IFwiXCI7XG4gICAgICBzdGF0ZS5oaXN0b3J5Qm9vdHN0cmFwcGVkID0gZmFsc2U7XG4gICAgICBzdGF0ZS5zdHJlYW1Sb3cgPSBudWxsO1xuICAgICAgc3RhdGUuc3RyZWFtQnVmID0gXCJcIjtcbiAgICAgIHRpbWVsaW5lU3RvcmUuY2xlYXIoKTtcbiAgICB9XG4gICAgaWYgKHN0YXRlLmhpc3RvcnlCb290c3RyYXBwZWQgJiYgIXJlcGxhY2UpIHtcbiAgICAgIHN0YXRlLmJvb3RzdHJhcHBpbmcgPSB0cnVlO1xuICAgICAgY29uc3Qgcm93cyA9IEFycmF5LmZyb20oXG4gICAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQucXVlcnlTZWxlY3RvckFsbChcIi5jaGF0LXJvd1wiKSxcbiAgICAgICk7XG4gICAgICByb3dzLmZvckVhY2goKHJvdykgPT4ge1xuICAgICAgICBjb25zdCBleGlzdGluZ0lkID0gcm93LmRhdGFzZXQubWVzc2FnZUlkO1xuICAgICAgICBpZiAoZXhpc3RpbmdJZCAmJiB0aW1lbGluZVN0b3JlLm1hcC5oYXMoZXhpc3RpbmdJZCkpIHtcbiAgICAgICAgICBjb25zdCBjdXJyZW50Um9sZSA9IHJvdy5kYXRhc2V0LnJvbGUgfHwgXCJcIjtcbiAgICAgICAgICBpZiAoY3VycmVudFJvbGUpIHtcbiAgICAgICAgICAgIGRlY29yYXRlUm93KHJvdywgY3VycmVudFJvbGUpO1xuICAgICAgICAgIH1cbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgYnViYmxlID0gcm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgICAgIGNvbnN0IG1ldGEgPSBidWJibGU/LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1tZXRhXCIpIHx8IG51bGw7XG4gICAgICAgIGNvbnN0IHJvbGUgPVxuICAgICAgICAgIHJvdy5kYXRhc2V0LnJvbGUgfHxcbiAgICAgICAgICAocm93LmNsYXNzTGlzdC5jb250YWlucyhcImNoYXQtdXNlclwiKVxuICAgICAgICAgICAgPyBcInVzZXJcIlxuICAgICAgICAgICAgOiByb3cuY2xhc3NMaXN0LmNvbnRhaW5zKFwiY2hhdC1hc3Npc3RhbnRcIilcbiAgICAgICAgICAgICAgPyBcImFzc2lzdGFudFwiXG4gICAgICAgICAgICAgIDogXCJzeXN0ZW1cIik7XG4gICAgICAgIGNvbnN0IHRleHQgPVxuICAgICAgICAgIHJvdy5kYXRhc2V0LnJhd1RleHQgJiYgcm93LmRhdGFzZXQucmF3VGV4dC5sZW5ndGggPiAwXG4gICAgICAgICAgICA/IHJvdy5kYXRhc2V0LnJhd1RleHRcbiAgICAgICAgICAgIDogYnViYmxlXG4gICAgICAgICAgICAgID8gZXh0cmFjdEJ1YmJsZVRleHQoYnViYmxlKVxuICAgICAgICAgICAgICA6IHJvdy50ZXh0Q29udGVudC50cmltKCk7XG4gICAgICAgIGNvbnN0IHRpbWVzdGFtcCA9XG4gICAgICAgICAgcm93LmRhdGFzZXQudGltZXN0YW1wICYmIHJvdy5kYXRhc2V0LnRpbWVzdGFtcC5sZW5ndGggPiAwXG4gICAgICAgICAgICA/IHJvdy5kYXRhc2V0LnRpbWVzdGFtcFxuICAgICAgICAgICAgOiBtZXRhXG4gICAgICAgICAgICAgID8gbWV0YS50ZXh0Q29udGVudC50cmltKClcbiAgICAgICAgICAgICAgOiBub3dJU08oKTtcbiAgICAgICAgY29uc3QgbWVzc2FnZUlkID0gdGltZWxpbmVTdG9yZS5yZWdpc3Rlcih7XG4gICAgICAgICAgaWQ6IGV4aXN0aW5nSWQsXG4gICAgICAgICAgcm9sZSxcbiAgICAgICAgICB0ZXh0LFxuICAgICAgICAgIHRpbWVzdGFtcCxcbiAgICAgICAgICByb3csXG4gICAgICAgIH0pO1xuICAgICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBtZXNzYWdlSWQ7XG4gICAgICAgIHJvdy5kYXRhc2V0LnJvbGUgPSByb2xlO1xuICAgICAgICByb3cuZGF0YXNldC5yYXdUZXh0ID0gdGV4dDtcbiAgICAgICAgcm93LmRhdGFzZXQudGltZXN0YW1wID0gdGltZXN0YW1wO1xuICAgICAgICBkZWNvcmF0ZVJvdyhyb3csIHJvbGUpO1xuICAgICAgfSk7XG4gICAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gZmFsc2U7XG4gICAgICByZWFwcGx5VHJhbnNjcmlwdEZpbHRlcigpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gdHJ1ZTtcbiAgICBlbnRyaWVzXG4gICAgICAuc2xpY2UoKVxuICAgICAgLnJldmVyc2UoKVxuICAgICAgLmZvckVhY2goKGl0ZW0pID0+IHtcbiAgICAgICAgaWYgKGl0ZW0ucXVlcnkpIHtcbiAgICAgICAgICBhcHBlbmRNZXNzYWdlKFwidXNlclwiLCBpdGVtLnF1ZXJ5LCB7XG4gICAgICAgICAgICB0aW1lc3RhbXA6IGl0ZW0udGltZXN0YW1wLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIGlmIChpdGVtLnJlc3BvbnNlKSB7XG4gICAgICAgICAgYXBwZW5kTWVzc2FnZShcImFzc2lzdGFudFwiLCBpdGVtLnJlc3BvbnNlLCB7XG4gICAgICAgICAgICB0aW1lc3RhbXA6IGl0ZW0udGltZXN0YW1wLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gZmFsc2U7XG4gICAgc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCA9IHRydWU7XG4gICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IGZhbHNlIH0pO1xuICAgIGhpZGVTY3JvbGxCdXR0b24oKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHN0YXJ0U3RyZWFtKCkge1xuICAgIHN0YXRlLnN0cmVhbUJ1ZiA9IFwiXCI7XG4gICAgY29uc3QgdHMgPSBub3dJU08oKTtcbiAgICBzdGF0ZS5zdHJlYW1NZXNzYWdlSWQgPSB0aW1lbGluZVN0b3JlLm1ha2VNZXNzYWdlSWQoKTtcbiAgICBzdGF0ZS5zdHJlYW1Sb3cgPSBsaW5lKFxuICAgICAgXCJhc3Npc3RhbnRcIixcbiAgICAgICc8ZGl2IGNsYXNzPVwiY2hhdC1idWJibGVcIj48c3BhbiBjbGFzcz1cImNoYXQtY3Vyc29yXCI+XHUyNThEPC9zcGFuPjwvZGl2PicsXG4gICAgICB7XG4gICAgICAgIHJhd1RleHQ6IFwiXCIsXG4gICAgICAgIHRpbWVzdGFtcDogdHMsXG4gICAgICAgIG1lc3NhZ2VJZDogc3RhdGUuc3RyZWFtTWVzc2FnZUlkLFxuICAgICAgICBtZXRhZGF0YTogeyBzdHJlYW1pbmc6IHRydWUgfSxcbiAgICAgIH0sXG4gICAgKTtcbiAgICBzZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHRzIH0pO1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHNldENvbXBvc2VyU3RhdHVzKFwiUlx1MDBFOXBvbnNlIGVuIGNvdXJzXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzU3RyZWFtaW5nKCkge1xuICAgIHJldHVybiBCb29sZWFuKHN0YXRlLnN0cmVhbVJvdyk7XG4gIH1cblxuICBmdW5jdGlvbiBoYXNTdHJlYW1CdWZmZXIoKSB7XG4gICAgcmV0dXJuIEJvb2xlYW4oc3RhdGUuc3RyZWFtQnVmKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGVuZFN0cmVhbShkZWx0YSkge1xuICAgIGlmICghc3RhdGUuc3RyZWFtUm93KSB7XG4gICAgICBzdGFydFN0cmVhbSgpO1xuICAgIH1cbiAgICBjb25zdCBzaG91bGRTdGljayA9IGlzQXRCb3R0b20oKTtcbiAgICBzdGF0ZS5zdHJlYW1CdWYgKz0gZGVsdGEgfHwgXCJcIjtcbiAgICBjb25zdCBidWJibGUgPSBzdGF0ZS5zdHJlYW1Sb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICBpZiAoYnViYmxlKSB7XG4gICAgICBidWJibGUuaW5uZXJIVE1MID0gYCR7cmVuZGVyTWFya2Rvd24oc3RhdGUuc3RyZWFtQnVmKX08c3BhbiBjbGFzcz1cImNoYXQtY3Vyc29yXCI+XHUyNThEPC9zcGFuPmA7XG4gICAgfVxuICAgIGlmIChzdGF0ZS5zdHJlYW1NZXNzYWdlSWQpIHtcbiAgICAgIHRpbWVsaW5lU3RvcmUudXBkYXRlKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCwge1xuICAgICAgICB0ZXh0OiBzdGF0ZS5zdHJlYW1CdWYsXG4gICAgICAgIG1ldGFkYXRhOiB7IHN0cmVhbWluZzogdHJ1ZSB9LFxuICAgICAgfSk7XG4gICAgfVxuICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogbm93SVNPKCkgfSk7XG4gICAgaWYgKHNob3VsZFN0aWNrKSB7XG4gICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogZmFsc2UgfSk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZW5kU3RyZWFtKGRhdGEpIHtcbiAgICBpZiAoIXN0YXRlLnN0cmVhbVJvdykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBidWJibGUgPSBzdGF0ZS5zdHJlYW1Sb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICBpZiAoYnViYmxlKSB7XG4gICAgICBidWJibGUuaW5uZXJIVE1MID0gcmVuZGVyTWFya2Rvd24oc3RhdGUuc3RyZWFtQnVmKTtcbiAgICAgIGNvbnN0IG1ldGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiZGl2XCIpO1xuICAgICAgbWV0YS5jbGFzc05hbWUgPSBcImNoYXQtbWV0YVwiO1xuICAgICAgY29uc3QgdHMgPSBkYXRhICYmIGRhdGEudGltZXN0YW1wID8gZGF0YS50aW1lc3RhbXAgOiBub3dJU08oKTtcbiAgICAgIG1ldGEudGV4dENvbnRlbnQgPSBmb3JtYXRUaW1lc3RhbXAodHMpO1xuICAgICAgaWYgKGRhdGEgJiYgZGF0YS5lcnJvcikge1xuICAgICAgICBtZXRhLmNsYXNzTGlzdC5hZGQoXCJ0ZXh0LWRhbmdlclwiKTtcbiAgICAgICAgbWV0YS50ZXh0Q29udGVudCA9IGAke21ldGEudGV4dENvbnRlbnR9IFx1MjAyMiAke2RhdGEuZXJyb3J9YDtcbiAgICAgIH1cbiAgICAgIGJ1YmJsZS5hcHBlbmRDaGlsZChtZXRhKTtcbiAgICAgIGRlY29yYXRlUm93KHN0YXRlLnN0cmVhbVJvdywgXCJhc3Npc3RhbnRcIik7XG4gICAgICBoaWdobGlnaHRSb3coc3RhdGUuc3RyZWFtUm93LCBcImFzc2lzdGFudFwiKTtcbiAgICAgIGlmIChpc0F0Qm90dG9tKCkpIHtcbiAgICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IHRydWUgfSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBzaG93U2Nyb2xsQnV0dG9uKCk7XG4gICAgICB9XG4gICAgICBpZiAoc3RhdGUuc3RyZWFtTWVzc2FnZUlkKSB7XG4gICAgICAgIHRpbWVsaW5lU3RvcmUudXBkYXRlKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCwge1xuICAgICAgICAgIHRleHQ6IHN0YXRlLnN0cmVhbUJ1ZixcbiAgICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICAgIG1ldGFkYXRhOiB7XG4gICAgICAgICAgICBzdHJlYW1pbmc6IG51bGwsXG4gICAgICAgICAgICAuLi4oZGF0YSAmJiBkYXRhLmVycm9yID8geyBlcnJvcjogZGF0YS5lcnJvciB9IDogeyBlcnJvcjogbnVsbCB9KSxcbiAgICAgICAgICB9LFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogdHMgfSk7XG4gICAgfVxuICAgIGNvbnN0IGhhc0Vycm9yID0gQm9vbGVhbihkYXRhICYmIGRhdGEuZXJyb3IpO1xuICAgIHNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgaGFzRXJyb3JcbiAgICAgICAgPyBcIlJcdTAwRTlwb25zZSBpbmRpc3BvbmlibGUuIENvbnN1bHRleiBsZXMgam91cm5hdXguXCJcbiAgICAgICAgOiBcIlJcdTAwRTlwb25zZSByZVx1MDBFN3VlLlwiLFxuICAgICAgaGFzRXJyb3IgPyBcImRhbmdlclwiIDogXCJzdWNjZXNzXCIsXG4gICAgKTtcbiAgICBzY2hlZHVsZUNvbXBvc2VySWRsZShoYXNFcnJvciA/IDYwMDAgOiAzNTAwKTtcbiAgICBzdGF0ZS5zdHJlYW1Sb3cgPSBudWxsO1xuICAgIHN0YXRlLnN0cmVhbUJ1ZiA9IFwiXCI7XG4gICAgc3RhdGUuc3RyZWFtTWVzc2FnZUlkID0gbnVsbDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhzdWdnZXN0aW9ucykge1xuICAgIGlmICghZWxlbWVudHMucXVpY2tBY3Rpb25zKSByZXR1cm47XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHN1Z2dlc3Rpb25zKSB8fCBzdWdnZXN0aW9ucy5sZW5ndGggPT09IDApIHJldHVybjtcbiAgICBjb25zdCBidXR0b25zID0gQXJyYXkuZnJvbShcbiAgICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5xdWVyeVNlbGVjdG9yQWxsKFwiYnV0dG9uLnFhXCIpLFxuICAgICk7XG4gICAgY29uc3QgbG9va3VwID0gbmV3IE1hcCgpO1xuICAgIGJ1dHRvbnMuZm9yRWFjaCgoYnRuKSA9PiBsb29rdXAuc2V0KGJ0bi5kYXRhc2V0LmFjdGlvbiwgYnRuKSk7XG4gICAgY29uc3QgZnJhZyA9IGRvY3VtZW50LmNyZWF0ZURvY3VtZW50RnJhZ21lbnQoKTtcbiAgICBzdWdnZXN0aW9ucy5mb3JFYWNoKChrZXkpID0+IHtcbiAgICAgIGlmIChsb29rdXAuaGFzKGtleSkpIHtcbiAgICAgICAgZnJhZy5hcHBlbmRDaGlsZChsb29rdXAuZ2V0KGtleSkpO1xuICAgICAgICBsb29rdXAuZGVsZXRlKGtleSk7XG4gICAgICB9XG4gICAgfSk7XG4gICAgbG9va3VwLmZvckVhY2goKGJ0bikgPT4gZnJhZy5hcHBlbmRDaGlsZChidG4pKTtcbiAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMuaW5uZXJIVE1MID0gXCJcIjtcbiAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMuYXBwZW5kQ2hpbGQoZnJhZyk7XG4gIH1cblxuICBmdW5jdGlvbiBmb3JtYXRQZXJmKGQpIHtcbiAgICBjb25zdCBiaXRzID0gW107XG4gICAgaWYgKGQgJiYgdHlwZW9mIGQuY3B1ICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICBjb25zdCBjcHUgPSBOdW1iZXIoZC5jcHUpO1xuICAgICAgaWYgKCFOdW1iZXIuaXNOYU4oY3B1KSkge1xuICAgICAgICBiaXRzLnB1c2goYENQVSAke2NwdS50b0ZpeGVkKDApfSVgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGQgJiYgdHlwZW9mIGQudHRmYl9tcyAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgY29uc3QgdHRmYiA9IE51bWJlcihkLnR0ZmJfbXMpO1xuICAgICAgaWYgKCFOdW1iZXIuaXNOYU4odHRmYikpIHtcbiAgICAgICAgYml0cy5wdXNoKGBUVEZCICR7dHRmYn0gbXNgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGJpdHMuam9pbihcIiBcdTIwMjIgXCIpIHx8IFwibWlzZSBcdTAwRTAgam91clwiO1xuICB9XG5cbiAgZnVuY3Rpb24gYXR0YWNoRXZlbnRzKCkge1xuICAgIGlmIChlbGVtZW50cy5jb21wb3Nlcikge1xuICAgICAgZWxlbWVudHMuY29tcG9zZXIuYWRkRXZlbnRMaXN0ZW5lcihcInN1Ym1pdFwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgY29uc3QgdGV4dCA9IChlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIikudHJpbSgpO1xuICAgICAgICBlbWl0KFwic3VibWl0XCIsIHsgdGV4dCB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5tb2RlU2VsZWN0KSB7XG4gICAgICBlbGVtZW50cy5tb2RlU2VsZWN0LmFkZEV2ZW50TGlzdGVuZXIoXCJjaGFuZ2VcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGNvbnN0IHZhbHVlID0gZXZlbnQudGFyZ2V0LnZhbHVlIHx8IFwiXCI7XG4gICAgICAgIGNvbnN0IG5leHRNb2RlID0gbm9ybWFsaXNlTW9kZSh2YWx1ZSk7XG4gICAgICAgIHNldE1vZGUobmV4dE1vZGUpO1xuICAgICAgICBlbWl0KFwibW9kZS1jaGFuZ2VcIiwgeyBtb2RlOiBuZXh0TW9kZSB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5xdWlja0FjdGlvbnMpIHtcbiAgICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGNvbnN0IHRhcmdldCA9IGV2ZW50LnRhcmdldDtcbiAgICAgICAgaWYgKCEodGFyZ2V0IGluc3RhbmNlb2YgSFRNTEJ1dHRvbkVsZW1lbnQpKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGFjdGlvbiA9IHRhcmdldC5kYXRhc2V0LmFjdGlvbjtcbiAgICAgICAgaWYgKCFhY3Rpb24pIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgZW1pdChcInF1aWNrLWFjdGlvblwiLCB7IGFjdGlvbiB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQuYWRkRXZlbnRMaXN0ZW5lcihcImlucHV0XCIsIChldmVudCkgPT4ge1xuICAgICAgICBlbWl0KFwiZmlsdGVyLWNoYW5nZVwiLCB7IHZhbHVlOiBldmVudC50YXJnZXQudmFsdWUgfHwgXCJcIiB9KTtcbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQuYWRkRXZlbnRMaXN0ZW5lcihcImtleWRvd25cIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGlmIChldmVudC5rZXkgPT09IFwiRXNjYXBlXCIpIHtcbiAgICAgICAgICBldmVudC5wcmV2ZW50RGVmYXVsdCgpO1xuICAgICAgICAgIGVtaXQoXCJmaWx0ZXItY2xlYXJcIik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJDbGVhcikge1xuICAgICAgZWxlbWVudHMuZmlsdGVyQ2xlYXIuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgZW1pdChcImZpbHRlci1jbGVhclwiKTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5leHBvcnRKc29uKSB7XG4gICAgICBlbGVtZW50cy5leHBvcnRKc29uLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PlxuICAgICAgICBlbWl0KFwiZXhwb3J0XCIsIHsgZm9ybWF0OiBcImpzb25cIiB9KSxcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5leHBvcnRNYXJrZG93bikge1xuICAgICAgZWxlbWVudHMuZXhwb3J0TWFya2Rvd24uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+XG4gICAgICAgIGVtaXQoXCJleHBvcnRcIiwgeyBmb3JtYXQ6IFwibWFya2Rvd25cIiB9KSxcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5leHBvcnRDb3B5KSB7XG4gICAgICBlbGVtZW50cy5leHBvcnRDb3B5LmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiBlbWl0KFwiZXhwb3J0LWNvcHlcIikpO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwiaW5wdXRcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIHVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgICAgYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgICAgY29uc3QgdmFsdWUgPSBldmVudC50YXJnZXQudmFsdWUgfHwgXCJcIjtcbiAgICAgICAgaWYgKCF2YWx1ZS50cmltKCkpIHtcbiAgICAgICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICAgICAgfVxuICAgICAgICBlbWl0KFwicHJvbXB0LWlucHV0XCIsIHsgdmFsdWUgfSk7XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwicGFzdGVcIiwgKCkgPT4ge1xuICAgICAgICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgdXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgICAgICAgIGF1dG9zaXplUHJvbXB0KCk7XG4gICAgICAgICAgZW1pdChcInByb21wdC1pbnB1dFwiLCB7IHZhbHVlOiBlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIiB9KTtcbiAgICAgICAgfSwgMCk7XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgaWYgKChldmVudC5jdHJsS2V5IHx8IGV2ZW50Lm1ldGFLZXkpICYmIGV2ZW50LmtleSA9PT0gXCJFbnRlclwiKSB7XG4gICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICBlbWl0KFwic3VibWl0XCIsIHsgdGV4dDogKGVsZW1lbnRzLnByb21wdC52YWx1ZSB8fCBcIlwiKS50cmltKCkgfSk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMucHJvbXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJmb2N1c1wiLCAoKSA9PiB7XG4gICAgICAgIHNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiUlx1MDBFOWRpZ2V6IHZvdHJlIG1lc3NhZ2UsIHB1aXMgQ3RybCtFbnRyXHUwMEU5ZSBwb3VyIGwnZW52b3llci5cIixcbiAgICAgICAgICBcImluZm9cIixcbiAgICAgICAgKTtcbiAgICAgICAgc2NoZWR1bGVDb21wb3NlcklkbGUoNDAwMCk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudHJhbnNjcmlwdCkge1xuICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5hZGRFdmVudExpc3RlbmVyKFwic2Nyb2xsXCIsICgpID0+IHtcbiAgICAgICAgaWYgKGlzQXRCb3R0b20oKSkge1xuICAgICAgICAgIGhpZGVTY3JvbGxCdXR0b24oKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBzaG93U2Nyb2xsQnV0dG9uKCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5zY3JvbGxCb3R0b20pIHtcbiAgICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogdHJ1ZSB9KTtcbiAgICAgICAgaWYgKGVsZW1lbnRzLnByb21wdCkge1xuICAgICAgICAgIGVsZW1lbnRzLnByb21wdC5mb2N1cygpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcInJlc2l6ZVwiLCAoKSA9PiB7XG4gICAgICBpZiAoaXNBdEJvdHRvbSgpKSB7XG4gICAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiBmYWxzZSB9KTtcbiAgICAgIH1cbiAgICB9KTtcblxuICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMoKTtcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcIm9ubGluZVwiLCAoKSA9PiB7XG4gICAgICB1cGRhdGVOZXR3b3JrU3RhdHVzKCk7XG4gICAgICBhbm5vdW5jZUNvbm5lY3Rpb24oXCJDb25uZXhpb24gclx1MDBFOXNlYXUgcmVzdGF1clx1MDBFOWUuXCIsIFwiaW5mb1wiKTtcbiAgICB9KTtcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcIm9mZmxpbmVcIiwgKCkgPT4ge1xuICAgICAgdXBkYXRlTmV0d29ya1N0YXR1cygpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29ubmV4aW9uIHJcdTAwRTlzZWF1IHBlcmR1ZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgfSk7XG5cbiAgICBjb25zdCB0b2dnbGVCdG4gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcInRvZ2dsZS1kYXJrLW1vZGVcIik7XG4gICAgY29uc3QgZGFya01vZGVLZXkgPSBcImRhcmstbW9kZVwiO1xuXG4gICAgZnVuY3Rpb24gYXBwbHlEYXJrTW9kZShlbmFibGVkKSB7XG4gICAgICBkb2N1bWVudC5ib2R5LmNsYXNzTGlzdC50b2dnbGUoXCJkYXJrLW1vZGVcIiwgZW5hYmxlZCk7XG4gICAgICBpZiAodG9nZ2xlQnRuKSB7XG4gICAgICAgIHRvZ2dsZUJ0bi50ZXh0Q29udGVudCA9IGVuYWJsZWQgPyBcIk1vZGUgQ2xhaXJcIiA6IFwiTW9kZSBTb21icmVcIjtcbiAgICAgICAgdG9nZ2xlQnRuLnNldEF0dHJpYnV0ZShcImFyaWEtcHJlc3NlZFwiLCBlbmFibGVkID8gXCJ0cnVlXCIgOiBcImZhbHNlXCIpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICBhcHBseURhcmtNb2RlKHdpbmRvdy5sb2NhbFN0b3JhZ2UuZ2V0SXRlbShkYXJrTW9kZUtleSkgPT09IFwiMVwiKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byByZWFkIGRhcmsgbW9kZSBwcmVmZXJlbmNlXCIsIGVycik7XG4gICAgfVxuXG4gICAgaWYgKHRvZ2dsZUJ0bikge1xuICAgICAgdG9nZ2xlQnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIGNvbnN0IGVuYWJsZWQgPSAhZG9jdW1lbnQuYm9keS5jbGFzc0xpc3QuY29udGFpbnMoXCJkYXJrLW1vZGVcIik7XG4gICAgICAgIGFwcGx5RGFya01vZGUoZW5hYmxlZCk7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5zZXRJdGVtKGRhcmtNb2RlS2V5LCBlbmFibGVkID8gXCIxXCIgOiBcIjBcIik7XG4gICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byBwZXJzaXN0IGRhcmsgbW9kZSBwcmVmZXJlbmNlXCIsIGVycik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy52b2ljZVRvZ2dsZSkge1xuICAgICAgZWxlbWVudHMudm9pY2VUb2dnbGUuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgZW1pdChcInZvaWNlLXRvZ2dsZVwiKTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy52b2ljZUF1dG9TZW5kKSB7XG4gICAgICBlbGVtZW50cy52b2ljZUF1dG9TZW5kLmFkZEV2ZW50TGlzdGVuZXIoXCJjaGFuZ2VcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGVtaXQoXCJ2b2ljZS1hdXRvc2VuZC1jaGFuZ2VcIiwgeyBlbmFibGVkOiBldmVudC50YXJnZXQuY2hlY2tlZCB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy52b2ljZVBsYXliYWNrKSB7XG4gICAgICBlbGVtZW50cy52b2ljZVBsYXliYWNrLmFkZEV2ZW50TGlzdGVuZXIoXCJjaGFuZ2VcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGVtaXQoXCJ2b2ljZS1wbGF5YmFjay1jaGFuZ2VcIiwgeyBlbmFibGVkOiBldmVudC50YXJnZXQuY2hlY2tlZCB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy52b2ljZVN0b3BQbGF5YmFjaykge1xuICAgICAgZWxlbWVudHMudm9pY2VTdG9wUGxheWJhY2suYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgZW1pdChcInZvaWNlLXN0b3AtcGxheWJhY2tcIik7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdCkge1xuICAgICAgZWxlbWVudHMudm9pY2VWb2ljZVNlbGVjdC5hZGRFdmVudExpc3RlbmVyKFwiY2hhbmdlXCIsIChldmVudCkgPT4ge1xuICAgICAgICBlbWl0KFwidm9pY2Utdm9pY2UtY2hhbmdlXCIsIHsgdm9pY2VVUkk6IGV2ZW50LnRhcmdldC52YWx1ZSB8fCBudWxsIH0pO1xuICAgICAgfSk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gaW5pdGlhbGlzZSgpIHtcbiAgICBzZXRNb2RlKHN0YXRlLm1vZGUsIHsgc2tpcFN0YXR1czogdHJ1ZSB9KTtcbiAgICBzZXREaWFnbm9zdGljcyh7IGNvbm5lY3RlZEF0OiBudWxsLCBsYXN0TWVzc2FnZUF0OiBudWxsLCBsYXRlbmN5TXM6IG51bGwgfSk7XG4gICAgdXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgIGF1dG9zaXplUHJvbXB0KCk7XG4gICAgc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCk7XG4gICAgc2V0Vm9pY2VUcmFuc2NyaXB0KFwiXCIsIHsgc3RhdGU6IFwiaWRsZVwiLCBwbGFjZWhvbGRlcjogXCJcIiB9KTtcbiAgICBhdHRhY2hFdmVudHMoKTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgZWxlbWVudHMsXG4gICAgb24sXG4gICAgZW1pdCxcbiAgICBpbml0aWFsaXNlLFxuICAgIHJlbmRlckhpc3RvcnksXG4gICAgYXBwZW5kTWVzc2FnZSxcbiAgICBzZXRCdXN5LFxuICAgIHNob3dFcnJvcixcbiAgICBoaWRlRXJyb3IsXG4gICAgc2V0Q29tcG9zZXJTdGF0dXMsXG4gICAgc2V0Q29tcG9zZXJTdGF0dXNJZGxlLFxuICAgIHNjaGVkdWxlQ29tcG9zZXJJZGxlLFxuICAgIHVwZGF0ZVByb21wdE1ldHJpY3MsXG4gICAgYXV0b3NpemVQcm9tcHQsXG4gICAgc3RhcnRTdHJlYW0sXG4gICAgYXBwZW5kU3RyZWFtLFxuICAgIGVuZFN0cmVhbSxcbiAgICBhbm5vdW5jZUNvbm5lY3Rpb24sXG4gICAgdXBkYXRlQ29ubmVjdGlvbk1ldGEsXG4gICAgc2V0RGlhZ25vc3RpY3MsXG4gICAgYXBwbHlRdWlja0FjdGlvbk9yZGVyaW5nLFxuICAgIGFwcGx5VHJhbnNjcmlwdEZpbHRlcixcbiAgICByZWFwcGx5VHJhbnNjcmlwdEZpbHRlcixcbiAgICBjbGVhclRyYW5zY3JpcHRGaWx0ZXIsXG4gICAgc2V0V3NTdGF0dXMsXG4gICAgdXBkYXRlTmV0d29ya1N0YXR1cyxcbiAgICBzY3JvbGxUb0JvdHRvbSxcbiAgICBzZXRWb2ljZVN0YXR1cyxcbiAgICBzY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSxcbiAgICBzZXRWb2ljZUF2YWlsYWJpbGl0eSxcbiAgICBzZXRWb2ljZUxpc3RlbmluZyxcbiAgICBzZXRWb2ljZVRyYW5zY3JpcHQsXG4gICAgc2V0Vm9pY2VQcmVmZXJlbmNlcyxcbiAgICBzZXRWb2ljZVNwZWFraW5nLFxuICAgIHNldFZvaWNlVm9pY2VPcHRpb25zLFxuICAgIHNldE1vZGUsXG4gICAgcmVuZGVyRW1iZWRkaW5nRGV0YWlscyhyb3csIGVtYmVkZGluZ0RhdGEsIG1ldGFkYXRhID0ge30pIHtcbiAgICAgIGF0dGFjaEVtYmVkZGluZ0RldGFpbHMocm93LCBlbWJlZGRpbmdEYXRhLCBtZXRhZGF0YSk7XG4gICAgfSxcbiAgICBzZXQgZGlhZ25vc3RpY3ModmFsdWUpIHtcbiAgICAgIE9iamVjdC5hc3NpZ24oZGlhZ25vc3RpY3MsIHZhbHVlKTtcbiAgICB9LFxuICAgIGdldCBkaWFnbm9zdGljcygpIHtcbiAgICAgIHJldHVybiB7IC4uLmRpYWdub3N0aWNzIH07XG4gICAgfSxcbiAgICBnZXQgbW9kZSgpIHtcbiAgICAgIHJldHVybiBzdGF0ZS5tb2RlO1xuICAgIH0sXG4gICAgZm9ybWF0VGltZXN0YW1wLFxuICAgIG5vd0lTTyxcbiAgICBmb3JtYXRQZXJmLFxuICAgIGlzU3RyZWFtaW5nLFxuICAgIGhhc1N0cmVhbUJ1ZmZlcixcbiAgICBnZXQgdm9pY2VTdGF0dXNEZWZhdWx0KCkge1xuICAgICAgcmV0dXJuIHZvaWNlU3RhdHVzRGVmYXVsdDtcbiAgICB9LFxuICB9O1xufVxuIiwgImNvbnN0IERFRkFVTFRfU1RPUkFHRV9LRVkgPSBcIm1vbmdhcnNfand0XCI7XG5cbmZ1bmN0aW9uIGhhc0xvY2FsU3RvcmFnZSgpIHtcbiAgdHJ5IHtcbiAgICByZXR1cm4gdHlwZW9mIHdpbmRvdyAhPT0gXCJ1bmRlZmluZWRcIiAmJiBCb29sZWFuKHdpbmRvdy5sb2NhbFN0b3JhZ2UpO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLndhcm4oXCJBY2Nlc3NpbmcgbG9jYWxTdG9yYWdlIGZhaWxlZFwiLCBlcnIpO1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlQXV0aFNlcnZpY2UoY29uZmlnID0ge30pIHtcbiAgY29uc3Qgc3RvcmFnZUtleSA9IGNvbmZpZy5zdG9yYWdlS2V5IHx8IERFRkFVTFRfU1RPUkFHRV9LRVk7XG4gIGxldCBmYWxsYmFja1Rva2VuID1cbiAgICB0eXBlb2YgY29uZmlnLnRva2VuID09PSBcInN0cmluZ1wiICYmIGNvbmZpZy50b2tlbi50cmltKCkgIT09IFwiXCJcbiAgICAgID8gY29uZmlnLnRva2VuLnRyaW0oKVxuICAgICAgOiB1bmRlZmluZWQ7XG5cbiAgZnVuY3Rpb24gcGVyc2lzdFRva2VuKHRva2VuKSB7XG4gICAgaWYgKHR5cGVvZiB0b2tlbiA9PT0gXCJzdHJpbmdcIikge1xuICAgICAgdG9rZW4gPSB0b2tlbi50cmltKCk7XG4gICAgfVxuICAgIGZhbGxiYWNrVG9rZW4gPSB0b2tlbiB8fCB1bmRlZmluZWQ7XG4gICAgaWYgKCF0b2tlbikge1xuICAgICAgY2xlYXJUb2tlbigpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGlmICghaGFzTG9jYWxTdG9yYWdlKCkpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5zZXRJdGVtKHN0b3JhZ2VLZXksIHRva2VuKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byBwZXJzaXN0IEpXVCBpbiBsb2NhbFN0b3JhZ2VcIiwgZXJyKTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiByZWFkU3RvcmVkVG9rZW4oKSB7XG4gICAgaWYgKCFoYXNMb2NhbFN0b3JhZ2UoKSkge1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgY29uc3Qgc3RvcmVkID0gd2luZG93LmxvY2FsU3RvcmFnZS5nZXRJdGVtKHN0b3JhZ2VLZXkpO1xuICAgICAgcmV0dXJuIHN0b3JlZCB8fCB1bmRlZmluZWQ7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcmVhZCBKV1QgZnJvbSBsb2NhbFN0b3JhZ2VcIiwgZXJyKTtcbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gY2xlYXJUb2tlbigpIHtcbiAgICBmYWxsYmFja1Rva2VuID0gdW5kZWZpbmVkO1xuXG4gICAgaWYgKCFoYXNMb2NhbFN0b3JhZ2UoKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICB3aW5kb3cubG9jYWxTdG9yYWdlLnJlbW92ZUl0ZW0oc3RvcmFnZUtleSk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gY2xlYXIgSldUIGZyb20gbG9jYWxTdG9yYWdlXCIsIGVycik7XG4gICAgfVxuICB9XG5cbiAgaWYgKGZhbGxiYWNrVG9rZW4pIHtcbiAgICBwZXJzaXN0VG9rZW4oZmFsbGJhY2tUb2tlbik7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBnZXRKd3QoKSB7XG4gICAgY29uc3Qgc3RvcmVkID0gcmVhZFN0b3JlZFRva2VuKCk7XG4gICAgaWYgKHN0b3JlZCkge1xuICAgICAgcmV0dXJuIHN0b3JlZDtcbiAgICB9XG4gICAgaWYgKGZhbGxiYWNrVG9rZW4pIHtcbiAgICAgIHJldHVybiBmYWxsYmFja1Rva2VuO1xuICAgIH1cbiAgICB0aHJvdyBuZXcgRXJyb3IoXCJNaXNzaW5nIEpXVCBmb3IgY2hhdCBhdXRoZW50aWNhdGlvbi5cIik7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGdldEp3dCxcbiAgICBwZXJzaXN0VG9rZW4sXG4gICAgY2xlYXJUb2tlbixcbiAgICBzdG9yYWdlS2V5LFxuICB9O1xufVxuIiwgImltcG9ydCB7IGFwaVVybCB9IGZyb20gXCIuLi9jb25maWcuanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUh0dHBTZXJ2aWNlKHsgY29uZmlnLCBhdXRoIH0pIHtcbiAgYXN5bmMgZnVuY3Rpb24gYXV0aG9yaXNlZEZldGNoKHBhdGgsIG9wdGlvbnMgPSB7fSkge1xuICAgIGxldCBqd3Q7XG4gICAgdHJ5IHtcbiAgICAgIGp3dCA9IGF3YWl0IGF1dGguZ2V0Snd0KCk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAvLyBTdXJmYWNlIGEgY29uc2lzdGVudCBlcnJvciBhbmQgcHJlc2VydmUgYWJvcnQgc2VtYW50aWNzXG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXCJBdXRob3JpemF0aW9uIGZhaWxlZDogbWlzc2luZyBvciB1bnJlYWRhYmxlIEpXVFwiKTtcbiAgICB9XG4gICAgY29uc3QgaGVhZGVycyA9IG5ldyBIZWFkZXJzKG9wdGlvbnMuaGVhZGVycyB8fCB7fSk7XG4gICAgaWYgKCFoZWFkZXJzLmhhcyhcIkF1dGhvcml6YXRpb25cIikpIHtcbiAgICAgIGhlYWRlcnMuc2V0KFwiQXV0aG9yaXphdGlvblwiLCBgQmVhcmVyICR7and0fWApO1xuICAgIH1cbiAgICByZXR1cm4gZmV0Y2goYXBpVXJsKGNvbmZpZywgcGF0aCksIHsgLi4ub3B0aW9ucywgaGVhZGVycyB9KTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGZldGNoVGlja2V0KCkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL2F1dGgvd3MvdGlja2V0XCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgfSk7XG4gICAgaWYgKCFyZXNwLm9rKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFRpY2tldCBlcnJvcjogJHtyZXNwLnN0YXR1c31gKTtcbiAgICB9XG4gICAgY29uc3QgYm9keSA9IGF3YWl0IHJlc3AuanNvbigpO1xuICAgIGlmICghYm9keSB8fCAhYm9keS50aWNrZXQpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcIlRpY2tldCByZXNwb25zZSBpbnZhbGlkZVwiKTtcbiAgICB9XG4gICAgcmV0dXJuIGJvZHkudGlja2V0O1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gcG9zdENoYXQobWVzc2FnZSkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL2NvbnZlcnNhdGlvbi9jaGF0XCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgICBoZWFkZXJzOiB7IFwiQ29udGVudC1UeXBlXCI6IFwiYXBwbGljYXRpb24vanNvblwiIH0sXG4gICAgICBib2R5OiBKU09OLnN0cmluZ2lmeSh7IG1lc3NhZ2UgfSksXG4gICAgfSk7XG4gICAgaWYgKCFyZXNwLm9rKSB7XG4gICAgICBjb25zdCBwYXlsb2FkID0gYXdhaXQgcmVzcC50ZXh0KCk7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXNwLnN0YXR1c306ICR7cGF5bG9hZH1gKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3A7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBwb3N0RW1iZWQodGV4dCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgaWYgKCFjb25maWcuZW1iZWRTZXJ2aWNlVXJsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIFwiU2VydmljZSBkJ2VtYmVkZGluZyBpbmRpc3BvbmlibGU6IGF1Y3VuZSBVUkwgY29uZmlndXJcdTAwRTllLlwiXG4gICAgICApO1xuICAgIH1cbiAgICBjb25zdCBwYXlsb2FkID0ge1xuICAgICAgaW5wdXRzOiBbdGV4dF0sXG4gICAgfTtcbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKG9wdGlvbnMsIFwibm9ybWFsaXNlXCIpKSB7XG4gICAgICBwYXlsb2FkLm5vcm1hbGlzZSA9IEJvb2xlYW4ob3B0aW9ucy5ub3JtYWxpc2UpO1xuICAgIH0gZWxzZSB7XG4gICAgICBwYXlsb2FkLm5vcm1hbGlzZSA9IGZhbHNlO1xuICAgIH1cbiAgICBjb25zdCByZXNwID0gYXdhaXQgYXV0aG9yaXNlZEZldGNoKGNvbmZpZy5lbWJlZFNlcnZpY2VVcmwsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgICBoZWFkZXJzOiB7IFwiQ29udGVudC1UeXBlXCI6IFwiYXBwbGljYXRpb24vanNvblwiIH0sXG4gICAgICBib2R5OiBKU09OLnN0cmluZ2lmeShwYXlsb2FkKSxcbiAgICB9KTtcbiAgICBpZiAoIXJlc3Aub2spIHtcbiAgICAgIGNvbnN0IGJvZHlUZXh0ID0gYXdhaXQgcmVzcC50ZXh0KCk7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXNwLnN0YXR1c306ICR7Ym9keVRleHR9YCk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGEgPSBhd2FpdCByZXNwLmpzb24oKTtcbiAgICBpZiAoIWRhdGEgfHwgIUFycmF5LmlzQXJyYXkoZGF0YS52ZWN0b3JzKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFwiRW1iZWRkaW5nIHJlc3BvbnNlIGludmFsaWRlOiB2ZWN0ZXVycyBtYW5xdWFudHNcIik7XG4gICAgfVxuICAgIHJldHVybiBkYXRhO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gcG9zdFN1Z2dlc3Rpb25zKHByb21wdCkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL3VpL3N1Z2dlc3Rpb25zXCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgICBoZWFkZXJzOiB7IFwiQ29udGVudC1UeXBlXCI6IFwiYXBwbGljYXRpb24vanNvblwiIH0sXG4gICAgICBib2R5OiBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgIHByb21wdCxcbiAgICAgICAgYWN0aW9uczogW1wiY29kZVwiLCBcInN1bW1hcml6ZVwiLCBcImV4cGxhaW5cIl0sXG4gICAgICB9KSxcbiAgICB9KTtcbiAgICBpZiAoIXJlc3Aub2spIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgU3VnZ2VzdGlvbiBlcnJvcjogJHtyZXNwLnN0YXR1c31gKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3AuanNvbigpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBmZXRjaFRpY2tldCxcbiAgICBwb3N0Q2hhdCxcbiAgICBwb3N0RW1iZWQsXG4gICAgcG9zdFN1Z2dlc3Rpb25zLFxuICB9O1xufVxuIiwgImltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuLi91dGlscy90aW1lLmpzXCI7XG5cbmZ1bmN0aW9uIGJ1aWxkRXhwb3J0RmlsZW5hbWUoZXh0ZW5zaW9uKSB7XG4gIGNvbnN0IHN0YW1wID0gbm93SVNPKCkucmVwbGFjZSgvWzouXS9nLCBcIi1cIik7XG4gIHJldHVybiBgbW9uZ2Fycy1jaGF0LSR7c3RhbXB9LiR7ZXh0ZW5zaW9ufWA7XG59XG5cbmZ1bmN0aW9uIGJ1aWxkTWFya2Rvd25FeHBvcnQoaXRlbXMpIHtcbiAgY29uc3QgbGluZXMgPSBbXCIjIEhpc3RvcmlxdWUgZGUgY29udmVyc2F0aW9uIG1vbkdBUlNcIiwgXCJcIl07XG4gIGl0ZW1zLmZvckVhY2goKGl0ZW0pID0+IHtcbiAgICBjb25zdCByb2xlID0gaXRlbS5yb2xlID8gaXRlbS5yb2xlLnRvVXBwZXJDYXNlKCkgOiBcIk1FU1NBR0VcIjtcbiAgICBsaW5lcy5wdXNoKGAjIyAke3JvbGV9YCk7XG4gICAgaWYgKGl0ZW0udGltZXN0YW1wKSB7XG4gICAgICBsaW5lcy5wdXNoKGAqSG9yb2RhdGFnZVx1MDBBMDoqICR7aXRlbS50aW1lc3RhbXB9YCk7XG4gICAgfVxuICAgIGlmIChpdGVtLm1ldGFkYXRhICYmIE9iamVjdC5rZXlzKGl0ZW0ubWV0YWRhdGEpLmxlbmd0aCA+IDApIHtcbiAgICAgIE9iamVjdC5lbnRyaWVzKGl0ZW0ubWV0YWRhdGEpLmZvckVhY2goKFtrZXksIHZhbHVlXSkgPT4ge1xuICAgICAgICBsaW5lcy5wdXNoKGAqJHtrZXl9XHUwMEEwOiogJHt2YWx1ZX1gKTtcbiAgICAgIH0pO1xuICAgIH1cbiAgICBsaW5lcy5wdXNoKFwiXCIpO1xuICAgIGxpbmVzLnB1c2goaXRlbS50ZXh0IHx8IFwiXCIpO1xuICAgIGxpbmVzLnB1c2goXCJcIik7XG4gIH0pO1xuICByZXR1cm4gbGluZXMuam9pbihcIlxcblwiKTtcbn1cblxuZnVuY3Rpb24gZG93bmxvYWRCbG9iKGZpbGVuYW1lLCB0ZXh0LCB0eXBlKSB7XG4gIGlmICghd2luZG93LlVSTCB8fCB0eXBlb2Ygd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwgIT09IFwiZnVuY3Rpb25cIikge1xuICAgIGNvbnNvbGUud2FybihcIkJsb2IgZXhwb3J0IHVuc3VwcG9ydGVkIGluIHRoaXMgZW52aXJvbm1lbnRcIik7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbdGV4dF0sIHsgdHlwZSB9KTtcbiAgY29uc3QgdXJsID0gVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTtcbiAgY29uc3QgYW5jaG9yID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImFcIik7XG4gIGFuY2hvci5ocmVmID0gdXJsO1xuICBhbmNob3IuZG93bmxvYWQgPSBmaWxlbmFtZTtcbiAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChhbmNob3IpO1xuICBhbmNob3IuY2xpY2soKTtcbiAgZG9jdW1lbnQuYm9keS5yZW1vdmVDaGlsZChhbmNob3IpO1xuICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiBVUkwucmV2b2tlT2JqZWN0VVJMKHVybCksIDApO1xuICByZXR1cm4gdHJ1ZTtcbn1cblxuYXN5bmMgZnVuY3Rpb24gY29weVRvQ2xpcGJvYXJkKHRleHQpIHtcbiAgaWYgKCF0ZXh0KSByZXR1cm4gZmFsc2U7XG4gIHRyeSB7XG4gICAgaWYgKG5hdmlnYXRvci5jbGlwYm9hcmQgJiYgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQpIHtcbiAgICAgIGF3YWl0IG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KHRleHQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB0ZXh0YXJlYSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJ0ZXh0YXJlYVwiKTtcbiAgICAgIHRleHRhcmVhLnZhbHVlID0gdGV4dDtcbiAgICAgIHRleHRhcmVhLnNldEF0dHJpYnV0ZShcInJlYWRvbmx5XCIsIFwicmVhZG9ubHlcIik7XG4gICAgICB0ZXh0YXJlYS5zdHlsZS5wb3NpdGlvbiA9IFwiYWJzb2x1dGVcIjtcbiAgICAgIHRleHRhcmVhLnN0eWxlLmxlZnQgPSBcIi05OTk5cHhcIjtcbiAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgdGV4dGFyZWEuc2VsZWN0KCk7XG4gICAgICBkb2N1bWVudC5leGVjQ29tbWFuZChcImNvcHlcIik7XG4gICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKHRleHRhcmVhKTtcbiAgICB9XG4gICAgcmV0dXJuIHRydWU7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIGNvbnNvbGUud2FybihcIkNvcHkgY29udmVyc2F0aW9uIGZhaWxlZFwiLCBlcnIpO1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRXhwb3J0ZXIoeyB0aW1lbGluZVN0b3JlLCBhbm5vdW5jZSB9KSB7XG4gIGZ1bmN0aW9uIGNvbGxlY3RUcmFuc2NyaXB0KCkge1xuICAgIHJldHVybiB0aW1lbGluZVN0b3JlLmNvbGxlY3QoKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGV4cG9ydENvbnZlcnNhdGlvbihmb3JtYXQpIHtcbiAgICBjb25zdCBpdGVtcyA9IGNvbGxlY3RUcmFuc2NyaXB0KCk7XG4gICAgaWYgKCFpdGVtcy5sZW5ndGgpIHtcbiAgICAgIGFubm91bmNlKFwiQXVjdW4gbWVzc2FnZSBcdTAwRTAgZXhwb3J0ZXIuXCIsIFwid2FybmluZ1wiKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKGZvcm1hdCA9PT0gXCJqc29uXCIpIHtcbiAgICAgIGNvbnN0IHBheWxvYWQgPSB7XG4gICAgICAgIGV4cG9ydGVkX2F0OiBub3dJU08oKSxcbiAgICAgICAgY291bnQ6IGl0ZW1zLmxlbmd0aCxcbiAgICAgICAgaXRlbXMsXG4gICAgICB9O1xuICAgICAgaWYgKFxuICAgICAgICBkb3dubG9hZEJsb2IoXG4gICAgICAgICAgYnVpbGRFeHBvcnRGaWxlbmFtZShcImpzb25cIiksXG4gICAgICAgICAgSlNPTi5zdHJpbmdpZnkocGF5bG9hZCwgbnVsbCwgMiksXG4gICAgICAgICAgXCJhcHBsaWNhdGlvbi9qc29uXCIsXG4gICAgICAgIClcbiAgICAgICkge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBKU09OIGdcdTAwRTluXHUwMEU5clx1MDBFOS5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYW5ub3VuY2UoXCJFeHBvcnQgbm9uIHN1cHBvcnRcdTAwRTkgZGFucyBjZSBuYXZpZ2F0ZXVyLlwiLCBcImRhbmdlclwiKTtcbiAgICAgIH1cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKGZvcm1hdCA9PT0gXCJtYXJrZG93blwiKSB7XG4gICAgICBpZiAoXG4gICAgICAgIGRvd25sb2FkQmxvYihcbiAgICAgICAgICBidWlsZEV4cG9ydEZpbGVuYW1lKFwibWRcIiksXG4gICAgICAgICAgYnVpbGRNYXJrZG93bkV4cG9ydChpdGVtcyksXG4gICAgICAgICAgXCJ0ZXh0L21hcmtkb3duXCIsXG4gICAgICAgIClcbiAgICAgICkge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBNYXJrZG93biBnXHUwMEU5blx1MDBFOXJcdTAwRTkuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGFubm91bmNlKFwiRXhwb3J0IG5vbiBzdXBwb3J0XHUwMEU5IGRhbnMgY2UgbmF2aWdhdGV1ci5cIiwgXCJkYW5nZXJcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gY29weUNvbnZlcnNhdGlvblRvQ2xpcGJvYXJkKCkge1xuICAgIGNvbnN0IGl0ZW1zID0gY29sbGVjdFRyYW5zY3JpcHQoKTtcbiAgICBpZiAoIWl0ZW1zLmxlbmd0aCkge1xuICAgICAgYW5ub3VuY2UoXCJBdWN1biBtZXNzYWdlIFx1MDBFMCBjb3BpZXIuXCIsIFwid2FybmluZ1wiKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdGV4dCA9IGJ1aWxkTWFya2Rvd25FeHBvcnQoaXRlbXMpO1xuICAgIGlmIChhd2FpdCBjb3B5VG9DbGlwYm9hcmQodGV4dCkpIHtcbiAgICAgIGFubm91bmNlKFwiQ29udmVyc2F0aW9uIGNvcGlcdTAwRTllIGF1IHByZXNzZS1wYXBpZXJzLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgfSBlbHNlIHtcbiAgICAgIGFubm91bmNlKFwiSW1wb3NzaWJsZSBkZSBjb3BpZXIgbGEgY29udmVyc2F0aW9uLlwiLCBcImRhbmdlclwiKTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGV4cG9ydENvbnZlcnNhdGlvbixcbiAgICBjb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgbm93SVNPIH0gZnJvbSBcIi4uL3V0aWxzL3RpbWUuanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVNvY2tldENsaWVudCh7IGNvbmZpZywgaHR0cCwgdWksIG9uRXZlbnQgfSkge1xuICBsZXQgd3M7XG4gIGxldCB3c0hCZWF0O1xuICBsZXQgcmVjb25uZWN0QmFja29mZiA9IDUwMDtcbiAgY29uc3QgQkFDS09GRl9NQVggPSA4MDAwO1xuICBsZXQgcmV0cnlUaW1lciA9IG51bGw7XG4gIGxldCBkaXNwb3NlZCA9IGZhbHNlO1xuXG4gIGZ1bmN0aW9uIGNsZWFySGVhcnRiZWF0KCkge1xuICAgIGlmICh3c0hCZWF0KSB7XG4gICAgICBjbGVhckludGVydmFsKHdzSEJlYXQpO1xuICAgICAgd3NIQmVhdCA9IG51bGw7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gc2NoZWR1bGVSZWNvbm5lY3QoZGVsYXlCYXNlKSB7XG4gICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICByZXR1cm4gMDtcbiAgICB9XG4gICAgY29uc3Qgaml0dGVyID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICogMjUwKTtcbiAgICBjb25zdCBkZWxheSA9IE1hdGgubWluKEJBQ0tPRkZfTUFYLCBkZWxheUJhc2UgKyBqaXR0ZXIpO1xuICAgIGlmIChyZXRyeVRpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQocmV0cnlUaW1lcik7XG4gICAgfVxuICAgIHJldHJ5VGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICByZXRyeVRpbWVyID0gbnVsbDtcbiAgICAgIHJlY29ubmVjdEJhY2tvZmYgPSBNYXRoLm1pbihcbiAgICAgICAgQkFDS09GRl9NQVgsXG4gICAgICAgIE1hdGgubWF4KDUwMCwgcmVjb25uZWN0QmFja29mZiAqIDIpLFxuICAgICAgKTtcbiAgICAgIHZvaWQgb3BlblNvY2tldCgpO1xuICAgIH0sIGRlbGF5KTtcbiAgICByZXR1cm4gZGVsYXk7XG4gIH1cblxuICBmdW5jdGlvbiBzYWZlU2VuZChvYmopIHtcbiAgICB0cnkge1xuICAgICAgaWYgKHdzICYmIHdzLnJlYWR5U3RhdGUgPT09IFdlYlNvY2tldC5PUEVOKSB7XG4gICAgICAgIHdzLnNlbmQoSlNPTi5zdHJpbmdpZnkob2JqKSk7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gc2VuZCBvdmVyIFdlYlNvY2tldFwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIG9wZW5Tb2NrZXQoKSB7XG4gICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiT2J0ZW50aW9uIGRcdTIwMTl1biB0aWNrZXQgZGUgY29ubmV4aW9uXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgICAgIGNvbnN0IHRpY2tldCA9IGF3YWl0IGh0dHAuZmV0Y2hUaWNrZXQoKTtcbiAgICAgIGlmIChkaXNwb3NlZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IHdzVXJsID0gbmV3IFVSTChcIi93cy9jaGF0L1wiLCBjb25maWcuYmFzZVVybCk7XG4gICAgICB3c1VybC5wcm90b2NvbCA9IGNvbmZpZy5iYXNlVXJsLnByb3RvY29sID09PSBcImh0dHBzOlwiID8gXCJ3c3M6XCIgOiBcIndzOlwiO1xuICAgICAgd3NVcmwuc2VhcmNoUGFyYW1zLnNldChcInRcIiwgdGlja2V0KTtcblxuICAgICAgaWYgKHdzKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgd3MuY2xvc2UoKTtcbiAgICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFwiV2ViU29ja2V0IGNsb3NlIGJlZm9yZSByZWNvbm5lY3QgZmFpbGVkXCIsIGVycik7XG4gICAgICAgIH1cbiAgICAgICAgd3MgPSBudWxsO1xuICAgICAgfVxuXG4gICAgICB3cyA9IG5ldyBXZWJTb2NrZXQod3NVcmwudG9TdHJpbmcoKSk7XG4gICAgICB1aS5zZXRXc1N0YXR1cyhcImNvbm5lY3RpbmdcIik7XG4gICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIkNvbm5leGlvbiBhdSBzZXJ2ZXVyXHUyMDI2XCIsIFwiaW5mb1wiKTtcblxuICAgICAgd3Mub25vcGVuID0gKCkgPT4ge1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgaWYgKHJldHJ5VGltZXIpIHtcbiAgICAgICAgICBjbGVhclRpbWVvdXQocmV0cnlUaW1lcik7XG4gICAgICAgICAgcmV0cnlUaW1lciA9IG51bGw7XG4gICAgICAgIH1cbiAgICAgICAgcmVjb25uZWN0QmFja29mZiA9IDUwMDtcbiAgICAgICAgY29uc3QgY29ubmVjdGVkQXQgPSBub3dJU08oKTtcbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJvbmxpbmVcIik7XG4gICAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFxuICAgICAgICAgIGBDb25uZWN0XHUwMEU5IGxlICR7dWkuZm9ybWF0VGltZXN0YW1wKGNvbm5lY3RlZEF0KX1gLFxuICAgICAgICAgIFwic3VjY2Vzc1wiLFxuICAgICAgICApO1xuICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGNvbm5lY3RlZEF0LCBsYXN0TWVzc2FnZUF0OiBjb25uZWN0ZWRBdCB9KTtcbiAgICAgICAgdWkuaGlkZUVycm9yKCk7XG4gICAgICAgIGNsZWFySGVhcnRiZWF0KCk7XG4gICAgICAgIHdzSEJlYXQgPSB3aW5kb3cuc2V0SW50ZXJ2YWwoKCkgPT4ge1xuICAgICAgICAgIHNhZmVTZW5kKHsgdHlwZTogXCJjbGllbnQucGluZ1wiLCB0czogbm93SVNPKCkgfSk7XG4gICAgICAgIH0sIDIwMDAwKTtcbiAgICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJDb25uZWN0XHUwMEU5LiBWb3VzIHBvdXZleiBcdTAwRTljaGFuZ2VyLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgfTtcblxuICAgICAgd3Mub25tZXNzYWdlID0gKGV2dCkgPT4ge1xuICAgICAgICBjb25zdCByZWNlaXZlZEF0ID0gbm93SVNPKCk7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgZXYgPSBKU09OLnBhcnNlKGV2dC5kYXRhKTtcbiAgICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHJlY2VpdmVkQXQgfSk7XG4gICAgICAgICAgb25FdmVudChldik7XG4gICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgIGNvbnNvbGUuZXJyb3IoXCJCYWQgZXZlbnQgcGF5bG9hZFwiLCBlcnIsIGV2dC5kYXRhKTtcbiAgICAgICAgfVxuICAgICAgfTtcblxuICAgICAgd3Mub25jbG9zZSA9ICgpID0+IHtcbiAgICAgICAgY2xlYXJIZWFydGJlYXQoKTtcbiAgICAgICAgd3MgPSBudWxsO1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJvZmZsaW5lXCIpO1xuICAgICAgICB1aS5zZXREaWFnbm9zdGljcyh7IGxhdGVuY3lNczogdW5kZWZpbmVkIH0pO1xuICAgICAgICBjb25zdCBkZWxheSA9IHNjaGVkdWxlUmVjb25uZWN0KHJlY29ubmVjdEJhY2tvZmYpO1xuICAgICAgICBjb25zdCBzZWNvbmRzID0gTWF0aC5tYXgoMSwgTWF0aC5yb3VuZChkZWxheSAvIDEwMDApKTtcbiAgICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEoXG4gICAgICAgICAgYERcdTAwRTljb25uZWN0XHUwMEU5LiBOb3V2ZWxsZSB0ZW50YXRpdmUgZGFucyAke3NlY29uZHN9IHNcdTIwMjZgLFxuICAgICAgICAgIFwid2FybmluZ1wiLFxuICAgICAgICApO1xuICAgICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgICBcIkNvbm5leGlvbiBwZXJkdWUuIFJlY29ubmV4aW9uIGF1dG9tYXRpcXVlXHUyMDI2XCIsXG4gICAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICAgICk7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfTtcblxuICAgICAgd3Mub25lcnJvciA9IChlcnIpID0+IHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIldlYlNvY2tldCBlcnJvclwiLCBlcnIpO1xuICAgICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgdWkuc2V0V3NTdGF0dXMoXCJlcnJvclwiLCBcIkVycmV1ciBXZWJTb2NrZXRcIik7XG4gICAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiRXJyZXVyIFdlYlNvY2tldCBkXHUwMEU5dGVjdFx1MDBFOWUuXCIsIFwiZGFuZ2VyXCIpO1xuICAgICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcIlVuZSBlcnJldXIgclx1MDBFOXNlYXUgZXN0IHN1cnZlbnVlLlwiLCBcImRhbmdlclwiKTtcbiAgICAgICAgdWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNjAwMCk7XG4gICAgICB9O1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS5lcnJvcihlcnIpO1xuICAgICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGNvbnN0IG1lc3NhZ2UgPSBlcnIgaW5zdGFuY2VvZiBFcnJvciA/IGVyci5tZXNzYWdlIDogU3RyaW5nKGVycik7XG4gICAgICB1aS5zZXRXc1N0YXR1cyhcImVycm9yXCIsIG1lc3NhZ2UpO1xuICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEobWVzc2FnZSwgXCJkYW5nZXJcIik7XG4gICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgXCJDb25uZXhpb24gaW5kaXNwb25pYmxlLiBOb3V2ZWwgZXNzYWkgYmllbnRcdTAwRjR0LlwiLFxuICAgICAgICBcImRhbmdlclwiLFxuICAgICAgKTtcbiAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgc2NoZWR1bGVSZWNvbm5lY3QocmVjb25uZWN0QmFja29mZik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZGlzcG9zZSgpIHtcbiAgICBkaXNwb3NlZCA9IHRydWU7XG4gICAgaWYgKHJldHJ5VGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChyZXRyeVRpbWVyKTtcbiAgICAgIHJldHJ5VGltZXIgPSBudWxsO1xuICAgIH1cbiAgICBjbGVhckhlYXJ0YmVhdCgpO1xuICAgIGlmICh3cykge1xuICAgICAgdHJ5IHtcbiAgICAgICAgd3MuY2xvc2UoKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICBjb25zb2xlLndhcm4oXCJXZWJTb2NrZXQgY2xvc2UgZHVyaW5nIGRpc3Bvc2UgZmFpbGVkXCIsIGVycik7XG4gICAgICB9XG4gICAgICB3cyA9IG51bGw7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBvcGVuOiBvcGVuU29ja2V0LFxuICAgIHNlbmQ6IHNhZmVTZW5kLFxuICAgIGRpc3Bvc2UsXG4gIH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlKHsgaHR0cCwgdWkgfSkge1xuICBsZXQgdGltZXIgPSBudWxsO1xuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlKHByb21wdCkge1xuICAgIGlmICh0aW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHRpbWVyKTtcbiAgICB9XG4gICAgdGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCksIDIyMCk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCkge1xuICAgIGlmICghcHJvbXB0IHx8IHByb21wdC50cmltKCkubGVuZ3RoIDwgMykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0cnkge1xuICAgICAgY29uc3QgcGF5bG9hZCA9IGF3YWl0IGh0dHAucG9zdFN1Z2dlc3Rpb25zKHByb21wdC50cmltKCkpO1xuICAgICAgaWYgKHBheWxvYWQgJiYgQXJyYXkuaXNBcnJheShwYXlsb2FkLmFjdGlvbnMpKSB7XG4gICAgICAgIHVpLmFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhwYXlsb2FkLmFjdGlvbnMpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcIkFVSSBzdWdnZXN0aW9uIGZldGNoIGZhaWxlZFwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgc2NoZWR1bGUsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgY3JlYXRlRW1pdHRlciB9IGZyb20gXCIuLi91dGlscy9lbWl0dGVyLmpzXCI7XG5pbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBub3JtYWxpemVUZXh0KHZhbHVlKSB7XG4gIGlmICghdmFsdWUpIHtcbiAgICByZXR1cm4gXCJcIjtcbiAgfVxuICByZXR1cm4gU3RyaW5nKHZhbHVlKS5yZXBsYWNlKC9cXHMrL2csIFwiIFwiKS50cmltKCk7XG59XG5cbmZ1bmN0aW9uIGRlc2NyaWJlUmVjb2duaXRpb25FcnJvcihjb2RlLCBmYWxsYmFjayA9IFwiXCIpIHtcbiAgc3dpdGNoIChjb2RlKSB7XG4gICAgY2FzZSBcIm5vdC1hbGxvd2VkXCI6XG4gICAgY2FzZSBcInNlcnZpY2Utbm90LWFsbG93ZWRcIjpcbiAgICAgIHJldHVybiAoXG4gICAgICAgIFwiQWNjXHUwMEU4cyBhdSBtaWNyb3Bob25lIHJlZnVzXHUwMEU5LiBBdXRvcmlzZXogbGEgZGljdFx1MDBFOWUgdm9jYWxlIGRhbnMgdm90cmUgbmF2aWdhdGV1ci5cIlxuICAgICAgKTtcbiAgICBjYXNlIFwibmV0d29ya1wiOlxuICAgICAgcmV0dXJuIFwiTGEgcmVjb25uYWlzc2FuY2Ugdm9jYWxlIGEgXHUwMEU5dFx1MDBFOSBpbnRlcnJvbXB1ZSBwYXIgdW4gcHJvYmxcdTAwRThtZSByXHUwMEU5c2VhdS5cIjtcbiAgICBjYXNlIFwibm8tc3BlZWNoXCI6XG4gICAgICByZXR1cm4gXCJBdWN1bmUgdm9peCBkXHUwMEU5dGVjdFx1MDBFOWUuIEVzc2F5ZXogZGUgcGFybGVyIHBsdXMgcHJcdTAwRThzIGR1IG1pY3JvLlwiO1xuICAgIGNhc2UgXCJhYm9ydGVkXCI6XG4gICAgICByZXR1cm4gXCJMYSBkaWN0XHUwMEU5ZSBhIFx1MDBFOXRcdTAwRTkgaW50ZXJyb21wdWUuXCI7XG4gICAgY2FzZSBcImF1ZGlvLWNhcHR1cmVcIjpcbiAgICAgIHJldHVybiBcIkF1Y3VuIG1pY3JvcGhvbmUgZGlzcG9uaWJsZS4gVlx1MDBFOXJpZmlleiB2b3RyZSBtYXRcdTAwRTlyaWVsLlwiO1xuICAgIGNhc2UgXCJiYWQtZ3JhbW1hclwiOlxuICAgICAgcmV0dXJuIFwiTGUgc2VydmljZSBkZSBkaWN0XHUwMEU5ZSBhIHJlbmNvbnRyXHUwMEU5IHVuZSBlcnJldXIgZGUgdHJhaXRlbWVudC5cIjtcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGZhbGxiYWNrIHx8IFwiTGEgcmVjb25uYWlzc2FuY2Ugdm9jYWxlIGEgcmVuY29udHJcdTAwRTkgdW5lIGVycmV1ciBpbmF0dGVuZHVlLlwiO1xuICB9XG59XG5cbmZ1bmN0aW9uIG1hcFZvaWNlKHZvaWNlKSB7XG4gIHJldHVybiB7XG4gICAgbmFtZTogdm9pY2UubmFtZSxcbiAgICBsYW5nOiB2b2ljZS5sYW5nLFxuICAgIHZvaWNlVVJJOiB2b2ljZS52b2ljZVVSSSxcbiAgICBkZWZhdWx0OiBCb29sZWFuKHZvaWNlLmRlZmF1bHQpLFxuICAgIGxvY2FsU2VydmljZTogQm9vbGVhbih2b2ljZS5sb2NhbFNlcnZpY2UpLFxuICB9O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3BlZWNoU2VydmljZSh7IGRlZmF1bHRMYW5ndWFnZSB9ID0ge30pIHtcbiAgY29uc3QgZW1pdHRlciA9IGNyZWF0ZUVtaXR0ZXIoKTtcbiAgY29uc3QgZ2xvYmFsU2NvcGUgPSB0eXBlb2Ygd2luZG93ICE9PSBcInVuZGVmaW5lZFwiID8gd2luZG93IDoge307XG4gIGNvbnN0IFJlY29nbml0aW9uQ3RvciA9XG4gICAgZ2xvYmFsU2NvcGUuU3BlZWNoUmVjb2duaXRpb24gfHwgZ2xvYmFsU2NvcGUud2Via2l0U3BlZWNoUmVjb2duaXRpb24gfHwgbnVsbDtcbiAgY29uc3QgcmVjb2duaXRpb25TdXBwb3J0ZWQgPSBCb29sZWFuKFJlY29nbml0aW9uQ3Rvcik7XG4gIGNvbnN0IHN5bnRoZXNpc1N1cHBvcnRlZCA9IEJvb2xlYW4oZ2xvYmFsU2NvcGUuc3BlZWNoU3ludGhlc2lzKTtcbiAgY29uc3Qgc3ludGggPSBzeW50aGVzaXNTdXBwb3J0ZWQgPyBnbG9iYWxTY29wZS5zcGVlY2hTeW50aGVzaXMgOiBudWxsO1xuXG4gIGxldCByZWNvZ25pdGlvbiA9IG51bGw7XG4gIGNvbnN0IG5hdmlnYXRvckxhbmd1YWdlID1cbiAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgPyBuYXZpZ2F0b3IubGFuZ3VhZ2VcbiAgICAgIDogbnVsbDtcbiAgbGV0IHJlY29nbml0aW9uTGFuZyA9XG4gICAgZGVmYXVsdExhbmd1YWdlIHx8IG5hdmlnYXRvckxhbmd1YWdlIHx8IFwiZnItQ0FcIjtcbiAgbGV0IG1hbnVhbFN0b3AgPSBmYWxzZTtcbiAgbGV0IGxpc3RlbmluZyA9IGZhbHNlO1xuICBsZXQgc3BlYWtpbmcgPSBmYWxzZTtcbiAgbGV0IHByZWZlcnJlZFZvaWNlVVJJID0gbnVsbDtcbiAgbGV0IHZvaWNlc0NhY2hlID0gW107XG4gIGxldCBtaWNyb3Bob25lUHJpbWVkID0gZmFsc2U7XG4gIGxldCBtaWNyb3Bob25lUHJpbWluZyA9IG51bGw7XG5cbiAgY29uc3QgdXNlckFnZW50ID1cbiAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci51c2VyQWdlbnRcbiAgICAgID8gbmF2aWdhdG9yLnVzZXJBZ2VudC50b0xvd2VyQ2FzZSgpXG4gICAgICA6IFwiXCI7XG4gIGNvbnN0IHBsYXRmb3JtID1cbiAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci5wbGF0Zm9ybVxuICAgICAgPyBuYXZpZ2F0b3IucGxhdGZvcm0udG9Mb3dlckNhc2UoKVxuICAgICAgOiBcIlwiO1xuICBjb25zdCBtYXhUb3VjaFBvaW50cyA9XG4gICAgdHlwZW9mIG5hdmlnYXRvciAhPT0gXCJ1bmRlZmluZWRcIiAmJiB0eXBlb2YgbmF2aWdhdG9yLm1heFRvdWNoUG9pbnRzID09PSBcIm51bWJlclwiXG4gICAgICA/IG5hdmlnYXRvci5tYXhUb3VjaFBvaW50c1xuICAgICAgOiAwO1xuICBjb25zdCBpc0FwcGxlTW9iaWxlID1cbiAgICAvaXBob25lfGlwYWR8aXBvZC8udGVzdCh1c2VyQWdlbnQpIHx8XG4gICAgKHBsYXRmb3JtID09PSBcIm1hY2ludGVsXCIgJiYgbWF4VG91Y2hQb2ludHMgPiAxKTtcbiAgY29uc3QgaXNTYWZhcmkgPVxuICAgIC9zYWZhcmkvLnRlc3QodXNlckFnZW50KSAmJlxuICAgICEvY3Jpb3N8Znhpb3N8Y2hyb21lfGFuZHJvaWR8ZWRnZXxlZGd8b3ByfG9wZXJhLy50ZXN0KHVzZXJBZ2VudCk7XG5cbiAgZnVuY3Rpb24gcmVxdWlyZXNNaWNyb3Bob25lUHJpbWluZygpIHtcbiAgICBpZiAoIXJlY29nbml0aW9uU3VwcG9ydGVkKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGlmIChtaWNyb3Bob25lUHJpbWVkKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGlmICh0eXBlb2YgbmF2aWdhdG9yID09PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGlmIChcbiAgICAgICFuYXZpZ2F0b3IubWVkaWFEZXZpY2VzIHx8XG4gICAgICB0eXBlb2YgbmF2aWdhdG9yLm1lZGlhRGV2aWNlcy5nZXRVc2VyTWVkaWEgIT09IFwiZnVuY3Rpb25cIlxuICAgICkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgICByZXR1cm4gaXNBcHBsZU1vYmlsZSAmJiBpc1NhZmFyaTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHJlbGVhc2VTdHJlYW0oc3RyZWFtKSB7XG4gICAgaWYgKCFzdHJlYW0pIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdHJhY2tzID0gdHlwZW9mIHN0cmVhbS5nZXRUcmFja3MgPT09IFwiZnVuY3Rpb25cIiA/IHN0cmVhbS5nZXRUcmFja3MoKSA6IFtdO1xuICAgIHRyYWNrcy5mb3JFYWNoKCh0cmFjaykgPT4ge1xuICAgICAgdHJ5IHtcbiAgICAgICAgdHJhY2suc3RvcCgpO1xuICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgIGNvbnNvbGUuZGVidWcoXCJVbmFibGUgdG8gc3RvcCB0cmFja1wiLCBlcnIpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZW5zdXJlTWljcm9waG9uZUFjY2VzcygpIHtcbiAgICBpZiAoIXJlcXVpcmVzTWljcm9waG9uZVByaW1pbmcoKSkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICAgIGlmIChtaWNyb3Bob25lUHJpbWVkKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgaWYgKG1pY3JvcGhvbmVQcmltaW5nKSB7XG4gICAgICB0cnkge1xuICAgICAgICByZXR1cm4gYXdhaXQgbWljcm9waG9uZVByaW1pbmc7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cbiAgICBtaWNyb3Bob25lUHJpbWluZyA9IG5hdmlnYXRvci5tZWRpYURldmljZXNcbiAgICAgIC5nZXRVc2VyTWVkaWEoeyBhdWRpbzogdHJ1ZSB9KVxuICAgICAgLnRoZW4oKHN0cmVhbSkgPT4ge1xuICAgICAgICBtaWNyb3Bob25lUHJpbWVkID0gdHJ1ZTtcbiAgICAgICAgcmVsZWFzZVN0cmVhbShzdHJlYW0pO1xuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgIH0pXG4gICAgICAuY2F0Y2goKGVycikgPT4ge1xuICAgICAgICBlbWl0RXJyb3Ioe1xuICAgICAgICAgIHNvdXJjZTogXCJyZWNvZ25pdGlvblwiLFxuICAgICAgICAgIGNvZGU6IFwicGVybWlzc2lvbi1kZW5pZWRcIixcbiAgICAgICAgICBtZXNzYWdlOlxuICAgICAgICAgICAgXCJBdXRvcmlzYXRpb24gZHUgbWljcm9waG9uZSByZWZ1c1x1MDBFOWUuIEFjdGl2ZXogbCdhY2NcdTAwRThzIGRhbnMgbGVzIHJcdTAwRTlnbGFnZXMgZGUgU2FmYXJpLlwiLFxuICAgICAgICAgIGRldGFpbHM6IGVycixcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH0pXG4gICAgICAuZmluYWxseSgoKSA9PiB7XG4gICAgICAgIG1pY3JvcGhvbmVQcmltaW5nID0gbnVsbDtcbiAgICAgIH0pO1xuICAgIHJldHVybiBtaWNyb3Bob25lUHJpbWluZztcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzUGVybWlzc2lvbkVycm9yKGVycm9yKSB7XG4gICAgaWYgKCFlcnJvcikge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgICBjb25zdCBjb2RlID1cbiAgICAgIHR5cGVvZiBlcnJvciA9PT0gXCJzdHJpbmdcIlxuICAgICAgICA/IGVycm9yXG4gICAgICAgIDogZXJyb3IubmFtZSB8fCBlcnJvci5jb2RlIHx8IGVycm9yLm1lc3NhZ2UgfHwgXCJcIjtcbiAgICBjb25zdCBub3JtYWxpc2VkID0gU3RyaW5nKGNvZGUpLnRvTG93ZXJDYXNlKCk7XG4gICAgcmV0dXJuIFtcbiAgICAgIFwibm90YWxsb3dlZGVycm9yXCIsXG4gICAgICBcIm5vdC1hbGxvd2VkXCIsXG4gICAgICBcInNlcnZpY2Utbm90LWFsbG93ZWRcIixcbiAgICAgIFwic2VjdXJpdHllcnJvclwiLFxuICAgICAgXCJwZXJtaXNzaW9uZGVuaWVkZXJyb3JcIixcbiAgICAgIFwiYWJvcnRlcnJvclwiLFxuICAgIF0uc29tZSgoY2FuZGlkYXRlKSA9PiBub3JtYWxpc2VkLmluY2x1ZGVzKGNhbmRpZGF0ZSkpO1xuICB9XG5cbiAgZnVuY3Rpb24gZW1pdEVycm9yKHBheWxvYWQpIHtcbiAgICBjb25zdCBlbnJpY2hlZCA9IHtcbiAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICAuLi5wYXlsb2FkLFxuICAgIH07XG4gICAgY29uc29sZS5lcnJvcihcIlNwZWVjaCBzZXJ2aWNlIGVycm9yXCIsIGVucmljaGVkKTtcbiAgICBlbWl0dGVyLmVtaXQoXCJlcnJvclwiLCBlbnJpY2hlZCk7XG4gIH1cblxuICBmdW5jdGlvbiBlbnN1cmVSZWNvZ25pdGlvbigpIHtcbiAgICBpZiAoIXJlY29nbml0aW9uU3VwcG9ydGVkKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgaWYgKHJlY29nbml0aW9uKSB7XG4gICAgICByZXR1cm4gcmVjb2duaXRpb247XG4gICAgfVxuICAgIHJlY29nbml0aW9uID0gbmV3IFJlY29nbml0aW9uQ3RvcigpO1xuICAgIHJlY29nbml0aW9uLmxhbmcgPSByZWNvZ25pdGlvbkxhbmc7XG4gICAgcmVjb2duaXRpb24uY29udGludW91cyA9IGZhbHNlO1xuICAgIHJlY29nbml0aW9uLmludGVyaW1SZXN1bHRzID0gdHJ1ZTtcbiAgICByZWNvZ25pdGlvbi5tYXhBbHRlcm5hdGl2ZXMgPSAxO1xuXG4gICAgcmVjb2duaXRpb24ub25zdGFydCA9ICgpID0+IHtcbiAgICAgIGxpc3RlbmluZyA9IHRydWU7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJsaXN0ZW5pbmctY2hhbmdlXCIsIHtcbiAgICAgICAgbGlzdGVuaW5nOiB0cnVlLFxuICAgICAgICByZWFzb246IFwic3RhcnRcIixcbiAgICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIH0pO1xuICAgIH07XG5cbiAgICByZWNvZ25pdGlvbi5vbmVuZCA9ICgpID0+IHtcbiAgICAgIGNvbnN0IHJlYXNvbiA9IG1hbnVhbFN0b3AgPyBcIm1hbnVhbFwiIDogXCJlbmRlZFwiO1xuICAgICAgbGlzdGVuaW5nID0gZmFsc2U7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJsaXN0ZW5pbmctY2hhbmdlXCIsIHtcbiAgICAgICAgbGlzdGVuaW5nOiBmYWxzZSxcbiAgICAgICAgcmVhc29uLFxuICAgICAgICB0aW1lc3RhbXA6IG5vd0lTTygpLFxuICAgICAgfSk7XG4gICAgICBtYW51YWxTdG9wID0gZmFsc2U7XG4gICAgfTtcblxuICAgIHJlY29nbml0aW9uLm9uZXJyb3IgPSAoZXZlbnQpID0+IHtcbiAgICAgIGxpc3RlbmluZyA9IGZhbHNlO1xuICAgICAgY29uc3QgY29kZSA9IGV2ZW50LmVycm9yIHx8IFwidW5rbm93blwiO1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgIGNvZGUsXG4gICAgICAgIG1lc3NhZ2U6IGRlc2NyaWJlUmVjb2duaXRpb25FcnJvcihjb2RlLCBldmVudC5tZXNzYWdlKSxcbiAgICAgICAgZXZlbnQsXG4gICAgICB9KTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcImxpc3RlbmluZy1jaGFuZ2VcIiwge1xuICAgICAgICBsaXN0ZW5pbmc6IGZhbHNlLFxuICAgICAgICByZWFzb246IFwiZXJyb3JcIixcbiAgICAgICAgY29kZSxcbiAgICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIH0pO1xuICAgIH07XG5cbiAgICByZWNvZ25pdGlvbi5vbnJlc3VsdCA9IChldmVudCkgPT4ge1xuICAgICAgaWYgKCFldmVudC5yZXN1bHRzKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGZvciAobGV0IGkgPSBldmVudC5yZXN1bHRJbmRleDsgaSA8IGV2ZW50LnJlc3VsdHMubGVuZ3RoOyBpICs9IDEpIHtcbiAgICAgICAgY29uc3QgcmVzdWx0ID0gZXZlbnQucmVzdWx0c1tpXTtcbiAgICAgICAgaWYgKCFyZXN1bHQgfHwgcmVzdWx0Lmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGFsdGVybmF0aXZlID0gcmVzdWx0WzBdO1xuICAgICAgICBjb25zdCB0cmFuc2NyaXB0ID0gbm9ybWFsaXplVGV4dChhbHRlcm5hdGl2ZT8udHJhbnNjcmlwdCB8fCBcIlwiKTtcbiAgICAgICAgaWYgKCF0cmFuc2NyaXB0KSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgZW1pdHRlci5lbWl0KFwidHJhbnNjcmlwdFwiLCB7XG4gICAgICAgICAgdHJhbnNjcmlwdCxcbiAgICAgICAgICBpc0ZpbmFsOiBCb29sZWFuKHJlc3VsdC5pc0ZpbmFsKSxcbiAgICAgICAgICBjb25maWRlbmNlOlxuICAgICAgICAgICAgdHlwZW9mIGFsdGVybmF0aXZlLmNvbmZpZGVuY2UgPT09IFwibnVtYmVyXCJcbiAgICAgICAgICAgICAgPyBhbHRlcm5hdGl2ZS5jb25maWRlbmNlXG4gICAgICAgICAgICAgIDogbnVsbCxcbiAgICAgICAgICB0aW1lc3RhbXA6IG5vd0lTTygpLFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICB9O1xuXG4gICAgcmVjb2duaXRpb24ub25hdWRpb2VuZCA9ICgpID0+IHtcbiAgICAgIGVtaXR0ZXIuZW1pdChcImF1ZGlvLWVuZFwiLCB7IHRpbWVzdGFtcDogbm93SVNPKCkgfSk7XG4gICAgfTtcblxuICAgIHJlY29nbml0aW9uLm9uc3BlZWNoZW5kID0gKCkgPT4ge1xuICAgICAgZW1pdHRlci5lbWl0KFwic3BlZWNoLWVuZFwiLCB7IHRpbWVzdGFtcDogbm93SVNPKCkgfSk7XG4gICAgfTtcblxuICAgIHJldHVybiByZWNvZ25pdGlvbjtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIHN0YXJ0TGlzdGVuaW5nKG9wdGlvbnMgPSB7fSkge1xuICAgIGlmICghcmVjb2duaXRpb25TdXBwb3J0ZWQpIHtcbiAgICAgIGVtaXRFcnJvcih7XG4gICAgICAgIHNvdXJjZTogXCJyZWNvZ25pdGlvblwiLFxuICAgICAgICBjb2RlOiBcInVuc3VwcG9ydGVkXCIsXG4gICAgICAgIG1lc3NhZ2U6IFwiTGEgZGljdFx1MDBFOWUgdm9jYWxlIG4nZXN0IHBhcyBkaXNwb25pYmxlIHN1ciBjZXQgYXBwYXJlaWwuXCIsXG4gICAgICB9KTtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgY29uc3QgaW5zdGFuY2UgPSBlbnN1cmVSZWNvZ25pdGlvbigpO1xuICAgIGlmICghaW5zdGFuY2UpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgaWYgKGxpc3RlbmluZykge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICAgIG1hbnVhbFN0b3AgPSBmYWxzZTtcbiAgICByZWNvZ25pdGlvbkxhbmcgPSBub3JtYWxpemVUZXh0KG9wdGlvbnMubGFuZ3VhZ2UpIHx8IHJlY29nbml0aW9uTGFuZztcbiAgICBpbnN0YW5jZS5sYW5nID0gcmVjb2duaXRpb25MYW5nO1xuICAgIGluc3RhbmNlLmludGVyaW1SZXN1bHRzID0gb3B0aW9ucy5pbnRlcmltUmVzdWx0cyAhPT0gZmFsc2U7XG4gICAgaW5zdGFuY2UuY29udGludW91cyA9IEJvb2xlYW4ob3B0aW9ucy5jb250aW51b3VzKTtcbiAgICBpbnN0YW5jZS5tYXhBbHRlcm5hdGl2ZXMgPSBvcHRpb25zLm1heEFsdGVybmF0aXZlcyB8fCAxO1xuICAgIGlmIChyZXF1aXJlc01pY3JvcGhvbmVQcmltaW5nKCkpIHtcbiAgICAgIGNvbnN0IGdyYW50ZWQgPSBhd2FpdCBlbnN1cmVNaWNyb3Bob25lQWNjZXNzKCk7XG4gICAgICBpZiAoIWdyYW50ZWQpIHtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cbiAgICB0cnkge1xuICAgICAgaW5zdGFuY2Uuc3RhcnQoKTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgaWYgKHJlcXVpcmVzTWljcm9waG9uZVByaW1pbmcoKSAmJiAhbWljcm9waG9uZVByaW1lZCAmJiBpc1Blcm1pc3Npb25FcnJvcihlcnIpKSB7XG4gICAgICAgIGNvbnN0IGdyYW50ZWQgPSBhd2FpdCBlbnN1cmVNaWNyb3Bob25lQWNjZXNzKCk7XG4gICAgICAgIGlmIChncmFudGVkKSB7XG4gICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgIGluc3RhbmNlLnN0YXJ0KCk7XG4gICAgICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgICAgICB9IGNhdGNoIChyZXRyeUVycikge1xuICAgICAgICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgICAgICAgIGNvZGU6IFwic3RhcnQtZmFpbGVkXCIsXG4gICAgICAgICAgICAgIG1lc3NhZ2U6XG4gICAgICAgICAgICAgICAgcmV0cnlFcnIgJiYgcmV0cnlFcnIubWVzc2FnZVxuICAgICAgICAgICAgICAgICAgPyByZXRyeUVyci5tZXNzYWdlXG4gICAgICAgICAgICAgICAgICA6IFwiSW1wb3NzaWJsZSBkZSBkXHUwMEU5bWFycmVyIGxhIHJlY29ubmFpc3NhbmNlIHZvY2FsZS5cIixcbiAgICAgICAgICAgICAgZGV0YWlsczogcmV0cnlFcnIsXG4gICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGVtaXRFcnJvcih7XG4gICAgICAgIHNvdXJjZTogXCJyZWNvZ25pdGlvblwiLFxuICAgICAgICBjb2RlOiBcInN0YXJ0LWZhaWxlZFwiLFxuICAgICAgICBtZXNzYWdlOlxuICAgICAgICAgIGVyciAmJiBlcnIubWVzc2FnZVxuICAgICAgICAgICAgPyBlcnIubWVzc2FnZVxuICAgICAgICAgICAgOiBcIkltcG9zc2libGUgZGUgZFx1MDBFOW1hcnJlciBsYSByZWNvbm5haXNzYW5jZSB2b2NhbGUuXCIsXG4gICAgICAgIGRldGFpbHM6IGVycixcbiAgICAgIH0pO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHN0b3BMaXN0ZW5pbmcob3B0aW9ucyA9IHt9KSB7XG4gICAgaWYgKCFyZWNvZ25pdGlvbikge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBtYW51YWxTdG9wID0gdHJ1ZTtcbiAgICB0cnkge1xuICAgICAgaWYgKG9wdGlvbnMgJiYgb3B0aW9ucy5hYm9ydCAmJiB0eXBlb2YgcmVjb2duaXRpb24uYWJvcnQgPT09IFwiZnVuY3Rpb25cIikge1xuICAgICAgICByZWNvZ25pdGlvbi5hYm9ydCgpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmVjb2duaXRpb24uc3RvcCgpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInJlY29nbml0aW9uXCIsXG4gICAgICAgIGNvZGU6IFwic3RvcC1mYWlsZWRcIixcbiAgICAgICAgbWVzc2FnZTogXCJBcnJcdTAwRUF0IGRlIGxhIGRpY3RcdTAwRTllIGltcG9zc2libGUuXCIsXG4gICAgICAgIGRldGFpbHM6IGVycixcbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGZpbmRWb2ljZSh1cmkpIHtcbiAgICBpZiAoIXVyaSB8fCAhc3ludGgpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCB2b2ljZXMgPSBzeW50aC5nZXRWb2ljZXMoKTtcbiAgICByZXR1cm4gdm9pY2VzLmZpbmQoKHZvaWNlKSA9PiB2b2ljZS52b2ljZVVSSSA9PT0gdXJpKSB8fCBudWxsO1xuICB9XG5cbiAgZnVuY3Rpb24gcmVmcmVzaFZvaWNlcygpIHtcbiAgICBpZiAoIXN5bnRoKSB7XG4gICAgICByZXR1cm4gW107XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICB2b2ljZXNDYWNoZSA9IHN5bnRoLmdldFZvaWNlcygpO1xuICAgICAgY29uc3QgcGF5bG9hZCA9IHZvaWNlc0NhY2hlLm1hcChtYXBWb2ljZSk7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJ2b2ljZXNcIiwgeyB2b2ljZXM6IHBheWxvYWQgfSk7XG4gICAgICByZXR1cm4gcGF5bG9hZDtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGVtaXRFcnJvcih7XG4gICAgICAgIHNvdXJjZTogXCJzeW50aGVzaXNcIixcbiAgICAgICAgY29kZTogXCJ2b2ljZXMtZmFpbGVkXCIsXG4gICAgICAgIG1lc3NhZ2U6IFwiSW1wb3NzaWJsZSBkZSByXHUwMEU5Y3VwXHUwMEU5cmVyIGxhIGxpc3RlIGRlcyB2b2l4IGRpc3BvbmlibGVzLlwiLFxuICAgICAgICBkZXRhaWxzOiBlcnIsXG4gICAgICB9KTtcbiAgICAgIHJldHVybiBbXTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBzcGVhayh0ZXh0LCBvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIXN5bnRoZXNpc1N1cHBvcnRlZCkge1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInN5bnRoZXNpc1wiLFxuICAgICAgICBjb2RlOiBcInVuc3VwcG9ydGVkXCIsXG4gICAgICAgIG1lc3NhZ2U6IFwiTGEgc3ludGhcdTAwRThzZSB2b2NhbGUgbidlc3QgcGFzIGRpc3BvbmlibGUgc3VyIGNldCBhcHBhcmVpbC5cIixcbiAgICAgIH0pO1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGNvbnN0IGNvbnRlbnQgPSBub3JtYWxpemVUZXh0KHRleHQpO1xuICAgIGlmICghY29udGVudCkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIGlmIChsaXN0ZW5pbmcpIHtcbiAgICAgIHN0b3BMaXN0ZW5pbmcoeyBhYm9ydDogdHJ1ZSB9KTtcbiAgICB9XG4gICAgc3RvcFNwZWFraW5nKCk7XG4gICAgY29uc3QgdXR0ZXJhbmNlID0gbmV3IFNwZWVjaFN5bnRoZXNpc1V0dGVyYW5jZShjb250ZW50KTtcbiAgICB1dHRlcmFuY2UubGFuZyA9IG5vcm1hbGl6ZVRleHQob3B0aW9ucy5sYW5nKSB8fCByZWNvZ25pdGlvbkxhbmc7XG4gICAgY29uc3QgcmF0ZSA9IE51bWJlcihvcHRpb25zLnJhdGUpO1xuICAgIGlmICghTnVtYmVyLmlzTmFOKHJhdGUpICYmIHJhdGUgPiAwKSB7XG4gICAgICB1dHRlcmFuY2UucmF0ZSA9IE1hdGgubWluKHJhdGUsIDIpO1xuICAgIH1cbiAgICBjb25zdCBwaXRjaCA9IE51bWJlcihvcHRpb25zLnBpdGNoKTtcbiAgICBpZiAoIU51bWJlci5pc05hTihwaXRjaCkgJiYgcGl0Y2ggPiAwKSB7XG4gICAgICB1dHRlcmFuY2UucGl0Y2ggPSBNYXRoLm1pbihwaXRjaCwgMik7XG4gICAgfVxuICAgIGNvbnN0IHZvaWNlID1cbiAgICAgIGZpbmRWb2ljZShvcHRpb25zLnZvaWNlVVJJKSB8fCBmaW5kVm9pY2UocHJlZmVycmVkVm9pY2VVUkkpIHx8IG51bGw7XG4gICAgaWYgKHZvaWNlKSB7XG4gICAgICB1dHRlcmFuY2Uudm9pY2UgPSB2b2ljZTtcbiAgICB9XG5cbiAgICB1dHRlcmFuY2Uub25zdGFydCA9ICgpID0+IHtcbiAgICAgIHNwZWFraW5nID0gdHJ1ZTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcInNwZWFraW5nLWNoYW5nZVwiLCB7XG4gICAgICAgIHNwZWFraW5nOiB0cnVlLFxuICAgICAgICB1dHRlcmFuY2UsXG4gICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICB9KTtcbiAgICB9O1xuXG4gICAgdXR0ZXJhbmNlLm9uZW5kID0gKCkgPT4ge1xuICAgICAgc3BlYWtpbmcgPSBmYWxzZTtcbiAgICAgIGVtaXR0ZXIuZW1pdChcInNwZWFraW5nLWNoYW5nZVwiLCB7XG4gICAgICAgIHNwZWFraW5nOiBmYWxzZSxcbiAgICAgICAgdXR0ZXJhbmNlLFxuICAgICAgICB0aW1lc3RhbXA6IG5vd0lTTygpLFxuICAgICAgfSk7XG4gICAgfTtcblxuICAgIHV0dGVyYW5jZS5vbmVycm9yID0gKGV2ZW50KSA9PiB7XG4gICAgICBzcGVha2luZyA9IGZhbHNlO1xuICAgICAgZW1pdEVycm9yKHtcbiAgICAgICAgc291cmNlOiBcInN5bnRoZXNpc1wiLFxuICAgICAgICBjb2RlOiBldmVudC5lcnJvciB8fCBcInVua25vd25cIixcbiAgICAgICAgbWVzc2FnZTpcbiAgICAgICAgICBldmVudCAmJiBldmVudC5tZXNzYWdlXG4gICAgICAgICAgICA/IGV2ZW50Lm1lc3NhZ2VcbiAgICAgICAgICAgIDogXCJMYSBzeW50aFx1MDBFOHNlIHZvY2FsZSBhIHJlbmNvbnRyXHUwMEU5IHVuZSBlcnJldXIuXCIsXG4gICAgICAgIGV2ZW50LFxuICAgICAgfSk7XG4gICAgICBlbWl0dGVyLmVtaXQoXCJzcGVha2luZy1jaGFuZ2VcIiwge1xuICAgICAgICBzcGVha2luZzogZmFsc2UsXG4gICAgICAgIHV0dGVyYW5jZSxcbiAgICAgICAgcmVhc29uOiBcImVycm9yXCIsXG4gICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICB9KTtcbiAgICB9O1xuXG4gICAgc3ludGguc3BlYWsodXR0ZXJhbmNlKTtcbiAgICByZXR1cm4gdXR0ZXJhbmNlO1xuICB9XG5cbiAgZnVuY3Rpb24gc3RvcFNwZWFraW5nKCkge1xuICAgIGlmICghc3ludGhlc2lzU3VwcG9ydGVkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmIChzeW50aC5zcGVha2luZyB8fCBzeW50aC5wZW5kaW5nKSB7XG4gICAgICBzeW50aC5jYW5jZWwoKTtcbiAgICB9XG4gICAgaWYgKHNwZWFraW5nKSB7XG4gICAgICBzcGVha2luZyA9IGZhbHNlO1xuICAgICAgZW1pdHRlci5lbWl0KFwic3BlYWtpbmctY2hhbmdlXCIsIHtcbiAgICAgICAgc3BlYWtpbmc6IGZhbHNlLFxuICAgICAgICByZWFzb246IFwiY2FuY2VsXCIsXG4gICAgICAgIHRpbWVzdGFtcDogbm93SVNPKCksXG4gICAgICB9KTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBzZXRQcmVmZXJyZWRWb2ljZSh1cmkpIHtcbiAgICBwcmVmZXJyZWRWb2ljZVVSSSA9IHVyaSB8fCBudWxsO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0TGFuZ3VhZ2UobGFuZykge1xuICAgIGNvbnN0IG5leHQgPSBub3JtYWxpemVUZXh0KGxhbmcpO1xuICAgIGlmIChuZXh0KSB7XG4gICAgICByZWNvZ25pdGlvbkxhbmcgPSBuZXh0O1xuICAgICAgaWYgKHJlY29nbml0aW9uKSB7XG4gICAgICAgIHJlY29nbml0aW9uLmxhbmcgPSByZWNvZ25pdGlvbkxhbmc7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgaWYgKHN5bnRoZXNpc1N1cHBvcnRlZCkge1xuICAgIHJlZnJlc2hWb2ljZXMoKTtcbiAgICBpZiAoc3ludGguYWRkRXZlbnRMaXN0ZW5lcikge1xuICAgICAgc3ludGguYWRkRXZlbnRMaXN0ZW5lcihcInZvaWNlc2NoYW5nZWRcIiwgcmVmcmVzaFZvaWNlcyk7XG4gICAgfSBlbHNlIGlmIChcIm9udm9pY2VzY2hhbmdlZFwiIGluIHN5bnRoKSB7XG4gICAgICBzeW50aC5vbnZvaWNlc2NoYW5nZWQgPSByZWZyZXNoVm9pY2VzO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgb246IGVtaXR0ZXIub24sXG4gICAgb2ZmOiBlbWl0dGVyLm9mZixcbiAgICBzdGFydExpc3RlbmluZyxcbiAgICBzdG9wTGlzdGVuaW5nLFxuICAgIHNwZWFrLFxuICAgIHN0b3BTcGVha2luZyxcbiAgICBzZXRQcmVmZXJyZWRWb2ljZSxcbiAgICBzZXRMYW5ndWFnZSxcbiAgICByZWZyZXNoVm9pY2VzLFxuICAgIGdldFZvaWNlczogKCkgPT4gdm9pY2VzQ2FjaGUubWFwKG1hcFZvaWNlKSxcbiAgICBnZXRQcmVmZXJyZWRWb2ljZTogKCkgPT4gcHJlZmVycmVkVm9pY2VVUkksXG4gICAgaXNSZWNvZ25pdGlvblN1cHBvcnRlZDogKCkgPT4gcmVjb2duaXRpb25TdXBwb3J0ZWQsXG4gICAgaXNTeW50aGVzaXNTdXBwb3J0ZWQ6ICgpID0+IHN5bnRoZXNpc1N1cHBvcnRlZCxcbiAgfTtcbn1cbiIsICJpbXBvcnQgeyByZXNvbHZlQ29uZmlnIH0gZnJvbSBcIi4vY29uZmlnLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVUaW1lbGluZVN0b3JlIH0gZnJvbSBcIi4vc3RhdGUvdGltZWxpbmVTdG9yZS5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlQ2hhdFVpIH0gZnJvbSBcIi4vdWkvY2hhdFVpLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVBdXRoU2VydmljZSB9IGZyb20gXCIuL3NlcnZpY2VzL2F1dGguanNcIjtcbmltcG9ydCB7IGNyZWF0ZUh0dHBTZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvaHR0cC5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlRXhwb3J0ZXIgfSBmcm9tIFwiLi9zZXJ2aWNlcy9leHBvcnRlci5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlU29ja2V0Q2xpZW50IH0gZnJvbSBcIi4vc2VydmljZXMvc29ja2V0LmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVTdWdnZXN0aW9uU2VydmljZSB9IGZyb20gXCIuL3NlcnZpY2VzL3N1Z2dlc3Rpb25zLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVTcGVlY2hTZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvc3BlZWNoLmpzXCI7XG5pbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi91dGlscy90aW1lLmpzXCI7XG5cbmZ1bmN0aW9uIHF1ZXJ5RWxlbWVudHMoZG9jKSB7XG4gIGNvbnN0IGJ5SWQgPSAoaWQpID0+IGRvYy5nZXRFbGVtZW50QnlJZChpZCk7XG4gIHJldHVybiB7XG4gICAgdHJhbnNjcmlwdDogYnlJZChcInRyYW5zY3JpcHRcIiksXG4gICAgY29tcG9zZXI6IGJ5SWQoXCJjb21wb3NlclwiKSxcbiAgICBwcm9tcHQ6IGJ5SWQoXCJwcm9tcHRcIiksXG4gICAgc2VuZDogYnlJZChcInNlbmRcIiksXG4gICAgbW9kZVNlbGVjdDogYnlJZChcImNoYXQtbW9kZVwiKSxcbiAgICB3c1N0YXR1czogYnlJZChcIndzLXN0YXR1c1wiKSxcbiAgICBxdWlja0FjdGlvbnM6IGJ5SWQoXCJxdWljay1hY3Rpb25zXCIpLFxuICAgIGNvbm5lY3Rpb246IGJ5SWQoXCJjb25uZWN0aW9uXCIpLFxuICAgIGVycm9yQWxlcnQ6IGJ5SWQoXCJlcnJvci1hbGVydFwiKSxcbiAgICBlcnJvck1lc3NhZ2U6IGJ5SWQoXCJlcnJvci1tZXNzYWdlXCIpLFxuICAgIHNjcm9sbEJvdHRvbTogYnlJZChcInNjcm9sbC1ib3R0b21cIiksXG4gICAgY29tcG9zZXJTdGF0dXM6IGJ5SWQoXCJjb21wb3Nlci1zdGF0dXNcIiksXG4gICAgcHJvbXB0Q291bnQ6IGJ5SWQoXCJwcm9tcHQtY291bnRcIiksXG4gICAgY29ubmVjdGlvbk1ldGE6IGJ5SWQoXCJjb25uZWN0aW9uLW1ldGFcIiksXG4gICAgZmlsdGVySW5wdXQ6IGJ5SWQoXCJjaGF0LXNlYXJjaFwiKSxcbiAgICBmaWx0ZXJDbGVhcjogYnlJZChcImNoYXQtc2VhcmNoLWNsZWFyXCIpLFxuICAgIGZpbHRlckVtcHR5OiBieUlkKFwiZmlsdGVyLWVtcHR5XCIpLFxuICAgIGZpbHRlckhpbnQ6IGJ5SWQoXCJjaGF0LXNlYXJjaC1oaW50XCIpLFxuICAgIGV4cG9ydEpzb246IGJ5SWQoXCJleHBvcnQtanNvblwiKSxcbiAgICBleHBvcnRNYXJrZG93bjogYnlJZChcImV4cG9ydC1tYXJrZG93blwiKSxcbiAgICBleHBvcnRDb3B5OiBieUlkKFwiZXhwb3J0LWNvcHlcIiksXG4gICAgZGlhZ0Nvbm5lY3RlZDogYnlJZChcImRpYWctY29ubmVjdGVkXCIpLFxuICAgIGRpYWdMYXN0TWVzc2FnZTogYnlJZChcImRpYWctbGFzdC1tZXNzYWdlXCIpLFxuICAgIGRpYWdMYXRlbmN5OiBieUlkKFwiZGlhZy1sYXRlbmN5XCIpLFxuICAgIGRpYWdOZXR3b3JrOiBieUlkKFwiZGlhZy1uZXR3b3JrXCIpLFxuICAgIHZvaWNlQ29udHJvbHM6IGJ5SWQoXCJ2b2ljZS1jb250cm9sc1wiKSxcbiAgICB2b2ljZVJlY29nbml0aW9uR3JvdXA6IGJ5SWQoXCJ2b2ljZS1yZWNvZ25pdGlvbi1ncm91cFwiKSxcbiAgICB2b2ljZVN5bnRoZXNpc0dyb3VwOiBieUlkKFwidm9pY2Utc3ludGhlc2lzLWdyb3VwXCIpLFxuICAgIHZvaWNlVG9nZ2xlOiBieUlkKFwidm9pY2UtdG9nZ2xlXCIpLFxuICAgIHZvaWNlU3RhdHVzOiBieUlkKFwidm9pY2Utc3RhdHVzXCIpLFxuICAgIHZvaWNlVHJhbnNjcmlwdDogYnlJZChcInZvaWNlLXRyYW5zY3JpcHRcIiksXG4gICAgdm9pY2VBdXRvU2VuZDogYnlJZChcInZvaWNlLWF1dG8tc2VuZFwiKSxcbiAgICB2b2ljZVBsYXliYWNrOiBieUlkKFwidm9pY2UtcGxheWJhY2tcIiksXG4gICAgdm9pY2VTdG9wUGxheWJhY2s6IGJ5SWQoXCJ2b2ljZS1zdG9wLXBsYXliYWNrXCIpLFxuICAgIHZvaWNlVm9pY2VTZWxlY3Q6IGJ5SWQoXCJ2b2ljZS12b2ljZS1zZWxlY3RcIiksXG4gICAgdm9pY2VTcGVha2luZ0luZGljYXRvcjogYnlJZChcInZvaWNlLXNwZWFraW5nLWluZGljYXRvclwiKSxcbiAgfTtcbn1cblxuZnVuY3Rpb24gcmVhZEhpc3RvcnkoZG9jKSB7XG4gIGNvbnN0IGhpc3RvcnlFbGVtZW50ID0gZG9jLmdldEVsZW1lbnRCeUlkKFwiY2hhdC1oaXN0b3J5XCIpO1xuICBpZiAoIWhpc3RvcnlFbGVtZW50KSB7XG4gICAgcmV0dXJuIFtdO1xuICB9XG4gIGNvbnN0IHBheWxvYWQgPSBoaXN0b3J5RWxlbWVudC50ZXh0Q29udGVudCB8fCBcIm51bGxcIjtcbiAgaGlzdG9yeUVsZW1lbnQucmVtb3ZlKCk7XG4gIHRyeSB7XG4gICAgY29uc3QgcGFyc2VkID0gSlNPTi5wYXJzZShwYXlsb2FkKTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheShwYXJzZWQpKSB7XG4gICAgICByZXR1cm4gcGFyc2VkO1xuICAgIH1cbiAgICBpZiAocGFyc2VkICYmIHBhcnNlZC5lcnJvcikge1xuICAgICAgcmV0dXJuIHsgZXJyb3I6IHBhcnNlZC5lcnJvciB9O1xuICAgIH1cbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS5lcnJvcihcIlVuYWJsZSB0byBwYXJzZSBjaGF0IGhpc3RvcnlcIiwgZXJyKTtcbiAgfVxuICByZXR1cm4gW107XG59XG5cbmZ1bmN0aW9uIGVuc3VyZUVsZW1lbnRzKGVsZW1lbnRzKSB7XG4gIHJldHVybiBCb29sZWFuKGVsZW1lbnRzLnRyYW5zY3JpcHQgJiYgZWxlbWVudHMuY29tcG9zZXIgJiYgZWxlbWVudHMucHJvbXB0KTtcbn1cblxuY29uc3QgUVVJQ0tfUFJFU0VUUyA9IHtcbiAgY29kZTogXCJKZSBzb3VoYWl0ZSBcdTAwRTljcmlyZSBkdSBjb2RlXHUyMDI2XCIsXG4gIHN1bW1hcml6ZTogXCJSXHUwMEU5c3VtZSBsYSBkZXJuaVx1MDBFOHJlIGNvbnZlcnNhdGlvbi5cIixcbiAgZXhwbGFpbjogXCJFeHBsaXF1ZSB0YSBkZXJuaVx1MDBFOHJlIHJcdTAwRTlwb25zZSBwbHVzIHNpbXBsZW1lbnQuXCIsXG59O1xuXG5leHBvcnQgY2xhc3MgQ2hhdEFwcCB7XG4gIGNvbnN0cnVjdG9yKGRvYyA9IGRvY3VtZW50LCByYXdDb25maWcgPSB3aW5kb3cuY2hhdENvbmZpZyB8fCB7fSkge1xuICAgIHRoaXMuZG9jID0gZG9jO1xuICAgIHRoaXMuY29uZmlnID0gcmVzb2x2ZUNvbmZpZyhyYXdDb25maWcpO1xuICAgIHRoaXMuZWxlbWVudHMgPSBxdWVyeUVsZW1lbnRzKGRvYyk7XG4gICAgaWYgKCFlbnN1cmVFbGVtZW50cyh0aGlzLmVsZW1lbnRzKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAod2luZG93Lm1hcmtlZCAmJiB0eXBlb2Ygd2luZG93Lm1hcmtlZC5zZXRPcHRpb25zID09PSBcImZ1bmN0aW9uXCIpIHtcbiAgICAgIHdpbmRvdy5tYXJrZWQuc2V0T3B0aW9ucyh7XG4gICAgICAgIGJyZWFrczogdHJ1ZSxcbiAgICAgICAgZ2ZtOiB0cnVlLFxuICAgICAgICBoZWFkZXJJZHM6IGZhbHNlLFxuICAgICAgICBtYW5nbGU6IGZhbHNlLFxuICAgICAgfSk7XG4gICAgfVxuICAgIHRoaXMudGltZWxpbmVTdG9yZSA9IGNyZWF0ZVRpbWVsaW5lU3RvcmUoKTtcbiAgICB0aGlzLnVpID0gY3JlYXRlQ2hhdFVpKHtcbiAgICAgIGVsZW1lbnRzOiB0aGlzLmVsZW1lbnRzLFxuICAgICAgdGltZWxpbmVTdG9yZTogdGhpcy50aW1lbGluZVN0b3JlLFxuICAgIH0pO1xuICAgIHRoaXMubW9kZSA9IHRoaXMudWkubW9kZSB8fCBcImNoYXRcIjtcbiAgICB0aGlzLmF1dGggPSBjcmVhdGVBdXRoU2VydmljZSh0aGlzLmNvbmZpZyk7XG4gICAgdGhpcy5odHRwID0gY3JlYXRlSHR0cFNlcnZpY2UoeyBjb25maWc6IHRoaXMuY29uZmlnLCBhdXRoOiB0aGlzLmF1dGggfSk7XG4gICAgdGhpcy5lbWJlZGRpbmdBdmFpbGFibGUgPSBCb29sZWFuKHRoaXMuY29uZmlnLmVtYmVkU2VydmljZVVybCk7XG4gICAgdGhpcy5lbWJlZE9wdGlvbkxhYmVsID0gbnVsbDtcbiAgICB0aGlzLmNvbmZpZ3VyZU1vZGVBdmFpbGFiaWxpdHkoKTtcbiAgICB0aGlzLmV4cG9ydGVyID0gY3JlYXRlRXhwb3J0ZXIoe1xuICAgICAgdGltZWxpbmVTdG9yZTogdGhpcy50aW1lbGluZVN0b3JlLFxuICAgICAgYW5ub3VuY2U6IChtZXNzYWdlLCB2YXJpYW50KSA9PlxuICAgICAgICB0aGlzLnVpLmFubm91bmNlQ29ubmVjdGlvbihtZXNzYWdlLCB2YXJpYW50KSxcbiAgICB9KTtcbiAgICB0aGlzLnN1Z2dlc3Rpb25zID0gY3JlYXRlU3VnZ2VzdGlvblNlcnZpY2Uoe1xuICAgICAgaHR0cDogdGhpcy5odHRwLFxuICAgICAgdWk6IHRoaXMudWksXG4gICAgfSk7XG4gICAgdGhpcy5zb2NrZXQgPSBjcmVhdGVTb2NrZXRDbGllbnQoe1xuICAgICAgY29uZmlnOiB0aGlzLmNvbmZpZyxcbiAgICAgIGh0dHA6IHRoaXMuaHR0cCxcbiAgICAgIHVpOiB0aGlzLnVpLFxuICAgICAgb25FdmVudDogKGV2KSA9PiB0aGlzLmhhbmRsZVNvY2tldEV2ZW50KGV2KSxcbiAgICB9KTtcblxuICAgIHRoaXMuc2V0dXBWb2ljZUZlYXR1cmVzKCk7XG5cbiAgICBjb25zdCBoaXN0b3J5UGF5bG9hZCA9IHJlYWRIaXN0b3J5KGRvYyk7XG4gICAgaWYgKGhpc3RvcnlQYXlsb2FkICYmIGhpc3RvcnlQYXlsb2FkLmVycm9yKSB7XG4gICAgICB0aGlzLnVpLnNob3dFcnJvcihoaXN0b3J5UGF5bG9hZC5lcnJvcik7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGhpc3RvcnlQYXlsb2FkKSkge1xuICAgICAgdGhpcy51aS5yZW5kZXJIaXN0b3J5KGhpc3RvcnlQYXlsb2FkKTtcbiAgICB9XG5cbiAgICB0aGlzLnJlZ2lzdGVyVWlIYW5kbGVycygpO1xuICAgIHRoaXMudWkuaW5pdGlhbGlzZSgpO1xuICAgIHRoaXMuc29ja2V0Lm9wZW4oKTtcbiAgfVxuXG4gIGNvbmZpZ3VyZU1vZGVBdmFpbGFiaWxpdHkoKSB7XG4gICAgY29uc3Qgc2VsZWN0ID0gdGhpcy5lbGVtZW50cy5tb2RlU2VsZWN0O1xuICAgIGlmICghc2VsZWN0KSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IG9wdGlvbiA9IHNlbGVjdC5xdWVyeVNlbGVjdG9yKCdvcHRpb25bdmFsdWU9XCJlbWJlZFwiXScpO1xuICAgIGlmICghb3B0aW9uKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICghdGhpcy5lbWJlZE9wdGlvbkxhYmVsKSB7XG4gICAgICB0aGlzLmVtYmVkT3B0aW9uTGFiZWwgPSBvcHRpb24udGV4dENvbnRlbnQudHJpbSgpIHx8IFwiRW1iZWRkaW5nXCI7XG4gICAgfVxuICAgIGlmICh0aGlzLmVtYmVkZGluZ0F2YWlsYWJsZSkge1xuICAgICAgb3B0aW9uLmRpc2FibGVkID0gZmFsc2U7XG4gICAgICBvcHRpb24ucmVtb3ZlQXR0cmlidXRlKFwiYXJpYS1kaXNhYmxlZFwiKTtcbiAgICAgIG9wdGlvbi50ZXh0Q29udGVudCA9IHRoaXMuZW1iZWRPcHRpb25MYWJlbDtcbiAgICB9IGVsc2Uge1xuICAgICAgb3B0aW9uLmRpc2FibGVkID0gdHJ1ZTtcbiAgICAgIG9wdGlvbi5zZXRBdHRyaWJ1dGUoXCJhcmlhLWRpc2FibGVkXCIsIFwidHJ1ZVwiKTtcbiAgICAgIG9wdGlvbi50ZXh0Q29udGVudCA9IGAke3RoaXMuZW1iZWRPcHRpb25MYWJlbH0gKGluZGlzcG9uaWJsZSlgO1xuICAgICAgaWYgKHNlbGVjdC52YWx1ZSA9PT0gXCJlbWJlZFwiKSB7XG4gICAgICAgIHNlbGVjdC52YWx1ZSA9IFwiY2hhdFwiO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMudWkgJiYgdHlwZW9mIHRoaXMudWkuc2V0TW9kZSA9PT0gXCJmdW5jdGlvblwiKSB7XG4gICAgICAgIHRoaXMudWkuc2V0TW9kZShcImNoYXRcIiwgeyBmb3JjZVN0YXR1czogdHJ1ZSB9KTtcbiAgICAgIH1cbiAgICAgIHRoaXMubW9kZSA9IFwiY2hhdFwiO1xuICAgIH1cbiAgfVxuXG4gIHJlZ2lzdGVyVWlIYW5kbGVycygpIHtcbiAgICB0aGlzLnVpLm9uKFwic3VibWl0XCIsIGFzeW5jICh7IHRleHQgfSkgPT4ge1xuICAgICAgY29uc3QgdmFsdWUgPSAodGV4dCB8fCBcIlwiKS50cmltKCk7XG4gICAgICBjb25zdCByZXF1ZXN0TW9kZSA9IHRoaXMubW9kZSA9PT0gXCJlbWJlZFwiID8gXCJlbWJlZFwiIDogXCJjaGF0XCI7XG4gICAgICBpZiAoIXZhbHVlKSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJTYWlzaXNzZXogdW4gbWVzc2FnZSBhdmFudCBkXHUyMDE5ZW52b3llci5cIixcbiAgICAgICAgICBcIndhcm5pbmdcIixcbiAgICAgICAgKTtcbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgaWYgKHJlcXVlc3RNb2RlID09PSBcImVtYmVkXCIgJiYgIXRoaXMuZW1iZWRkaW5nQXZhaWxhYmxlKSB7XG4gICAgICAgIHRoaXMudWkuc2V0TW9kZShcImNoYXRcIiwgeyBmb3JjZVN0YXR1czogdHJ1ZSB9KTtcbiAgICAgICAgaWYgKHRoaXMuZWxlbWVudHMubW9kZVNlbGVjdCkge1xuICAgICAgICAgIHRoaXMuZWxlbWVudHMubW9kZVNlbGVjdC52YWx1ZSA9IFwiY2hhdFwiO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMubW9kZSA9IFwiY2hhdFwiO1xuICAgICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiU2VydmljZSBkJ2VtYmVkZGluZyBpbmRpc3BvbmlibGUuIE1vZGUgQ2hhdCByXHUwMEU5dGFibGkuXCIsXG4gICAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICAgICk7XG4gICAgICAgIHRoaXMudWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNTAwMCk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMudWkuaGlkZUVycm9yKCk7XG4gICAgICBjb25zdCBzdWJtaXR0ZWRBdCA9IG5vd0lTTygpO1xuICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwidXNlclwiLCB2YWx1ZSwge1xuICAgICAgICB0aW1lc3RhbXA6IHN1Ym1pdHRlZEF0LFxuICAgICAgICBtZXRhZGF0YTogeyBzdWJtaXR0ZWQ6IHRydWUsIG1vZGU6IHJlcXVlc3RNb2RlIH0sXG4gICAgICB9KTtcbiAgICAgIGlmICh0aGlzLmVsZW1lbnRzLnByb21wdCkge1xuICAgICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC52YWx1ZSA9IFwiXCI7XG4gICAgICB9XG4gICAgICB0aGlzLnVpLnVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgIGlmIChyZXF1ZXN0TW9kZSA9PT0gXCJlbWJlZFwiKSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJDYWxjdWwgZGUgbCdlbWJlZGRpbmdcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy51aS5zZXRDb21wb3NlclN0YXR1cyhcIk1lc3NhZ2UgZW52b3lcdTAwRTlcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgfVxuICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIHRoaXMudWkuc2V0QnVzeSh0cnVlKTtcbiAgICAgIGlmIChyZXF1ZXN0TW9kZSA9PT0gXCJjaGF0XCIpIHtcbiAgICAgICAgdGhpcy51aS5hcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcoW1wiY29kZVwiLCBcInN1bW1hcml6ZVwiLCBcImV4cGxhaW5cIl0pO1xuICAgICAgfVxuXG4gICAgICB0cnkge1xuICAgICAgICBpZiAocmVxdWVzdE1vZGUgPT09IFwiZW1iZWRcIikge1xuICAgICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgdGhpcy5odHRwLnBvc3RFbWJlZCh2YWx1ZSk7XG4gICAgICAgICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICAgICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC5mb2N1cygpO1xuICAgICAgICAgIH1cbiAgICAgICAgICB0aGlzLnVpLnNldEJ1c3koZmFsc2UpO1xuICAgICAgICAgIHRoaXMucHJlc2VudEVtYmVkZGluZ1Jlc3VsdChyZXNwb25zZSk7XG4gICAgICAgICAgdGhpcy51aS5zZXRDb21wb3NlclN0YXR1cyhcIlZlY3RldXIgZ1x1MDBFOW5cdTAwRTlyXHUwMEU5LlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBhd2FpdCB0aGlzLmh0dHAucG9zdENoYXQodmFsdWUpO1xuICAgICAgICAgIGlmICh0aGlzLmVsZW1lbnRzLnByb21wdCkge1xuICAgICAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQuZm9jdXMoKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdGhpcy51aS5zdGFydFN0cmVhbSgpO1xuICAgICAgICB9XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgdGhpcy51aS5zaG93RXJyb3IoZXJyLCB7XG4gICAgICAgICAgbWV0YWRhdGE6IHsgc3RhZ2U6IFwic3VibWl0XCIsIG1vZGU6IHJlcXVlc3RNb2RlIH0sXG4gICAgICAgIH0pO1xuICAgICAgICBpZiAocmVxdWVzdE1vZGUgPT09IFwiZW1iZWRcIikge1xuICAgICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgICBcIkdcdTAwRTluXHUwMEU5cmF0aW9uIGQnZW1iZWRkaW5nIGltcG9zc2libGUuIFZcdTAwRTlyaWZpZXogbGEgY29ubmV4aW9uLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgICBcIkVudm9pIGltcG9zc2libGUuIFZcdTAwRTlyaWZpZXogbGEgY29ubmV4aW9uLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNjAwMCk7XG4gICAgICB9XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwibW9kZS1jaGFuZ2VcIiwgKHsgbW9kZSB9KSA9PiB7XG4gICAgICBjb25zdCByZXF1ZXN0ZWRNb2RlID0gbW9kZSA9PT0gXCJlbWJlZFwiID8gXCJlbWJlZFwiIDogXCJjaGF0XCI7XG4gICAgICBpZiAocmVxdWVzdGVkTW9kZSA9PT0gXCJlbWJlZFwiICYmICF0aGlzLmVtYmVkZGluZ0F2YWlsYWJsZSkge1xuICAgICAgICB0aGlzLmNvbmZpZ3VyZU1vZGVBdmFpbGFiaWxpdHkoKTtcbiAgICAgICAgdGhpcy51aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgICBcIlNlcnZpY2UgZCdlbWJlZGRpbmcgaW5kaXNwb25pYmxlLiBNb2RlIENoYXQgclx1MDBFOXRhYmxpLlwiLFxuICAgICAgICAgIFwid2FybmluZ1wiLFxuICAgICAgICApO1xuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDUwMDApO1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5tb2RlID09PSByZXF1ZXN0ZWRNb2RlKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMubW9kZSA9IHJlcXVlc3RlZE1vZGU7XG4gICAgICB0aGlzLnVpLnNldE1vZGUocmVxdWVzdGVkTW9kZSk7XG4gICAgICBpZiAocmVxdWVzdGVkTW9kZSA9PT0gXCJlbWJlZFwiKSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJNb2RlIEVtYmVkZGluZyBhY3Rpdlx1MDBFOS4gTGVzIHJlcXVcdTAwRUF0ZXMgcmVudm9pZW50IGRlcyB2ZWN0ZXVycy5cIixcbiAgICAgICAgICBcImluZm9cIixcbiAgICAgICAgKTtcbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg1MDAwKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJNb2RlIENoYXQgYWN0aXZcdTAwRTkuIExlcyByXHUwMEU5cG9uc2VzIHNlcm9udCBnXHUwMEU5blx1MDBFOXJcdTAwRTllcyBwYXIgbGUgTExNLlwiLFxuICAgICAgICAgIFwiaW5mb1wiLFxuICAgICAgICApO1xuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgfVxuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcInF1aWNrLWFjdGlvblwiLCAoeyBhY3Rpb24gfSkgPT4ge1xuICAgICAgaWYgKCFhY3Rpb24pIHJldHVybjtcbiAgICAgIGNvbnN0IHByZXNldCA9IFFVSUNLX1BSRVNFVFNbYWN0aW9uXSB8fCBhY3Rpb247XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSBwcmVzZXQ7XG4gICAgICB9XG4gICAgICB0aGlzLnVpLnVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJTdWdnZXN0aW9uIGVudm95XHUwMEU5ZVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgdGhpcy51aS5lbWl0KFwic3VibWl0XCIsIHsgdGV4dDogcHJlc2V0IH0pO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcImZpbHRlci1jaGFuZ2VcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgdGhpcy51aS5hcHBseVRyYW5zY3JpcHRGaWx0ZXIodmFsdWUsIHsgcHJlc2VydmVJbnB1dDogdHJ1ZSB9KTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJmaWx0ZXItY2xlYXJcIiwgKCkgPT4ge1xuICAgICAgdGhpcy51aS5jbGVhclRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJleHBvcnRcIiwgKHsgZm9ybWF0IH0pID0+IHtcbiAgICAgIHRoaXMuZXhwb3J0ZXIuZXhwb3J0Q29udmVyc2F0aW9uKGZvcm1hdCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwiZXhwb3J0LWNvcHlcIiwgKCkgPT4ge1xuICAgICAgdGhpcy5leHBvcnRlci5jb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJ2b2ljZS10b2dnbGVcIiwgKCkgPT4ge1xuICAgICAgdGhpcy50b2dnbGVWb2ljZUxpc3RlbmluZygpLmNhdGNoKChlcnIpID0+IHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIlZvaWNlIHRvZ2dsZSBmYWlsZWRcIiwgZXJyKTtcbiAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcInZvaWNlLWF1dG9zZW5kLWNoYW5nZVwiLCAoeyBlbmFibGVkIH0pID0+IHtcbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VBdXRvU2VuZENoYW5nZShCb29sZWFuKGVuYWJsZWQpKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJ2b2ljZS1wbGF5YmFjay1jaGFuZ2VcIiwgKHsgZW5hYmxlZCB9KSA9PiB7XG4gICAgICB0aGlzLmhhbmRsZVZvaWNlUGxheWJhY2tDaGFuZ2UoQm9vbGVhbihlbmFibGVkKSk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwidm9pY2Utc3RvcC1wbGF5YmFja1wiLCAoKSA9PiB7XG4gICAgICB0aGlzLnN0b3BWb2ljZVBsYXliYWNrKCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwidm9pY2Utdm9pY2UtY2hhbmdlXCIsICh7IHZvaWNlVVJJIH0pID0+IHtcbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VWb2ljZUNoYW5nZSh2b2ljZVVSSSB8fCBudWxsKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJwcm9tcHQtaW5wdXRcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgaWYgKCF2YWx1ZSB8fCAhdmFsdWUudHJpbSgpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmVsZW1lbnRzLnNlbmQgJiYgdGhpcy5lbGVtZW50cy5zZW5kLmRpc2FibGVkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMuc3VnZ2VzdGlvbnMuc2NoZWR1bGUodmFsdWUpO1xuICAgIH0pO1xuICB9XG5cbiAgbG9hZFZvaWNlUHJlZmVyZW5jZXMoZGVmYXVsdExhbmd1YWdlKSB7XG4gICAgY29uc3QgZmFsbGJhY2sgPSB7XG4gICAgICBhdXRvU2VuZDogdHJ1ZSxcbiAgICAgIHBsYXliYWNrOiB0cnVlLFxuICAgICAgdm9pY2VVUkk6IG51bGwsXG4gICAgICBsYW5ndWFnZTogZGVmYXVsdExhbmd1YWdlLFxuICAgIH07XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHJhdyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2UuZ2V0SXRlbShcImNoYXQudm9pY2VcIik7XG4gICAgICBpZiAoIXJhdykge1xuICAgICAgICByZXR1cm4gZmFsbGJhY2s7XG4gICAgICB9XG4gICAgICBjb25zdCBwYXJzZWQgPSBKU09OLnBhcnNlKHJhdyk7XG4gICAgICBpZiAoIXBhcnNlZCB8fCB0eXBlb2YgcGFyc2VkICE9PSBcIm9iamVjdFwiKSB7XG4gICAgICAgIHJldHVybiBmYWxsYmFjaztcbiAgICAgIH1cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGF1dG9TZW5kOlxuICAgICAgICAgIHR5cGVvZiBwYXJzZWQuYXV0b1NlbmQgPT09IFwiYm9vbGVhblwiXG4gICAgICAgICAgICA/IHBhcnNlZC5hdXRvU2VuZFxuICAgICAgICAgICAgOiBmYWxsYmFjay5hdXRvU2VuZCxcbiAgICAgICAgcGxheWJhY2s6XG4gICAgICAgICAgdHlwZW9mIHBhcnNlZC5wbGF5YmFjayA9PT0gXCJib29sZWFuXCJcbiAgICAgICAgICAgID8gcGFyc2VkLnBsYXliYWNrXG4gICAgICAgICAgICA6IGZhbGxiYWNrLnBsYXliYWNrLFxuICAgICAgICB2b2ljZVVSSTpcbiAgICAgICAgICB0eXBlb2YgcGFyc2VkLnZvaWNlVVJJID09PSBcInN0cmluZ1wiICYmIHBhcnNlZC52b2ljZVVSSS5sZW5ndGggPiAwXG4gICAgICAgICAgICA/IHBhcnNlZC52b2ljZVVSSVxuICAgICAgICAgICAgOiBudWxsLFxuICAgICAgICBsYW5ndWFnZTpcbiAgICAgICAgICB0eXBlb2YgcGFyc2VkLmxhbmd1YWdlID09PSBcInN0cmluZ1wiICYmIHBhcnNlZC5sYW5ndWFnZVxuICAgICAgICAgICAgPyBwYXJzZWQubGFuZ3VhZ2VcbiAgICAgICAgICAgIDogZmFsbGJhY2subGFuZ3VhZ2UsXG4gICAgICB9O1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHJlYWQgdm9pY2UgcHJlZmVyZW5jZXNcIiwgZXJyKTtcbiAgICAgIHJldHVybiBmYWxsYmFjaztcbiAgICB9XG4gIH1cblxuICBwZXJzaXN0Vm9pY2VQcmVmZXJlbmNlcygpIHtcbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0cnkge1xuICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5zZXRJdGVtKFxuICAgICAgICBcImNoYXQudm9pY2VcIixcbiAgICAgICAgSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgIGF1dG9TZW5kOiBCb29sZWFuKHRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCksXG4gICAgICAgICAgcGxheWJhY2s6IEJvb2xlYW4odGhpcy52b2ljZVByZWZzLnBsYXliYWNrKSxcbiAgICAgICAgICB2b2ljZVVSSTogdGhpcy52b2ljZVByZWZzLnZvaWNlVVJJIHx8IG51bGwsXG4gICAgICAgICAgbGFuZ3VhZ2U6IHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSB8fCBudWxsLFxuICAgICAgICB9KSxcbiAgICAgICk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcGVyc2lzdCB2b2ljZSBwcmVmZXJlbmNlc1wiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIHNldHVwVm9pY2VGZWF0dXJlcygpIHtcbiAgICBjb25zdCBkb2NMYW5nID0gKFxuICAgICAgdGhpcy5kb2M/LmRvY3VtZW50RWxlbWVudD8uZ2V0QXR0cmlidXRlKFwibGFuZ1wiKSB8fCBcIlwiXG4gICAgKS50cmltKCk7XG4gICAgY29uc3QgbmF2aWdhdG9yTGFuZyA9XG4gICAgICB0eXBlb2YgbmF2aWdhdG9yICE9PSBcInVuZGVmaW5lZFwiICYmIG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgICA/IG5hdmlnYXRvci5sYW5ndWFnZVxuICAgICAgICA6IG51bGw7XG4gICAgY29uc3QgZGVmYXVsdExhbmd1YWdlID0gZG9jTGFuZyB8fCBuYXZpZ2F0b3JMYW5nIHx8IFwiZnItQ0FcIjtcbiAgICB0aGlzLnZvaWNlUHJlZnMgPSB0aGlzLmxvYWRWb2ljZVByZWZlcmVuY2VzKGRlZmF1bHRMYW5ndWFnZSk7XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMubGFuZ3VhZ2UpIHtcbiAgICAgIHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSA9IGRlZmF1bHRMYW5ndWFnZTtcbiAgICAgIHRoaXMucGVyc2lzdFZvaWNlUHJlZmVyZW5jZXMoKTtcbiAgICB9XG4gICAgdGhpcy52b2ljZVN0YXRlID0ge1xuICAgICAgZW5hYmxlZDogZmFsc2UsXG4gICAgICBsaXN0ZW5pbmc6IGZhbHNlLFxuICAgICAgYXdhaXRpbmdSZXNwb25zZTogZmFsc2UsXG4gICAgICBtYW51YWxTdG9wOiBmYWxzZSxcbiAgICAgIHJlc3RhcnRUaW1lcjogbnVsbCxcbiAgICAgIGxhc3RUcmFuc2NyaXB0OiBcIlwiLFxuICAgIH07XG4gICAgdGhpcy5zcGVlY2ggPSBjcmVhdGVTcGVlY2hTZXJ2aWNlKHtcbiAgICAgIGRlZmF1bHRMYW5ndWFnZTogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlLFxuICAgIH0pO1xuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMudm9pY2VVUkkpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSk7XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMubGFuZ3VhZ2UpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldExhbmd1YWdlKHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSk7XG4gICAgfVxuICAgIGNvbnN0IHJlY29nbml0aW9uU3VwcG9ydGVkID0gdGhpcy5zcGVlY2guaXNSZWNvZ25pdGlvblN1cHBvcnRlZCgpO1xuICAgIGNvbnN0IHN5bnRoZXNpc1N1cHBvcnRlZCA9IHRoaXMuc3BlZWNoLmlzU3ludGhlc2lzU3VwcG9ydGVkKCk7XG4gICAgdGhpcy51aS5zZXRWb2ljZUF2YWlsYWJpbGl0eSh7XG4gICAgICByZWNvZ25pdGlvbjogcmVjb2duaXRpb25TdXBwb3J0ZWQsXG4gICAgICBzeW50aGVzaXM6IHN5bnRoZXNpc1N1cHBvcnRlZCxcbiAgICB9KTtcbiAgICB0aGlzLnVpLnNldFZvaWNlUHJlZmVyZW5jZXModGhpcy52b2ljZVByZWZzKTtcbiAgICBpZiAocmVjb2duaXRpb25TdXBwb3J0ZWQpIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiQWN0aXZleiBsZSBtaWNybyBwb3VyIGRpY3RlciB2b3RyZSBtZXNzYWdlLlwiLFxuICAgICAgICBcIm11dGVkXCIsXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoc3ludGhlc2lzU3VwcG9ydGVkKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkxlY3R1cmUgdm9jYWxlIGRpc3BvbmlibGUuIExhIGRpY3RcdTAwRTllIG5cdTAwRTljZXNzaXRlIHVuIG5hdmlnYXRldXIgY29tcGF0aWJsZS5cIixcbiAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICApO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkxlcyBmb25jdGlvbm5hbGl0XHUwMEU5cyB2b2NhbGVzIG5lIHNvbnQgcGFzIGRpc3BvbmlibGVzIGRhbnMgY2UgbmF2aWdhdGV1ci5cIixcbiAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICk7XG4gICAgfVxuICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUocmVjb2duaXRpb25TdXBwb3J0ZWQgPyA1MDAwIDogNzAwMCk7XG4gICAgdGhpcy5zcGVlY2gub24oXCJsaXN0ZW5pbmctY2hhbmdlXCIsIChwYXlsb2FkKSA9PlxuICAgICAgdGhpcy5oYW5kbGVWb2ljZUxpc3RlbmluZ0NoYW5nZShwYXlsb2FkKSxcbiAgICApO1xuICAgIHRoaXMuc3BlZWNoLm9uKFwidHJhbnNjcmlwdFwiLCAocGF5bG9hZCkgPT5cbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VUcmFuc2NyaXB0KHBheWxvYWQpLFxuICAgICk7XG4gICAgdGhpcy5zcGVlY2gub24oXCJlcnJvclwiLCAocGF5bG9hZCkgPT4gdGhpcy5oYW5kbGVWb2ljZUVycm9yKHBheWxvYWQpKTtcbiAgICB0aGlzLnNwZWVjaC5vbihcInNwZWFraW5nLWNoYW5nZVwiLCAocGF5bG9hZCkgPT5cbiAgICAgIHRoaXMuaGFuZGxlVm9pY2VTcGVha2luZ0NoYW5nZShwYXlsb2FkKSxcbiAgICApO1xuICAgIHRoaXMuc3BlZWNoLm9uKFwidm9pY2VzXCIsICh7IHZvaWNlcyB9KSA9PlxuICAgICAgdGhpcy5oYW5kbGVWb2ljZVZvaWNlcyhBcnJheS5pc0FycmF5KHZvaWNlcykgPyB2b2ljZXMgOiBbXSksXG4gICAgKTtcbiAgfVxuXG4gIGFzeW5jIHRvZ2dsZVZvaWNlTGlzdGVuaW5nKCkge1xuICAgIGlmICghdGhpcy5zcGVlY2ggfHwgIXRoaXMuc3BlZWNoLmlzUmVjb2duaXRpb25TdXBwb3J0ZWQoKSkge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcbiAgICAgICAgXCJMYSBkaWN0XHUwMEU5ZSB2b2NhbGUgbidlc3QgcGFzIGRpc3BvbmlibGUgZGFucyBjZSBuYXZpZ2F0ZXVyLlwiLFxuICAgICAgICBcImRhbmdlclwiLFxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5saXN0ZW5pbmcgfHwgdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUubWFudWFsU3RvcCA9IHRydWU7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpIHtcbiAgICAgICAgd2luZG93LmNsZWFyVGltZW91dCh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKTtcbiAgICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgICB9XG4gICAgICB0aGlzLnNwZWVjaC5zdG9wTGlzdGVuaW5nKCk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiRGljdFx1MDBFOWUgaW50ZXJyb21wdWUuXCIsIFwibXV0ZWRcIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlU3RhdGUubWFudWFsU3RvcCA9IGZhbHNlO1xuICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gdHJ1ZTtcbiAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKSB7XG4gICAgICB3aW5kb3cuY2xlYXJUaW1lb3V0KHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpO1xuICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgfVxuICAgIGNvbnN0IHN0YXJ0ZWQgPSBhd2FpdCB0aGlzLnNwZWVjaC5zdGFydExpc3RlbmluZyh7XG4gICAgICBsYW5ndWFnZTogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlLFxuICAgICAgaW50ZXJpbVJlc3VsdHM6IHRydWUsXG4gICAgICBjb250aW51b3VzOiBmYWxzZSxcbiAgICB9KTtcbiAgICBpZiAoIXN0YXJ0ZWQpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkltcG9zc2libGUgZGUgZFx1MDBFOW1hcnJlciBsYSBkaWN0XHUwMEU5ZS4gVlx1MDBFOXJpZmlleiBsZSBtaWNyby5cIixcbiAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICk7XG4gICAgfVxuICB9XG5cbiAgaGFuZGxlVm9pY2VMaXN0ZW5pbmdDaGFuZ2UocGF5bG9hZCA9IHt9KSB7XG4gICAgY29uc3QgbGlzdGVuaW5nID0gQm9vbGVhbihwYXlsb2FkLmxpc3RlbmluZyk7XG4gICAgdGhpcy52b2ljZVN0YXRlLmxpc3RlbmluZyA9IGxpc3RlbmluZztcbiAgICBpZiAobGlzdGVuaW5nKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlTGlzdGVuaW5nKHRydWUpO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVRyYW5zY3JpcHQoXCJcIiwgeyBzdGF0ZTogXCJpZGxlXCIgfSk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICBcIkVuIFx1MDBFOWNvdXRlXHUyMDI2IFBhcmxleiBsb3JzcXVlIHZvdXMgXHUwMEVBdGVzIHByXHUwMEVBdC5cIixcbiAgICAgICAgXCJpbmZvXCIsXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnVpLnNldFZvaWNlTGlzdGVuaW5nKGZhbHNlKTtcbiAgICBpZiAocGF5bG9hZC5yZWFzb24gPT09IFwibWFudWFsXCIpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5tYW51YWxTdG9wID0gZmFsc2U7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkRpY3RcdTAwRTllIGludGVycm9tcHVlLlwiLCBcIm11dGVkXCIpO1xuICAgICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSgzNTAwKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHBheWxvYWQucmVhc29uID09PSBcImVycm9yXCIpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuYXdhaXRpbmdSZXNwb25zZSA9IGZhbHNlO1xuICAgICAgY29uc3QgbWVzc2FnZSA9XG4gICAgICAgIHBheWxvYWQuY29kZSA9PT0gXCJub3QtYWxsb3dlZFwiXG4gICAgICAgICAgPyBcIkF1dG9yaXNleiBsJ2FjY1x1MDBFOHMgYXUgbWljcm9waG9uZSBwb3VyIGNvbnRpbnVlci5cIlxuICAgICAgICAgIDogXCJMYSBkaWN0XHUwMEU5ZSB2b2NhbGUgcydlc3QgaW50ZXJyb21wdWUuIFJcdTAwRTllc3NheWV6LlwiO1xuICAgICAgY29uc3QgdG9uZSA9IHBheWxvYWQuY29kZSA9PT0gXCJub3QtYWxsb3dlZFwiID8gXCJkYW5nZXJcIiA6IFwid2FybmluZ1wiO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhtZXNzYWdlLCB0b25lKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAodGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgJiYgIXRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlKSB7XG4gICAgICB0aGlzLm1heWJlUmVzdGFydFZvaWNlTGlzdGVuaW5nKDY1MCk7XG4gICAgfVxuICB9XG5cbiAgaGFuZGxlVm9pY2VUcmFuc2NyaXB0KHBheWxvYWQgPSB7fSkge1xuICAgIGNvbnN0IHRyYW5zY3JpcHQgPVxuICAgICAgdHlwZW9mIHBheWxvYWQudHJhbnNjcmlwdCA9PT0gXCJzdHJpbmdcIiA/IHBheWxvYWQudHJhbnNjcmlwdCA6IFwiXCI7XG4gICAgY29uc3QgaXNGaW5hbCA9IEJvb2xlYW4ocGF5bG9hZC5pc0ZpbmFsKTtcbiAgICBjb25zdCBjb25maWRlbmNlID1cbiAgICAgIHR5cGVvZiBwYXlsb2FkLmNvbmZpZGVuY2UgPT09IFwibnVtYmVyXCIgPyBwYXlsb2FkLmNvbmZpZGVuY2UgOiBudWxsO1xuICAgIGlmICh0cmFuc2NyaXB0KSB7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUubGFzdFRyYW5zY3JpcHQgPSB0cmFuc2NyaXB0O1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVRyYW5zY3JpcHQodHJhbnNjcmlwdCwge1xuICAgICAgICBzdGF0ZTogaXNGaW5hbCA/IFwiZmluYWxcIiA6IFwiaW50ZXJpbVwiLFxuICAgICAgfSk7XG4gICAgfVxuICAgIGlmICghaXNGaW5hbCkge1xuICAgICAgaWYgKHRyYW5zY3JpcHQpIHtcbiAgICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIlRyYW5zY3JpcHRpb24gZW4gY291cnNcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgfVxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAoIXRyYW5zY3JpcHQpIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXCJBdWN1biB0ZXh0ZSBuJ2EgXHUwMEU5dFx1MDBFOSByZWNvbm51LlwiLCBcIndhcm5pbmdcIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDMwMDApO1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UgPSBmYWxzZTtcbiAgICAgIGlmICghdGhpcy52b2ljZVByZWZzLmF1dG9TZW5kKSB7XG4gICAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICB9XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQpIHtcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gdHJ1ZTtcbiAgICAgIGNvbnN0IGNvbmZpZGVuY2VQY3QgPVxuICAgICAgICBjb25maWRlbmNlICE9PSBudWxsXG4gICAgICAgICAgPyBNYXRoLnJvdW5kKE1hdGgubWF4KDAsIE1hdGgubWluKDEsIGNvbmZpZGVuY2UpKSAqIDEwMClcbiAgICAgICAgICA6IG51bGw7XG4gICAgICBpZiAoY29uZmlkZW5jZVBjdCAhPT0gbnVsbCkge1xuICAgICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFxuICAgICAgICAgIGBFbnZvaSBkdSBtZXNzYWdlIGRpY3RcdTAwRTkgKCR7Y29uZmlkZW5jZVBjdH0lIGRlIGNvbmZpYW5jZSlcdTIwMjZgLFxuICAgICAgICAgIFwiaW5mb1wiLFxuICAgICAgICApO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkVudm9pIGR1IG1lc3NhZ2UgZGljdFx1MDBFOVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB9XG4gICAgICB0aGlzLnN1Ym1pdFZvaWNlUHJvbXB0KHRyYW5zY3JpcHQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSB0cmFuc2NyaXB0O1xuICAgICAgfVxuICAgICAgdGhpcy51aS51cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICB0aGlzLnVpLmF1dG9zaXplUHJvbXB0KCk7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiTWVzc2FnZSBkaWN0XHUwMEU5LiBWXHUwMEU5cmlmaWV6IGF2YW50IGwnZW52b2kuXCIsIFwiaW5mb1wiKTtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoNDUwMCk7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgIH1cbiAgfVxuXG4gIGhhbmRsZVZvaWNlRXJyb3IocGF5bG9hZCA9IHt9KSB7XG4gICAgY29uc3QgbWVzc2FnZSA9XG4gICAgICB0eXBlb2YgcGF5bG9hZC5tZXNzYWdlID09PSBcInN0cmluZ1wiICYmIHBheWxvYWQubWVzc2FnZS5sZW5ndGggPiAwXG4gICAgICAgID8gcGF5bG9hZC5tZXNzYWdlXG4gICAgICAgIDogXCJVbmUgZXJyZXVyIHZvY2FsZSBlc3Qgc3VydmVudWUuXCI7XG4gICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhtZXNzYWdlLCBcImRhbmdlclwiKTtcbiAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gZmFsc2U7XG4gICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpIHtcbiAgICAgIHdpbmRvdy5jbGVhclRpbWVvdXQodGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lcik7XG4gICAgICB0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSg2MDAwKTtcbiAgfVxuXG4gIGhhbmRsZVZvaWNlU3BlYWtpbmdDaGFuZ2UocGF5bG9hZCA9IHt9KSB7XG4gICAgY29uc3Qgc3BlYWtpbmcgPSBCb29sZWFuKHBheWxvYWQuc3BlYWtpbmcpO1xuICAgIHRoaXMudWkuc2V0Vm9pY2VTcGVha2luZyhzcGVha2luZyk7XG4gICAgaWYgKHNwZWFraW5nKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiTGVjdHVyZSBkZSBsYSByXHUwMEU5cG9uc2VcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAoXG4gICAgICB0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQgJiZcbiAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkICYmXG4gICAgICAhdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2VcbiAgICApIHtcbiAgICAgIHRoaXMubWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoODAwKTtcbiAgICB9XG4gICAgdGhpcy51aS5zY2hlZHVsZVZvaWNlU3RhdHVzSWRsZSgzNTAwKTtcbiAgfVxuXG4gIGhhbmRsZVZvaWNlVm9pY2VzKHZvaWNlcyA9IFtdKSB7XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHZvaWNlcykpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgbGV0IHNlbGVjdGVkVXJpID0gdGhpcy52b2ljZVByZWZzLnZvaWNlVVJJO1xuICAgIGlmICghc2VsZWN0ZWRVcmkgJiYgdm9pY2VzLmxlbmd0aCA+IDApIHtcbiAgICAgIGNvbnN0IHByZWZlcnJlZCA9IHZvaWNlcy5maW5kKCh2b2ljZSkgPT4ge1xuICAgICAgICBpZiAoIXZvaWNlIHx8ICF2b2ljZS5sYW5nKSB7XG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGxhbmcgPSBTdHJpbmcodm9pY2UubGFuZykudG9Mb3dlckNhc2UoKTtcbiAgICAgICAgY29uc3QgdGFyZ2V0ID0gKHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSB8fCBcIlwiKS50b0xvd2VyQ2FzZSgpO1xuICAgICAgICByZXR1cm4gdGFyZ2V0ICYmIGxhbmcuc3RhcnRzV2l0aCh0YXJnZXQuc2xpY2UoMCwgMikpO1xuICAgICAgfSk7XG4gICAgICBpZiAocHJlZmVycmVkKSB7XG4gICAgICAgIHNlbGVjdGVkVXJpID0gcHJlZmVycmVkLnZvaWNlVVJJIHx8IG51bGw7XG4gICAgICAgIHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSA9IHNlbGVjdGVkVXJpO1xuICAgICAgICB0aGlzLnBlcnNpc3RWb2ljZVByZWZlcmVuY2VzKCk7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMudWkuc2V0Vm9pY2VWb2ljZU9wdGlvbnModm9pY2VzLCBzZWxlY3RlZFVyaSB8fCBudWxsKTtcbiAgICBpZiAoc2VsZWN0ZWRVcmkpIHtcbiAgICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHNlbGVjdGVkVXJpKTtcbiAgICB9XG4gIH1cblxuICBoYW5kbGVWb2ljZUF1dG9TZW5kQ2hhbmdlKGVuYWJsZWQpIHtcbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQgPSBCb29sZWFuKGVuYWJsZWQpO1xuICAgIHRoaXMucGVyc2lzdFZvaWNlUHJlZmVyZW5jZXMoKTtcbiAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCkge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQgPSBmYWxzZTtcbiAgICAgIGlmICh0aGlzLnZvaWNlU3RhdGUubGlzdGVuaW5nKSB7XG4gICAgICAgIHRoaXMuc3BlZWNoLnN0b3BMaXN0ZW5pbmcoKTtcbiAgICAgIH1cbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiTW9kZSBtYW51ZWwgYWN0aXZcdTAwRTkuIFV0aWxpc2V6IGxlIG1pY3JvIHBvdXIgcmVtcGxpciBsZSBjaGFtcC5cIixcbiAgICAgICAgXCJtdXRlZFwiLFxuICAgICAgKTtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoNDAwMCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgIFwiTGVzIG1lc3NhZ2VzIGRpY3RcdTAwRTlzIHNlcm9udCBlbnZveVx1MDBFOXMgYXV0b21hdGlxdWVtZW50LlwiLFxuICAgICAgICBcImluZm9cIixcbiAgICAgICk7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgIH1cbiAgfVxuXG4gIGhhbmRsZVZvaWNlUGxheWJhY2tDaGFuZ2UoZW5hYmxlZCkge1xuICAgIGlmICghdGhpcy52b2ljZVByZWZzKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IG5leHQgPSBCb29sZWFuKGVuYWJsZWQpO1xuICAgIHRoaXMudm9pY2VQcmVmcy5wbGF5YmFjayA9IG5leHQ7XG4gICAgdGhpcy5wZXJzaXN0Vm9pY2VQcmVmZXJlbmNlcygpO1xuICAgIGlmICghbmV4dCkge1xuICAgICAgdGhpcy5zdG9wVm9pY2VQbGF5YmFjaygpO1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIkxlY3R1cmUgdm9jYWxlIGRcdTAwRTlzYWN0aXZcdTAwRTllLlwiLCBcIm11dGVkXCIpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiTGVjdHVyZSB2b2NhbGUgYWN0aXZcdTAwRTllLlwiLCBcImluZm9cIik7XG4gICAgfVxuICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gIH1cblxuICBoYW5kbGVWb2ljZVZvaWNlQ2hhbmdlKHZvaWNlVVJJKSB7XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdmFsdWUgPSB2b2ljZVVSSSAmJiB2b2ljZVVSSS5sZW5ndGggPiAwID8gdm9pY2VVUkkgOiBudWxsO1xuICAgIHRoaXMudm9pY2VQcmVmcy52b2ljZVVSSSA9IHZhbHVlO1xuICAgIHRoaXMuc3BlZWNoLnNldFByZWZlcnJlZFZvaWNlKHZhbHVlKTtcbiAgICB0aGlzLnBlcnNpc3RWb2ljZVByZWZlcmVuY2VzKCk7XG4gICAgaWYgKHZhbHVlKSB7XG4gICAgICB0aGlzLnVpLnNldFZvaWNlU3RhdHVzKFwiVm9peCBzXHUwMEU5bGVjdGlvbm5cdTAwRTllIG1pc2UgXHUwMEUwIGpvdXIuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy51aS5zZXRWb2ljZVN0YXR1cyhcIlZvaXggcGFyIGRcdTAwRTlmYXV0IGR1IHN5c3RcdTAwRThtZSB1dGlsaXNcdTAwRTllLlwiLCBcIm11dGVkXCIpO1xuICAgIH1cbiAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDMwMDApO1xuICB9XG5cbiAgc3RvcFZvaWNlUGxheWJhY2soKSB7XG4gICAgaWYgKCF0aGlzLnNwZWVjaCB8fCAhdGhpcy5zcGVlY2guaXNTeW50aGVzaXNTdXBwb3J0ZWQoKSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0aGlzLnNwZWVjaC5zdG9wU3BlYWtpbmcoKTtcbiAgICB0aGlzLnVpLnNldFZvaWNlU3BlYWtpbmcoZmFsc2UpO1xuICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXCJMZWN0dXJlIHZvY2FsZSBpbnRlcnJvbXB1ZS5cIiwgXCJtdXRlZFwiKTtcbiAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDMwMDApO1xuICB9XG5cbiAgbWF5YmVSZXN0YXJ0Vm9pY2VMaXN0ZW5pbmcoZGVsYXkgPSA2NTApIHtcbiAgICBpZiAoIXRoaXMuc3BlZWNoIHx8ICF0aGlzLnNwZWVjaC5pc1JlY29nbml0aW9uU3VwcG9ydGVkKCkpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKCF0aGlzLnZvaWNlUHJlZnMuYXV0b1NlbmQgfHwgIXRoaXMudm9pY2VTdGF0ZS5lbmFibGVkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUubGlzdGVuaW5nIHx8IHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyKSB7XG4gICAgICB3aW5kb3cuY2xlYXJUaW1lb3V0KHRoaXMudm9pY2VTdGF0ZS5yZXN0YXJ0VGltZXIpO1xuICAgIH1cbiAgICB0aGlzLnZvaWNlU3RhdGUucmVzdGFydFRpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgdGhpcy52b2ljZVN0YXRlLnJlc3RhcnRUaW1lciA9IG51bGw7XG4gICAgICBpZiAoIXRoaXMudm9pY2VQcmVmcy5hdXRvU2VuZCB8fCAhdGhpcy52b2ljZVN0YXRlLmVuYWJsZWQpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMudm9pY2VTdGF0ZS5saXN0ZW5pbmcgfHwgdGhpcy52b2ljZVN0YXRlLmF3YWl0aW5nUmVzcG9uc2UpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgY29uc3QgYXR0ZW1wdCA9IHRoaXMuc3BlZWNoLnN0YXJ0TGlzdGVuaW5nKHtcbiAgICAgICAgbGFuZ3VhZ2U6IHRoaXMudm9pY2VQcmVmcy5sYW5ndWFnZSxcbiAgICAgICAgaW50ZXJpbVJlc3VsdHM6IHRydWUsXG4gICAgICAgIGNvbnRpbnVvdXM6IGZhbHNlLFxuICAgICAgfSk7XG4gICAgICBQcm9taXNlLnJlc29sdmUoYXR0ZW1wdClcbiAgICAgICAgLnRoZW4oKHN0YXJ0ZWQpID0+IHtcbiAgICAgICAgICBpZiAoc3RhcnRlZCkge1xuICAgICAgICAgICAgcmV0dXJuO1xuICAgICAgICAgIH1cbiAgICAgICAgICB0aGlzLnZvaWNlU3RhdGUuZW5hYmxlZCA9IGZhbHNlO1xuICAgICAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgICAgICBcIkltcG9zc2libGUgZGUgcmVsYW5jZXIgbGEgZGljdFx1MDBFOWUgdm9jYWxlLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9KVxuICAgICAgICAuY2F0Y2goKGVycikgPT4ge1xuICAgICAgICAgIHRoaXMudm9pY2VTdGF0ZS5lbmFibGVkID0gZmFsc2U7XG4gICAgICAgICAgY29uc29sZS5lcnJvcihcIkF1dG9tYXRpYyB2b2ljZSByZXN0YXJ0IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXG4gICAgICAgICAgICBcIkltcG9zc2libGUgZGUgcmVsYW5jZXIgbGEgZGljdFx1MDBFOWUgdm9jYWxlLlwiLFxuICAgICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgICApO1xuICAgICAgICB9KTtcbiAgICB9LCBkZWxheSk7XG4gIH1cblxuICBzdWJtaXRWb2ljZVByb21wdCh0ZXh0KSB7XG4gICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC52YWx1ZSA9IHRleHQ7XG4gICAgfVxuICAgIHRoaXMudWkudXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICB0aGlzLnVpLmVtaXQoXCJzdWJtaXRcIiwgeyB0ZXh0IH0pO1xuICB9XG5cbiAgZ2V0TGF0ZXN0QXNzaXN0YW50VGV4dCgpIHtcbiAgICBpZiAoIXRoaXMudGltZWxpbmVTdG9yZSB8fCAhdGhpcy50aW1lbGluZVN0b3JlLm9yZGVyKSB7XG4gICAgICByZXR1cm4gXCJcIjtcbiAgICB9XG4gICAgZm9yIChsZXQgaSA9IHRoaXMudGltZWxpbmVTdG9yZS5vcmRlci5sZW5ndGggLSAxOyBpID49IDA7IGkgLT0gMSkge1xuICAgICAgY29uc3QgaWQgPSB0aGlzLnRpbWVsaW5lU3RvcmUub3JkZXJbaV07XG4gICAgICBjb25zdCBlbnRyeSA9IHRoaXMudGltZWxpbmVTdG9yZS5tYXAuZ2V0KGlkKTtcbiAgICAgIGlmIChlbnRyeSAmJiBlbnRyeS5yb2xlID09PSBcImFzc2lzdGFudFwiICYmIGVudHJ5LnRleHQpIHtcbiAgICAgICAgcmV0dXJuIGVudHJ5LnRleHQ7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBcIlwiO1xuICB9XG5cbiAgZm9ybWF0RW1iZWRkaW5nUmVzcG9uc2UocmVzdWx0KSB7XG4gICAgY29uc3Qgc2FmZUlubGluZSA9ICh2YWx1ZSkgPT4ge1xuICAgICAgaWYgKHZhbHVlID09PSBudWxsIHx8IHR5cGVvZiB2YWx1ZSA9PT0gXCJ1bmRlZmluZWRcIiB8fCB2YWx1ZSA9PT0gXCJcIikge1xuICAgICAgICByZXR1cm4gXCJcdTIwMTRcIjtcbiAgICAgIH1cbiAgICAgIHJldHVybiBgXFxgJHtTdHJpbmcodmFsdWUpLnJlcGxhY2UoL2AvZywgXCJcXFxcYFwiKX1cXGBgO1xuICAgIH07XG4gICAgY29uc3QgdmVjdG9ycyA9IEFycmF5LmlzQXJyYXkocmVzdWx0Py52ZWN0b3JzKSA/IHJlc3VsdC52ZWN0b3JzIDogW107XG4gICAgY29uc3QgZGltc0NhbmRpZGF0ZSA9XG4gICAgICB0eXBlb2YgcmVzdWx0Py5kaW1zID09PSBcIm51bWJlclwiID8gcmVzdWx0LmRpbXMgOiBOdW1iZXIocmVzdWx0Py5kaW1zKTtcbiAgICBjb25zdCBkaW1zID0gTnVtYmVyLmlzRmluaXRlKGRpbXNDYW5kaWRhdGUpXG4gICAgICA/IE51bWJlcihkaW1zQ2FuZGlkYXRlKVxuICAgICAgOiBBcnJheS5pc0FycmF5KHZlY3RvcnNbMF0pXG4gICAgICAgID8gdmVjdG9yc1swXS5sZW5ndGhcbiAgICAgICAgOiAwO1xuICAgIGNvbnN0IGNvdW50Q2FuZGlkYXRlID1cbiAgICAgIHR5cGVvZiByZXN1bHQ/LmNvdW50ID09PSBcIm51bWJlclwiID8gcmVzdWx0LmNvdW50IDogTnVtYmVyKHJlc3VsdD8uY291bnQpO1xuICAgIGNvbnN0IGNvdW50ID0gTnVtYmVyLmlzRmluaXRlKGNvdW50Q2FuZGlkYXRlKVxuICAgICAgPyBOdW1iZXIoY291bnRDYW5kaWRhdGUpXG4gICAgICA6IHZlY3RvcnMubGVuZ3RoO1xuICAgIGNvbnN0IG5vcm1hbGlzZWQgPSBCb29sZWFuKHJlc3VsdD8ubm9ybWFsaXNlZCk7XG4gICAgY29uc3Qgc3VtbWFyeUxpbmVzID0gW1xuICAgICAgYC0gKipCYWNrZW5kIDoqKiAke3NhZmVJbmxpbmUocmVzdWx0Py5iYWNrZW5kID8/IFwiaW5jb25udVwiKX1gLFxuICAgICAgYC0gKipNb2RcdTAwRThsZSA6KiogJHtzYWZlSW5saW5lKHJlc3VsdD8ubW9kZWwgPz8gXCJpbmNvbm51XCIpfWAsXG4gICAgICBgLSAqKkRpbWVuc2lvbnMgOioqICR7ZGltcyB8fCAwfWAsXG4gICAgICBgLSAqKlZlY3RldXJzIGdcdTAwRTluXHUwMEU5clx1MDBFOXMgOioqICR7Y291bnR9YCxcbiAgICAgIGAtICoqTm9ybWFsaXNhdGlvbiBhcHBsaXF1XHUwMEU5ZSA6KiogJHtub3JtYWxpc2VkID8gXCJPdWlcIiA6IFwiTm9uXCJ9YCxcbiAgICBdO1xuXG4gICAgY29uc3QgdmVjdG9yU2VjdGlvbnMgPSBbXTtcbiAgICB2ZWN0b3JzLmZvckVhY2goKHZlY3RvciwgaW5kZXgpID0+IHtcbiAgICAgIGlmICghQXJyYXkuaXNBcnJheSh2ZWN0b3IpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHByZXZpZXdMZW5ndGggPSBNYXRoLm1pbigxMiwgdmVjdG9yLmxlbmd0aCk7XG4gICAgICBjb25zdCBwcmV2aWV3VmFsdWVzID0gdmVjdG9yLnNsaWNlKDAsIHByZXZpZXdMZW5ndGgpLm1hcCgodmFsdWUpID0+IHtcbiAgICAgICAgY29uc3QgbnVtZXJpYyA9IHR5cGVvZiB2YWx1ZSA9PT0gXCJudW1iZXJcIiA/IHZhbHVlIDogTnVtYmVyKHZhbHVlKTtcbiAgICAgICAgaWYgKE51bWJlci5pc0Zpbml0ZShudW1lcmljKSkge1xuICAgICAgICAgIHJldHVybiBOdW1iZXIucGFyc2VGbG9hdChudW1lcmljLnRvRml4ZWQoNikpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB2YWx1ZTtcbiAgICAgIH0pO1xuICAgICAgY29uc3QgcHJldmlld0pzb24gPSBKU09OLnN0cmluZ2lmeShwcmV2aWV3VmFsdWVzLCBudWxsLCAyKTtcbiAgICAgIGxldCBzZWN0aW9uID0gW1xuICAgICAgICBgKioke3ZlY3RvcnMubGVuZ3RoID4gMSA/IGBWZWN0ZXVyICR7aW5kZXggKyAxfWAgOiBcIlZlY3RldXJcIn0qKmAsXG4gICAgICAgIFwiYGBganNvblwiLFxuICAgICAgICBgJHtwcmV2aWV3SnNvbn0ke3ZlY3Rvci5sZW5ndGggPiBwcmV2aWV3TGVuZ3RoID8gXCJcXG4vLyBcdTIwMjZcIiA6IFwiXCJ9YCxcbiAgICAgICAgXCJgYGBcIixcbiAgICAgIF0uam9pbihcIlxcblwiKTtcbiAgICAgIGlmICh2ZWN0b3IubGVuZ3RoID4gcHJldmlld0xlbmd0aCkge1xuICAgICAgICBjb25zdCBmdWxsVmVjdG9yID0gdmVjdG9yLm1hcCgodmFsdWUpID0+IHtcbiAgICAgICAgICBjb25zdCBudW1lcmljID0gdHlwZW9mIHZhbHVlID09PSBcIm51bWJlclwiID8gdmFsdWUgOiBOdW1iZXIodmFsdWUpO1xuICAgICAgICAgIHJldHVybiBOdW1iZXIuaXNGaW5pdGUobnVtZXJpYykgPyBudW1lcmljIDogdmFsdWU7XG4gICAgICAgIH0pO1xuICAgICAgICBzZWN0aW9uICs9IGBcXG5cXG48ZGV0YWlscz48c3VtbWFyeT5WZWN0ZXVyIGNvbXBsZXQgJHtpbmRleCArIDF9PC9zdW1tYXJ5PlxcblxcblxcYFxcYFxcYGpzb25cXG4ke0pTT04uc3RyaW5naWZ5KFxuICAgICAgICAgIGZ1bGxWZWN0b3IsXG4gICAgICAgICAgbnVsbCxcbiAgICAgICAgICAyLFxuICAgICAgICApfVxcblxcYFxcYFxcYFxcblxcbjwvZGV0YWlscz5gO1xuICAgICAgfVxuICAgICAgdmVjdG9yU2VjdGlvbnMucHVzaChzZWN0aW9uKTtcbiAgICB9KTtcblxuICAgIGNvbnN0IHNlY3Rpb25zID0gW1wiIyMjIFJcdTAwRTlzdWx0YXQgZCdlbWJlZGRpbmdcIiwgc3VtbWFyeUxpbmVzLmpvaW4oXCJcXG5cIildO1xuICAgIGlmICh2ZWN0b3JTZWN0aW9ucy5sZW5ndGggPiAwKSB7XG4gICAgICBzZWN0aW9ucy5wdXNoKHZlY3RvclNlY3Rpb25zLmpvaW4oXCJcXG5cXG5cIikpO1xuICAgIH0gZWxzZSB7XG4gICAgICBzZWN0aW9ucy5wdXNoKFwiKipBdWN1bmUgY29tcG9zYW50ZSBkJ2VtYmVkZGluZyBuJ2EgXHUwMEU5dFx1MDBFOSByZW52b3lcdTAwRTllLioqXCIpO1xuICAgIH1cbiAgICByZXR1cm4gc2VjdGlvbnMuam9pbihcIlxcblxcblwiKTtcbiAgfVxuXG4gIHByZXNlbnRFbWJlZGRpbmdSZXN1bHQocmVzdWx0KSB7XG4gICAgY29uc3QgdmVjdG9ycyA9IEFycmF5LmlzQXJyYXkocmVzdWx0Py52ZWN0b3JzKSA/IHJlc3VsdC52ZWN0b3JzIDogW107XG4gICAgY29uc3QgZGltc0NhbmRpZGF0ZSA9XG4gICAgICB0eXBlb2YgcmVzdWx0Py5kaW1zID09PSBcIm51bWJlclwiID8gcmVzdWx0LmRpbXMgOiBOdW1iZXIocmVzdWx0Py5kaW1zKTtcbiAgICBjb25zdCBkaW1zID0gTnVtYmVyLmlzRmluaXRlKGRpbXNDYW5kaWRhdGUpXG4gICAgICA/IE51bWJlcihkaW1zQ2FuZGlkYXRlKVxuICAgICAgOiBBcnJheS5pc0FycmF5KHZlY3RvcnNbMF0pXG4gICAgICAgID8gdmVjdG9yc1swXS5sZW5ndGhcbiAgICAgICAgOiAwO1xuICAgIGNvbnN0IGNvdW50Q2FuZGlkYXRlID1cbiAgICAgIHR5cGVvZiByZXN1bHQ/LmNvdW50ID09PSBcIm51bWJlclwiID8gcmVzdWx0LmNvdW50IDogTnVtYmVyKHJlc3VsdD8uY291bnQpO1xuICAgIGNvbnN0IGNvdW50ID0gTnVtYmVyLmlzRmluaXRlKGNvdW50Q2FuZGlkYXRlKVxuICAgICAgPyBOdW1iZXIoY291bnRDYW5kaWRhdGUpXG4gICAgICA6IHZlY3RvcnMubGVuZ3RoO1xuICAgIGNvbnN0IG5vcm1hbGlzZWQgPSBCb29sZWFuKHJlc3VsdD8ubm9ybWFsaXNlZCk7XG4gICAgY29uc3QgbWV0YUJpdHMgPSBbXCJFbWJlZGRpbmdcIl07XG4gICAgaWYgKGRpbXMpIHtcbiAgICAgIG1ldGFCaXRzLnB1c2goYCR7ZGltc30gZGltc2ApO1xuICAgIH1cbiAgICBpZiAoY291bnQpIHtcbiAgICAgIG1ldGFCaXRzLnB1c2goYCR7Y291bnR9IHZlY3RldXIke2NvdW50ID4gMSA/IFwic1wiIDogXCJcIn1gKTtcbiAgICB9XG4gICAgaWYgKG5vcm1hbGlzZWQpIHtcbiAgICAgIG1ldGFCaXRzLnB1c2goXCJOb3JtYWxpc1x1MDBFOVwiKTtcbiAgICB9XG4gICAgY29uc3QgbWVzc2FnZSA9IHRoaXMuZm9ybWF0RW1iZWRkaW5nUmVzcG9uc2UocmVzdWx0KTtcbiAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJhc3Npc3RhbnRcIiwgbWVzc2FnZSwge1xuICAgICAgdGltZXN0YW1wOiBub3dJU08oKSxcbiAgICAgIG1ldGFTdWZmaXg6IG1ldGFCaXRzLmpvaW4oXCIgXHUyMDIyIFwiKSxcbiAgICAgIG1ldGFkYXRhOiB7XG4gICAgICAgIG1vZGU6IFwiZW1iZWRcIixcbiAgICAgICAgZGltcyxcbiAgICAgICAgYmFja2VuZDpcbiAgICAgICAgICB0eXBlb2YgcmVzdWx0Py5iYWNrZW5kID09PSBcInN0cmluZ1wiICYmIHJlc3VsdC5iYWNrZW5kXG4gICAgICAgICAgICA/IHJlc3VsdC5iYWNrZW5kXG4gICAgICAgICAgICA6IG51bGwsXG4gICAgICAgIG1vZGVsOlxuICAgICAgICAgIHR5cGVvZiByZXN1bHQ/Lm1vZGVsID09PSBcInN0cmluZ1wiICYmIHJlc3VsdC5tb2RlbFxuICAgICAgICAgICAgPyByZXN1bHQubW9kZWxcbiAgICAgICAgICAgIDogbnVsbCxcbiAgICAgICAgY291bnQsXG4gICAgICAgIG5vcm1hbGlzZWQsXG4gICAgICB9LFxuICAgICAgZW1iZWRkaW5nRGF0YToge1xuICAgICAgICBiYWNrZW5kOlxuICAgICAgICAgIHR5cGVvZiByZXN1bHQ/LmJhY2tlbmQgPT09IFwic3RyaW5nXCIgJiYgcmVzdWx0LmJhY2tlbmRcbiAgICAgICAgICAgID8gcmVzdWx0LmJhY2tlbmRcbiAgICAgICAgICAgIDogbnVsbCxcbiAgICAgICAgbW9kZWw6XG4gICAgICAgICAgdHlwZW9mIHJlc3VsdD8ubW9kZWwgPT09IFwic3RyaW5nXCIgJiYgcmVzdWx0Lm1vZGVsXG4gICAgICAgICAgICA/IHJlc3VsdC5tb2RlbFxuICAgICAgICAgICAgOiBudWxsLFxuICAgICAgICBkaW1zLFxuICAgICAgICBjb3VudCxcbiAgICAgICAgbm9ybWFsaXNlZCxcbiAgICAgICAgdmVjdG9ycyxcbiAgICAgICAgcmF3OiByZXN1bHQsXG4gICAgICB9LFxuICAgIH0pO1xuICB9XG5cbiAgaGFuZGxlVm9pY2VBc3Npc3RhbnRDb21wbGV0aW9uKCkge1xuICAgIGlmICghdGhpcy52b2ljZVByZWZzKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGxhdGVzdCA9IHRoaXMuZ2V0TGF0ZXN0QXNzaXN0YW50VGV4dCgpO1xuICAgIHRoaXMudm9pY2VTdGF0ZS5hd2FpdGluZ1Jlc3BvbnNlID0gZmFsc2U7XG4gICAgaWYgKCFsYXRlc3QpIHtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gICAgICB0aGlzLm1heWJlUmVzdGFydFZvaWNlTGlzdGVuaW5nKDgwMCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmIChcbiAgICAgIHRoaXMudm9pY2VQcmVmcy5wbGF5YmFjayAmJlxuICAgICAgdGhpcy5zcGVlY2ggJiZcbiAgICAgIHRoaXMuc3BlZWNoLmlzU3ludGhlc2lzU3VwcG9ydGVkKClcbiAgICApIHtcbiAgICAgIHRoaXMudWkuc2V0Vm9pY2VTdGF0dXMoXCJMZWN0dXJlIGRlIGxhIHJcdTAwRTlwb25zZVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICBjb25zdCB1dHRlcmFuY2UgPSB0aGlzLnNwZWVjaC5zcGVhayhsYXRlc3QsIHtcbiAgICAgICAgbGFuZzogdGhpcy52b2ljZVByZWZzLmxhbmd1YWdlLFxuICAgICAgICB2b2ljZVVSSTogdGhpcy52b2ljZVByZWZzLnZvaWNlVVJJLFxuICAgICAgfSk7XG4gICAgICBpZiAoIXV0dGVyYW5jZSkge1xuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlVm9pY2VTdGF0dXNJZGxlKDM1MDApO1xuICAgICAgICB0aGlzLm1heWJlUmVzdGFydFZvaWNlTGlzdGVuaW5nKDgwMCk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudWkuc2NoZWR1bGVWb2ljZVN0YXR1c0lkbGUoMzUwMCk7XG4gICAgICB0aGlzLm1heWJlUmVzdGFydFZvaWNlTGlzdGVuaW5nKDgwMCk7XG4gICAgfVxuICB9XG5cbiAgaGFuZGxlU29ja2V0RXZlbnQoZXYpIHtcbiAgICBjb25zdCB0eXBlID0gZXYgJiYgZXYudHlwZSA/IGV2LnR5cGUgOiBcIlwiO1xuICAgIGNvbnN0IGRhdGEgPSBldiAmJiBldi5kYXRhID8gZXYuZGF0YSA6IHt9O1xuICAgIHN3aXRjaCAodHlwZSkge1xuICAgICAgY2FzZSBcIndzLmNvbm5lY3RlZFwiOiB7XG4gICAgICAgIGlmIChkYXRhICYmIGRhdGEub3JpZ2luKSB7XG4gICAgICAgICAgdGhpcy51aS5hbm5vdW5jZUNvbm5lY3Rpb24oYENvbm5lY3RcdTAwRTkgdmlhICR7ZGF0YS5vcmlnaW59YCk7XG4gICAgICAgICAgdGhpcy51aS51cGRhdGVDb25uZWN0aW9uTWV0YShcbiAgICAgICAgICAgIGBDb25uZWN0XHUwMEU5IHZpYSAke2RhdGEub3JpZ2lufWAsXG4gICAgICAgICAgICBcInN1Y2Nlc3NcIixcbiAgICAgICAgICApO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRoaXMudWkuYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29ubmVjdFx1MDBFOSBhdSBzZXJ2ZXVyLlwiKTtcbiAgICAgICAgICB0aGlzLnVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiQ29ubmVjdFx1MDBFOSBhdSBzZXJ2ZXVyLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiaGlzdG9yeS5zbmFwc2hvdFwiOiB7XG4gICAgICAgIGlmIChkYXRhICYmIEFycmF5LmlzQXJyYXkoZGF0YS5pdGVtcykpIHtcbiAgICAgICAgICB0aGlzLnVpLnJlbmRlckhpc3RvcnkoZGF0YS5pdGVtcywgeyByZXBsYWNlOiB0cnVlIH0pO1xuICAgICAgICB9XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImFpX21vZGVsLnJlc3BvbnNlX2NodW5rXCI6IHtcbiAgICAgICAgY29uc3QgZGVsdGEgPVxuICAgICAgICAgIHR5cGVvZiBkYXRhLmRlbHRhID09PSBcInN0cmluZ1wiID8gZGF0YS5kZWx0YSA6IGRhdGEudGV4dCB8fCBcIlwiO1xuICAgICAgICB0aGlzLnVpLmFwcGVuZFN0cmVhbShkZWx0YSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImFpX21vZGVsLnJlc3BvbnNlX2NvbXBsZXRlXCI6IHtcbiAgICAgICAgaWYgKGRhdGEgJiYgZGF0YS50ZXh0ICYmICF0aGlzLnVpLmhhc1N0cmVhbUJ1ZmZlcigpKSB7XG4gICAgICAgICAgdGhpcy51aS5hcHBlbmRTdHJlYW0oZGF0YS50ZXh0KTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLmVuZFN0cmVhbShkYXRhKTtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgaWYgKGRhdGEgJiYgdHlwZW9mIGRhdGEubGF0ZW5jeV9tcyAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgICAgIHRoaXMudWkuc2V0RGlhZ25vc3RpY3MoeyBsYXRlbmN5TXM6IE51bWJlcihkYXRhLmxhdGVuY3lfbXMpIH0pO1xuICAgICAgICB9XG4gICAgICAgIGlmIChkYXRhICYmIGRhdGEub2sgPT09IGZhbHNlICYmIGRhdGEuZXJyb3IpIHtcbiAgICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJzeXN0ZW1cIiwgZGF0YS5lcnJvciwge1xuICAgICAgICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMuaGFuZGxlVm9pY2VBc3Npc3RhbnRDb21wbGV0aW9uKCk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImNoYXQubWVzc2FnZVwiOiB7XG4gICAgICAgIGlmICghdGhpcy51aS5pc1N0cmVhbWluZygpKSB7XG4gICAgICAgICAgdGhpcy51aS5zdGFydFN0cmVhbSgpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChcbiAgICAgICAgICBkYXRhICYmXG4gICAgICAgICAgdHlwZW9mIGRhdGEucmVzcG9uc2UgPT09IFwic3RyaW5nXCIgJiZcbiAgICAgICAgICAhdGhpcy51aS5oYXNTdHJlYW1CdWZmZXIoKVxuICAgICAgICApIHtcbiAgICAgICAgICB0aGlzLnVpLmFwcGVuZFN0cmVhbShkYXRhLnJlc3BvbnNlKTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLmVuZFN0cmVhbShkYXRhKTtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgdGhpcy5oYW5kbGVWb2ljZUFzc2lzdGFudENvbXBsZXRpb24oKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiZXZvbHV0aW9uX2VuZ2luZS50cmFpbmluZ19jb21wbGV0ZVwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIGBcdTAwQzl2b2x1dGlvbiBtaXNlIFx1MDBFMCBqb3VyICR7ZGF0YSAmJiBkYXRhLnZlcnNpb24gPyBkYXRhLnZlcnNpb24gOiBcIlwifWAsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJva1wiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImV2b2x1dGlvbl9lbmdpbmUudHJhaW5pbmdfZmFpbGVkXCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFxuICAgICAgICAgIFwic3lzdGVtXCIsXG4gICAgICAgICAgYFx1MDBDOWNoZWMgZGUgbCdcdTAwRTl2b2x1dGlvbiA6ICR7ZGF0YSAmJiBkYXRhLmVycm9yID8gZGF0YS5lcnJvciA6IFwiaW5jb25udVwifWAsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInNsZWVwX3RpbWVfY29tcHV0ZS5waGFzZV9zdGFydFwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIFwiT3B0aW1pc2F0aW9uIGVuIGFycmlcdTAwRThyZS1wbGFuIGRcdTAwRTltYXJyXHUwMEU5ZVx1MjAyNlwiLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiaGludFwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInNsZWVwX3RpbWVfY29tcHV0ZS5jcmVhdGl2ZV9waGFzZVwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIGBFeHBsb3JhdGlvbiBkZSAke051bWJlcihkYXRhICYmIGRhdGEuaWRlYXMgPyBkYXRhLmlkZWFzIDogMSl9IGlkXHUwMEU5ZXNcdTIwMjZgLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiaGludFwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInBlcmZvcm1hbmNlLmFsZXJ0XCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwic3lzdGVtXCIsIGBQZXJmIDogJHt0aGlzLnVpLmZvcm1hdFBlcmYoZGF0YSl9YCwge1xuICAgICAgICAgIHZhcmlhbnQ6IFwid2FyblwiLFxuICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgIH0pO1xuICAgICAgICBpZiAoZGF0YSAmJiB0eXBlb2YgZGF0YS50dGZiX21zICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICAgICAgdGhpcy51aS5zZXREaWFnbm9zdGljcyh7IGxhdGVuY3lNczogTnVtYmVyKGRhdGEudHRmYl9tcykgfSk7XG4gICAgICAgIH1cbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwidWkuc3VnZ2VzdGlvbnNcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhcbiAgICAgICAgICBBcnJheS5pc0FycmF5KGRhdGEuYWN0aW9ucykgPyBkYXRhLmFjdGlvbnMgOiBbXSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBkZWZhdWx0OlxuICAgICAgICBpZiAodHlwZSAmJiB0eXBlLnN0YXJ0c1dpdGgoXCJ3cy5cIikpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcIlVuaGFuZGxlZCBldmVudFwiLCBldik7XG4gICAgfVxuICB9XG59XG4iLCAiLyoqXG4gKiBFbnRyeSBwb2ludCBmb3IgdGhlIGNoYXQgYXBwbGljYXRpb24uXG4gKiBFeHBlY3RzIHdpbmRvdy5jaGF0Q29uZmlnIHRvIGJlIGRlZmluZWQgYnkgdGhlIHNlcnZlci1yZW5kZXJlZCB0ZW1wbGF0ZS5cbiAqIEZhbGxzIGJhY2sgdG8gYW4gZW1wdHkgY29uZmlnIGlmIG5vdCBwcmVzZW50LlxuICogUmVxdWlyZWQgY29uZmlnIHNoYXBlOiB7IGFwaVVybD8sIHdzVXJsPywgdG9rZW4/LCAuLi4gfVxuICovXG5pbXBvcnQgeyBDaGF0QXBwIH0gZnJvbSBcIi4vYXBwLmpzXCI7XG5cbm5ldyBDaGF0QXBwKGRvY3VtZW50LCB3aW5kb3cuY2hhdENvbmZpZyB8fCB7fSk7XG4iXSwKICAibWFwcGluZ3MiOiAiOztBQUFPLFdBQVMsY0FBYyxNQUFNLENBQUMsR0FBRztBQUN0QyxVQUFNLFNBQVMsRUFBRSxHQUFHLElBQUk7QUFDeEIsVUFBTSxZQUFZLE9BQU8sY0FBYyxPQUFPLFNBQVM7QUFDdkQsUUFBSTtBQUNGLGFBQU8sVUFBVSxJQUFJLElBQUksU0FBUztBQUFBLElB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
