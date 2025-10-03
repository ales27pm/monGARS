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
    const composerStatusDefault = elements.composerStatus && elements.composerStatus.textContent.trim() || "Appuyez sur Ctrl+Entr\xE9e pour envoyer rapidement.";
    const filterHintDefault = elements.filterHint && elements.filterHint.textContent.trim() || "Utilisez le filtre pour limiter l'historique. Appuyez sur \xC9chap pour effacer.";
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
    function hideError() {
      if (!elements.errorAlert) return;
      elements.errorAlert.classList.add("d-none");
      if (elements.errorMessage) {
        elements.errorMessage.textContent = "";
      }
    }
    function showError(message) {
      if (!elements.errorAlert || !elements.errorMessage) return;
      elements.errorMessage.textContent = message;
      elements.errorAlert.classList.remove("d-none");
    }
    function setComposerStatus(message, tone = "muted") {
      if (!elements.composerStatus) return;
      const tones = ["muted", "info", "success", "danger", "warning"];
      elements.composerStatus.textContent = message;
      tones.forEach((t) => elements.composerStatus.classList.remove(`text-${t}`));
      elements.composerStatus.classList.add(`text-${tone}`);
    }
    function setComposerStatusIdle() {
      setComposerStatus(composerStatusDefault, "muted");
    }
    function scheduleComposerIdle(delay = 3500) {
      if (state.resetStatusTimer) {
        clearTimeout(state.resetStatusTimer);
      }
      state.resetStatusTimer = window.setTimeout(() => {
        setComposerStatusIdle();
      }, delay);
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
    }
    function initialise() {
      setDiagnostics({ connectedAt: null, lastMessageAt: null, latencyMs: null });
      updatePromptMetrics();
      autosizePrompt();
      setComposerStatusIdle();
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
      hasStreamBuffer
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
    let fallbackToken = typeof config.token === "string" && config.token.trim() !== "" ? config.token : void 0;
    function persistToken(token) {
      if (!token) {
        return;
      }
      fallbackToken = token;
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
      throw new Error(
        `Missing JWT (store it in localStorage as '${storageKey}' or provide it in the chat config).`
      );
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
      diagNetwork: byId("diag-network")
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
          metadata: { submitted: true }
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
          const message = err instanceof Error ? err.message : String(err);
          this.ui.showError(message);
          this.ui.appendMessage("system", message, {
            variant: "error",
            allowMarkdown: false,
            metadata: { stage: "submit" }
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
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsic3JjL2NvbmZpZy5qcyIsICJzcmMvdXRpbHMvdGltZS5qcyIsICJzcmMvc3RhdGUvdGltZWxpbmVTdG9yZS5qcyIsICJzcmMvdXRpbHMvZW1pdHRlci5qcyIsICJzcmMvdXRpbHMvZG9tLmpzIiwgInNyYy9zZXJ2aWNlcy9tYXJrZG93bi5qcyIsICJzcmMvdWkvY2hhdFVpLmpzIiwgInNyYy9zZXJ2aWNlcy9hdXRoLmpzIiwgInNyYy9zZXJ2aWNlcy9odHRwLmpzIiwgInNyYy9zZXJ2aWNlcy9leHBvcnRlci5qcyIsICJzcmMvc2VydmljZXMvc29ja2V0LmpzIiwgInNyYy9zZXJ2aWNlcy9zdWdnZXN0aW9ucy5qcyIsICJzcmMvYXBwLmpzIiwgInNyYy9pbmRleC5qcyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiZXhwb3J0IGZ1bmN0aW9uIHJlc29sdmVDb25maWcocmF3ID0ge30pIHtcbiAgY29uc3QgY29uZmlnID0geyAuLi5yYXcgfTtcbiAgY29uc3QgY2FuZGlkYXRlID0gY29uZmlnLmZhc3RhcGlVcmwgfHwgd2luZG93LmxvY2F0aW9uLm9yaWdpbjtcbiAgdHJ5IHtcbiAgICBjb25maWcuYmFzZVVybCA9IG5ldyBVUkwoY2FuZGlkYXRlKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS5lcnJvcihcIkludmFsaWQgRkFTVEFQSSBVUkxcIiwgZXJyLCBjYW5kaWRhdGUpO1xuICAgIGNvbmZpZy5iYXNlVXJsID0gbmV3IFVSTCh3aW5kb3cubG9jYXRpb24ub3JpZ2luKTtcbiAgfVxuICByZXR1cm4gY29uZmlnO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXBpVXJsKGNvbmZpZywgcGF0aCkge1xuICByZXR1cm4gbmV3IFVSTChwYXRoLCBjb25maWcuYmFzZVVybCkudG9TdHJpbmcoKTtcbn1cbiIsICJleHBvcnQgZnVuY3Rpb24gbm93SVNPKCkge1xuICByZXR1cm4gbmV3IERhdGUoKS50b0lTT1N0cmluZygpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZm9ybWF0VGltZXN0YW1wKHRzKSB7XG4gIGlmICghdHMpIHJldHVybiBcIlwiO1xuICB0cnkge1xuICAgIHJldHVybiBuZXcgRGF0ZSh0cykudG9Mb2NhbGVTdHJpbmcoXCJmci1DQVwiKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgcmV0dXJuIFN0cmluZyh0cyk7XG4gIH1cbn1cbiIsICJpbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBtYWtlTWVzc2FnZUlkKCkge1xuICByZXR1cm4gYG1zZy0ke0RhdGUubm93KCkudG9TdHJpbmcoMzYpfS0ke01hdGgucmFuZG9tKCkudG9TdHJpbmcoMzYpLnNsaWNlKDIsIDgpfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVUaW1lbGluZVN0b3JlKCkge1xuICBjb25zdCBvcmRlciA9IFtdO1xuICBjb25zdCBtYXAgPSBuZXcgTWFwKCk7XG5cbiAgZnVuY3Rpb24gcmVnaXN0ZXIoe1xuICAgIGlkLFxuICAgIHJvbGUsXG4gICAgdGV4dCA9IFwiXCIsXG4gICAgdGltZXN0YW1wID0gbm93SVNPKCksXG4gICAgcm93LFxuICAgIG1ldGFkYXRhID0ge30sXG4gIH0pIHtcbiAgICBjb25zdCBtZXNzYWdlSWQgPSBpZCB8fCBtYWtlTWVzc2FnZUlkKCk7XG4gICAgaWYgKCFtYXAuaGFzKG1lc3NhZ2VJZCkpIHtcbiAgICAgIG9yZGVyLnB1c2gobWVzc2FnZUlkKTtcbiAgICB9XG4gICAgbWFwLnNldChtZXNzYWdlSWQsIHtcbiAgICAgIGlkOiBtZXNzYWdlSWQsXG4gICAgICByb2xlLFxuICAgICAgdGV4dCxcbiAgICAgIHRpbWVzdGFtcCxcbiAgICAgIHJvdyxcbiAgICAgIG1ldGFkYXRhOiB7IC4uLm1ldGFkYXRhIH0sXG4gICAgfSk7XG4gICAgaWYgKHJvdykge1xuICAgICAgcm93LmRhdGFzZXQubWVzc2FnZUlkID0gbWVzc2FnZUlkO1xuICAgICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgICByb3cuZGF0YXNldC5yYXdUZXh0ID0gdGV4dDtcbiAgICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IHRpbWVzdGFtcDtcbiAgICB9XG4gICAgcmV0dXJuIG1lc3NhZ2VJZDtcbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZShpZCwgcGF0Y2ggPSB7fSkge1xuICAgIGlmICghbWFwLmhhcyhpZCkpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cbiAgICBjb25zdCBlbnRyeSA9IG1hcC5nZXQoaWQpO1xuICAgIGNvbnN0IG5leHQgPSB7IC4uLmVudHJ5LCAuLi5wYXRjaCB9O1xuICAgIGlmIChwYXRjaCAmJiB0eXBlb2YgcGF0Y2gubWV0YWRhdGEgPT09IFwib2JqZWN0XCIgJiYgcGF0Y2gubWV0YWRhdGEgIT09IG51bGwpIHtcbiAgICAgIGNvbnN0IG1lcmdlZCA9IHsgLi4uZW50cnkubWV0YWRhdGEgfTtcbiAgICAgIE9iamVjdC5lbnRyaWVzKHBhdGNoLm1ldGFkYXRhKS5mb3JFYWNoKChba2V5LCB2YWx1ZV0pID0+IHtcbiAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgfHwgdmFsdWUgPT09IG51bGwpIHtcbiAgICAgICAgICBkZWxldGUgbWVyZ2VkW2tleV07XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgbWVyZ2VkW2tleV0gPSB2YWx1ZTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBuZXh0Lm1ldGFkYXRhID0gbWVyZ2VkO1xuICAgIH1cbiAgICBtYXAuc2V0KGlkLCBuZXh0KTtcbiAgICBjb25zdCB7IHJvdyB9ID0gbmV4dDtcbiAgICBpZiAocm93ICYmIHJvdy5pc0Nvbm5lY3RlZCkge1xuICAgICAgaWYgKG5leHQudGV4dCAhPT0gZW50cnkudGV4dCkge1xuICAgICAgICByb3cuZGF0YXNldC5yYXdUZXh0ID0gbmV4dC50ZXh0IHx8IFwiXCI7XG4gICAgICB9XG4gICAgICBpZiAobmV4dC50aW1lc3RhbXAgIT09IGVudHJ5LnRpbWVzdGFtcCkge1xuICAgICAgICByb3cuZGF0YXNldC50aW1lc3RhbXAgPSBuZXh0LnRpbWVzdGFtcCB8fCBcIlwiO1xuICAgICAgfVxuICAgICAgaWYgKG5leHQucm9sZSAmJiBuZXh0LnJvbGUgIT09IGVudHJ5LnJvbGUpIHtcbiAgICAgICAgcm93LmRhdGFzZXQucm9sZSA9IG5leHQucm9sZTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5leHQ7XG4gIH1cblxuICBmdW5jdGlvbiBjb2xsZWN0KCkge1xuICAgIHJldHVybiBvcmRlclxuICAgICAgLm1hcCgoaWQpID0+IHtcbiAgICAgICAgY29uc3QgZW50cnkgPSBtYXAuZ2V0KGlkKTtcbiAgICAgICAgaWYgKCFlbnRyeSkge1xuICAgICAgICAgIHJldHVybiBudWxsO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgcm9sZTogZW50cnkucm9sZSxcbiAgICAgICAgICB0ZXh0OiBlbnRyeS50ZXh0LFxuICAgICAgICAgIHRpbWVzdGFtcDogZW50cnkudGltZXN0YW1wLFxuICAgICAgICAgIC4uLihlbnRyeS5tZXRhZGF0YSAmJlxuICAgICAgICAgICAgT2JqZWN0LmtleXMoZW50cnkubWV0YWRhdGEpLmxlbmd0aCA+IDAgJiYge1xuICAgICAgICAgICAgICBtZXRhZGF0YTogeyAuLi5lbnRyeS5tZXRhZGF0YSB9LFxuICAgICAgICAgICAgfSksXG4gICAgICAgIH07XG4gICAgICB9KVxuICAgICAgLmZpbHRlcihCb29sZWFuKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGNsZWFyKCkge1xuICAgIG9yZGVyLmxlbmd0aCA9IDA7XG4gICAgbWFwLmNsZWFyKCk7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIHJlZ2lzdGVyLFxuICAgIHVwZGF0ZSxcbiAgICBjb2xsZWN0LFxuICAgIGNsZWFyLFxuICAgIG9yZGVyLFxuICAgIG1hcCxcbiAgICBtYWtlTWVzc2FnZUlkLFxuICB9O1xufVxuIiwgImV4cG9ydCBmdW5jdGlvbiBjcmVhdGVFbWl0dGVyKCkge1xuICBjb25zdCBsaXN0ZW5lcnMgPSBuZXcgTWFwKCk7XG5cbiAgZnVuY3Rpb24gb24oZXZlbnQsIGhhbmRsZXIpIHtcbiAgICBpZiAoIWxpc3RlbmVycy5oYXMoZXZlbnQpKSB7XG4gICAgICBsaXN0ZW5lcnMuc2V0KGV2ZW50LCBuZXcgU2V0KCkpO1xuICAgIH1cbiAgICBsaXN0ZW5lcnMuZ2V0KGV2ZW50KS5hZGQoaGFuZGxlcik7XG4gICAgcmV0dXJuICgpID0+IG9mZihldmVudCwgaGFuZGxlcik7XG4gIH1cblxuICBmdW5jdGlvbiBvZmYoZXZlbnQsIGhhbmRsZXIpIHtcbiAgICBpZiAoIWxpc3RlbmVycy5oYXMoZXZlbnQpKSByZXR1cm47XG4gICAgY29uc3QgYnVja2V0ID0gbGlzdGVuZXJzLmdldChldmVudCk7XG4gICAgYnVja2V0LmRlbGV0ZShoYW5kbGVyKTtcbiAgICBpZiAoYnVja2V0LnNpemUgPT09IDApIHtcbiAgICAgIGxpc3RlbmVycy5kZWxldGUoZXZlbnQpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGVtaXQoZXZlbnQsIHBheWxvYWQpIHtcbiAgICBpZiAoIWxpc3RlbmVycy5oYXMoZXZlbnQpKSByZXR1cm47XG4gICAgbGlzdGVuZXJzLmdldChldmVudCkuZm9yRWFjaCgoaGFuZGxlcikgPT4ge1xuICAgICAgdHJ5IHtcbiAgICAgICAgaGFuZGxlcihwYXlsb2FkKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICBjb25zb2xlLmVycm9yKFwiRW1pdHRlciBoYW5kbGVyIGVycm9yXCIsIGVycik7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICByZXR1cm4geyBvbiwgb2ZmLCBlbWl0IH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGVzY2FwZUhUTUwoc3RyKSB7XG4gIHJldHVybiBTdHJpbmcoc3RyKS5yZXBsYWNlKFxuICAgIC9bJjw+XCInXS9nLFxuICAgIChjaCkgPT5cbiAgICAgICh7XG4gICAgICAgIFwiJlwiOiBcIiZhbXA7XCIsXG4gICAgICAgIFwiPFwiOiBcIiZsdDtcIixcbiAgICAgICAgXCI+XCI6IFwiJmd0O1wiLFxuICAgICAgICAnXCInOiBcIiZxdW90O1wiLFxuICAgICAgICBcIidcIjogXCImIzM5O1wiLFxuICAgICAgfSlbY2hdLFxuICApO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaHRtbFRvVGV4dChodG1sKSB7XG4gIGNvbnN0IHBhcnNlciA9IG5ldyBET01QYXJzZXIoKTtcbiAgY29uc3QgZG9jID0gcGFyc2VyLnBhcnNlRnJvbVN0cmluZyhodG1sLCBcInRleHQvaHRtbFwiKTtcbiAgcmV0dXJuIGRvYy5ib2R5LnRleHRDb250ZW50IHx8IFwiXCI7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBleHRyYWN0QnViYmxlVGV4dChidWJibGUpIHtcbiAgY29uc3QgY2xvbmUgPSBidWJibGUuY2xvbmVOb2RlKHRydWUpO1xuICBjbG9uZVxuICAgIC5xdWVyeVNlbGVjdG9yQWxsKFwiLmNvcHktYnRuLCAuY2hhdC1tZXRhXCIpXG4gICAgLmZvckVhY2goKG5vZGUpID0+IG5vZGUucmVtb3ZlKCkpO1xuICByZXR1cm4gY2xvbmUudGV4dENvbnRlbnQudHJpbSgpO1xufVxuIiwgImltcG9ydCB7IGVzY2FwZUhUTUwgfSBmcm9tIFwiLi4vdXRpbHMvZG9tLmpzXCI7XG5cbmV4cG9ydCBmdW5jdGlvbiByZW5kZXJNYXJrZG93bih0ZXh0KSB7XG4gIGlmICh0ZXh0ID09IG51bGwpIHtcbiAgICByZXR1cm4gXCJcIjtcbiAgfVxuICBjb25zdCB2YWx1ZSA9IFN0cmluZyh0ZXh0KTtcbiAgY29uc3QgZmFsbGJhY2sgPSAoKSA9PiB7XG4gICAgY29uc3QgZXNjYXBlZCA9IGVzY2FwZUhUTUwodmFsdWUpO1xuICAgIHJldHVybiBlc2NhcGVkLnJlcGxhY2UoL1xcbi9nLCBcIjxicj5cIik7XG4gIH07XG4gIHRyeSB7XG4gICAgaWYgKHdpbmRvdy5tYXJrZWQgJiYgdHlwZW9mIHdpbmRvdy5tYXJrZWQucGFyc2UgPT09IFwiZnVuY3Rpb25cIikge1xuICAgICAgY29uc3QgcmVuZGVyZWQgPSB3aW5kb3cubWFya2VkLnBhcnNlKHZhbHVlKTtcbiAgICAgIGlmICh3aW5kb3cuRE9NUHVyaWZ5ICYmIHR5cGVvZiB3aW5kb3cuRE9NUHVyaWZ5LnNhbml0aXplID09PSBcImZ1bmN0aW9uXCIpIHtcbiAgICAgICAgcmV0dXJuIHdpbmRvdy5ET01QdXJpZnkuc2FuaXRpemUocmVuZGVyZWQsIHtcbiAgICAgICAgICBBTExPV19VTktOT1dOX1BST1RPQ09MUzogZmFsc2UsXG4gICAgICAgICAgVVNFX1BST0ZJTEVTOiB7IGh0bWw6IHRydWUgfSxcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgICAvLyBGYWxsYmFjazogZXNjYXBlIHJhdyB0ZXh0IGFuZCBkbyBtaW5pbWFsIGZvcm1hdHRpbmcgdG8gYXZvaWQgWFNTXG4gICAgICBjb25zdCBlc2NhcGVkID0gZXNjYXBlSFRNTCh2YWx1ZSk7XG4gICAgICByZXR1cm4gZXNjYXBlZC5yZXBsYWNlKC9cXG4vZywgXCI8YnI+XCIpO1xuICAgIH1cbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS53YXJuKFwiTWFya2Rvd24gcmVuZGVyaW5nIGZhaWxlZFwiLCBlcnIpO1xuICB9XG4gIHJldHVybiBmYWxsYmFjaygpO1xufVxuIiwgImltcG9ydCB7IGNyZWF0ZUVtaXR0ZXIgfSBmcm9tIFwiLi4vdXRpbHMvZW1pdHRlci5qc1wiO1xuaW1wb3J0IHsgaHRtbFRvVGV4dCwgZXh0cmFjdEJ1YmJsZVRleHQsIGVzY2FwZUhUTUwgfSBmcm9tIFwiLi4vdXRpbHMvZG9tLmpzXCI7XG5pbXBvcnQgeyByZW5kZXJNYXJrZG93biB9IGZyb20gXCIuLi9zZXJ2aWNlcy9tYXJrZG93bi5qc1wiO1xuaW1wb3J0IHsgZm9ybWF0VGltZXN0YW1wLCBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlQ2hhdFVpKHsgZWxlbWVudHMsIHRpbWVsaW5lU3RvcmUgfSkge1xuICBjb25zdCBlbWl0dGVyID0gY3JlYXRlRW1pdHRlcigpO1xuXG4gIGNvbnN0IHNlbmRJZGxlTWFya3VwID0gZWxlbWVudHMuc2VuZCA/IGVsZW1lbnRzLnNlbmQuaW5uZXJIVE1MIDogXCJcIjtcbiAgY29uc3Qgc2VuZElkbGVMYWJlbCA9XG4gICAgKGVsZW1lbnRzLnNlbmQgJiYgZWxlbWVudHMuc2VuZC5nZXRBdHRyaWJ1dGUoXCJkYXRhLWlkbGUtbGFiZWxcIikpIHx8XG4gICAgKGVsZW1lbnRzLnNlbmQgPyBlbGVtZW50cy5zZW5kLnRleHRDb250ZW50LnRyaW0oKSA6IFwiRW52b3llclwiKTtcbiAgY29uc3Qgc2VuZEJ1c3lNYXJrdXAgPVxuICAgICc8c3BhbiBjbGFzcz1cInNwaW5uZXItYm9yZGVyIHNwaW5uZXItYm9yZGVyLXNtIG1lLTFcIiByb2xlPVwic3RhdHVzXCIgYXJpYS1oaWRkZW49XCJ0cnVlXCI+PC9zcGFuPkVudm9pXHUyMDI2JztcbiAgY29uc3QgY29tcG9zZXJTdGF0dXNEZWZhdWx0ID1cbiAgICAoZWxlbWVudHMuY29tcG9zZXJTdGF0dXMgJiYgZWxlbWVudHMuY29tcG9zZXJTdGF0dXMudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiQXBwdXlleiBzdXIgQ3RybCtFbnRyXHUwMEU5ZSBwb3VyIGVudm95ZXIgcmFwaWRlbWVudC5cIjtcbiAgY29uc3QgZmlsdGVySGludERlZmF1bHQgPVxuICAgIChlbGVtZW50cy5maWx0ZXJIaW50ICYmIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiVXRpbGlzZXogbGUgZmlsdHJlIHBvdXIgbGltaXRlciBsJ2hpc3RvcmlxdWUuIEFwcHV5ZXogc3VyIFx1MDBDOWNoYXAgcG91ciBlZmZhY2VyLlwiO1xuICBjb25zdCBwcm9tcHRNYXggPSBOdW1iZXIoZWxlbWVudHMucHJvbXB0Py5nZXRBdHRyaWJ1dGUoXCJtYXhsZW5ndGhcIikpIHx8IG51bGw7XG4gIGNvbnN0IHByZWZlcnNSZWR1Y2VkTW90aW9uID1cbiAgICB3aW5kb3cubWF0Y2hNZWRpYSAmJlxuICAgIHdpbmRvdy5tYXRjaE1lZGlhKFwiKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246IHJlZHVjZSlcIikubWF0Y2hlcztcbiAgY29uc3QgU0NST0xMX1RIUkVTSE9MRCA9IDE0MDtcbiAgY29uc3QgUFJPTVBUX01BWF9IRUlHSFQgPSAzMjA7XG5cbiAgY29uc3QgZGlhZ25vc3RpY3MgPSB7XG4gICAgY29ubmVjdGVkQXQ6IG51bGwsXG4gICAgbGFzdE1lc3NhZ2VBdDogbnVsbCxcbiAgICBsYXRlbmN5TXM6IG51bGwsXG4gIH07XG5cbiAgY29uc3Qgc3RhdGUgPSB7XG4gICAgcmVzZXRTdGF0dXNUaW1lcjogbnVsbCxcbiAgICBoaWRlU2Nyb2xsVGltZXI6IG51bGwsXG4gICAgYWN0aXZlRmlsdGVyOiBcIlwiLFxuICAgIGhpc3RvcnlCb290c3RyYXBwZWQ6IGVsZW1lbnRzLnRyYW5zY3JpcHQuY2hpbGRFbGVtZW50Q291bnQgPiAwLFxuICAgIGJvb3RzdHJhcHBpbmc6IGZhbHNlLFxuICAgIHN0cmVhbVJvdzogbnVsbCxcbiAgICBzdHJlYW1CdWY6IFwiXCIsXG4gICAgc3RyZWFtTWVzc2FnZUlkOiBudWxsLFxuICB9O1xuXG4gIGNvbnN0IHN0YXR1c0xhYmVscyA9IHtcbiAgICBvZmZsaW5lOiBcIkhvcnMgbGlnbmVcIixcbiAgICBjb25uZWN0aW5nOiBcIkNvbm5leGlvblx1MjAyNlwiLFxuICAgIG9ubGluZTogXCJFbiBsaWduZVwiLFxuICAgIGVycm9yOiBcIkVycmV1clwiLFxuICB9O1xuXG4gIGZ1bmN0aW9uIG9uKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgcmV0dXJuIGVtaXR0ZXIub24oZXZlbnQsIGhhbmRsZXIpO1xuICB9XG5cbiAgZnVuY3Rpb24gZW1pdChldmVudCwgcGF5bG9hZCkge1xuICAgIGVtaXR0ZXIuZW1pdChldmVudCwgcGF5bG9hZCk7XG4gIH1cblxuICBmdW5jdGlvbiBzZXRCdXN5KGJ1c3kpIHtcbiAgICBlbGVtZW50cy50cmFuc2NyaXB0LnNldEF0dHJpYnV0ZShcImFyaWEtYnVzeVwiLCBidXN5ID8gXCJ0cnVlXCIgOiBcImZhbHNlXCIpO1xuICAgIGlmIChlbGVtZW50cy5zZW5kKSB7XG4gICAgICBlbGVtZW50cy5zZW5kLmRpc2FibGVkID0gQm9vbGVhbihidXN5KTtcbiAgICAgIGVsZW1lbnRzLnNlbmQuc2V0QXR0cmlidXRlKFwiYXJpYS1idXN5XCIsIGJ1c3kgPyBcInRydWVcIiA6IFwiZmFsc2VcIik7XG4gICAgICBpZiAoYnVzeSkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLmlubmVySFRNTCA9IHNlbmRCdXN5TWFya3VwO1xuICAgICAgfSBlbHNlIGlmIChzZW5kSWRsZU1hcmt1cCkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLmlubmVySFRNTCA9IHNlbmRJZGxlTWFya3VwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZWxlbWVudHMuc2VuZC50ZXh0Q29udGVudCA9IHNlbmRJZGxlTGFiZWw7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gaGlkZUVycm9yKCkge1xuICAgIGlmICghZWxlbWVudHMuZXJyb3JBbGVydCkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLmVycm9yQWxlcnQuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICBpZiAoZWxlbWVudHMuZXJyb3JNZXNzYWdlKSB7XG4gICAgICBlbGVtZW50cy5lcnJvck1lc3NhZ2UudGV4dENvbnRlbnQgPSBcIlwiO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNob3dFcnJvcihtZXNzYWdlKSB7XG4gICAgaWYgKCFlbGVtZW50cy5lcnJvckFsZXJ0IHx8ICFlbGVtZW50cy5lcnJvck1lc3NhZ2UpIHJldHVybjtcbiAgICBlbGVtZW50cy5lcnJvck1lc3NhZ2UudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIGVsZW1lbnRzLmVycm9yQWxlcnQuY2xhc3NMaXN0LnJlbW92ZShcImQtbm9uZVwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldENvbXBvc2VyU3RhdHVzKG1lc3NhZ2UsIHRvbmUgPSBcIm11dGVkXCIpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzKSByZXR1cm47XG4gICAgY29uc3QgdG9uZXMgPSBbXCJtdXRlZFwiLCBcImluZm9cIiwgXCJzdWNjZXNzXCIsIFwiZGFuZ2VyXCIsIFwid2FybmluZ1wiXTtcbiAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgdG9uZXMuZm9yRWFjaCgodCkgPT4gZWxlbWVudHMuY29tcG9zZXJTdGF0dXMuY2xhc3NMaXN0LnJlbW92ZShgdGV4dC0ke3R9YCkpO1xuICAgIGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzLmNsYXNzTGlzdC5hZGQoYHRleHQtJHt0b25lfWApO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCkge1xuICAgIHNldENvbXBvc2VyU3RhdHVzKGNvbXBvc2VyU3RhdHVzRGVmYXVsdCwgXCJtdXRlZFwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlQ29tcG9zZXJJZGxlKGRlbGF5ID0gMzUwMCkge1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHN0YXRlLnJlc2V0U3RhdHVzVGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICB9LCBkZWxheSk7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVQcm9tcHRNZXRyaWNzKCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0Q291bnQgfHwgIWVsZW1lbnRzLnByb21wdCkgcmV0dXJuO1xuICAgIGNvbnN0IHZhbHVlID0gZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCI7XG4gICAgaWYgKHByb21wdE1heCkge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9IC8gJHtwcm9tcHRNYXh9YDtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9YDtcbiAgICB9XG4gICAgZWxlbWVudHMucHJvbXB0Q291bnQuY2xhc3NMaXN0LnJlbW92ZShcInRleHQtd2FybmluZ1wiLCBcInRleHQtZGFuZ2VyXCIpO1xuICAgIGlmIChwcm9tcHRNYXgpIHtcbiAgICAgIGNvbnN0IHJlbWFpbmluZyA9IHByb21wdE1heCAtIHZhbHVlLmxlbmd0aDtcbiAgICAgIGlmIChyZW1haW5pbmcgPD0gNSkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC1kYW5nZXJcIik7XG4gICAgICB9IGVsc2UgaWYgKHJlbWFpbmluZyA8PSAyMCkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC13YXJuaW5nXCIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGF1dG9zaXplUHJvbXB0KCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0KSByZXR1cm47XG4gICAgZWxlbWVudHMucHJvbXB0LnN0eWxlLmhlaWdodCA9IFwiYXV0b1wiO1xuICAgIGNvbnN0IG5leHRIZWlnaHQgPSBNYXRoLm1pbihcbiAgICAgIGVsZW1lbnRzLnByb21wdC5zY3JvbGxIZWlnaHQsXG4gICAgICBQUk9NUFRfTUFYX0hFSUdIVCxcbiAgICApO1xuICAgIGVsZW1lbnRzLnByb21wdC5zdHlsZS5oZWlnaHQgPSBgJHtuZXh0SGVpZ2h0fXB4YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzQXRCb3R0b20oKSB7XG4gICAgaWYgKCFlbGVtZW50cy50cmFuc2NyaXB0KSByZXR1cm4gdHJ1ZTtcbiAgICBjb25zdCBkaXN0YW5jZSA9XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCAtXG4gICAgICAoZWxlbWVudHMudHJhbnNjcmlwdC5zY3JvbGxUb3AgKyBlbGVtZW50cy50cmFuc2NyaXB0LmNsaWVudEhlaWdodCk7XG4gICAgcmV0dXJuIGRpc3RhbmNlIDw9IFNDUk9MTF9USFJFU0hPTEQ7XG4gIH1cblxuICBmdW5jdGlvbiBzY3JvbGxUb0JvdHRvbShvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybjtcbiAgICBjb25zdCBzbW9vdGggPSBvcHRpb25zLnNtb290aCAhPT0gZmFsc2UgJiYgIXByZWZlcnNSZWR1Y2VkTW90aW9uO1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuc2Nyb2xsVG8oe1xuICAgICAgdG9wOiBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCxcbiAgICAgIGJlaGF2aW9yOiBzbW9vdGggPyBcInNtb290aFwiIDogXCJhdXRvXCIsXG4gICAgfSk7XG4gICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICB9XG5cbiAgZnVuY3Rpb24gc2hvd1Njcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGlmIChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpO1xuICAgICAgc3RhdGUuaGlkZVNjcm9sbFRpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5yZW1vdmUoXCJkLW5vbmVcIik7XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJpcy12aXNpYmxlXCIpO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5zZXRBdHRyaWJ1dGUoXCJhcmlhLWhpZGRlblwiLCBcImZhbHNlXCIpO1xuICB9XG5cbiAgZnVuY3Rpb24gaGlkZVNjcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKFwiaXMtdmlzaWJsZVwiKTtcbiAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uc2V0QXR0cmlidXRlKFwiYXJpYS1oaWRkZW5cIiwgXCJ0cnVlXCIpO1xuICAgIHN0YXRlLmhpZGVTY3JvbGxUaW1lciA9IHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGlmIChlbGVtZW50cy5zY3JvbGxCb3R0b20pIHtcbiAgICAgICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgfSwgMjAwKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGhhbmRsZUNvcHkoYnViYmxlKSB7XG4gICAgY29uc3QgdGV4dCA9IGV4dHJhY3RCdWJibGVUZXh0KGJ1YmJsZSk7XG4gICAgaWYgKCF0ZXh0KSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICBpZiAobmF2aWdhdG9yLmNsaXBib2FyZCAmJiBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCkge1xuICAgICAgICBhd2FpdCBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCh0ZXh0KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IHRleHRhcmVhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRleHRhcmVhXCIpO1xuICAgICAgICB0ZXh0YXJlYS52YWx1ZSA9IHRleHQ7XG4gICAgICAgIHRleHRhcmVhLnNldEF0dHJpYnV0ZShcInJlYWRvbmx5XCIsIFwicmVhZG9ubHlcIik7XG4gICAgICAgIHRleHRhcmVhLnN0eWxlLnBvc2l0aW9uID0gXCJhYnNvbHV0ZVwiO1xuICAgICAgICB0ZXh0YXJlYS5zdHlsZS5sZWZ0ID0gXCItOTk5OXB4XCI7XG4gICAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgICB0ZXh0YXJlYS5zZWxlY3QoKTtcbiAgICAgICAgZG9jdW1lbnQuZXhlY0NvbW1hbmQoXCJjb3B5XCIpO1xuICAgICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKHRleHRhcmVhKTtcbiAgICAgIH1cbiAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcIkNvbnRlbnUgY29waVx1MDBFOSBkYW5zIGxlIHByZXNzZS1wYXBpZXJzLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJDb3B5IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiSW1wb3NzaWJsZSBkZSBjb3BpZXIgbGUgbWVzc2FnZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZGVjb3JhdGVSb3cocm93LCByb2xlKSB7XG4gICAgY29uc3QgYnViYmxlID0gcm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKCFidWJibGUpIHJldHVybjtcbiAgICBpZiAocm9sZSA9PT0gXCJhc3Npc3RhbnRcIiB8fCByb2xlID09PSBcInVzZXJcIikge1xuICAgICAgYnViYmxlLmNsYXNzTGlzdC5hZGQoXCJoYXMtdG9vbHNcIik7XG4gICAgICBidWJibGUucXVlcnlTZWxlY3RvckFsbChcIi5jb3B5LWJ0blwiKS5mb3JFYWNoKChidG4pID0+IGJ0bi5yZW1vdmUoKSk7XG4gICAgICBjb25zdCBjb3B5QnRuID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImJ1dHRvblwiKTtcbiAgICAgIGNvcHlCdG4udHlwZSA9IFwiYnV0dG9uXCI7XG4gICAgICBjb3B5QnRuLmNsYXNzTmFtZSA9IFwiY29weS1idG5cIjtcbiAgICAgIGNvcHlCdG4uaW5uZXJIVE1MID1cbiAgICAgICAgJzxzcGFuIGFyaWEtaGlkZGVuPVwidHJ1ZVwiPlx1MjlDOTwvc3Bhbj48c3BhbiBjbGFzcz1cInZpc3VhbGx5LWhpZGRlblwiPkNvcGllciBsZSBtZXNzYWdlPC9zcGFuPic7XG4gICAgICBjb3B5QnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiBoYW5kbGVDb3B5KGJ1YmJsZSkpO1xuICAgICAgYnViYmxlLmFwcGVuZENoaWxkKGNvcHlCdG4pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGhpZ2hsaWdodFJvdyhyb3csIHJvbGUpIHtcbiAgICBpZiAoIXJvdyB8fCBzdGF0ZS5ib290c3RyYXBwaW5nIHx8IHJvbGUgPT09IFwic3lzdGVtXCIpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgcm93LmNsYXNzTGlzdC5hZGQoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgfSwgNjAwKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGxpbmUocm9sZSwgaHRtbCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qgc2hvdWxkU3RpY2sgPSBpc0F0Qm90dG9tKCk7XG4gICAgY29uc3Qgcm93ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRpdlwiKTtcbiAgICByb3cuY2xhc3NOYW1lID0gYGNoYXQtcm93IGNoYXQtJHtyb2xlfWA7XG4gICAgcm93LmlubmVySFRNTCA9IGh0bWw7XG4gICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IG9wdGlvbnMucmF3VGV4dCB8fCBcIlwiO1xuICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IG9wdGlvbnMudGltZXN0YW1wIHx8IFwiXCI7XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5hcHBlbmRDaGlsZChyb3cpO1xuICAgIGRlY29yYXRlUm93KHJvdywgcm9sZSk7XG4gICAgaWYgKG9wdGlvbnMucmVnaXN0ZXIgIT09IGZhbHNlKSB7XG4gICAgICBjb25zdCB0cyA9IG9wdGlvbnMudGltZXN0YW1wIHx8IG5vd0lTTygpO1xuICAgICAgY29uc3QgdGV4dCA9XG4gICAgICAgIG9wdGlvbnMucmF3VGV4dCAmJiBvcHRpb25zLnJhd1RleHQubGVuZ3RoID4gMFxuICAgICAgICAgID8gb3B0aW9ucy5yYXdUZXh0XG4gICAgICAgICAgOiBodG1sVG9UZXh0KGh0bWwpO1xuICAgICAgY29uc3QgaWQgPSB0aW1lbGluZVN0b3JlLnJlZ2lzdGVyKHtcbiAgICAgICAgaWQ6IG9wdGlvbnMubWVzc2FnZUlkLFxuICAgICAgICByb2xlLFxuICAgICAgICB0ZXh0LFxuICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICByb3csXG4gICAgICAgIG1ldGFkYXRhOiBvcHRpb25zLm1ldGFkYXRhIHx8IHt9LFxuICAgICAgfSk7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBpZDtcbiAgICB9IGVsc2UgaWYgKG9wdGlvbnMubWVzc2FnZUlkKSB7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBvcHRpb25zLm1lc3NhZ2VJZDtcbiAgICB9IGVsc2UgaWYgKCFyb3cuZGF0YXNldC5tZXNzYWdlSWQpIHtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IHRpbWVsaW5lU3RvcmUubWFrZU1lc3NhZ2VJZCgpO1xuICAgIH1cbiAgICBpZiAoc2hvdWxkU3RpY2spIHtcbiAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiAhc3RhdGUuYm9vdHN0cmFwcGluZyB9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgc2hvd1Njcm9sbEJ1dHRvbigpO1xuICAgIH1cbiAgICBoaWdobGlnaHRSb3cocm93LCByb2xlKTtcbiAgICBpZiAoc3RhdGUuYWN0aXZlRmlsdGVyKSB7XG4gICAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIoc3RhdGUuYWN0aXZlRmlsdGVyLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfVxuICAgIHJldHVybiByb3c7XG4gIH1cblxuICBmdW5jdGlvbiBidWlsZEJ1YmJsZSh7XG4gICAgdGV4dCxcbiAgICB0aW1lc3RhbXAsXG4gICAgdmFyaWFudCxcbiAgICBtZXRhU3VmZml4LFxuICAgIGFsbG93TWFya2Rvd24gPSB0cnVlLFxuICB9KSB7XG4gICAgY29uc3QgY2xhc3NlcyA9IFtcImNoYXQtYnViYmxlXCJdO1xuICAgIGlmICh2YXJpYW50KSB7XG4gICAgICBjbGFzc2VzLnB1c2goYGNoYXQtYnViYmxlLSR7dmFyaWFudH1gKTtcbiAgICB9XG4gICAgY29uc3QgY29udGVudCA9IGFsbG93TWFya2Rvd25cbiAgICAgID8gcmVuZGVyTWFya2Rvd24odGV4dClcbiAgICAgIDogZXNjYXBlSFRNTChTdHJpbmcodGV4dCkpO1xuICAgIGNvbnN0IG1ldGFCaXRzID0gW107XG4gICAgaWYgKHRpbWVzdGFtcCkge1xuICAgICAgbWV0YUJpdHMucHVzaChmb3JtYXRUaW1lc3RhbXAodGltZXN0YW1wKSk7XG4gICAgfVxuICAgIGlmIChtZXRhU3VmZml4KSB7XG4gICAgICBtZXRhQml0cy5wdXNoKG1ldGFTdWZmaXgpO1xuICAgIH1cbiAgICBjb25zdCBtZXRhSHRtbCA9XG4gICAgICBtZXRhQml0cy5sZW5ndGggPiAwXG4gICAgICAgID8gYDxkaXYgY2xhc3M9XCJjaGF0LW1ldGFcIj4ke2VzY2FwZUhUTUwobWV0YUJpdHMuam9pbihcIiBcdTIwMjIgXCIpKX08L2Rpdj5gXG4gICAgICAgIDogXCJcIjtcbiAgICByZXR1cm4gYDxkaXYgY2xhc3M9XCIke2NsYXNzZXMuam9pbihcIiBcIil9XCI+JHtjb250ZW50fSR7bWV0YUh0bWx9PC9kaXY+YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGVuZE1lc3NhZ2Uocm9sZSwgdGV4dCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qge1xuICAgICAgdGltZXN0YW1wLFxuICAgICAgdmFyaWFudCxcbiAgICAgIG1ldGFTdWZmaXgsXG4gICAgICBhbGxvd01hcmtkb3duID0gdHJ1ZSxcbiAgICAgIG1lc3NhZ2VJZCxcbiAgICAgIHJlZ2lzdGVyID0gdHJ1ZSxcbiAgICAgIG1ldGFkYXRhLFxuICAgIH0gPSBvcHRpb25zO1xuICAgIGNvbnN0IGJ1YmJsZSA9IGJ1aWxkQnViYmxlKHtcbiAgICAgIHRleHQsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICB2YXJpYW50LFxuICAgICAgbWV0YVN1ZmZpeCxcbiAgICAgIGFsbG93TWFya2Rvd24sXG4gICAgfSk7XG4gICAgY29uc3Qgcm93ID0gbGluZShyb2xlLCBidWJibGUsIHtcbiAgICAgIHJhd1RleHQ6IHRleHQsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICBtZXNzYWdlSWQsXG4gICAgICByZWdpc3RlcixcbiAgICAgIG1ldGFkYXRhLFxuICAgIH0pO1xuICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogdGltZXN0YW1wIHx8IG5vd0lTTygpIH0pO1xuICAgIHJldHVybiByb3c7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVEaWFnbm9zdGljRmllbGQoZWwsIHZhbHVlKSB7XG4gICAgaWYgKCFlbCkgcmV0dXJuO1xuICAgIGVsLnRleHRDb250ZW50ID0gdmFsdWUgfHwgXCJcdTIwMTRcIjtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldERpYWdub3N0aWNzKHBhdGNoKSB7XG4gICAgT2JqZWN0LmFzc2lnbihkaWFnbm9zdGljcywgcGF0Y2gpO1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocGF0Y2gsIFwiY29ubmVjdGVkQXRcIikpIHtcbiAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChcbiAgICAgICAgZWxlbWVudHMuZGlhZ0Nvbm5lY3RlZCxcbiAgICAgICAgZGlhZ25vc3RpY3MuY29ubmVjdGVkQXRcbiAgICAgICAgICA/IGZvcm1hdFRpbWVzdGFtcChkaWFnbm9zdGljcy5jb25uZWN0ZWRBdClcbiAgICAgICAgICA6IFwiXHUyMDE0XCIsXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHBhdGNoLCBcImxhc3RNZXNzYWdlQXRcIikpIHtcbiAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChcbiAgICAgICAgZWxlbWVudHMuZGlhZ0xhc3RNZXNzYWdlLFxuICAgICAgICBkaWFnbm9zdGljcy5sYXN0TWVzc2FnZUF0XG4gICAgICAgICAgPyBmb3JtYXRUaW1lc3RhbXAoZGlhZ25vc3RpY3MubGFzdE1lc3NhZ2VBdClcbiAgICAgICAgICA6IFwiXHUyMDE0XCIsXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHBhdGNoLCBcImxhdGVuY3lNc1wiKSkge1xuICAgICAgaWYgKHR5cGVvZiBkaWFnbm9zdGljcy5sYXRlbmN5TXMgPT09IFwibnVtYmVyXCIpIHtcbiAgICAgICAgdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKFxuICAgICAgICAgIGVsZW1lbnRzLmRpYWdMYXRlbmN5LFxuICAgICAgICAgIGAke01hdGgubWF4KDAsIE1hdGgucm91bmQoZGlhZ25vc3RpY3MubGF0ZW5jeU1zKSl9IG1zYCxcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChlbGVtZW50cy5kaWFnTGF0ZW5jeSwgXCJcdTIwMTRcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gdXBkYXRlTmV0d29ya1N0YXR1cygpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmRpYWdOZXR3b3JrKSByZXR1cm47XG4gICAgY29uc3Qgb25saW5lID0gbmF2aWdhdG9yLm9uTGluZTtcbiAgICBlbGVtZW50cy5kaWFnTmV0d29yay50ZXh0Q29udGVudCA9IG9ubGluZSA/IFwiRW4gbGlnbmVcIiA6IFwiSG9ycyBsaWduZVwiO1xuICAgIGVsZW1lbnRzLmRpYWdOZXR3b3JrLmNsYXNzTGlzdC50b2dnbGUoXCJ0ZXh0LWRhbmdlclwiLCAhb25saW5lKTtcbiAgICBlbGVtZW50cy5kaWFnTmV0d29yay5jbGFzc0xpc3QudG9nZ2xlKFwidGV4dC1zdWNjZXNzXCIsIG9ubGluZSk7XG4gIH1cblxuICBmdW5jdGlvbiBhbm5vdW5jZUNvbm5lY3Rpb24obWVzc2FnZSwgdmFyaWFudCA9IFwiaW5mb1wiKSB7XG4gICAgaWYgKCFlbGVtZW50cy5jb25uZWN0aW9uKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGNsYXNzTGlzdCA9IGVsZW1lbnRzLmNvbm5lY3Rpb24uY2xhc3NMaXN0O1xuICAgIEFycmF5LmZyb20oY2xhc3NMaXN0KVxuICAgICAgLmZpbHRlcigoY2xzKSA9PiBjbHMuc3RhcnRzV2l0aChcImFsZXJ0LVwiKSAmJiBjbHMgIT09IFwiYWxlcnRcIilcbiAgICAgIC5mb3JFYWNoKChjbHMpID0+IGNsYXNzTGlzdC5yZW1vdmUoY2xzKSk7XG4gICAgY2xhc3NMaXN0LmFkZChcImFsZXJ0XCIpO1xuICAgIGNsYXNzTGlzdC5hZGQoYGFsZXJ0LSR7dmFyaWFudH1gKTtcbiAgICBlbGVtZW50cy5jb25uZWN0aW9uLnRleHRDb250ZW50ID0gbWVzc2FnZTtcbiAgICBjbGFzc0xpc3QucmVtb3ZlKFwidmlzdWFsbHktaGlkZGVuXCIpO1xuICAgIHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGNsYXNzTGlzdC5hZGQoXCJ2aXN1YWxseS1oaWRkZW5cIik7XG4gICAgfSwgNDAwMCk7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVDb25uZWN0aW9uTWV0YShtZXNzYWdlLCB0b25lID0gXCJtdXRlZFwiKSB7XG4gICAgaWYgKCFlbGVtZW50cy5jb25uZWN0aW9uTWV0YSkgcmV0dXJuO1xuICAgIGNvbnN0IHRvbmVzID0gW1wibXV0ZWRcIiwgXCJpbmZvXCIsIFwic3VjY2Vzc1wiLCBcImRhbmdlclwiLCBcIndhcm5pbmdcIl07XG4gICAgZWxlbWVudHMuY29ubmVjdGlvbk1ldGEudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIHRvbmVzLmZvckVhY2goKHQpID0+IGVsZW1lbnRzLmNvbm5lY3Rpb25NZXRhLmNsYXNzTGlzdC5yZW1vdmUoYHRleHQtJHt0fWApKTtcbiAgICBlbGVtZW50cy5jb25uZWN0aW9uTWV0YS5jbGFzc0xpc3QuYWRkKGB0ZXh0LSR7dG9uZX1gKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldFdzU3RhdHVzKHN0YXRlLCB0aXRsZSkge1xuICAgIGlmICghZWxlbWVudHMud3NTdGF0dXMpIHJldHVybjtcbiAgICBjb25zdCBsYWJlbCA9IHN0YXR1c0xhYmVsc1tzdGF0ZV0gfHwgc3RhdGU7XG4gICAgZWxlbWVudHMud3NTdGF0dXMudGV4dENvbnRlbnQgPSBsYWJlbDtcbiAgICBlbGVtZW50cy53c1N0YXR1cy5jbGFzc05hbWUgPSBgYmFkZ2Ugd3MtYmFkZ2UgJHtzdGF0ZX1gO1xuICAgIGlmICh0aXRsZSkge1xuICAgICAgZWxlbWVudHMud3NTdGF0dXMudGl0bGUgPSB0aXRsZTtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMud3NTdGF0dXMucmVtb3ZlQXR0cmlidXRlKFwidGl0bGVcIik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gbm9ybWFsaXplU3RyaW5nKHN0cikge1xuICAgIGNvbnN0IHZhbHVlID0gU3RyaW5nKHN0ciB8fCBcIlwiKTtcbiAgICB0cnkge1xuICAgICAgcmV0dXJuIHZhbHVlXG4gICAgICAgIC5ub3JtYWxpemUoXCJORkRcIilcbiAgICAgICAgLnJlcGxhY2UoL1tcXHUwMzAwLVxcdTAzNmZdL2csIFwiXCIpXG4gICAgICAgIC50b0xvd2VyQ2FzZSgpO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgcmV0dXJuIHZhbHVlLnRvTG93ZXJDYXNlKCk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gYXBwbHlUcmFuc2NyaXB0RmlsdGVyKHF1ZXJ5LCBvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybiAwO1xuICAgIGNvbnN0IHsgcHJlc2VydmVJbnB1dCA9IGZhbHNlIH0gPSBvcHRpb25zO1xuICAgIGNvbnN0IHJhd1F1ZXJ5ID0gdHlwZW9mIHF1ZXJ5ID09PSBcInN0cmluZ1wiID8gcXVlcnkgOiBcIlwiO1xuICAgIGlmICghcHJlc2VydmVJbnB1dCAmJiBlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQudmFsdWUgPSByYXdRdWVyeTtcbiAgICB9XG4gICAgY29uc3QgdHJpbW1lZCA9IHJhd1F1ZXJ5LnRyaW0oKTtcbiAgICBzdGF0ZS5hY3RpdmVGaWx0ZXIgPSB0cmltbWVkO1xuICAgIGNvbnN0IG5vcm1hbGl6ZWQgPSBub3JtYWxpemVTdHJpbmcodHJpbW1lZCk7XG4gICAgbGV0IG1hdGNoZXMgPSAwO1xuICAgIGNvbnN0IHJvd3MgPSBBcnJheS5mcm9tKGVsZW1lbnRzLnRyYW5zY3JpcHQucXVlcnlTZWxlY3RvckFsbChcIi5jaGF0LXJvd1wiKSk7XG4gICAgcm93cy5mb3JFYWNoKChyb3cpID0+IHtcbiAgICAgIHJvdy5jbGFzc0xpc3QucmVtb3ZlKFwiY2hhdC1oaWRkZW5cIiwgXCJjaGF0LWZpbHRlci1tYXRjaFwiKTtcbiAgICAgIGlmICghbm9ybWFsaXplZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBjb25zdCByYXcgPSByb3cuZGF0YXNldC5yYXdUZXh0IHx8IFwiXCI7XG4gICAgICBjb25zdCBub3JtYWxpemVkUm93ID0gbm9ybWFsaXplU3RyaW5nKHJhdyk7XG4gICAgICBpZiAobm9ybWFsaXplZFJvdy5pbmNsdWRlcyhub3JtYWxpemVkKSkge1xuICAgICAgICByb3cuY2xhc3NMaXN0LmFkZChcImNoYXQtZmlsdGVyLW1hdGNoXCIpO1xuICAgICAgICBtYXRjaGVzICs9IDE7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByb3cuY2xhc3NMaXN0LmFkZChcImNoYXQtaGlkZGVuXCIpO1xuICAgICAgfVxuICAgIH0pO1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuY2xhc3NMaXN0LnRvZ2dsZShcImZpbHRlcmVkXCIsIEJvb2xlYW4odHJpbW1lZCkpO1xuICAgIGlmIChlbGVtZW50cy5maWx0ZXJFbXB0eSkge1xuICAgICAgaWYgKHRyaW1tZWQgJiYgbWF0Y2hlcyA9PT0gMCkge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5jbGFzc0xpc3QucmVtb3ZlKFwiZC1ub25lXCIpO1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5zZXRBdHRyaWJ1dGUoXG4gICAgICAgICAgXCJhcmlhLWxpdmVcIixcbiAgICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5nZXRBdHRyaWJ1dGUoXCJhcmlhLWxpdmVcIikgfHwgXCJwb2xpdGVcIixcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJIaW50KSB7XG4gICAgICBpZiAodHJpbW1lZCkge1xuICAgICAgICBsZXQgc3VtbWFyeSA9IFwiQXVjdW4gbWVzc2FnZSBuZSBjb3JyZXNwb25kLlwiO1xuICAgICAgICBpZiAobWF0Y2hlcyA9PT0gMSkge1xuICAgICAgICAgIHN1bW1hcnkgPSBcIjEgbWVzc2FnZSBjb3JyZXNwb25kLlwiO1xuICAgICAgICB9IGVsc2UgaWYgKG1hdGNoZXMgPiAxKSB7XG4gICAgICAgICAgc3VtbWFyeSA9IGAke21hdGNoZXN9IG1lc3NhZ2VzIGNvcnJlc3BvbmRlbnQuYDtcbiAgICAgICAgfVxuICAgICAgICBlbGVtZW50cy5maWx0ZXJIaW50LnRleHRDb250ZW50ID0gc3VtbWFyeTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQgPSBmaWx0ZXJIaW50RGVmYXVsdDtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG1hdGNoZXM7XG4gIH1cblxuICBmdW5jdGlvbiByZWFwcGx5VHJhbnNjcmlwdEZpbHRlcigpIHtcbiAgICBpZiAoc3RhdGUuYWN0aXZlRmlsdGVyKSB7XG4gICAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIoc3RhdGUuYWN0aXZlRmlsdGVyLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfSBlbHNlIGlmIChlbGVtZW50cy50cmFuc2NyaXB0KSB7XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LmNsYXNzTGlzdC5yZW1vdmUoXCJmaWx0ZXJlZFwiKTtcbiAgICAgIGNvbnN0IHJvd3MgPSBBcnJheS5mcm9tKFxuICAgICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnF1ZXJ5U2VsZWN0b3JBbGwoXCIuY2hhdC1yb3dcIiksXG4gICAgICApO1xuICAgICAgcm93cy5mb3JFYWNoKChyb3cpID0+IHtcbiAgICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LWhpZGRlblwiLCBcImNoYXQtZmlsdGVyLW1hdGNoXCIpO1xuICAgICAgfSk7XG4gICAgICBpZiAoZWxlbWVudHMuZmlsdGVyRW1wdHkpIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICAgIH1cbiAgICAgIGlmIChlbGVtZW50cy5maWx0ZXJIaW50KSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQgPSBmaWx0ZXJIaW50RGVmYXVsdDtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBjbGVhclRyYW5zY3JpcHRGaWx0ZXIoZm9jdXMgPSB0cnVlKSB7XG4gICAgc3RhdGUuYWN0aXZlRmlsdGVyID0gXCJcIjtcbiAgICBpZiAoZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LnZhbHVlID0gXCJcIjtcbiAgICB9XG4gICAgcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICBpZiAoZm9jdXMgJiYgZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LmZvY3VzKCk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gcmVuZGVySGlzdG9yeShlbnRyaWVzLCBvcHRpb25zID0ge30pIHtcbiAgICBjb25zdCB7IHJlcGxhY2UgPSBmYWxzZSB9ID0gb3B0aW9ucztcbiAgICBpZiAoIUFycmF5LmlzQXJyYXkoZW50cmllcykgfHwgZW50cmllcy5sZW5ndGggPT09IDApIHtcbiAgICAgIGlmIChyZXBsYWNlKSB7XG4gICAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuaW5uZXJIVE1MID0gXCJcIjtcbiAgICAgICAgc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCA9IGZhbHNlO1xuICAgICAgICBoaWRlU2Nyb2xsQnV0dG9uKCk7XG4gICAgICAgIHRpbWVsaW5lU3RvcmUuY2xlYXIoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHJlcGxhY2UpIHtcbiAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuaW5uZXJIVE1MID0gXCJcIjtcbiAgICAgIHN0YXRlLmhpc3RvcnlCb290c3RyYXBwZWQgPSBmYWxzZTtcbiAgICAgIHN0YXRlLnN0cmVhbVJvdyA9IG51bGw7XG4gICAgICBzdGF0ZS5zdHJlYW1CdWYgPSBcIlwiO1xuICAgICAgdGltZWxpbmVTdG9yZS5jbGVhcigpO1xuICAgIH1cbiAgICBpZiAoc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCAmJiAhcmVwbGFjZSkge1xuICAgICAgc3RhdGUuYm9vdHN0cmFwcGluZyA9IHRydWU7XG4gICAgICBjb25zdCByb3dzID0gQXJyYXkuZnJvbShcbiAgICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5xdWVyeVNlbGVjdG9yQWxsKFwiLmNoYXQtcm93XCIpLFxuICAgICAgKTtcbiAgICAgIHJvd3MuZm9yRWFjaCgocm93KSA9PiB7XG4gICAgICAgIGNvbnN0IGV4aXN0aW5nSWQgPSByb3cuZGF0YXNldC5tZXNzYWdlSWQ7XG4gICAgICAgIGlmIChleGlzdGluZ0lkICYmIHRpbWVsaW5lU3RvcmUubWFwLmhhcyhleGlzdGluZ0lkKSkge1xuICAgICAgICAgIGNvbnN0IGN1cnJlbnRSb2xlID0gcm93LmRhdGFzZXQucm9sZSB8fCBcIlwiO1xuICAgICAgICAgIGlmIChjdXJyZW50Um9sZSkge1xuICAgICAgICAgICAgZGVjb3JhdGVSb3cocm93LCBjdXJyZW50Um9sZSk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCBidWJibGUgPSByb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICAgICAgY29uc3QgbWV0YSA9IGJ1YmJsZT8ucXVlcnlTZWxlY3RvcihcIi5jaGF0LW1ldGFcIikgfHwgbnVsbDtcbiAgICAgICAgY29uc3Qgcm9sZSA9XG4gICAgICAgICAgcm93LmRhdGFzZXQucm9sZSB8fFxuICAgICAgICAgIChyb3cuY2xhc3NMaXN0LmNvbnRhaW5zKFwiY2hhdC11c2VyXCIpXG4gICAgICAgICAgICA/IFwidXNlclwiXG4gICAgICAgICAgICA6IHJvdy5jbGFzc0xpc3QuY29udGFpbnMoXCJjaGF0LWFzc2lzdGFudFwiKVxuICAgICAgICAgICAgPyBcImFzc2lzdGFudFwiXG4gICAgICAgICAgICA6IFwic3lzdGVtXCIpO1xuICAgICAgICBjb25zdCB0ZXh0ID1cbiAgICAgICAgICByb3cuZGF0YXNldC5yYXdUZXh0ICYmIHJvdy5kYXRhc2V0LnJhd1RleHQubGVuZ3RoID4gMFxuICAgICAgICAgICAgPyByb3cuZGF0YXNldC5yYXdUZXh0XG4gICAgICAgICAgICA6IGJ1YmJsZVxuICAgICAgICAgICAgPyBleHRyYWN0QnViYmxlVGV4dChidWJibGUpXG4gICAgICAgICAgICA6IHJvdy50ZXh0Q29udGVudC50cmltKCk7XG4gICAgICAgIGNvbnN0IHRpbWVzdGFtcCA9XG4gICAgICAgICAgcm93LmRhdGFzZXQudGltZXN0YW1wICYmIHJvdy5kYXRhc2V0LnRpbWVzdGFtcC5sZW5ndGggPiAwXG4gICAgICAgICAgICA/IHJvdy5kYXRhc2V0LnRpbWVzdGFtcFxuICAgICAgICAgICAgOiBtZXRhXG4gICAgICAgICAgICA/IG1ldGEudGV4dENvbnRlbnQudHJpbSgpXG4gICAgICAgICAgICA6IG5vd0lTTygpO1xuICAgICAgICBjb25zdCBtZXNzYWdlSWQgPSB0aW1lbGluZVN0b3JlLnJlZ2lzdGVyKHtcbiAgICAgICAgICBpZDogZXhpc3RpbmdJZCxcbiAgICAgICAgICByb2xlLFxuICAgICAgICAgIHRleHQsXG4gICAgICAgICAgdGltZXN0YW1wLFxuICAgICAgICAgIHJvdyxcbiAgICAgICAgfSk7XG4gICAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IG1lc3NhZ2VJZDtcbiAgICAgICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgICAgIHJvdy5kYXRhc2V0LnJhd1RleHQgPSB0ZXh0O1xuICAgICAgICByb3cuZGF0YXNldC50aW1lc3RhbXAgPSB0aW1lc3RhbXA7XG4gICAgICAgIGRlY29yYXRlUm93KHJvdywgcm9sZSk7XG4gICAgICB9KTtcbiAgICAgIHN0YXRlLmJvb3RzdHJhcHBpbmcgPSBmYWxzZTtcbiAgICAgIHJlYXBwbHlUcmFuc2NyaXB0RmlsdGVyKCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHN0YXRlLmJvb3RzdHJhcHBpbmcgPSB0cnVlO1xuICAgIGVudHJpZXNcbiAgICAgIC5zbGljZSgpXG4gICAgICAucmV2ZXJzZSgpXG4gICAgICAuZm9yRWFjaCgoaXRlbSkgPT4ge1xuICAgICAgICBpZiAoaXRlbS5xdWVyeSkge1xuICAgICAgICAgIGFwcGVuZE1lc3NhZ2UoXCJ1c2VyXCIsIGl0ZW0ucXVlcnksIHtcbiAgICAgICAgICAgIHRpbWVzdGFtcDogaXRlbS50aW1lc3RhbXAsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGl0ZW0ucmVzcG9uc2UpIHtcbiAgICAgICAgICBhcHBlbmRNZXNzYWdlKFwiYXNzaXN0YW50XCIsIGl0ZW0ucmVzcG9uc2UsIHtcbiAgICAgICAgICAgIHRpbWVzdGFtcDogaXRlbS50aW1lc3RhbXAsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIHN0YXRlLmJvb3RzdHJhcHBpbmcgPSBmYWxzZTtcbiAgICBzdGF0ZS5oaXN0b3J5Qm9vdHN0cmFwcGVkID0gdHJ1ZTtcbiAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogZmFsc2UgfSk7XG4gICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICB9XG5cbiAgZnVuY3Rpb24gc3RhcnRTdHJlYW0oKSB7XG4gICAgc3RhdGUuc3RyZWFtQnVmID0gXCJcIjtcbiAgICBjb25zdCB0cyA9IG5vd0lTTygpO1xuICAgIHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCA9IHRpbWVsaW5lU3RvcmUubWFrZU1lc3NhZ2VJZCgpO1xuICAgIHN0YXRlLnN0cmVhbVJvdyA9IGxpbmUoXG4gICAgICBcImFzc2lzdGFudFwiLFxuICAgICAgJzxkaXYgY2xhc3M9XCJjaGF0LWJ1YmJsZVwiPjxzcGFuIGNsYXNzPVwiY2hhdC1jdXJzb3JcIj5cdTI1OEQ8L3NwYW4+PC9kaXY+JyxcbiAgICAgIHtcbiAgICAgICAgcmF3VGV4dDogXCJcIixcbiAgICAgICAgdGltZXN0YW1wOiB0cyxcbiAgICAgICAgbWVzc2FnZUlkOiBzdGF0ZS5zdHJlYW1NZXNzYWdlSWQsXG4gICAgICAgIG1ldGFkYXRhOiB7IHN0cmVhbWluZzogdHJ1ZSB9LFxuICAgICAgfSxcbiAgICApO1xuICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogdHMgfSk7XG4gICAgaWYgKHN0YXRlLnJlc2V0U3RhdHVzVGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKTtcbiAgICB9XG4gICAgc2V0Q29tcG9zZXJTdGF0dXMoXCJSXHUwMEU5cG9uc2UgZW4gY291cnNcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICB9XG5cbiAgZnVuY3Rpb24gaXNTdHJlYW1pbmcoKSB7XG4gICAgcmV0dXJuIEJvb2xlYW4oc3RhdGUuc3RyZWFtUm93KTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGhhc1N0cmVhbUJ1ZmZlcigpIHtcbiAgICByZXR1cm4gQm9vbGVhbihzdGF0ZS5zdHJlYW1CdWYpO1xuICB9XG5cbiAgZnVuY3Rpb24gYXBwZW5kU3RyZWFtKGRlbHRhKSB7XG4gICAgaWYgKCFzdGF0ZS5zdHJlYW1Sb3cpIHtcbiAgICAgIHN0YXJ0U3RyZWFtKCk7XG4gICAgfVxuICAgIGNvbnN0IHNob3VsZFN0aWNrID0gaXNBdEJvdHRvbSgpO1xuICAgIHN0YXRlLnN0cmVhbUJ1ZiArPSBkZWx0YSB8fCBcIlwiO1xuICAgIGNvbnN0IGJ1YmJsZSA9IHN0YXRlLnN0cmVhbVJvdy5xdWVyeVNlbGVjdG9yKFwiLmNoYXQtYnViYmxlXCIpO1xuICAgIGlmIChidWJibGUpIHtcbiAgICAgIGJ1YmJsZS5pbm5lckhUTUwgPSBgJHtyZW5kZXJNYXJrZG93bihzdGF0ZS5zdHJlYW1CdWYpfTxzcGFuIGNsYXNzPVwiY2hhdC1jdXJzb3JcIj5cdTI1OEQ8L3NwYW4+YDtcbiAgICB9XG4gICAgaWYgKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCkge1xuICAgICAgdGltZWxpbmVTdG9yZS51cGRhdGUoc3RhdGUuc3RyZWFtTWVzc2FnZUlkLCB7XG4gICAgICAgIHRleHQ6IHN0YXRlLnN0cmVhbUJ1ZixcbiAgICAgICAgbWV0YWRhdGE6IHsgc3RyZWFtaW5nOiB0cnVlIH0sXG4gICAgICB9KTtcbiAgICB9XG4gICAgc2V0RGlhZ25vc3RpY3MoeyBsYXN0TWVzc2FnZUF0OiBub3dJU08oKSB9KTtcbiAgICBpZiAoc2hvdWxkU3RpY2spIHtcbiAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiBmYWxzZSB9KTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBlbmRTdHJlYW0oZGF0YSkge1xuICAgIGlmICghc3RhdGUuc3RyZWFtUm93KSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGJ1YmJsZSA9IHN0YXRlLnN0cmVhbVJvdy5xdWVyeVNlbGVjdG9yKFwiLmNoYXQtYnViYmxlXCIpO1xuICAgIGlmIChidWJibGUpIHtcbiAgICAgIGJ1YmJsZS5pbm5lckhUTUwgPSByZW5kZXJNYXJrZG93bihzdGF0ZS5zdHJlYW1CdWYpO1xuICAgICAgY29uc3QgbWV0YSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJkaXZcIik7XG4gICAgICBtZXRhLmNsYXNzTmFtZSA9IFwiY2hhdC1tZXRhXCI7XG4gICAgICBjb25zdCB0cyA9IGRhdGEgJiYgZGF0YS50aW1lc3RhbXAgPyBkYXRhLnRpbWVzdGFtcCA6IG5vd0lTTygpO1xuICAgICAgbWV0YS50ZXh0Q29udGVudCA9IGZvcm1hdFRpbWVzdGFtcCh0cyk7XG4gICAgICBpZiAoZGF0YSAmJiBkYXRhLmVycm9yKSB7XG4gICAgICAgIG1ldGEuY2xhc3NMaXN0LmFkZChcInRleHQtZGFuZ2VyXCIpO1xuICAgICAgICBtZXRhLnRleHRDb250ZW50ID0gYCR7bWV0YS50ZXh0Q29udGVudH0gXHUyMDIyICR7ZGF0YS5lcnJvcn1gO1xuICAgICAgfVxuICAgICAgYnViYmxlLmFwcGVuZENoaWxkKG1ldGEpO1xuICAgICAgZGVjb3JhdGVSb3coc3RhdGUuc3RyZWFtUm93LCBcImFzc2lzdGFudFwiKTtcbiAgICAgIGhpZ2hsaWdodFJvdyhzdGF0ZS5zdHJlYW1Sb3csIFwiYXNzaXN0YW50XCIpO1xuICAgICAgaWYgKGlzQXRCb3R0b20oKSkge1xuICAgICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogdHJ1ZSB9KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHNob3dTY3JvbGxCdXR0b24oKTtcbiAgICAgIH1cbiAgICAgIGlmIChzdGF0ZS5zdHJlYW1NZXNzYWdlSWQpIHtcbiAgICAgICAgdGltZWxpbmVTdG9yZS51cGRhdGUoc3RhdGUuc3RyZWFtTWVzc2FnZUlkLCB7XG4gICAgICAgICAgdGV4dDogc3RhdGUuc3RyZWFtQnVmLFxuICAgICAgICAgIHRpbWVzdGFtcDogdHMsXG4gICAgICAgICAgbWV0YWRhdGE6IHtcbiAgICAgICAgICAgIHN0cmVhbWluZzogbnVsbCxcbiAgICAgICAgICAgIC4uLihkYXRhICYmIGRhdGEuZXJyb3IgPyB7IGVycm9yOiBkYXRhLmVycm9yIH0gOiB7IGVycm9yOiBudWxsIH0pLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgICAgc2V0RGlhZ25vc3RpY3MoeyBsYXN0TWVzc2FnZUF0OiB0cyB9KTtcbiAgICB9XG4gICAgY29uc3QgaGFzRXJyb3IgPSBCb29sZWFuKGRhdGEgJiYgZGF0YS5lcnJvcik7XG4gICAgc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICBoYXNFcnJvclxuICAgICAgICA/IFwiUlx1MDBFOXBvbnNlIGluZGlzcG9uaWJsZS4gQ29uc3VsdGV6IGxlcyBqb3VybmF1eC5cIlxuICAgICAgICA6IFwiUlx1MDBFOXBvbnNlIHJlXHUwMEU3dWUuXCIsXG4gICAgICBoYXNFcnJvciA/IFwiZGFuZ2VyXCIgOiBcInN1Y2Nlc3NcIixcbiAgICApO1xuICAgIHNjaGVkdWxlQ29tcG9zZXJJZGxlKGhhc0Vycm9yID8gNjAwMCA6IDM1MDApO1xuICAgIHN0YXRlLnN0cmVhbVJvdyA9IG51bGw7XG4gICAgc3RhdGUuc3RyZWFtQnVmID0gXCJcIjtcbiAgICBzdGF0ZS5zdHJlYW1NZXNzYWdlSWQgPSBudWxsO1xuICB9XG5cbiAgZnVuY3Rpb24gYXBwbHlRdWlja0FjdGlvbk9yZGVyaW5nKHN1Z2dlc3Rpb25zKSB7XG4gICAgaWYgKCFlbGVtZW50cy5xdWlja0FjdGlvbnMpIHJldHVybjtcbiAgICBpZiAoIUFycmF5LmlzQXJyYXkoc3VnZ2VzdGlvbnMpIHx8IHN1Z2dlc3Rpb25zLmxlbmd0aCA9PT0gMCkgcmV0dXJuO1xuICAgIGNvbnN0IGJ1dHRvbnMgPSBBcnJheS5mcm9tKFxuICAgICAgZWxlbWVudHMucXVpY2tBY3Rpb25zLnF1ZXJ5U2VsZWN0b3JBbGwoXCJidXR0b24ucWFcIiksXG4gICAgKTtcbiAgICBjb25zdCBsb29rdXAgPSBuZXcgTWFwKCk7XG4gICAgYnV0dG9ucy5mb3JFYWNoKChidG4pID0+IGxvb2t1cC5zZXQoYnRuLmRhdGFzZXQuYWN0aW9uLCBidG4pKTtcbiAgICBjb25zdCBmcmFnID0gZG9jdW1lbnQuY3JlYXRlRG9jdW1lbnRGcmFnbWVudCgpO1xuICAgIHN1Z2dlc3Rpb25zLmZvckVhY2goKGtleSkgPT4ge1xuICAgICAgaWYgKGxvb2t1cC5oYXMoa2V5KSkge1xuICAgICAgICBmcmFnLmFwcGVuZENoaWxkKGxvb2t1cC5nZXQoa2V5KSk7XG4gICAgICAgIGxvb2t1cC5kZWxldGUoa2V5KTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBsb29rdXAuZm9yRWFjaCgoYnRuKSA9PiBmcmFnLmFwcGVuZENoaWxkKGJ0bikpO1xuICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5pbm5lckhUTUwgPSBcIlwiO1xuICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5hcHBlbmRDaGlsZChmcmFnKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGZvcm1hdFBlcmYoZCkge1xuICAgIGNvbnN0IGJpdHMgPSBbXTtcbiAgICBpZiAoZCAmJiB0eXBlb2YgZC5jcHUgIT09IFwidW5kZWZpbmVkXCIpIHtcbiAgICAgIGNvbnN0IGNwdSA9IE51bWJlcihkLmNwdSk7XG4gICAgICBpZiAoIU51bWJlci5pc05hTihjcHUpKSB7XG4gICAgICAgIGJpdHMucHVzaChgQ1BVICR7Y3B1LnRvRml4ZWQoMCl9JWApO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAoZCAmJiB0eXBlb2YgZC50dGZiX21zICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICBjb25zdCB0dGZiID0gTnVtYmVyKGQudHRmYl9tcyk7XG4gICAgICBpZiAoIU51bWJlci5pc05hTih0dGZiKSkge1xuICAgICAgICBiaXRzLnB1c2goYFRURkIgJHt0dGZifSBtc2ApO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gYml0cy5qb2luKFwiIFx1MjAyMiBcIikgfHwgXCJtaXNlIFx1MDBFMCBqb3VyXCI7XG4gIH1cblxuICBmdW5jdGlvbiBhdHRhY2hFdmVudHMoKSB7XG4gICAgaWYgKGVsZW1lbnRzLmNvbXBvc2VyKSB7XG4gICAgICBlbGVtZW50cy5jb21wb3Nlci5hZGRFdmVudExpc3RlbmVyKFwic3VibWl0XCIsIChldmVudCkgPT4ge1xuICAgICAgICBldmVudC5wcmV2ZW50RGVmYXVsdCgpO1xuICAgICAgICBjb25zdCB0ZXh0ID0gKGVsZW1lbnRzLnByb21wdC52YWx1ZSB8fCBcIlwiKS50cmltKCk7XG4gICAgICAgIGVtaXQoXCJzdWJtaXRcIiwgeyB0ZXh0IH0pO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLnF1aWNrQWN0aW9ucykge1xuICAgICAgZWxlbWVudHMucXVpY2tBY3Rpb25zLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgY29uc3QgdGFyZ2V0ID0gZXZlbnQudGFyZ2V0O1xuICAgICAgICBpZiAoISh0YXJnZXQgaW5zdGFuY2VvZiBIVE1MQnV0dG9uRWxlbWVudCkpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgYWN0aW9uID0gdGFyZ2V0LmRhdGFzZXQuYWN0aW9uO1xuICAgICAgICBpZiAoIWFjdGlvbikge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuICAgICAgICBlbWl0KFwicXVpY2stYWN0aW9uXCIsIHsgYWN0aW9uIH0pO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLmZpbHRlcklucHV0KSB7XG4gICAgICBlbGVtZW50cy5maWx0ZXJJbnB1dC5hZGRFdmVudExpc3RlbmVyKFwiaW5wdXRcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGVtaXQoXCJmaWx0ZXItY2hhbmdlXCIsIHsgdmFsdWU6IGV2ZW50LnRhcmdldC52YWx1ZSB8fCBcIlwiIH0pO1xuICAgICAgfSk7XG4gICAgICBlbGVtZW50cy5maWx0ZXJJbnB1dC5hZGRFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgaWYgKGV2ZW50LmtleSA9PT0gXCJFc2NhcGVcIikge1xuICAgICAgICAgIGV2ZW50LnByZXZlbnREZWZhdWx0KCk7XG4gICAgICAgICAgZW1pdChcImZpbHRlci1jbGVhclwiKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLmZpbHRlckNsZWFyKSB7XG4gICAgICBlbGVtZW50cy5maWx0ZXJDbGVhci5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgICAgICBlbWl0KFwiZmlsdGVyLWNsZWFyXCIpO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLmV4cG9ydEpzb24pIHtcbiAgICAgIGVsZW1lbnRzLmV4cG9ydEpzb24uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+XG4gICAgICAgIGVtaXQoXCJleHBvcnRcIiwgeyBmb3JtYXQ6IFwianNvblwiIH0pLFxuICAgICAgKTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLmV4cG9ydE1hcmtkb3duKSB7XG4gICAgICBlbGVtZW50cy5leHBvcnRNYXJrZG93bi5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT5cbiAgICAgICAgZW1pdChcImV4cG9ydFwiLCB7IGZvcm1hdDogXCJtYXJrZG93blwiIH0pLFxuICAgICAgKTtcbiAgICB9XG4gICAgaWYgKGVsZW1lbnRzLmV4cG9ydENvcHkpIHtcbiAgICAgIGVsZW1lbnRzLmV4cG9ydENvcHkuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IGVtaXQoXCJleHBvcnQtY29weVwiKSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLnByb21wdCkge1xuICAgICAgZWxlbWVudHMucHJvbXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJpbnB1dFwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgdXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgICAgICBhdXRvc2l6ZVByb21wdCgpO1xuICAgICAgICBjb25zdCB2YWx1ZSA9IGV2ZW50LnRhcmdldC52YWx1ZSB8fCBcIlwiO1xuICAgICAgICBpZiAoIXZhbHVlLnRyaW0oKSkge1xuICAgICAgICAgIHNldENvbXBvc2VyU3RhdHVzSWRsZSgpO1xuICAgICAgICB9XG4gICAgICAgIGVtaXQoXCJwcm9tcHQtaW5wdXRcIiwgeyB2YWx1ZSB9KTtcbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMucHJvbXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJwYXN0ZVwiLCAoKSA9PiB7XG4gICAgICAgIHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgICB1cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICAgICAgYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgICAgICBlbWl0KFwicHJvbXB0LWlucHV0XCIsIHsgdmFsdWU6IGVsZW1lbnRzLnByb21wdC52YWx1ZSB8fCBcIlwiIH0pO1xuICAgICAgICB9LCAwKTtcbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMucHJvbXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJrZXlkb3duXCIsIChldmVudCkgPT4ge1xuICAgICAgICBpZiAoKGV2ZW50LmN0cmxLZXkgfHwgZXZlbnQubWV0YUtleSkgJiYgZXZlbnQua2V5ID09PSBcIkVudGVyXCIpIHtcbiAgICAgICAgICBldmVudC5wcmV2ZW50RGVmYXVsdCgpO1xuICAgICAgICAgIGVtaXQoXCJzdWJtaXRcIiwgeyB0ZXh0OiAoZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCIpLnRyaW0oKSB9KTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBlbGVtZW50cy5wcm9tcHQuYWRkRXZlbnRMaXN0ZW5lcihcImZvY3VzXCIsICgpID0+IHtcbiAgICAgICAgc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJSXHUwMEU5ZGlnZXogdm90cmUgbWVzc2FnZSwgcHVpcyBDdHJsK0VudHJcdTAwRTllIHBvdXIgbCdlbnZveWVyLlwiLFxuICAgICAgICAgIFwiaW5mb1wiLFxuICAgICAgICApO1xuICAgICAgICBzY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy50cmFuc2NyaXB0KSB7XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJzY3JvbGxcIiwgKCkgPT4ge1xuICAgICAgICBpZiAoaXNBdEJvdHRvbSgpKSB7XG4gICAgICAgICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHNob3dTY3JvbGxCdXR0b24oKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRzLnNjcm9sbEJvdHRvbSkge1xuICAgICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiB0cnVlIH0pO1xuICAgICAgICBpZiAoZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICAgICAgZWxlbWVudHMucHJvbXB0LmZvY3VzKCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKFwicmVzaXplXCIsICgpID0+IHtcbiAgICAgIGlmIChpc0F0Qm90dG9tKCkpIHtcbiAgICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IGZhbHNlIH0pO1xuICAgICAgfVxuICAgIH0pO1xuXG4gICAgdXBkYXRlTmV0d29ya1N0YXR1cygpO1xuICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKFwib25saW5lXCIsICgpID0+IHtcbiAgICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMoKTtcbiAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcIkNvbm5leGlvbiByXHUwMEU5c2VhdSByZXN0YXVyXHUwMEU5ZS5cIiwgXCJpbmZvXCIpO1xuICAgIH0pO1xuICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKFwib2ZmbGluZVwiLCAoKSA9PiB7XG4gICAgICB1cGRhdGVOZXR3b3JrU3RhdHVzKCk7XG4gICAgICBhbm5vdW5jZUNvbm5lY3Rpb24oXCJDb25uZXhpb24gclx1MDBFOXNlYXUgcGVyZHVlLlwiLCBcImRhbmdlclwiKTtcbiAgICB9KTtcblxuICAgIGNvbnN0IHRvZ2dsZUJ0biA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwidG9nZ2xlLWRhcmstbW9kZVwiKTtcbiAgICBjb25zdCBkYXJrTW9kZUtleSA9IFwiZGFyay1tb2RlXCI7XG5cbiAgICBmdW5jdGlvbiBhcHBseURhcmtNb2RlKGVuYWJsZWQpIHtcbiAgICAgIGRvY3VtZW50LmJvZHkuY2xhc3NMaXN0LnRvZ2dsZShcImRhcmstbW9kZVwiLCBlbmFibGVkKTtcbiAgICAgIGlmICh0b2dnbGVCdG4pIHtcbiAgICAgICAgdG9nZ2xlQnRuLnRleHRDb250ZW50ID0gZW5hYmxlZCA/IFwiTW9kZSBDbGFpclwiIDogXCJNb2RlIFNvbWJyZVwiO1xuICAgICAgICB0b2dnbGVCdG4uc2V0QXR0cmlidXRlKFwiYXJpYS1wcmVzc2VkXCIsIGVuYWJsZWQgPyBcInRydWVcIiA6IFwiZmFsc2VcIik7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIGFwcGx5RGFya01vZGUod2luZG93LmxvY2FsU3RvcmFnZS5nZXRJdGVtKGRhcmtNb2RlS2V5KSA9PT0gXCIxXCIpO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHJlYWQgZGFyayBtb2RlIHByZWZlcmVuY2VcIiwgZXJyKTtcbiAgICB9XG5cbiAgICBpZiAodG9nZ2xlQnRuKSB7XG4gICAgICB0b2dnbGVCdG4uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgY29uc3QgZW5hYmxlZCA9ICFkb2N1bWVudC5ib2R5LmNsYXNzTGlzdC5jb250YWlucyhcImRhcmstbW9kZVwiKTtcbiAgICAgICAgYXBwbHlEYXJrTW9kZShlbmFibGVkKTtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICB3aW5kb3cubG9jYWxTdG9yYWdlLnNldEl0ZW0oZGFya01vZGVLZXksIGVuYWJsZWQgPyBcIjFcIiA6IFwiMFwiKTtcbiAgICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHBlcnNpc3QgZGFyayBtb2RlIHByZWZlcmVuY2VcIiwgZXJyKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gaW5pdGlhbGlzZSgpIHtcbiAgICBzZXREaWFnbm9zdGljcyh7IGNvbm5lY3RlZEF0OiBudWxsLCBsYXN0TWVzc2FnZUF0OiBudWxsLCBsYXRlbmN5TXM6IG51bGwgfSk7XG4gICAgdXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgIGF1dG9zaXplUHJvbXB0KCk7XG4gICAgc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCk7XG4gICAgYXR0YWNoRXZlbnRzKCk7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGVsZW1lbnRzLFxuICAgIG9uLFxuICAgIGVtaXQsXG4gICAgaW5pdGlhbGlzZSxcbiAgICByZW5kZXJIaXN0b3J5LFxuICAgIGFwcGVuZE1lc3NhZ2UsXG4gICAgc2V0QnVzeSxcbiAgICBzaG93RXJyb3IsXG4gICAgaGlkZUVycm9yLFxuICAgIHNldENvbXBvc2VyU3RhdHVzLFxuICAgIHNldENvbXBvc2VyU3RhdHVzSWRsZSxcbiAgICBzY2hlZHVsZUNvbXBvc2VySWRsZSxcbiAgICB1cGRhdGVQcm9tcHRNZXRyaWNzLFxuICAgIGF1dG9zaXplUHJvbXB0LFxuICAgIHN0YXJ0U3RyZWFtLFxuICAgIGFwcGVuZFN0cmVhbSxcbiAgICBlbmRTdHJlYW0sXG4gICAgYW5ub3VuY2VDb25uZWN0aW9uLFxuICAgIHVwZGF0ZUNvbm5lY3Rpb25NZXRhLFxuICAgIHNldERpYWdub3N0aWNzLFxuICAgIGFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyxcbiAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIsXG4gICAgcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIsXG4gICAgY2xlYXJUcmFuc2NyaXB0RmlsdGVyLFxuICAgIHNldFdzU3RhdHVzLFxuICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMsXG4gICAgc2Nyb2xsVG9Cb3R0b20sXG4gICAgc2V0IGRpYWdub3N0aWNzKHZhbHVlKSB7XG4gICAgICBPYmplY3QuYXNzaWduKGRpYWdub3N0aWNzLCB2YWx1ZSk7XG4gICAgfSxcbiAgICBnZXQgZGlhZ25vc3RpY3MoKSB7XG4gICAgICByZXR1cm4geyAuLi5kaWFnbm9zdGljcyB9O1xuICAgIH0sXG4gICAgZm9ybWF0VGltZXN0YW1wLFxuICAgIG5vd0lTTyxcbiAgICBmb3JtYXRQZXJmLFxuICAgIGlzU3RyZWFtaW5nLFxuICAgIGhhc1N0cmVhbUJ1ZmZlcixcbiAgfTtcbn1cbiIsICJjb25zdCBERUZBVUxUX1NUT1JBR0VfS0VZID0gXCJtb25nYXJzX2p3dFwiO1xuXG5mdW5jdGlvbiBoYXNMb2NhbFN0b3JhZ2UoKSB7XG4gIHRyeSB7XG4gICAgcmV0dXJuIHR5cGVvZiB3aW5kb3cgIT09IFwidW5kZWZpbmVkXCIgJiYgQm9vbGVhbih3aW5kb3cubG9jYWxTdG9yYWdlKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS53YXJuKFwiQWNjZXNzaW5nIGxvY2FsU3RvcmFnZSBmYWlsZWRcIiwgZXJyKTtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUF1dGhTZXJ2aWNlKGNvbmZpZyA9IHt9KSB7XG4gIGNvbnN0IHN0b3JhZ2VLZXkgPSBjb25maWcuc3RvcmFnZUtleSB8fCBERUZBVUxUX1NUT1JBR0VfS0VZO1xuICBsZXQgZmFsbGJhY2tUb2tlbiA9XG4gICAgdHlwZW9mIGNvbmZpZy50b2tlbiA9PT0gXCJzdHJpbmdcIiAmJiBjb25maWcudG9rZW4udHJpbSgpICE9PSBcIlwiXG4gICAgICA/IGNvbmZpZy50b2tlblxuICAgICAgOiB1bmRlZmluZWQ7XG5cbiAgZnVuY3Rpb24gcGVyc2lzdFRva2VuKHRva2VuKSB7XG4gICAgaWYgKCF0b2tlbikge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBmYWxsYmFja1Rva2VuID0gdG9rZW47XG5cbiAgICBpZiAoIWhhc0xvY2FsU3RvcmFnZSgpKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2Uuc2V0SXRlbShzdG9yYWdlS2V5LCB0b2tlbik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcGVyc2lzdCBKV1QgaW4gbG9jYWxTdG9yYWdlXCIsIGVycik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gcmVhZFN0b3JlZFRva2VuKCkge1xuICAgIGlmICghaGFzTG9jYWxTdG9yYWdlKCkpIHtcbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHN0b3JlZCA9IHdpbmRvdy5sb2NhbFN0b3JhZ2UuZ2V0SXRlbShzdG9yYWdlS2V5KTtcbiAgICAgIHJldHVybiBzdG9yZWQgfHwgdW5kZWZpbmVkO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHJlYWQgSldUIGZyb20gbG9jYWxTdG9yYWdlXCIsIGVycik7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGNsZWFyVG9rZW4oKSB7XG4gICAgZmFsbGJhY2tUb2tlbiA9IHVuZGVmaW5lZDtcblxuICAgIGlmICghaGFzTG9jYWxTdG9yYWdlKCkpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0cnkge1xuICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKHN0b3JhZ2VLZXkpO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIGNsZWFyIEpXVCBmcm9tIGxvY2FsU3RvcmFnZVwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIGlmIChmYWxsYmFja1Rva2VuKSB7XG4gICAgcGVyc2lzdFRva2VuKGZhbGxiYWNrVG9rZW4pO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZ2V0Snd0KCkge1xuICAgIGNvbnN0IHN0b3JlZCA9IHJlYWRTdG9yZWRUb2tlbigpO1xuICAgIGlmIChzdG9yZWQpIHtcbiAgICAgIHJldHVybiBzdG9yZWQ7XG4gICAgfVxuICAgIGlmIChmYWxsYmFja1Rva2VuKSB7XG4gICAgICByZXR1cm4gZmFsbGJhY2tUb2tlbjtcbiAgICB9XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgYE1pc3NpbmcgSldUIChzdG9yZSBpdCBpbiBsb2NhbFN0b3JhZ2UgYXMgJyR7c3RvcmFnZUtleX0nIG9yIHByb3ZpZGUgaXQgaW4gdGhlIGNoYXQgY29uZmlnKS5gLFxuICAgICk7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGdldEp3dCxcbiAgICBwZXJzaXN0VG9rZW4sXG4gICAgY2xlYXJUb2tlbixcbiAgICBzdG9yYWdlS2V5LFxuICB9O1xufVxuIiwgImltcG9ydCB7IGFwaVVybCB9IGZyb20gXCIuLi9jb25maWcuanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUh0dHBTZXJ2aWNlKHsgY29uZmlnLCBhdXRoIH0pIHtcbiAgYXN5bmMgZnVuY3Rpb24gYXV0aG9yaXNlZEZldGNoKHBhdGgsIG9wdGlvbnMgPSB7fSkge1xuICAgIGxldCBqd3Q7XG4gICAgdHJ5IHtcbiAgICAgIGp3dCA9IGF3YWl0IGF1dGguZ2V0Snd0KCk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAvLyBTdXJmYWNlIGEgY29uc2lzdGVudCBlcnJvciBhbmQgcHJlc2VydmUgYWJvcnQgc2VtYW50aWNzXG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXCJBdXRob3JpemF0aW9uIGZhaWxlZDogbWlzc2luZyBvciB1bnJlYWRhYmxlIEpXVFwiKTtcbiAgICB9XG4gICAgY29uc3QgaGVhZGVycyA9IG5ldyBIZWFkZXJzKG9wdGlvbnMuaGVhZGVycyB8fCB7fSk7XG4gICAgaWYgKCFoZWFkZXJzLmhhcyhcIkF1dGhvcml6YXRpb25cIikpIHtcbiAgICAgIGhlYWRlcnMuc2V0KFwiQXV0aG9yaXphdGlvblwiLCBgQmVhcmVyICR7and0fWApO1xuICAgIH1cbiAgICByZXR1cm4gZmV0Y2goYXBpVXJsKGNvbmZpZywgcGF0aCksIHsgLi4ub3B0aW9ucywgaGVhZGVycyB9KTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGZldGNoVGlja2V0KCkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL2F1dGgvd3MvdGlja2V0XCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgfSk7XG4gICAgaWYgKCFyZXNwLm9rKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFRpY2tldCBlcnJvcjogJHtyZXNwLnN0YXR1c31gKTtcbiAgICB9XG4gICAgY29uc3QgYm9keSA9IGF3YWl0IHJlc3AuanNvbigpO1xuICAgIGlmICghYm9keSB8fCAhYm9keS50aWNrZXQpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcIlRpY2tldCByZXNwb25zZSBpbnZhbGlkZVwiKTtcbiAgICB9XG4gICAgcmV0dXJuIGJvZHkudGlja2V0O1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gcG9zdENoYXQobWVzc2FnZSkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL2NvbnZlcnNhdGlvbi9jaGF0XCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgICBoZWFkZXJzOiB7IFwiQ29udGVudC1UeXBlXCI6IFwiYXBwbGljYXRpb24vanNvblwiIH0sXG4gICAgICBib2R5OiBKU09OLnN0cmluZ2lmeSh7IG1lc3NhZ2UgfSksXG4gICAgfSk7XG4gICAgaWYgKCFyZXNwLm9rKSB7XG4gICAgICBjb25zdCBwYXlsb2FkID0gYXdhaXQgcmVzcC50ZXh0KCk7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEhUVFAgJHtyZXNwLnN0YXR1c306ICR7cGF5bG9hZH1gKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3A7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBwb3N0U3VnZ2VzdGlvbnMocHJvbXB0KSB7XG4gICAgY29uc3QgcmVzcCA9IGF3YWl0IGF1dGhvcmlzZWRGZXRjaChcIi9hcGkvdjEvdWkvc3VnZ2VzdGlvbnNcIiwge1xuICAgICAgbWV0aG9kOiBcIlBPU1RcIixcbiAgICAgIGhlYWRlcnM6IHsgXCJDb250ZW50LVR5cGVcIjogXCJhcHBsaWNhdGlvbi9qc29uXCIgfSxcbiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHtcbiAgICAgICAgcHJvbXB0LFxuICAgICAgICBhY3Rpb25zOiBbXCJjb2RlXCIsIFwic3VtbWFyaXplXCIsIFwiZXhwbGFpblwiXSxcbiAgICAgIH0pLFxuICAgIH0pO1xuICAgIGlmICghcmVzcC5vaykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBTdWdnZXN0aW9uIGVycm9yOiAke3Jlc3Auc3RhdHVzfWApO1xuICAgIH1cbiAgICByZXR1cm4gcmVzcC5qc29uKCk7XG4gIH1cblxuICByZXR1cm4ge1xuICAgIGZldGNoVGlja2V0LFxuICAgIHBvc3RDaGF0LFxuICAgIHBvc3RTdWdnZXN0aW9ucyxcbiAgfTtcbn1cbiIsICJpbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBidWlsZEV4cG9ydEZpbGVuYW1lKGV4dGVuc2lvbikge1xuICBjb25zdCBzdGFtcCA9IG5vd0lTTygpLnJlcGxhY2UoL1s6Ll0vZywgXCItXCIpO1xuICByZXR1cm4gYG1vbmdhcnMtY2hhdC0ke3N0YW1wfS4ke2V4dGVuc2lvbn1gO1xufVxuXG5mdW5jdGlvbiBidWlsZE1hcmtkb3duRXhwb3J0KGl0ZW1zKSB7XG4gIGNvbnN0IGxpbmVzID0gW1wiIyBIaXN0b3JpcXVlIGRlIGNvbnZlcnNhdGlvbiBtb25HQVJTXCIsIFwiXCJdO1xuICBpdGVtcy5mb3JFYWNoKChpdGVtKSA9PiB7XG4gICAgY29uc3Qgcm9sZSA9IGl0ZW0ucm9sZSA/IGl0ZW0ucm9sZS50b1VwcGVyQ2FzZSgpIDogXCJNRVNTQUdFXCI7XG4gICAgbGluZXMucHVzaChgIyMgJHtyb2xlfWApO1xuICAgIGlmIChpdGVtLnRpbWVzdGFtcCkge1xuICAgICAgbGluZXMucHVzaChgKkhvcm9kYXRhZ2VcdTAwQTA6KiAke2l0ZW0udGltZXN0YW1wfWApO1xuICAgIH1cbiAgICBpZiAoaXRlbS5tZXRhZGF0YSAmJiBPYmplY3Qua2V5cyhpdGVtLm1ldGFkYXRhKS5sZW5ndGggPiAwKSB7XG4gICAgICBPYmplY3QuZW50cmllcyhpdGVtLm1ldGFkYXRhKS5mb3JFYWNoKChba2V5LCB2YWx1ZV0pID0+IHtcbiAgICAgICAgbGluZXMucHVzaChgKiR7a2V5fVx1MDBBMDoqICR7dmFsdWV9YCk7XG4gICAgICB9KTtcbiAgICB9XG4gICAgbGluZXMucHVzaChcIlwiKTtcbiAgICBsaW5lcy5wdXNoKGl0ZW0udGV4dCB8fCBcIlwiKTtcbiAgICBsaW5lcy5wdXNoKFwiXCIpO1xuICB9KTtcbiAgcmV0dXJuIGxpbmVzLmpvaW4oXCJcXG5cIik7XG59XG5cbmZ1bmN0aW9uIGRvd25sb2FkQmxvYihmaWxlbmFtZSwgdGV4dCwgdHlwZSkge1xuICBpZiAoIXdpbmRvdy5VUkwgfHwgdHlwZW9mIHdpbmRvdy5VUkwuY3JlYXRlT2JqZWN0VVJMICE9PSBcImZ1bmN0aW9uXCIpIHtcbiAgICBjb25zb2xlLndhcm4oXCJCbG9iIGV4cG9ydCB1bnN1cHBvcnRlZCBpbiB0aGlzIGVudmlyb25tZW50XCIpO1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuICBjb25zdCBibG9iID0gbmV3IEJsb2IoW3RleHRdLCB7IHR5cGUgfSk7XG4gIGNvbnN0IHVybCA9IFVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG4gIGNvbnN0IGFuY2hvciA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJhXCIpO1xuICBhbmNob3IuaHJlZiA9IHVybDtcbiAgYW5jaG9yLmRvd25sb2FkID0gZmlsZW5hbWU7XG4gIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoYW5jaG9yKTtcbiAgYW5jaG9yLmNsaWNrKCk7XG4gIGRvY3VtZW50LmJvZHkucmVtb3ZlQ2hpbGQoYW5jaG9yKTtcbiAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4gVVJMLnJldm9rZU9iamVjdFVSTCh1cmwpLCAwKTtcbiAgcmV0dXJuIHRydWU7XG59XG5cbmFzeW5jIGZ1bmN0aW9uIGNvcHlUb0NsaXBib2FyZCh0ZXh0KSB7XG4gIGlmICghdGV4dCkgcmV0dXJuIGZhbHNlO1xuICB0cnkge1xuICAgIGlmIChuYXZpZ2F0b3IuY2xpcGJvYXJkICYmIG5hdmlnYXRvci5jbGlwYm9hcmQud3JpdGVUZXh0KSB7XG4gICAgICBhd2FpdCBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCh0ZXh0KTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgdGV4dGFyZWEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidGV4dGFyZWFcIik7XG4gICAgICB0ZXh0YXJlYS52YWx1ZSA9IHRleHQ7XG4gICAgICB0ZXh0YXJlYS5zZXRBdHRyaWJ1dGUoXCJyZWFkb25seVwiLCBcInJlYWRvbmx5XCIpO1xuICAgICAgdGV4dGFyZWEuc3R5bGUucG9zaXRpb24gPSBcImFic29sdXRlXCI7XG4gICAgICB0ZXh0YXJlYS5zdHlsZS5sZWZ0ID0gXCItOTk5OXB4XCI7XG4gICAgICBkb2N1bWVudC5ib2R5LmFwcGVuZENoaWxkKHRleHRhcmVhKTtcbiAgICAgIHRleHRhcmVhLnNlbGVjdCgpO1xuICAgICAgZG9jdW1lbnQuZXhlY0NvbW1hbmQoXCJjb3B5XCIpO1xuICAgICAgZG9jdW1lbnQuYm9keS5yZW1vdmVDaGlsZCh0ZXh0YXJlYSk7XG4gICAgfVxuICAgIHJldHVybiB0cnVlO1xuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLndhcm4oXCJDb3B5IGNvbnZlcnNhdGlvbiBmYWlsZWRcIiwgZXJyKTtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUV4cG9ydGVyKHsgdGltZWxpbmVTdG9yZSwgYW5ub3VuY2UgfSkge1xuICBmdW5jdGlvbiBjb2xsZWN0VHJhbnNjcmlwdCgpIHtcbiAgICByZXR1cm4gdGltZWxpbmVTdG9yZS5jb2xsZWN0KCk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBleHBvcnRDb252ZXJzYXRpb24oZm9ybWF0KSB7XG4gICAgY29uc3QgaXRlbXMgPSBjb2xsZWN0VHJhbnNjcmlwdCgpO1xuICAgIGlmICghaXRlbXMubGVuZ3RoKSB7XG4gICAgICBhbm5vdW5jZShcIkF1Y3VuIG1lc3NhZ2UgXHUwMEUwIGV4cG9ydGVyLlwiLCBcIndhcm5pbmdcIik7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmIChmb3JtYXQgPT09IFwianNvblwiKSB7XG4gICAgICBjb25zdCBwYXlsb2FkID0ge1xuICAgICAgICBleHBvcnRlZF9hdDogbm93SVNPKCksXG4gICAgICAgIGNvdW50OiBpdGVtcy5sZW5ndGgsXG4gICAgICAgIGl0ZW1zLFxuICAgICAgfTtcbiAgICAgIGlmIChcbiAgICAgICAgZG93bmxvYWRCbG9iKFxuICAgICAgICAgIGJ1aWxkRXhwb3J0RmlsZW5hbWUoXCJqc29uXCIpLFxuICAgICAgICAgIEpTT04uc3RyaW5naWZ5KHBheWxvYWQsIG51bGwsIDIpLFxuICAgICAgICAgIFwiYXBwbGljYXRpb24vanNvblwiLFxuICAgICAgICApXG4gICAgICApIHtcbiAgICAgICAgYW5ub3VuY2UoXCJFeHBvcnQgSlNPTiBnXHUwMEU5blx1MDBFOXJcdTAwRTkuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGFubm91bmNlKFwiRXhwb3J0IG5vbiBzdXBwb3J0XHUwMEU5IGRhbnMgY2UgbmF2aWdhdGV1ci5cIiwgXCJkYW5nZXJcIik7XG4gICAgICB9XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmIChmb3JtYXQgPT09IFwibWFya2Rvd25cIikge1xuICAgICAgaWYgKFxuICAgICAgICBkb3dubG9hZEJsb2IoXG4gICAgICAgICAgYnVpbGRFeHBvcnRGaWxlbmFtZShcIm1kXCIpLFxuICAgICAgICAgIGJ1aWxkTWFya2Rvd25FeHBvcnQoaXRlbXMpLFxuICAgICAgICAgIFwidGV4dC9tYXJrZG93blwiLFxuICAgICAgICApXG4gICAgICApIHtcbiAgICAgICAgYW5ub3VuY2UoXCJFeHBvcnQgTWFya2Rvd24gZ1x1MDBFOW5cdTAwRTlyXHUwMEU5LlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBub24gc3VwcG9ydFx1MDBFOSBkYW5zIGNlIG5hdmlnYXRldXIuXCIsIFwiZGFuZ2VyXCIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGNvcHlDb252ZXJzYXRpb25Ub0NsaXBib2FyZCgpIHtcbiAgICBjb25zdCBpdGVtcyA9IGNvbGxlY3RUcmFuc2NyaXB0KCk7XG4gICAgaWYgKCFpdGVtcy5sZW5ndGgpIHtcbiAgICAgIGFubm91bmNlKFwiQXVjdW4gbWVzc2FnZSBcdTAwRTAgY29waWVyLlwiLCBcIndhcm5pbmdcIik7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IHRleHQgPSBidWlsZE1hcmtkb3duRXhwb3J0KGl0ZW1zKTtcbiAgICBpZiAoYXdhaXQgY29weVRvQ2xpcGJvYXJkKHRleHQpKSB7XG4gICAgICBhbm5vdW5jZShcIkNvbnZlcnNhdGlvbiBjb3BpXHUwMEU5ZSBhdSBwcmVzc2UtcGFwaWVycy5cIiwgXCJzdWNjZXNzXCIpO1xuICAgIH0gZWxzZSB7XG4gICAgICBhbm5vdW5jZShcIkltcG9zc2libGUgZGUgY29waWVyIGxhIGNvbnZlcnNhdGlvbi5cIiwgXCJkYW5nZXJcIik7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBleHBvcnRDb252ZXJzYXRpb24sXG4gICAgY29weUNvbnZlcnNhdGlvblRvQ2xpcGJvYXJkLFxuICB9O1xufVxuIiwgImltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuLi91dGlscy90aW1lLmpzXCI7XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTb2NrZXRDbGllbnQoeyBjb25maWcsIGh0dHAsIHVpLCBvbkV2ZW50IH0pIHtcbiAgbGV0IHdzO1xuICBsZXQgd3NIQmVhdDtcbiAgbGV0IHJlY29ubmVjdEJhY2tvZmYgPSA1MDA7XG4gIGNvbnN0IEJBQ0tPRkZfTUFYID0gODAwMDtcbiAgbGV0IHJldHJ5VGltZXIgPSBudWxsO1xuICBsZXQgZGlzcG9zZWQgPSBmYWxzZTtcblxuICBmdW5jdGlvbiBjbGVhckhlYXJ0YmVhdCgpIHtcbiAgICBpZiAod3NIQmVhdCkge1xuICAgICAgY2xlYXJJbnRlcnZhbCh3c0hCZWF0KTtcbiAgICAgIHdzSEJlYXQgPSBudWxsO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlUmVjb25uZWN0KGRlbGF5QmFzZSkge1xuICAgIGlmIChkaXNwb3NlZCkge1xuICAgICAgcmV0dXJuIDA7XG4gICAgfVxuICAgIGNvbnN0IGppdHRlciA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIDI1MCk7XG4gICAgY29uc3QgZGVsYXkgPSBNYXRoLm1pbihCQUNLT0ZGX01BWCwgZGVsYXlCYXNlICsgaml0dGVyKTtcbiAgICBpZiAocmV0cnlUaW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHJldHJ5VGltZXIpO1xuICAgIH1cbiAgICByZXRyeVRpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgcmV0cnlUaW1lciA9IG51bGw7XG4gICAgICByZWNvbm5lY3RCYWNrb2ZmID0gTWF0aC5taW4oXG4gICAgICAgIEJBQ0tPRkZfTUFYLFxuICAgICAgICBNYXRoLm1heCg1MDAsIHJlY29ubmVjdEJhY2tvZmYgKiAyKSxcbiAgICAgICk7XG4gICAgICB2b2lkIG9wZW5Tb2NrZXQoKTtcbiAgICB9LCBkZWxheSk7XG4gICAgcmV0dXJuIGRlbGF5O1xuICB9XG5cbiAgZnVuY3Rpb24gc2FmZVNlbmQob2JqKSB7XG4gICAgdHJ5IHtcbiAgICAgIGlmICh3cyAmJiB3cy5yZWFkeVN0YXRlID09PSBXZWJTb2NrZXQuT1BFTikge1xuICAgICAgICB3cy5zZW5kKEpTT04uc3RyaW5naWZ5KG9iaikpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHNlbmQgb3ZlciBXZWJTb2NrZXRcIiwgZXJyKTtcbiAgICB9XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBvcGVuU29ja2V0KCkge1xuICAgIGlmIChkaXNwb3NlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIk9idGVudGlvbiBkXHUyMDE5dW4gdGlja2V0IGRlIGNvbm5leGlvblx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICBjb25zdCB0aWNrZXQgPSBhd2FpdCBodHRwLmZldGNoVGlja2V0KCk7XG4gICAgICBpZiAoZGlzcG9zZWQpIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuXG4gICAgICBjb25zdCB3c1VybCA9IG5ldyBVUkwoXCIvd3MvY2hhdC9cIiwgY29uZmlnLmJhc2VVcmwpO1xuICAgICAgd3NVcmwucHJvdG9jb2wgPSBjb25maWcuYmFzZVVybC5wcm90b2NvbCA9PT0gXCJodHRwczpcIiA/IFwid3NzOlwiIDogXCJ3czpcIjtcbiAgICAgIHdzVXJsLnNlYXJjaFBhcmFtcy5zZXQoXCJ0XCIsIHRpY2tldCk7XG5cbiAgICAgIGlmICh3cykge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIHdzLmNsb3NlKCk7XG4gICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcIldlYlNvY2tldCBjbG9zZSBiZWZvcmUgcmVjb25uZWN0IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgICB9XG4gICAgICAgIHdzID0gbnVsbDtcbiAgICAgIH1cblxuICAgICAgd3MgPSBuZXcgV2ViU29ja2V0KHdzVXJsLnRvU3RyaW5nKCkpO1xuICAgICAgdWkuc2V0V3NTdGF0dXMoXCJjb25uZWN0aW5nXCIpO1xuICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEoXCJDb25uZXhpb24gYXUgc2VydmV1clx1MjAyNlwiLCBcImluZm9cIik7XG5cbiAgICAgIHdzLm9ub3BlbiA9ICgpID0+IHtcbiAgICAgICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGlmIChyZXRyeVRpbWVyKSB7XG4gICAgICAgICAgY2xlYXJUaW1lb3V0KHJldHJ5VGltZXIpO1xuICAgICAgICAgIHJldHJ5VGltZXIgPSBudWxsO1xuICAgICAgICB9XG4gICAgICAgIHJlY29ubmVjdEJhY2tvZmYgPSA1MDA7XG4gICAgICAgIGNvbnN0IGNvbm5lY3RlZEF0ID0gbm93SVNPKCk7XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwib25saW5lXCIpO1xuICAgICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcbiAgICAgICAgICBgQ29ubmVjdFx1MDBFOSBsZSAke3VpLmZvcm1hdFRpbWVzdGFtcChjb25uZWN0ZWRBdCl9YCxcbiAgICAgICAgICBcInN1Y2Nlc3NcIixcbiAgICAgICAgKTtcbiAgICAgICAgdWkuc2V0RGlhZ25vc3RpY3MoeyBjb25uZWN0ZWRBdCwgbGFzdE1lc3NhZ2VBdDogY29ubmVjdGVkQXQgfSk7XG4gICAgICAgIHVpLmhpZGVFcnJvcigpO1xuICAgICAgICBjbGVhckhlYXJ0YmVhdCgpO1xuICAgICAgICB3c0hCZWF0ID0gd2luZG93LnNldEludGVydmFsKCgpID0+IHtcbiAgICAgICAgICBzYWZlU2VuZCh7IHR5cGU6IFwiY2xpZW50LnBpbmdcIiwgdHM6IG5vd0lTTygpIH0pO1xuICAgICAgICB9LCAyMDAwMCk7XG4gICAgICAgIHVpLnNldENvbXBvc2VyU3RhdHVzKFwiQ29ubmVjdFx1MDBFOS4gVm91cyBwb3V2ZXogXHUwMEU5Y2hhbmdlci5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgICB1aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIH07XG5cbiAgICAgIHdzLm9ubWVzc2FnZSA9IChldnQpID0+IHtcbiAgICAgICAgY29uc3QgcmVjZWl2ZWRBdCA9IG5vd0lTTygpO1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGV2ID0gSlNPTi5wYXJzZShldnQuZGF0YSk7XG4gICAgICAgICAgdWkuc2V0RGlhZ25vc3RpY3MoeyBsYXN0TWVzc2FnZUF0OiByZWNlaXZlZEF0IH0pO1xuICAgICAgICAgIG9uRXZlbnQoZXYpO1xuICAgICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgICBjb25zb2xlLmVycm9yKFwiQmFkIGV2ZW50IHBheWxvYWRcIiwgZXJyLCBldnQuZGF0YSk7XG4gICAgICAgIH1cbiAgICAgIH07XG5cbiAgICAgIHdzLm9uY2xvc2UgPSAoKSA9PiB7XG4gICAgICAgIGNsZWFySGVhcnRiZWF0KCk7XG4gICAgICAgIHdzID0gbnVsbDtcbiAgICAgICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwib2ZmbGluZVwiKTtcbiAgICAgICAgdWkuc2V0RGlhZ25vc3RpY3MoeyBsYXRlbmN5TXM6IHVuZGVmaW5lZCB9KTtcbiAgICAgICAgY29uc3QgZGVsYXkgPSBzY2hlZHVsZVJlY29ubmVjdChyZWNvbm5lY3RCYWNrb2ZmKTtcbiAgICAgICAgY29uc3Qgc2Vjb25kcyA9IE1hdGgubWF4KDEsIE1hdGgucm91bmQoZGVsYXkgLyAxMDAwKSk7XG4gICAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFxuICAgICAgICAgIGBEXHUwMEU5Y29ubmVjdFx1MDBFOS4gTm91dmVsbGUgdGVudGF0aXZlIGRhbnMgJHtzZWNvbmRzfSBzXHUyMDI2YCxcbiAgICAgICAgICBcIndhcm5pbmdcIixcbiAgICAgICAgKTtcbiAgICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJDb25uZXhpb24gcGVyZHVlLiBSZWNvbm5leGlvbiBhdXRvbWF0aXF1ZVx1MjAyNlwiLFxuICAgICAgICAgIFwid2FybmluZ1wiLFxuICAgICAgICApO1xuICAgICAgICB1aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg2MDAwKTtcbiAgICAgIH07XG5cbiAgICAgIHdzLm9uZXJyb3IgPSAoZXJyKSA9PiB7XG4gICAgICAgIGNvbnNvbGUuZXJyb3IoXCJXZWJTb2NrZXQgZXJyb3JcIiwgZXJyKTtcbiAgICAgICAgaWYgKGRpc3Bvc2VkKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwiZXJyb3JcIiwgXCJFcnJldXIgV2ViU29ja2V0XCIpO1xuICAgICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIkVycmV1ciBXZWJTb2NrZXQgZFx1MDBFOXRlY3RcdTAwRTllLlwiLCBcImRhbmdlclwiKTtcbiAgICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJVbmUgZXJyZXVyIHJcdTAwRTlzZWF1IGVzdCBzdXJ2ZW51ZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUuZXJyb3IoZXJyKTtcbiAgICAgIGlmIChkaXNwb3NlZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBjb25zdCBtZXNzYWdlID0gZXJyIGluc3RhbmNlb2YgRXJyb3IgPyBlcnIubWVzc2FnZSA6IFN0cmluZyhlcnIpO1xuICAgICAgdWkuc2V0V3NTdGF0dXMoXCJlcnJvclwiLCBtZXNzYWdlKTtcbiAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKG1lc3NhZ2UsIFwiZGFuZ2VyXCIpO1xuICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgIFwiQ29ubmV4aW9uIGluZGlzcG9uaWJsZS4gTm91dmVsIGVzc2FpIGJpZW50XHUwMEY0dC5cIixcbiAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICk7XG4gICAgICB1aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg2MDAwKTtcbiAgICAgIHNjaGVkdWxlUmVjb25uZWN0KHJlY29ubmVjdEJhY2tvZmYpO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGRpc3Bvc2UoKSB7XG4gICAgZGlzcG9zZWQgPSB0cnVlO1xuICAgIGlmIChyZXRyeVRpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQocmV0cnlUaW1lcik7XG4gICAgICByZXRyeVRpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgY2xlYXJIZWFydGJlYXQoKTtcbiAgICBpZiAod3MpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIHdzLmNsb3NlKCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFwiV2ViU29ja2V0IGNsb3NlIGR1cmluZyBkaXNwb3NlIGZhaWxlZFwiLCBlcnIpO1xuICAgICAgfVxuICAgICAgd3MgPSBudWxsO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgb3Blbjogb3BlblNvY2tldCxcbiAgICBzZW5kOiBzYWZlU2VuZCxcbiAgICBkaXNwb3NlLFxuICB9O1xufVxuIiwgImV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTdWdnZXN0aW9uU2VydmljZSh7IGh0dHAsIHVpIH0pIHtcbiAgbGV0IHRpbWVyID0gbnVsbDtcblxuICBmdW5jdGlvbiBzY2hlZHVsZShwcm9tcHQpIHtcbiAgICBpZiAodGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dCh0aW1lcik7XG4gICAgfVxuICAgIHRpbWVyID0gd2luZG93LnNldFRpbWVvdXQoKCkgPT4gZmV0Y2hTdWdnZXN0aW9ucyhwcm9tcHQpLCAyMjApO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZmV0Y2hTdWdnZXN0aW9ucyhwcm9tcHQpIHtcbiAgICBpZiAoIXByb21wdCB8fCBwcm9tcHQudHJpbSgpLmxlbmd0aCA8IDMpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHBheWxvYWQgPSBhd2FpdCBodHRwLnBvc3RTdWdnZXN0aW9ucyhwcm9tcHQudHJpbSgpKTtcbiAgICAgIGlmIChwYXlsb2FkICYmIEFycmF5LmlzQXJyYXkocGF5bG9hZC5hY3Rpb25zKSkge1xuICAgICAgICB1aS5hcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcocGF5bG9hZC5hY3Rpb25zKTtcbiAgICAgIH1cbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUuZGVidWcoXCJBVUkgc3VnZ2VzdGlvbiBmZXRjaCBmYWlsZWRcIiwgZXJyKTtcbiAgICB9XG4gIH1cblxuICByZXR1cm4ge1xuICAgIHNjaGVkdWxlLFxuICB9O1xufVxuIiwgImltcG9ydCB7IHJlc29sdmVDb25maWcgfSBmcm9tIFwiLi9jb25maWcuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVRpbWVsaW5lU3RvcmUgfSBmcm9tIFwiLi9zdGF0ZS90aW1lbGluZVN0b3JlLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVDaGF0VWkgfSBmcm9tIFwiLi91aS9jaGF0VWkuanNcIjtcbmltcG9ydCB7IGNyZWF0ZUF1dGhTZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvYXV0aC5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlSHR0cFNlcnZpY2UgfSBmcm9tIFwiLi9zZXJ2aWNlcy9odHRwLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVFeHBvcnRlciB9IGZyb20gXCIuL3NlcnZpY2VzL2V4cG9ydGVyLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVTb2NrZXRDbGllbnQgfSBmcm9tIFwiLi9zZXJ2aWNlcy9zb2NrZXQuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlIH0gZnJvbSBcIi4vc2VydmljZXMvc3VnZ2VzdGlvbnMuanNcIjtcbmltcG9ydCB7IG5vd0lTTyB9IGZyb20gXCIuL3V0aWxzL3RpbWUuanNcIjtcblxuZnVuY3Rpb24gcXVlcnlFbGVtZW50cyhkb2MpIHtcbiAgY29uc3QgYnlJZCA9IChpZCkgPT4gZG9jLmdldEVsZW1lbnRCeUlkKGlkKTtcbiAgcmV0dXJuIHtcbiAgICB0cmFuc2NyaXB0OiBieUlkKFwidHJhbnNjcmlwdFwiKSxcbiAgICBjb21wb3NlcjogYnlJZChcImNvbXBvc2VyXCIpLFxuICAgIHByb21wdDogYnlJZChcInByb21wdFwiKSxcbiAgICBzZW5kOiBieUlkKFwic2VuZFwiKSxcbiAgICB3c1N0YXR1czogYnlJZChcIndzLXN0YXR1c1wiKSxcbiAgICBxdWlja0FjdGlvbnM6IGJ5SWQoXCJxdWljay1hY3Rpb25zXCIpLFxuICAgIGNvbm5lY3Rpb246IGJ5SWQoXCJjb25uZWN0aW9uXCIpLFxuICAgIGVycm9yQWxlcnQ6IGJ5SWQoXCJlcnJvci1hbGVydFwiKSxcbiAgICBlcnJvck1lc3NhZ2U6IGJ5SWQoXCJlcnJvci1tZXNzYWdlXCIpLFxuICAgIHNjcm9sbEJvdHRvbTogYnlJZChcInNjcm9sbC1ib3R0b21cIiksXG4gICAgY29tcG9zZXJTdGF0dXM6IGJ5SWQoXCJjb21wb3Nlci1zdGF0dXNcIiksXG4gICAgcHJvbXB0Q291bnQ6IGJ5SWQoXCJwcm9tcHQtY291bnRcIiksXG4gICAgY29ubmVjdGlvbk1ldGE6IGJ5SWQoXCJjb25uZWN0aW9uLW1ldGFcIiksXG4gICAgZmlsdGVySW5wdXQ6IGJ5SWQoXCJjaGF0LXNlYXJjaFwiKSxcbiAgICBmaWx0ZXJDbGVhcjogYnlJZChcImNoYXQtc2VhcmNoLWNsZWFyXCIpLFxuICAgIGZpbHRlckVtcHR5OiBieUlkKFwiZmlsdGVyLWVtcHR5XCIpLFxuICAgIGZpbHRlckhpbnQ6IGJ5SWQoXCJjaGF0LXNlYXJjaC1oaW50XCIpLFxuICAgIGV4cG9ydEpzb246IGJ5SWQoXCJleHBvcnQtanNvblwiKSxcbiAgICBleHBvcnRNYXJrZG93bjogYnlJZChcImV4cG9ydC1tYXJrZG93blwiKSxcbiAgICBleHBvcnRDb3B5OiBieUlkKFwiZXhwb3J0LWNvcHlcIiksXG4gICAgZGlhZ0Nvbm5lY3RlZDogYnlJZChcImRpYWctY29ubmVjdGVkXCIpLFxuICAgIGRpYWdMYXN0TWVzc2FnZTogYnlJZChcImRpYWctbGFzdC1tZXNzYWdlXCIpLFxuICAgIGRpYWdMYXRlbmN5OiBieUlkKFwiZGlhZy1sYXRlbmN5XCIpLFxuICAgIGRpYWdOZXR3b3JrOiBieUlkKFwiZGlhZy1uZXR3b3JrXCIpLFxuICB9O1xufVxuXG5mdW5jdGlvbiByZWFkSGlzdG9yeShkb2MpIHtcbiAgY29uc3QgaGlzdG9yeUVsZW1lbnQgPSBkb2MuZ2V0RWxlbWVudEJ5SWQoXCJjaGF0LWhpc3RvcnlcIik7XG4gIGlmICghaGlzdG9yeUVsZW1lbnQpIHtcbiAgICByZXR1cm4gW107XG4gIH1cbiAgY29uc3QgcGF5bG9hZCA9IGhpc3RvcnlFbGVtZW50LnRleHRDb250ZW50IHx8IFwibnVsbFwiO1xuICBoaXN0b3J5RWxlbWVudC5yZW1vdmUoKTtcbiAgdHJ5IHtcbiAgICBjb25zdCBwYXJzZWQgPSBKU09OLnBhcnNlKHBheWxvYWQpO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHBhcnNlZCkpIHtcbiAgICAgIHJldHVybiBwYXJzZWQ7XG4gICAgfVxuICAgIGlmIChwYXJzZWQgJiYgcGFyc2VkLmVycm9yKSB7XG4gICAgICByZXR1cm4geyBlcnJvcjogcGFyc2VkLmVycm9yIH07XG4gICAgfVxuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLmVycm9yKFwiVW5hYmxlIHRvIHBhcnNlIGNoYXQgaGlzdG9yeVwiLCBlcnIpO1xuICB9XG4gIHJldHVybiBbXTtcbn1cblxuZnVuY3Rpb24gZW5zdXJlRWxlbWVudHMoZWxlbWVudHMpIHtcbiAgcmV0dXJuIEJvb2xlYW4oZWxlbWVudHMudHJhbnNjcmlwdCAmJiBlbGVtZW50cy5jb21wb3NlciAmJiBlbGVtZW50cy5wcm9tcHQpO1xufVxuXG5jb25zdCBRVUlDS19QUkVTRVRTID0ge1xuICBjb2RlOiBcIkplIHNvdWhhaXRlIFx1MDBFOWNyaXJlIGR1IGNvZGVcdTIwMjZcIixcbiAgc3VtbWFyaXplOiBcIlJcdTAwRTlzdW1lIGxhIGRlcm5pXHUwMEU4cmUgY29udmVyc2F0aW9uLlwiLFxuICBleHBsYWluOiBcIkV4cGxpcXVlIHRhIGRlcm5pXHUwMEU4cmUgclx1MDBFOXBvbnNlIHBsdXMgc2ltcGxlbWVudC5cIixcbn07XG5cbmV4cG9ydCBjbGFzcyBDaGF0QXBwIHtcbiAgY29uc3RydWN0b3IoZG9jID0gZG9jdW1lbnQsIHJhd0NvbmZpZyA9IHdpbmRvdy5jaGF0Q29uZmlnIHx8IHt9KSB7XG4gICAgdGhpcy5kb2MgPSBkb2M7XG4gICAgdGhpcy5jb25maWcgPSByZXNvbHZlQ29uZmlnKHJhd0NvbmZpZyk7XG4gICAgdGhpcy5lbGVtZW50cyA9IHF1ZXJ5RWxlbWVudHMoZG9jKTtcbiAgICBpZiAoIWVuc3VyZUVsZW1lbnRzKHRoaXMuZWxlbWVudHMpKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGlmICh3aW5kb3cubWFya2VkICYmIHR5cGVvZiB3aW5kb3cubWFya2VkLnNldE9wdGlvbnMgPT09IFwiZnVuY3Rpb25cIikge1xuICAgICAgd2luZG93Lm1hcmtlZC5zZXRPcHRpb25zKHtcbiAgICAgICAgYnJlYWtzOiB0cnVlLFxuICAgICAgICBnZm06IHRydWUsXG4gICAgICAgIGhlYWRlcklkczogZmFsc2UsXG4gICAgICAgIG1hbmdsZTogZmFsc2UsXG4gICAgICB9KTtcbiAgICB9XG4gICAgdGhpcy50aW1lbGluZVN0b3JlID0gY3JlYXRlVGltZWxpbmVTdG9yZSgpO1xuICAgIHRoaXMudWkgPSBjcmVhdGVDaGF0VWkoe1xuICAgICAgZWxlbWVudHM6IHRoaXMuZWxlbWVudHMsXG4gICAgICB0aW1lbGluZVN0b3JlOiB0aGlzLnRpbWVsaW5lU3RvcmUsXG4gICAgfSk7XG4gICAgdGhpcy5hdXRoID0gY3JlYXRlQXV0aFNlcnZpY2UodGhpcy5jb25maWcpO1xuICAgIHRoaXMuaHR0cCA9IGNyZWF0ZUh0dHBTZXJ2aWNlKHsgY29uZmlnOiB0aGlzLmNvbmZpZywgYXV0aDogdGhpcy5hdXRoIH0pO1xuICAgIHRoaXMuZXhwb3J0ZXIgPSBjcmVhdGVFeHBvcnRlcih7XG4gICAgICB0aW1lbGluZVN0b3JlOiB0aGlzLnRpbWVsaW5lU3RvcmUsXG4gICAgICBhbm5vdW5jZTogKG1lc3NhZ2UsIHZhcmlhbnQpID0+XG4gICAgICAgIHRoaXMudWkuYW5ub3VuY2VDb25uZWN0aW9uKG1lc3NhZ2UsIHZhcmlhbnQpLFxuICAgIH0pO1xuICAgIHRoaXMuc3VnZ2VzdGlvbnMgPSBjcmVhdGVTdWdnZXN0aW9uU2VydmljZSh7XG4gICAgICBodHRwOiB0aGlzLmh0dHAsXG4gICAgICB1aTogdGhpcy51aSxcbiAgICB9KTtcbiAgICB0aGlzLnNvY2tldCA9IGNyZWF0ZVNvY2tldENsaWVudCh7XG4gICAgICBjb25maWc6IHRoaXMuY29uZmlnLFxuICAgICAgaHR0cDogdGhpcy5odHRwLFxuICAgICAgdWk6IHRoaXMudWksXG4gICAgICBvbkV2ZW50OiAoZXYpID0+IHRoaXMuaGFuZGxlU29ja2V0RXZlbnQoZXYpLFxuICAgIH0pO1xuXG4gICAgY29uc3QgaGlzdG9yeVBheWxvYWQgPSByZWFkSGlzdG9yeShkb2MpO1xuICAgIGlmIChoaXN0b3J5UGF5bG9hZCAmJiBoaXN0b3J5UGF5bG9hZC5lcnJvcikge1xuICAgICAgdGhpcy51aS5zaG93RXJyb3IoaGlzdG9yeVBheWxvYWQuZXJyb3IpO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheShoaXN0b3J5UGF5bG9hZCkpIHtcbiAgICAgIHRoaXMudWkucmVuZGVySGlzdG9yeShoaXN0b3J5UGF5bG9hZCk7XG4gICAgfVxuXG4gICAgdGhpcy5yZWdpc3RlclVpSGFuZGxlcnMoKTtcbiAgICB0aGlzLnVpLmluaXRpYWxpc2UoKTtcbiAgICB0aGlzLnNvY2tldC5vcGVuKCk7XG4gIH1cblxuICByZWdpc3RlclVpSGFuZGxlcnMoKSB7XG4gICAgdGhpcy51aS5vbihcInN1Ym1pdFwiLCBhc3luYyAoeyB0ZXh0IH0pID0+IHtcbiAgICAgIGNvbnN0IHZhbHVlID0gKHRleHQgfHwgXCJcIikudHJpbSgpO1xuICAgICAgaWYgKCF2YWx1ZSkge1xuICAgICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiU2Fpc2lzc2V6IHVuIG1lc3NhZ2UgYXZhbnQgZFx1MjAxOWVudm95ZXIuXCIsXG4gICAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICAgICk7XG4gICAgICAgIHRoaXMudWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNDAwMCk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMudWkuaGlkZUVycm9yKCk7XG4gICAgICBjb25zdCBzdWJtaXR0ZWRBdCA9IG5vd0lTTygpO1xuICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwidXNlclwiLCB2YWx1ZSwge1xuICAgICAgICB0aW1lc3RhbXA6IHN1Ym1pdHRlZEF0LFxuICAgICAgICBtZXRhZGF0YTogeyBzdWJtaXR0ZWQ6IHRydWUgfSxcbiAgICAgIH0pO1xuICAgICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICAgIHRoaXMuZWxlbWVudHMucHJvbXB0LnZhbHVlID0gXCJcIjtcbiAgICAgIH1cbiAgICAgIHRoaXMudWkudXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgICAgdGhpcy51aS5hdXRvc2l6ZVByb21wdCgpO1xuICAgICAgdGhpcy51aS5zZXRDb21wb3NlclN0YXR1cyhcIk1lc3NhZ2UgZW52b3lcdTAwRTlcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIHRoaXMudWkuc2V0QnVzeSh0cnVlKTtcbiAgICAgIHRoaXMudWkuYXBwbHlRdWlja0FjdGlvbk9yZGVyaW5nKFtcImNvZGVcIiwgXCJzdW1tYXJpemVcIiwgXCJleHBsYWluXCJdKTtcblxuICAgICAgdHJ5IHtcbiAgICAgICAgYXdhaXQgdGhpcy5odHRwLnBvc3RDaGF0KHZhbHVlKTtcbiAgICAgICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQuZm9jdXMoKTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLnN0YXJ0U3RyZWFtKCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgY29uc3QgbWVzc2FnZSA9IGVyciBpbnN0YW5jZW9mIEVycm9yID8gZXJyLm1lc3NhZ2UgOiBTdHJpbmcoZXJyKTtcbiAgICAgICAgdGhpcy51aS5zaG93RXJyb3IobWVzc2FnZSk7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcInN5c3RlbVwiLCBtZXNzYWdlLCB7XG4gICAgICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgIG1ldGFkYXRhOiB7IHN0YWdlOiBcInN1Ym1pdFwiIH0sXG4gICAgICAgIH0pO1xuICAgICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiRW52b2kgaW1wb3NzaWJsZS4gVlx1MDBFOXJpZmlleiBsYSBjb25uZXhpb24uXCIsXG4gICAgICAgICAgXCJkYW5nZXJcIixcbiAgICAgICAgKTtcbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg2MDAwKTtcbiAgICAgIH1cbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJxdWljay1hY3Rpb25cIiwgKHsgYWN0aW9uIH0pID0+IHtcbiAgICAgIGlmICghYWN0aW9uKSByZXR1cm47XG4gICAgICBjb25zdCBwcmVzZXQgPSBRVUlDS19QUkVTRVRTW2FjdGlvbl0gfHwgYWN0aW9uO1xuICAgICAgaWYgKHRoaXMuZWxlbWVudHMucHJvbXB0KSB7XG4gICAgICAgIHRoaXMuZWxlbWVudHMucHJvbXB0LnZhbHVlID0gcHJlc2V0O1xuICAgICAgfVxuICAgICAgdGhpcy51aS51cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICB0aGlzLnVpLmF1dG9zaXplUHJvbXB0KCk7XG4gICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFwiU3VnZ2VzdGlvbiBlbnZveVx1MDBFOWVcdTIwMjZcIiwgXCJpbmZvXCIpO1xuICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIHRoaXMudWkuZW1pdChcInN1Ym1pdFwiLCB7IHRleHQ6IHByZXNldCB9KTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJmaWx0ZXItY2hhbmdlXCIsICh7IHZhbHVlIH0pID0+IHtcbiAgICAgIHRoaXMudWkuYXBwbHlUcmFuc2NyaXB0RmlsdGVyKHZhbHVlLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwiZmlsdGVyLWNsZWFyXCIsICgpID0+IHtcbiAgICAgIHRoaXMudWkuY2xlYXJUcmFuc2NyaXB0RmlsdGVyKCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwiZXhwb3J0XCIsICh7IGZvcm1hdCB9KSA9PiB7XG4gICAgICB0aGlzLmV4cG9ydGVyLmV4cG9ydENvbnZlcnNhdGlvbihmb3JtYXQpO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcImV4cG9ydC1jb3B5XCIsICgpID0+IHtcbiAgICAgIHRoaXMuZXhwb3J0ZXIuY29weUNvbnZlcnNhdGlvblRvQ2xpcGJvYXJkKCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwicHJvbXB0LWlucHV0XCIsICh7IHZhbHVlIH0pID0+IHtcbiAgICAgIGlmICghdmFsdWUgfHwgIXZhbHVlLnRyaW0oKSkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5zZW5kICYmIHRoaXMuZWxlbWVudHMuc2VuZC5kaXNhYmxlZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICB0aGlzLnN1Z2dlc3Rpb25zLnNjaGVkdWxlKHZhbHVlKTtcbiAgICB9KTtcbiAgfVxuXG4gIGhhbmRsZVNvY2tldEV2ZW50KGV2KSB7XG4gICAgY29uc3QgdHlwZSA9IGV2ICYmIGV2LnR5cGUgPyBldi50eXBlIDogXCJcIjtcbiAgICBjb25zdCBkYXRhID0gZXYgJiYgZXYuZGF0YSA/IGV2LmRhdGEgOiB7fTtcbiAgICBzd2l0Y2ggKHR5cGUpIHtcbiAgICAgIGNhc2UgXCJ3cy5jb25uZWN0ZWRcIjoge1xuICAgICAgICBpZiAoZGF0YSAmJiBkYXRhLm9yaWdpbikge1xuICAgICAgICAgIHRoaXMudWkuYW5ub3VuY2VDb25uZWN0aW9uKGBDb25uZWN0XHUwMEU5IHZpYSAke2RhdGEub3JpZ2lufWApO1xuICAgICAgICAgIHRoaXMudWkudXBkYXRlQ29ubmVjdGlvbk1ldGEoXG4gICAgICAgICAgICBgQ29ubmVjdFx1MDBFOSB2aWEgJHtkYXRhLm9yaWdpbn1gLFxuICAgICAgICAgICAgXCJzdWNjZXNzXCIsXG4gICAgICAgICAgKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aGlzLnVpLmFubm91bmNlQ29ubmVjdGlvbihcIkNvbm5lY3RcdTAwRTkgYXUgc2VydmV1ci5cIik7XG4gICAgICAgICAgdGhpcy51aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIkNvbm5lY3RcdTAwRTkgYXUgc2VydmV1ci5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNDAwMCk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImhpc3Rvcnkuc25hcHNob3RcIjoge1xuICAgICAgICBpZiAoZGF0YSAmJiBBcnJheS5pc0FycmF5KGRhdGEuaXRlbXMpKSB7XG4gICAgICAgICAgdGhpcy51aS5yZW5kZXJIaXN0b3J5KGRhdGEuaXRlbXMsIHsgcmVwbGFjZTogdHJ1ZSB9KTtcbiAgICAgICAgfVxuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJhaV9tb2RlbC5yZXNwb25zZV9jaHVua1wiOiB7XG4gICAgICAgIGNvbnN0IGRlbHRhID1cbiAgICAgICAgICB0eXBlb2YgZGF0YS5kZWx0YSA9PT0gXCJzdHJpbmdcIiA/IGRhdGEuZGVsdGEgOiBkYXRhLnRleHQgfHwgXCJcIjtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRTdHJlYW0oZGVsdGEpO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJhaV9tb2RlbC5yZXNwb25zZV9jb21wbGV0ZVwiOiB7XG4gICAgICAgIGlmIChkYXRhICYmIGRhdGEudGV4dCAmJiAhdGhpcy51aS5oYXNTdHJlYW1CdWZmZXIoKSkge1xuICAgICAgICAgIHRoaXMudWkuYXBwZW5kU3RyZWFtKGRhdGEudGV4dCk7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy51aS5lbmRTdHJlYW0oZGF0YSk7XG4gICAgICAgIHRoaXMudWkuc2V0QnVzeShmYWxzZSk7XG4gICAgICAgIGlmIChkYXRhICYmIHR5cGVvZiBkYXRhLmxhdGVuY3lfbXMgIT09IFwidW5kZWZpbmVkXCIpIHtcbiAgICAgICAgICB0aGlzLnVpLnNldERpYWdub3N0aWNzKHsgbGF0ZW5jeU1zOiBOdW1iZXIoZGF0YS5sYXRlbmN5X21zKSB9KTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoZGF0YSAmJiBkYXRhLm9rID09PSBmYWxzZSAmJiBkYXRhLmVycm9yKSB7XG4gICAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwic3lzdGVtXCIsIGRhdGEuZXJyb3IsIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiZXJyb3JcIixcbiAgICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfVxuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJjaGF0Lm1lc3NhZ2VcIjoge1xuICAgICAgICBpZiAoIXRoaXMudWkuaXNTdHJlYW1pbmcoKSkge1xuICAgICAgICAgIHRoaXMudWkuc3RhcnRTdHJlYW0oKTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoXG4gICAgICAgICAgZGF0YSAmJlxuICAgICAgICAgIHR5cGVvZiBkYXRhLnJlc3BvbnNlID09PSBcInN0cmluZ1wiICYmXG4gICAgICAgICAgIXRoaXMudWkuaGFzU3RyZWFtQnVmZmVyKClcbiAgICAgICAgKSB7XG4gICAgICAgICAgdGhpcy51aS5hcHBlbmRTdHJlYW0oZGF0YS5yZXNwb25zZSk7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy51aS5lbmRTdHJlYW0oZGF0YSk7XG4gICAgICAgIHRoaXMudWkuc2V0QnVzeShmYWxzZSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImV2b2x1dGlvbl9lbmdpbmUudHJhaW5pbmdfY29tcGxldGVcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXG4gICAgICAgICAgXCJzeXN0ZW1cIixcbiAgICAgICAgICBgXHUwMEM5dm9sdXRpb24gbWlzZSBcdTAwRTAgam91ciAke2RhdGEgJiYgZGF0YS52ZXJzaW9uID8gZGF0YS52ZXJzaW9uIDogXCJcIn1gLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwib2tcIixcbiAgICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgICB9LFxuICAgICAgICApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJldm9sdXRpb25fZW5naW5lLnRyYWluaW5nX2ZhaWxlZFwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIGBcdTAwQzljaGVjIGRlIGwnXHUwMEU5dm9sdXRpb24gOiAke2RhdGEgJiYgZGF0YS5lcnJvciA/IGRhdGEuZXJyb3IgOiBcImluY29ubnVcIn1gLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiZXJyb3JcIixcbiAgICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgICB9LFxuICAgICAgICApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJzbGVlcF90aW1lX2NvbXB1dGUucGhhc2Vfc3RhcnRcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXG4gICAgICAgICAgXCJzeXN0ZW1cIixcbiAgICAgICAgICBcIk9wdGltaXNhdGlvbiBlbiBhcnJpXHUwMEU4cmUtcGxhbiBkXHUwMEU5bWFyclx1MDBFOWVcdTIwMjZcIixcbiAgICAgICAgICB7XG4gICAgICAgICAgICB2YXJpYW50OiBcImhpbnRcIixcbiAgICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgICB9LFxuICAgICAgICApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJzbGVlcF90aW1lX2NvbXB1dGUuY3JlYXRpdmVfcGhhc2VcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXG4gICAgICAgICAgXCJzeXN0ZW1cIixcbiAgICAgICAgICBgRXhwbG9yYXRpb24gZGUgJHtOdW1iZXIoZGF0YSAmJiBkYXRhLmlkZWFzID8gZGF0YS5pZGVhcyA6IDEpfSBpZFx1MDBFOWVzXHUyMDI2YCxcbiAgICAgICAgICB7XG4gICAgICAgICAgICB2YXJpYW50OiBcImhpbnRcIixcbiAgICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgICAgbWV0YWRhdGE6IHsgZXZlbnQ6IHR5cGUgfSxcbiAgICAgICAgICB9LFxuICAgICAgICApO1xuICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICAgIGNhc2UgXCJwZXJmb3JtYW5jZS5hbGVydFwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcInN5c3RlbVwiLCBgUGVyZiA6ICR7dGhpcy51aS5mb3JtYXRQZXJmKGRhdGEpfWAsIHtcbiAgICAgICAgICB2YXJpYW50OiBcIndhcm5cIixcbiAgICAgICAgICBhbGxvd01hcmtkb3duOiBmYWxzZSxcbiAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICB9KTtcbiAgICAgICAgaWYgKGRhdGEgJiYgdHlwZW9mIGRhdGEudHRmYl9tcyAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgICAgIHRoaXMudWkuc2V0RGlhZ25vc3RpY3MoeyBsYXRlbmN5TXM6IE51bWJlcihkYXRhLnR0ZmJfbXMpIH0pO1xuICAgICAgICB9XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInVpLnN1Z2dlc3Rpb25zXCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcoXG4gICAgICAgICAgQXJyYXkuaXNBcnJheShkYXRhLmFjdGlvbnMpID8gZGF0YS5hY3Rpb25zIDogW10sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgZGVmYXVsdDpcbiAgICAgICAgaWYgKHR5cGUgJiYgdHlwZS5zdGFydHNXaXRoKFwid3MuXCIpKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnNvbGUuZGVidWcoXCJVbmhhbmRsZWQgZXZlbnRcIiwgZXYpO1xuICAgIH1cbiAgfVxufVxuIiwgIi8qKlxuICogRW50cnkgcG9pbnQgZm9yIHRoZSBjaGF0IGFwcGxpY2F0aW9uLlxuICogRXhwZWN0cyB3aW5kb3cuY2hhdENvbmZpZyB0byBiZSBkZWZpbmVkIGJ5IHRoZSBzZXJ2ZXItcmVuZGVyZWQgdGVtcGxhdGUuXG4gKiBGYWxscyBiYWNrIHRvIGFuIGVtcHR5IGNvbmZpZyBpZiBub3QgcHJlc2VudC5cbiAqIFJlcXVpcmVkIGNvbmZpZyBzaGFwZTogeyBhcGlVcmw/LCB3c1VybD8sIHRva2VuPywgLi4uIH1cbiAqL1xuaW1wb3J0IHsgQ2hhdEFwcCB9IGZyb20gXCIuL2FwcC5qc1wiO1xuXG5uZXcgQ2hhdEFwcChkb2N1bWVudCwgd2luZG93LmNoYXRDb25maWcgfHwge30pO1xuIl0sCiAgIm1hcHBpbmdzIjogIjs7QUFBTyxXQUFTLGNBQWMsTUFBTSxDQUFDLEdBQUc7QUFDdEMsVUFBTSxTQUFTLEVBQUUsR0FBRyxJQUFJO0FBQ3hCLFVBQU0sWUFBWSxPQUFPLGNBQWMsT0FBTyxTQUFTO0FBQ3ZELFFBQUk7QUFDRixhQUFPLFVBQVUsSUFBSSxJQUFJLFNBQVM7QUFBQSxJQUNwQyxTQUFTLEtBQUs7QUFDWixjQUFRLE1BQU0sdUJBQXVCLEtBQUssU0FBUztBQUNuRCxhQUFPLFVBQVUsSUFBSSxJQUFJLE9BQU8sU0FBUyxNQUFNO0FBQUEsSUFDakQ7QUFDQSxXQUFPO0FBQUEsRUFDVDtBQUVPLFdBQVMsT0FBTyxRQUFRLE1BQU07QUFDbkMsV0FBTyxJQUFJLElBQUksTUFBTSxPQUFPLE9BQU8sRUFBRSxTQUFTO0FBQUEsRUFDaEQ7OztBQ2RPLFdBQVMsU0FBUztBQUN2QixZQUFPLG9CQUFJLEtBQUssR0FBRSxZQUFZO0FBQUEsRUFDaEM7QUFFTyxXQUFTLGdCQUFnQixJQUFJO0FBQ2xDLFFBQUksQ0FBQyxHQUFJLFFBQU87QUFDaEIsUUFBSTtBQUNGLGFBQU8sSUFBSSxLQUFLLEVBQUUsRUFBRSxlQUFlLE9BQU87QUFBQSxJQUM1QyxTQUFTLEtBQUs7QUFDWixhQUFPLE9BQU8sRUFBRTtBQUFBLElBQ2xCO0FBQUEsRUFDRjs7O0FDVEEsV0FBUyxnQkFBZ0I7QUFDdkIsV0FBTyxPQUFPLEtBQUssSUFBSSxFQUFFLFNBQVMsRUFBRSxDQUFDLElBQUksS0FBSyxPQUFPLEVBQUUsU0FBUyxFQUFFLEVBQUUsTUFBTSxHQUFHLENBQUMsQ0FBQztBQUFBLEVBQ2pGO0FBRU8sV0FBUyxzQkFBc0I7QUFDcEMsVUFBTSxRQUFRLENBQUM7QUFDZixVQUFNLE1BQU0sb0JBQUksSUFBSTtBQUVwQixhQUFTLFNBQVM7QUFBQSxNQUNoQjtBQUFBLE1BQ0E7QUFBQSxNQUNBLE9BQU87QUFBQSxNQUNQLFlBQVksT0FBTztBQUFBLE1BQ25CO0FBQUEsTUFDQSxXQUFXLENBQUM7QUFBQSxJQUNkLEdBQUc7QUFDRCxZQUFNLFlBQVksTUFBTSxjQUFjO0FBQ3RDLFVBQUksQ0FBQyxJQUFJLElBQUksU0FBUyxHQUFHO0FBQ3ZCLGNBQU0sS0FBSyxTQUFTO0FBQUEsTUFDdEI7QUFDQSxVQUFJLElBQUksV0FBVztBQUFBLFFBQ2pCLElBQUk7QUFBQSxRQUNKO0FBQUEsUUFDQTtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsUUFDQSxVQUFVLEVBQUUsR0FBRyxTQUFTO0FBQUEsTUFDMUIsQ0FBQztBQUNELFVBQUksS0FBSztBQUNQLFlBQUksUUFBUSxZQUFZO0FBQ3hCLFlBQUksUUFBUSxPQUFPO0FBQ25CLFlBQUksUUFBUSxVQUFVO0FBQ3RCLFlBQUksUUFBUSxZQUFZO0FBQUEsTUFDMUI7QUFDQSxhQUFPO0FBQUEsSUFDVDtBQUVBLGFBQVMsT0FBTyxJQUFJLFFBQVEsQ0FBQyxHQUFHO0FBQzlCLFVBQUksQ0FBQyxJQUFJLElBQUksRUFBRSxHQUFHO0FBQ2hCLGVBQU87QUFBQSxNQUNUO0FBQ0EsWUFBTSxRQUFRLElBQUksSUFBSSxFQUFFO0FBQ3hCLFlBQU0sT0FBTyxFQUFFLEdBQUcsT0FBTyxHQUFHLE1BQU07QUFDbEMsVUFBSSxTQUFTLE9BQU8sTUFBTSxhQUFhLFlBQVksTUFBTSxhQUFhLE1BQU07QUFDMUUsY0FBTSxTQUFTLEVBQUUsR0FBRyxNQUFNLFNBQVM7QUFDbkMsZUFBTyxRQUFRLE1BQU0sUUFBUSxFQUFFLFFBQVEsQ0FBQyxDQUFDLEtBQUssS0FBSyxNQUFNO0FBQ3ZELGNBQUksVUFBVSxVQUFhLFVBQVUsTUFBTTtBQUN6QyxtQkFBTyxPQUFPLEdBQUc7QUFBQSxVQUNuQixPQUFPO0FBQ0wsbUJBQU8sR0FBRyxJQUFJO0FBQUEsVUFDaEI7QUFBQSxRQUNGLENBQUM7QUFDRCxhQUFLLFdBQVc7QUFBQSxNQUNsQjtBQUNBLFVBQUksSUFBSSxJQUFJLElBQUk7QUFDaEIsWUFBTSxFQUFFLElBQUksSUFBSTtBQUNoQixVQUFJLE9BQU8sSUFBSSxhQUFhO0FBQzFCLFlBQUksS0FBSyxTQUFTLE1BQU0sTUFBTTtBQUM1QixjQUFJLFFBQVEsVUFBVSxLQUFLLFFBQVE7QUFBQSxRQUNyQztBQUNBLFlBQUksS0FBSyxjQUFjLE1BQU0sV0FBVztBQUN0QyxjQUFJLFFBQVEsWUFBWSxLQUFLLGFBQWE7QUFBQSxRQUM1QztBQUNBLFlBQUksS0FBSyxRQUFRLEtBQUssU0FBUyxNQUFNLE1BQU07QUFDekMsY0FBSSxRQUFRLE9BQU8sS0FBSztBQUFBLFFBQzFCO0FBQUEsTUFDRjtBQUNBLGFBQU87QUFBQSxJQUNUO0FBRUEsYUFBUyxVQUFVO0FBQ2pCLGFBQU8sTUFDSixJQUFJLENBQUMsT0FBTztBQUNYLGNBQU0sUUFBUSxJQUFJLElBQUksRUFBRTtBQUN4QixZQUFJLENBQUMsT0FBTztBQUNWLGlCQUFPO0FBQUEsUUFDVDtBQUNBLGVBQU87QUFBQSxVQUNMLE1BQU0sTUFBTTtBQUFBLFVBQ1osTUFBTSxNQUFNO0FBQUEsVUFDWixXQUFXLE1BQU07QUFBQSxVQUNqQixHQUFJLE1BQU0sWUFDUixPQUFPLEtBQUssTUFBTSxRQUFRLEVBQUUsU0FBUyxLQUFLO0FBQUEsWUFDeEMsVUFBVSxFQUFFLEdBQUcsTUFBTSxTQUFTO0FBQUEsVUFDaEM7QUFBQSxRQUNKO0FBQUEsTUFDRixDQUFDLEVBQ0EsT0FBTyxPQUFPO0FBQUEsSUFDbkI7QUFFQSxhQUFTLFFBQVE7QUFDZixZQUFNLFNBQVM7QUFDZixVQUFJLE1BQU07QUFBQSxJQUNaO0FBRUEsV0FBTztBQUFBLE1BQ0w7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxJQUNGO0FBQUEsRUFDRjs7O0FDMUdPLFdBQVMsZ0JBQWdCO0FBQzlCLFVBQU0sWUFBWSxvQkFBSSxJQUFJO0FBRTFCLGFBQVMsR0FBRyxPQUFPLFNBQVM7QUFDMUIsVUFBSSxDQUFDLFVBQVUsSUFBSSxLQUFLLEdBQUc7QUFDekIsa0JBQVUsSUFBSSxPQUFPLG9CQUFJLElBQUksQ0FBQztBQUFBLE1BQ2hDO0FBQ0EsZ0JBQVUsSUFBSSxLQUFLLEVBQUUsSUFBSSxPQUFPO0FBQ2hDLGFBQU8sTUFBTSxJQUFJLE9BQU8sT0FBTztBQUFBLElBQ2pDO0FBRUEsYUFBUyxJQUFJLE9BQU8sU0FBUztBQUMzQixVQUFJLENBQUMsVUFBVSxJQUFJLEtBQUssRUFBRztBQUMzQixZQUFNLFNBQVMsVUFBVSxJQUFJLEtBQUs7QUFDbEMsYUFBTyxPQUFPLE9BQU87QUFDckIsVUFBSSxPQUFPLFNBQVMsR0FBRztBQUNyQixrQkFBVSxPQUFPLEtBQUs7QUFBQSxNQUN4QjtBQUFBLElBQ0Y7QUFFQSxhQUFTLEtBQUssT0FBTyxTQUFTO0FBQzVCLFVBQUksQ0FBQyxVQUFVLElBQUksS0FBSyxFQUFHO0FBQzNCLGdCQUFVLElBQUksS0FBSyxFQUFFLFFBQVEsQ0FBQyxZQUFZO0FBQ3hDLFlBQUk7QUFDRixrQkFBUSxPQUFPO0FBQUEsUUFDakIsU0FBUyxLQUFLO0FBQ1osa0JBQVEsTUFBTSx5QkFBeUIsR0FBRztBQUFBLFFBQzVDO0FBQUEsTUFDRixDQUFDO0FBQUEsSUFDSDtBQUVBLFdBQU8sRUFBRSxJQUFJLEtBQUssS0FBSztBQUFBLEVBQ3pCOzs7QUNoQ08sV0FBUyxXQUFXLEtBQUs7QUFDOUIsV0FBTyxPQUFPLEdBQUcsRUFBRTtBQUFBLE1BQ2pCO0FBQUEsTUFDQSxDQUFDLFFBQ0U7QUFBQSxRQUNDLEtBQUs7QUFBQSxRQUNMLEtBQUs7QUFBQSxRQUNMLEtBQUs7QUFBQSxRQUNMLEtBQUs7QUFBQSxRQUNMLEtBQUs7QUFBQSxNQUNQLEdBQUcsRUFBRTtBQUFBLElBQ1Q7QUFBQSxFQUNGO0FBRU8sV0FBUyxXQUFXLE1BQU07QUFDL0IsVUFBTSxTQUFTLElBQUksVUFBVTtBQUM3QixVQUFNLE1BQU0sT0FBTyxnQkFBZ0IsTUFBTSxXQUFXO0FBQ3BELFdBQU8sSUFBSSxLQUFLLGVBQWU7QUFBQSxFQUNqQztBQUVPLFdBQVMsa0JBQWtCLFFBQVE7QUFDeEMsVUFBTSxRQUFRLE9BQU8sVUFBVSxJQUFJO0FBQ25DLFVBQ0csaUJBQWlCLHVCQUF1QixFQUN4QyxRQUFRLENBQUMsU0FBUyxLQUFLLE9BQU8sQ0FBQztBQUNsQyxXQUFPLE1BQU0sWUFBWSxLQUFLO0FBQUEsRUFDaEM7OztBQ3hCTyxXQUFTLGVBQWUsTUFBTTtBQUNuQyxRQUFJLFFBQVEsTUFBTTtBQUNoQixhQUFPO0FBQUEsSUFDVDtBQUNBLFVBQU0sUUFBUSxPQUFPLElBQUk7QUFDekIsVUFBTSxXQUFXLE1BQU07QUFDckIsWUFBTSxVQUFVLFdBQVcsS0FBSztBQUNoQyxhQUFPLFFBQVEsUUFBUSxPQUFPLE1BQU07QUFBQSxJQUN0QztBQUNBLFFBQUk7QUFDRixVQUFJLE9BQU8sVUFBVSxPQUFPLE9BQU8sT0FBTyxVQUFVLFlBQVk7QUFDOUQsY0FBTSxXQUFXLE9BQU8sT0FBTyxNQUFNLEtBQUs7QUFDMUMsWUFBSSxPQUFPLGFBQWEsT0FBTyxPQUFPLFVBQVUsYUFBYSxZQUFZO0FBQ3ZFLGlCQUFPLE9BQU8sVUFBVSxTQUFTLFVBQVU7QUFBQSxZQUN6Qyx5QkFBeUI7QUFBQSxZQUN6QixjQUFjLEVBQUUsTUFBTSxLQUFLO0FBQUEsVUFDN0IsQ0FBQztBQUFBLFFBQ0g7QUFFQSxjQUFNLFVBQVUsV0FBVyxLQUFLO0FBQ2hDLGVBQU8sUUFBUSxRQUFRLE9BQU8sTUFBTTtBQUFBLE1BQ3RDO0FBQUEsSUFDRixTQUFTLEtBQUs7QUFDWixjQUFRLEtBQUssNkJBQTZCLEdBQUc7QUFBQSxJQUMvQztBQUNBLFdBQU8sU0FBUztBQUFBLEVBQ2xCOzs7QUN2Qk8sV0FBUyxhQUFhLEVBQUUsVUFBVSxjQUFjLEdBQUc7QUFMMUQ7QUFNRSxVQUFNLFVBQVUsY0FBYztBQUU5QixVQUFNLGlCQUFpQixTQUFTLE9BQU8sU0FBUyxLQUFLLFlBQVk7QUFDakUsVUFBTSxnQkFDSCxTQUFTLFFBQVEsU0FBUyxLQUFLLGFBQWEsaUJBQWlCLE1BQzdELFNBQVMsT0FBTyxTQUFTLEtBQUssWUFBWSxLQUFLLElBQUk7QUFDdEQsVUFBTSxpQkFDSjtBQUNGLFVBQU0sd0JBQ0gsU0FBUyxrQkFBa0IsU0FBUyxlQUFlLFlBQVksS0FBSyxLQUNyRTtBQUNGLFVBQU0sb0JBQ0gsU0FBUyxjQUFjLFNBQVMsV0FBVyxZQUFZLEtBQUssS0FDN0Q7QUFDRixVQUFNLFlBQVksUUFBTyxjQUFTLFdBQVQsbUJBQWlCLGFBQWEsWUFBWSxLQUFLO0FBQ3hFLFVBQU0sdUJBQ0osT0FBTyxjQUNQLE9BQU8sV0FBVyxrQ0FBa0MsRUFBRTtBQUN4RCxVQUFNLG1CQUFtQjtBQUN6QixVQUFNLG9CQUFvQjtBQUUxQixVQUFNLGNBQWM7QUFBQSxNQUNsQixhQUFhO0FBQUEsTUFDYixlQUFlO0FBQUEsTUFDZixXQUFXO0FBQUEsSUFDYjtBQUVBLFVBQU0sUUFBUTtBQUFBLE1BQ1osa0JBQWtCO0FBQUEsTUFDbEIsaUJBQWlCO0FBQUEsTUFDakIsY0FBYztBQUFBLE1BQ2QscUJBQXFCLFNBQVMsV0FBVyxvQkFBb0I7QUFBQSxNQUM3RCxlQUFlO0FBQUEsTUFDZixXQUFXO0FBQUEsTUFDWCxXQUFXO0FBQUEsTUFDWCxpQkFBaUI7QUFBQSxJQUNuQjtBQUVBLFVBQU0sZUFBZTtBQUFBLE1BQ25CLFNBQVM7QUFBQSxNQUNULFlBQVk7QUFBQSxNQUNaLFFBQVE7QUFBQSxNQUNSLE9BQU87QUFBQSxJQUNUO0FBRUEsYUFBUyxHQUFHLE9BQU8sU0FBUztBQUMxQixhQUFPLFFBQVEsR0FBRyxPQUFPLE9BQU87QUFBQSxJQUNsQztBQUVBLGFBQVMsS0FBSyxPQUFPLFNBQVM7QUFDNUIsY0FBUSxLQUFLLE9BQU8sT0FBTztBQUFBLElBQzdCO0FBRUEsYUFBUyxRQUFRLE1BQU07QUFDckIsZUFBUyxXQUFXLGFBQWEsYUFBYSxPQUFPLFNBQVMsT0FBTztBQUNyRSxVQUFJLFNBQVMsTUFBTTtBQUNqQixpQkFBUyxLQUFLLFdBQVcsUUFBUSxJQUFJO0FBQ3JDLGlCQUFTLEtBQUssYUFBYSxhQUFhLE9BQU8sU0FBUyxPQUFPO0FBQy9ELFlBQUksTUFBTTtBQUNSLG1CQUFTLEtBQUssWUFBWTtBQUFBLFFBQzVCLFdBQVcsZ0JBQWdCO0FBQ3pCLG1CQUFTLEtBQUssWUFBWTtBQUFBLFFBQzVCLE9BQU87QUFDTCxtQkFBUyxLQUFLLGNBQWM7QUFBQSxRQUM5QjtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBRUEsYUFBUyxZQUFZO0FBQ25CLFVBQUksQ0FBQyxTQUFTLFdBQVk7QUFDMUIsZUFBUyxXQUFXLFVBQVUsSUFBSSxRQUFRO0FBQzFDLFVBQUksU0FBUyxjQUFjO0FBQ3pCLGlCQUFTLGFBQWEsY0FBYztBQUFBLE1BQ3RDO0FBQUEsSUFDRjtBQUVBLGFBQVMsVUFBVSxTQUFTO0FBQzFCLFVBQUksQ0FBQyxTQUFTLGNBQWMsQ0FBQyxTQUFTLGFBQWM7QUFDcEQsZUFBUyxhQUFhLGNBQWM7QUFDcEMsZUFBUyxXQUFXLFVBQVUsT0FBTyxRQUFRO0FBQUEsSUFDL0M7QUFFQSxhQUFTLGtCQUFrQixTQUFTLE9BQU8sU0FBUztBQUNsRCxVQUFJLENBQUMsU0FBUyxlQUFnQjtBQUM5QixZQUFNLFFBQVEsQ0FBQyxTQUFTLFFBQVEsV0FBVyxVQUFVLFNBQVM7QUFDOUQsZUFBUyxlQUFlLGNBQWM7QUFDdEMsWUFBTSxRQUFRLENBQUMsTUFBTSxTQUFTLGVBQWUsVUFBVSxPQUFPLFFBQVEsQ0FBQyxFQUFFLENBQUM7QUFDMUUsZUFBUyxlQUFlLFVBQVUsSUFBSSxRQUFRLElBQUksRUFBRTtBQUFBLElBQ3REO0FBRUEsYUFBUyx3QkFBd0I7QUFDL0Isd0JBQWtCLHVCQUF1QixPQUFPO0FBQUEsSUFDbEQ7QUFFQSxhQUFTLHFCQUFxQixRQUFRLE1BQU07QUFDMUMsVUFBSSxNQUFNLGtCQUFrQjtBQUMxQixxQkFBYSxNQUFNLGdCQUFnQjtBQUFBLE1BQ3JDO0FBQ0EsWUFBTSxtQkFBbUIsT0FBTyxXQUFXLE1BQU07QUFDL0MsOEJBQXNCO0FBQUEsTUFDeEIsR0FBRyxLQUFLO0FBQUEsSUFDVjtBQUVBLGFBQVMsc0JBQXNCO0FBQzdCLFVBQUksQ0FBQyxTQUFTLGVBQWUsQ0FBQyxTQUFTLE9BQVE7QUFDL0MsWUFBTSxRQUFRLFNBQVMsT0FBTyxTQUFTO0FBQ3ZDLFVBQUksV0FBVztBQUNiLGlCQUFTLFlBQVksY0FBYyxHQUFHLE1BQU0sTUFBTSxNQUFNLFNBQVM7QUFBQSxNQUNuRSxPQUFPO0FBQ0wsaUJBQVMsWUFBWSxjQUFjLEdBQUcsTUFBTSxNQUFNO0FBQUEsTUFDcEQ7QUFDQSxlQUFTLFlBQVksVUFBVSxPQUFPLGdCQUFnQixhQUFhO0FBQ25FLFVBQUksV0FBVztBQUNiLGNBQU0sWUFBWSxZQUFZLE1BQU07QUFDcEMsWUFBSSxhQUFhLEdBQUc7QUFDbEIsbUJBQVMsWUFBWSxVQUFVLElBQUksYUFBYTtBQUFBLFFBQ2xELFdBQVcsYUFBYSxJQUFJO0FBQzFCLG1CQUFTLFlBQVksVUFBVSxJQUFJLGNBQWM7QUFBQSxRQUNuRDtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBRUEsYUFBUyxpQkFBaUI7QUFDeEIsVUFBSSxDQUFDLFNBQVMsT0FBUTtBQUN0QixlQUFTLE9BQU8sTUFBTSxTQUFTO0FBQy9CLFlBQU0sYUFBYSxLQUFLO0FBQUEsUUFDdEIsU0FBUyxPQUFPO0FBQUEsUUFDaEI7QUFBQSxNQUNGO0FBQ0EsZUFBUyxPQUFPLE1BQU0sU0FBUyxHQUFHLFVBQVU7QUFBQSxJQUM5QztBQUVBLGFBQVMsYUFBYTtBQUNwQixVQUFJLENBQUMsU0FBUyxXQUFZLFFBQU87QUFDakMsWUFBTSxXQUNKLFNBQVMsV0FBVyxnQkFDbkIsU0FBUyxXQUFXLFlBQVksU0FBUyxXQUFXO0FBQ3ZELGFBQU8sWUFBWTtBQUFBLElBQ3JCO0FBRUEsYUFBUyxlQUFlLFVBQVUsQ0FBQyxHQUFHO0FBQ3BDLFVBQUksQ0FBQyxTQUFTLFdBQVk7QUFDMUIsWUFBTSxTQUFTLFFBQVEsV0FBVyxTQUFTLENBQUM7QUFDNUMsZUFBUyxXQUFXLFNBQVM7QUFBQSxRQUMzQixLQUFLLFNBQVMsV0FBVztBQUFBLFFBQ3pCLFVBQVUsU0FBUyxXQUFXO0FBQUEsTUFDaEMsQ0FBQztBQUNELHVCQUFpQjtBQUFBLElBQ25CO0FBRUEsYUFBUyxtQkFBbUI7QUFDMUIsVUFBSSxDQUFDLFNBQVMsYUFBYztBQUM1QixVQUFJLE1BQU0saUJBQWlCO0FBQ3pCLHFCQUFhLE1BQU0sZUFBZTtBQUNsQyxjQUFNLGtCQUFrQjtBQUFBLE1BQzFCO0FBQ0EsZUFBUyxhQUFhLFVBQVUsT0FBTyxRQUFRO0FBQy9DLGVBQVMsYUFBYSxVQUFVLElBQUksWUFBWTtBQUNoRCxlQUFTLGFBQWEsYUFBYSxlQUFlLE9BQU87QUFBQSxJQUMzRDtBQUVBLGFBQVMsbUJBQW1CO0FBQzFCLFVBQUksQ0FBQyxTQUFTLGFBQWM7QUFDNUIsZUFBUyxhQUFhLFVBQVUsT0FBTyxZQUFZO0FBQ25ELGVBQVMsYUFBYSxhQUFhLGVBQWUsTUFBTTtBQUN4RCxZQUFNLGtCQUFrQixPQUFPLFdBQVcsTUFBTTtBQUM5QyxZQUFJLFNBQVMsY0FBYztBQUN6QixtQkFBUyxhQUFhLFVBQVUsSUFBSSxRQUFRO0FBQUEsUUFDOUM7QUFBQSxNQUNGLEdBQUcsR0FBRztBQUFBLElBQ1I7QUFFQSxtQkFBZSxXQUFXLFFBQVE7QUFDaEMsWUFBTSxPQUFPLGtCQUFrQixNQUFNO0FBQ3JDLFVBQUksQ0FBQyxNQUFNO0FBQ1Q7QUFBQSxNQUNGO0FBQ0EsVUFBSTtBQUNGLFlBQUksVUFBVSxhQUFhLFVBQVUsVUFBVSxXQUFXO0FBQ3hELGdCQUFNLFVBQVUsVUFBVSxVQUFVLElBQUk7QUFBQSxRQUMxQyxPQUFPO0FBQ0wsZ0JBQU0sV0FBVyxTQUFTLGNBQWMsVUFBVTtBQUNsRCxtQkFBUyxRQUFRO0FBQ2pCLG1CQUFTLGFBQWEsWUFBWSxVQUFVO0FBQzVDLG1CQUFTLE1BQU0sV0FBVztBQUMxQixtQkFBUyxNQUFNLE9BQU87QUFDdEIsbUJBQVMsS0FBSyxZQUFZLFFBQVE7QUFDbEMsbUJBQVMsT0FBTztBQUNoQixtQkFBUyxZQUFZLE1BQU07QUFDM0IsbUJBQVMsS0FBSyxZQUFZLFFBQVE7QUFBQSxRQUNwQztBQUNBLDJCQUFtQiw0Q0FBeUMsU0FBUztBQUFBLE1BQ3ZFLFNBQVMsS0FBSztBQUNaLGdCQUFRLEtBQUssZUFBZSxHQUFHO0FBQy9CLDJCQUFtQixvQ0FBb0MsUUFBUTtBQUFBLE1BQ2pFO0FBQUEsSUFDRjtBQUVBLGFBQVMsWUFBWSxLQUFLLE1BQU07QUFDOUIsWUFBTSxTQUFTLElBQUksY0FBYyxjQUFjO0FBQy9DLFVBQUksQ0FBQyxPQUFRO0FBQ2IsVUFBSSxTQUFTLGVBQWUsU0FBUyxRQUFRO0FBQzNDLGVBQU8sVUFBVSxJQUFJLFdBQVc7QUFDaEMsZUFBTyxpQkFBaUIsV0FBVyxFQUFFLFFBQVEsQ0FBQyxRQUFRLElBQUksT0FBTyxDQUFDO0FBQ2xFLGNBQU0sVUFBVSxTQUFTLGNBQWMsUUFBUTtBQUMvQyxnQkFBUSxPQUFPO0FBQ2YsZ0JBQVEsWUFBWTtBQUNwQixnQkFBUSxZQUNOO0FBQ0YsZ0JBQVEsaUJBQWlCLFNBQVMsTUFBTSxXQUFXLE1BQU0sQ0FBQztBQUMxRCxlQUFPLFlBQVksT0FBTztBQUFBLE1BQzVCO0FBQUEsSUFDRjtBQUVBLGFBQVMsYUFBYSxLQUFLLE1BQU07QUFDL0IsVUFBSSxDQUFDLE9BQU8sTUFBTSxpQkFBaUIsU0FBUyxVQUFVO0FBQ3BEO0FBQUEsTUFDRjtBQUNBLFVBQUksVUFBVSxJQUFJLG9CQUFvQjtBQUN0QyxhQUFPLFdBQVcsTUFBTTtBQUN0QixZQUFJLFVBQVUsT0FBTyxvQkFBb0I7QUFBQSxNQUMzQyxHQUFHLEdBQUc7QUFBQSxJQUNSO0FBRUEsYUFBUyxLQUFLLE1BQU0sTUFBTSxVQUFVLENBQUMsR0FBRztBQUN0QyxZQUFNLGNBQWMsV0FBVztBQUMvQixZQUFNLE1BQU0sU0FBUyxjQUFjLEtBQUs7QUFDeEMsVUFBSSxZQUFZLGlCQUFpQixJQUFJO0FBQ3JDLFVBQUksWUFBWTtBQUNoQixVQUFJLFFBQVEsT0FBTztBQUNuQixVQUFJLFFBQVEsVUFBVSxRQUFRLFdBQVc7QUFDekMsVUFBSSxRQUFRLFlBQVksUUFBUSxhQUFhO0FBQzdDLGVBQVMsV0FBVyxZQUFZLEdBQUc7QUFDbkMsa0JBQVksS0FBSyxJQUFJO0FBQ3JCLFVBQUksUUFBUSxhQUFhLE9BQU87QUFDOUIsY0FBTSxLQUFLLFFBQVEsYUFBYSxPQUFPO0FBQ3ZDLGNBQU0sT0FDSixRQUFRLFdBQVcsUUFBUSxRQUFRLFNBQVMsSUFDeEMsUUFBUSxVQUNSLFdBQVcsSUFBSTtBQUNyQixjQUFNLEtBQUssY0FBYyxTQUFTO0FBQUEsVUFDaEMsSUFBSSxRQUFRO0FBQUEsVUFDWjtBQUFBLFVBQ0E7QUFBQSxVQUNBLFdBQVc7QUFBQSxVQUNYO0FBQUEsVUFDQSxVQUFVLFFBQVEsWUFBWSxDQUFDO0FBQUEsUUFDakMsQ0FBQztBQUNELFlBQUksUUFBUSxZQUFZO0FBQUEsTUFDMUIsV0FBVyxRQUFRLFdBQVc7QUFDNUIsWUFBSSxRQUFRLFlBQVksUUFBUTtBQUFBLE1BQ2xDLFdBQVcsQ0FBQyxJQUFJLFFBQVEsV0FBVztBQUNqQyxZQUFJLFFBQVEsWUFBWSxjQUFjLGNBQWM7QUFBQSxNQUN0RDtBQUNBLFVBQUksYUFBYTtBQUNmLHVCQUFlLEVBQUUsUUFBUSxDQUFDLE1BQU0sY0FBYyxDQUFDO0FBQUEsTUFDakQsT0FBTztBQUNMLHlCQUFpQjtBQUFBLE1BQ25CO0FBQ0EsbUJBQWEsS0FBSyxJQUFJO0FBQ3RCLFVBQUksTUFBTSxjQUFjO0FBQ3RCLDhCQUFzQixNQUFNLGNBQWMsRUFBRSxlQUFlLEtBQUssQ0FBQztBQUFBLE1BQ25FO0FBQ0EsYUFBTztBQUFBLElBQ1Q7QUFFQSxhQUFTLFlBQVk7QUFBQSxNQUNuQjtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0EsZ0JBQWdCO0FBQUEsSUFDbEIsR0FBRztBQUNELFlBQU0sVUFBVSxDQUFDLGFBQWE7QUFDOUIsVUFBSSxTQUFTO0FBQ1gsZ0JBQVEsS0FBSyxlQUFlLE9BQU8sRUFBRTtBQUFBLE1BQ3ZDO0FBQ0EsWUFBTSxVQUFVLGdCQUNaLGVBQWUsSUFBSSxJQUNuQixXQUFXLE9BQU8sSUFBSSxDQUFDO0FBQzNCLFlBQU0sV0FBVyxDQUFDO0FBQ2xCLFVBQUksV0FBVztBQUNiLGlCQUFTLEtBQUssZ0JBQWdCLFNBQVMsQ0FBQztBQUFBLE1BQzFDO0FBQ0EsVUFBSSxZQUFZO0FBQ2QsaUJBQVMsS0FBSyxVQUFVO0FBQUEsTUFDMUI7QUFDQSxZQUFNLFdBQ0osU0FBUyxTQUFTLElBQ2QsMEJBQTBCLFdBQVcsU0FBUyxLQUFLLFVBQUssQ0FBQyxDQUFDLFdBQzFEO0FBQ04sYUFBTyxlQUFlLFFBQVEsS0FBSyxHQUFHLENBQUMsS0FBSyxPQUFPLEdBQUcsUUFBUTtBQUFBLElBQ2hFO0FBRUEsYUFBUyxjQUFjLE1BQU0sTUFBTSxVQUFVLENBQUMsR0FBRztBQUMvQyxZQUFNO0FBQUEsUUFDSjtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsUUFDQSxnQkFBZ0I7QUFBQSxRQUNoQjtBQUFBLFFBQ0EsV0FBVztBQUFBLFFBQ1g7QUFBQSxNQUNGLElBQUk7QUFDSixZQUFNLFNBQVMsWUFBWTtBQUFBLFFBQ3pCO0FBQUEsUUFDQTtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsUUFDQTtBQUFBLE1BQ0YsQ0FBQztBQUNELFlBQU0sTUFBTSxLQUFLLE1BQU0sUUFBUTtBQUFBLFFBQzdCLFNBQVM7QUFBQSxRQUNUO0FBQUEsUUFDQTtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsTUFDRixDQUFDO0FBQ0QscUJBQWUsRUFBRSxlQUFlLGFBQWEsT0FBTyxFQUFFLENBQUM7QUFDdkQsYUFBTztBQUFBLElBQ1Q7QUFFQSxhQUFTLHNCQUFzQixJQUFJLE9BQU87QUFDeEMsVUFBSSxDQUFDLEdBQUk7QUFDVCxTQUFHLGNBQWMsU0FBUztBQUFBLElBQzVCO0FBRUEsYUFBUyxlQUFlLE9BQU87QUFDN0IsYUFBTyxPQUFPLGFBQWEsS0FBSztBQUNoQyxVQUFJLE9BQU8sVUFBVSxlQUFlLEtBQUssT0FBTyxhQUFhLEdBQUc7QUFDOUQ7QUFBQSxVQUNFLFNBQVM7QUFBQSxVQUNULFlBQVksY0FDUixnQkFBZ0IsWUFBWSxXQUFXLElBQ3ZDO0FBQUEsUUFDTjtBQUFBLE1BQ0Y7QUFDQSxVQUFJLE9BQU8sVUFBVSxlQUFlLEtBQUssT0FBTyxlQUFlLEdBQUc7QUFDaEU7QUFBQSxVQUNFLFNBQVM7QUFBQSxVQUNULFlBQVksZ0JBQ1IsZ0JBQWdCLFlBQVksYUFBYSxJQUN6QztBQUFBLFFBQ047QUFBQSxNQUNGO0FBQ0EsVUFBSSxPQUFPLFVBQVUsZUFBZSxLQUFLLE9BQU8sV0FBVyxHQUFHO0FBQzVELFlBQUksT0FBTyxZQUFZLGNBQWMsVUFBVTtBQUM3QztBQUFBLFlBQ0UsU0FBUztBQUFBLFlBQ1QsR0FBRyxLQUFLLElBQUksR0FBRyxLQUFLLE1BQU0sWUFBWSxTQUFTLENBQUMsQ0FBQztBQUFBLFVBQ25EO0FBQUEsUUFDRixPQUFPO0FBQ0wsZ0NBQXNCLFNBQVMsYUFBYSxRQUFHO0FBQUEsUUFDakQ7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUVBLGFBQVMsc0JBQXNCO0FBQzdCLFVBQUksQ0FBQyxTQUFTLFlBQWE7QUFDM0IsWUFBTSxTQUFTLFVBQVU7QUFDekIsZUFBUyxZQUFZLGNBQWMsU0FBUyxhQUFhO0FBQ3pELGVBQVMsWUFBWSxVQUFVLE9BQU8sZUFBZSxDQUFDLE1BQU07QUFDNUQsZUFBUyxZQUFZLFVBQVUsT0FBTyxnQkFBZ0IsTUFBTTtBQUFBLElBQzlEO0FBRUEsYUFBUyxtQkFBbUIsU0FBUyxVQUFVLFFBQVE7QUFDckQsVUFBSSxDQUFDLFNBQVMsWUFBWTtBQUN4QjtBQUFBLE1BQ0Y7QUFDQSxZQUFNLFlBQVksU0FBUyxXQUFXO0FBQ3RDLFlBQU0sS0FBSyxTQUFTLEVBQ2pCLE9BQU8sQ0FBQyxRQUFRLElBQUksV0FBVyxRQUFRLEtBQUssUUFBUSxPQUFPLEVBQzNELFFBQVEsQ0FBQyxRQUFRLFVBQVUsT0FBTyxHQUFHLENBQUM7QUFDekMsZ0JBQVUsSUFBSSxPQUFPO0FBQ3JCLGdCQUFVLElBQUksU0FBUyxPQUFPLEVBQUU7QUFDaEMsZUFBUyxXQUFXLGNBQWM7QUFDbEMsZ0JBQVUsT0FBTyxpQkFBaUI7QUFDbEMsYUFBTyxXQUFXLE1BQU07QUFDdEIsa0JBQVUsSUFBSSxpQkFBaUI7QUFBQSxNQUNqQyxHQUFHLEdBQUk7QUFBQSxJQUNUO0FBRUEsYUFBUyxxQkFBcUIsU0FBUyxPQUFPLFNBQVM7QUFDckQsVUFBSSxDQUFDLFNBQVMsZUFBZ0I7QUFDOUIsWUFBTSxRQUFRLENBQUMsU0FBUyxRQUFRLFdBQVcsVUFBVSxTQUFTO0FBQzlELGVBQVMsZUFBZSxjQUFjO0FBQ3RDLFlBQU0sUUFBUSxDQUFDLE1BQU0sU0FBUyxlQUFlLFVBQVUsT0FBTyxRQUFRLENBQUMsRUFBRSxDQUFDO0FBQzFFLGVBQVMsZUFBZSxVQUFVLElBQUksUUFBUSxJQUFJLEVBQUU7QUFBQSxJQUN0RDtBQUVBLGFBQVMsWUFBWUEsUUFBTyxPQUFPO0FBQ2pDLFVBQUksQ0FBQyxTQUFTLFNBQVU7QUFDeEIsWUFBTSxRQUFRLGFBQWFBLE1BQUssS0FBS0E7QUFDckMsZUFBUyxTQUFTLGNBQWM7QUFDaEMsZUFBUyxTQUFTLFlBQVksa0JBQWtCQSxNQUFLO0FBQ3JELFVBQUksT0FBTztBQUNULGlCQUFTLFNBQVMsUUFBUTtBQUFBLE1BQzVCLE9BQU87QUFDTCxpQkFBUyxTQUFTLGdCQUFnQixPQUFPO0FBQUEsTUFDM0M7QUFBQSxJQUNGO0FBRUEsYUFBUyxnQkFBZ0IsS0FBSztBQUM1QixZQUFNLFFBQVEsT0FBTyxPQUFPLEVBQUU7QUFDOUIsVUFBSTtBQUNGLGVBQU8sTUFDSixVQUFVLEtBQUssRUFDZixRQUFRLG9CQUFvQixFQUFFLEVBQzlCLFlBQVk7QUFBQSxNQUNqQixTQUFTLEtBQUs7QUFDWixlQUFPLE1BQU0sWUFBWTtBQUFBLE1BQzNCO0FBQUEsSUFDRjtBQUVBLGFBQVMsc0JBQXNCLE9BQU8sVUFBVSxDQUFDLEdBQUc7QUFDbEQsVUFBSSxDQUFDLFNBQVMsV0FBWSxRQUFPO0FBQ2pDLFlBQU0sRUFBRSxnQkFBZ0IsTUFBTSxJQUFJO0FBQ2xDLFlBQU0sV0FBVyxPQUFPLFVBQVUsV0FBVyxRQUFRO0FBQ3JELFVBQUksQ0FBQyxpQkFBaUIsU0FBUyxhQUFhO0FBQzFDLGlCQUFTLFlBQVksUUFBUTtBQUFBLE1BQy9CO0FBQ0EsWUFBTSxVQUFVLFNBQVMsS0FBSztBQUM5QixZQUFNLGVBQWU7QUFDckIsWUFBTSxhQUFhLGdCQUFnQixPQUFPO0FBQzFDLFVBQUksVUFBVTtBQUNkLFlBQU0sT0FBTyxNQUFNLEtBQUssU0FBUyxXQUFXLGlCQUFpQixXQUFXLENBQUM7QUFDekUsV0FBSyxRQUFRLENBQUMsUUFBUTtBQUNwQixZQUFJLFVBQVUsT0FBTyxlQUFlLG1CQUFtQjtBQUN2RCxZQUFJLENBQUMsWUFBWTtBQUNmO0FBQUEsUUFDRjtBQUNBLGNBQU0sTUFBTSxJQUFJLFFBQVEsV0FBVztBQUNuQyxjQUFNLGdCQUFnQixnQkFBZ0IsR0FBRztBQUN6QyxZQUFJLGNBQWMsU0FBUyxVQUFVLEdBQUc7QUFDdEMsY0FBSSxVQUFVLElBQUksbUJBQW1CO0FBQ3JDLHFCQUFXO0FBQUEsUUFDYixPQUFPO0FBQ0wsY0FBSSxVQUFVLElBQUksYUFBYTtBQUFBLFFBQ2pDO0FBQUEsTUFDRixDQUFDO0FBQ0QsZUFBUyxXQUFXLFVBQVUsT0FBTyxZQUFZLFFBQVEsT0FBTyxDQUFDO0FBQ2pFLFVBQUksU0FBUyxhQUFhO0FBQ3hCLFlBQUksV0FBVyxZQUFZLEdBQUc7QUFDNUIsbUJBQVMsWUFBWSxVQUFVLE9BQU8sUUFBUTtBQUM5QyxtQkFBUyxZQUFZO0FBQUEsWUFDbkI7QUFBQSxZQUNBLFNBQVMsWUFBWSxhQUFhLFdBQVcsS0FBSztBQUFBLFVBQ3BEO0FBQUEsUUFDRixPQUFPO0FBQ0wsbUJBQVMsWUFBWSxVQUFVLElBQUksUUFBUTtBQUFBLFFBQzdDO0FBQUEsTUFDRjtBQUNBLFVBQUksU0FBUyxZQUFZO0FBQ3ZCLFlBQUksU0FBUztBQUNYLGNBQUksVUFBVTtBQUNkLGNBQUksWUFBWSxHQUFHO0FBQ2pCLHNCQUFVO0FBQUEsVUFDWixXQUFXLFVBQVUsR0FBRztBQUN0QixzQkFBVSxHQUFHLE9BQU87QUFBQSxVQUN0QjtBQUNBLG1CQUFTLFdBQVcsY0FBYztBQUFBLFFBQ3BDLE9BQU87QUFDTCxtQkFBUyxXQUFXLGNBQWM7QUFBQSxRQUNwQztBQUFBLE1BQ0Y7QUFDQSxhQUFPO0FBQUEsSUFDVDtBQUVBLGFBQVMsMEJBQTBCO0FBQ2pDLFVBQUksTUFBTSxjQUFjO0FBQ3RCLDhCQUFzQixNQUFNLGNBQWMsRUFBRSxlQUFlLEtBQUssQ0FBQztBQUFBLE1BQ25FLFdBQVcsU0FBUyxZQUFZO0FBQzlCLGlCQUFTLFdBQVcsVUFBVSxPQUFPLFVBQVU7QUFDL0MsY0FBTSxPQUFPLE1BQU07QUFBQSxVQUNqQixTQUFTLFdBQVcsaUJBQWlCLFdBQVc7QUFBQSxRQUNsRDtBQUNBLGFBQUssUUFBUSxDQUFDLFFBQVE7QUFDcEIsY0FBSSxVQUFVLE9BQU8sZUFBZSxtQkFBbUI7QUFBQSxRQUN6RCxDQUFDO0FBQ0QsWUFBSSxTQUFTLGFBQWE7QUFDeEIsbUJBQVMsWUFBWSxVQUFVLElBQUksUUFBUTtBQUFBLFFBQzdDO0FBQ0EsWUFBSSxTQUFTLFlBQVk7QUFDdkIsbUJBQVMsV0FBVyxjQUFjO0FBQUEsUUFDcEM7QUFBQSxNQUNGO0FBQUEsSUFDRjtBQUVBLGFBQVMsc0JBQXNCLFFBQVEsTUFBTTtBQUMzQyxZQUFNLGVBQWU7QUFDckIsVUFBSSxTQUFTLGFBQWE7QUFDeEIsaUJBQVMsWUFBWSxRQUFRO0FBQUEsTUFDL0I7QUFDQSw4QkFBd0I7QUFDeEIsVUFBSSxTQUFTLFNBQVMsYUFBYTtBQUNqQyxpQkFBUyxZQUFZLE1BQU07QUFBQSxNQUM3QjtBQUFBLElBQ0Y7QUFFQSxhQUFTLGNBQWMsU0FBUyxVQUFVLENBQUMsR0FBRztBQUM1QyxZQUFNLEVBQUUsVUFBVSxNQUFNLElBQUk7QUFDNUIsVUFBSSxDQUFDLE1BQU0sUUFBUSxPQUFPLEtBQUssUUFBUSxXQUFXLEdBQUc7QUFDbkQsWUFBSSxTQUFTO0FBQ1gsbUJBQVMsV0FBVyxZQUFZO0FBQ2hDLGdCQUFNLHNCQUFzQjtBQUM1QiwyQkFBaUI7QUFDakIsd0JBQWMsTUFBTTtBQUFBLFFBQ3RCO0FBQ0E7QUFBQSxNQUNGO0FBQ0EsVUFBSSxTQUFTO0FBQ1gsaUJBQVMsV0FBVyxZQUFZO0FBQ2hDLGNBQU0sc0JBQXNCO0FBQzVCLGNBQU0sWUFBWTtBQUNsQixjQUFNLFlBQVk7QUFDbEIsc0JBQWMsTUFBTTtBQUFBLE1BQ3RCO0FBQ0EsVUFBSSxNQUFNLHVCQUF1QixDQUFDLFNBQVM7QUFDekMsY0FBTSxnQkFBZ0I7QUFDdEIsY0FBTSxPQUFPLE1BQU07QUFBQSxVQUNqQixTQUFTLFdBQVcsaUJBQWlCLFdBQVc7QUFBQSxRQUNsRDtBQUNBLGFBQUssUUFBUSxDQUFDLFFBQVE7QUFDcEIsZ0JBQU0sYUFBYSxJQUFJLFFBQVE7QUFDL0IsY0FBSSxjQUFjLGNBQWMsSUFBSSxJQUFJLFVBQVUsR0FBRztBQUNuRCxrQkFBTSxjQUFjLElBQUksUUFBUSxRQUFRO0FBQ3hDLGdCQUFJLGFBQWE7QUFDZiwwQkFBWSxLQUFLLFdBQVc7QUFBQSxZQUM5QjtBQUNBO0FBQUEsVUFDRjtBQUNBLGdCQUFNLFNBQVMsSUFBSSxjQUFjLGNBQWM7QUFDL0MsZ0JBQU0sUUFBTyxpQ0FBUSxjQUFjLGtCQUFpQjtBQUNwRCxnQkFBTSxPQUNKLElBQUksUUFBUSxTQUNYLElBQUksVUFBVSxTQUFTLFdBQVcsSUFDL0IsU0FDQSxJQUFJLFVBQVUsU0FBUyxnQkFBZ0IsSUFDdkMsY0FDQTtBQUNOLGdCQUFNLE9BQ0osSUFBSSxRQUFRLFdBQVcsSUFBSSxRQUFRLFFBQVEsU0FBUyxJQUNoRCxJQUFJLFFBQVEsVUFDWixTQUNBLGtCQUFrQixNQUFNLElBQ3hCLElBQUksWUFBWSxLQUFLO0FBQzNCLGdCQUFNLFlBQ0osSUFBSSxRQUFRLGFBQWEsSUFBSSxRQUFRLFVBQVUsU0FBUyxJQUNwRCxJQUFJLFFBQVEsWUFDWixPQUNBLEtBQUssWUFBWSxLQUFLLElBQ3RCLE9BQU87QUFDYixnQkFBTSxZQUFZLGNBQWMsU0FBUztBQUFBLFlBQ3ZDLElBQUk7QUFBQSxZQUNKO0FBQUEsWUFDQTtBQUFBLFlBQ0E7QUFBQSxZQUNBO0FBQUEsVUFDRixDQUFDO0FBQ0QsY0FBSSxRQUFRLFlBQVk7QUFDeEIsY0FBSSxRQUFRLE9BQU87QUFDbkIsY0FBSSxRQUFRLFVBQVU7QUFDdEIsY0FBSSxRQUFRLFlBQVk7QUFDeEIsc0JBQVksS0FBSyxJQUFJO0FBQUEsUUFDdkIsQ0FBQztBQUNELGNBQU0sZ0JBQWdCO0FBQ3RCLGdDQUF3QjtBQUN4QjtBQUFBLE1BQ0Y7QUFDQSxZQUFNLGdCQUFnQjtBQUN0QixjQUNHLE1BQU0sRUFDTixRQUFRLEVBQ1IsUUFBUSxDQUFDLFNBQVM7QUFDakIsWUFBSSxLQUFLLE9BQU87QUFDZCx3QkFBYyxRQUFRLEtBQUssT0FBTztBQUFBLFlBQ2hDLFdBQVcsS0FBSztBQUFBLFVBQ2xCLENBQUM7QUFBQSxRQUNIO0FBQ0EsWUFBSSxLQUFLLFVBQVU7QUFDakIsd0JBQWMsYUFBYSxLQUFLLFVBQVU7QUFBQSxZQUN4QyxXQUFXLEtBQUs7QUFBQSxVQUNsQixDQUFDO0FBQUEsUUFDSDtBQUFBLE1BQ0YsQ0FBQztBQUNILFlBQU0sZ0JBQWdCO0FBQ3RCLFlBQU0sc0JBQXNCO0FBQzVCLHFCQUFlLEVBQUUsUUFBUSxNQUFNLENBQUM7QUFDaEMsdUJBQWlCO0FBQUEsSUFDbkI7QUFFQSxhQUFTLGNBQWM7QUFDckIsWUFBTSxZQUFZO0FBQ2xCLFlBQU0sS0FBSyxPQUFPO0FBQ2xCLFlBQU0sa0JBQWtCLGNBQWMsY0FBYztBQUNwRCxZQUFNLFlBQVk7QUFBQSxRQUNoQjtBQUFBLFFBQ0E7QUFBQSxRQUNBO0FBQUEsVUFDRSxTQUFTO0FBQUEsVUFDVCxXQUFXO0FBQUEsVUFDWCxXQUFXLE1BQU07QUFBQSxVQUNqQixVQUFVLEVBQUUsV0FBVyxLQUFLO0FBQUEsUUFDOUI7QUFBQSxNQUNGO0FBQ0EscUJBQWUsRUFBRSxlQUFlLEdBQUcsQ0FBQztBQUNwQyxVQUFJLE1BQU0sa0JBQWtCO0FBQzFCLHFCQUFhLE1BQU0sZ0JBQWdCO0FBQUEsTUFDckM7QUFDQSx3QkFBa0IsNkJBQXFCLE1BQU07QUFBQSxJQUMvQztBQUVBLGFBQVMsY0FBYztBQUNyQixhQUFPLFFBQVEsTUFBTSxTQUFTO0FBQUEsSUFDaEM7QUFFQSxhQUFTLGtCQUFrQjtBQUN6QixhQUFPLFFBQVEsTUFBTSxTQUFTO0FBQUEsSUFDaEM7QUFFQSxhQUFTLGFBQWEsT0FBTztBQUMzQixVQUFJLENBQUMsTUFBTSxXQUFXO0FBQ3BCLG9CQUFZO0FBQUEsTUFDZDtBQUNBLFlBQU0sY0FBYyxXQUFXO0FBQy9CLFlBQU0sYUFBYSxTQUFTO0FBQzVCLFlBQU0sU0FBUyxNQUFNLFVBQVUsY0FBYyxjQUFjO0FBQzNELFVBQUksUUFBUTtBQUNWLGVBQU8sWUFBWSxHQUFHLGVBQWUsTUFBTSxTQUFTLENBQUM7QUFBQSxNQUN2RDtBQUNBLFVBQUksTUFBTSxpQkFBaUI7QUFDekIsc0JBQWMsT0FBTyxNQUFNLGlCQUFpQjtBQUFBLFVBQzFDLE1BQU0sTUFBTTtBQUFBLFVBQ1osVUFBVSxFQUFFLFdBQVcsS0FBSztBQUFBLFFBQzlCLENBQUM7QUFBQSxNQUNIO0FBQ0EscUJBQWUsRUFBRSxlQUFlLE9BQU8sRUFBRSxDQUFDO0FBQzFDLFVBQUksYUFBYTtBQUNmLHVCQUFlLEVBQUUsUUFBUSxNQUFNLENBQUM7QUFBQSxNQUNsQztBQUFBLElBQ0Y7QUFFQSxhQUFTLFVBQVUsTUFBTTtBQUN2QixVQUFJLENBQUMsTUFBTSxXQUFXO0FBQ3BCO0FBQUEsTUFDRjtBQUNBLFlBQU0sU0FBUyxNQUFNLFVBQVUsY0FBYyxjQUFjO0FBQzNELFVBQUksUUFBUTtBQUNWLGVBQU8sWUFBWSxlQUFlLE1BQU0sU0FBUztBQUNqRCxjQUFNLE9BQU8sU0FBUyxjQUFjLEtBQUs7QUFDekMsYUFBSyxZQUFZO0FBQ2pCLGNBQU0sS0FBSyxRQUFRLEtBQUssWUFBWSxLQUFLLFlBQVksT0FBTztBQUM1RCxhQUFLLGNBQWMsZ0JBQWdCLEVBQUU7QUFDckMsWUFBSSxRQUFRLEtBQUssT0FBTztBQUN0QixlQUFLLFVBQVUsSUFBSSxhQUFhO0FBQ2hDLGVBQUssY0FBYyxHQUFHLEtBQUssV0FBVyxXQUFNLEtBQUssS0FBSztBQUFBLFFBQ3hEO0FBQ0EsZUFBTyxZQUFZLElBQUk7QUFDdkIsb0JBQVksTUFBTSxXQUFXLFdBQVc7QUFDeEMscUJBQWEsTUFBTSxXQUFXLFdBQVc7QUFDekMsWUFBSSxXQUFXLEdBQUc7QUFDaEIseUJBQWUsRUFBRSxRQUFRLEtBQUssQ0FBQztBQUFBLFFBQ2pDLE9BQU87QUFDTCwyQkFBaUI7QUFBQSxRQUNuQjtBQUNBLFlBQUksTUFBTSxpQkFBaUI7QUFDekIsd0JBQWMsT0FBTyxNQUFNLGlCQUFpQjtBQUFBLFlBQzFDLE1BQU0sTUFBTTtBQUFBLFlBQ1osV0FBVztBQUFBLFlBQ1gsVUFBVTtBQUFBLGNBQ1IsV0FBVztBQUFBLGNBQ1gsR0FBSSxRQUFRLEtBQUssUUFBUSxFQUFFLE9BQU8sS0FBSyxNQUFNLElBQUksRUFBRSxPQUFPLEtBQUs7QUFBQSxZQUNqRTtBQUFBLFVBQ0YsQ0FBQztBQUFBLFFBQ0g7QUFDQSx1QkFBZSxFQUFFLGVBQWUsR0FBRyxDQUFDO0FBQUEsTUFDdEM7QUFDQSxZQUFNLFdBQVcsUUFBUSxRQUFRLEtBQUssS0FBSztBQUMzQztBQUFBLFFBQ0UsV0FDSSxxREFDQTtBQUFBLFFBQ0osV0FBVyxXQUFXO0FBQUEsTUFDeEI7QUFDQSwyQkFBcUIsV0FBVyxNQUFPLElBQUk7QUFDM0MsWUFBTSxZQUFZO0FBQ2xCLFlBQU0sWUFBWTtBQUNsQixZQUFNLGtCQUFrQjtBQUFBLElBQzFCO0FBRUEsYUFBUyx5QkFBeUIsYUFBYTtBQUM3QyxVQUFJLENBQUMsU0FBUyxhQUFjO0FBQzVCLFVBQUksQ0FBQyxNQUFNLFFBQVEsV0FBVyxLQUFLLFlBQVksV0FBVyxFQUFHO0FBQzdELFlBQU0sVUFBVSxNQUFNO0FBQUEsUUFDcEIsU0FBUyxhQUFhLGlCQUFpQixXQUFXO0FBQUEsTUFDcEQ7QUFDQSxZQUFNLFNBQVMsb0JBQUksSUFBSTtBQUN2QixjQUFRLFFBQVEsQ0FBQyxRQUFRLE9BQU8sSUFBSSxJQUFJLFFBQVEsUUFBUSxHQUFHLENBQUM7QUFDNUQsWUFBTSxPQUFPLFNBQVMsdUJBQXVCO0FBQzdDLGtCQUFZLFFBQVEsQ0FBQyxRQUFRO0FBQzNCLFlBQUksT0FBTyxJQUFJLEdBQUcsR0FBRztBQUNuQixlQUFLLFlBQVksT0FBTyxJQUFJLEdBQUcsQ0FBQztBQUNoQyxpQkFBTyxPQUFPLEdBQUc7QUFBQSxRQUNuQjtBQUFBLE1BQ0YsQ0FBQztBQUNELGFBQU8sUUFBUSxDQUFDLFFBQVEsS0FBSyxZQUFZLEdBQUcsQ0FBQztBQUM3QyxlQUFTLGFBQWEsWUFBWTtBQUNsQyxlQUFTLGFBQWEsWUFBWSxJQUFJO0FBQUEsSUFDeEM7QUFFQSxhQUFTLFdBQVcsR0FBRztBQUNyQixZQUFNLE9BQU8sQ0FBQztBQUNkLFVBQUksS0FBSyxPQUFPLEVBQUUsUUFBUSxhQUFhO0FBQ3JDLGNBQU0sTUFBTSxPQUFPLEVBQUUsR0FBRztBQUN4QixZQUFJLENBQUMsT0FBTyxNQUFNLEdBQUcsR0FBRztBQUN0QixlQUFLLEtBQUssT0FBTyxJQUFJLFFBQVEsQ0FBQyxDQUFDLEdBQUc7QUFBQSxRQUNwQztBQUFBLE1BQ0Y7QUFDQSxVQUFJLEtBQUssT0FBTyxFQUFFLFlBQVksYUFBYTtBQUN6QyxjQUFNLE9BQU8sT0FBTyxFQUFFLE9BQU87QUFDN0IsWUFBSSxDQUFDLE9BQU8sTUFBTSxJQUFJLEdBQUc7QUFDdkIsZUFBSyxLQUFLLFFBQVEsSUFBSSxLQUFLO0FBQUEsUUFDN0I7QUFBQSxNQUNGO0FBQ0EsYUFBTyxLQUFLLEtBQUssVUFBSyxLQUFLO0FBQUEsSUFDN0I7QUFFQSxhQUFTLGVBQWU7QUFDdEIsVUFBSSxTQUFTLFVBQVU7QUFDckIsaUJBQVMsU0FBUyxpQkFBaUIsVUFBVSxDQUFDLFVBQVU7QUFDdEQsZ0JBQU0sZUFBZTtBQUNyQixnQkFBTSxRQUFRLFNBQVMsT0FBTyxTQUFTLElBQUksS0FBSztBQUNoRCxlQUFLLFVBQVUsRUFBRSxLQUFLLENBQUM7QUFBQSxRQUN6QixDQUFDO0FBQUEsTUFDSDtBQUVBLFVBQUksU0FBUyxjQUFjO0FBQ3pCLGlCQUFTLGFBQWEsaUJBQWlCLFNBQVMsQ0FBQyxVQUFVO0FBQ3pELGdCQUFNLFNBQVMsTUFBTTtBQUNyQixjQUFJLEVBQUUsa0JBQWtCLG9CQUFvQjtBQUMxQztBQUFBLFVBQ0Y7QUFDQSxnQkFBTSxTQUFTLE9BQU8sUUFBUTtBQUM5QixjQUFJLENBQUMsUUFBUTtBQUNYO0FBQUEsVUFDRjtBQUNBLGVBQUssZ0JBQWdCLEVBQUUsT0FBTyxDQUFDO0FBQUEsUUFDakMsQ0FBQztBQUFBLE1BQ0g7QUFFQSxVQUFJLFNBQVMsYUFBYTtBQUN4QixpQkFBUyxZQUFZLGlCQUFpQixTQUFTLENBQUMsVUFBVTtBQUN4RCxlQUFLLGlCQUFpQixFQUFFLE9BQU8sTUFBTSxPQUFPLFNBQVMsR0FBRyxDQUFDO0FBQUEsUUFDM0QsQ0FBQztBQUNELGlCQUFTLFlBQVksaUJBQWlCLFdBQVcsQ0FBQyxVQUFVO0FBQzFELGNBQUksTUFBTSxRQUFRLFVBQVU7QUFDMUIsa0JBQU0sZUFBZTtBQUNyQixpQkFBSyxjQUFjO0FBQUEsVUFDckI7QUFBQSxRQUNGLENBQUM7QUFBQSxNQUNIO0FBRUEsVUFBSSxTQUFTLGFBQWE7QUFDeEIsaUJBQVMsWUFBWSxpQkFBaUIsU0FBUyxNQUFNO0FBQ25ELGVBQUssY0FBYztBQUFBLFFBQ3JCLENBQUM7QUFBQSxNQUNIO0FBRUEsVUFBSSxTQUFTLFlBQVk7QUFDdkIsaUJBQVMsV0FBVztBQUFBLFVBQWlCO0FBQUEsVUFBUyxNQUM1QyxLQUFLLFVBQVUsRUFBRSxRQUFRLE9BQU8sQ0FBQztBQUFBLFFBQ25DO0FBQUEsTUFDRjtBQUNBLFVBQUksU0FBUyxnQkFBZ0I7QUFDM0IsaUJBQVMsZUFBZTtBQUFBLFVBQWlCO0FBQUEsVUFBUyxNQUNoRCxLQUFLLFVBQVUsRUFBRSxRQUFRLFdBQVcsQ0FBQztBQUFBLFFBQ3ZDO0FBQUEsTUFDRjtBQUNBLFVBQUksU0FBUyxZQUFZO0FBQ3ZCLGlCQUFTLFdBQVcsaUJBQWlCLFNBQVMsTUFBTSxLQUFLLGFBQWEsQ0FBQztBQUFBLE1BQ3pFO0FBRUEsVUFBSSxTQUFTLFFBQVE7QUFDbkIsaUJBQVMsT0FBTyxpQkFBaUIsU0FBUyxDQUFDLFVBQVU7QUFDbkQsOEJBQW9CO0FBQ3BCLHlCQUFlO0FBQ2YsZ0JBQU0sUUFBUSxNQUFNLE9BQU8sU0FBUztBQUNwQyxjQUFJLENBQUMsTUFBTSxLQUFLLEdBQUc7QUFDakIsa0NBQXNCO0FBQUEsVUFDeEI7QUFDQSxlQUFLLGdCQUFnQixFQUFFLE1BQU0sQ0FBQztBQUFBLFFBQ2hDLENBQUM7QUFDRCxpQkFBUyxPQUFPLGlCQUFpQixTQUFTLE1BQU07QUFDOUMsaUJBQU8sV0FBVyxNQUFNO0FBQ3RCLGdDQUFvQjtBQUNwQiwyQkFBZTtBQUNmLGlCQUFLLGdCQUFnQixFQUFFLE9BQU8sU0FBUyxPQUFPLFNBQVMsR0FBRyxDQUFDO0FBQUEsVUFDN0QsR0FBRyxDQUFDO0FBQUEsUUFDTixDQUFDO0FBQ0QsaUJBQVMsT0FBTyxpQkFBaUIsV0FBVyxDQUFDLFVBQVU7QUFDckQsZUFBSyxNQUFNLFdBQVcsTUFBTSxZQUFZLE1BQU0sUUFBUSxTQUFTO0FBQzdELGtCQUFNLGVBQWU7QUFDckIsaUJBQUssVUFBVSxFQUFFLE9BQU8sU0FBUyxPQUFPLFNBQVMsSUFBSSxLQUFLLEVBQUUsQ0FBQztBQUFBLFVBQy9EO0FBQUEsUUFDRixDQUFDO0FBQ0QsaUJBQVMsT0FBTyxpQkFBaUIsU0FBUyxNQUFNO0FBQzlDO0FBQUEsWUFDRTtBQUFBLFlBQ0E7QUFBQSxVQUNGO0FBQ0EsK0JBQXFCLEdBQUk7QUFBQSxRQUMzQixDQUFDO0FBQUEsTUFDSDtBQUVBLFVBQUksU0FBUyxZQUFZO0FBQ3ZCLGlCQUFTLFdBQVcsaUJBQWlCLFVBQVUsTUFBTTtBQUNuRCxjQUFJLFdBQVcsR0FBRztBQUNoQiw2QkFBaUI7QUFBQSxVQUNuQixPQUFPO0FBQ0wsNkJBQWlCO0FBQUEsVUFDbkI7QUFBQSxRQUNGLENBQUM7QUFBQSxNQUNIO0FBRUEsVUFBSSxTQUFTLGNBQWM7QUFDekIsaUJBQVMsYUFBYSxpQkFBaUIsU0FBUyxNQUFNO0FBQ3BELHlCQUFlLEVBQUUsUUFBUSxLQUFLLENBQUM7QUFDL0IsY0FBSSxTQUFTLFFBQVE7QUFDbkIscUJBQVMsT0FBTyxNQUFNO0FBQUEsVUFDeEI7QUFBQSxRQUNGLENBQUM7QUFBQSxNQUNIO0FBRUEsYUFBTyxpQkFBaUIsVUFBVSxNQUFNO0FBQ3RDLFlBQUksV0FBVyxHQUFHO0FBQ2hCLHlCQUFlLEVBQUUsUUFBUSxNQUFNLENBQUM7QUFBQSxRQUNsQztBQUFBLE1BQ0YsQ0FBQztBQUVELDBCQUFvQjtBQUNwQixhQUFPLGlCQUFpQixVQUFVLE1BQU07QUFDdEMsNEJBQW9CO0FBQ3BCLDJCQUFtQixxQ0FBK0IsTUFBTTtBQUFBLE1BQzFELENBQUM7QUFDRCxhQUFPLGlCQUFpQixXQUFXLE1BQU07QUFDdkMsNEJBQW9CO0FBQ3BCLDJCQUFtQiwrQkFBNEIsUUFBUTtBQUFBLE1BQ3pELENBQUM7QUFFRCxZQUFNLFlBQVksU0FBUyxlQUFlLGtCQUFrQjtBQUM1RCxZQUFNLGNBQWM7QUFFcEIsZUFBUyxjQUFjLFNBQVM7QUFDOUIsaUJBQVMsS0FBSyxVQUFVLE9BQU8sYUFBYSxPQUFPO0FBQ25ELFlBQUksV0FBVztBQUNiLG9CQUFVLGNBQWMsVUFBVSxlQUFlO0FBQ2pELG9CQUFVLGFBQWEsZ0JBQWdCLFVBQVUsU0FBUyxPQUFPO0FBQUEsUUFDbkU7QUFBQSxNQUNGO0FBRUEsVUFBSTtBQUNGLHNCQUFjLE9BQU8sYUFBYSxRQUFRLFdBQVcsTUFBTSxHQUFHO0FBQUEsTUFDaEUsU0FBUyxLQUFLO0FBQ1osZ0JBQVEsS0FBSyx1Q0FBdUMsR0FBRztBQUFBLE1BQ3pEO0FBRUEsVUFBSSxXQUFXO0FBQ2Isa0JBQVUsaUJBQWlCLFNBQVMsTUFBTTtBQUN4QyxnQkFBTSxVQUFVLENBQUMsU0FBUyxLQUFLLFVBQVUsU0FBUyxXQUFXO0FBQzdELHdCQUFjLE9BQU87QUFDckIsY0FBSTtBQUNGLG1CQUFPLGFBQWEsUUFBUSxhQUFhLFVBQVUsTUFBTSxHQUFHO0FBQUEsVUFDOUQsU0FBUyxLQUFLO0FBQ1osb0JBQVEsS0FBSywwQ0FBMEMsR0FBRztBQUFBLFVBQzVEO0FBQUEsUUFDRixDQUFDO0FBQUEsTUFDSDtBQUFBLElBQ0Y7QUFFQSxhQUFTLGFBQWE7QUFDcEIscUJBQWUsRUFBRSxhQUFhLE1BQU0sZUFBZSxNQUFNLFdBQVcsS0FBSyxDQUFDO0FBQzFFLDBCQUFvQjtBQUNwQixxQkFBZTtBQUNmLDRCQUFzQjtBQUN0QixtQkFBYTtBQUFBLElBQ2Y7QUFFQSxXQUFPO0FBQUEsTUFDTDtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQSxJQUFJLFlBQVksT0FBTztBQUNyQixlQUFPLE9BQU8sYUFBYSxLQUFLO0FBQUEsTUFDbEM7QUFBQSxNQUNBLElBQUksY0FBYztBQUNoQixlQUFPLEVBQUUsR0FBRyxZQUFZO0FBQUEsTUFDMUI7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLElBQ0Y7QUFBQSxFQUNGOzs7QUNyNkJBLE1BQU0sc0JBQXNCO0FBRTVCLFdBQVMsa0JBQWtCO0FBQ3pCLFFBQUk7QUFDRixhQUFPLE9BQU8sV0FBVyxlQUFlLFFBQVEsT0FBTyxZQUFZO0FBQUEsSUFDckUsU0FBUyxLQUFLO0FBQ1osY0FBUSxLQUFLLGlDQUFpQyxHQUFHO0FBQ2pELGFBQU87QUFBQSxJQUNUO0FBQUEsRUFDRjtBQUVPLFdBQVMsa0JBQWtCLFNBQVMsQ0FBQyxHQUFHO0FBQzdDLFVBQU0sYUFBYSxPQUFPLGNBQWM7QUFDeEMsUUFBSSxnQkFDRixPQUFPLE9BQU8sVUFBVSxZQUFZLE9BQU8sTUFBTSxLQUFLLE1BQU0sS0FDeEQsT0FBTyxRQUNQO0FBRU4sYUFBUyxhQUFhLE9BQU87QUFDM0IsVUFBSSxDQUFDLE9BQU87QUFDVjtBQUFBLE1BQ0Y7QUFDQSxzQkFBZ0I7QUFFaEIsVUFBSSxDQUFDLGdCQUFnQixHQUFHO0FBQ3RCO0FBQUEsTUFDRjtBQUVBLFVBQUk7QUFDRixlQUFPLGFBQWEsUUFBUSxZQUFZLEtBQUs7QUFBQSxNQUMvQyxTQUFTLEtBQUs7QUFDWixnQkFBUSxLQUFLLHlDQUF5QyxHQUFHO0FBQUEsTUFDM0Q7QUFBQSxJQUNGO0FBRUEsYUFBUyxrQkFBa0I7QUFDekIsVUFBSSxDQUFDLGdCQUFnQixHQUFHO0FBQ3RCLGVBQU87QUFBQSxNQUNUO0FBRUEsVUFBSTtBQUNGLGNBQU0sU0FBUyxPQUFPLGFBQWEsUUFBUSxVQUFVO0FBQ3JELGVBQU8sVUFBVTtBQUFBLE1BQ25CLFNBQVMsS0FBSztBQUNaLGdCQUFRLEtBQUssd0NBQXdDLEdBQUc7QUFDeEQsZUFBTztBQUFBLE1BQ1Q7QUFBQSxJQUNGO0FBRUEsYUFBUyxhQUFhO0FBQ3BCLHNCQUFnQjtBQUVoQixVQUFJLENBQUMsZ0JBQWdCLEdBQUc7QUFDdEI7QUFBQSxNQUNGO0FBRUEsVUFBSTtBQUNGLGVBQU8sYUFBYSxXQUFXLFVBQVU7QUFBQSxNQUMzQyxTQUFTLEtBQUs7QUFDWixnQkFBUSxLQUFLLHlDQUF5QyxHQUFHO0FBQUEsTUFDM0Q7QUFBQSxJQUNGO0FBRUEsUUFBSSxlQUFlO0FBQ2pCLG1CQUFhLGFBQWE7QUFBQSxJQUM1QjtBQUVBLG1CQUFlLFNBQVM7QUFDdEIsWUFBTSxTQUFTLGdCQUFnQjtBQUMvQixVQUFJLFFBQVE7QUFDVixlQUFPO0FBQUEsTUFDVDtBQUNBLFVBQUksZUFBZTtBQUNqQixlQUFPO0FBQUEsTUFDVDtBQUNBLFlBQU0sSUFBSTtBQUFBLFFBQ1IsNkNBQTZDLFVBQVU7QUFBQSxNQUN6RDtBQUFBLElBQ0Y7QUFFQSxXQUFPO0FBQUEsTUFDTDtBQUFBLE1BQ0E7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLElBQ0Y7QUFBQSxFQUNGOzs7QUNwRk8sV0FBUyxrQkFBa0IsRUFBRSxRQUFRLEtBQUssR0FBRztBQUNsRCxtQkFBZSxnQkFBZ0IsTUFBTSxVQUFVLENBQUMsR0FBRztBQUNqRCxVQUFJO0FBQ0osVUFBSTtBQUNGLGNBQU0sTUFBTSxLQUFLLE9BQU87QUFBQSxNQUMxQixTQUFTLEtBQUs7QUFFWixjQUFNLElBQUksTUFBTSxpREFBaUQ7QUFBQSxNQUNuRTtBQUNBLFlBQU0sVUFBVSxJQUFJLFFBQVEsUUFBUSxXQUFXLENBQUMsQ0FBQztBQUNqRCxVQUFJLENBQUMsUUFBUSxJQUFJLGVBQWUsR0FBRztBQUNqQyxnQkFBUSxJQUFJLGlCQUFpQixVQUFVLEdBQUcsRUFBRTtBQUFBLE1BQzlDO0FBQ0EsYUFBTyxNQUFNLE9BQU8sUUFBUSxJQUFJLEdBQUcsRUFBRSxHQUFHLFNBQVMsUUFBUSxDQUFDO0FBQUEsSUFDNUQ7QUFFQSxtQkFBZSxjQUFjO0FBQzNCLFlBQU0sT0FBTyxNQUFNLGdCQUFnQiwwQkFBMEI7QUFBQSxRQUMzRCxRQUFRO0FBQUEsTUFDVixDQUFDO0FBQ0QsVUFBSSxDQUFDLEtBQUssSUFBSTtBQUNaLGNBQU0sSUFBSSxNQUFNLGlCQUFpQixLQUFLLE1BQU0sRUFBRTtBQUFBLE1BQ2hEO0FBQ0EsWUFBTSxPQUFPLE1BQU0sS0FBSyxLQUFLO0FBQzdCLFVBQUksQ0FBQyxRQUFRLENBQUMsS0FBSyxRQUFRO0FBQ3pCLGNBQU0sSUFBSSxNQUFNLDBCQUEwQjtBQUFBLE1BQzVDO0FBQ0EsYUFBTyxLQUFLO0FBQUEsSUFDZDtBQUVBLG1CQUFlLFNBQVMsU0FBUztBQUMvQixZQUFNLE9BQU8sTUFBTSxnQkFBZ0IsNkJBQTZCO0FBQUEsUUFDOUQsUUFBUTtBQUFBLFFBQ1IsU0FBUyxFQUFFLGdCQUFnQixtQkFBbUI7QUFBQSxRQUM5QyxNQUFNLEtBQUssVUFBVSxFQUFFLFFBQVEsQ0FBQztBQUFBLE1BQ2xDLENBQUM7QUFDRCxVQUFJLENBQUMsS0FBSyxJQUFJO0FBQ1osY0FBTSxVQUFVLE1BQU0sS0FBSyxLQUFLO0FBQ2hDLGNBQU0sSUFBSSxNQUFNLFFBQVEsS0FBSyxNQUFNLEtBQUssT0FBTyxFQUFFO0FBQUEsTUFDbkQ7QUFDQSxhQUFPO0FBQUEsSUFDVDtBQUVBLG1CQUFlLGdCQUFnQixRQUFRO0FBQ3JDLFlBQU0sT0FBTyxNQUFNLGdCQUFnQiwwQkFBMEI7QUFBQSxRQUMzRCxRQUFRO0FBQUEsUUFDUixTQUFTLEVBQUUsZ0JBQWdCLG1CQUFtQjtBQUFBLFFBQzlDLE1BQU0sS0FBSyxVQUFVO0FBQUEsVUFDbkI7QUFBQSxVQUNBLFNBQVMsQ0FBQyxRQUFRLGFBQWEsU0FBUztBQUFBLFFBQzFDLENBQUM7QUFBQSxNQUNILENBQUM7QUFDRCxVQUFJLENBQUMsS0FBSyxJQUFJO0FBQ1osY0FBTSxJQUFJLE1BQU0scUJBQXFCLEtBQUssTUFBTSxFQUFFO0FBQUEsTUFDcEQ7QUFDQSxhQUFPLEtBQUssS0FBSztBQUFBLElBQ25CO0FBRUEsV0FBTztBQUFBLE1BQ0w7QUFBQSxNQUNBO0FBQUEsTUFDQTtBQUFBLElBQ0Y7QUFBQSxFQUNGOzs7QUMvREEsV0FBUyxvQkFBb0IsV0FBVztBQUN0QyxVQUFNLFFBQVEsT0FBTyxFQUFFLFFBQVEsU0FBUyxHQUFHO0FBQzNDLFdBQU8sZ0JBQWdCLEtBQUssSUFBSSxTQUFTO0FBQUEsRUFDM0M7QUFFQSxXQUFTLG9CQUFvQixPQUFPO0FBQ2xDLFVBQU0sUUFBUSxDQUFDLHdDQUF3QyxFQUFFO0FBQ3pELFVBQU0sUUFBUSxDQUFDLFNBQVM7QUFDdEIsWUFBTSxPQUFPLEtBQUssT0FBTyxLQUFLLEtBQUssWUFBWSxJQUFJO0FBQ25ELFlBQU0sS0FBSyxNQUFNLElBQUksRUFBRTtBQUN2QixVQUFJLEtBQUssV0FBVztBQUNsQixjQUFNLEtBQUsscUJBQWtCLEtBQUssU0FBUyxFQUFFO0FBQUEsTUFDL0M7QUFDQSxVQUFJLEtBQUssWUFBWSxPQUFPLEtBQUssS0FBSyxRQUFRLEVBQUUsU0FBUyxHQUFHO0FBQzFELGVBQU8sUUFBUSxLQUFLLFFBQVEsRUFBRSxRQUFRLENBQUMsQ0FBQyxLQUFLLEtBQUssTUFBTTtBQUN0RCxnQkFBTSxLQUFLLElBQUksR0FBRyxVQUFPLEtBQUssRUFBRTtBQUFBLFFBQ2xDLENBQUM7QUFBQSxNQUNIO0FBQ0EsWUFBTSxLQUFLLEVBQUU7QUFDYixZQUFNLEtBQUssS0FBSyxRQUFRLEVBQUU7QUFDMUIsWUFBTSxLQUFLLEVBQUU7QUFBQSxJQUNmLENBQUM7QUFDRCxXQUFPLE1BQU0sS0FBSyxJQUFJO0FBQUEsRUFDeEI7QUFFQSxXQUFTLGFBQWEsVUFBVSxNQUFNLE1BQU07QUFDMUMsUUFBSSxDQUFDLE9BQU8sT0FBTyxPQUFPLE9BQU8sSUFBSSxvQkFBb0IsWUFBWTtBQUNuRSxjQUFRLEtBQUssNkNBQTZDO0FBQzFELGFBQU87QUFBQSxJQUNUO0FBQ0EsVUFBTSxPQUFPLElBQUksS0FBSyxDQUFDLElBQUksR0FBRyxFQUFFLEtBQUssQ0FBQztBQUN0QyxVQUFNLE1BQU0sSUFBSSxnQkFBZ0IsSUFBSTtBQUNwQyxVQUFNLFNBQVMsU0FBUyxjQUFjLEdBQUc7QUFDekMsV0FBTyxPQUFPO0FBQ2QsV0FBTyxXQUFXO0FBQ2xCLGFBQVMsS0FBSyxZQUFZLE1BQU07QUFDaEMsV0FBTyxNQUFNO0FBQ2IsYUFBUyxLQUFLLFlBQVksTUFBTTtBQUNoQyxXQUFPLFdBQVcsTUFBTSxJQUFJLGdCQUFnQixHQUFHLEdBQUcsQ0FBQztBQUNuRCxXQUFPO0FBQUEsRUFDVDtBQUVBLGlCQUFlLGdCQUFnQixNQUFNO0FBQ25DLFFBQUksQ0FBQyxLQUFNLFFBQU87QUFDbEIsUUFBSTtBQUNGLFVBQUksVUFBVSxhQUFhLFVBQVUsVUFBVSxXQUFXO0FBQ3hELGNBQU0sVUFBVSxVQUFVLFVBQVUsSUFBSTtBQUFBLE1BQzFDLE9BQU87QUFDTCxjQUFNLFdBQVcsU0FBUyxjQUFjLFVBQVU7QUFDbEQsaUJBQVMsUUFBUTtBQUNqQixpQkFBUyxhQUFhLFlBQVksVUFBVTtBQUM1QyxpQkFBUyxNQUFNLFdBQVc7QUFDMUIsaUJBQVMsTUFBTSxPQUFPO0FBQ3RCLGlCQUFTLEtBQUssWUFBWSxRQUFRO0FBQ2xDLGlCQUFTLE9BQU87QUFDaEIsaUJBQVMsWUFBWSxNQUFNO0FBQzNCLGlCQUFTLEtBQUssWUFBWSxRQUFRO0FBQUEsTUFDcEM7QUFDQSxhQUFPO0FBQUEsSUFDVCxTQUFTLEtBQUs7QUFDWixjQUFRLEtBQUssNEJBQTRCLEdBQUc7QUFDNUMsYUFBTztBQUFBLElBQ1Q7QUFBQSxFQUNGO0FBRU8sV0FBUyxlQUFlLEVBQUUsZUFBZSxTQUFTLEdBQUc7QUFDMUQsYUFBUyxvQkFBb0I7QUFDM0IsYUFBTyxjQUFjLFFBQVE7QUFBQSxJQUMvQjtBQUVBLG1CQUFlLG1CQUFtQixRQUFRO0FBQ3hDLFlBQU0sUUFBUSxrQkFBa0I7QUFDaEMsVUFBSSxDQUFDLE1BQU0sUUFBUTtBQUNqQixpQkFBUyxnQ0FBNkIsU0FBUztBQUMvQztBQUFBLE1BQ0Y7QUFDQSxVQUFJLFdBQVcsUUFBUTtBQUNyQixjQUFNLFVBQVU7QUFBQSxVQUNkLGFBQWEsT0FBTztBQUFBLFVBQ3BCLE9BQU8sTUFBTTtBQUFBLFVBQ2I7QUFBQSxRQUNGO0FBQ0EsWUFDRTtBQUFBLFVBQ0Usb0JBQW9CLE1BQU07QUFBQSxVQUMxQixLQUFLLFVBQVUsU0FBUyxNQUFNLENBQUM7QUFBQSxVQUMvQjtBQUFBLFFBQ0YsR0FDQTtBQUNBLG1CQUFTLGdDQUF1QixTQUFTO0FBQUEsUUFDM0MsT0FBTztBQUNMLG1CQUFTLDhDQUEyQyxRQUFRO0FBQUEsUUFDOUQ7QUFDQTtBQUFBLE1BQ0Y7QUFDQSxVQUFJLFdBQVcsWUFBWTtBQUN6QixZQUNFO0FBQUEsVUFDRSxvQkFBb0IsSUFBSTtBQUFBLFVBQ3hCLG9CQUFvQixLQUFLO0FBQUEsVUFDekI7QUFBQSxRQUNGLEdBQ0E7QUFDQSxtQkFBUyxvQ0FBMkIsU0FBUztBQUFBLFFBQy9DLE9BQU87QUFDTCxtQkFBUyw4Q0FBMkMsUUFBUTtBQUFBLFFBQzlEO0FBQUEsTUFDRjtBQUFBLElBQ0Y7QUFFQSxtQkFBZSw4QkFBOEI7QUFDM0MsWUFBTSxRQUFRLGtCQUFrQjtBQUNoQyxVQUFJLENBQUMsTUFBTSxRQUFRO0FBQ2pCLGlCQUFTLDhCQUEyQixTQUFTO0FBQzdDO0FBQUEsTUFDRjtBQUNBLFlBQU0sT0FBTyxvQkFBb0IsS0FBSztBQUN0QyxVQUFJLE1BQU0sZ0JBQWdCLElBQUksR0FBRztBQUMvQixpQkFBUyw2Q0FBMEMsU0FBUztBQUFBLE1BQzlELE9BQU87QUFDTCxpQkFBUyx5Q0FBeUMsUUFBUTtBQUFBLE1BQzVEO0FBQUEsSUFDRjtBQUVBLFdBQU87QUFBQSxNQUNMO0FBQUEsTUFDQTtBQUFBLElBQ0Y7QUFBQSxFQUNGOzs7QUNoSU8sV0FBUyxtQkFBbUIsRUFBRSxRQUFRLE1BQU0sSUFBSSxRQUFRLEdBQUc7QUFDaEUsUUFBSTtBQUNKLFFBQUk7QUFDSixRQUFJLG1CQUFtQjtBQUN2QixVQUFNLGNBQWM7QUFDcEIsUUFBSSxhQUFhO0FBQ2pCLFFBQUksV0FBVztBQUVmLGFBQVMsaUJBQWlCO0FBQ3hCLFVBQUksU0FBUztBQUNYLHNCQUFjLE9BQU87QUFDckIsa0JBQVU7QUFBQSxNQUNaO0FBQUEsSUFDRjtBQUVBLGFBQVMsa0JBQWtCLFdBQVc7QUFDcEMsVUFBSSxVQUFVO0FBQ1osZUFBTztBQUFBLE1BQ1Q7QUFDQSxZQUFNLFNBQVMsS0FBSyxNQUFNLEtBQUssT0FBTyxJQUFJLEdBQUc7QUFDN0MsWUFBTSxRQUFRLEtBQUssSUFBSSxhQUFhLFlBQVksTUFBTTtBQUN0RCxVQUFJLFlBQVk7QUFDZCxxQkFBYSxVQUFVO0FBQUEsTUFDekI7QUFDQSxtQkFBYSxPQUFPLFdBQVcsTUFBTTtBQUNuQyxxQkFBYTtBQUNiLDJCQUFtQixLQUFLO0FBQUEsVUFDdEI7QUFBQSxVQUNBLEtBQUssSUFBSSxLQUFLLG1CQUFtQixDQUFDO0FBQUEsUUFDcEM7QUFDQSxhQUFLLFdBQVc7QUFBQSxNQUNsQixHQUFHLEtBQUs7QUFDUixhQUFPO0FBQUEsSUFDVDtBQUVBLGFBQVMsU0FBUyxLQUFLO0FBQ3JCLFVBQUk7QUFDRixZQUFJLE1BQU0sR0FBRyxlQUFlLFVBQVUsTUFBTTtBQUMxQyxhQUFHLEtBQUssS0FBSyxVQUFVLEdBQUcsQ0FBQztBQUFBLFFBQzdCO0FBQUEsTUFDRixTQUFTLEtBQUs7QUFDWixnQkFBUSxLQUFLLGlDQUFpQyxHQUFHO0FBQUEsTUFDbkQ7QUFBQSxJQUNGO0FBRUEsbUJBQWUsYUFBYTtBQUMxQixVQUFJLFVBQVU7QUFDWjtBQUFBLE1BQ0Y7QUFFQSxVQUFJO0FBQ0YsV0FBRyxxQkFBcUIsaURBQXVDLE1BQU07QUFDckUsY0FBTSxTQUFTLE1BQU0sS0FBSyxZQUFZO0FBQ3RDLFlBQUksVUFBVTtBQUNaO0FBQUEsUUFDRjtBQUVBLGNBQU0sUUFBUSxJQUFJLElBQUksYUFBYSxPQUFPLE9BQU87QUFDakQsY0FBTSxXQUFXLE9BQU8sUUFBUSxhQUFhLFdBQVcsU0FBUztBQUNqRSxjQUFNLGFBQWEsSUFBSSxLQUFLLE1BQU07QUFFbEMsWUFBSSxJQUFJO0FBQ04sY0FBSTtBQUNGLGVBQUcsTUFBTTtBQUFBLFVBQ1gsU0FBUyxLQUFLO0FBQ1osb0JBQVEsS0FBSywyQ0FBMkMsR0FBRztBQUFBLFVBQzdEO0FBQ0EsZUFBSztBQUFBLFFBQ1A7QUFFQSxhQUFLLElBQUksVUFBVSxNQUFNLFNBQVMsQ0FBQztBQUNuQyxXQUFHLFlBQVksWUFBWTtBQUMzQixXQUFHLHFCQUFxQiw4QkFBeUIsTUFBTTtBQUV2RCxXQUFHLFNBQVMsTUFBTTtBQUNoQixjQUFJLFVBQVU7QUFDWjtBQUFBLFVBQ0Y7QUFDQSxjQUFJLFlBQVk7QUFDZCx5QkFBYSxVQUFVO0FBQ3ZCLHlCQUFhO0FBQUEsVUFDZjtBQUNBLDZCQUFtQjtBQUNuQixnQkFBTSxjQUFjLE9BQU87QUFDM0IsYUFBRyxZQUFZLFFBQVE7QUFDdkIsYUFBRztBQUFBLFlBQ0Qsa0JBQWUsR0FBRyxnQkFBZ0IsV0FBVyxDQUFDO0FBQUEsWUFDOUM7QUFBQSxVQUNGO0FBQ0EsYUFBRyxlQUFlLEVBQUUsYUFBYSxlQUFlLFlBQVksQ0FBQztBQUM3RCxhQUFHLFVBQVU7QUFDYix5QkFBZTtBQUNmLG9CQUFVLE9BQU8sWUFBWSxNQUFNO0FBQ2pDLHFCQUFTLEVBQUUsTUFBTSxlQUFlLElBQUksT0FBTyxFQUFFLENBQUM7QUFBQSxVQUNoRCxHQUFHLEdBQUs7QUFDUixhQUFHLGtCQUFrQix5Q0FBbUMsU0FBUztBQUNqRSxhQUFHLHFCQUFxQixHQUFJO0FBQUEsUUFDOUI7QUFFQSxXQUFHLFlBQVksQ0FBQyxRQUFRO0FBQ3RCLGdCQUFNLGFBQWEsT0FBTztBQUMxQixjQUFJO0FBQ0Ysa0JBQU0sS0FBSyxLQUFLLE1BQU0sSUFBSSxJQUFJO0FBQzlCLGVBQUcsZUFBZSxFQUFFLGVBQWUsV0FBVyxDQUFDO0FBQy9DLG9CQUFRLEVBQUU7QUFBQSxVQUNaLFNBQVMsS0FBSztBQUNaLG9CQUFRLE1BQU0scUJBQXFCLEtBQUssSUFBSSxJQUFJO0FBQUEsVUFDbEQ7QUFBQSxRQUNGO0FBRUEsV0FBRyxVQUFVLE1BQU07QUFDakIseUJBQWU7QUFDZixlQUFLO0FBQ0wsY0FBSSxVQUFVO0FBQ1o7QUFBQSxVQUNGO0FBQ0EsYUFBRyxZQUFZLFNBQVM7QUFDeEIsYUFBRyxlQUFlLEVBQUUsV0FBVyxPQUFVLENBQUM7QUFDMUMsZ0JBQU0sUUFBUSxrQkFBa0IsZ0JBQWdCO0FBQ2hELGdCQUFNLFVBQVUsS0FBSyxJQUFJLEdBQUcsS0FBSyxNQUFNLFFBQVEsR0FBSSxDQUFDO0FBQ3BELGFBQUc7QUFBQSxZQUNELDZDQUF1QyxPQUFPO0FBQUEsWUFDOUM7QUFBQSxVQUNGO0FBQ0EsYUFBRztBQUFBLFlBQ0Q7QUFBQSxZQUNBO0FBQUEsVUFDRjtBQUNBLGFBQUcscUJBQXFCLEdBQUk7QUFBQSxRQUM5QjtBQUVBLFdBQUcsVUFBVSxDQUFDLFFBQVE7QUFDcEIsa0JBQVEsTUFBTSxtQkFBbUIsR0FBRztBQUNwQyxjQUFJLFVBQVU7QUFDWjtBQUFBLFVBQ0Y7QUFDQSxhQUFHLFlBQVksU0FBUyxrQkFBa0I7QUFDMUMsYUFBRyxxQkFBcUIsb0NBQThCLFFBQVE7QUFDOUQsYUFBRyxrQkFBa0Isc0NBQW1DLFFBQVE7QUFDaEUsYUFBRyxxQkFBcUIsR0FBSTtBQUFBLFFBQzlCO0FBQUEsTUFDRixTQUFTLEtBQUs7QUFDWixnQkFBUSxNQUFNLEdBQUc7QUFDakIsWUFBSSxVQUFVO0FBQ1o7QUFBQSxRQUNGO0FBQ0EsY0FBTSxVQUFVLGVBQWUsUUFBUSxJQUFJLFVBQVUsT0FBTyxHQUFHO0FBQy9ELFdBQUcsWUFBWSxTQUFTLE9BQU87QUFDL0IsV0FBRyxxQkFBcUIsU0FBUyxRQUFRO0FBQ3pDLFdBQUc7QUFBQSxVQUNEO0FBQUEsVUFDQTtBQUFBLFFBQ0Y7QUFDQSxXQUFHLHFCQUFxQixHQUFJO0FBQzVCLDBCQUFrQixnQkFBZ0I7QUFBQSxNQUNwQztBQUFBLElBQ0Y7QUFFQSxhQUFTLFVBQVU7QUFDakIsaUJBQVc7QUFDWCxVQUFJLFlBQVk7QUFDZCxxQkFBYSxVQUFVO0FBQ3ZCLHFCQUFhO0FBQUEsTUFDZjtBQUNBLHFCQUFlO0FBQ2YsVUFBSSxJQUFJO0FBQ04sWUFBSTtBQUNGLGFBQUcsTUFBTTtBQUFBLFFBQ1gsU0FBUyxLQUFLO0FBQ1osa0JBQVEsS0FBSyx5Q0FBeUMsR0FBRztBQUFBLFFBQzNEO0FBQ0EsYUFBSztBQUFBLE1BQ1A7QUFBQSxJQUNGO0FBRUEsV0FBTztBQUFBLE1BQ0wsTUFBTTtBQUFBLE1BQ04sTUFBTTtBQUFBLE1BQ047QUFBQSxJQUNGO0FBQUEsRUFDRjs7O0FDdExPLFdBQVMsd0JBQXdCLEVBQUUsTUFBTSxHQUFHLEdBQUc7QUFDcEQsUUFBSSxRQUFRO0FBRVosYUFBUyxTQUFTLFFBQVE7QUFDeEIsVUFBSSxPQUFPO0FBQ1QscUJBQWEsS0FBSztBQUFBLE1BQ3BCO0FBQ0EsY0FBUSxPQUFPLFdBQVcsTUFBTSxpQkFBaUIsTUFBTSxHQUFHLEdBQUc7QUFBQSxJQUMvRDtBQUVBLG1CQUFlLGlCQUFpQixRQUFRO0FBQ3RDLFVBQUksQ0FBQyxVQUFVLE9BQU8sS0FBSyxFQUFFLFNBQVMsR0FBRztBQUN2QztBQUFBLE1BQ0Y7QUFDQSxVQUFJO0FBQ0YsY0FBTSxVQUFVLE1BQU0sS0FBSyxnQkFBZ0IsT0FBTyxLQUFLLENBQUM7QUFDeEQsWUFBSSxXQUFXLE1BQU0sUUFBUSxRQUFRLE9BQU8sR0FBRztBQUM3QyxhQUFHLHlCQUF5QixRQUFRLE9BQU87QUFBQSxRQUM3QztBQUFBLE1BQ0YsU0FBUyxLQUFLO0FBQ1osZ0JBQVEsTUFBTSwrQkFBK0IsR0FBRztBQUFBLE1BQ2xEO0FBQUEsSUFDRjtBQUVBLFdBQU87QUFBQSxNQUNMO0FBQUEsSUFDRjtBQUFBLEVBQ0Y7OztBQ2pCQSxXQUFTLGNBQWMsS0FBSztBQUMxQixVQUFNLE9BQU8sQ0FBQyxPQUFPLElBQUksZUFBZSxFQUFFO0FBQzFDLFdBQU87QUFBQSxNQUNMLFlBQVksS0FBSyxZQUFZO0FBQUEsTUFDN0IsVUFBVSxLQUFLLFVBQVU7QUFBQSxNQUN6QixRQUFRLEtBQUssUUFBUTtBQUFBLE1BQ3JCLE1BQU0sS0FBSyxNQUFNO0FBQUEsTUFDakIsVUFBVSxLQUFLLFdBQVc7QUFBQSxNQUMxQixjQUFjLEtBQUssZUFBZTtBQUFBLE1BQ2xDLFlBQVksS0FBSyxZQUFZO0FBQUEsTUFDN0IsWUFBWSxLQUFLLGFBQWE7QUFBQSxNQUM5QixjQUFjLEtBQUssZUFBZTtBQUFBLE1BQ2xDLGNBQWMsS0FBSyxlQUFlO0FBQUEsTUFDbEMsZ0JBQWdCLEtBQUssaUJBQWlCO0FBQUEsTUFDdEMsYUFBYSxLQUFLLGNBQWM7QUFBQSxNQUNoQyxnQkFBZ0IsS0FBSyxpQkFBaUI7QUFBQSxNQUN0QyxhQUFhLEtBQUssYUFBYTtBQUFBLE1BQy9CLGFBQWEsS0FBSyxtQkFBbUI7QUFBQSxNQUNyQyxhQUFhLEtBQUssY0FBYztBQUFBLE1BQ2hDLFlBQVksS0FBSyxrQkFBa0I7QUFBQSxNQUNuQyxZQUFZLEtBQUssYUFBYTtBQUFBLE1BQzlCLGdCQUFnQixLQUFLLGlCQUFpQjtBQUFBLE1BQ3RDLFlBQVksS0FBSyxhQUFhO0FBQUEsTUFDOUIsZUFBZSxLQUFLLGdCQUFnQjtBQUFBLE1BQ3BDLGlCQUFpQixLQUFLLG1CQUFtQjtBQUFBLE1BQ3pDLGFBQWEsS0FBSyxjQUFjO0FBQUEsTUFDaEMsYUFBYSxLQUFLLGNBQWM7QUFBQSxJQUNsQztBQUFBLEVBQ0Y7QUFFQSxXQUFTLFlBQVksS0FBSztBQUN4QixVQUFNLGlCQUFpQixJQUFJLGVBQWUsY0FBYztBQUN4RCxRQUFJLENBQUMsZ0JBQWdCO0FBQ25CLGFBQU8sQ0FBQztBQUFBLElBQ1Y7QUFDQSxVQUFNLFVBQVUsZUFBZSxlQUFlO0FBQzlDLG1CQUFlLE9BQU87QUFDdEIsUUFBSTtBQUNGLFlBQU0sU0FBUyxLQUFLLE1BQU0sT0FBTztBQUNqQyxVQUFJLE1BQU0sUUFBUSxNQUFNLEdBQUc7QUFDekIsZUFBTztBQUFBLE1BQ1Q7QUFDQSxVQUFJLFVBQVUsT0FBTyxPQUFPO0FBQzFCLGVBQU8sRUFBRSxPQUFPLE9BQU8sTUFBTTtBQUFBLE1BQy9CO0FBQUEsSUFDRixTQUFTLEtBQUs7QUFDWixjQUFRLE1BQU0sZ0NBQWdDLEdBQUc7QUFBQSxJQUNuRDtBQUNBLFdBQU8sQ0FBQztBQUFBLEVBQ1Y7QUFFQSxXQUFTLGVBQWUsVUFBVTtBQUNoQyxXQUFPLFFBQVEsU0FBUyxjQUFjLFNBQVMsWUFBWSxTQUFTLE1BQU07QUFBQSxFQUM1RTtBQUVBLE1BQU0sZ0JBQWdCO0FBQUEsSUFDcEIsTUFBTTtBQUFBLElBQ04sV0FBVztBQUFBLElBQ1gsU0FBUztBQUFBLEVBQ1g7QUFFTyxNQUFNLFVBQU4sTUFBYztBQUFBLElBQ25CLFlBQVksTUFBTSxVQUFVLFlBQVksT0FBTyxjQUFjLENBQUMsR0FBRztBQUMvRCxXQUFLLE1BQU07QUFDWCxXQUFLLFNBQVMsY0FBYyxTQUFTO0FBQ3JDLFdBQUssV0FBVyxjQUFjLEdBQUc7QUFDakMsVUFBSSxDQUFDLGVBQWUsS0FBSyxRQUFRLEdBQUc7QUFDbEM7QUFBQSxNQUNGO0FBQ0EsVUFBSSxPQUFPLFVBQVUsT0FBTyxPQUFPLE9BQU8sZUFBZSxZQUFZO0FBQ25FLGVBQU8sT0FBTyxXQUFXO0FBQUEsVUFDdkIsUUFBUTtBQUFBLFVBQ1IsS0FBSztBQUFBLFVBQ0wsV0FBVztBQUFBLFVBQ1gsUUFBUTtBQUFBLFFBQ1YsQ0FBQztBQUFBLE1BQ0g7QUFDQSxXQUFLLGdCQUFnQixvQkFBb0I7QUFDekMsV0FBSyxLQUFLLGFBQWE7QUFBQSxRQUNyQixVQUFVLEtBQUs7QUFBQSxRQUNmLGVBQWUsS0FBSztBQUFBLE1BQ3RCLENBQUM7QUFDRCxXQUFLLE9BQU8sa0JBQWtCLEtBQUssTUFBTTtBQUN6QyxXQUFLLE9BQU8sa0JBQWtCLEVBQUUsUUFBUSxLQUFLLFFBQVEsTUFBTSxLQUFLLEtBQUssQ0FBQztBQUN0RSxXQUFLLFdBQVcsZUFBZTtBQUFBLFFBQzdCLGVBQWUsS0FBSztBQUFBLFFBQ3BCLFVBQVUsQ0FBQyxTQUFTLFlBQ2xCLEtBQUssR0FBRyxtQkFBbUIsU0FBUyxPQUFPO0FBQUEsTUFDL0MsQ0FBQztBQUNELFdBQUssY0FBYyx3QkFBd0I7QUFBQSxRQUN6QyxNQUFNLEtBQUs7QUFBQSxRQUNYLElBQUksS0FBSztBQUFBLE1BQ1gsQ0FBQztBQUNELFdBQUssU0FBUyxtQkFBbUI7QUFBQSxRQUMvQixRQUFRLEtBQUs7QUFBQSxRQUNiLE1BQU0sS0FBSztBQUFBLFFBQ1gsSUFBSSxLQUFLO0FBQUEsUUFDVCxTQUFTLENBQUMsT0FBTyxLQUFLLGtCQUFrQixFQUFFO0FBQUEsTUFDNUMsQ0FBQztBQUVELFlBQU0saUJBQWlCLFlBQVksR0FBRztBQUN0QyxVQUFJLGtCQUFrQixlQUFlLE9BQU87QUFDMUMsYUFBSyxHQUFHLFVBQVUsZUFBZSxLQUFLO0FBQUEsTUFDeEMsV0FBVyxNQUFNLFFBQVEsY0FBYyxHQUFHO0FBQ3hDLGFBQUssR0FBRyxjQUFjLGNBQWM7QUFBQSxNQUN0QztBQUVBLFdBQUssbUJBQW1CO0FBQ3hCLFdBQUssR0FBRyxXQUFXO0FBQ25CLFdBQUssT0FBTyxLQUFLO0FBQUEsSUFDbkI7QUFBQSxJQUVBLHFCQUFxQjtBQUNuQixXQUFLLEdBQUcsR0FBRyxVQUFVLE9BQU8sRUFBRSxLQUFLLE1BQU07QUFDdkMsY0FBTSxTQUFTLFFBQVEsSUFBSSxLQUFLO0FBQ2hDLFlBQUksQ0FBQyxPQUFPO0FBQ1YsZUFBSyxHQUFHO0FBQUEsWUFDTjtBQUFBLFlBQ0E7QUFBQSxVQUNGO0FBQ0EsZUFBSyxHQUFHLHFCQUFxQixHQUFJO0FBQ2pDO0FBQUEsUUFDRjtBQUNBLGFBQUssR0FBRyxVQUFVO0FBQ2xCLGNBQU0sY0FBYyxPQUFPO0FBQzNCLGFBQUssR0FBRyxjQUFjLFFBQVEsT0FBTztBQUFBLFVBQ25DLFdBQVc7QUFBQSxVQUNYLFVBQVUsRUFBRSxXQUFXLEtBQUs7QUFBQSxRQUM5QixDQUFDO0FBQ0QsWUFBSSxLQUFLLFNBQVMsUUFBUTtBQUN4QixlQUFLLFNBQVMsT0FBTyxRQUFRO0FBQUEsUUFDL0I7QUFDQSxhQUFLLEdBQUcsb0JBQW9CO0FBQzVCLGFBQUssR0FBRyxlQUFlO0FBQ3ZCLGFBQUssR0FBRyxrQkFBa0IsMkJBQW1CLE1BQU07QUFDbkQsYUFBSyxHQUFHLHFCQUFxQixHQUFJO0FBQ2pDLGFBQUssR0FBRyxRQUFRLElBQUk7QUFDcEIsYUFBSyxHQUFHLHlCQUF5QixDQUFDLFFBQVEsYUFBYSxTQUFTLENBQUM7QUFFakUsWUFBSTtBQUNGLGdCQUFNLEtBQUssS0FBSyxTQUFTLEtBQUs7QUFDOUIsY0FBSSxLQUFLLFNBQVMsUUFBUTtBQUN4QixpQkFBSyxTQUFTLE9BQU8sTUFBTTtBQUFBLFVBQzdCO0FBQ0EsZUFBSyxHQUFHLFlBQVk7QUFBQSxRQUN0QixTQUFTLEtBQUs7QUFDWixlQUFLLEdBQUcsUUFBUSxLQUFLO0FBQ3JCLGdCQUFNLFVBQVUsZUFBZSxRQUFRLElBQUksVUFBVSxPQUFPLEdBQUc7QUFDL0QsZUFBSyxHQUFHLFVBQVUsT0FBTztBQUN6QixlQUFLLEdBQUcsY0FBYyxVQUFVLFNBQVM7QUFBQSxZQUN2QyxTQUFTO0FBQUEsWUFDVCxlQUFlO0FBQUEsWUFDZixVQUFVLEVBQUUsT0FBTyxTQUFTO0FBQUEsVUFDOUIsQ0FBQztBQUNELGVBQUssR0FBRztBQUFBLFlBQ047QUFBQSxZQUNBO0FBQUEsVUFDRjtBQUNBLGVBQUssR0FBRyxxQkFBcUIsR0FBSTtBQUFBLFFBQ25DO0FBQUEsTUFDRixDQUFDO0FBRUQsV0FBSyxHQUFHLEdBQUcsZ0JBQWdCLENBQUMsRUFBRSxPQUFPLE1BQU07QUFDekMsWUFBSSxDQUFDLE9BQVE7QUFDYixjQUFNLFNBQVMsY0FBYyxNQUFNLEtBQUs7QUFDeEMsWUFBSSxLQUFLLFNBQVMsUUFBUTtBQUN4QixlQUFLLFNBQVMsT0FBTyxRQUFRO0FBQUEsUUFDL0I7QUFDQSxhQUFLLEdBQUcsb0JBQW9CO0FBQzVCLGFBQUssR0FBRyxlQUFlO0FBQ3ZCLGFBQUssR0FBRyxrQkFBa0IsK0JBQXVCLE1BQU07QUFDdkQsYUFBSyxHQUFHLHFCQUFxQixHQUFJO0FBQ2pDLGFBQUssR0FBRyxLQUFLLFVBQVUsRUFBRSxNQUFNLE9BQU8sQ0FBQztBQUFBLE1BQ3pDLENBQUM7QUFFRCxXQUFLLEdBQUcsR0FBRyxpQkFBaUIsQ0FBQyxFQUFFLE1BQU0sTUFBTTtBQUN6QyxhQUFLLEdBQUcsc0JBQXNCLE9BQU8sRUFBRSxlQUFlLEtBQUssQ0FBQztBQUFBLE1BQzlELENBQUM7QUFFRCxXQUFLLEdBQUcsR0FBRyxnQkFBZ0IsTUFBTTtBQUMvQixhQUFLLEdBQUcsc0JBQXNCO0FBQUEsTUFDaEMsQ0FBQztBQUVELFdBQUssR0FBRyxHQUFHLFVBQVUsQ0FBQyxFQUFFLE9BQU8sTUFBTTtBQUNuQyxhQUFLLFNBQVMsbUJBQW1CLE1BQU07QUFBQSxNQUN6QyxDQUFDO0FBRUQsV0FBSyxHQUFHLEdBQUcsZUFBZSxNQUFNO0FBQzlCLGFBQUssU0FBUyw0QkFBNEI7QUFBQSxNQUM1QyxDQUFDO0FBRUQsV0FBSyxHQUFHLEdBQUcsZ0JBQWdCLENBQUMsRUFBRSxNQUFNLE1BQU07QUFDeEMsWUFBSSxDQUFDLFNBQVMsQ0FBQyxNQUFNLEtBQUssR0FBRztBQUMzQjtBQUFBLFFBQ0Y7QUFDQSxZQUFJLEtBQUssU0FBUyxRQUFRLEtBQUssU0FBUyxLQUFLLFVBQVU7QUFDckQ7QUFBQSxRQUNGO0FBQ0EsYUFBSyxZQUFZLFNBQVMsS0FBSztBQUFBLE1BQ2pDLENBQUM7QUFBQSxJQUNIO0FBQUEsSUFFQSxrQkFBa0IsSUFBSTtBQUNwQixZQUFNLE9BQU8sTUFBTSxHQUFHLE9BQU8sR0FBRyxPQUFPO0FBQ3ZDLFlBQU0sT0FBTyxNQUFNLEdBQUcsT0FBTyxHQUFHLE9BQU8sQ0FBQztBQUN4QyxjQUFRLE1BQU07QUFBQSxRQUNaLEtBQUssZ0JBQWdCO0FBQ25CLGNBQUksUUFBUSxLQUFLLFFBQVE7QUFDdkIsaUJBQUssR0FBRyxtQkFBbUIsbUJBQWdCLEtBQUssTUFBTSxFQUFFO0FBQ3hELGlCQUFLLEdBQUc7QUFBQSxjQUNOLG1CQUFnQixLQUFLLE1BQU07QUFBQSxjQUMzQjtBQUFBLFlBQ0Y7QUFBQSxVQUNGLE9BQU87QUFDTCxpQkFBSyxHQUFHLG1CQUFtQix5QkFBc0I7QUFDakQsaUJBQUssR0FBRyxxQkFBcUIsMkJBQXdCLFNBQVM7QUFBQSxVQUNoRTtBQUNBLGVBQUssR0FBRyxxQkFBcUIsR0FBSTtBQUNqQztBQUFBLFFBQ0Y7QUFBQSxRQUNBLEtBQUssb0JBQW9CO0FBQ3ZCLGNBQUksUUFBUSxNQUFNLFFBQVEsS0FBSyxLQUFLLEdBQUc7QUFDckMsaUJBQUssR0FBRyxjQUFjLEtBQUssT0FBTyxFQUFFLFNBQVMsS0FBSyxDQUFDO0FBQUEsVUFDckQ7QUFDQTtBQUFBLFFBQ0Y7QUFBQSxRQUNBLEtBQUssMkJBQTJCO0FBQzlCLGdCQUFNLFFBQ0osT0FBTyxLQUFLLFVBQVUsV0FBVyxLQUFLLFFBQVEsS0FBSyxRQUFRO0FBQzdELGVBQUssR0FBRyxhQUFhLEtBQUs7QUFDMUI7QUFBQSxRQUNGO0FBQUEsUUFDQSxLQUFLLDhCQUE4QjtBQUNqQyxjQUFJLFFBQVEsS0FBSyxRQUFRLENBQUMsS0FBSyxHQUFHLGdCQUFnQixHQUFHO0FBQ25ELGlCQUFLLEdBQUcsYUFBYSxLQUFLLElBQUk7QUFBQSxVQUNoQztBQUNBLGVBQUssR0FBRyxVQUFVLElBQUk7QUFDdEIsZUFBSyxHQUFHLFFBQVEsS0FBSztBQUNyQixjQUFJLFFBQVEsT0FBTyxLQUFLLGVBQWUsYUFBYTtBQUNsRCxpQkFBSyxHQUFHLGVBQWUsRUFBRSxXQUFXLE9BQU8sS0FBSyxVQUFVLEVBQUUsQ0FBQztBQUFBLFVBQy9EO0FBQ0EsY0FBSSxRQUFRLEtBQUssT0FBTyxTQUFTLEtBQUssT0FBTztBQUMzQyxpQkFBSyxHQUFHLGNBQWMsVUFBVSxLQUFLLE9BQU87QUFBQSxjQUMxQyxTQUFTO0FBQUEsY0FDVCxlQUFlO0FBQUEsY0FDZixVQUFVLEVBQUUsT0FBTyxLQUFLO0FBQUEsWUFDMUIsQ0FBQztBQUFBLFVBQ0g7QUFDQTtBQUFBLFFBQ0Y7QUFBQSxRQUNBLEtBQUssZ0JBQWdCO0FBQ25CLGNBQUksQ0FBQyxLQUFLLEdBQUcsWUFBWSxHQUFHO0FBQzFCLGlCQUFLLEdBQUcsWUFBWTtBQUFBLFVBQ3RCO0FBQ0EsY0FDRSxRQUNBLE9BQU8sS0FBSyxhQUFhLFlBQ3pCLENBQUMsS0FBSyxHQUFHLGdCQUFnQixHQUN6QjtBQUNBLGlCQUFLLEdBQUcsYUFBYSxLQUFLLFFBQVE7QUFBQSxVQUNwQztBQUNBLGVBQUssR0FBRyxVQUFVLElBQUk7QUFDdEIsZUFBSyxHQUFHLFFBQVEsS0FBSztBQUNyQjtBQUFBLFFBQ0Y7QUFBQSxRQUNBLEtBQUssc0NBQXNDO0FBQ3pDLGVBQUssR0FBRztBQUFBLFlBQ047QUFBQSxZQUNBLCtCQUF5QixRQUFRLEtBQUssVUFBVSxLQUFLLFVBQVUsRUFBRTtBQUFBLFlBQ2pFO0FBQUEsY0FDRSxTQUFTO0FBQUEsY0FDVCxlQUFlO0FBQUEsY0FDZixVQUFVLEVBQUUsT0FBTyxLQUFLO0FBQUEsWUFDMUI7QUFBQSxVQUNGO0FBQ0E7QUFBQSxRQUNGO0FBQUEsUUFDQSxLQUFLLG9DQUFvQztBQUN2QyxlQUFLLEdBQUc7QUFBQSxZQUNOO0FBQUEsWUFDQSxnQ0FBMEIsUUFBUSxLQUFLLFFBQVEsS0FBSyxRQUFRLFNBQVM7QUFBQSxZQUNyRTtBQUFBLGNBQ0UsU0FBUztBQUFBLGNBQ1QsZUFBZTtBQUFBLGNBQ2YsVUFBVSxFQUFFLE9BQU8sS0FBSztBQUFBLFlBQzFCO0FBQUEsVUFDRjtBQUNBO0FBQUEsUUFDRjtBQUFBLFFBQ0EsS0FBSyxrQ0FBa0M7QUFDckMsZUFBSyxHQUFHO0FBQUEsWUFDTjtBQUFBLFlBQ0E7QUFBQSxZQUNBO0FBQUEsY0FDRSxTQUFTO0FBQUEsY0FDVCxlQUFlO0FBQUEsY0FDZixVQUFVLEVBQUUsT0FBTyxLQUFLO0FBQUEsWUFDMUI7QUFBQSxVQUNGO0FBQ0E7QUFBQSxRQUNGO0FBQUEsUUFDQSxLQUFLLHFDQUFxQztBQUN4QyxlQUFLLEdBQUc7QUFBQSxZQUNOO0FBQUEsWUFDQSxrQkFBa0IsT0FBTyxRQUFRLEtBQUssUUFBUSxLQUFLLFFBQVEsQ0FBQyxDQUFDO0FBQUEsWUFDN0Q7QUFBQSxjQUNFLFNBQVM7QUFBQSxjQUNULGVBQWU7QUFBQSxjQUNmLFVBQVUsRUFBRSxPQUFPLEtBQUs7QUFBQSxZQUMxQjtBQUFBLFVBQ0Y7QUFDQTtBQUFBLFFBQ0Y7QUFBQSxRQUNBLEtBQUsscUJBQXFCO0FBQ3hCLGVBQUssR0FBRyxjQUFjLFVBQVUsVUFBVSxLQUFLLEdBQUcsV0FBVyxJQUFJLENBQUMsSUFBSTtBQUFBLFlBQ3BFLFNBQVM7QUFBQSxZQUNULGVBQWU7QUFBQSxZQUNmLFVBQVUsRUFBRSxPQUFPLEtBQUs7QUFBQSxVQUMxQixDQUFDO0FBQ0QsY0FBSSxRQUFRLE9BQU8sS0FBSyxZQUFZLGFBQWE7QUFDL0MsaUJBQUssR0FBRyxlQUFlLEVBQUUsV0FBVyxPQUFPLEtBQUssT0FBTyxFQUFFLENBQUM7QUFBQSxVQUM1RDtBQUNBO0FBQUEsUUFDRjtBQUFBLFFBQ0EsS0FBSyxrQkFBa0I7QUFDckIsZUFBSyxHQUFHO0FBQUEsWUFDTixNQUFNLFFBQVEsS0FBSyxPQUFPLElBQUksS0FBSyxVQUFVLENBQUM7QUFBQSxVQUNoRDtBQUNBO0FBQUEsUUFDRjtBQUFBLFFBQ0E7QUFDRSxjQUFJLFFBQVEsS0FBSyxXQUFXLEtBQUssR0FBRztBQUNsQztBQUFBLFVBQ0Y7QUFDQSxrQkFBUSxNQUFNLG1CQUFtQixFQUFFO0FBQUEsTUFDdkM7QUFBQSxJQUNGO0FBQUEsRUFDRjs7O0FDblZBLE1BQUksUUFBUSxVQUFVLE9BQU8sY0FBYyxDQUFDLENBQUM7IiwKICAibmFtZXMiOiBbInN0YXRlIl0KfQo=
