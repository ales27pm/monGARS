(() => {
  // webapp/static/js/src/config.js
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

  // webapp/static/js/src/utils/time.js
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

  // webapp/static/js/src/state/timelineStore.js
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
    function update(id, patch) {
      if (!map.has(id)) {
        return null;
      }
      const entry = map.get(id);
      const next = { ...entry, ...patch };
      if (patch.metadata) {
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
      if (next.row) {
        next.row.dataset.rawText = next.text || "";
        next.row.dataset.timestamp = next.timestamp || "";
        next.row.dataset.role = next.role || entry.role;
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

  // webapp/static/js/src/utils/emitter.js
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

  // webapp/static/js/src/utils/dom.js
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
    const container = document.createElement("div");
    container.innerHTML = html;
    return container.textContent || "";
  }
  function extractBubbleText(bubble) {
    const clone = bubble.cloneNode(true);
    clone.querySelectorAll(".copy-btn, .chat-meta").forEach((node) => node.remove());
    return clone.textContent.trim();
  }

  // webapp/static/js/src/services/markdown.js
  function renderMarkdown(text) {
    if (text == null) {
      return "";
    }
    const value = String(text);
    try {
      if (window.marked && typeof window.marked.parse === "function") {
        const rendered = window.marked.parse(value);
        if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
          return window.DOMPurify.sanitize(rendered, {
            ALLOW_UNKNOWN_PROTOCOLS: false,
            USE_PROFILES: { html: true }
          });
        }
        return rendered;
      }
    } catch (err) {
      console.warn("Markdown rendering failed", err);
    }
    const escaped = escapeHTML(value);
    return escaped.replace(/\n/g, "<br>");
  }

  // webapp/static/js/src/ui/chatUi.js
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

  // webapp/static/js/src/services/auth.js
  function createAuthService(config) {
    function persistToken(token) {
      if (!token) return;
      try {
        window.localStorage.setItem("jwt", token);
      } catch (err) {
        console.warn("Unable to persist JWT in localStorage", err);
      }
    }
    if (config.token) {
      persistToken(config.token);
    }
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
    return {
      getJwt,
      persistToken
    };
  }

  // webapp/static/js/src/services/http.js
  function createHttpService({ config, auth }) {
    async function authorisedFetch(path, options = {}) {
      const jwt = await auth.getJwt();
      const headers = {
        ...options.headers || {},
        Authorization: `Bearer ${jwt}`
      };
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

  // webapp/static/js/src/services/exporter.js
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

  // webapp/static/js/src/services/socket.js
  function createSocketClient({ config, http, ui, onEvent }) {
    let ws;
    let wsHBeat;
    let reconnectBackoff = 500;
    const BACKOFF_MAX = 8e3;
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
        ui.updateConnectionMeta("Obtention d\u2019un ticket de connexion\u2026", "info");
        const ticket = await http.fetchTicket();
        const wsUrl = new URL("/ws/chat/", config.baseUrl);
        wsUrl.protocol = config.baseUrl.protocol === "https:" ? "wss:" : "ws:";
        wsUrl.searchParams.set("t", ticket);
        ws = new WebSocket(wsUrl.toString());
        ui.setWsStatus("connecting");
        ui.updateConnectionMeta("Connexion au serveur\u2026", "info");
        ws.onopen = () => {
          ui.setWsStatus("online");
          const connectedAt = nowISO();
          ui.updateConnectionMeta(
            `Connect\xE9 le ${ui.formatTimestamp(connectedAt)}`,
            "success"
          );
          ui.setDiagnostics({ connectedAt, lastMessageAt: connectedAt });
          ui.hideError();
          wsHBeat = window.setInterval(() => {
            safeSend({ type: "client.ping", ts: nowISO() });
          }, 2e4);
          reconnectBackoff = 500;
          ui.setComposerStatus("Connect\xE9. Vous pouvez \xE9changer.", "success");
          ui.scheduleComposerIdle(4e3);
        };
        ws.onmessage = (evt) => {
          try {
            const ev = JSON.parse(evt.data);
            onEvent(ev);
          } catch (err) {
            console.error("Bad event payload", err, evt.data);
          }
        };
        ws.onclose = () => {
          ui.setWsStatus("offline");
          if (wsHBeat) {
            clearInterval(wsHBeat);
          }
          ui.setDiagnostics({ latencyMs: void 0 });
          const delay = reconnectBackoff + Math.floor(Math.random() * 250);
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
          reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
          window.setTimeout(openSocket, delay);
        };
        ws.onerror = (err) => {
          console.error("WebSocket error", err);
          ui.setWsStatus("error", "Erreur WebSocket");
          ui.updateConnectionMeta("Erreur WebSocket d\xE9tect\xE9e.", "danger");
          ui.setComposerStatus("Une erreur r\xE9seau est survenue.", "danger");
          ui.scheduleComposerIdle(6e3);
        };
      } catch (err) {
        console.error(err);
        const message = err instanceof Error ? err.message : String(err);
        ui.setWsStatus("error", message);
        ui.updateConnectionMeta(message, "danger");
        ui.setComposerStatus(
          "Connexion indisponible. Nouvel essai bient\xF4t.",
          "danger"
        );
        ui.scheduleComposerIdle(6e3);
        const delay = Math.min(BACKOFF_MAX, reconnectBackoff);
        reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
        window.setTimeout(openSocket, delay);
      }
    }
    function dispose() {
      if (wsHBeat) {
        clearInterval(wsHBeat);
      }
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
    return {
      open: openSocket,
      send: safeSend,
      dispose
    };
  }

  // webapp/static/js/src/services/suggestions.js
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

  // webapp/static/js/src/app.js
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

  // webapp/static/js/src/index.js
  new ChatApp(document, window.chatConfig || {});
})();
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsic3JjL2NvbmZpZy5qcyIsICJzcmMvdXRpbHMvdGltZS5qcyIsICJzcmMvc3RhdGUvdGltZWxpbmVTdG9yZS5qcyIsICJzcmMvdXRpbHMvZW1pdHRlci5qcyIsICJzcmMvdXRpbHMvZG9tLmpzIiwgInNyYy9zZXJ2aWNlcy9tYXJrZG93bi5qcyIsICJzcmMvdWkvY2hhdFVpLmpzIiwgInNyYy9zZXJ2aWNlcy9hdXRoLmpzIiwgInNyYy9zZXJ2aWNlcy9odHRwLmpzIiwgInNyYy9zZXJ2aWNlcy9leHBvcnRlci5qcyIsICJzcmMvc2VydmljZXMvc29ja2V0LmpzIiwgInNyYy9zZXJ2aWNlcy9zdWdnZXN0aW9ucy5qcyIsICJzcmMvYXBwLmpzIiwgInNyYy9pbmRleC5qcyJdLAogICJzb3VyY2VzQ29udGVudCI6IFsiZXhwb3J0IGZ1bmN0aW9uIHJlc29sdmVDb25maWcocmF3ID0ge30pIHtcbiAgY29uc3QgY29uZmlnID0geyAuLi5yYXcgfTtcbiAgY29uc3QgY2FuZGlkYXRlID0gY29uZmlnLmZhc3RhcGlVcmwgfHwgd2luZG93LmxvY2F0aW9uLm9yaWdpbjtcbiAgdHJ5IHtcbiAgICBjb25maWcuYmFzZVVybCA9IG5ldyBVUkwoY2FuZGlkYXRlKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS5lcnJvcihcIkludmFsaWQgRkFTVEFQSSBVUkxcIiwgZXJyLCBjYW5kaWRhdGUpO1xuICAgIGNvbmZpZy5iYXNlVXJsID0gbmV3IFVSTCh3aW5kb3cubG9jYXRpb24ub3JpZ2luKTtcbiAgfVxuICByZXR1cm4gY29uZmlnO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXBpVXJsKGNvbmZpZywgcGF0aCkge1xuICByZXR1cm4gbmV3IFVSTChwYXRoLCBjb25maWcuYmFzZVVybCkudG9TdHJpbmcoKTtcbn1cbiIsICJleHBvcnQgZnVuY3Rpb24gbm93SVNPKCkge1xuICByZXR1cm4gbmV3IERhdGUoKS50b0lTT1N0cmluZygpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZm9ybWF0VGltZXN0YW1wKHRzKSB7XG4gIGlmICghdHMpIHJldHVybiBcIlwiO1xuICB0cnkge1xuICAgIHJldHVybiBuZXcgRGF0ZSh0cykudG9Mb2NhbGVTdHJpbmcoXCJmci1DQVwiKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgcmV0dXJuIFN0cmluZyh0cyk7XG4gIH1cbn1cbiIsICJpbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBtYWtlTWVzc2FnZUlkKCkge1xuICByZXR1cm4gYG1zZy0ke0RhdGUubm93KCkudG9TdHJpbmcoMzYpfS0ke01hdGgucmFuZG9tKCkudG9TdHJpbmcoMzYpLnNsaWNlKDIsIDgpfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVUaW1lbGluZVN0b3JlKCkge1xuICBjb25zdCBvcmRlciA9IFtdO1xuICBjb25zdCBtYXAgPSBuZXcgTWFwKCk7XG5cbiAgZnVuY3Rpb24gcmVnaXN0ZXIoe1xuICAgIGlkLFxuICAgIHJvbGUsXG4gICAgdGV4dCA9IFwiXCIsXG4gICAgdGltZXN0YW1wID0gbm93SVNPKCksXG4gICAgcm93LFxuICAgIG1ldGFkYXRhID0ge30sXG4gIH0pIHtcbiAgICBjb25zdCBtZXNzYWdlSWQgPSBpZCB8fCBtYWtlTWVzc2FnZUlkKCk7XG4gICAgaWYgKCFtYXAuaGFzKG1lc3NhZ2VJZCkpIHtcbiAgICAgIG9yZGVyLnB1c2gobWVzc2FnZUlkKTtcbiAgICB9XG4gICAgbWFwLnNldChtZXNzYWdlSWQsIHtcbiAgICAgIGlkOiBtZXNzYWdlSWQsXG4gICAgICByb2xlLFxuICAgICAgdGV4dCxcbiAgICAgIHRpbWVzdGFtcCxcbiAgICAgIHJvdyxcbiAgICAgIG1ldGFkYXRhOiB7IC4uLm1ldGFkYXRhIH0sXG4gICAgfSk7XG4gICAgaWYgKHJvdykge1xuICAgICAgcm93LmRhdGFzZXQubWVzc2FnZUlkID0gbWVzc2FnZUlkO1xuICAgICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgICByb3cuZGF0YXNldC5yYXdUZXh0ID0gdGV4dDtcbiAgICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IHRpbWVzdGFtcDtcbiAgICB9XG4gICAgcmV0dXJuIG1lc3NhZ2VJZDtcbiAgfVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZShpZCwgcGF0Y2gpIHtcbiAgICBpZiAoIW1hcC5oYXMoaWQpKSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gICAgY29uc3QgZW50cnkgPSBtYXAuZ2V0KGlkKTtcbiAgICBjb25zdCBuZXh0ID0geyAuLi5lbnRyeSwgLi4ucGF0Y2ggfTtcbiAgICBpZiAocGF0Y2gubWV0YWRhdGEpIHtcbiAgICAgIGNvbnN0IG1lcmdlZCA9IHsgLi4uZW50cnkubWV0YWRhdGEgfTtcbiAgICAgIE9iamVjdC5lbnRyaWVzKHBhdGNoLm1ldGFkYXRhKS5mb3JFYWNoKChba2V5LCB2YWx1ZV0pID0+IHtcbiAgICAgICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQgfHwgdmFsdWUgPT09IG51bGwpIHtcbiAgICAgICAgICBkZWxldGUgbWVyZ2VkW2tleV07XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgbWVyZ2VkW2tleV0gPSB2YWx1ZTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBuZXh0Lm1ldGFkYXRhID0gbWVyZ2VkO1xuICAgIH1cbiAgICBtYXAuc2V0KGlkLCBuZXh0KTtcbiAgICBpZiAobmV4dC5yb3cpIHtcbiAgICAgIG5leHQucm93LmRhdGFzZXQucmF3VGV4dCA9IG5leHQudGV4dCB8fCBcIlwiO1xuICAgICAgbmV4dC5yb3cuZGF0YXNldC50aW1lc3RhbXAgPSBuZXh0LnRpbWVzdGFtcCB8fCBcIlwiO1xuICAgICAgbmV4dC5yb3cuZGF0YXNldC5yb2xlID0gbmV4dC5yb2xlIHx8IGVudHJ5LnJvbGU7XG4gICAgfVxuICAgIHJldHVybiBuZXh0O1xuICB9XG5cbiAgZnVuY3Rpb24gY29sbGVjdCgpIHtcbiAgICByZXR1cm4gb3JkZXJcbiAgICAgIC5tYXAoKGlkKSA9PiB7XG4gICAgICAgIGNvbnN0IGVudHJ5ID0gbWFwLmdldChpZCk7XG4gICAgICAgIGlmICghZW50cnkpIHtcbiAgICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIHJvbGU6IGVudHJ5LnJvbGUsXG4gICAgICAgICAgdGV4dDogZW50cnkudGV4dCxcbiAgICAgICAgICB0aW1lc3RhbXA6IGVudHJ5LnRpbWVzdGFtcCxcbiAgICAgICAgICAuLi4oZW50cnkubWV0YWRhdGEgJiZcbiAgICAgICAgICAgIE9iamVjdC5rZXlzKGVudHJ5Lm1ldGFkYXRhKS5sZW5ndGggPiAwICYmIHtcbiAgICAgICAgICAgICAgbWV0YWRhdGE6IHsgLi4uZW50cnkubWV0YWRhdGEgfSxcbiAgICAgICAgICAgIH0pLFxuICAgICAgICB9O1xuICAgICAgfSlcbiAgICAgIC5maWx0ZXIoQm9vbGVhbik7XG4gIH1cblxuICBmdW5jdGlvbiBjbGVhcigpIHtcbiAgICBvcmRlci5sZW5ndGggPSAwO1xuICAgIG1hcC5jbGVhcigpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICByZWdpc3RlcixcbiAgICB1cGRhdGUsXG4gICAgY29sbGVjdCxcbiAgICBjbGVhcixcbiAgICBvcmRlcixcbiAgICBtYXAsXG4gICAgbWFrZU1lc3NhZ2VJZCxcbiAgfTtcbn1cbiIsICJleHBvcnQgZnVuY3Rpb24gY3JlYXRlRW1pdHRlcigpIHtcbiAgY29uc3QgbGlzdGVuZXJzID0gbmV3IE1hcCgpO1xuXG4gIGZ1bmN0aW9uIG9uKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkge1xuICAgICAgbGlzdGVuZXJzLnNldChldmVudCwgbmV3IFNldCgpKTtcbiAgICB9XG4gICAgbGlzdGVuZXJzLmdldChldmVudCkuYWRkKGhhbmRsZXIpO1xuICAgIHJldHVybiAoKSA9PiBvZmYoZXZlbnQsIGhhbmRsZXIpO1xuICB9XG5cbiAgZnVuY3Rpb24gb2ZmKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkgcmV0dXJuO1xuICAgIGNvbnN0IGJ1Y2tldCA9IGxpc3RlbmVycy5nZXQoZXZlbnQpO1xuICAgIGJ1Y2tldC5kZWxldGUoaGFuZGxlcik7XG4gICAgaWYgKGJ1Y2tldC5zaXplID09PSAwKSB7XG4gICAgICBsaXN0ZW5lcnMuZGVsZXRlKGV2ZW50KTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBlbWl0KGV2ZW50LCBwYXlsb2FkKSB7XG4gICAgaWYgKCFsaXN0ZW5lcnMuaGFzKGV2ZW50KSkgcmV0dXJuO1xuICAgIGxpc3RlbmVycy5nZXQoZXZlbnQpLmZvckVhY2goKGhhbmRsZXIpID0+IHtcbiAgICAgIHRyeSB7XG4gICAgICAgIGhhbmRsZXIocGF5bG9hZCk7XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgY29uc29sZS5lcnJvcihcIkVtaXR0ZXIgaGFuZGxlciBlcnJvclwiLCBlcnIpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgcmV0dXJuIHsgb24sIG9mZiwgZW1pdCB9O1xufVxuIiwgImV4cG9ydCBmdW5jdGlvbiBlc2NhcGVIVE1MKHN0cikge1xuICByZXR1cm4gU3RyaW5nKHN0cikucmVwbGFjZShcbiAgICAvWyY8PlwiJ10vZyxcbiAgICAoY2gpID0+XG4gICAgICAoe1xuICAgICAgICBcIiZcIjogXCImYW1wO1wiLFxuICAgICAgICBcIjxcIjogXCImbHQ7XCIsXG4gICAgICAgIFwiPlwiOiBcIiZndDtcIixcbiAgICAgICAgJ1wiJzogXCImcXVvdDtcIixcbiAgICAgICAgXCInXCI6IFwiJiMzOTtcIixcbiAgICAgIH0pW2NoXSxcbiAgKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGh0bWxUb1RleHQoaHRtbCkge1xuICBjb25zdCBjb250YWluZXIgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiZGl2XCIpO1xuICBjb250YWluZXIuaW5uZXJIVE1MID0gaHRtbDtcbiAgcmV0dXJuIGNvbnRhaW5lci50ZXh0Q29udGVudCB8fCBcIlwiO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZXh0cmFjdEJ1YmJsZVRleHQoYnViYmxlKSB7XG4gIGNvbnN0IGNsb25lID0gYnViYmxlLmNsb25lTm9kZSh0cnVlKTtcbiAgY2xvbmVcbiAgICAucXVlcnlTZWxlY3RvckFsbChcIi5jb3B5LWJ0biwgLmNoYXQtbWV0YVwiKVxuICAgIC5mb3JFYWNoKChub2RlKSA9PiBub2RlLnJlbW92ZSgpKTtcbiAgcmV0dXJuIGNsb25lLnRleHRDb250ZW50LnRyaW0oKTtcbn1cbiIsICJpbXBvcnQgeyBlc2NhcGVIVE1MIH0gZnJvbSBcIi4uL3V0aWxzL2RvbS5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gcmVuZGVyTWFya2Rvd24odGV4dCkge1xuICBpZiAodGV4dCA9PSBudWxsKSB7XG4gICAgcmV0dXJuIFwiXCI7XG4gIH1cbiAgY29uc3QgdmFsdWUgPSBTdHJpbmcodGV4dCk7XG4gIHRyeSB7XG4gICAgaWYgKHdpbmRvdy5tYXJrZWQgJiYgdHlwZW9mIHdpbmRvdy5tYXJrZWQucGFyc2UgPT09IFwiZnVuY3Rpb25cIikge1xuICAgICAgY29uc3QgcmVuZGVyZWQgPSB3aW5kb3cubWFya2VkLnBhcnNlKHZhbHVlKTtcbiAgICAgIGlmICh3aW5kb3cuRE9NUHVyaWZ5ICYmIHR5cGVvZiB3aW5kb3cuRE9NUHVyaWZ5LnNhbml0aXplID09PSBcImZ1bmN0aW9uXCIpIHtcbiAgICAgICAgcmV0dXJuIHdpbmRvdy5ET01QdXJpZnkuc2FuaXRpemUocmVuZGVyZWQsIHtcbiAgICAgICAgICBBTExPV19VTktOT1dOX1BST1RPQ09MUzogZmFsc2UsXG4gICAgICAgICAgVVNFX1BST0ZJTEVTOiB7IGh0bWw6IHRydWUgfSxcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVuZGVyZWQ7XG4gICAgfVxuICB9IGNhdGNoIChlcnIpIHtcbiAgICBjb25zb2xlLndhcm4oXCJNYXJrZG93biByZW5kZXJpbmcgZmFpbGVkXCIsIGVycik7XG4gIH1cbiAgY29uc3QgZXNjYXBlZCA9IGVzY2FwZUhUTUwodmFsdWUpO1xuICByZXR1cm4gZXNjYXBlZC5yZXBsYWNlKC9cXG4vZywgXCI8YnI+XCIpO1xufVxuIiwgImltcG9ydCB7IGNyZWF0ZUVtaXR0ZXIgfSBmcm9tIFwiLi4vdXRpbHMvZW1pdHRlci5qc1wiO1xuaW1wb3J0IHsgaHRtbFRvVGV4dCwgZXh0cmFjdEJ1YmJsZVRleHQsIGVzY2FwZUhUTUwgfSBmcm9tIFwiLi4vdXRpbHMvZG9tLmpzXCI7XG5pbXBvcnQgeyByZW5kZXJNYXJrZG93biB9IGZyb20gXCIuLi9zZXJ2aWNlcy9tYXJrZG93bi5qc1wiO1xuaW1wb3J0IHsgZm9ybWF0VGltZXN0YW1wLCBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlQ2hhdFVpKHsgZWxlbWVudHMsIHRpbWVsaW5lU3RvcmUgfSkge1xuICBjb25zdCBlbWl0dGVyID0gY3JlYXRlRW1pdHRlcigpO1xuXG4gIGNvbnN0IHNlbmRJZGxlTWFya3VwID0gZWxlbWVudHMuc2VuZCA/IGVsZW1lbnRzLnNlbmQuaW5uZXJIVE1MIDogXCJcIjtcbiAgY29uc3Qgc2VuZElkbGVMYWJlbCA9XG4gICAgKGVsZW1lbnRzLnNlbmQgJiYgZWxlbWVudHMuc2VuZC5nZXRBdHRyaWJ1dGUoXCJkYXRhLWlkbGUtbGFiZWxcIikpIHx8XG4gICAgKGVsZW1lbnRzLnNlbmQgPyBlbGVtZW50cy5zZW5kLnRleHRDb250ZW50LnRyaW0oKSA6IFwiRW52b3llclwiKTtcbiAgY29uc3Qgc2VuZEJ1c3lNYXJrdXAgPVxuICAgICc8c3BhbiBjbGFzcz1cInNwaW5uZXItYm9yZGVyIHNwaW5uZXItYm9yZGVyLXNtIG1lLTFcIiByb2xlPVwic3RhdHVzXCIgYXJpYS1oaWRkZW49XCJ0cnVlXCI+PC9zcGFuPkVudm9pXHUyMDI2JztcbiAgY29uc3QgY29tcG9zZXJTdGF0dXNEZWZhdWx0ID1cbiAgICAoZWxlbWVudHMuY29tcG9zZXJTdGF0dXMgJiYgZWxlbWVudHMuY29tcG9zZXJTdGF0dXMudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiQXBwdXlleiBzdXIgQ3RybCtFbnRyXHUwMEU5ZSBwb3VyIGVudm95ZXIgcmFwaWRlbWVudC5cIjtcbiAgY29uc3QgZmlsdGVySGludERlZmF1bHQgPVxuICAgIChlbGVtZW50cy5maWx0ZXJIaW50ICYmIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQudHJpbSgpKSB8fFxuICAgIFwiVXRpbGlzZXogbGUgZmlsdHJlIHBvdXIgbGltaXRlciBsJ2hpc3RvcmlxdWUuIEFwcHV5ZXogc3VyIFx1MDBDOWNoYXAgcG91ciBlZmZhY2VyLlwiO1xuICBjb25zdCBwcm9tcHRNYXggPSBOdW1iZXIoZWxlbWVudHMucHJvbXB0Py5nZXRBdHRyaWJ1dGUoXCJtYXhsZW5ndGhcIikpIHx8IG51bGw7XG4gIGNvbnN0IHByZWZlcnNSZWR1Y2VkTW90aW9uID1cbiAgICB3aW5kb3cubWF0Y2hNZWRpYSAmJlxuICAgIHdpbmRvdy5tYXRjaE1lZGlhKFwiKHByZWZlcnMtcmVkdWNlZC1tb3Rpb246IHJlZHVjZSlcIikubWF0Y2hlcztcbiAgY29uc3QgU0NST0xMX1RIUkVTSE9MRCA9IDE0MDtcbiAgY29uc3QgUFJPTVBUX01BWF9IRUlHSFQgPSAzMjA7XG5cbiAgY29uc3QgZGlhZ25vc3RpY3MgPSB7XG4gICAgY29ubmVjdGVkQXQ6IG51bGwsXG4gICAgbGFzdE1lc3NhZ2VBdDogbnVsbCxcbiAgICBsYXRlbmN5TXM6IG51bGwsXG4gIH07XG5cbiAgY29uc3Qgc3RhdGUgPSB7XG4gICAgcmVzZXRTdGF0dXNUaW1lcjogbnVsbCxcbiAgICBoaWRlU2Nyb2xsVGltZXI6IG51bGwsXG4gICAgYWN0aXZlRmlsdGVyOiBcIlwiLFxuICAgIGhpc3RvcnlCb290c3RyYXBwZWQ6IGVsZW1lbnRzLnRyYW5zY3JpcHQuY2hpbGRFbGVtZW50Q291bnQgPiAwLFxuICAgIGJvb3RzdHJhcHBpbmc6IGZhbHNlLFxuICAgIHN0cmVhbVJvdzogbnVsbCxcbiAgICBzdHJlYW1CdWY6IFwiXCIsXG4gICAgc3RyZWFtTWVzc2FnZUlkOiBudWxsLFxuICB9O1xuXG4gIGNvbnN0IHN0YXR1c0xhYmVscyA9IHtcbiAgICBvZmZsaW5lOiBcIkhvcnMgbGlnbmVcIixcbiAgICBjb25uZWN0aW5nOiBcIkNvbm5leGlvblx1MjAyNlwiLFxuICAgIG9ubGluZTogXCJFbiBsaWduZVwiLFxuICAgIGVycm9yOiBcIkVycmV1clwiLFxuICB9O1xuXG4gIGZ1bmN0aW9uIG9uKGV2ZW50LCBoYW5kbGVyKSB7XG4gICAgcmV0dXJuIGVtaXR0ZXIub24oZXZlbnQsIGhhbmRsZXIpO1xuICB9XG5cbiAgZnVuY3Rpb24gZW1pdChldmVudCwgcGF5bG9hZCkge1xuICAgIGVtaXR0ZXIuZW1pdChldmVudCwgcGF5bG9hZCk7XG4gIH1cblxuICBmdW5jdGlvbiBzZXRCdXN5KGJ1c3kpIHtcbiAgICBlbGVtZW50cy50cmFuc2NyaXB0LnNldEF0dHJpYnV0ZShcImFyaWEtYnVzeVwiLCBidXN5ID8gXCJ0cnVlXCIgOiBcImZhbHNlXCIpO1xuICAgIGlmIChlbGVtZW50cy5zZW5kKSB7XG4gICAgICBlbGVtZW50cy5zZW5kLmRpc2FibGVkID0gQm9vbGVhbihidXN5KTtcbiAgICAgIGVsZW1lbnRzLnNlbmQuc2V0QXR0cmlidXRlKFwiYXJpYS1idXN5XCIsIGJ1c3kgPyBcInRydWVcIiA6IFwiZmFsc2VcIik7XG4gICAgICBpZiAoYnVzeSkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLmlubmVySFRNTCA9IHNlbmRCdXN5TWFya3VwO1xuICAgICAgfSBlbHNlIGlmIChzZW5kSWRsZU1hcmt1cCkge1xuICAgICAgICBlbGVtZW50cy5zZW5kLmlubmVySFRNTCA9IHNlbmRJZGxlTWFya3VwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZWxlbWVudHMuc2VuZC50ZXh0Q29udGVudCA9IHNlbmRJZGxlTGFiZWw7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gaGlkZUVycm9yKCkge1xuICAgIGlmICghZWxlbWVudHMuZXJyb3JBbGVydCkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLmVycm9yQWxlcnQuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICBpZiAoZWxlbWVudHMuZXJyb3JNZXNzYWdlKSB7XG4gICAgICBlbGVtZW50cy5lcnJvck1lc3NhZ2UudGV4dENvbnRlbnQgPSBcIlwiO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIHNob3dFcnJvcihtZXNzYWdlKSB7XG4gICAgaWYgKCFlbGVtZW50cy5lcnJvckFsZXJ0IHx8ICFlbGVtZW50cy5lcnJvck1lc3NhZ2UpIHJldHVybjtcbiAgICBlbGVtZW50cy5lcnJvck1lc3NhZ2UudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIGVsZW1lbnRzLmVycm9yQWxlcnQuY2xhc3NMaXN0LnJlbW92ZShcImQtbm9uZVwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldENvbXBvc2VyU3RhdHVzKG1lc3NhZ2UsIHRvbmUgPSBcIm11dGVkXCIpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzKSByZXR1cm47XG4gICAgY29uc3QgdG9uZXMgPSBbXCJtdXRlZFwiLCBcImluZm9cIiwgXCJzdWNjZXNzXCIsIFwiZGFuZ2VyXCIsIFwid2FybmluZ1wiXTtcbiAgICBlbGVtZW50cy5jb21wb3NlclN0YXR1cy50ZXh0Q29udGVudCA9IG1lc3NhZ2U7XG4gICAgdG9uZXMuZm9yRWFjaCgodCkgPT4gZWxlbWVudHMuY29tcG9zZXJTdGF0dXMuY2xhc3NMaXN0LnJlbW92ZShgdGV4dC0ke3R9YCkpO1xuICAgIGVsZW1lbnRzLmNvbXBvc2VyU3RhdHVzLmNsYXNzTGlzdC5hZGQoYHRleHQtJHt0b25lfWApO1xuICB9XG5cbiAgZnVuY3Rpb24gc2V0Q29tcG9zZXJTdGF0dXNJZGxlKCkge1xuICAgIHNldENvbXBvc2VyU3RhdHVzKGNvbXBvc2VyU3RhdHVzRGVmYXVsdCwgXCJtdXRlZFwiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlQ29tcG9zZXJJZGxlKGRlbGF5ID0gMzUwMCkge1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHN0YXRlLnJlc2V0U3RhdHVzVGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICB9LCBkZWxheSk7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVQcm9tcHRNZXRyaWNzKCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0Q291bnQgfHwgIWVsZW1lbnRzLnByb21wdCkgcmV0dXJuO1xuICAgIGNvbnN0IHZhbHVlID0gZWxlbWVudHMucHJvbXB0LnZhbHVlIHx8IFwiXCI7XG4gICAgaWYgKHByb21wdE1heCkge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9IC8gJHtwcm9tcHRNYXh9YDtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMucHJvbXB0Q291bnQudGV4dENvbnRlbnQgPSBgJHt2YWx1ZS5sZW5ndGh9YDtcbiAgICB9XG4gICAgZWxlbWVudHMucHJvbXB0Q291bnQuY2xhc3NMaXN0LnJlbW92ZShcInRleHQtd2FybmluZ1wiLCBcInRleHQtZGFuZ2VyXCIpO1xuICAgIGlmIChwcm9tcHRNYXgpIHtcbiAgICAgIGNvbnN0IHJlbWFpbmluZyA9IHByb21wdE1heCAtIHZhbHVlLmxlbmd0aDtcbiAgICAgIGlmIChyZW1haW5pbmcgPD0gNSkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC1kYW5nZXJcIik7XG4gICAgICB9IGVsc2UgaWYgKHJlbWFpbmluZyA8PSAyMCkge1xuICAgICAgICBlbGVtZW50cy5wcm9tcHRDb3VudC5jbGFzc0xpc3QuYWRkKFwidGV4dC13YXJuaW5nXCIpO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGF1dG9zaXplUHJvbXB0KCkge1xuICAgIGlmICghZWxlbWVudHMucHJvbXB0KSByZXR1cm47XG4gICAgZWxlbWVudHMucHJvbXB0LnN0eWxlLmhlaWdodCA9IFwiYXV0b1wiO1xuICAgIGNvbnN0IG5leHRIZWlnaHQgPSBNYXRoLm1pbihcbiAgICAgIGVsZW1lbnRzLnByb21wdC5zY3JvbGxIZWlnaHQsXG4gICAgICBQUk9NUFRfTUFYX0hFSUdIVCxcbiAgICApO1xuICAgIGVsZW1lbnRzLnByb21wdC5zdHlsZS5oZWlnaHQgPSBgJHtuZXh0SGVpZ2h0fXB4YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzQXRCb3R0b20oKSB7XG4gICAgaWYgKCFlbGVtZW50cy50cmFuc2NyaXB0KSByZXR1cm4gdHJ1ZTtcbiAgICBjb25zdCBkaXN0YW5jZSA9XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCAtXG4gICAgICAoZWxlbWVudHMudHJhbnNjcmlwdC5zY3JvbGxUb3AgKyBlbGVtZW50cy50cmFuc2NyaXB0LmNsaWVudEhlaWdodCk7XG4gICAgcmV0dXJuIGRpc3RhbmNlIDw9IFNDUk9MTF9USFJFU0hPTEQ7XG4gIH1cblxuICBmdW5jdGlvbiBzY3JvbGxUb0JvdHRvbShvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybjtcbiAgICBjb25zdCBzbW9vdGggPSBvcHRpb25zLnNtb290aCAhPT0gZmFsc2UgJiYgIXByZWZlcnNSZWR1Y2VkTW90aW9uO1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuc2Nyb2xsVG8oe1xuICAgICAgdG9wOiBlbGVtZW50cy50cmFuc2NyaXB0LnNjcm9sbEhlaWdodCxcbiAgICAgIGJlaGF2aW9yOiBzbW9vdGggPyBcInNtb290aFwiIDogXCJhdXRvXCIsXG4gICAgfSk7XG4gICAgaGlkZVNjcm9sbEJ1dHRvbigpO1xuICB9XG5cbiAgZnVuY3Rpb24gc2hvd1Njcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGlmIChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpIHtcbiAgICAgIGNsZWFyVGltZW91dChzdGF0ZS5oaWRlU2Nyb2xsVGltZXIpO1xuICAgICAgc3RhdGUuaGlkZVNjcm9sbFRpbWVyID0gbnVsbDtcbiAgICB9XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5yZW1vdmUoXCJkLW5vbmVcIik7XG4gICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJpcy12aXNpYmxlXCIpO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5zZXRBdHRyaWJ1dGUoXCJhcmlhLWhpZGRlblwiLCBcImZhbHNlXCIpO1xuICB9XG5cbiAgZnVuY3Rpb24gaGlkZVNjcm9sbEJ1dHRvbigpIHtcbiAgICBpZiAoIWVsZW1lbnRzLnNjcm9sbEJvdHRvbSkgcmV0dXJuO1xuICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5jbGFzc0xpc3QucmVtb3ZlKFwiaXMtdmlzaWJsZVwiKTtcbiAgICBlbGVtZW50cy5zY3JvbGxCb3R0b20uc2V0QXR0cmlidXRlKFwiYXJpYS1oaWRkZW5cIiwgXCJ0cnVlXCIpO1xuICAgIHN0YXRlLmhpZGVTY3JvbGxUaW1lciA9IHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGlmIChlbGVtZW50cy5zY3JvbGxCb3R0b20pIHtcbiAgICAgICAgZWxlbWVudHMuc2Nyb2xsQm90dG9tLmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgfSwgMjAwKTtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIGhhbmRsZUNvcHkoYnViYmxlKSB7XG4gICAgY29uc3QgdGV4dCA9IGV4dHJhY3RCdWJibGVUZXh0KGJ1YmJsZSk7XG4gICAgaWYgKCF0ZXh0KSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIHRyeSB7XG4gICAgICBpZiAobmF2aWdhdG9yLmNsaXBib2FyZCAmJiBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCkge1xuICAgICAgICBhd2FpdCBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCh0ZXh0KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IHRleHRhcmVhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRleHRhcmVhXCIpO1xuICAgICAgICB0ZXh0YXJlYS52YWx1ZSA9IHRleHQ7XG4gICAgICAgIHRleHRhcmVhLnNldEF0dHJpYnV0ZShcInJlYWRvbmx5XCIsIFwicmVhZG9ubHlcIik7XG4gICAgICAgIHRleHRhcmVhLnN0eWxlLnBvc2l0aW9uID0gXCJhYnNvbHV0ZVwiO1xuICAgICAgICB0ZXh0YXJlYS5zdHlsZS5sZWZ0ID0gXCItOTk5OXB4XCI7XG4gICAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQodGV4dGFyZWEpO1xuICAgICAgICB0ZXh0YXJlYS5zZWxlY3QoKTtcbiAgICAgICAgZG9jdW1lbnQuZXhlY0NvbW1hbmQoXCJjb3B5XCIpO1xuICAgICAgICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKHRleHRhcmVhKTtcbiAgICAgIH1cbiAgICAgIGFubm91bmNlQ29ubmVjdGlvbihcIkNvbnRlbnUgY29waVx1MDBFOSBkYW5zIGxlIHByZXNzZS1wYXBpZXJzLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJDb3B5IGZhaWxlZFwiLCBlcnIpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiSW1wb3NzaWJsZSBkZSBjb3BpZXIgbGUgbWVzc2FnZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZGVjb3JhdGVSb3cocm93LCByb2xlKSB7XG4gICAgY29uc3QgYnViYmxlID0gcm93LnF1ZXJ5U2VsZWN0b3IoXCIuY2hhdC1idWJibGVcIik7XG4gICAgaWYgKCFidWJibGUpIHJldHVybjtcbiAgICBpZiAocm9sZSA9PT0gXCJhc3Npc3RhbnRcIiB8fCByb2xlID09PSBcInVzZXJcIikge1xuICAgICAgYnViYmxlLmNsYXNzTGlzdC5hZGQoXCJoYXMtdG9vbHNcIik7XG4gICAgICBidWJibGUucXVlcnlTZWxlY3RvckFsbChcIi5jb3B5LWJ0blwiKS5mb3JFYWNoKChidG4pID0+IGJ0bi5yZW1vdmUoKSk7XG4gICAgICBjb25zdCBjb3B5QnRuID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImJ1dHRvblwiKTtcbiAgICAgIGNvcHlCdG4udHlwZSA9IFwiYnV0dG9uXCI7XG4gICAgICBjb3B5QnRuLmNsYXNzTmFtZSA9IFwiY29weS1idG5cIjtcbiAgICAgIGNvcHlCdG4uaW5uZXJIVE1MID1cbiAgICAgICAgJzxzcGFuIGFyaWEtaGlkZGVuPVwidHJ1ZVwiPlx1MjlDOTwvc3Bhbj48c3BhbiBjbGFzcz1cInZpc3VhbGx5LWhpZGRlblwiPkNvcGllciBsZSBtZXNzYWdlPC9zcGFuPic7XG4gICAgICBjb3B5QnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiBoYW5kbGVDb3B5KGJ1YmJsZSkpO1xuICAgICAgYnViYmxlLmFwcGVuZENoaWxkKGNvcHlCdG4pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGhpZ2hsaWdodFJvdyhyb3csIHJvbGUpIHtcbiAgICBpZiAoIXJvdyB8fCBzdGF0ZS5ib290c3RyYXBwaW5nIHx8IHJvbGUgPT09IFwic3lzdGVtXCIpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgcm93LmNsYXNzTGlzdC5hZGQoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LXJvdy1oaWdobGlnaHRcIik7XG4gICAgfSwgNjAwKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGxpbmUocm9sZSwgaHRtbCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qgc2hvdWxkU3RpY2sgPSBpc0F0Qm90dG9tKCk7XG4gICAgY29uc3Qgcm93ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImRpdlwiKTtcbiAgICByb3cuY2xhc3NOYW1lID0gYGNoYXQtcm93IGNoYXQtJHtyb2xlfWA7XG4gICAgcm93LmlubmVySFRNTCA9IGh0bWw7XG4gICAgcm93LmRhdGFzZXQucm9sZSA9IHJvbGU7XG4gICAgcm93LmRhdGFzZXQucmF3VGV4dCA9IG9wdGlvbnMucmF3VGV4dCB8fCBcIlwiO1xuICAgIHJvdy5kYXRhc2V0LnRpbWVzdGFtcCA9IG9wdGlvbnMudGltZXN0YW1wIHx8IFwiXCI7XG4gICAgZWxlbWVudHMudHJhbnNjcmlwdC5hcHBlbmRDaGlsZChyb3cpO1xuICAgIGRlY29yYXRlUm93KHJvdywgcm9sZSk7XG4gICAgaWYgKG9wdGlvbnMucmVnaXN0ZXIgIT09IGZhbHNlKSB7XG4gICAgICBjb25zdCB0cyA9IG9wdGlvbnMudGltZXN0YW1wIHx8IG5vd0lTTygpO1xuICAgICAgY29uc3QgdGV4dCA9XG4gICAgICAgIG9wdGlvbnMucmF3VGV4dCAmJiBvcHRpb25zLnJhd1RleHQubGVuZ3RoID4gMFxuICAgICAgICAgID8gb3B0aW9ucy5yYXdUZXh0XG4gICAgICAgICAgOiBodG1sVG9UZXh0KGh0bWwpO1xuICAgICAgY29uc3QgaWQgPSB0aW1lbGluZVN0b3JlLnJlZ2lzdGVyKHtcbiAgICAgICAgaWQ6IG9wdGlvbnMubWVzc2FnZUlkLFxuICAgICAgICByb2xlLFxuICAgICAgICB0ZXh0LFxuICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICByb3csXG4gICAgICAgIG1ldGFkYXRhOiBvcHRpb25zLm1ldGFkYXRhIHx8IHt9LFxuICAgICAgfSk7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBpZDtcbiAgICB9IGVsc2UgaWYgKG9wdGlvbnMubWVzc2FnZUlkKSB7XG4gICAgICByb3cuZGF0YXNldC5tZXNzYWdlSWQgPSBvcHRpb25zLm1lc3NhZ2VJZDtcbiAgICB9IGVsc2UgaWYgKCFyb3cuZGF0YXNldC5tZXNzYWdlSWQpIHtcbiAgICAgIHJvdy5kYXRhc2V0Lm1lc3NhZ2VJZCA9IHRpbWVsaW5lU3RvcmUubWFrZU1lc3NhZ2VJZCgpO1xuICAgIH1cbiAgICBpZiAoc2hvdWxkU3RpY2spIHtcbiAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiAhc3RhdGUuYm9vdHN0cmFwcGluZyB9KTtcbiAgICB9IGVsc2Uge1xuICAgICAgc2hvd1Njcm9sbEJ1dHRvbigpO1xuICAgIH1cbiAgICBoaWdobGlnaHRSb3cocm93LCByb2xlKTtcbiAgICBpZiAoc3RhdGUuYWN0aXZlRmlsdGVyKSB7XG4gICAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIoc3RhdGUuYWN0aXZlRmlsdGVyLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfVxuICAgIHJldHVybiByb3c7XG4gIH1cblxuICBmdW5jdGlvbiBidWlsZEJ1YmJsZSh7XG4gICAgdGV4dCxcbiAgICB0aW1lc3RhbXAsXG4gICAgdmFyaWFudCxcbiAgICBtZXRhU3VmZml4LFxuICAgIGFsbG93TWFya2Rvd24gPSB0cnVlLFxuICB9KSB7XG4gICAgY29uc3QgY2xhc3NlcyA9IFtcImNoYXQtYnViYmxlXCJdO1xuICAgIGlmICh2YXJpYW50KSB7XG4gICAgICBjbGFzc2VzLnB1c2goYGNoYXQtYnViYmxlLSR7dmFyaWFudH1gKTtcbiAgICB9XG4gICAgY29uc3QgY29udGVudCA9IGFsbG93TWFya2Rvd25cbiAgICAgID8gcmVuZGVyTWFya2Rvd24odGV4dClcbiAgICAgIDogZXNjYXBlSFRNTChTdHJpbmcodGV4dCkpO1xuICAgIGNvbnN0IG1ldGFCaXRzID0gW107XG4gICAgaWYgKHRpbWVzdGFtcCkge1xuICAgICAgbWV0YUJpdHMucHVzaChmb3JtYXRUaW1lc3RhbXAodGltZXN0YW1wKSk7XG4gICAgfVxuICAgIGlmIChtZXRhU3VmZml4KSB7XG4gICAgICBtZXRhQml0cy5wdXNoKG1ldGFTdWZmaXgpO1xuICAgIH1cbiAgICBjb25zdCBtZXRhSHRtbCA9XG4gICAgICBtZXRhQml0cy5sZW5ndGggPiAwXG4gICAgICAgID8gYDxkaXYgY2xhc3M9XCJjaGF0LW1ldGFcIj4ke2VzY2FwZUhUTUwobWV0YUJpdHMuam9pbihcIiBcdTIwMjIgXCIpKX08L2Rpdj5gXG4gICAgICAgIDogXCJcIjtcbiAgICByZXR1cm4gYDxkaXYgY2xhc3M9XCIke2NsYXNzZXMuam9pbihcIiBcIil9XCI+JHtjb250ZW50fSR7bWV0YUh0bWx9PC9kaXY+YDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGVuZE1lc3NhZ2Uocm9sZSwgdGV4dCwgb3B0aW9ucyA9IHt9KSB7XG4gICAgY29uc3Qge1xuICAgICAgdGltZXN0YW1wLFxuICAgICAgdmFyaWFudCxcbiAgICAgIG1ldGFTdWZmaXgsXG4gICAgICBhbGxvd01hcmtkb3duID0gdHJ1ZSxcbiAgICAgIG1lc3NhZ2VJZCxcbiAgICAgIHJlZ2lzdGVyID0gdHJ1ZSxcbiAgICAgIG1ldGFkYXRhLFxuICAgIH0gPSBvcHRpb25zO1xuICAgIGNvbnN0IGJ1YmJsZSA9IGJ1aWxkQnViYmxlKHtcbiAgICAgIHRleHQsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICB2YXJpYW50LFxuICAgICAgbWV0YVN1ZmZpeCxcbiAgICAgIGFsbG93TWFya2Rvd24sXG4gICAgfSk7XG4gICAgY29uc3Qgcm93ID0gbGluZShyb2xlLCBidWJibGUsIHtcbiAgICAgIHJhd1RleHQ6IHRleHQsXG4gICAgICB0aW1lc3RhbXAsXG4gICAgICBtZXNzYWdlSWQsXG4gICAgICByZWdpc3RlcixcbiAgICAgIG1ldGFkYXRhLFxuICAgIH0pO1xuICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogdGltZXN0YW1wIHx8IG5vd0lTTygpIH0pO1xuICAgIHJldHVybiByb3c7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVEaWFnbm9zdGljRmllbGQoZWwsIHZhbHVlKSB7XG4gICAgaWYgKCFlbCkgcmV0dXJuO1xuICAgIGVsLnRleHRDb250ZW50ID0gdmFsdWUgfHwgXCJcdTIwMTRcIjtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldERpYWdub3N0aWNzKHBhdGNoKSB7XG4gICAgT2JqZWN0LmFzc2lnbihkaWFnbm9zdGljcywgcGF0Y2gpO1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocGF0Y2gsIFwiY29ubmVjdGVkQXRcIikpIHtcbiAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChcbiAgICAgICAgZWxlbWVudHMuZGlhZ0Nvbm5lY3RlZCxcbiAgICAgICAgZGlhZ25vc3RpY3MuY29ubmVjdGVkQXRcbiAgICAgICAgICA/IGZvcm1hdFRpbWVzdGFtcChkaWFnbm9zdGljcy5jb25uZWN0ZWRBdClcbiAgICAgICAgICA6IFwiXHUyMDE0XCIsXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHBhdGNoLCBcImxhc3RNZXNzYWdlQXRcIikpIHtcbiAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChcbiAgICAgICAgZWxlbWVudHMuZGlhZ0xhc3RNZXNzYWdlLFxuICAgICAgICBkaWFnbm9zdGljcy5sYXN0TWVzc2FnZUF0XG4gICAgICAgICAgPyBmb3JtYXRUaW1lc3RhbXAoZGlhZ25vc3RpY3MubGFzdE1lc3NhZ2VBdClcbiAgICAgICAgICA6IFwiXHUyMDE0XCIsXG4gICAgICApO1xuICAgIH1cbiAgICBpZiAoT2JqZWN0LnByb3RvdHlwZS5oYXNPd25Qcm9wZXJ0eS5jYWxsKHBhdGNoLCBcImxhdGVuY3lNc1wiKSkge1xuICAgICAgaWYgKHR5cGVvZiBkaWFnbm9zdGljcy5sYXRlbmN5TXMgPT09IFwibnVtYmVyXCIpIHtcbiAgICAgICAgdXBkYXRlRGlhZ25vc3RpY0ZpZWxkKFxuICAgICAgICAgIGVsZW1lbnRzLmRpYWdMYXRlbmN5LFxuICAgICAgICAgIGAke01hdGgubWF4KDAsIE1hdGgucm91bmQoZGlhZ25vc3RpY3MubGF0ZW5jeU1zKSl9IG1zYCxcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHVwZGF0ZURpYWdub3N0aWNGaWVsZChlbGVtZW50cy5kaWFnTGF0ZW5jeSwgXCJcdTIwMTRcIik7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gdXBkYXRlTmV0d29ya1N0YXR1cygpIHtcbiAgICBpZiAoIWVsZW1lbnRzLmRpYWdOZXR3b3JrKSByZXR1cm47XG4gICAgY29uc3Qgb25saW5lID0gbmF2aWdhdG9yLm9uTGluZTtcbiAgICBlbGVtZW50cy5kaWFnTmV0d29yay50ZXh0Q29udGVudCA9IG9ubGluZSA/IFwiRW4gbGlnbmVcIiA6IFwiSG9ycyBsaWduZVwiO1xuICAgIGVsZW1lbnRzLmRpYWdOZXR3b3JrLmNsYXNzTGlzdC50b2dnbGUoXCJ0ZXh0LWRhbmdlclwiLCAhb25saW5lKTtcbiAgICBlbGVtZW50cy5kaWFnTmV0d29yay5jbGFzc0xpc3QudG9nZ2xlKFwidGV4dC1zdWNjZXNzXCIsIG9ubGluZSk7XG4gIH1cblxuICBmdW5jdGlvbiBhbm5vdW5jZUNvbm5lY3Rpb24obWVzc2FnZSwgdmFyaWFudCA9IFwiaW5mb1wiKSB7XG4gICAgaWYgKCFlbGVtZW50cy5jb25uZWN0aW9uKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGNsYXNzTGlzdCA9IGVsZW1lbnRzLmNvbm5lY3Rpb24uY2xhc3NMaXN0O1xuICAgIEFycmF5LmZyb20oY2xhc3NMaXN0KVxuICAgICAgLmZpbHRlcigoY2xzKSA9PiBjbHMuc3RhcnRzV2l0aChcImFsZXJ0LVwiKSAmJiBjbHMgIT09IFwiYWxlcnRcIilcbiAgICAgIC5mb3JFYWNoKChjbHMpID0+IGNsYXNzTGlzdC5yZW1vdmUoY2xzKSk7XG4gICAgY2xhc3NMaXN0LmFkZChcImFsZXJ0XCIpO1xuICAgIGNsYXNzTGlzdC5hZGQoYGFsZXJ0LSR7dmFyaWFudH1gKTtcbiAgICBlbGVtZW50cy5jb25uZWN0aW9uLnRleHRDb250ZW50ID0gbWVzc2FnZTtcbiAgICBjbGFzc0xpc3QucmVtb3ZlKFwidmlzdWFsbHktaGlkZGVuXCIpO1xuICAgIHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGNsYXNzTGlzdC5hZGQoXCJ2aXN1YWxseS1oaWRkZW5cIik7XG4gICAgfSwgNDAwMCk7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVDb25uZWN0aW9uTWV0YShtZXNzYWdlLCB0b25lID0gXCJtdXRlZFwiKSB7XG4gICAgaWYgKCFlbGVtZW50cy5jb25uZWN0aW9uTWV0YSkgcmV0dXJuO1xuICAgIGNvbnN0IHRvbmVzID0gW1wibXV0ZWRcIiwgXCJpbmZvXCIsIFwic3VjY2Vzc1wiLCBcImRhbmdlclwiLCBcIndhcm5pbmdcIl07XG4gICAgZWxlbWVudHMuY29ubmVjdGlvbk1ldGEudGV4dENvbnRlbnQgPSBtZXNzYWdlO1xuICAgIHRvbmVzLmZvckVhY2goKHQpID0+IGVsZW1lbnRzLmNvbm5lY3Rpb25NZXRhLmNsYXNzTGlzdC5yZW1vdmUoYHRleHQtJHt0fWApKTtcbiAgICBlbGVtZW50cy5jb25uZWN0aW9uTWV0YS5jbGFzc0xpc3QuYWRkKGB0ZXh0LSR7dG9uZX1gKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHNldFdzU3RhdHVzKHN0YXRlLCB0aXRsZSkge1xuICAgIGlmICghZWxlbWVudHMud3NTdGF0dXMpIHJldHVybjtcbiAgICBjb25zdCBsYWJlbCA9IHN0YXR1c0xhYmVsc1tzdGF0ZV0gfHwgc3RhdGU7XG4gICAgZWxlbWVudHMud3NTdGF0dXMudGV4dENvbnRlbnQgPSBsYWJlbDtcbiAgICBlbGVtZW50cy53c1N0YXR1cy5jbGFzc05hbWUgPSBgYmFkZ2Ugd3MtYmFkZ2UgJHtzdGF0ZX1gO1xuICAgIGlmICh0aXRsZSkge1xuICAgICAgZWxlbWVudHMud3NTdGF0dXMudGl0bGUgPSB0aXRsZTtcbiAgICB9IGVsc2Uge1xuICAgICAgZWxlbWVudHMud3NTdGF0dXMucmVtb3ZlQXR0cmlidXRlKFwidGl0bGVcIik7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gbm9ybWFsaXplU3RyaW5nKHN0cikge1xuICAgIGNvbnN0IHZhbHVlID0gU3RyaW5nKHN0ciB8fCBcIlwiKTtcbiAgICB0cnkge1xuICAgICAgcmV0dXJuIHZhbHVlXG4gICAgICAgIC5ub3JtYWxpemUoXCJORkRcIilcbiAgICAgICAgLnJlcGxhY2UoL1tcXHUwMzAwLVxcdTAzNmZdL2csIFwiXCIpXG4gICAgICAgIC50b0xvd2VyQ2FzZSgpO1xuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgcmV0dXJuIHZhbHVlLnRvTG93ZXJDYXNlKCk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gYXBwbHlUcmFuc2NyaXB0RmlsdGVyKHF1ZXJ5LCBvcHRpb25zID0ge30pIHtcbiAgICBpZiAoIWVsZW1lbnRzLnRyYW5zY3JpcHQpIHJldHVybiAwO1xuICAgIGNvbnN0IHsgcHJlc2VydmVJbnB1dCA9IGZhbHNlIH0gPSBvcHRpb25zO1xuICAgIGNvbnN0IHJhd1F1ZXJ5ID0gdHlwZW9mIHF1ZXJ5ID09PSBcInN0cmluZ1wiID8gcXVlcnkgOiBcIlwiO1xuICAgIGlmICghcHJlc2VydmVJbnB1dCAmJiBlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQudmFsdWUgPSByYXdRdWVyeTtcbiAgICB9XG4gICAgY29uc3QgdHJpbW1lZCA9IHJhd1F1ZXJ5LnRyaW0oKTtcbiAgICBzdGF0ZS5hY3RpdmVGaWx0ZXIgPSB0cmltbWVkO1xuICAgIGNvbnN0IG5vcm1hbGl6ZWQgPSBub3JtYWxpemVTdHJpbmcodHJpbW1lZCk7XG4gICAgbGV0IG1hdGNoZXMgPSAwO1xuICAgIGNvbnN0IHJvd3MgPSBBcnJheS5mcm9tKGVsZW1lbnRzLnRyYW5zY3JpcHQucXVlcnlTZWxlY3RvckFsbChcIi5jaGF0LXJvd1wiKSk7XG4gICAgcm93cy5mb3JFYWNoKChyb3cpID0+IHtcbiAgICAgIHJvdy5jbGFzc0xpc3QucmVtb3ZlKFwiY2hhdC1oaWRkZW5cIiwgXCJjaGF0LWZpbHRlci1tYXRjaFwiKTtcbiAgICAgIGlmICghbm9ybWFsaXplZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBjb25zdCByYXcgPSByb3cuZGF0YXNldC5yYXdUZXh0IHx8IFwiXCI7XG4gICAgICBjb25zdCBub3JtYWxpemVkUm93ID0gbm9ybWFsaXplU3RyaW5nKHJhdyk7XG4gICAgICBpZiAobm9ybWFsaXplZFJvdy5pbmNsdWRlcyhub3JtYWxpemVkKSkge1xuICAgICAgICByb3cuY2xhc3NMaXN0LmFkZChcImNoYXQtZmlsdGVyLW1hdGNoXCIpO1xuICAgICAgICBtYXRjaGVzICs9IDE7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByb3cuY2xhc3NMaXN0LmFkZChcImNoYXQtaGlkZGVuXCIpO1xuICAgICAgfVxuICAgIH0pO1xuICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuY2xhc3NMaXN0LnRvZ2dsZShcImZpbHRlcmVkXCIsIEJvb2xlYW4odHJpbW1lZCkpO1xuICAgIGlmIChlbGVtZW50cy5maWx0ZXJFbXB0eSkge1xuICAgICAgaWYgKHRyaW1tZWQgJiYgbWF0Y2hlcyA9PT0gMCkge1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5jbGFzc0xpc3QucmVtb3ZlKFwiZC1ub25lXCIpO1xuICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5zZXRBdHRyaWJ1dGUoXG4gICAgICAgICAgXCJhcmlhLWxpdmVcIixcbiAgICAgICAgICBlbGVtZW50cy5maWx0ZXJFbXB0eS5nZXRBdHRyaWJ1dGUoXCJhcmlhLWxpdmVcIikgfHwgXCJwb2xpdGVcIixcbiAgICAgICAgKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckVtcHR5LmNsYXNzTGlzdC5hZGQoXCJkLW5vbmVcIik7XG4gICAgICB9XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJIaW50KSB7XG4gICAgICBpZiAodHJpbW1lZCkge1xuICAgICAgICBsZXQgc3VtbWFyeSA9IFwiQXVjdW4gbWVzc2FnZSBuZSBjb3JyZXNwb25kLlwiO1xuICAgICAgICBpZiAobWF0Y2hlcyA9PT0gMSkge1xuICAgICAgICAgIHN1bW1hcnkgPSBcIjEgbWVzc2FnZSBjb3JyZXNwb25kLlwiO1xuICAgICAgICB9IGVsc2UgaWYgKG1hdGNoZXMgPiAxKSB7XG4gICAgICAgICAgc3VtbWFyeSA9IGAke21hdGNoZXN9IG1lc3NhZ2VzIGNvcnJlc3BvbmRlbnQuYDtcbiAgICAgICAgfVxuICAgICAgICBlbGVtZW50cy5maWx0ZXJIaW50LnRleHRDb250ZW50ID0gc3VtbWFyeTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQgPSBmaWx0ZXJIaW50RGVmYXVsdDtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG1hdGNoZXM7XG4gIH1cblxuICBmdW5jdGlvbiByZWFwcGx5VHJhbnNjcmlwdEZpbHRlcigpIHtcbiAgICBpZiAoc3RhdGUuYWN0aXZlRmlsdGVyKSB7XG4gICAgICBhcHBseVRyYW5zY3JpcHRGaWx0ZXIoc3RhdGUuYWN0aXZlRmlsdGVyLCB7IHByZXNlcnZlSW5wdXQ6IHRydWUgfSk7XG4gICAgfSBlbHNlIGlmIChlbGVtZW50cy50cmFuc2NyaXB0KSB7XG4gICAgICBlbGVtZW50cy50cmFuc2NyaXB0LmNsYXNzTGlzdC5yZW1vdmUoXCJmaWx0ZXJlZFwiKTtcbiAgICAgIGNvbnN0IHJvd3MgPSBBcnJheS5mcm9tKFxuICAgICAgICBlbGVtZW50cy50cmFuc2NyaXB0LnF1ZXJ5U2VsZWN0b3JBbGwoXCIuY2hhdC1yb3dcIiksXG4gICAgICApO1xuICAgICAgcm93cy5mb3JFYWNoKChyb3cpID0+IHtcbiAgICAgICAgcm93LmNsYXNzTGlzdC5yZW1vdmUoXCJjaGF0LWhpZGRlblwiLCBcImNoYXQtZmlsdGVyLW1hdGNoXCIpO1xuICAgICAgfSk7XG4gICAgICBpZiAoZWxlbWVudHMuZmlsdGVyRW1wdHkpIHtcbiAgICAgICAgZWxlbWVudHMuZmlsdGVyRW1wdHkuY2xhc3NMaXN0LmFkZChcImQtbm9uZVwiKTtcbiAgICAgIH1cbiAgICAgIGlmIChlbGVtZW50cy5maWx0ZXJIaW50KSB7XG4gICAgICAgIGVsZW1lbnRzLmZpbHRlckhpbnQudGV4dENvbnRlbnQgPSBmaWx0ZXJIaW50RGVmYXVsdDtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBjbGVhclRyYW5zY3JpcHRGaWx0ZXIoZm9jdXMgPSB0cnVlKSB7XG4gICAgc3RhdGUuYWN0aXZlRmlsdGVyID0gXCJcIjtcbiAgICBpZiAoZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LnZhbHVlID0gXCJcIjtcbiAgICB9XG4gICAgcmVhcHBseVRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICBpZiAoZm9jdXMgJiYgZWxlbWVudHMuZmlsdGVySW5wdXQpIHtcbiAgICAgIGVsZW1lbnRzLmZpbHRlcklucHV0LmZvY3VzKCk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gcmVuZGVySGlzdG9yeShlbnRyaWVzLCBvcHRpb25zID0ge30pIHtcbiAgICBjb25zdCB7IHJlcGxhY2UgPSBmYWxzZSB9ID0gb3B0aW9ucztcbiAgICBpZiAoIUFycmF5LmlzQXJyYXkoZW50cmllcykgfHwgZW50cmllcy5sZW5ndGggPT09IDApIHtcbiAgICAgIGlmIChyZXBsYWNlKSB7XG4gICAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuaW5uZXJIVE1MID0gXCJcIjtcbiAgICAgICAgc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCA9IGZhbHNlO1xuICAgICAgICBoaWRlU2Nyb2xsQnV0dG9uKCk7XG4gICAgICAgIHRpbWVsaW5lU3RvcmUuY2xlYXIoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHJlcGxhY2UpIHtcbiAgICAgIGVsZW1lbnRzLnRyYW5zY3JpcHQuaW5uZXJIVE1MID0gXCJcIjtcbiAgICAgIHN0YXRlLmhpc3RvcnlCb290c3RyYXBwZWQgPSBmYWxzZTtcbiAgICAgIHN0YXRlLnN0cmVhbVJvdyA9IG51bGw7XG4gICAgICBzdGF0ZS5zdHJlYW1CdWYgPSBcIlwiO1xuICAgICAgdGltZWxpbmVTdG9yZS5jbGVhcigpO1xuICAgIH1cbiAgICBpZiAoc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCAmJiAhcmVwbGFjZSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gdHJ1ZTtcbiAgICBlbnRyaWVzXG4gICAgICAuc2xpY2UoKVxuICAgICAgLnJldmVyc2UoKVxuICAgICAgLmZvckVhY2goKGl0ZW0pID0+IHtcbiAgICAgICAgaWYgKGl0ZW0ucXVlcnkpIHtcbiAgICAgICAgICBhcHBlbmRNZXNzYWdlKFwidXNlclwiLCBpdGVtLnF1ZXJ5LCB7XG4gICAgICAgICAgICB0aW1lc3RhbXA6IGl0ZW0udGltZXN0YW1wLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIGlmIChpdGVtLnJlc3BvbnNlKSB7XG4gICAgICAgICAgYXBwZW5kTWVzc2FnZShcImFzc2lzdGFudFwiLCBpdGVtLnJlc3BvbnNlLCB7XG4gICAgICAgICAgICB0aW1lc3RhbXA6IGl0ZW0udGltZXN0YW1wLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICBzdGF0ZS5ib290c3RyYXBwaW5nID0gZmFsc2U7XG4gICAgc3RhdGUuaGlzdG9yeUJvb3RzdHJhcHBlZCA9IHRydWU7XG4gICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IGZhbHNlIH0pO1xuICAgIGhpZGVTY3JvbGxCdXR0b24oKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIHN0YXJ0U3RyZWFtKCkge1xuICAgIHN0YXRlLnN0cmVhbUJ1ZiA9IFwiXCI7XG4gICAgY29uc3QgdHMgPSBub3dJU08oKTtcbiAgICBzdGF0ZS5zdHJlYW1NZXNzYWdlSWQgPSB0aW1lbGluZVN0b3JlLm1ha2VNZXNzYWdlSWQoKTtcbiAgICBzdGF0ZS5zdHJlYW1Sb3cgPSBsaW5lKFxuICAgICAgXCJhc3Npc3RhbnRcIixcbiAgICAgICc8ZGl2IGNsYXNzPVwiY2hhdC1idWJibGVcIj48c3BhbiBjbGFzcz1cImNoYXQtY3Vyc29yXCI+XHUyNThEPC9zcGFuPjwvZGl2PicsXG4gICAgICB7XG4gICAgICAgIHJhd1RleHQ6IFwiXCIsXG4gICAgICAgIHRpbWVzdGFtcDogdHMsXG4gICAgICAgIG1lc3NhZ2VJZDogc3RhdGUuc3RyZWFtTWVzc2FnZUlkLFxuICAgICAgICBtZXRhZGF0YTogeyBzdHJlYW1pbmc6IHRydWUgfSxcbiAgICAgIH0sXG4gICAgKTtcbiAgICBzZXREaWFnbm9zdGljcyh7IGxhc3RNZXNzYWdlQXQ6IHRzIH0pO1xuICAgIGlmIChzdGF0ZS5yZXNldFN0YXR1c1RpbWVyKSB7XG4gICAgICBjbGVhclRpbWVvdXQoc3RhdGUucmVzZXRTdGF0dXNUaW1lcik7XG4gICAgfVxuICAgIHNldENvbXBvc2VyU3RhdHVzKFwiUlx1MDBFOXBvbnNlIGVuIGNvdXJzXHUyMDI2XCIsIFwiaW5mb1wiKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGlzU3RyZWFtaW5nKCkge1xuICAgIHJldHVybiBCb29sZWFuKHN0YXRlLnN0cmVhbVJvdyk7XG4gIH1cblxuICBmdW5jdGlvbiBoYXNTdHJlYW1CdWZmZXIoKSB7XG4gICAgcmV0dXJuIEJvb2xlYW4oc3RhdGUuc3RyZWFtQnVmKTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGVuZFN0cmVhbShkZWx0YSkge1xuICAgIGlmICghc3RhdGUuc3RyZWFtUm93KSB7XG4gICAgICBzdGFydFN0cmVhbSgpO1xuICAgIH1cbiAgICBjb25zdCBzaG91bGRTdGljayA9IGlzQXRCb3R0b20oKTtcbiAgICBzdGF0ZS5zdHJlYW1CdWYgKz0gZGVsdGEgfHwgXCJcIjtcbiAgICBjb25zdCBidWJibGUgPSBzdGF0ZS5zdHJlYW1Sb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICBpZiAoYnViYmxlKSB7XG4gICAgICBidWJibGUuaW5uZXJIVE1MID0gYCR7cmVuZGVyTWFya2Rvd24oc3RhdGUuc3RyZWFtQnVmKX08c3BhbiBjbGFzcz1cImNoYXQtY3Vyc29yXCI+XHUyNThEPC9zcGFuPmA7XG4gICAgfVxuICAgIGlmIChzdGF0ZS5zdHJlYW1NZXNzYWdlSWQpIHtcbiAgICAgIHRpbWVsaW5lU3RvcmUudXBkYXRlKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCwge1xuICAgICAgICB0ZXh0OiBzdGF0ZS5zdHJlYW1CdWYsXG4gICAgICAgIG1ldGFkYXRhOiB7IHN0cmVhbWluZzogdHJ1ZSB9LFxuICAgICAgfSk7XG4gICAgfVxuICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogbm93SVNPKCkgfSk7XG4gICAgaWYgKHNob3VsZFN0aWNrKSB7XG4gICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogZmFsc2UgfSk7XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gZW5kU3RyZWFtKGRhdGEpIHtcbiAgICBpZiAoIXN0YXRlLnN0cmVhbVJvdykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBidWJibGUgPSBzdGF0ZS5zdHJlYW1Sb3cucXVlcnlTZWxlY3RvcihcIi5jaGF0LWJ1YmJsZVwiKTtcbiAgICBpZiAoYnViYmxlKSB7XG4gICAgICBidWJibGUuaW5uZXJIVE1MID0gcmVuZGVyTWFya2Rvd24oc3RhdGUuc3RyZWFtQnVmKTtcbiAgICAgIGNvbnN0IG1ldGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiZGl2XCIpO1xuICAgICAgbWV0YS5jbGFzc05hbWUgPSBcImNoYXQtbWV0YVwiO1xuICAgICAgY29uc3QgdHMgPSBkYXRhICYmIGRhdGEudGltZXN0YW1wID8gZGF0YS50aW1lc3RhbXAgOiBub3dJU08oKTtcbiAgICAgIG1ldGEudGV4dENvbnRlbnQgPSBmb3JtYXRUaW1lc3RhbXAodHMpO1xuICAgICAgaWYgKGRhdGEgJiYgZGF0YS5lcnJvcikge1xuICAgICAgICBtZXRhLmNsYXNzTGlzdC5hZGQoXCJ0ZXh0LWRhbmdlclwiKTtcbiAgICAgICAgbWV0YS50ZXh0Q29udGVudCA9IGAke21ldGEudGV4dENvbnRlbnR9IFx1MjAyMiAke2RhdGEuZXJyb3J9YDtcbiAgICAgIH1cbiAgICAgIGJ1YmJsZS5hcHBlbmRDaGlsZChtZXRhKTtcbiAgICAgIGRlY29yYXRlUm93KHN0YXRlLnN0cmVhbVJvdywgXCJhc3Npc3RhbnRcIik7XG4gICAgICBoaWdobGlnaHRSb3coc3RhdGUuc3RyZWFtUm93LCBcImFzc2lzdGFudFwiKTtcbiAgICAgIGlmIChpc0F0Qm90dG9tKCkpIHtcbiAgICAgICAgc2Nyb2xsVG9Cb3R0b20oeyBzbW9vdGg6IHRydWUgfSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBzaG93U2Nyb2xsQnV0dG9uKCk7XG4gICAgICB9XG4gICAgICBpZiAoc3RhdGUuc3RyZWFtTWVzc2FnZUlkKSB7XG4gICAgICAgIHRpbWVsaW5lU3RvcmUudXBkYXRlKHN0YXRlLnN0cmVhbU1lc3NhZ2VJZCwge1xuICAgICAgICAgIHRleHQ6IHN0YXRlLnN0cmVhbUJ1ZixcbiAgICAgICAgICB0aW1lc3RhbXA6IHRzLFxuICAgICAgICAgIG1ldGFkYXRhOiB7XG4gICAgICAgICAgICBzdHJlYW1pbmc6IG51bGwsXG4gICAgICAgICAgICAuLi4oZGF0YSAmJiBkYXRhLmVycm9yID8geyBlcnJvcjogZGF0YS5lcnJvciB9IDogeyBlcnJvcjogbnVsbCB9KSxcbiAgICAgICAgICB9LFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICAgIHNldERpYWdub3N0aWNzKHsgbGFzdE1lc3NhZ2VBdDogdHMgfSk7XG4gICAgfVxuICAgIGNvbnN0IGhhc0Vycm9yID0gQm9vbGVhbihkYXRhICYmIGRhdGEuZXJyb3IpO1xuICAgIHNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgaGFzRXJyb3JcbiAgICAgICAgPyBcIlJcdTAwRTlwb25zZSBpbmRpc3BvbmlibGUuIENvbnN1bHRleiBsZXMgam91cm5hdXguXCJcbiAgICAgICAgOiBcIlJcdTAwRTlwb25zZSByZVx1MDBFN3VlLlwiLFxuICAgICAgaGFzRXJyb3IgPyBcImRhbmdlclwiIDogXCJzdWNjZXNzXCIsXG4gICAgKTtcbiAgICBzY2hlZHVsZUNvbXBvc2VySWRsZShoYXNFcnJvciA/IDYwMDAgOiAzNTAwKTtcbiAgICBzdGF0ZS5zdHJlYW1Sb3cgPSBudWxsO1xuICAgIHN0YXRlLnN0cmVhbUJ1ZiA9IFwiXCI7XG4gICAgc3RhdGUuc3RyZWFtTWVzc2FnZUlkID0gbnVsbDtcbiAgfVxuXG4gIGZ1bmN0aW9uIGFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhzdWdnZXN0aW9ucykge1xuICAgIGlmICghZWxlbWVudHMucXVpY2tBY3Rpb25zKSByZXR1cm47XG4gICAgaWYgKCFBcnJheS5pc0FycmF5KHN1Z2dlc3Rpb25zKSB8fCBzdWdnZXN0aW9ucy5sZW5ndGggPT09IDApIHJldHVybjtcbiAgICBjb25zdCBidXR0b25zID0gQXJyYXkuZnJvbShcbiAgICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5xdWVyeVNlbGVjdG9yQWxsKFwiYnV0dG9uLnFhXCIpLFxuICAgICk7XG4gICAgY29uc3QgbG9va3VwID0gbmV3IE1hcCgpO1xuICAgIGJ1dHRvbnMuZm9yRWFjaCgoYnRuKSA9PiBsb29rdXAuc2V0KGJ0bi5kYXRhc2V0LmFjdGlvbiwgYnRuKSk7XG4gICAgY29uc3QgZnJhZyA9IGRvY3VtZW50LmNyZWF0ZURvY3VtZW50RnJhZ21lbnQoKTtcbiAgICBzdWdnZXN0aW9ucy5mb3JFYWNoKChrZXkpID0+IHtcbiAgICAgIGlmIChsb29rdXAuaGFzKGtleSkpIHtcbiAgICAgICAgZnJhZy5hcHBlbmRDaGlsZChsb29rdXAuZ2V0KGtleSkpO1xuICAgICAgICBsb29rdXAuZGVsZXRlKGtleSk7XG4gICAgICB9XG4gICAgfSk7XG4gICAgbG9va3VwLmZvckVhY2goKGJ0bikgPT4gZnJhZy5hcHBlbmRDaGlsZChidG4pKTtcbiAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMuaW5uZXJIVE1MID0gXCJcIjtcbiAgICBlbGVtZW50cy5xdWlja0FjdGlvbnMuYXBwZW5kQ2hpbGQoZnJhZyk7XG4gIH1cblxuICBmdW5jdGlvbiBmb3JtYXRQZXJmKGQpIHtcbiAgICBjb25zdCBiaXRzID0gW107XG4gICAgaWYgKGQgJiYgdHlwZW9mIGQuY3B1ICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICBjb25zdCBjcHUgPSBOdW1iZXIoZC5jcHUpO1xuICAgICAgaWYgKCFOdW1iZXIuaXNOYU4oY3B1KSkge1xuICAgICAgICBiaXRzLnB1c2goYENQVSAke2NwdS50b0ZpeGVkKDApfSVgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGQgJiYgdHlwZW9mIGQudHRmYl9tcyAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgY29uc3QgdHRmYiA9IE51bWJlcihkLnR0ZmJfbXMpO1xuICAgICAgaWYgKCFOdW1iZXIuaXNOYU4odHRmYikpIHtcbiAgICAgICAgYml0cy5wdXNoKGBUVEZCICR7dHRmYn0gbXNgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGJpdHMuam9pbihcIiBcdTIwMjIgXCIpIHx8IFwibWlzZSBcdTAwRTAgam91clwiO1xuICB9XG5cbiAgZnVuY3Rpb24gYXR0YWNoRXZlbnRzKCkge1xuICAgIGlmIChlbGVtZW50cy5jb21wb3Nlcikge1xuICAgICAgZWxlbWVudHMuY29tcG9zZXIuYWRkRXZlbnRMaXN0ZW5lcihcInN1Ym1pdFwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgY29uc3QgdGV4dCA9IChlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIikudHJpbSgpO1xuICAgICAgICBlbWl0KFwic3VibWl0XCIsIHsgdGV4dCB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5xdWlja0FjdGlvbnMpIHtcbiAgICAgIGVsZW1lbnRzLnF1aWNrQWN0aW9ucy5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGNvbnN0IHRhcmdldCA9IGV2ZW50LnRhcmdldDtcbiAgICAgICAgaWYgKCEodGFyZ2V0IGluc3RhbmNlb2YgSFRNTEJ1dHRvbkVsZW1lbnQpKSB7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGFjdGlvbiA9IHRhcmdldC5kYXRhc2V0LmFjdGlvbjtcbiAgICAgICAgaWYgKCFhY3Rpb24pIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgZW1pdChcInF1aWNrLWFjdGlvblwiLCB7IGFjdGlvbiB9KTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJJbnB1dCkge1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQuYWRkRXZlbnRMaXN0ZW5lcihcImlucHV0XCIsIChldmVudCkgPT4ge1xuICAgICAgICBlbWl0KFwiZmlsdGVyLWNoYW5nZVwiLCB7IHZhbHVlOiBldmVudC50YXJnZXQudmFsdWUgfHwgXCJcIiB9KTtcbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMuZmlsdGVySW5wdXQuYWRkRXZlbnRMaXN0ZW5lcihcImtleWRvd25cIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIGlmIChldmVudC5rZXkgPT09IFwiRXNjYXBlXCIpIHtcbiAgICAgICAgICBldmVudC5wcmV2ZW50RGVmYXVsdCgpO1xuICAgICAgICAgIGVtaXQoXCJmaWx0ZXItY2xlYXJcIik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5maWx0ZXJDbGVhcikge1xuICAgICAgZWxlbWVudHMuZmlsdGVyQ2xlYXIuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICAgICAgZW1pdChcImZpbHRlci1jbGVhclwiKTtcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5leHBvcnRKc29uKSB7XG4gICAgICBlbGVtZW50cy5leHBvcnRKc29uLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PlxuICAgICAgICBlbWl0KFwiZXhwb3J0XCIsIHsgZm9ybWF0OiBcImpzb25cIiB9KSxcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5leHBvcnRNYXJrZG93bikge1xuICAgICAgZWxlbWVudHMuZXhwb3J0TWFya2Rvd24uYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+XG4gICAgICAgIGVtaXQoXCJleHBvcnRcIiwgeyBmb3JtYXQ6IFwibWFya2Rvd25cIiB9KSxcbiAgICAgICk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50cy5leHBvcnRDb3B5KSB7XG4gICAgICBlbGVtZW50cy5leHBvcnRDb3B5LmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiBlbWl0KFwiZXhwb3J0LWNvcHlcIikpO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwiaW5wdXRcIiwgKGV2ZW50KSA9PiB7XG4gICAgICAgIHVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgICAgYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgICAgY29uc3QgdmFsdWUgPSBldmVudC50YXJnZXQudmFsdWUgfHwgXCJcIjtcbiAgICAgICAgaWYgKCF2YWx1ZS50cmltKCkpIHtcbiAgICAgICAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUoKTtcbiAgICAgICAgfVxuICAgICAgICBlbWl0KFwicHJvbXB0LWlucHV0XCIsIHsgdmFsdWUgfSk7XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwicGFzdGVcIiwgKCkgPT4ge1xuICAgICAgICB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgdXBkYXRlUHJvbXB0TWV0cmljcygpO1xuICAgICAgICAgIGF1dG9zaXplUHJvbXB0KCk7XG4gICAgICAgICAgZW1pdChcInByb21wdC1pbnB1dFwiLCB7IHZhbHVlOiBlbGVtZW50cy5wcm9tcHQudmFsdWUgfHwgXCJcIiB9KTtcbiAgICAgICAgfSwgMCk7XG4gICAgICB9KTtcbiAgICAgIGVsZW1lbnRzLnByb21wdC5hZGRFdmVudExpc3RlbmVyKFwia2V5ZG93blwiLCAoZXZlbnQpID0+IHtcbiAgICAgICAgaWYgKChldmVudC5jdHJsS2V5IHx8IGV2ZW50Lm1ldGFLZXkpICYmIGV2ZW50LmtleSA9PT0gXCJFbnRlclwiKSB7XG4gICAgICAgICAgZXZlbnQucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICBlbWl0KFwic3VibWl0XCIsIHsgdGV4dDogKGVsZW1lbnRzLnByb21wdC52YWx1ZSB8fCBcIlwiKS50cmltKCkgfSk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgICAgZWxlbWVudHMucHJvbXB0LmFkZEV2ZW50TGlzdGVuZXIoXCJmb2N1c1wiLCAoKSA9PiB7XG4gICAgICAgIHNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiUlx1MDBFOWRpZ2V6IHZvdHJlIG1lc3NhZ2UsIHB1aXMgQ3RybCtFbnRyXHUwMEU5ZSBwb3VyIGwnZW52b3llci5cIixcbiAgICAgICAgICBcImluZm9cIixcbiAgICAgICAgKTtcbiAgICAgICAgc2NoZWR1bGVDb21wb3NlcklkbGUoNDAwMCk7XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoZWxlbWVudHMudHJhbnNjcmlwdCkge1xuICAgICAgZWxlbWVudHMudHJhbnNjcmlwdC5hZGRFdmVudExpc3RlbmVyKFwic2Nyb2xsXCIsICgpID0+IHtcbiAgICAgICAgaWYgKGlzQXRCb3R0b20oKSkge1xuICAgICAgICAgIGhpZGVTY3JvbGxCdXR0b24oKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBzaG93U2Nyb2xsQnV0dG9uKCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIGlmIChlbGVtZW50cy5zY3JvbGxCb3R0b20pIHtcbiAgICAgIGVsZW1lbnRzLnNjcm9sbEJvdHRvbS5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgICAgICBzY3JvbGxUb0JvdHRvbSh7IHNtb290aDogdHJ1ZSB9KTtcbiAgICAgICAgaWYgKGVsZW1lbnRzLnByb21wdCkge1xuICAgICAgICAgIGVsZW1lbnRzLnByb21wdC5mb2N1cygpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcInJlc2l6ZVwiLCAoKSA9PiB7XG4gICAgICBpZiAoaXNBdEJvdHRvbSgpKSB7XG4gICAgICAgIHNjcm9sbFRvQm90dG9tKHsgc21vb3RoOiBmYWxzZSB9KTtcbiAgICAgIH1cbiAgICB9KTtcblxuICAgIHVwZGF0ZU5ldHdvcmtTdGF0dXMoKTtcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcIm9ubGluZVwiLCAoKSA9PiB7XG4gICAgICB1cGRhdGVOZXR3b3JrU3RhdHVzKCk7XG4gICAgICBhbm5vdW5jZUNvbm5lY3Rpb24oXCJDb25uZXhpb24gclx1MDBFOXNlYXUgcmVzdGF1clx1MDBFOWUuXCIsIFwiaW5mb1wiKTtcbiAgICB9KTtcbiAgICB3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcihcIm9mZmxpbmVcIiwgKCkgPT4ge1xuICAgICAgdXBkYXRlTmV0d29ya1N0YXR1cygpO1xuICAgICAgYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29ubmV4aW9uIHJcdTAwRTlzZWF1IHBlcmR1ZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgfSk7XG5cbiAgICBjb25zdCB0b2dnbGVCdG4gPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcInRvZ2dsZS1kYXJrLW1vZGVcIik7XG4gICAgY29uc3QgZGFya01vZGVLZXkgPSBcImRhcmstbW9kZVwiO1xuXG4gICAgZnVuY3Rpb24gYXBwbHlEYXJrTW9kZShlbmFibGVkKSB7XG4gICAgICBkb2N1bWVudC5ib2R5LmNsYXNzTGlzdC50b2dnbGUoXCJkYXJrLW1vZGVcIiwgZW5hYmxlZCk7XG4gICAgICBpZiAodG9nZ2xlQnRuKSB7XG4gICAgICAgIHRvZ2dsZUJ0bi50ZXh0Q29udGVudCA9IGVuYWJsZWQgPyBcIk1vZGUgQ2xhaXJcIiA6IFwiTW9kZSBTb21icmVcIjtcbiAgICAgICAgdG9nZ2xlQnRuLnNldEF0dHJpYnV0ZShcImFyaWEtcHJlc3NlZFwiLCBlbmFibGVkID8gXCJ0cnVlXCIgOiBcImZhbHNlXCIpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICBhcHBseURhcmtNb2RlKHdpbmRvdy5sb2NhbFN0b3JhZ2UuZ2V0SXRlbShkYXJrTW9kZUtleSkgPT09IFwiMVwiKTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byByZWFkIGRhcmsgbW9kZSBwcmVmZXJlbmNlXCIsIGVycik7XG4gICAgfVxuXG4gICAgaWYgKHRvZ2dsZUJ0bikge1xuICAgICAgdG9nZ2xlQnRuLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgICAgIGNvbnN0IGVuYWJsZWQgPSAhZG9jdW1lbnQuYm9keS5jbGFzc0xpc3QuY29udGFpbnMoXCJkYXJrLW1vZGVcIik7XG4gICAgICAgIGFwcGx5RGFya01vZGUoZW5hYmxlZCk7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5zZXRJdGVtKGRhcmtNb2RlS2V5LCBlbmFibGVkID8gXCIxXCIgOiBcIjBcIik7XG4gICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcIlVuYWJsZSB0byBwZXJzaXN0IGRhcmsgbW9kZSBwcmVmZXJlbmNlXCIsIGVycik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIGluaXRpYWxpc2UoKSB7XG4gICAgc2V0RGlhZ25vc3RpY3MoeyBjb25uZWN0ZWRBdDogbnVsbCwgbGFzdE1lc3NhZ2VBdDogbnVsbCwgbGF0ZW5jeU1zOiBudWxsIH0pO1xuICAgIHVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICBhdXRvc2l6ZVByb21wdCgpO1xuICAgIHNldENvbXBvc2VyU3RhdHVzSWRsZSgpO1xuICAgIGF0dGFjaEV2ZW50cygpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBlbGVtZW50cyxcbiAgICBvbixcbiAgICBlbWl0LFxuICAgIGluaXRpYWxpc2UsXG4gICAgcmVuZGVySGlzdG9yeSxcbiAgICBhcHBlbmRNZXNzYWdlLFxuICAgIHNldEJ1c3ksXG4gICAgc2hvd0Vycm9yLFxuICAgIGhpZGVFcnJvcixcbiAgICBzZXRDb21wb3NlclN0YXR1cyxcbiAgICBzZXRDb21wb3NlclN0YXR1c0lkbGUsXG4gICAgc2NoZWR1bGVDb21wb3NlcklkbGUsXG4gICAgdXBkYXRlUHJvbXB0TWV0cmljcyxcbiAgICBhdXRvc2l6ZVByb21wdCxcbiAgICBzdGFydFN0cmVhbSxcbiAgICBhcHBlbmRTdHJlYW0sXG4gICAgZW5kU3RyZWFtLFxuICAgIGFubm91bmNlQ29ubmVjdGlvbixcbiAgICB1cGRhdGVDb25uZWN0aW9uTWV0YSxcbiAgICBzZXREaWFnbm9zdGljcyxcbiAgICBhcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcsXG4gICAgYXBwbHlUcmFuc2NyaXB0RmlsdGVyLFxuICAgIHJlYXBwbHlUcmFuc2NyaXB0RmlsdGVyLFxuICAgIGNsZWFyVHJhbnNjcmlwdEZpbHRlcixcbiAgICBzZXRXc1N0YXR1cyxcbiAgICB1cGRhdGVOZXR3b3JrU3RhdHVzLFxuICAgIHNjcm9sbFRvQm90dG9tLFxuICAgIHNldCBkaWFnbm9zdGljcyh2YWx1ZSkge1xuICAgICAgT2JqZWN0LmFzc2lnbihkaWFnbm9zdGljcywgdmFsdWUpO1xuICAgIH0sXG4gICAgZ2V0IGRpYWdub3N0aWNzKCkge1xuICAgICAgcmV0dXJuIHsgLi4uZGlhZ25vc3RpY3MgfTtcbiAgICB9LFxuICAgIGZvcm1hdFRpbWVzdGFtcCxcbiAgICBub3dJU08sXG4gICAgZm9ybWF0UGVyZixcbiAgICBpc1N0cmVhbWluZyxcbiAgICBoYXNTdHJlYW1CdWZmZXIsXG4gIH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUF1dGhTZXJ2aWNlKGNvbmZpZykge1xuICBmdW5jdGlvbiBwZXJzaXN0VG9rZW4odG9rZW4pIHtcbiAgICBpZiAoIXRva2VuKSByZXR1cm47XG4gICAgdHJ5IHtcbiAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2Uuc2V0SXRlbShcImp3dFwiLCB0b2tlbik7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcGVyc2lzdCBKV1QgaW4gbG9jYWxTdG9yYWdlXCIsIGVycik7XG4gICAgfVxuICB9XG5cbiAgaWYgKGNvbmZpZy50b2tlbikge1xuICAgIHBlcnNpc3RUb2tlbihjb25maWcudG9rZW4pO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZ2V0Snd0KCkge1xuICAgIHRyeSB7XG4gICAgICBjb25zdCBzdG9yZWQgPSB3aW5kb3cubG9jYWxTdG9yYWdlLmdldEl0ZW0oXCJqd3RcIik7XG4gICAgICBpZiAoc3RvcmVkKSB7XG4gICAgICAgIHJldHVybiBzdG9yZWQ7XG4gICAgICB9XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBjb25zb2xlLndhcm4oXCJVbmFibGUgdG8gcmVhZCBKV1QgZnJvbSBsb2NhbFN0b3JhZ2VcIiwgZXJyKTtcbiAgICB9XG4gICAgaWYgKGNvbmZpZy50b2tlbikge1xuICAgICAgcmV0dXJuIGNvbmZpZy50b2tlbjtcbiAgICB9XG4gICAgdGhyb3cgbmV3IEVycm9yKFwiTWlzc2luZyBKV1QgKHN0b3JlIGl0IGluIGxvY2FsU3RvcmFnZSBhcyAnand0JykuXCIpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBnZXRKd3QsXG4gICAgcGVyc2lzdFRva2VuLFxuICB9O1xufVxuIiwgImltcG9ydCB7IGFwaVVybCB9IGZyb20gXCIuLi9jb25maWcuanNcIjtcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUh0dHBTZXJ2aWNlKHsgY29uZmlnLCBhdXRoIH0pIHtcbiAgYXN5bmMgZnVuY3Rpb24gYXV0aG9yaXNlZEZldGNoKHBhdGgsIG9wdGlvbnMgPSB7fSkge1xuICAgIGNvbnN0IGp3dCA9IGF3YWl0IGF1dGguZ2V0Snd0KCk7XG4gICAgY29uc3QgaGVhZGVycyA9IHtcbiAgICAgIC4uLihvcHRpb25zLmhlYWRlcnMgfHwge30pLFxuICAgICAgQXV0aG9yaXphdGlvbjogYEJlYXJlciAke2p3dH1gLFxuICAgIH07XG4gICAgcmV0dXJuIGZldGNoKGFwaVVybChjb25maWcsIHBhdGgpLCB7IC4uLm9wdGlvbnMsIGhlYWRlcnMgfSk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBmZXRjaFRpY2tldCgpIHtcbiAgICBjb25zdCByZXNwID0gYXdhaXQgYXV0aG9yaXNlZEZldGNoKFwiL2FwaS92MS9hdXRoL3dzL3RpY2tldFwiLCB7XG4gICAgICBtZXRob2Q6IFwiUE9TVFwiLFxuICAgIH0pO1xuICAgIGlmICghcmVzcC5vaykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBUaWNrZXQgZXJyb3I6ICR7cmVzcC5zdGF0dXN9YCk7XG4gICAgfVxuICAgIGNvbnN0IGJvZHkgPSBhd2FpdCByZXNwLmpzb24oKTtcbiAgICBpZiAoIWJvZHkgfHwgIWJvZHkudGlja2V0KSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXCJUaWNrZXQgcmVzcG9uc2UgaW52YWxpZGVcIik7XG4gICAgfVxuICAgIHJldHVybiBib2R5LnRpY2tldDtcbiAgfVxuXG4gIGFzeW5jIGZ1bmN0aW9uIHBvc3RDaGF0KG1lc3NhZ2UpIHtcbiAgICBjb25zdCByZXNwID0gYXdhaXQgYXV0aG9yaXNlZEZldGNoKFwiL2FwaS92MS9jb252ZXJzYXRpb24vY2hhdFwiLCB7XG4gICAgICBtZXRob2Q6IFwiUE9TVFwiLFxuICAgICAgaGVhZGVyczogeyBcIkNvbnRlbnQtVHlwZVwiOiBcImFwcGxpY2F0aW9uL2pzb25cIiB9LFxuICAgICAgYm9keTogSlNPTi5zdHJpbmdpZnkoeyBtZXNzYWdlIH0pLFxuICAgIH0pO1xuICAgIGlmICghcmVzcC5vaykge1xuICAgICAgY29uc3QgcGF5bG9hZCA9IGF3YWl0IHJlc3AudGV4dCgpO1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBIVFRQICR7cmVzcC5zdGF0dXN9OiAke3BheWxvYWR9YCk7XG4gICAgfVxuICAgIHJldHVybiByZXNwO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gcG9zdFN1Z2dlc3Rpb25zKHByb21wdCkge1xuICAgIGNvbnN0IHJlc3AgPSBhd2FpdCBhdXRob3Jpc2VkRmV0Y2goXCIvYXBpL3YxL3VpL3N1Z2dlc3Rpb25zXCIsIHtcbiAgICAgIG1ldGhvZDogXCJQT1NUXCIsXG4gICAgICBoZWFkZXJzOiB7IFwiQ29udGVudC1UeXBlXCI6IFwiYXBwbGljYXRpb24vanNvblwiIH0sXG4gICAgICBib2R5OiBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgIHByb21wdCxcbiAgICAgICAgYWN0aW9uczogW1wiY29kZVwiLCBcInN1bW1hcml6ZVwiLCBcImV4cGxhaW5cIl0sXG4gICAgICB9KSxcbiAgICB9KTtcbiAgICBpZiAoIXJlc3Aub2spIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgU3VnZ2VzdGlvbiBlcnJvcjogJHtyZXNwLnN0YXR1c31gKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3AuanNvbigpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBmZXRjaFRpY2tldCxcbiAgICBwb3N0Q2hhdCxcbiAgICBwb3N0U3VnZ2VzdGlvbnMsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgbm93SVNPIH0gZnJvbSBcIi4uL3V0aWxzL3RpbWUuanNcIjtcblxuZnVuY3Rpb24gYnVpbGRFeHBvcnRGaWxlbmFtZShleHRlbnNpb24pIHtcbiAgY29uc3Qgc3RhbXAgPSBub3dJU08oKS5yZXBsYWNlKC9bOi5dL2csIFwiLVwiKTtcbiAgcmV0dXJuIGBtb25nYXJzLWNoYXQtJHtzdGFtcH0uJHtleHRlbnNpb259YDtcbn1cblxuZnVuY3Rpb24gYnVpbGRNYXJrZG93bkV4cG9ydChpdGVtcykge1xuICBjb25zdCBsaW5lcyA9IFtcIiMgSGlzdG9yaXF1ZSBkZSBjb252ZXJzYXRpb24gbW9uR0FSU1wiLCBcIlwiXTtcbiAgaXRlbXMuZm9yRWFjaCgoaXRlbSkgPT4ge1xuICAgIGNvbnN0IHJvbGUgPSBpdGVtLnJvbGUgPyBpdGVtLnJvbGUudG9VcHBlckNhc2UoKSA6IFwiTUVTU0FHRVwiO1xuICAgIGxpbmVzLnB1c2goYCMjICR7cm9sZX1gKTtcbiAgICBpZiAoaXRlbS50aW1lc3RhbXApIHtcbiAgICAgIGxpbmVzLnB1c2goYCpIb3JvZGF0YWdlXHUwMEEwOiogJHtpdGVtLnRpbWVzdGFtcH1gKTtcbiAgICB9XG4gICAgaWYgKGl0ZW0ubWV0YWRhdGEgJiYgT2JqZWN0LmtleXMoaXRlbS5tZXRhZGF0YSkubGVuZ3RoID4gMCkge1xuICAgICAgT2JqZWN0LmVudHJpZXMoaXRlbS5tZXRhZGF0YSkuZm9yRWFjaCgoW2tleSwgdmFsdWVdKSA9PiB7XG4gICAgICAgIGxpbmVzLnB1c2goYCoke2tleX1cdTAwQTA6KiAke3ZhbHVlfWApO1xuICAgICAgfSk7XG4gICAgfVxuICAgIGxpbmVzLnB1c2goXCJcIik7XG4gICAgbGluZXMucHVzaChpdGVtLnRleHQgfHwgXCJcIik7XG4gICAgbGluZXMucHVzaChcIlwiKTtcbiAgfSk7XG4gIHJldHVybiBsaW5lcy5qb2luKFwiXFxuXCIpO1xufVxuXG5mdW5jdGlvbiBkb3dubG9hZEJsb2IoZmlsZW5hbWUsIHRleHQsIHR5cGUpIHtcbiAgaWYgKCF3aW5kb3cuVVJMIHx8IHR5cGVvZiB3aW5kb3cuVVJMLmNyZWF0ZU9iamVjdFVSTCAhPT0gXCJmdW5jdGlvblwiKSB7XG4gICAgY29uc29sZS53YXJuKFwiQmxvYiBleHBvcnQgdW5zdXBwb3J0ZWQgaW4gdGhpcyBlbnZpcm9ubWVudFwiKTtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgY29uc3QgYmxvYiA9IG5ldyBCbG9iKFt0ZXh0XSwgeyB0eXBlIH0pO1xuICBjb25zdCB1cmwgPSBVUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpO1xuICBjb25zdCBhbmNob3IgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiYVwiKTtcbiAgYW5jaG9yLmhyZWYgPSB1cmw7XG4gIGFuY2hvci5kb3dubG9hZCA9IGZpbGVuYW1lO1xuICBkb2N1bWVudC5ib2R5LmFwcGVuZENoaWxkKGFuY2hvcik7XG4gIGFuY2hvci5jbGljaygpO1xuICBkb2N1bWVudC5ib2R5LnJlbW92ZUNoaWxkKGFuY2hvcik7XG4gIHdpbmRvdy5zZXRUaW1lb3V0KCgpID0+IFVSTC5yZXZva2VPYmplY3RVUkwodXJsKSwgMCk7XG4gIHJldHVybiB0cnVlO1xufVxuXG5hc3luYyBmdW5jdGlvbiBjb3B5VG9DbGlwYm9hcmQodGV4dCkge1xuICBpZiAoIXRleHQpIHJldHVybiBmYWxzZTtcbiAgdHJ5IHtcbiAgICBpZiAobmF2aWdhdG9yLmNsaXBib2FyZCAmJiBuYXZpZ2F0b3IuY2xpcGJvYXJkLndyaXRlVGV4dCkge1xuICAgICAgYXdhaXQgbmF2aWdhdG9yLmNsaXBib2FyZC53cml0ZVRleHQodGV4dCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IHRleHRhcmVhID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcInRleHRhcmVhXCIpO1xuICAgICAgdGV4dGFyZWEudmFsdWUgPSB0ZXh0O1xuICAgICAgdGV4dGFyZWEuc2V0QXR0cmlidXRlKFwicmVhZG9ubHlcIiwgXCJyZWFkb25seVwiKTtcbiAgICAgIHRleHRhcmVhLnN0eWxlLnBvc2l0aW9uID0gXCJhYnNvbHV0ZVwiO1xuICAgICAgdGV4dGFyZWEuc3R5bGUubGVmdCA9IFwiLTk5OTlweFwiO1xuICAgICAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZCh0ZXh0YXJlYSk7XG4gICAgICB0ZXh0YXJlYS5zZWxlY3QoKTtcbiAgICAgIGRvY3VtZW50LmV4ZWNDb21tYW5kKFwiY29weVwiKTtcbiAgICAgIGRvY3VtZW50LmJvZHkucmVtb3ZlQ2hpbGQodGV4dGFyZWEpO1xuICAgIH1cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgY29uc29sZS53YXJuKFwiQ29weSBjb252ZXJzYXRpb24gZmFpbGVkXCIsIGVycik7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVFeHBvcnRlcih7IHRpbWVsaW5lU3RvcmUsIGFubm91bmNlIH0pIHtcbiAgZnVuY3Rpb24gY29sbGVjdFRyYW5zY3JpcHQoKSB7XG4gICAgcmV0dXJuIHRpbWVsaW5lU3RvcmUuY29sbGVjdCgpO1xuICB9XG5cbiAgYXN5bmMgZnVuY3Rpb24gZXhwb3J0Q29udmVyc2F0aW9uKGZvcm1hdCkge1xuICAgIGNvbnN0IGl0ZW1zID0gY29sbGVjdFRyYW5zY3JpcHQoKTtcbiAgICBpZiAoIWl0ZW1zLmxlbmd0aCkge1xuICAgICAgYW5ub3VuY2UoXCJBdWN1biBtZXNzYWdlIFx1MDBFMCBleHBvcnRlci5cIiwgXCJ3YXJuaW5nXCIpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAoZm9ybWF0ID09PSBcImpzb25cIikge1xuICAgICAgY29uc3QgcGF5bG9hZCA9IHtcbiAgICAgICAgZXhwb3J0ZWRfYXQ6IG5vd0lTTygpLFxuICAgICAgICBjb3VudDogaXRlbXMubGVuZ3RoLFxuICAgICAgICBpdGVtcyxcbiAgICAgIH07XG4gICAgICBpZiAoXG4gICAgICAgIGRvd25sb2FkQmxvYihcbiAgICAgICAgICBidWlsZEV4cG9ydEZpbGVuYW1lKFwianNvblwiKSxcbiAgICAgICAgICBKU09OLnN0cmluZ2lmeShwYXlsb2FkLCBudWxsLCAyKSxcbiAgICAgICAgICBcImFwcGxpY2F0aW9uL2pzb25cIixcbiAgICAgICAgKVxuICAgICAgKSB7XG4gICAgICAgIGFubm91bmNlKFwiRXhwb3J0IEpTT04gZ1x1MDBFOW5cdTAwRTlyXHUwMEU5LlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBhbm5vdW5jZShcIkV4cG9ydCBub24gc3VwcG9ydFx1MDBFOSBkYW5zIGNlIG5hdmlnYXRldXIuXCIsIFwiZGFuZ2VyXCIpO1xuICAgICAgfVxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAoZm9ybWF0ID09PSBcIm1hcmtkb3duXCIpIHtcbiAgICAgIGlmIChcbiAgICAgICAgZG93bmxvYWRCbG9iKFxuICAgICAgICAgIGJ1aWxkRXhwb3J0RmlsZW5hbWUoXCJtZFwiKSxcbiAgICAgICAgICBidWlsZE1hcmtkb3duRXhwb3J0KGl0ZW1zKSxcbiAgICAgICAgICBcInRleHQvbWFya2Rvd25cIixcbiAgICAgICAgKVxuICAgICAgKSB7XG4gICAgICAgIGFubm91bmNlKFwiRXhwb3J0IE1hcmtkb3duIGdcdTAwRTluXHUwMEU5clx1MDBFOS5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgYW5ub3VuY2UoXCJFeHBvcnQgbm9uIHN1cHBvcnRcdTAwRTkgZGFucyBjZSBuYXZpZ2F0ZXVyLlwiLCBcImRhbmdlclwiKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBjb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQoKSB7XG4gICAgY29uc3QgaXRlbXMgPSBjb2xsZWN0VHJhbnNjcmlwdCgpO1xuICAgIGlmICghaXRlbXMubGVuZ3RoKSB7XG4gICAgICBhbm5vdW5jZShcIkF1Y3VuIG1lc3NhZ2UgXHUwMEUwIGNvcGllci5cIiwgXCJ3YXJuaW5nXCIpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCB0ZXh0ID0gYnVpbGRNYXJrZG93bkV4cG9ydChpdGVtcyk7XG4gICAgaWYgKGF3YWl0IGNvcHlUb0NsaXBib2FyZCh0ZXh0KSkge1xuICAgICAgYW5ub3VuY2UoXCJDb252ZXJzYXRpb24gY29waVx1MDBFOWUgYXUgcHJlc3NlLXBhcGllcnMuXCIsIFwic3VjY2Vzc1wiKTtcbiAgICB9IGVsc2Uge1xuICAgICAgYW5ub3VuY2UoXCJJbXBvc3NpYmxlIGRlIGNvcGllciBsYSBjb252ZXJzYXRpb24uXCIsIFwiZGFuZ2VyXCIpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgZXhwb3J0Q29udmVyc2F0aW9uLFxuICAgIGNvcHlDb252ZXJzYXRpb25Ub0NsaXBib2FyZCxcbiAgfTtcbn1cbiIsICJpbXBvcnQgeyBub3dJU08gfSBmcm9tIFwiLi4vdXRpbHMvdGltZS5qc1wiO1xuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU29ja2V0Q2xpZW50KHsgY29uZmlnLCBodHRwLCB1aSwgb25FdmVudCB9KSB7XG4gIGxldCB3cztcbiAgbGV0IHdzSEJlYXQ7XG4gIGxldCByZWNvbm5lY3RCYWNrb2ZmID0gNTAwO1xuICBjb25zdCBCQUNLT0ZGX01BWCA9IDgwMDA7XG5cbiAgZnVuY3Rpb24gc2FmZVNlbmQob2JqKSB7XG4gICAgdHJ5IHtcbiAgICAgIGlmICh3cyAmJiB3cy5yZWFkeVN0YXRlID09PSBXZWJTb2NrZXQuT1BFTikge1xuICAgICAgICB3cy5zZW5kKEpTT04uc3RyaW5naWZ5KG9iaikpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS53YXJuKFwiVW5hYmxlIHRvIHNlbmQgb3ZlciBXZWJTb2NrZXRcIiwgZXJyKTtcbiAgICB9XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBvcGVuU29ja2V0KCkge1xuICAgIHRyeSB7XG4gICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIk9idGVudGlvbiBkXHUyMDE5dW4gdGlja2V0IGRlIGNvbm5leGlvblx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICBjb25zdCB0aWNrZXQgPSBhd2FpdCBodHRwLmZldGNoVGlja2V0KCk7XG4gICAgICBjb25zdCB3c1VybCA9IG5ldyBVUkwoXCIvd3MvY2hhdC9cIiwgY29uZmlnLmJhc2VVcmwpO1xuICAgICAgd3NVcmwucHJvdG9jb2wgPSBjb25maWcuYmFzZVVybC5wcm90b2NvbCA9PT0gXCJodHRwczpcIiA/IFwid3NzOlwiIDogXCJ3czpcIjtcbiAgICAgIHdzVXJsLnNlYXJjaFBhcmFtcy5zZXQoXCJ0XCIsIHRpY2tldCk7XG5cbiAgICAgIHdzID0gbmV3IFdlYlNvY2tldCh3c1VybC50b1N0cmluZygpKTtcbiAgICAgIHVpLnNldFdzU3RhdHVzKFwiY29ubmVjdGluZ1wiKTtcbiAgICAgIHVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiQ29ubmV4aW9uIGF1IHNlcnZldXJcdTIwMjZcIiwgXCJpbmZvXCIpO1xuXG4gICAgICB3cy5vbm9wZW4gPSAoKSA9PiB7XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwib25saW5lXCIpO1xuICAgICAgICBjb25zdCBjb25uZWN0ZWRBdCA9IG5vd0lTTygpO1xuICAgICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcbiAgICAgICAgICBgQ29ubmVjdFx1MDBFOSBsZSAke3VpLmZvcm1hdFRpbWVzdGFtcChjb25uZWN0ZWRBdCl9YCxcbiAgICAgICAgICBcInN1Y2Nlc3NcIixcbiAgICAgICAgKTtcbiAgICAgICAgdWkuc2V0RGlhZ25vc3RpY3MoeyBjb25uZWN0ZWRBdCwgbGFzdE1lc3NhZ2VBdDogY29ubmVjdGVkQXQgfSk7XG4gICAgICAgIHVpLmhpZGVFcnJvcigpO1xuICAgICAgICB3c0hCZWF0ID0gd2luZG93LnNldEludGVydmFsKCgpID0+IHtcbiAgICAgICAgICBzYWZlU2VuZCh7IHR5cGU6IFwiY2xpZW50LnBpbmdcIiwgdHM6IG5vd0lTTygpIH0pO1xuICAgICAgICB9LCAyMDAwMCk7XG4gICAgICAgIHJlY29ubmVjdEJhY2tvZmYgPSA1MDA7XG4gICAgICAgIHVpLnNldENvbXBvc2VyU3RhdHVzKFwiQ29ubmVjdFx1MDBFOS4gVm91cyBwb3V2ZXogXHUwMEU5Y2hhbmdlci5cIiwgXCJzdWNjZXNzXCIpO1xuICAgICAgICB1aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgIH07XG5cbiAgICAgIHdzLm9ubWVzc2FnZSA9IChldnQpID0+IHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBldiA9IEpTT04ucGFyc2UoZXZ0LmRhdGEpO1xuICAgICAgICAgIG9uRXZlbnQoZXYpO1xuICAgICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgICBjb25zb2xlLmVycm9yKFwiQmFkIGV2ZW50IHBheWxvYWRcIiwgZXJyLCBldnQuZGF0YSk7XG4gICAgICAgIH1cbiAgICAgIH07XG5cbiAgICAgIHdzLm9uY2xvc2UgPSAoKSA9PiB7XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwib2ZmbGluZVwiKTtcbiAgICAgICAgaWYgKHdzSEJlYXQpIHtcbiAgICAgICAgICBjbGVhckludGVydmFsKHdzSEJlYXQpO1xuICAgICAgICB9XG4gICAgICAgIHVpLnNldERpYWdub3N0aWNzKHsgbGF0ZW5jeU1zOiB1bmRlZmluZWQgfSk7XG4gICAgICAgIGNvbnN0IGRlbGF5ID0gcmVjb25uZWN0QmFja29mZiArIE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIDI1MCk7XG4gICAgICAgIGNvbnN0IHNlY29uZHMgPSBNYXRoLm1heCgxLCBNYXRoLnJvdW5kKGRlbGF5IC8gMTAwMCkpO1xuICAgICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcbiAgICAgICAgICBgRFx1MDBFOWNvbm5lY3RcdTAwRTkuIE5vdXZlbGxlIHRlbnRhdGl2ZSBkYW5zICR7c2Vjb25kc30gc1x1MjAyNmAsXG4gICAgICAgICAgXCJ3YXJuaW5nXCIsXG4gICAgICAgICk7XG4gICAgICAgIHVpLnNldENvbXBvc2VyU3RhdHVzKFxuICAgICAgICAgIFwiQ29ubmV4aW9uIHBlcmR1ZS4gUmVjb25uZXhpb24gYXV0b21hdGlxdWVcdTIwMjZcIixcbiAgICAgICAgICBcIndhcm5pbmdcIixcbiAgICAgICAgKTtcbiAgICAgICAgdWkuc2NoZWR1bGVDb21wb3NlcklkbGUoNjAwMCk7XG4gICAgICAgIHJlY29ubmVjdEJhY2tvZmYgPSBNYXRoLm1pbihCQUNLT0ZGX01BWCwgcmVjb25uZWN0QmFja29mZiAqIDIpO1xuICAgICAgICB3aW5kb3cuc2V0VGltZW91dChvcGVuU29ja2V0LCBkZWxheSk7XG4gICAgICB9O1xuXG4gICAgICB3cy5vbmVycm9yID0gKGVycikgPT4ge1xuICAgICAgICBjb25zb2xlLmVycm9yKFwiV2ViU29ja2V0IGVycm9yXCIsIGVycik7XG4gICAgICAgIHVpLnNldFdzU3RhdHVzKFwiZXJyb3JcIiwgXCJFcnJldXIgV2ViU29ja2V0XCIpO1xuICAgICAgICB1aS51cGRhdGVDb25uZWN0aW9uTWV0YShcIkVycmV1ciBXZWJTb2NrZXQgZFx1MDBFOXRlY3RcdTAwRTllLlwiLCBcImRhbmdlclwiKTtcbiAgICAgICAgdWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJVbmUgZXJyZXVyIHJcdTAwRTlzZWF1IGVzdCBzdXJ2ZW51ZS5cIiwgXCJkYW5nZXJcIik7XG4gICAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfTtcbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGNvbnNvbGUuZXJyb3IoZXJyKTtcbiAgICAgIGNvbnN0IG1lc3NhZ2UgPSBlcnIgaW5zdGFuY2VvZiBFcnJvciA/IGVyci5tZXNzYWdlIDogU3RyaW5nKGVycik7XG4gICAgICB1aS5zZXRXc1N0YXR1cyhcImVycm9yXCIsIG1lc3NhZ2UpO1xuICAgICAgdWkudXBkYXRlQ29ubmVjdGlvbk1ldGEobWVzc2FnZSwgXCJkYW5nZXJcIik7XG4gICAgICB1aS5zZXRDb21wb3NlclN0YXR1cyhcbiAgICAgICAgXCJDb25uZXhpb24gaW5kaXNwb25pYmxlLiBOb3V2ZWwgZXNzYWkgYmllbnRcdTAwRjR0LlwiLFxuICAgICAgICBcImRhbmdlclwiLFxuICAgICAgKTtcbiAgICAgIHVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgY29uc3QgZGVsYXkgPSBNYXRoLm1pbihCQUNLT0ZGX01BWCwgcmVjb25uZWN0QmFja29mZik7XG4gICAgICByZWNvbm5lY3RCYWNrb2ZmID0gTWF0aC5taW4oQkFDS09GRl9NQVgsIHJlY29ubmVjdEJhY2tvZmYgKiAyKTtcbiAgICAgIHdpbmRvdy5zZXRUaW1lb3V0KG9wZW5Tb2NrZXQsIGRlbGF5KTtcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBkaXNwb3NlKCkge1xuICAgIGlmICh3c0hCZWF0KSB7XG4gICAgICBjbGVhckludGVydmFsKHdzSEJlYXQpO1xuICAgIH1cbiAgICBpZiAod3MgJiYgd3MucmVhZHlTdGF0ZSA9PT0gV2ViU29ja2V0Lk9QRU4pIHtcbiAgICAgIHdzLmNsb3NlKCk7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBvcGVuOiBvcGVuU29ja2V0LFxuICAgIHNlbmQ6IHNhZmVTZW5kLFxuICAgIGRpc3Bvc2UsXG4gIH07XG59XG4iLCAiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlKHsgaHR0cCwgdWkgfSkge1xuICBsZXQgdGltZXIgPSBudWxsO1xuXG4gIGZ1bmN0aW9uIHNjaGVkdWxlKHByb21wdCkge1xuICAgIGlmICh0aW1lcikge1xuICAgICAgY2xlYXJUaW1lb3V0KHRpbWVyKTtcbiAgICB9XG4gICAgdGltZXIgPSB3aW5kb3cuc2V0VGltZW91dCgoKSA9PiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCksIDIyMCk7XG4gIH1cblxuICBhc3luYyBmdW5jdGlvbiBmZXRjaFN1Z2dlc3Rpb25zKHByb21wdCkge1xuICAgIGlmICghcHJvbXB0IHx8IHByb21wdC50cmltKCkubGVuZ3RoIDwgMykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICB0cnkge1xuICAgICAgY29uc3QgcGF5bG9hZCA9IGF3YWl0IGh0dHAucG9zdFN1Z2dlc3Rpb25zKHByb21wdC50cmltKCkpO1xuICAgICAgaWYgKHBheWxvYWQgJiYgQXJyYXkuaXNBcnJheShwYXlsb2FkLmFjdGlvbnMpKSB7XG4gICAgICAgIHVpLmFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhwYXlsb2FkLmFjdGlvbnMpO1xuICAgICAgfVxuICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgY29uc29sZS5kZWJ1ZyhcIkFVSSBzdWdnZXN0aW9uIGZldGNoIGZhaWxlZFwiLCBlcnIpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB7XG4gICAgc2NoZWR1bGUsXG4gIH07XG59XG4iLCAiaW1wb3J0IHsgcmVzb2x2ZUNvbmZpZyB9IGZyb20gXCIuL2NvbmZpZy5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlVGltZWxpbmVTdG9yZSB9IGZyb20gXCIuL3N0YXRlL3RpbWVsaW5lU3RvcmUuanNcIjtcbmltcG9ydCB7IGNyZWF0ZUNoYXRVaSB9IGZyb20gXCIuL3VpL2NoYXRVaS5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlQXV0aFNlcnZpY2UgfSBmcm9tIFwiLi9zZXJ2aWNlcy9hdXRoLmpzXCI7XG5pbXBvcnQgeyBjcmVhdGVIdHRwU2VydmljZSB9IGZyb20gXCIuL3NlcnZpY2VzL2h0dHAuanNcIjtcbmltcG9ydCB7IGNyZWF0ZUV4cG9ydGVyIH0gZnJvbSBcIi4vc2VydmljZXMvZXhwb3J0ZXIuanNcIjtcbmltcG9ydCB7IGNyZWF0ZVNvY2tldENsaWVudCB9IGZyb20gXCIuL3NlcnZpY2VzL3NvY2tldC5qc1wiO1xuaW1wb3J0IHsgY3JlYXRlU3VnZ2VzdGlvblNlcnZpY2UgfSBmcm9tIFwiLi9zZXJ2aWNlcy9zdWdnZXN0aW9ucy5qc1wiO1xuaW1wb3J0IHsgbm93SVNPIH0gZnJvbSBcIi4vdXRpbHMvdGltZS5qc1wiO1xuXG5mdW5jdGlvbiBxdWVyeUVsZW1lbnRzKGRvYykge1xuICBjb25zdCBieUlkID0gKGlkKSA9PiBkb2MuZ2V0RWxlbWVudEJ5SWQoaWQpO1xuICByZXR1cm4ge1xuICAgIHRyYW5zY3JpcHQ6IGJ5SWQoXCJ0cmFuc2NyaXB0XCIpLFxuICAgIGNvbXBvc2VyOiBieUlkKFwiY29tcG9zZXJcIiksXG4gICAgcHJvbXB0OiBieUlkKFwicHJvbXB0XCIpLFxuICAgIHNlbmQ6IGJ5SWQoXCJzZW5kXCIpLFxuICAgIHdzU3RhdHVzOiBieUlkKFwid3Mtc3RhdHVzXCIpLFxuICAgIHF1aWNrQWN0aW9uczogYnlJZChcInF1aWNrLWFjdGlvbnNcIiksXG4gICAgY29ubmVjdGlvbjogYnlJZChcImNvbm5lY3Rpb25cIiksXG4gICAgZXJyb3JBbGVydDogYnlJZChcImVycm9yLWFsZXJ0XCIpLFxuICAgIGVycm9yTWVzc2FnZTogYnlJZChcImVycm9yLW1lc3NhZ2VcIiksXG4gICAgc2Nyb2xsQm90dG9tOiBieUlkKFwic2Nyb2xsLWJvdHRvbVwiKSxcbiAgICBjb21wb3NlclN0YXR1czogYnlJZChcImNvbXBvc2VyLXN0YXR1c1wiKSxcbiAgICBwcm9tcHRDb3VudDogYnlJZChcInByb21wdC1jb3VudFwiKSxcbiAgICBjb25uZWN0aW9uTWV0YTogYnlJZChcImNvbm5lY3Rpb24tbWV0YVwiKSxcbiAgICBmaWx0ZXJJbnB1dDogYnlJZChcImNoYXQtc2VhcmNoXCIpLFxuICAgIGZpbHRlckNsZWFyOiBieUlkKFwiY2hhdC1zZWFyY2gtY2xlYXJcIiksXG4gICAgZmlsdGVyRW1wdHk6IGJ5SWQoXCJmaWx0ZXItZW1wdHlcIiksXG4gICAgZmlsdGVySGludDogYnlJZChcImNoYXQtc2VhcmNoLWhpbnRcIiksXG4gICAgZXhwb3J0SnNvbjogYnlJZChcImV4cG9ydC1qc29uXCIpLFxuICAgIGV4cG9ydE1hcmtkb3duOiBieUlkKFwiZXhwb3J0LW1hcmtkb3duXCIpLFxuICAgIGV4cG9ydENvcHk6IGJ5SWQoXCJleHBvcnQtY29weVwiKSxcbiAgICBkaWFnQ29ubmVjdGVkOiBieUlkKFwiZGlhZy1jb25uZWN0ZWRcIiksXG4gICAgZGlhZ0xhc3RNZXNzYWdlOiBieUlkKFwiZGlhZy1sYXN0LW1lc3NhZ2VcIiksXG4gICAgZGlhZ0xhdGVuY3k6IGJ5SWQoXCJkaWFnLWxhdGVuY3lcIiksXG4gICAgZGlhZ05ldHdvcms6IGJ5SWQoXCJkaWFnLW5ldHdvcmtcIiksXG4gIH07XG59XG5cbmZ1bmN0aW9uIHJlYWRIaXN0b3J5KGRvYykge1xuICBjb25zdCBoaXN0b3J5RWxlbWVudCA9IGRvYy5nZXRFbGVtZW50QnlJZChcImNoYXQtaGlzdG9yeVwiKTtcbiAgaWYgKCFoaXN0b3J5RWxlbWVudCkge1xuICAgIHJldHVybiBbXTtcbiAgfVxuICBjb25zdCBwYXlsb2FkID0gaGlzdG9yeUVsZW1lbnQudGV4dENvbnRlbnQgfHwgXCJudWxsXCI7XG4gIGhpc3RvcnlFbGVtZW50LnJlbW92ZSgpO1xuICB0cnkge1xuICAgIGNvbnN0IHBhcnNlZCA9IEpTT04ucGFyc2UocGF5bG9hZCk7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkocGFyc2VkKSkge1xuICAgICAgcmV0dXJuIHBhcnNlZDtcbiAgICB9XG4gICAgaWYgKHBhcnNlZCAmJiBwYXJzZWQuZXJyb3IpIHtcbiAgICAgIHJldHVybiB7IGVycm9yOiBwYXJzZWQuZXJyb3IgfTtcbiAgICB9XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIGNvbnNvbGUuZXJyb3IoXCJVbmFibGUgdG8gcGFyc2UgY2hhdCBoaXN0b3J5XCIsIGVycik7XG4gIH1cbiAgcmV0dXJuIFtdO1xufVxuXG5mdW5jdGlvbiBlbnN1cmVFbGVtZW50cyhlbGVtZW50cykge1xuICByZXR1cm4gQm9vbGVhbihlbGVtZW50cy50cmFuc2NyaXB0ICYmIGVsZW1lbnRzLmNvbXBvc2VyICYmIGVsZW1lbnRzLnByb21wdCk7XG59XG5cbmNvbnN0IFFVSUNLX1BSRVNFVFMgPSB7XG4gIGNvZGU6IFwiSmUgc291aGFpdGUgXHUwMEU5Y3JpcmUgZHUgY29kZVx1MjAyNlwiLFxuICBzdW1tYXJpemU6IFwiUlx1MDBFOXN1bWUgbGEgZGVybmlcdTAwRThyZSBjb252ZXJzYXRpb24uXCIsXG4gIGV4cGxhaW46IFwiRXhwbGlxdWUgdGEgZGVybmlcdTAwRThyZSByXHUwMEU5cG9uc2UgcGx1cyBzaW1wbGVtZW50LlwiLFxufTtcblxuZXhwb3J0IGNsYXNzIENoYXRBcHAge1xuICBjb25zdHJ1Y3Rvcihkb2MgPSBkb2N1bWVudCwgcmF3Q29uZmlnID0gd2luZG93LmNoYXRDb25maWcgfHwge30pIHtcbiAgICB0aGlzLmRvYyA9IGRvYztcbiAgICB0aGlzLmNvbmZpZyA9IHJlc29sdmVDb25maWcocmF3Q29uZmlnKTtcbiAgICB0aGlzLmVsZW1lbnRzID0gcXVlcnlFbGVtZW50cyhkb2MpO1xuICAgIGlmICghZW5zdXJlRWxlbWVudHModGhpcy5lbGVtZW50cykpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHdpbmRvdy5tYXJrZWQgJiYgdHlwZW9mIHdpbmRvdy5tYXJrZWQuc2V0T3B0aW9ucyA9PT0gXCJmdW5jdGlvblwiKSB7XG4gICAgICB3aW5kb3cubWFya2VkLnNldE9wdGlvbnMoe1xuICAgICAgICBicmVha3M6IHRydWUsXG4gICAgICAgIGdmbTogdHJ1ZSxcbiAgICAgICAgaGVhZGVySWRzOiBmYWxzZSxcbiAgICAgICAgbWFuZ2xlOiBmYWxzZSxcbiAgICAgIH0pO1xuICAgIH1cbiAgICB0aGlzLnRpbWVsaW5lU3RvcmUgPSBjcmVhdGVUaW1lbGluZVN0b3JlKCk7XG4gICAgdGhpcy51aSA9IGNyZWF0ZUNoYXRVaSh7XG4gICAgICBlbGVtZW50czogdGhpcy5lbGVtZW50cyxcbiAgICAgIHRpbWVsaW5lU3RvcmU6IHRoaXMudGltZWxpbmVTdG9yZSxcbiAgICB9KTtcbiAgICB0aGlzLmF1dGggPSBjcmVhdGVBdXRoU2VydmljZSh0aGlzLmNvbmZpZyk7XG4gICAgdGhpcy5odHRwID0gY3JlYXRlSHR0cFNlcnZpY2UoeyBjb25maWc6IHRoaXMuY29uZmlnLCBhdXRoOiB0aGlzLmF1dGggfSk7XG4gICAgdGhpcy5leHBvcnRlciA9IGNyZWF0ZUV4cG9ydGVyKHtcbiAgICAgIHRpbWVsaW5lU3RvcmU6IHRoaXMudGltZWxpbmVTdG9yZSxcbiAgICAgIGFubm91bmNlOiAobWVzc2FnZSwgdmFyaWFudCkgPT5cbiAgICAgICAgdGhpcy51aS5hbm5vdW5jZUNvbm5lY3Rpb24obWVzc2FnZSwgdmFyaWFudCksXG4gICAgfSk7XG4gICAgdGhpcy5zdWdnZXN0aW9ucyA9IGNyZWF0ZVN1Z2dlc3Rpb25TZXJ2aWNlKHtcbiAgICAgIGh0dHA6IHRoaXMuaHR0cCxcbiAgICAgIHVpOiB0aGlzLnVpLFxuICAgIH0pO1xuICAgIHRoaXMuc29ja2V0ID0gY3JlYXRlU29ja2V0Q2xpZW50KHtcbiAgICAgIGNvbmZpZzogdGhpcy5jb25maWcsXG4gICAgICBodHRwOiB0aGlzLmh0dHAsXG4gICAgICB1aTogdGhpcy51aSxcbiAgICAgIG9uRXZlbnQ6IChldikgPT4gdGhpcy5oYW5kbGVTb2NrZXRFdmVudChldiksXG4gICAgfSk7XG5cbiAgICBjb25zdCBoaXN0b3J5UGF5bG9hZCA9IHJlYWRIaXN0b3J5KGRvYyk7XG4gICAgaWYgKGhpc3RvcnlQYXlsb2FkICYmIGhpc3RvcnlQYXlsb2FkLmVycm9yKSB7XG4gICAgICB0aGlzLnVpLnNob3dFcnJvcihoaXN0b3J5UGF5bG9hZC5lcnJvcik7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGhpc3RvcnlQYXlsb2FkKSkge1xuICAgICAgdGhpcy51aS5yZW5kZXJIaXN0b3J5KGhpc3RvcnlQYXlsb2FkKTtcbiAgICB9XG5cbiAgICB0aGlzLnJlZ2lzdGVyVWlIYW5kbGVycygpO1xuICAgIHRoaXMudWkuaW5pdGlhbGlzZSgpO1xuICAgIHRoaXMuc29ja2V0Lm9wZW4oKTtcbiAgfVxuXG4gIHJlZ2lzdGVyVWlIYW5kbGVycygpIHtcbiAgICB0aGlzLnVpLm9uKFwic3VibWl0XCIsIGFzeW5jICh7IHRleHQgfSkgPT4ge1xuICAgICAgY29uc3QgdmFsdWUgPSAodGV4dCB8fCBcIlwiKS50cmltKCk7XG4gICAgICBpZiAoIXZhbHVlKSB7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJTYWlzaXNzZXogdW4gbWVzc2FnZSBhdmFudCBkXHUyMDE5ZW52b3llci5cIixcbiAgICAgICAgICBcIndhcm5pbmdcIixcbiAgICAgICAgKTtcbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgdGhpcy51aS5oaWRlRXJyb3IoKTtcbiAgICAgIGNvbnN0IHN1Ym1pdHRlZEF0ID0gbm93SVNPKCk7XG4gICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJ1c2VyXCIsIHZhbHVlLCB7XG4gICAgICAgIHRpbWVzdGFtcDogc3VibWl0dGVkQXQsXG4gICAgICAgIG1ldGFkYXRhOiB7IHN1Ym1pdHRlZDogdHJ1ZSB9LFxuICAgICAgfSk7XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSBcIlwiO1xuICAgICAgfVxuICAgICAgdGhpcy51aS51cGRhdGVQcm9tcHRNZXRyaWNzKCk7XG4gICAgICB0aGlzLnVpLmF1dG9zaXplUHJvbXB0KCk7XG4gICAgICB0aGlzLnVpLnNldENvbXBvc2VyU3RhdHVzKFwiTWVzc2FnZSBlbnZveVx1MDBFOVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgdGhpcy51aS5zZXRCdXN5KHRydWUpO1xuICAgICAgdGhpcy51aS5hcHBseVF1aWNrQWN0aW9uT3JkZXJpbmcoW1wiY29kZVwiLCBcInN1bW1hcml6ZVwiLCBcImV4cGxhaW5cIl0pO1xuXG4gICAgICB0cnkge1xuICAgICAgICBhd2FpdCB0aGlzLmh0dHAucG9zdENoYXQodmFsdWUpO1xuICAgICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgICB0aGlzLmVsZW1lbnRzLnByb21wdC5mb2N1cygpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMudWkuc3RhcnRTdHJlYW0oKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICB0aGlzLnVpLnNldEJ1c3koZmFsc2UpO1xuICAgICAgICBjb25zdCBtZXNzYWdlID0gZXJyIGluc3RhbmNlb2YgRXJyb3IgPyBlcnIubWVzc2FnZSA6IFN0cmluZyhlcnIpO1xuICAgICAgICB0aGlzLnVpLnNob3dFcnJvcihtZXNzYWdlKTtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwic3lzdGVtXCIsIG1lc3NhZ2UsIHtcbiAgICAgICAgICB2YXJpYW50OiBcImVycm9yXCIsXG4gICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgbWV0YWRhdGE6IHsgc3RhZ2U6IFwic3VibWl0XCIgfSxcbiAgICAgICAgfSk7XG4gICAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXG4gICAgICAgICAgXCJFbnZvaSBpbXBvc3NpYmxlLiBWXHUwMEU5cmlmaWV6IGxhIGNvbm5leGlvbi5cIixcbiAgICAgICAgICBcImRhbmdlclwiLFxuICAgICAgICApO1xuICAgICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDYwMDApO1xuICAgICAgfVxuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcInF1aWNrLWFjdGlvblwiLCAoeyBhY3Rpb24gfSkgPT4ge1xuICAgICAgaWYgKCFhY3Rpb24pIHJldHVybjtcbiAgICAgIGNvbnN0IHByZXNldCA9IFFVSUNLX1BSRVNFVFNbYWN0aW9uXSB8fCBhY3Rpb247XG4gICAgICBpZiAodGhpcy5lbGVtZW50cy5wcm9tcHQpIHtcbiAgICAgICAgdGhpcy5lbGVtZW50cy5wcm9tcHQudmFsdWUgPSBwcmVzZXQ7XG4gICAgICB9XG4gICAgICB0aGlzLnVpLnVwZGF0ZVByb21wdE1ldHJpY3MoKTtcbiAgICAgIHRoaXMudWkuYXV0b3NpemVQcm9tcHQoKTtcbiAgICAgIHRoaXMudWkuc2V0Q29tcG9zZXJTdGF0dXMoXCJTdWdnZXN0aW9uIGVudm95XHUwMEU5ZVx1MjAyNlwiLCBcImluZm9cIik7XG4gICAgICB0aGlzLnVpLnNjaGVkdWxlQ29tcG9zZXJJZGxlKDQwMDApO1xuICAgICAgdGhpcy51aS5lbWl0KFwic3VibWl0XCIsIHsgdGV4dDogcHJlc2V0IH0pO1xuICAgIH0pO1xuXG4gICAgdGhpcy51aS5vbihcImZpbHRlci1jaGFuZ2VcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgdGhpcy51aS5hcHBseVRyYW5zY3JpcHRGaWx0ZXIodmFsdWUsIHsgcHJlc2VydmVJbnB1dDogdHJ1ZSB9KTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJmaWx0ZXItY2xlYXJcIiwgKCkgPT4ge1xuICAgICAgdGhpcy51aS5jbGVhclRyYW5zY3JpcHRGaWx0ZXIoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJleHBvcnRcIiwgKHsgZm9ybWF0IH0pID0+IHtcbiAgICAgIHRoaXMuZXhwb3J0ZXIuZXhwb3J0Q29udmVyc2F0aW9uKGZvcm1hdCk7XG4gICAgfSk7XG5cbiAgICB0aGlzLnVpLm9uKFwiZXhwb3J0LWNvcHlcIiwgKCkgPT4ge1xuICAgICAgdGhpcy5leHBvcnRlci5jb3B5Q29udmVyc2F0aW9uVG9DbGlwYm9hcmQoKTtcbiAgICB9KTtcblxuICAgIHRoaXMudWkub24oXCJwcm9tcHQtaW5wdXRcIiwgKHsgdmFsdWUgfSkgPT4ge1xuICAgICAgaWYgKCF2YWx1ZSB8fCAhdmFsdWUudHJpbSgpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIGlmICh0aGlzLmVsZW1lbnRzLnNlbmQgJiYgdGhpcy5lbGVtZW50cy5zZW5kLmRpc2FibGVkKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICAgIHRoaXMuc3VnZ2VzdGlvbnMuc2NoZWR1bGUodmFsdWUpO1xuICAgIH0pO1xuICB9XG5cbiAgaGFuZGxlU29ja2V0RXZlbnQoZXYpIHtcbiAgICBjb25zdCB0eXBlID0gZXYgJiYgZXYudHlwZSA/IGV2LnR5cGUgOiBcIlwiO1xuICAgIGNvbnN0IGRhdGEgPSBldiAmJiBldi5kYXRhID8gZXYuZGF0YSA6IHt9O1xuICAgIHN3aXRjaCAodHlwZSkge1xuICAgICAgY2FzZSBcIndzLmNvbm5lY3RlZFwiOiB7XG4gICAgICAgIGlmIChkYXRhICYmIGRhdGEub3JpZ2luKSB7XG4gICAgICAgICAgdGhpcy51aS5hbm5vdW5jZUNvbm5lY3Rpb24oYENvbm5lY3RcdTAwRTkgdmlhICR7ZGF0YS5vcmlnaW59YCk7XG4gICAgICAgICAgdGhpcy51aS51cGRhdGVDb25uZWN0aW9uTWV0YShcbiAgICAgICAgICAgIGBDb25uZWN0XHUwMEU5IHZpYSAke2RhdGEub3JpZ2lufWAsXG4gICAgICAgICAgICBcInN1Y2Nlc3NcIixcbiAgICAgICAgICApO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRoaXMudWkuYW5ub3VuY2VDb25uZWN0aW9uKFwiQ29ubmVjdFx1MDBFOSBhdSBzZXJ2ZXVyLlwiKTtcbiAgICAgICAgICB0aGlzLnVpLnVwZGF0ZUNvbm5lY3Rpb25NZXRhKFwiQ29ubmVjdFx1MDBFOSBhdSBzZXJ2ZXVyLlwiLCBcInN1Y2Nlc3NcIik7XG4gICAgICAgIH1cbiAgICAgICAgdGhpcy51aS5zY2hlZHVsZUNvbXBvc2VySWRsZSg0MDAwKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiaGlzdG9yeS5zbmFwc2hvdFwiOiB7XG4gICAgICAgIGlmIChkYXRhICYmIEFycmF5LmlzQXJyYXkoZGF0YS5pdGVtcykpIHtcbiAgICAgICAgICB0aGlzLnVpLnJlbmRlckhpc3RvcnkoZGF0YS5pdGVtcywgeyByZXBsYWNlOiB0cnVlIH0pO1xuICAgICAgICB9XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImFpX21vZGVsLnJlc3BvbnNlX2NodW5rXCI6IHtcbiAgICAgICAgY29uc3QgZGVsdGEgPVxuICAgICAgICAgIHR5cGVvZiBkYXRhLmRlbHRhID09PSBcInN0cmluZ1wiID8gZGF0YS5kZWx0YSA6IGRhdGEudGV4dCB8fCBcIlwiO1xuICAgICAgICB0aGlzLnVpLmFwcGVuZFN0cmVhbShkZWx0YSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImFpX21vZGVsLnJlc3BvbnNlX2NvbXBsZXRlXCI6IHtcbiAgICAgICAgaWYgKGRhdGEgJiYgZGF0YS50ZXh0ICYmICF0aGlzLnVpLmhhc1N0cmVhbUJ1ZmZlcigpKSB7XG4gICAgICAgICAgdGhpcy51aS5hcHBlbmRTdHJlYW0oZGF0YS50ZXh0KTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLmVuZFN0cmVhbShkYXRhKTtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgaWYgKGRhdGEgJiYgdHlwZW9mIGRhdGEubGF0ZW5jeV9tcyAhPT0gXCJ1bmRlZmluZWRcIikge1xuICAgICAgICAgIHRoaXMudWkuc2V0RGlhZ25vc3RpY3MoeyBsYXRlbmN5TXM6IE51bWJlcihkYXRhLmxhdGVuY3lfbXMpIH0pO1xuICAgICAgICB9XG4gICAgICAgIGlmIChkYXRhICYmIGRhdGEub2sgPT09IGZhbHNlICYmIGRhdGEuZXJyb3IpIHtcbiAgICAgICAgICB0aGlzLnVpLmFwcGVuZE1lc3NhZ2UoXCJzeXN0ZW1cIiwgZGF0YS5lcnJvciwge1xuICAgICAgICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImNoYXQubWVzc2FnZVwiOiB7XG4gICAgICAgIGlmICghdGhpcy51aS5pc1N0cmVhbWluZygpKSB7XG4gICAgICAgICAgdGhpcy51aS5zdGFydFN0cmVhbSgpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChcbiAgICAgICAgICBkYXRhICYmXG4gICAgICAgICAgdHlwZW9mIGRhdGEucmVzcG9uc2UgPT09IFwic3RyaW5nXCIgJiZcbiAgICAgICAgICAhdGhpcy51aS5oYXNTdHJlYW1CdWZmZXIoKVxuICAgICAgICApIHtcbiAgICAgICAgICB0aGlzLnVpLmFwcGVuZFN0cmVhbShkYXRhLnJlc3BvbnNlKTtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnVpLmVuZFN0cmVhbShkYXRhKTtcbiAgICAgICAgdGhpcy51aS5zZXRCdXN5KGZhbHNlKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwiZXZvbHV0aW9uX2VuZ2luZS50cmFpbmluZ19jb21wbGV0ZVwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIGBcdTAwQzl2b2x1dGlvbiBtaXNlIFx1MDBFMCBqb3VyICR7ZGF0YSAmJiBkYXRhLnZlcnNpb24gPyBkYXRhLnZlcnNpb24gOiBcIlwifWAsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJva1wiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcImV2b2x1dGlvbl9lbmdpbmUudHJhaW5pbmdfZmFpbGVkXCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFxuICAgICAgICAgIFwic3lzdGVtXCIsXG4gICAgICAgICAgYFx1MDBDOWNoZWMgZGUgbCdcdTAwRTl2b2x1dGlvbiA6ICR7ZGF0YSAmJiBkYXRhLmVycm9yID8gZGF0YS5lcnJvciA6IFwiaW5jb25udVwifWAsXG4gICAgICAgICAge1xuICAgICAgICAgICAgdmFyaWFudDogXCJlcnJvclwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInNsZWVwX3RpbWVfY29tcHV0ZS5waGFzZV9zdGFydFwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIFwiT3B0aW1pc2F0aW9uIGVuIGFycmlcdTAwRThyZS1wbGFuIGRcdTAwRTltYXJyXHUwMEU5ZVx1MjAyNlwiLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiaGludFwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInNsZWVwX3RpbWVfY29tcHV0ZS5jcmVhdGl2ZV9waGFzZVwiOiB7XG4gICAgICAgIHRoaXMudWkuYXBwZW5kTWVzc2FnZShcbiAgICAgICAgICBcInN5c3RlbVwiLFxuICAgICAgICAgIGBFeHBsb3JhdGlvbiBkZSAke051bWJlcihkYXRhICYmIGRhdGEuaWRlYXMgPyBkYXRhLmlkZWFzIDogMSl9IGlkXHUwMEU5ZXNcdTIwMjZgLFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIHZhcmlhbnQ6IFwiaGludFwiLFxuICAgICAgICAgICAgYWxsb3dNYXJrZG93bjogZmFsc2UsXG4gICAgICAgICAgICBtZXRhZGF0YTogeyBldmVudDogdHlwZSB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgY2FzZSBcInBlcmZvcm1hbmNlLmFsZXJ0XCI6IHtcbiAgICAgICAgdGhpcy51aS5hcHBlbmRNZXNzYWdlKFwic3lzdGVtXCIsIGBQZXJmIDogJHt0aGlzLnVpLmZvcm1hdFBlcmYoZGF0YSl9YCwge1xuICAgICAgICAgIHZhcmlhbnQ6IFwid2FyblwiLFxuICAgICAgICAgIGFsbG93TWFya2Rvd246IGZhbHNlLFxuICAgICAgICAgIG1ldGFkYXRhOiB7IGV2ZW50OiB0eXBlIH0sXG4gICAgICAgIH0pO1xuICAgICAgICBpZiAoZGF0YSAmJiB0eXBlb2YgZGF0YS50dGZiX21zICE9PSBcInVuZGVmaW5lZFwiKSB7XG4gICAgICAgICAgdGhpcy51aS5zZXREaWFnbm9zdGljcyh7IGxhdGVuY3lNczogTnVtYmVyKGRhdGEudHRmYl9tcykgfSk7XG4gICAgICAgIH1cbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBjYXNlIFwidWkuc3VnZ2VzdGlvbnNcIjoge1xuICAgICAgICB0aGlzLnVpLmFwcGx5UXVpY2tBY3Rpb25PcmRlcmluZyhcbiAgICAgICAgICBBcnJheS5pc0FycmF5KGRhdGEuYWN0aW9ucykgPyBkYXRhLmFjdGlvbnMgOiBbXSxcbiAgICAgICAgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgICBkZWZhdWx0OlxuICAgICAgICBpZiAodHlwZSAmJiB0eXBlLnN0YXJ0c1dpdGgoXCJ3cy5cIikpIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgY29uc29sZS5kZWJ1ZyhcIlVuaGFuZGxlZCBldmVudFwiLCBldik7XG4gICAgfVxuICB9XG59XG4iLCAiaW1wb3J0IHsgQ2hhdEFwcCB9IGZyb20gXCIuL2FwcC5qc1wiO1xuXG5uZXcgQ2hhdEFwcChkb2N1bWVudCwgd2luZG93LmNoYXRDb25maWcgfHwge30pO1xuIl0sCiAgIm1hcHBpbmdzIjogIjs7QUFBTyxXQUFTLGNBQWMsTUFB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
