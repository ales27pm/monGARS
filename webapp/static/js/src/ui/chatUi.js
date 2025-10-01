import { createEmitter } from "../utils/emitter.js";
import { htmlToText, extractBubbleText, escapeHTML } from "../utils/dom.js";
import { renderMarkdown } from "../services/markdown.js";
import { formatTimestamp, nowISO } from "../utils/time.js";

export function createChatUi({ elements, timelineStore }) {
  const emitter = createEmitter();

  const sendIdleMarkup = elements.send ? elements.send.innerHTML : "";
  const sendIdleLabel =
    (elements.send && elements.send.getAttribute("data-idle-label")) ||
    (elements.send ? elements.send.textContent.trim() : "Envoyer");
  const sendBusyMarkup =
    '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Envoi…';
  const composerStatusDefault =
    (elements.composerStatus && elements.composerStatus.textContent.trim()) ||
    "Appuyez sur Ctrl+Entrée pour envoyer rapidement.";
  const filterHintDefault =
    (elements.filterHint && elements.filterHint.textContent.trim()) ||
    "Utilisez le filtre pour limiter l'historique. Appuyez sur Échap pour effacer.";
  const promptMax = Number(elements.prompt?.getAttribute("maxlength")) || null;
  const prefersReducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCROLL_THRESHOLD = 140;
  const PROMPT_MAX_HEIGHT = 320;

  const diagnostics = {
    connectedAt: null,
    lastMessageAt: null,
    latencyMs: null,
  };

  const state = {
    resetStatusTimer: null,
    hideScrollTimer: null,
    activeFilter: "",
    historyBootstrapped: elements.transcript.childElementCount > 0,
    bootstrapping: false,
    streamRow: null,
    streamBuf: "",
    streamMessageId: null,
  };

  const statusLabels = {
    offline: "Hors ligne",
    connecting: "Connexion…",
    online: "En ligne",
    error: "Erreur",
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
      PROMPT_MAX_HEIGHT,
    );
    elements.prompt.style.height = `${nextHeight}px`;
  }

  function isAtBottom() {
    if (!elements.transcript) return true;
    const distance =
      elements.transcript.scrollHeight -
      (elements.transcript.scrollTop + elements.transcript.clientHeight);
    return distance <= SCROLL_THRESHOLD;
  }

  function scrollToBottom(options = {}) {
    if (!elements.transcript) return;
    const smooth = options.smooth !== false && !prefersReducedMotion;
    elements.transcript.scrollTo({
      top: elements.transcript.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
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
      announceConnection("Contenu copié dans le presse-papiers.", "success");
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
      copyBtn.innerHTML =
        '<span aria-hidden="true">⧉</span><span class="visually-hidden">Copier le message</span>';
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
      const text =
        options.rawText && options.rawText.length > 0
          ? options.rawText
          : htmlToText(html);
      const id = timelineStore.register({
        id: options.messageId,
        role,
        text,
        timestamp: ts,
        row,
        metadata: options.metadata || {},
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
    allowMarkdown = true,
  }) {
    const classes = ["chat-bubble"];
    if (variant) {
      classes.push(`chat-bubble-${variant}`);
    }
    const content = allowMarkdown
      ? renderMarkdown(text)
      : escapeHTML(String(text));
    const metaBits = [];
    if (timestamp) {
      metaBits.push(formatTimestamp(timestamp));
    }
    if (metaSuffix) {
      metaBits.push(metaSuffix);
    }
    const metaHtml =
      metaBits.length > 0
        ? `<div class="chat-meta">${escapeHTML(metaBits.join(" • "))}</div>`
        : "";
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
      metadata,
    } = options;
    const bubble = buildBubble({
      text,
      timestamp,
      variant,
      metaSuffix,
      allowMarkdown,
    });
    const row = line(role, bubble, {
      rawText: text,
      timestamp,
      messageId,
      register,
      metadata,
    });
    setDiagnostics({ lastMessageAt: timestamp || nowISO() });
    return row;
  }

  function updateDiagnosticField(el, value) {
    if (!el) return;
    el.textContent = value || "—";
  }

  function setDiagnostics(patch) {
    Object.assign(diagnostics, patch);
    if (Object.prototype.hasOwnProperty.call(patch, "connectedAt")) {
      updateDiagnosticField(
        elements.diagConnected,
        diagnostics.connectedAt
          ? formatTimestamp(diagnostics.connectedAt)
          : "—",
      );
    }
    if (Object.prototype.hasOwnProperty.call(patch, "lastMessageAt")) {
      updateDiagnosticField(
        elements.diagLastMessage,
        diagnostics.lastMessageAt
          ? formatTimestamp(diagnostics.lastMessageAt)
          : "—",
      );
    }
    if (Object.prototype.hasOwnProperty.call(patch, "latencyMs")) {
      if (typeof diagnostics.latencyMs === "number") {
        updateDiagnosticField(
          elements.diagLatency,
          `${Math.max(0, Math.round(diagnostics.latencyMs))} ms`,
        );
      } else {
        updateDiagnosticField(elements.diagLatency, "—");
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
    Array.from(classList)
      .filter((cls) => cls.startsWith("alert-") && cls !== "alert")
      .forEach((cls) => classList.remove(cls));
    classList.add("alert");
    classList.add(`alert-${variant}`);
    elements.connection.textContent = message;
    classList.remove("visually-hidden");
    window.setTimeout(() => {
      classList.add("visually-hidden");
    }, 4000);
  }

  function updateConnectionMeta(message, tone = "muted") {
    if (!elements.connectionMeta) return;
    const tones = ["muted", "info", "success", "danger", "warning"];
    elements.connectionMeta.textContent = message;
    tones.forEach((t) => elements.connectionMeta.classList.remove(`text-${t}`));
    elements.connectionMeta.classList.add(`text-${tone}`);
  }

  function setWsStatus(state, title) {
    if (!elements.wsStatus) return;
    const label = statusLabels[state] || state;
    elements.wsStatus.textContent = label;
    elements.wsStatus.className = `badge ws-badge ${state}`;
    if (title) {
      elements.wsStatus.title = title;
    } else {
      elements.wsStatus.removeAttribute("title");
    }
  }

  function normalizeString(str) {
    const value = String(str || "");
    try {
      return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase();
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
          elements.filterEmpty.getAttribute("aria-live") || "polite",
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
        elements.transcript.querySelectorAll(".chat-row"),
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
        elements.transcript.querySelectorAll(".chat-row"),
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
        const meta = bubble?.querySelector(".chat-meta") || null;
        const role =
          row.dataset.role ||
          (row.classList.contains("chat-user")
            ? "user"
            : row.classList.contains("chat-assistant")
            ? "assistant"
            : "system");
        const text =
          row.dataset.rawText && row.dataset.rawText.length > 0
            ? row.dataset.rawText
            : bubble
            ? extractBubbleText(bubble)
            : row.textContent.trim();
        const timestamp =
          row.dataset.timestamp && row.dataset.timestamp.length > 0
            ? row.dataset.timestamp
            : meta
            ? meta.textContent.trim()
            : nowISO();
        const messageId = timelineStore.register({
          id: existingId,
          role,
          text,
          timestamp,
          row,
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
    entries
      .slice()
      .reverse()
      .forEach((item) => {
        if (item.query) {
          appendMessage("user", item.query, {
            timestamp: item.timestamp,
          });
        }
        if (item.response) {
          appendMessage("assistant", item.response, {
            timestamp: item.timestamp,
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
      '<div class="chat-bubble"><span class="chat-cursor">▍</span></div>',
      {
        rawText: "",
        timestamp: ts,
        messageId: state.streamMessageId,
        metadata: { streaming: true },
      },
    );
    setDiagnostics({ lastMessageAt: ts });
    if (state.resetStatusTimer) {
      clearTimeout(state.resetStatusTimer);
    }
    setComposerStatus("Réponse en cours…", "info");
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
      bubble.innerHTML = `${renderMarkdown(state.streamBuf)}<span class="chat-cursor">▍</span>`;
    }
    if (state.streamMessageId) {
      timelineStore.update(state.streamMessageId, {
        text: state.streamBuf,
        metadata: { streaming: true },
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
        meta.textContent = `${meta.textContent} • ${data.error}`;
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
            ...(data && data.error ? { error: data.error } : { error: null }),
          },
        });
      }
      setDiagnostics({ lastMessageAt: ts });
    }
    const hasError = Boolean(data && data.error);
    setComposerStatus(
      hasError
        ? "Réponse indisponible. Consultez les journaux."
        : "Réponse reçue.",
      hasError ? "danger" : "success",
    );
    scheduleComposerIdle(hasError ? 6000 : 3500);
    state.streamRow = null;
    state.streamBuf = "";
    state.streamMessageId = null;
  }

  function applyQuickActionOrdering(suggestions) {
    if (!elements.quickActions) return;
    if (!Array.isArray(suggestions) || suggestions.length === 0) return;
    const buttons = Array.from(
      elements.quickActions.querySelectorAll("button.qa"),
    );
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
    return bits.join(" • ") || "mise à jour";
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
      elements.exportJson.addEventListener("click", () =>
        emit("export", { format: "json" }),
      );
    }
    if (elements.exportMarkdown) {
      elements.exportMarkdown.addEventListener("click", () =>
        emit("export", { format: "markdown" }),
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
          "Rédigez votre message, puis Ctrl+Entrée pour l'envoyer.",
          "info",
        );
        scheduleComposerIdle(4000);
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
      announceConnection("Connexion réseau restaurée.", "info");
    });
    window.addEventListener("offline", () => {
      updateNetworkStatus();
      announceConnection("Connexion réseau perdue.", "danger");
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
    hasStreamBuffer,
  };
}
