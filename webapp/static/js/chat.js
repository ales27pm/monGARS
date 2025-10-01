/* monGARS chat frontend: event-driven UI wired to typed backend events */

(function () {
  const config = window.chatConfig || {};
  const els = {
    transcript: document.getElementById("transcript"),
    composer: document.getElementById("composer"),
    prompt: document.getElementById("prompt"),
    send: document.getElementById("send"),
    wsStatus: document.getElementById("ws-status"),
    quickActions: document.getElementById("quick-actions"),
    connection: document.getElementById("connection"),
    errorAlert: document.getElementById("error-alert"),
    errorMessage: document.getElementById("error-message"),
    scrollBottom: document.getElementById("scroll-bottom"),
    composerStatus: document.getElementById("composer-status"),
    promptCount: document.getElementById("prompt-count"),
    connectionMeta: document.getElementById("connection-meta"),
    filterInput: document.getElementById("chat-search"),
    filterClear: document.getElementById("chat-search-clear"),
    filterEmpty: document.getElementById("filter-empty"),
    filterHint: document.getElementById("chat-search-hint"),
    exportJson: document.getElementById("export-json"),
    exportMarkdown: document.getElementById("export-markdown"),
    exportCopy: document.getElementById("export-copy"),
    diagConnected: document.getElementById("diag-connected"),
    diagLastMessage: document.getElementById("diag-last-message"),
    diagLatency: document.getElementById("diag-latency"),
    diagNetwork: document.getElementById("diag-network"),
  };

  if (!els.transcript || !els.composer || !els.prompt) {
    return;
  }

  const sendIdleMarkup = els.send ? els.send.innerHTML : "";
  const sendIdleLabel =
    (els.send && els.send.getAttribute("data-idle-label")) ||
    (els.send ? els.send.textContent.trim() : "Envoyer");
  const sendBusyMarkup =
    '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Envoi…';
  const composerStatusDefault =
    (els.composerStatus && els.composerStatus.textContent.trim()) ||
    "Appuyez sur Ctrl+Entrée pour envoyer rapidement.";
  const filterHintDefault =
    (els.filterHint && els.filterHint.textContent.trim()) ||
    "Utilisez le filtre pour limiter l'historique. Appuyez sur Échap pour effacer.";
  const promptMax = Number(els.prompt.getAttribute("maxlength")) || null;
  const prefersReducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCROLL_THRESHOLD = 140;
  const PROMPT_MAX_HEIGHT = 320;

  if (window.marked && typeof window.marked.setOptions === "function") {
    window.marked.setOptions({
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false,
    });
  }

  const baseUrl = (() => {
    const candidate = config.fastapiUrl || window.location.origin;
    try {
      return new URL(candidate);
    } catch (err) {
      console.error("Invalid FASTAPI URL", err, candidate);
      return new URL(window.location.origin);
    }
  })();

  const apiUrl = (path) => new URL(path, baseUrl).toString();

  try {
    if (config.token) {
      window.localStorage.setItem("jwt", config.token);
    }
  } catch (err) {
    console.warn("Unable to persist JWT in localStorage", err);
  }

  const historyElement = document.getElementById("chat-history");
  let chatHistory = [];
  if (historyElement) {
    try {
      const parsed = JSON.parse(historyElement.textContent || "null");
      if (Array.isArray(parsed)) {
        chatHistory = parsed;
      } else if (parsed && parsed.error) {
        showError(parsed.error);
      }
    } catch (err) {
      console.error("Unable to parse chat history", err);
    }
    historyElement.remove();
  }

  let historyBootstrapped = els.transcript.childElementCount > 0;
  let bootstrapping = false;
  let resetStatusTimer = null;
  let hideScrollTimer = null;
  let activeFilter = "";
  const timelineOrder = [];
  const timelineMap = new Map();

  // ---- UX helpers ---------------------------------------------------------
  const nowISO = () => new Date().toISOString();
  const statusLabels = {
    offline: "Hors ligne",
    connecting: "Connexion…",
    online: "En ligne",
    error: "Erreur",
  };

  const diagnostics = {
    connectedAt: null,
    lastMessageAt: null,
    latencyMs: null,
  };

  function makeMessageId() {
    return `msg-${Date.now().toString(36)}-${Math.random()
      .toString(36)
      .slice(2, 8)}`;
  }

  function registerTimelineEntry({
    id,
    role,
    text = "",
    timestamp = nowISO(),
    row,
    metadata = {},
  }) {
    const messageId = id || makeMessageId();
    if (!timelineMap.has(messageId)) {
      timelineOrder.push(messageId);
    }
    timelineMap.set(messageId, {
      id: messageId,
      role,
      text,
      timestamp,
      row,
      metadata: { ...metadata },
    });
    if (row) {
      row.dataset.messageId = messageId;
      row.dataset.role = role;
      row.dataset.rawText = text;
      row.dataset.timestamp = timestamp;
    }
    return messageId;
  }

  function updateTimelineEntry(id, patch) {
    if (!timelineMap.has(id)) {
      return null;
    }
    const entry = timelineMap.get(id);
    const next = { ...entry, ...patch };
    if (patch.metadata) {
      const merged = { ...entry.metadata };
      Object.entries(patch.metadata).forEach(([key, value]) => {
        if (value === undefined || value === null) {
          delete merged[key];
        } else {
          merged[key] = value;
        }
      });
      next.metadata = merged;
    }
    timelineMap.set(id, next);
    if (next.row) {
      next.row.dataset.rawText = next.text || "";
      next.row.dataset.timestamp = next.timestamp || "";
      next.row.dataset.role = next.role || entry.role;
    }
    return next;
  }

  function collectTranscript() {
    return timelineOrder
      .map((id) => {
        const entry = timelineMap.get(id);
        if (!entry) {
          return null;
        }
        return {
          role: entry.role,
          text: entry.text,
          timestamp: entry.timestamp,
          ...(entry.metadata &&
            Object.keys(entry.metadata).length > 0 && {
              metadata: { ...entry.metadata },
            }),
        };
      })
      .filter(Boolean);
  }

  function downloadBlob(filename, text, type) {
    if (!window.URL || typeof window.URL.createObjectURL !== "function") {
      console.warn("Blob export unsupported in this environment");
      announceConnection("Export non supporté dans ce navigateur.", "danger");
      return;
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
  }

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
        lines.push(`*Horodatage :* ${item.timestamp}`);
      }
      if (item.metadata && Object.keys(item.metadata).length > 0) {
        Object.entries(item.metadata).forEach(([key, value]) => {
          lines.push(`*${key} :* ${value}`);
        });
      }
      lines.push("");
      lines.push(item.text || "");
      lines.push("");
    });
    return lines.join("\n");
  }

  async function exportConversation(format) {
    const items = collectTranscript();
    if (!items.length) {
      announceConnection("Aucun message à exporter.", "warning");
      return;
    }
    if (format === "json") {
      const payload = {
        exported_at: nowISO(),
        count: items.length,
        items,
      };
      downloadBlob(
        buildExportFilename("json"),
        JSON.stringify(payload, null, 2),
        "application/json",
      );
      announceConnection("Export JSON généré.", "success");
      return;
    }
    if (format === "markdown") {
      downloadBlob(
        buildExportFilename("md"),
        buildMarkdownExport(items),
        "text/markdown",
      );
      announceConnection("Export Markdown généré.", "success");
      return;
    }
  }

  async function copyConversationToClipboard() {
    const items = collectTranscript();
    if (!items.length) {
      announceConnection("Aucun message à copier.", "warning");
      return;
    }
    const text = buildMarkdownExport(items);
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
      announceConnection("Conversation copiée au presse-papiers.", "success");
    } catch (err) {
      console.warn("Copy conversation failed", err);
      announceConnection("Impossible de copier la conversation.", "danger");
    }
  }

  function renderMarkdown(text) {
    if (text == null) {
      return "";
    }
    const value = String(text);
    try {
      if (window.marked && typeof window.marked.parse === "function") {
        const rendered = window.marked.parse(value);
        if (
          window.DOMPurify &&
          typeof window.DOMPurify.sanitize === "function"
        ) {
          return window.DOMPurify.sanitize(rendered, {
            ALLOW_UNKNOWN_PROTOCOLS: false,
            USE_PROFILES: { html: true },
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

  function setBusy(busy) {
    els.transcript.setAttribute("aria-busy", busy ? "true" : "false");
    if (els.send) {
      els.send.disabled = Boolean(busy);
      els.send.setAttribute("aria-busy", busy ? "true" : "false");
      if (busy) {
        els.send.innerHTML = sendBusyMarkup;
      } else if (sendIdleMarkup) {
        els.send.innerHTML = sendIdleMarkup;
      } else {
        els.send.textContent = sendIdleLabel;
      }
    }
  }

  function hideError() {
    if (!els.errorAlert) return;
    els.errorAlert.classList.add("d-none");
    if (els.errorMessage) {
      els.errorMessage.textContent = "";
    }
  }

  function showError(message) {
    if (!els.errorAlert || !els.errorMessage) return;
    els.errorMessage.textContent = message;
    els.errorAlert.classList.remove("d-none");
  }

  function setComposerStatus(message, tone = "muted") {
    if (!els.composerStatus) return;
    const tones = ["muted", "info", "success", "danger", "warning"];
    els.composerStatus.textContent = message;
    tones.forEach((t) => els.composerStatus.classList.remove(`text-${t}`));
    els.composerStatus.classList.add(`text-${tone}`);
  }

  function setComposerStatusIdle() {
    setComposerStatus(composerStatusDefault, "muted");
  }

  function scheduleComposerIdle(delay = 3500) {
    if (resetStatusTimer) {
      clearTimeout(resetStatusTimer);
    }
    resetStatusTimer = window.setTimeout(() => {
      setComposerStatusIdle();
    }, delay);
  }

  function updatePromptMetrics() {
    if (!els.promptCount) return;
    const value = els.prompt.value || "";
    if (promptMax) {
      els.promptCount.textContent = `${value.length} / ${promptMax}`;
    } else {
      els.promptCount.textContent = `${value.length}`;
    }
    els.promptCount.classList.remove("text-warning", "text-danger");
    if (promptMax) {
      const remaining = promptMax - value.length;
      if (remaining <= 5) {
        els.promptCount.classList.add("text-danger");
      } else if (remaining <= 20) {
        els.promptCount.classList.add("text-warning");
      }
    }
  }

  function autosizePrompt() {
    if (!els.prompt) return;
    els.prompt.style.height = "auto";
    const nextHeight = Math.min(els.prompt.scrollHeight, PROMPT_MAX_HEIGHT);
    els.prompt.style.height = `${nextHeight}px`;
  }

  function isAtBottom() {
    if (!els.transcript) return true;
    const distance =
      els.transcript.scrollHeight -
      (els.transcript.scrollTop + els.transcript.clientHeight);
    return distance <= SCROLL_THRESHOLD;
  }

  function scrollToBottom(options = {}) {
    if (!els.transcript) return;
    const smooth = options.smooth !== false && !prefersReducedMotion;
    els.transcript.scrollTo({
      top: els.transcript.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
    hideScrollButton();
  }

  function showScrollButton() {
    if (!els.scrollBottom) return;
    if (hideScrollTimer) {
      clearTimeout(hideScrollTimer);
      hideScrollTimer = null;
    }
    els.scrollBottom.classList.remove("d-none");
    els.scrollBottom.classList.add("is-visible");
    els.scrollBottom.setAttribute("aria-hidden", "false");
  }

  function hideScrollButton() {
    if (!els.scrollBottom) return;
    els.scrollBottom.classList.remove("is-visible");
    els.scrollBottom.setAttribute("aria-hidden", "true");
    hideScrollTimer = window.setTimeout(() => {
      if (els.scrollBottom) {
        els.scrollBottom.classList.add("d-none");
      }
    }, 200);
  }

  function htmlToText(html) {
    const container = document.createElement("div");
    container.innerHTML = html;
    return container.textContent || "";
  }

  function extractBubbleText(bubble) {
    const clone = bubble.cloneNode(true);
    clone
      .querySelectorAll(".copy-btn, .chat-meta")
      .forEach((node) => node.remove());
    return clone.textContent.trim();
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
    if (!row || bootstrapping || role === "system") {
      return;
    }
    row.classList.add("chat-row-highlight");
    window.setTimeout(() => {
      row.classList.remove("chat-row-highlight");
    }, 600);
  }

  function updateConnectionMeta(message, tone = "muted") {
    if (!els.connectionMeta) return;
    const tones = ["muted", "info", "success", "danger", "warning"];
    els.connectionMeta.textContent = message;
    tones.forEach((t) => els.connectionMeta.classList.remove(`text-${t}`));
    els.connectionMeta.classList.add(`text-${tone}`);
  }

  function updateDiagnosticField(el, value) {
    if (!el) return;
    el.textContent = value || "—";
  }

  function setDiagnostics(patch) {
    Object.assign(diagnostics, patch);
    if (Object.prototype.hasOwnProperty.call(patch, "connectedAt")) {
      updateDiagnosticField(
        els.diagConnected,
        diagnostics.connectedAt
          ? formatTimestamp(diagnostics.connectedAt)
          : "—",
      );
    }
    if (Object.prototype.hasOwnProperty.call(patch, "lastMessageAt")) {
      updateDiagnosticField(
        els.diagLastMessage,
        diagnostics.lastMessageAt
          ? formatTimestamp(diagnostics.lastMessageAt)
          : "—",
      );
    }
    if (Object.prototype.hasOwnProperty.call(patch, "latencyMs")) {
      if (typeof diagnostics.latencyMs === "number") {
        updateDiagnosticField(
          els.diagLatency,
          `${Math.max(0, Math.round(diagnostics.latencyMs))} ms`,
        );
      } else {
        updateDiagnosticField(els.diagLatency, "—");
      }
    }
  }

  function updateNetworkStatus() {
    if (!els.diagNetwork) return;
    const online = navigator.onLine;
    els.diagNetwork.textContent = online ? "En ligne" : "Hors ligne";
    els.diagNetwork.classList.toggle("text-danger", !online);
    els.diagNetwork.classList.toggle("text-success", online);
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
    if (!els.transcript) return 0;
    const { preserveInput = false } = options;
    const rawQuery = typeof query === "string" ? query : "";
    if (!preserveInput && els.filterInput) {
      els.filterInput.value = rawQuery;
    }
    const trimmed = rawQuery.trim();
    activeFilter = trimmed;
    const normalized = normalizeString(trimmed);
    let matches = 0;
    const rows = Array.from(els.transcript.querySelectorAll(".chat-row"));
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
    els.transcript.classList.toggle("filtered", Boolean(trimmed));
    if (els.filterEmpty) {
      if (trimmed && matches === 0) {
        els.filterEmpty.classList.remove("d-none");
        els.filterEmpty.setAttribute(
          "aria-live",
          els.filterEmpty.getAttribute("aria-live") || "polite",
        );
      } else {
        els.filterEmpty.classList.add("d-none");
      }
    }
    if (els.filterHint) {
      if (trimmed) {
        let summary = "Aucun message ne correspond.";
        if (matches === 1) {
          summary = "1 message correspond.";
        } else if (matches > 1) {
          summary = `${matches} messages correspondent.`;
        }
        els.filterHint.textContent = summary;
      } else {
        els.filterHint.textContent = filterHintDefault;
      }
    }
    return matches;
  }

  function reapplyTranscriptFilter() {
    if (activeFilter) {
      applyTranscriptFilter(activeFilter, { preserveInput: true });
    } else if (els.transcript) {
      els.transcript.classList.remove("filtered");
      const rows = Array.from(els.transcript.querySelectorAll(".chat-row"));
      rows.forEach((row) => {
        row.classList.remove("chat-hidden", "chat-filter-match");
      });
      if (els.filterEmpty) {
        els.filterEmpty.classList.add("d-none");
      }
      if (els.filterHint) {
        els.filterHint.textContent = filterHintDefault;
      }
    }
  }

  function clearTranscriptFilter(focus = true) {
    activeFilter = "";
    if (els.filterInput) {
      els.filterInput.value = "";
    }
    reapplyTranscriptFilter();
    if (focus && els.filterInput) {
      els.filterInput.focus();
    }
  }

  function line(role, html, options = {}) {
    const shouldStick = isAtBottom();
    const row = document.createElement("div");
    row.className = `chat-row chat-${role}`;
    row.innerHTML = html;
    row.dataset.role = role;
    row.dataset.rawText = options.rawText || "";
    row.dataset.timestamp = options.timestamp || "";
    els.transcript.appendChild(row);
    decorateRow(row, role);
    if (options.register !== false) {
      const ts = options.timestamp || nowISO();
      const text =
        options.rawText && options.rawText.length > 0
          ? options.rawText
          : htmlToText(html);
      const id = registerTimelineEntry({
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
      row.dataset.messageId = makeMessageId();
    }
    if (shouldStick) {
      scrollToBottom({ smooth: !bootstrapping });
    } else {
      showScrollButton();
    }
    highlightRow(row, role);
    if (activeFilter) {
      applyTranscriptFilter(activeFilter, { preserveInput: true });
    }
    return row;
  }

  function escapeHTML(str) {
    return String(str).replace(
      /[&<>"']/g,
      (ch) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[ch],
    );
  }

  function formatTimestamp(ts) {
    if (!ts) return "";
    try {
      return new Date(ts).toLocaleString("fr-CA");
    } catch (err) {
      return String(ts);
    }
  }

  function renderHistory(entries, options = {}) {
    const { replace = false } = options;
    if (!Array.isArray(entries) || entries.length === 0) {
      if (replace) {
        els.transcript.innerHTML = "";
        historyBootstrapped = false;
        hideScrollButton();
        timelineMap.clear();
        timelineOrder.length = 0;
      }
      return;
    }
    if (replace) {
      els.transcript.innerHTML = "";
      historyBootstrapped = false;
      streamRow = null;
      streamBuf = "";
      timelineMap.clear();
      timelineOrder.length = 0;
    }
    if (historyBootstrapped && !replace) {
      return;
    }
    bootstrapping = true;
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
    bootstrapping = false;
    historyBootstrapped = true;
    scrollToBottom({ smooth: false });
    hideScrollButton();
  }

  setDiagnostics({ connectedAt: null, lastMessageAt: null, latencyMs: null });
  renderHistory(chatHistory);
  updatePromptMetrics();
  autosizePrompt();
  setComposerStatusIdle();

  // Streaming buffer for the current assistant message
  let streamRow = null;
  let streamBuf = "";
  let streamMessageId = null;

  function startStream() {
    streamBuf = "";
    const ts = nowISO();
    streamMessageId = makeMessageId();
    streamRow = line(
      "assistant",
      '<div class="chat-bubble"><span class="chat-cursor">▍</span></div>',
      {
        rawText: "",
        timestamp: ts,
        messageId: streamMessageId,
        metadata: { streaming: true },
      },
    );
    setDiagnostics({ lastMessageAt: ts });
    if (resetStatusTimer) {
      clearTimeout(resetStatusTimer);
    }
    setComposerStatus("Réponse en cours…", "info");
  }

  function appendStream(delta) {
    if (!streamRow) {
      startStream();
    }
    const shouldStick = isAtBottom();
    streamBuf += delta || "";
    const bubble = streamRow.querySelector(".chat-bubble");
    if (bubble) {
      bubble.innerHTML = `${renderMarkdown(streamBuf)}<span class="chat-cursor">▍</span>`;
    }
    if (streamMessageId) {
      updateTimelineEntry(streamMessageId, {
        text: streamBuf,
        metadata: { streaming: true },
      });
    }
    setDiagnostics({ lastMessageAt: nowISO() });
    if (shouldStick) {
      scrollToBottom({ smooth: false });
    }
  }

  function endStream(data) {
    if (!streamRow) {
      return;
    }
    const bubble = streamRow.querySelector(".chat-bubble");
    if (bubble) {
      bubble.innerHTML = renderMarkdown(streamBuf);
      const meta = document.createElement("div");
      meta.className = "chat-meta";
      const ts = data && data.timestamp ? data.timestamp : nowISO();
      meta.textContent = formatTimestamp(ts);
      if (data && data.error) {
        meta.classList.add("text-danger");
        meta.textContent = `${meta.textContent} • ${data.error}`;
      }
      bubble.appendChild(meta);
      decorateRow(streamRow, "assistant");
      highlightRow(streamRow, "assistant");
      if (isAtBottom()) {
        scrollToBottom({ smooth: true });
      } else {
        showScrollButton();
      }
      if (streamMessageId) {
        updateTimelineEntry(streamMessageId, {
          text: streamBuf,
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
    streamRow = null;
    streamBuf = "";
    streamMessageId = null;
  }

  function announceConnection(message, variant = "info") {
    if (!els.connection) {
      return;
    }
    const classList = els.connection.classList;
    Array.from(classList)
      .filter((cls) => cls.startsWith("alert-") && cls !== "alert")
      .forEach((cls) => classList.remove(cls));
    classList.add("alert");
    classList.add(`alert-${variant}`);
    els.connection.textContent = message;
    classList.remove("visually-hidden");
    window.setTimeout(() => {
      classList.add("visually-hidden");
    }, 4000);
  }

  // ---- WS ticket + socket -------------------------------------------------
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

  async function fetchTicket() {
    const jwt = await getJwt();
    const resp = await fetch(apiUrl("/api/v1/auth/ws/ticket"), {
      method: "POST",
      headers: { Authorization: `Bearer ${jwt}` },
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

  let ws;
  let wsHBeat;
  let reconnectBackoff = 500; // ms
  const BACKOFF_MAX = 8000;

  function setWsStatus(state, title) {
    if (!els.wsStatus) return;
    const label = statusLabels[state] || state;
    els.wsStatus.textContent = label;
    els.wsStatus.className = `badge ws-badge ${state}`;
    if (title) {
      els.wsStatus.title = title;
    } else {
      els.wsStatus.removeAttribute("title");
    }
  }

  async function openSocket() {
    try {
      updateConnectionMeta("Obtention d’un ticket de connexion…", "info");
      const ticket = await fetchTicket();
      const wsUrl = new URL("/ws/chat/", baseUrl);
      wsUrl.protocol = baseUrl.protocol === "https:" ? "wss:" : "ws:";
      wsUrl.searchParams.set("t", ticket);

      ws = new WebSocket(wsUrl.toString());
      setWsStatus("connecting");
      updateConnectionMeta("Connexion au serveur…", "info");

      ws.onopen = () => {
        setWsStatus("online");
        const connectedAt = nowISO();
        updateConnectionMeta(
          `Connecté le ${formatTimestamp(connectedAt)}`,
          "success",
        );
        setDiagnostics({ connectedAt, lastMessageAt: connectedAt });
        hideError();
        wsHBeat = window.setInterval(() => {
          safeSend({ type: "client.ping", ts: nowISO() });
        }, 20000);
        reconnectBackoff = 500;
        setComposerStatus("Connecté. Vous pouvez échanger.", "success");
        scheduleComposerIdle(4000);
      };

      ws.onmessage = (evt) => {
        try {
          const ev = JSON.parse(evt.data);
          handleEvent(ev);
        } catch (err) {
          console.error("Bad event payload", err, evt.data);
        }
      };

      ws.onclose = () => {
        setWsStatus("offline");
        if (wsHBeat) {
          clearInterval(wsHBeat);
        }
        setDiagnostics({ latencyMs: undefined });
        const delay = reconnectBackoff + Math.floor(Math.random() * 250);
        const seconds = Math.max(1, Math.round(delay / 1000));
        updateConnectionMeta(
          `Déconnecté. Nouvelle tentative dans ${seconds} s…`,
          "warning",
        );
        setComposerStatus(
          "Connexion perdue. Reconnexion automatique…",
          "warning",
        );
        scheduleComposerIdle(6000);
        reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
        window.setTimeout(openSocket, delay);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        setWsStatus("error", "Erreur WebSocket");
        updateConnectionMeta("Erreur WebSocket détectée.", "danger");
        setComposerStatus("Une erreur réseau est survenue.", "danger");
        scheduleComposerIdle(6000);
      };
    } catch (err) {
      console.error(err);
      const message = err instanceof Error ? err.message : String(err);
      setWsStatus("error", message);
      updateConnectionMeta(message, "danger");
      setComposerStatus(
        "Connexion indisponible. Nouvel essai bientôt.",
        "danger",
      );
      scheduleComposerIdle(6000);
      const delay = Math.min(BACKOFF_MAX, reconnectBackoff);
      reconnectBackoff = Math.min(BACKOFF_MAX, reconnectBackoff * 2);
      window.setTimeout(openSocket, delay);
    }
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

  // ---- Typed event router -------------------------------------------------
  function handleEvent(ev) {
    const type = ev && ev.type ? ev.type : "";
    const data = ev && ev.data ? ev.data : {};
    switch (type) {
      case "ws.connected": {
        if (data && data.origin) {
          announceConnection(`Connecté via ${data.origin}`);
          updateConnectionMeta(`Connecté via ${data.origin}`, "success");
        } else {
          announceConnection("Connecté au serveur.");
          updateConnectionMeta("Connecté au serveur.", "success");
        }
        scheduleComposerIdle(4000);
        break;
      }
      case "history.snapshot": {
        if (data && Array.isArray(data.items)) {
          renderHistory(data.items, { replace: true });
        }
        break;
      }
      case "ai_model.response_chunk": {
        const delta =
          typeof data.delta === "string" ? data.delta : data.text || "";
        appendStream(delta);
        break;
      }
      case "ai_model.response_complete": {
        if (data && data.text && !streamBuf) {
          appendStream(data.text);
        }
        endStream(data);
        setBusy(false);
        if (data && typeof data.latency_ms !== "undefined") {
          setDiagnostics({ latencyMs: Number(data.latency_ms) });
        }
        if (data && data.ok === false && data.error) {
          appendMessage("system", data.error, {
            variant: "error",
            allowMarkdown: false,
            metadata: { event: type },
          });
        }
        break;
      }
      case "chat.message": {
        if (!streamRow) {
          startStream();
        }
        if (data && typeof data.response === "string" && !streamBuf) {
          appendStream(data.response);
        }
        endStream(data);
        setBusy(false);
        break;
      }
      case "evolution_engine.training_complete": {
        appendMessage(
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
        appendMessage(
          "system",
          `Échec de l'évolution : ${
            data && data.error ? data.error : "inconnu"
          }`,
          {
            variant: "error",
            allowMarkdown: false,
            metadata: { event: type },
          },
        );
        break;
      }
      case "sleep_time_compute.phase_start": {
        appendMessage("system", "Optimisation en arrière-plan démarrée…", {
          variant: "hint",
          allowMarkdown: false,
          metadata: { event: type },
        });
        break;
      }
      case "sleep_time_compute.creative_phase": {
        appendMessage(
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
        appendMessage("system", `Perf : ${formatPerf(data)}`, {
          variant: "warn",
          allowMarkdown: false,
          metadata: { event: type },
        });
        if (data && typeof data.ttfb_ms !== "undefined") {
          setDiagnostics({ latencyMs: Number(data.ttfb_ms) });
        }
        break;
      }
      case "ui.suggestions": {
        applyQuickActionOrdering(
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

  function applyQuickActionOrdering(suggestions) {
    if (!els.quickActions) return;
    if (!Array.isArray(suggestions) || suggestions.length === 0) return;
    const buttons = Array.from(els.quickActions.querySelectorAll("button.qa"));
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
    els.quickActions.innerHTML = "";
    els.quickActions.appendChild(frag);
  }

  // ---- Debounced AUI suggestions -----------------------------------------
  let auiTimer = null;

  async function fetchSuggestionsDebounced() {
    if (auiTimer) {
      clearTimeout(auiTimer);
    }
    auiTimer = window.setTimeout(fetchSuggestions, 220);
  }

  async function fetchSuggestions() {
    const text = (els.prompt.value || "").trim();
    if (!text) {
      return;
    }
    if (els.send && els.send.disabled) {
      return;
    }
    if (text.length < 3) {
      return;
    }
    try {
      const jwt = await getJwt();
      const resp = await fetch(apiUrl("/api/v1/ui/suggestions"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${jwt}`,
        },
        body: JSON.stringify({
          prompt: text,
          actions: ["code", "summarize", "explain"],
        }),
      });
      if (!resp.ok) {
        return;
      }
      const payload = await resp.json();
      if (payload && Array.isArray(payload.actions)) {
        applyQuickActionOrdering(payload.actions);
      }
    } catch (err) {
      console.debug("AUI suggestion fetch failed", err);
    }
  }

  // ---- Submit & quick actions --------------------------------------------
  els.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = (els.prompt.value || "").trim();
    if (!text) {
      setComposerStatus("Saisissez un message avant d’envoyer.", "warning");
      scheduleComposerIdle(4000);
      return;
    }
    hideError();
    const submittedAt = nowISO();
    appendMessage("user", text, {
      timestamp: submittedAt,
      metadata: { submitted: true },
    });
    els.prompt.value = "";
    updatePromptMetrics();
    autosizePrompt();
    setComposerStatus("Message envoyé…", "info");
    scheduleComposerIdle(4000);
    setBusy(true);
    applyQuickActionOrdering(["code", "summarize", "explain"]);

    try {
      const jwt = await getJwt();
      const resp = await fetch(apiUrl("/api/v1/conversation/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${jwt}`,
        },
        body: JSON.stringify({ message: text }),
      });
      if (!resp.ok) {
        const payload = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${payload}`);
      }
      startStream();
      if (els.prompt) {
        els.prompt.focus();
      }
    } catch (err) {
      setBusy(false);
      showError(String(err));
      appendMessage("system", String(err), {
        variant: "error",
        allowMarkdown: false,
        metadata: { stage: "submit" },
      });
      setComposerStatus("Envoi impossible. Vérifiez la connexion.", "danger");
      scheduleComposerIdle(6000);
    }
  });

  if (els.quickActions) {
    els.quickActions.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLButtonElement)) {
        return;
      }
      const action = target.dataset.action;
      if (!action) {
        return;
      }
      const presets = {
        code: "Je souhaite écrire du code…",
        summarize: "Résume la dernière conversation.",
        explain: "Explique ta dernière réponse plus simplement.",
      };
      els.prompt.value = presets[action] || action;
      updatePromptMetrics();
      autosizePrompt();
      setComposerStatus("Suggestion envoyée…", "info");
      scheduleComposerIdle(4000);
      if (typeof els.composer.requestSubmit === "function") {
        els.composer.requestSubmit();
      } else {
        els.composer.dispatchEvent(new Event("submit"));
      }
    });
  }

  if (els.filterInput) {
    els.filterInput.addEventListener("input", (event) => {
      applyTranscriptFilter(event.target.value || "", { preserveInput: true });
    });
    els.filterInput.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        clearTranscriptFilter();
      }
    });
  }

  if (els.filterClear) {
    els.filterClear.addEventListener("click", () => {
      clearTranscriptFilter();
    });
  }

  if (els.exportJson) {
    els.exportJson.addEventListener("click", () => {
      exportConversation("json");
    });
  }

  if (els.exportMarkdown) {
    els.exportMarkdown.addEventListener("click", () => {
      exportConversation("markdown");
    });
  }

  if (els.exportCopy) {
    els.exportCopy.addEventListener("click", () => {
      copyConversationToClipboard();
    });
  }

  if (els.prompt) {
    els.prompt.addEventListener("input", (event) => {
      updatePromptMetrics();
      autosizePrompt();
      const value = event.target.value || "";
      if (!value.trim()) {
        setComposerStatusIdle();
      }
      fetchSuggestionsDebounced();
    });
    els.prompt.addEventListener("paste", () => {
      window.setTimeout(() => {
        updatePromptMetrics();
        autosizePrompt();
        fetchSuggestionsDebounced();
      }, 0);
    });
    els.prompt.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        if (typeof els.composer.requestSubmit === "function") {
          els.composer.requestSubmit();
        } else {
          els.composer.dispatchEvent(new Event("submit"));
        }
      }
    });
    els.prompt.addEventListener("focus", () => {
      setComposerStatus(
        "Rédigez votre message, puis Ctrl+Entrée pour l'envoyer.",
        "info",
      );
      scheduleComposerIdle(4000);
    });
  }

  if (els.transcript) {
    els.transcript.addEventListener("scroll", () => {
      if (isAtBottom()) {
        hideScrollButton();
      } else {
        showScrollButton();
      }
    });
  }

  if (els.scrollBottom) {
    els.scrollBottom.addEventListener("click", () => {
      scrollToBottom({ smooth: true });
      if (els.prompt) {
        els.prompt.focus();
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

  // ---- Dark mode toggle ---------------------------------------------------
  const darkModeKey = "dark-mode";
  const toggleBtn = document.getElementById("toggle-dark-mode");

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

  // ---- Boot ---------------------------------------------------------------
  openSocket();
})();
