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
  const SUPPORTED_TONES = ["muted", "info", "success", "danger", "warning"];
  const composerStatusDefault =
    (elements.composerStatus && elements.composerStatus.textContent.trim()) ||
    "Appuyez sur Ctrl+Entrée pour envoyer rapidement.";
  const filterHintDefault =
    (elements.filterHint && elements.filterHint.textContent.trim()) ||
    "Utilisez le filtre pour limiter l'historique. Appuyez sur Échap pour effacer.";
  const voiceStatusDefault =
    (elements.voiceStatus && elements.voiceStatus.textContent.trim()) ||
    "Vérification des capacités vocales…";
  const promptMax = Number(elements.prompt?.getAttribute("maxlength")) || null;
  const prefersReducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const SCROLL_THRESHOLD = 140;
  const PROMPT_MAX_HEIGHT = 320;
  const composerStatusEmbedding =
    (elements.composerStatus &&
      elements.composerStatus.getAttribute("data-embed-label")) ||
    "Mode Embedding : générez des vecteurs pour vos textes.";
  const promptPlaceholderDefault =
    (elements.prompt && elements.prompt.getAttribute("placeholder")) || "";
  const promptPlaceholderEmbedding =
    (elements.prompt &&
      elements.prompt.getAttribute("data-embed-placeholder")) ||
    "Entrez le texte à encoder…";
  const promptAriaDefault =
    (elements.prompt && elements.prompt.getAttribute("aria-label")) || "";
  const promptAriaEmbedding =
    (elements.prompt &&
      elements.prompt.getAttribute("data-embed-aria-label")) ||
    "Texte à encoder";
  const sendAriaDefault =
    (elements.send && elements.send.getAttribute("aria-label")) ||
    sendIdleLabel;
  const sendAriaEmbedding =
    (elements.send && elements.send.getAttribute("data-embed-aria-label")) ||
    "Générer un embedding";

  const diagnostics = {
    connectedAt: null,
    lastMessageAt: null,
    latencyMs: null,
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

  function normaliseMode(mode) {
    return mode === "embed" ? "embed" : "chat";
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
    elements.composerStatus.textContent = message;
    SUPPORTED_TONES.forEach((t) =>
      elements.composerStatus.classList.remove(`text-${t}`),
    );
    elements.composerStatus.classList.add(`text-${tone}`);
  }

  function setComposerStatusIdle() {
    const message =
      state.mode === "embed" ? composerStatusEmbedding : composerStatusDefault;
    setComposerStatus(message, "muted");
  }

  function scheduleComposerIdle(delay = 3500) {
    if (state.resetStatusTimer) {
      clearTimeout(state.resetStatusTimer);
    }
    state.resetStatusTimer = window.setTimeout(() => {
      setComposerStatusIdle();
    }, delay);
  }

  function setVoiceStatus(message, tone = "muted") {
    if (!elements.voiceStatus) return;
    if (state.voiceStatusTimer) {
      clearTimeout(state.voiceStatusTimer);
      state.voiceStatusTimer = null;
    }
    elements.voiceStatus.textContent = message;
    SUPPORTED_TONES.forEach((t) =>
      elements.voiceStatus.classList.remove(`text-${t}`),
    );
    elements.voiceStatus.classList.add(`text-${tone}`);
  }

  function scheduleVoiceStatusIdle(delay = 4000) {
    if (!elements.voiceStatus) return;
    if (state.voiceStatusTimer) {
      clearTimeout(state.voiceStatusTimer);
    }
    state.voiceStatusTimer = window.setTimeout(() => {
      setVoiceStatus(voiceStatusDefault, "muted");
      state.voiceStatusTimer = null;
    }, delay);
  }

  function setVoiceAvailability({
    recognition = false,
    synthesis = false,
  } = {}) {
    if (elements.voiceControls) {
      elements.voiceControls.classList.toggle(
        "d-none",
        !recognition && !synthesis,
      );
    }
    if (elements.voiceRecognitionGroup) {
      elements.voiceRecognitionGroup.classList.toggle("d-none", !recognition);
    }
    if (elements.voiceToggle) {
      elements.voiceToggle.disabled = !recognition;
      elements.voiceToggle.setAttribute(
        "title",
        recognition
          ? "Activer ou désactiver la dictée vocale."
          : "Dictée vocale indisponible dans ce navigateur.",
      );
      elements.voiceToggle.setAttribute("aria-pressed", "false");
      elements.voiceToggle.classList.remove("btn-danger");
      elements.voiceToggle.classList.add("btn-outline-secondary");
      elements.voiceToggle.textContent = "🎙️ Activer la dictée";
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
  }

  function setVoiceListening(listening) {
    if (!elements.voiceToggle) return;
    elements.voiceToggle.setAttribute(
      "aria-pressed",
      listening ? "true" : "false",
    );
    elements.voiceToggle.classList.toggle("btn-danger", listening);
    elements.voiceToggle.classList.toggle("btn-outline-secondary", !listening);
    elements.voiceToggle.textContent = listening
      ? "🛑 Arrêter l'écoute"
      : "🎙️ Activer la dictée";
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
    placeholder.textContent = voices.length
      ? "Voix par défaut du système"
      : "Aucune voix disponible";
    frag.appendChild(placeholder);
    voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.voiceURI || voice.name || "";
      const bits = [voice.name || voice.voiceURI || "Voix"];
      if (voice.lang) {
        bits.push(`(${voice.lang})`);
      }
      if (voice.default) {
        bits.push("• défaut");
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
      const placeholder =
        next === "embed"
          ? promptPlaceholderEmbedding
          : promptPlaceholderDefault;
      if (placeholder) {
        elements.prompt.setAttribute("placeholder", placeholder);
      } else {
        elements.prompt.removeAttribute("placeholder");
      }
      const ariaLabel =
        next === "embed" ? promptAriaEmbedding : promptAriaDefault;
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

  function normaliseNumeric(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatNumeric(value) {
    if (!Number.isFinite(value)) {
      return "—";
    }
    if (value === 0) {
      return "0";
    }
    const abs = Math.abs(value);
    if (abs >= 1000 || abs < 0.001) {
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
      preview,
    };
  }

  function createVectorStatsTable(stats) {
    const table = document.createElement("table");
    table.className =
      "table table-sm table-striped embedding-details-table mb-0";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    [
      "Vecteur",
      "Composantes",
      "Magnitude",
      "Moyenne",
      "Min",
      "Max",
      "Aperçu",
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
        stat.preview.length
          ? stat.preview.map((value) => formatNumeric(value)).join(", ")
          : "—",
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
    if (!row) {
      return;
    }
    const bubble = row.querySelector(".chat-bubble");
    if (!bubble) {
      return;
    }
    bubble
      .querySelectorAll(".embedding-details")
      .forEach((node) => node.remove());

    const vectors = Array.isArray(embeddingData.vectors)
      ? embeddingData.vectors.filter((vector) => Array.isArray(vector))
      : [];
    if (vectors.length === 0) {
      return;
    }

    const stats = vectors
      .map((vector, index) => summariseVector(vector, index))
      .filter((entry) => entry && entry.count >= 0);
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
    downloadBtn.textContent = "Télécharger le JSON";
    downloadBtn.addEventListener("click", () => {
      try {
        const payload =
          typeof embeddingData.raw === "object" && embeddingData.raw !== null
            ? embeddingData.raw
            : {
                backend: embeddingData.backend ?? metadata.backend ?? null,
                model: embeddingData.model ?? metadata.model ?? null,
                dims:
                  embeddingData.dims ??
                  metadata.dims ??
                  stats[0]?.count ??
                  null,
                normalised:
                  typeof embeddingData.normalised !== "undefined"
                    ? Boolean(embeddingData.normalised)
                    : Boolean(metadata.normalised),
                count: vectors.length,
                vectors,
              };
        const blob = new Blob([JSON.stringify(payload, null, 2)], {
          type: "application/json",
        });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        const slugSource = (
          embeddingData.model ||
          metadata.model ||
          "embedding"
        )
          .toString()
          .toLowerCase();
        const slug = slugSource
          .replace(/[^a-z0-9._-]+/g, "-")
          .replace(/^-+|-+$/g, "")
          .slice(0, 60);
        link.href = url;
        link.download = `embedding-${slug || "result"}-${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.setTimeout(() => {
          window.URL.revokeObjectURL(url);
        }, 1000);
      } catch (err) {
        console.warn("Unable to download embedding payload", err);
        announceConnection(
          "Impossible de télécharger le résultat d'embedding.",
          "danger",
        );
      }
    });
    header.appendChild(downloadBtn);
    cardBody.appendChild(header);

    const dimsCandidate = Number(embeddingData.dims ?? metadata.dims);
    const dims = Number.isFinite(dimsCandidate)
      ? Number(dimsCandidate)
      : Array.isArray(vectors[0])
        ? vectors[0].length
        : null;
    const validMagnitudeStats = stats.filter(
      (stat) =>
        typeof stat.magnitude === "number" && !Number.isNaN(stat.magnitude),
    );
    const totalMagnitude = validMagnitudeStats.reduce(
      (acc, stat) => acc + stat.magnitude,
      0,
    );
    const avgMagnitude =
      validMagnitudeStats.length > 0
        ? totalMagnitude / validMagnitudeStats.length
        : null;

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
        globalMin =
          globalMin === null ? stat.min : Math.min(globalMin, stat.min);
        globalMax =
          globalMax === null ? stat.max : Math.max(globalMax, stat.max);
      }
    });
    const aggregateMagnitude =
      componentCount > 0 ? Math.sqrt(componentSquares) : null;
    const aggregateMean =
      componentCount > 0 ? componentSum / componentCount : null;

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
      pushMeta("Modèle", String(embeddingData.model || metadata.model));
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
        typeof embeddingData.normalised !== "undefined"
          ? embeddingData.normalised
          : metadata.normalised,
      )
        ? "Oui"
        : "Non",
    );
    pushMeta("Magnitude moyenne", formatNumeric(avgMagnitude));
    pushMeta("Magnitude agrégée", formatNumeric(aggregateMagnitude));
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
      embeddingData,
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
    if (role === "assistant" && embeddingData) {
      attachEmbeddingDetails(row, embeddingData, metadata || {});
    }
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
    },
  };
}
