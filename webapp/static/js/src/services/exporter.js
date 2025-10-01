import { nowISO } from "../utils/time.js";

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

export function createExporter({ timelineStore, announce }) {
  function collectTranscript() {
    return timelineStore.collect();
  }

  async function exportConversation(format) {
    const items = collectTranscript();
    if (!items.length) {
      announce("Aucun message à exporter.", "warning");
      return;
    }
    if (format === "json") {
      const payload = {
        exported_at: nowISO(),
        count: items.length,
        items,
      };
      if (
        downloadBlob(
          buildExportFilename("json"),
          JSON.stringify(payload, null, 2),
          "application/json",
        )
      ) {
        announce("Export JSON généré.", "success");
      } else {
        announce("Export non supporté dans ce navigateur.", "danger");
      }
      return;
    }
    if (format === "markdown") {
      if (
        downloadBlob(
          buildExportFilename("md"),
          buildMarkdownExport(items),
          "text/markdown",
        )
      ) {
        announce("Export Markdown généré.", "success");
      } else {
        announce("Export non supporté dans ce navigateur.", "danger");
      }
    }
  }

  async function copyConversationToClipboard() {
    const items = collectTranscript();
    if (!items.length) {
      announce("Aucun message à copier.", "warning");
      return;
    }
    const text = buildMarkdownExport(items);
    if (await copyToClipboard(text)) {
      announce("Conversation copiée au presse-papiers.", "success");
    } else {
      announce("Impossible de copier la conversation.", "danger");
    }
  }

  return {
    exportConversation,
    copyConversationToClipboard,
  };
}
