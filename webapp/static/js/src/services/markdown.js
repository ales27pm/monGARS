import { escapeHTML } from "../utils/dom.js";

export function renderMarkdown(text) {
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
          USE_PROFILES: { html: true },
        });
      }
      return fallback();
    }
  } catch (err) {
    console.warn("Markdown rendering failed", err);
  }
  return fallback();
}
