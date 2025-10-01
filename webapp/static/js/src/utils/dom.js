export function escapeHTML(str) {
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

export function htmlToText(html) {
  const container = document.createElement("div");
  container.innerHTML = html;
  return container.textContent || "";
}

export function extractBubbleText(bubble) {
  const clone = bubble.cloneNode(true);
  clone
    .querySelectorAll(".copy-btn, .chat-meta")
    .forEach((node) => node.remove());
  return clone.textContent.trim();
}
