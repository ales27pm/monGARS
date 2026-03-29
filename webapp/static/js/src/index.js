/**
 * Entry point for the chat application.
 * Expects window.chatConfig to be defined by the server-rendered template.
 * Falls back to an empty config if not present.
 * Required config shape: { apiUrl?, wsUrl?, token?, ... }
 */
import { ChatApp } from "./app.js";
import { renderUserListPage } from "./ui/adminUserList.js";
import { renderChangePasswordPage } from "./ui/adminChangePassword.js";
import { createRoot } from "react-dom/client";
import { ModernChatApp } from "./ui/ModernChatApp.jsx";

const rawConfig = globalThis.chatConfig || {};
const pathname = globalThis.location?.pathname || "/";
const normalisedPath =
  pathname && pathname !== "/" ? pathname.replace(/\/+$/, "") || "/" : "/";
const isUserList = /^\/user\/list(?:$|\/)/.test(normalisedPath);
const isChangePassword = /^\/user\/change-password(?:$|\/)/.test(
  normalisedPath,
);
const params = new URLSearchParams(globalThis.location.search);
const isModernUi = params.get("ui") === "modern";

if (isModernUi && document.getElementById("chat")) {
  const container = document.getElementById("chat");
  container.innerHTML = "";
  const root = createRoot(container);
  root.render(<ModernChatApp />);
} else if (isUserList) {
  const mount = renderUserListPage({ doc: document, rawConfig });
  if (!mount) {
    new ChatApp(document, rawConfig);
  }
} else if (isChangePassword) {
  const mount = renderChangePasswordPage({ doc: document, rawConfig });
  if (!mount) {
    new ChatApp(document, rawConfig);
  }
} else {
  new ChatApp(document, rawConfig);
}
