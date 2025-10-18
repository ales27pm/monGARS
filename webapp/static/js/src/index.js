/**
 * Entry point for the chat application.
 * Expects window.chatConfig to be defined by the server-rendered template.
 * Falls back to an empty config if not present.
 * Required config shape: { apiUrl?, wsUrl?, token?, ... }
 */
import { ChatApp } from "./app.js";
import { renderUserListPage } from "./ui/adminUserList.js";
import { renderChangePasswordPage } from "./ui/adminChangePassword.js";

const rawConfig = window.chatConfig || {};
const pathname = window.location?.pathname || "/";
const normalisedPath =
  pathname && pathname !== "/" ? pathname.replace(/\/+$/, "") || "/" : "/";

if (normalisedPath.startsWith("/user/list")) {
  renderUserListPage({ doc: document, rawConfig });
} else if (normalisedPath.startsWith("/user/change-password")) {
  renderChangePasswordPage({ doc: document, rawConfig });
} else {
  new ChatApp(document, rawConfig);
}
