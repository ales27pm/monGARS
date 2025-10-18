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
const isUserList = /^\/user\/list(?:$|\/)/.test(normalisedPath);
const isChangePassword = /^\/user\/change-password(?:$|\/)/.test(
  normalisedPath,
);

let rendered = null;
if (isUserList) {
  rendered = renderUserListPage({ doc: document, rawConfig });
} else if (isChangePassword) {
  rendered = renderChangePasswordPage({ doc: document, rawConfig });
}

if (!rendered) {
  new ChatApp(document, rawConfig);
}
