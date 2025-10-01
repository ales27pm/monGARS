/**
 * Entry point for the chat application.
 * Expects window.chatConfig to be defined by the server-rendered template.
 * Falls back to an empty config if not present.
 * Required config shape: { apiUrl?, wsUrl?, token?, ... }
 */
import { ChatApp } from "./app.js";

new ChatApp(document, window.chatConfig || {});
