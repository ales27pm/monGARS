import { describe, expect, it, jest } from "@jest/globals";
import { ChatApp } from "../app.js";

describe("ChatApp.handleSocketEvent", () => {
  it("unwraps websocket message envelopes before dispatching", () => {
    const ui = {
      applyQuickActionOrdering: jest.fn(),
    };
    const ctx = {
      ui,
      handleSocketEvent(ev) {
        return ChatApp.prototype.handleSocketEvent.call(this, ev);
      },
    };

    ChatApp.prototype.handleSocketEvent.call(ctx, {
      type: "message",
      message: {
        type: "ui.suggestions",
        data: { actions: ["code", "explain"] },
      },
    });

    expect(ui.applyQuickActionOrdering).toHaveBeenCalledWith([
      "code",
      "explain",
    ]);
  });
});
