import {
  describe,
  expect,
  it,
  beforeEach,
  afterEach,
  jest,
} from "@jest/globals";
import { createSuggestionService } from "../suggestions.js";

describe("createSuggestionService", () => {
  let http;
  let ui;
  let service;

  beforeEach(() => {
    jest.useFakeTimers();
    http = {
      postSuggestions: jest.fn().mockResolvedValue({ actions: ["code", "explain"] }),
    };
    ui = {
      applyQuickActionOrdering: jest.fn(),
    };
    service = createSuggestionService({ http, ui });
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it("does not fetch suggestions for short prompts", async () => {
    service.schedule("hi");

    jest.advanceTimersByTime(500);
    await Promise.resolve();
    await Promise.resolve();

    expect(http.postSuggestions).not.toHaveBeenCalled();
    expect(ui.applyQuickActionOrdering).not.toHaveBeenCalled();
  });

  it("trims prompts and forwards them to the HTTP service", async () => {
    service.schedule("   explore  ");

    jest.advanceTimersByTime(220);
    await Promise.resolve();
    await Promise.resolve();

    expect(http.postSuggestions).toHaveBeenCalledTimes(1);
    expect(http.postSuggestions).toHaveBeenCalledWith("explore");
    expect(ui.applyQuickActionOrdering).toHaveBeenCalledWith(["code", "explain"]);
  });

  it("debounces rapid scheduling calls and only runs the latest prompt", async () => {
    service.schedule("first prompt");
    jest.advanceTimersByTime(100);
    service.schedule("second prompt");

    jest.advanceTimersByTime(220);
    await Promise.resolve();
    await Promise.resolve();

    expect(http.postSuggestions).toHaveBeenCalledTimes(1);
    expect(http.postSuggestions).toHaveBeenCalledWith("second prompt");
  });

  it("logs debug information when the HTTP call fails", async () => {
    const error = new Error("offline");
    http.postSuggestions.mockRejectedValueOnce(error);
    const debugSpy = jest.spyOn(console, "debug").mockImplementation(() => {});

    service.schedule("failing prompt");
    jest.advanceTimersByTime(220);
    await Promise.resolve();
    await Promise.resolve();

    expect(http.postSuggestions).toHaveBeenCalledTimes(1);
    expect(debugSpy).toHaveBeenCalled();
    const [message, receivedError] = debugSpy.mock.calls[0];
    expect(message).toBe("AUI suggestion fetch failed");
    expect(receivedError).toBe(error);

    debugSpy.mockRestore();
  });
});
