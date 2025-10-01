export function createSuggestionService({ http, ui }) {
  let timer = null;

  function schedule(prompt) {
    if (timer) {
      clearTimeout(timer);
    }
    timer = window.setTimeout(() => fetchSuggestions(prompt), 220);
  }

  async function fetchSuggestions(prompt) {
    if (!prompt || prompt.trim().length < 3) {
      return;
    }
    try {
      const payload = await http.postSuggestions(prompt.trim());
      if (payload && Array.isArray(payload.actions)) {
        ui.applyQuickActionOrdering(payload.actions);
      }
    } catch (err) {
      console.debug("AUI suggestion fetch failed", err);
    }
  }

  return {
    schedule,
  };
}
