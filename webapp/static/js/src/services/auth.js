const DEFAULT_STORAGE_KEY = "mongars_jwt";

function hasLocalStorage() {
  try {
    return typeof window !== "undefined" && Boolean(window.localStorage);
  } catch (err) {
    console.warn("Accessing localStorage failed", err);
    return false;
  }
}

export function createAuthService(config = {}) {
  const storageKey = config.storageKey || DEFAULT_STORAGE_KEY;
  let fallbackToken =
    typeof config.token === "string" && config.token.trim() !== ""
      ? config.token.trim()
      : undefined;

  function persistToken(token) {
    if (typeof token === "string") {
      token = token.trim();
    }
    fallbackToken = token || undefined;
    if (!token) {
      clearToken();
      return;
    }

    if (!hasLocalStorage()) {
      return;
    }

    try {
      window.localStorage.setItem(storageKey, token);
    } catch (err) {
      console.warn("Unable to persist JWT in localStorage", err);
    }
  }

  function readStoredToken() {
    if (!hasLocalStorage()) {
      return undefined;
    }

    try {
      const stored = window.localStorage.getItem(storageKey);
      return stored || undefined;
    } catch (err) {
      console.warn("Unable to read JWT from localStorage", err);
      return undefined;
    }
  }

  function clearToken() {
    fallbackToken = undefined;

    if (!hasLocalStorage()) {
      return;
    }

    try {
      window.localStorage.removeItem(storageKey);
    } catch (err) {
      console.warn("Unable to clear JWT from localStorage", err);
    }
  }

  if (fallbackToken) {
    persistToken(fallbackToken);
  }

  async function getJwt() {
    const stored = readStoredToken();
    if (stored) {
      return stored;
    }
    if (fallbackToken) {
      return fallbackToken;
    }
    throw new Error("Missing JWT for chat authentication.");
  }

  return {
    getJwt,
    persistToken,
    clearToken,
    storageKey,
  };
}
