// auth.js
// -------
// Small auth token helper.
//
// The backend expects a Bearer JWT on protected endpoints (e.g. /api/v1/auth/ws/ticket).
// Historically, a couple of storage keys have been used; we read them all to be
// tolerant during refactors.

const DEFAULT_STORAGE_KEY = "mongars_jwt";
const LEGACY_STORAGE_KEYS = ["jwt", "access_token"];

function normaliseJwt(token) {
  if (!token) return undefined;
  if (typeof token !== "string") return undefined;

  let t = token.trim();
  if (!t) return undefined;

  if (t.toLowerCase().startsWith("bearer ")) {
    t = t.slice(7).trim();
  }

  // Strip common accidental quoting.
  t = t.replace(/^"+|"+$/g, "").trim();

  if (!t) return undefined;
  if (t === "null" || t === "undefined") return undefined;

  return t;
}

function safeGet(storage, key) {
  try {
    return normaliseJwt(storage.getItem(key));
  } catch {
    return undefined;
  }
}

function safeSet(storage, key, value) {
  try {
    if (value === undefined || value === null) storage.removeItem(key);
    else storage.setItem(key, value);
  } catch {
    // ignore
  }
}

function safeDel(storage, key) {
  try {
    storage.removeItem(key);
  } catch {
    // ignore
  }
}

function readTokenFromStorage(primaryKey) {
  const keys = [primaryKey, ...LEGACY_STORAGE_KEYS.filter((k) => k !== primaryKey)];
  const storages = [];

  if (typeof window !== "undefined") {
    if (window.localStorage) storages.push(window.localStorage);
    if (window.sessionStorage) storages.push(window.sessionStorage);
  }

  for (const storage of storages) {
    for (const key of keys) {
      const v = safeGet(storage, key);
      if (v) return v;
    }
  }

  return undefined;
}

function seedTokenFromUrlOnce(primaryKey) {
  if (typeof window === "undefined") return undefined;
  try {
    const u = new URL(window.location.href);
    const fromQuery = u.searchParams.get("jwt") || u.searchParams.get("token");
    const v = normaliseJwt(fromQuery);
    if (!v) return undefined;

    // Persist and scrub the URL (avoid leaking tokens via copy/paste).
    safeSet(window.localStorage, primaryKey, v);
    u.searchParams.delete("jwt");
    u.searchParams.delete("token");
    window.history.replaceState({}, "", u.toString());
    return v;
  } catch {
    return undefined;
  }
}

export function createAuthService({ storageKey = DEFAULT_STORAGE_KEY } = {}) {
  let currentJwt = undefined;

  function getJwt() {
    if (currentJwt) return currentJwt;

    const seeded = seedTokenFromUrlOnce(storageKey);
    if (seeded) {
      currentJwt = seeded;
      return currentJwt;
    }

    currentJwt = readTokenFromStorage(storageKey);
    return currentJwt;
  }

  function persistToken(token, { alsoLegacyKeys = false } = {}) {
    const v = normaliseJwt(token);
    if (!v) return;

    currentJwt = v;

    if (typeof window === "undefined" || !window.localStorage) return;
    safeSet(window.localStorage, storageKey, v);

    if (alsoLegacyKeys) {
      for (const key of LEGACY_STORAGE_KEYS) {
        if (key !== storageKey) safeSet(window.localStorage, key, v);
      }
    }
  }

  function clearToken() {
    currentJwt = undefined;

    if (typeof window === "undefined") return;
    const storages = [];
    if (window.localStorage) storages.push(window.localStorage);
    if (window.sessionStorage) storages.push(window.sessionStorage);

    for (const storage of storages) {
      safeDel(storage, storageKey);
      for (const key of LEGACY_STORAGE_KEYS) safeDel(storage, key);
    }
  }

  return {
    getJwt,
    persistToken,
    clearToken,
    storageKey,
  };
}
