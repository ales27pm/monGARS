export function resolveConfig(raw = {}) {
  const config = { ...raw };
  const candidate = config.fastapiUrl || window.location.origin;
  try {
    config.baseUrl = new URL(candidate);
  } catch (err) {
    console.error("Invalid FASTAPI URL", err, candidate);
    config.baseUrl = new URL(window.location.origin);
  }
  return config;
}

export function apiUrl(config, path) {
  return new URL(path, config.baseUrl).toString();
}
