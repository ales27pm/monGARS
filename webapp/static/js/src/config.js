export function resolveConfig(raw = {}) {
  const config = { ...raw };
  const candidate = config.fastapiUrl || window.location.origin;
  try {
    config.baseUrl = new URL(candidate);
  } catch (err) {
    console.error("Invalid FASTAPI URL", err, candidate);
    config.baseUrl = new URL(window.location.origin);
  }
  const embedCandidate =
    typeof config.embedServiceUrl === "string"
      ? config.embedServiceUrl.trim()
      : "";
  if (embedCandidate) {
    try {
      const url = new URL(embedCandidate);
      if (url.protocol === "http:" || url.protocol === "https:") {
        config.embedServiceUrl = url.toString();
      } else {
        console.warn("Unsupported embedding service protocol", url.protocol);
        config.embedServiceUrl = null;
      }
    } catch (err) {
      console.warn("Invalid embedding service URL", err, embedCandidate);
      config.embedServiceUrl = null;
    }
  } else {
    config.embedServiceUrl = null;
  }
  return config;
}

export function apiUrl(config, path) {
  return new URL(path, config.baseUrl).toString();
}
