export const DEFAULT_ERROR_MESSAGE =
  "Une erreur inattendue est survenue. Veuillez réessayer.";

const ERROR_PREFIX_REGEX = /^\s*[\W_]*\s*(erreur|error)/i;

export function normaliseErrorText(error) {
  if (error instanceof Error) {
    const message = error.message ? error.message.trim() : "";
    if (message) {
      return message;
    }
  }

  if (typeof error === "string") {
    const trimmed = error.trim();
    return trimmed || DEFAULT_ERROR_MESSAGE;
  }

  if (error && typeof error === "object") {
    const candidate =
      (typeof error.message === "string" && error.message.trim()) ||
      (typeof error.error === "string" && error.error.trim());
    if (candidate) {
      return candidate;
    }

    try {
      const serialised = JSON.stringify(error);
      if (serialised && serialised !== "{}") {
        return serialised;
      }
    } catch (serialiseErr) {
      console.debug("Unable to serialise error payload", serialiseErr);
    }
  }

  if (typeof error === "undefined" || error === null) {
    return DEFAULT_ERROR_MESSAGE;
  }

  const fallback = String(error).trim();
  return fallback || DEFAULT_ERROR_MESSAGE;
}

export function resolveErrorText(errorOrText) {
  if (typeof errorOrText === "string") {
    const trimmed = errorOrText.trim();
    return trimmed || DEFAULT_ERROR_MESSAGE;
  }
  return normaliseErrorText(errorOrText);
}

export function computeErrorBubbleText(errorOrText, options = {}) {
  const { prefix } = options;
  const text = resolveErrorText(errorOrText);
  const basePrefix =
    options.prefix === null
      ? ""
      : typeof prefix === "string"
        ? prefix
        : "Erreur : ";
  const trimmedPrefix = basePrefix.trim().toLowerCase();
  const shouldPrefix =
    Boolean(basePrefix) &&
    !ERROR_PREFIX_REGEX.test(text) &&
    !(trimmedPrefix && text.toLowerCase().startsWith(trimmedPrefix));
  const bubbleText = shouldPrefix ? `${basePrefix}${text}` : text;
  return { text, bubbleText };
}
