import type { JSONValue } from './types';

const MODEL_SENTINELS = [
  '<|assistant|>',
  '<|user|>',
  '<|system|>',
  '<|eot_id|>',
  '<|start_header_id|>',
  '<|end_header_id|>',
] as const;

export const sanitizeAgentText = (
  raw: string,
  maximumCharacters: number,
): string => {
  let filtered = '';
  for (const character of raw) {
    const codePoint = character.codePointAt(0) ?? 0;
    if (
      character === '\n' ||
      character === '\t' ||
      (codePoint >= 32 && codePoint !== 127)
    ) {
      filtered += character;
    }
  }
  MODEL_SENTINELS.forEach((token) => {
    filtered = filtered.split(token).join('');
  });
  filtered = filtered
    .replace(/\r\n?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return [...filtered].slice(0, Math.max(1, maximumCharacters)).join('');
};

export const sanitizeError = (
  error: unknown,
  maximumCharacters = 500,
): string => {
  const raw =
    error instanceof Error ? error.message : 'Unknown execution error.';
  const redacted = raw
    .replace(/\bBearer\s+[^\s,;]+/gi, 'Bearer [REDACTED]')
    .replace(
      /\b(api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|secret|password)\b\s*[:=]\s*[^\s,;]+/gi,
      '$1=[REDACTED]',
    );
  return (
    sanitizeAgentText(redacted, maximumCharacters) || 'Execution failed safely.'
  );
};

export const stringifyObservation = (value: JSONValue): string => {
  if (typeof value === 'string') {
    return value;
  }
  const encoded = JSON.stringify(value);
  return encoded ?? 'Tool returned no observation.';
};
