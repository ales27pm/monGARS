import Config from 'react-native-config';

const DEFAULT_BASE_URL = 'https://localhost:8443';
const DEFAULT_VOICE_LOCALE = 'fr-CA';
const DEFAULT_HISTORY_LIMIT = 10;

function normaliseBaseUrl(value?: string | null): string {
  const trimmed = value?.trim();
  if (!trimmed) {
    return DEFAULT_BASE_URL;
  }

  return trimmed
    .replace(/\/api\/v1\/?$/, '')
    .replace(/\/ws\/chat\/?$/, '')
    .replace(/\/$/, '');
}

function toWebSocketUrl(baseUrl: string, explicit?: string | null): string {
  const trimmed = explicit?.trim();
  if (trimmed) {
    return trimmed.replace(/\/$/, '');
  }

  const url = new URL(baseUrl);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  url.pathname = '/ws/chat/';
  url.search = '';
  return url.toString().replace(/\/$/, '');
}

const baseUrl = normaliseBaseUrl(
  Config.MONGARS_BASE_URL ?? Config.MONGARS_API_URL ?? DEFAULT_BASE_URL,
);

const historyLimitRaw = Number(
  Config.MONGARS_HISTORY_LIMIT ?? DEFAULT_HISTORY_LIMIT,
);

export const settings = {
  baseUrl,
  apiBaseUrl: `${baseUrl}/api/v1`,
  websocketUrl: toWebSocketUrl(baseUrl, Config.MONGARS_WS_URL),
  embedServiceUrl: Config.MONGARS_EMBED_URL?.trim() || null,
  voiceLocale: Config.MONGARS_VOICE_LOCALE?.trim() || DEFAULT_VOICE_LOCALE,
  historyLimit:
    Number.isFinite(historyLimitRaw) && historyLimitRaw > 0
      ? historyLimitRaw
      : DEFAULT_HISTORY_LIMIT,
};
