import Config from 'react-native-config';

const API_URL = Config.MONGARS_API_URL ?? 'https://localhost:8443/api/v1';

export const settings = {
  apiUrl: API_URL.replace(/\/$/, ''),
  websocketUrl: Config.MONGARS_WS_URL ?? 'wss://localhost:8443/ws/chat',
  voiceLocale: Config.MONGARS_VOICE_LOCALE ?? 'fr-FR',
};
