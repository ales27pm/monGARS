import axios from 'axios';
import { settings } from './config';
import type { AuthCredentials, AuthToken } from '../types';

export async function authenticate(
  credentials: AuthCredentials,
): Promise<AuthToken> {
  const response = await axios.post(
    `${settings.apiUrl}/auth/token`,
    credentials,
    {
      timeout: 10000,
    },
  );
  return {
    accessToken: response.data.access_token,
    expiresAt: response.data.expires_at,
  };
}
