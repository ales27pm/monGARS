import axios from 'axios';
import { settings } from './config';
import type { AuthCredentials, AuthToken } from '../types';

export async function authenticate(
  credentials: AuthCredentials,
): Promise<AuthToken> {
  const username = credentials.username.trim();
  const body = new URLSearchParams();
  body.set('username', username);
  body.set('password', credentials.password);

  const response = await axios.post(`${settings.baseUrl}/token`, body.toString(), {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    timeout: 10000,
  });

  return {
    accessToken: response.data.access_token,
    tokenType: response.data.token_type ?? 'bearer',
    username,
  };
}
