export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: Date;
};

export type AuthCredentials = {
  username: string;
  password: string;
};

export type AuthToken = {
  accessToken: string;
  expiresAt: string;
};
