import { NativeModules, Platform } from 'react-native';

export const OUTLOOK_REQUIRED_SCOPES = [
  'User.Read',
  'Mail.ReadWrite',
  'Mail.Send',
  'offline_access',
] as const;

export type NativeOutlookConnectionStatus = {
  configured: boolean;
  connected: boolean;
  account: string | null;
  expiresAt: string | null;
  requiredScopes: string[];
  redirectUri: string;
  detail: string;
};

type NativeOutlookModule = {
  getOutlookConnectionStatus(ownerId: string): Promise<unknown>;
  configureOutlookClientID(ownerId: string, clientId: string): Promise<unknown>;
  connectOutlook(ownerId: string): Promise<unknown>;
  disconnectOutlook(ownerId: string): Promise<unknown>;
};

const candidate = NativeModules.CoreMLInferenceModule as
  | Partial<NativeOutlookModule>
  | undefined;

const nativeOutlookModule: NativeOutlookModule | null =
  Platform.OS === 'ios' &&
  candidate &&
  typeof candidate.getOutlookConnectionStatus === 'function' &&
  typeof candidate.configureOutlookClientID === 'function' &&
  typeof candidate.connectOutlook === 'function' &&
  typeof candidate.disconnectOutlook === 'function'
    ? (candidate as NativeOutlookModule)
    : null;

export const nativeOutlookModuleAvailable = nativeOutlookModule !== null;

export class NativeOutlookUnavailableError extends Error {
  readonly code = 'OUTLOOK_NATIVE_UNAVAILABLE';

  constructor() {
    super("La connexion Outlook native n'est pas liée à cette version iOS.");
    this.name = 'NativeOutlookUnavailableError';
  }
}

export class NativeOutlookContractError extends Error {
  readonly code = 'OUTLOOK_NATIVE_CONTRACT_INVALID';

  constructor(message: string) {
    super(`Réponse Outlook native invalide: ${message}`);
    this.name = 'NativeOutlookContractError';
  }
}

const ALLOWED_KEYS = new Set([
  'configured',
  'connected',
  'account',
  'expiresAt',
  'requiredScopes',
  'redirectUri',
  'detail',
]);
const REDIRECT_PATTERN = /^msauth\.[A-Za-z0-9.-]{1,255}:\/\/auth$/;

const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new NativeOutlookContractError('le statut doit être un objet.');
  }
  const source = value as Record<string, unknown>;
  if (Object.keys(source).some((key) => !ALLOWED_KEYS.has(key))) {
    throw new NativeOutlookContractError(
      'le statut contient un champ inattendu.',
    );
  }
  return source;
};

const boolean = (value: unknown, field: string): boolean => {
  if (typeof value !== 'boolean') {
    throw new NativeOutlookContractError(`${field} doit être booléen.`);
  }
  return value;
};

const boundedString = (
  value: unknown,
  field: string,
  maximum: number,
): string => {
  if (typeof value !== 'string') {
    throw new NativeOutlookContractError(`${field} doit être une chaîne.`);
  }
  const trimmed = value.trim();
  if (
    !trimmed ||
    trimmed.length > maximum ||
    [...trimmed].some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return codePoint <= 31 || (codePoint >= 127 && codePoint <= 159);
    })
  ) {
    throw new NativeOutlookContractError(`${field} est invalide.`);
  }
  return trimmed;
};

const nullableString = (
  value: unknown,
  field: string,
  maximum: number,
): string | null =>
  value === null ? null : boundedString(value, field, maximum);

export const normalizeOutlookConnectionStatus = (
  value: unknown,
): NativeOutlookConnectionStatus => {
  const source = record(value);
  const configured = boolean(source.configured, 'configured');
  const connected = boolean(source.connected, 'connected');
  if (connected && !configured) {
    throw new NativeOutlookContractError(
      'un compte connecté doit avoir une configuration active.',
    );
  }
  const account = nullableString(source.account, 'account', 512);
  if (connected && account === null) {
    throw new NativeOutlookContractError(
      'un compte connecté doit avoir un identifiant affichable.',
    );
  }
  const rawExpiresAt = nullableString(source.expiresAt, 'expiresAt', 64);
  const parsedExpiresAt = rawExpiresAt ? new Date(rawExpiresAt) : null;
  if (parsedExpiresAt && Number.isNaN(parsedExpiresAt.getTime())) {
    throw new NativeOutlookContractError('expiresAt doit être une date ISO.');
  }
  const expiresAt = parsedExpiresAt ? parsedExpiresAt.toISOString() : null;
  if (
    !Array.isArray(source.requiredScopes) ||
    source.requiredScopes.length !== OUTLOOK_REQUIRED_SCOPES.length ||
    source.requiredScopes.some(
      (scope, index) => scope !== OUTLOOK_REQUIRED_SCOPES[index],
    )
  ) {
    throw new NativeOutlookContractError(
      "requiredScopes ne correspond pas au contrat de l'agent.",
    );
  }
  const redirectUri = boundedString(source.redirectUri, 'redirectUri', 320);
  if (configured && !REDIRECT_PATTERN.test(redirectUri)) {
    throw new NativeOutlookContractError('redirectUri est invalide.');
  }
  return {
    configured,
    connected,
    account,
    expiresAt,
    requiredScopes: [...OUTLOOK_REQUIRED_SCOPES],
    redirectUri,
    detail: boundedString(source.detail, 'detail', 2_000),
  };
};

const requireModule = (): NativeOutlookModule => {
  if (!nativeOutlookModule) {
    throw new NativeOutlookUnavailableError();
  }
  return nativeOutlookModule;
};

const normalizedOwnerId = (ownerId: string): string =>
  boundedString(ownerId, 'ownerId', 256);

const normalizedClientId = (clientId: string): string => {
  const value = boundedString(clientId, 'clientId', 64).toLowerCase();
  if (
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
      value,
    )
  ) {
    throw new NativeOutlookContractError('clientId doit être un UUID.');
  }
  return value;
};

export const getNativeOutlookConnectionStatus = async (ownerId: string) => {
  const owner = normalizedOwnerId(ownerId);
  return normalizeOutlookConnectionStatus(
    await requireModule().getOutlookConnectionStatus(owner),
  );
};

export const connectNativeOutlook = async (ownerId: string) => {
  const owner = normalizedOwnerId(ownerId);
  return normalizeOutlookConnectionStatus(
    await requireModule().connectOutlook(owner),
  );
};

export const configureNativeOutlookClientId = async (
  ownerId: string,
  clientId: string,
) => {
  const owner = normalizedOwnerId(ownerId);
  const client = normalizedClientId(clientId);
  return normalizeOutlookConnectionStatus(
    await requireModule().configureOutlookClientID(owner, client),
  );
};

export const disconnectNativeOutlook = async (ownerId: string) => {
  const owner = normalizedOwnerId(ownerId);
  return normalizeOutlookConnectionStatus(
    await requireModule().disconnectOutlook(owner),
  );
};
