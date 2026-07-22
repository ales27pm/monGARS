import { NativeEventEmitter, NativeModules, Platform } from 'react-native';

export type NativeAppIntentHandoffKind =
  | 'ask'
  | 'memorySearch'
  | 'memoryAdd'
  | 'runTrigger'
  | 'diagnostics'
  | 'masked';

export type NativeAppIntentHandoff = {
  id: string;
  kind: NativeAppIntentHandoffKind;
  input?: string;
  createdAt: string;
  expiresAt: string;
  profileMatches: boolean;
};

export type NativeAppIntentHandoffSignal = Pick<
  NativeAppIntentHandoff,
  'id' | 'createdAt'
>;

export type NativeResolvedStoredTrigger = {
  id: string;
  title: string;
  prompt: string;
  repeats: boolean;
};

export type NativeAppIntentMemoryKind = 'memorySearch' | 'memoryAdd';

export type NativeAppIntentMemoryResult = {
  id: string;
  toolId: 'memory.recall' | 'memory.save';
  status: 'success' | 'unavailable' | 'denied' | 'failed' | 'cancelled';
  message: string;
  errorCode?: string;
};

type NativeAppIntentModule = {
  addListener(eventType: string): void;
  removeListeners(count: number): void;
  setActiveAppIntentProfile(ownerId: string): Promise<unknown>;
  getPendingAppIntentHandoff(ownerId: string): Promise<unknown>;
  acknowledgeAppIntentHandoff(request: {
    ownerId: string;
    id: string;
  }): Promise<unknown>;
  discardAppIntentHandoff(request: { id: string }): Promise<unknown>;
  executeAppIntentMemoryAction(request: {
    ownerId: string;
    id: string;
    kind: NativeAppIntentMemoryKind;
    input: string;
  }): Promise<unknown>;
  resolveStoredAgentTrigger(request: {
    ownerId: string;
    selector: string;
  }): Promise<unknown>;
};

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HANDOFF_KINDS = new Set<NativeAppIntentHandoffKind>([
  'ask',
  'memorySearch',
  'memoryAdd',
  'runTrigger',
  'diagnostics',
  'masked',
]);
const MEMORY_RESULT_STATUSES = new Set<NativeAppIntentMemoryResult['status']>([
  'success',
  'unavailable',
  'denied',
  'failed',
  'cancelled',
]);
const INPUT_MAXIMUM_BYTES: Record<NativeAppIntentHandoffKind, number> = {
  ask: 512,
  memorySearch: 192,
  memoryAdd: 186,
  runTrigger: 512,
  diagnostics: 0,
  masked: 0,
};

const utf8ByteLength = (value: string): number => {
  let length = 0;
  for (const character of value) {
    const codePoint = character.codePointAt(0) ?? 0;
    length +=
      codePoint <= 0x7f
        ? 1
        : codePoint <= 0x7ff
          ? 2
          : codePoint <= 0xffff
            ? 3
            : 4;
  }
  return length;
};

export class NativeAppIntentUnavailableError extends Error {
  readonly code = 'APP_INTENT_HANDOFF_UNAVAILABLE';

  constructor() {
    super(
      Platform.OS === 'ios'
        ? "Le pont App Intents n'est pas lié à cette version de monGARS."
        : 'Les App Intents monGARS sont disponibles uniquement sur iOS.',
    );
    this.name = 'NativeAppIntentUnavailableError';
  }
}

export class NativeAppIntentContractError extends Error {
  readonly code = 'APP_INTENT_HANDOFF_INVALID';

  constructor(message: string) {
    super(`Transfert App Intent invalide: ${message}`);
    this.name = 'NativeAppIntentContractError';
  }
}

const isNativeAppIntentModule = (
  value: unknown,
): value is NativeAppIntentModule => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<NativeAppIntentModule>;
  return (
    typeof candidate.addListener === 'function' &&
    typeof candidate.removeListeners === 'function' &&
    typeof candidate.setActiveAppIntentProfile === 'function' &&
    typeof candidate.getPendingAppIntentHandoff === 'function' &&
    typeof candidate.acknowledgeAppIntentHandoff === 'function' &&
    typeof candidate.discardAppIntentHandoff === 'function' &&
    typeof candidate.executeAppIntentMemoryAction === 'function' &&
    typeof candidate.resolveStoredAgentTrigger === 'function'
  );
};

const linkedModule = NativeModules.CoreMLInferenceModule as unknown;
const nativeModule =
  Platform.OS === 'ios' && isNativeAppIntentModule(linkedModule)
    ? linkedModule
    : null;

export const nativeAppIntentModuleAvailable = nativeModule !== null;
let eventEmitter: NativeEventEmitter | null = null;

const requireModule = (): NativeAppIntentModule => {
  if (!nativeModule) {
    throw new NativeAppIntentUnavailableError();
  }
  return nativeModule;
};

const record = (value: unknown, field: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new NativeAppIntentContractError(`${field} doit être un objet.`);
  }
  return value as Record<string, unknown>;
};

const requiredString = (
  value: unknown,
  field: string,
  maximumBytes: number,
): string => {
  if (typeof value !== 'string') {
    throw new NativeAppIntentContractError(`${field} doit être une chaîne.`);
  }
  const output = value.trim();
  if (!output || utf8ByteLength(output) > maximumBytes) {
    throw new NativeAppIntentContractError(`${field} est vide ou trop long.`);
  }
  return output;
};

const uuid = (value: unknown, field: string): string => {
  const output = requiredString(value, field, 64);
  if (!UUID_PATTERN.test(output)) {
    throw new NativeAppIntentContractError(`${field} doit être un UUID.`);
  }
  return output.toLowerCase();
};

const isoDate = (value: unknown, field: string): string => {
  const output = requiredString(value, field, 64);
  const parsed = new Date(output);
  if (Number.isNaN(parsed.getTime())) {
    throw new NativeAppIntentContractError(`${field} doit être une date ISO.`);
  }
  return parsed.toISOString();
};

const kind = (value: unknown): NativeAppIntentHandoffKind => {
  const output = requiredString(value, 'kind', 32);
  if (!HANDOFF_KINDS.has(output as NativeAppIntentHandoffKind)) {
    throw new NativeAppIntentContractError('kind est inconnu.');
  }
  return output as NativeAppIntentHandoffKind;
};

const normalizedInput = (
  value: unknown,
  handoffKind: NativeAppIntentHandoffKind,
): string | undefined => {
  if (handoffKind === 'diagnostics' || handoffKind === 'masked') {
    if (value !== undefined && value !== null) {
      throw new NativeAppIntentContractError(
        "diagnostics ne doit pas contenir d'entrée.",
      );
    }
    return undefined;
  }
  return requiredString(value, 'input', INPUT_MAXIMUM_BYTES[handoffKind]);
};

const normalizedOwnerId = (value: unknown): string => {
  const ownerId = requiredString(value, 'ownerId', 256);
  if (
    [...ownerId].some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return (
        codePoint <= 31 ||
        (codePoint >= 127 && codePoint <= 159) ||
        codePoint === 8_232 ||
        codePoint === 8_233
      );
    })
  ) {
    throw new NativeAppIntentContractError(
      'ownerId contient un caractère de contrôle.',
    );
  }
  return ownerId;
};

export const normalizeNativeAppIntentHandoff = (
  value: unknown,
): NativeAppIntentHandoff => {
  const source = record(value, 'handoff');
  const handoffKind = kind(source.kind);
  const createdAt = isoDate(source.createdAt, 'createdAt');
  const expiresAt = isoDate(source.expiresAt, 'expiresAt');
  if (new Date(expiresAt).getTime() <= new Date(createdAt).getTime()) {
    throw new NativeAppIntentContractError(
      'expiresAt doit être postérieur à createdAt.',
    );
  }
  if (typeof source.profileMatches !== 'boolean') {
    throw new NativeAppIntentContractError('profileMatches doit être booléen.');
  }
  if (!source.profileMatches && handoffKind !== 'masked') {
    throw new NativeAppIntentContractError(
      'kind doit être masqué pour un autre profil.',
    );
  }
  if (source.profileMatches && handoffKind === 'masked') {
    throw new NativeAppIntentContractError(
      'kind masqué ne peut pas appartenir au profil actif.',
    );
  }
  if (
    !source.profileMatches &&
    source.input !== undefined &&
    source.input !== null
  ) {
    throw new NativeAppIntentContractError(
      'input ne doit pas être révélé pour un autre profil.',
    );
  }
  const input = source.profileMatches
    ? normalizedInput(source.input, handoffKind)
    : undefined;
  return {
    id: uuid(source.id, 'id'),
    kind: handoffKind,
    ...(input ? { input } : {}),
    createdAt,
    expiresAt,
    profileMatches: source.profileMatches,
  };
};

export const normalizeNativeAppIntentHandoffSignal = (
  value: unknown,
): NativeAppIntentHandoffSignal => {
  const source = record(value, 'signal');
  if (source.kind !== undefined || source.input !== undefined) {
    throw new NativeAppIntentContractError(
      'signal ne doit révéler ni kind ni input.',
    );
  }
  return {
    id: uuid(source.id, 'id'),
    createdAt: isoDate(source.createdAt, 'createdAt'),
  };
};

const normalizeStoredTrigger = (
  value: unknown,
): NativeResolvedStoredTrigger => {
  const source = record(value, 'trigger');
  if (typeof source.repeats !== 'boolean') {
    throw new NativeAppIntentContractError(
      'trigger.repeats doit être booléen.',
    );
  }
  return {
    id: uuid(source.id, 'trigger.id'),
    title: requiredString(source.title, 'trigger.title', 1_000),
    prompt: requiredString(source.prompt, 'trigger.prompt', 512),
    repeats: source.repeats,
  };
};

export const setActiveNativeAppIntentProfile = async (
  rawOwnerId: string,
): Promise<void> => {
  const ownerId = normalizedOwnerId(rawOwnerId);
  const source = record(
    await requireModule().setActiveAppIntentProfile(ownerId),
    'profile',
  );
  if (source.active !== true) {
    throw new NativeAppIntentContractError('profile.active doit être vrai.');
  }
};

export const getPendingNativeAppIntentHandoff = async (
  rawOwnerId: string,
): Promise<NativeAppIntentHandoff | null> => {
  const ownerId = normalizedOwnerId(rawOwnerId);
  const value = await requireModule().getPendingAppIntentHandoff(ownerId);
  if (value === null || value === undefined) {
    return null;
  }
  return normalizeNativeAppIntentHandoff(value);
};

export const acknowledgeNativeAppIntentHandoff = async (
  rawOwnerId: string,
  rawId: string,
): Promise<boolean> => {
  const ownerId = normalizedOwnerId(rawOwnerId);
  const id = uuid(rawId, 'id');
  const source = record(
    await requireModule().acknowledgeAppIntentHandoff({ ownerId, id }),
    'acknowledgement',
  );
  if (uuid(source.id, 'acknowledgement.id') !== id) {
    throw new NativeAppIntentContractError(
      'acknowledgement.id ne correspond pas à la requête.',
    );
  }
  if (typeof source.acknowledged !== 'boolean') {
    throw new NativeAppIntentContractError(
      'acknowledgement.acknowledged doit être booléen.',
    );
  }
  return source.acknowledged;
};

export const discardNativeAppIntentHandoff = async (
  rawId: string,
): Promise<boolean> => {
  const id = uuid(rawId, 'id');
  const source = record(
    await requireModule().discardAppIntentHandoff({ id }),
    'discard',
  );
  if (uuid(source.id, 'discard.id') !== id) {
    throw new NativeAppIntentContractError(
      'discard.id ne correspond pas à la requête.',
    );
  }
  if (typeof source.discarded !== 'boolean') {
    throw new NativeAppIntentContractError(
      'discard.discarded doit être booléen.',
    );
  }
  return source.discarded;
};

export const executeNativeAppIntentMemoryAction = async (request: {
  ownerId: string;
  id: string;
  kind: NativeAppIntentMemoryKind;
  input: string;
}): Promise<NativeAppIntentMemoryResult> => {
  const ownerId = normalizedOwnerId(request.ownerId);
  const id = uuid(request.id, 'id');
  if (request.kind !== 'memorySearch' && request.kind !== 'memoryAdd') {
    throw new NativeAppIntentContractError('kind mémoire est inconnu.');
  }
  const input = normalizedInput(request.input, request.kind);
  if (!input) {
    throw new NativeAppIntentContractError('input mémoire est absent.');
  }
  const source = record(
    await requireModule().executeAppIntentMemoryAction({
      ownerId,
      id,
      kind: request.kind,
      input,
    }),
    'memoryResult',
  );
  if (uuid(source.id, 'memoryResult.id') !== id) {
    throw new NativeAppIntentContractError(
      'memoryResult.id ne correspond pas à la requête.',
    );
  }
  const expectedToolId =
    request.kind === 'memorySearch' ? 'memory.recall' : 'memory.save';
  if (source.toolId !== expectedToolId) {
    throw new NativeAppIntentContractError(
      'memoryResult.toolId ne correspond pas au type confirmé.',
    );
  }
  if (
    typeof source.status !== 'string' ||
    !MEMORY_RESULT_STATUSES.has(
      source.status as NativeAppIntentMemoryResult['status'],
    )
  ) {
    throw new NativeAppIntentContractError('memoryResult.status est inconnu.');
  }
  const message = requiredString(
    source.message,
    'memoryResult.message',
    16_000,
  );
  const errorCode =
    source.errorCode === undefined || source.errorCode === null
      ? undefined
      : requiredString(source.errorCode, 'memoryResult.errorCode', 120);
  return {
    id,
    toolId: expectedToolId,
    status: source.status as NativeAppIntentMemoryResult['status'],
    message,
    ...(errorCode ? { errorCode } : {}),
  };
};

export const resolveNativeStoredAgentTrigger = async (
  rawOwnerId: string,
  rawSelector: string,
): Promise<NativeResolvedStoredTrigger | null> => {
  const ownerId = normalizedOwnerId(rawOwnerId);
  const selector = requiredString(rawSelector, 'selector', 512);
  const value = await requireModule().resolveStoredAgentTrigger({
    ownerId,
    selector,
  });
  if (value === null || value === undefined) {
    return null;
  }
  return normalizeStoredTrigger(value);
};

export const subscribeNativeAppIntentHandoff = (
  listener: (signal: NativeAppIntentHandoffSignal) => void,
): (() => void) => {
  const module = requireModule();
  eventEmitter ??= new NativeEventEmitter(module);
  const subscription = eventEmitter.addListener(
    'onAppIntentHandoff',
    (value: unknown) => {
      try {
        listener(normalizeNativeAppIntentHandoffSignal(value));
      } catch (error) {
        console.debug('[appIntents] invalid handoff signal ignored', error);
      }
    },
  );
  return () => subscription.remove();
};
