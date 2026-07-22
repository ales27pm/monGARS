import { NativeEventEmitter, NativeModules, Platform } from 'react-native';
import { AGENT_TOOL_CATALOG } from '../agent/catalog';
import { INTENT_TOOL_IDS } from '../agent/routing';
import type {
  AgentIntent,
  AgentPermission,
  JSONObject,
  JSONValue,
} from '../agent/types';

export type NativeAgentHistoryMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type NativeAgentRunRequest = {
  runId: string;
  ownerId: string;
  prompt: string;
  history: NativeAgentHistoryMessage[];
  requestedIntent?: AgentIntent;
  allowedToolIds?: string[];
  approvalRecordId?: string;
  maxSteps?: number;
};

export type NativeAgentEvent = {
  type:
    | 'started'
    | 'routed'
    | 'model_turn'
    | 'repair_requested'
    | 'action_validated'
    | 'approval_required'
    | 'permission_required'
    | 'tool_started'
    | 'tool_finished'
    | 'final'
    | 'failure'
    | 'completed';
  toolId?: string;
  status?: string;
  stepIndex?: number;
  message?: string;
};

export type NativeAgentApproval = {
  recordId: string;
  toolId: string;
  arguments: JSONObject;
  displayName: string;
  risk: 'low' | 'moderate' | 'high' | 'critical';
  expiresAt: string;
};

type NativeAgentRunBase = {
  runId: string;
  intent: AgentIntent;
  events: NativeAgentEvent[];
  executedToolCount: number;
  modelTurnCount: number;
  usedRepairAttempt: boolean;
};

export type NativeAgentRunResult =
  | (NativeAgentRunBase & { status: 'final'; message: string })
  | (NativeAgentRunBase & { status: 'clarification'; message: string })
  | (NativeAgentRunBase & {
      status: 'approval_required';
      approval: NativeAgentApproval;
    })
  | (NativeAgentRunBase & {
      status: 'permission_required';
      permission: AgentPermission;
      message: string;
    })
  | (NativeAgentRunBase & { status: 'unavailable'; message: string })
  | (NativeAgentRunBase & {
      status: 'failed' | 'cancelled';
      code: string;
      message: string;
    });

export type NativeAgentCapabilities = {
  available: true;
  toolIds: string[];
  toolCount: number;
  supportsApprovals: true;
  maximumSteps: number;
};

export type NativeAgentApprovalBinding = {
  recordId: string;
  ownerId: string;
  prompt: string;
  toolId: string;
  arguments: JSONObject;
  expiresAt: string;
};

export type NativeAgentApprovalResult = {
  recordId: string;
  status: 'approved' | 'rejected' | 'expired';
};

export type NativeAgentPermissionResult = {
  permission: AgentPermission;
  state:
    | 'granted'
    | 'limited'
    | 'notDetermined'
    | 'denied'
    | 'restricted'
    | 'unavailable';
};

export type NativeAgentTriggerHandoff = {
  id: string;
  title: string;
  prompt: string;
  repeats: boolean;
};

export type NativeAgentTriggerSignal = {
  id: string;
  tappedAt: string;
};

type NativeAgentModule = {
  addListener(eventType: string): void;
  removeListeners(count: number): void;
  getAgentCapabilities(ownerId: string): Promise<unknown>;
  runAgent(request: NativeAgentRunRequest): Promise<unknown>;
  requestAgentPermission(permission: AgentPermission): Promise<unknown>;
  getPendingAgentTrigger(ownerId: string): Promise<unknown>;
  acknowledgePendingAgentTrigger(request: {
    ownerId: string;
    id: string;
  }): Promise<unknown>;
  approveAgent(binding: NativeAgentApprovalBinding): Promise<unknown>;
  rejectAgent(binding: NativeAgentApprovalBinding): Promise<unknown>;
  cancelAgent(runId: string): Promise<unknown>;
};

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const TOOL_IDS = new Set(AGENT_TOOL_CATALOG.map((tool) => tool.id));
const PERMISSIONS = new Set<AgentPermission>([
  'calendar',
  'reminders',
  'contacts',
  'location',
  'photos',
  'camera',
  'health',
  'motion',
  'alarms',
  'notifications',
]);
const INTENTS = new Set<AgentIntent>([
  'weather',
  'webSearch',
  'emailDraft',
  'messageDraft',
  'phoneCall',
  'contactSearch',
  'calendar',
  'reminder',
  'maps',
  'photos',
  'camera',
  'health',
  'motion',
  'files',
  'memory',
  'rag',
  'trigger',
  'alarm',
  'outlook',
  'note',
  'chat',
  'unknown',
]);
const EVENT_TYPES = new Set<NativeAgentEvent['type']>([
  'started',
  'routed',
  'model_turn',
  'repair_requested',
  'action_validated',
  'approval_required',
  'permission_required',
  'tool_started',
  'tool_finished',
  'final',
  'failure',
  'completed',
]);

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

const isNativeAgentModule = (value: unknown): value is NativeAgentModule => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<NativeAgentModule>;
  return (
    typeof candidate.getAgentCapabilities === 'function' &&
    typeof candidate.addListener === 'function' &&
    typeof candidate.removeListeners === 'function' &&
    typeof candidate.runAgent === 'function' &&
    typeof candidate.requestAgentPermission === 'function' &&
    typeof candidate.getPendingAgentTrigger === 'function' &&
    typeof candidate.acknowledgePendingAgentTrigger === 'function' &&
    typeof candidate.approveAgent === 'function' &&
    typeof candidate.rejectAgent === 'function' &&
    typeof candidate.cancelAgent === 'function'
  );
};

const linkedModule = NativeModules.CoreMLInferenceModule as unknown;
const nativeAgentModule =
  Platform.OS === 'ios' && isNativeAgentModule(linkedModule)
    ? linkedModule
    : null;

export const nativeAgentModuleAvailable = nativeAgentModule !== null;
let nativeAgentEventEmitter: NativeEventEmitter | null = null;

export class NativeAgentUnavailableError extends Error {
  readonly code = 'AGENT_MODULE_UNAVAILABLE';

  constructor() {
    super(
      Platform.OS === 'ios'
        ? "Le moteur d'outils local n'est pas lié à cette version de l'application."
        : "Le moteur d'outils local est disponible uniquement sur iOS.",
    );
    this.name = 'NativeAgentUnavailableError';
  }
}

export class NativeAgentContractError extends Error {
  readonly code = 'AGENT_NATIVE_CONTRACT_INVALID';

  constructor(message: string) {
    super(`Réponse native de l'agent invalide: ${message}`);
    this.name = 'NativeAgentContractError';
  }
}

const requireModule = (): NativeAgentModule => {
  if (!nativeAgentModule) {
    throw new NativeAgentUnavailableError();
  }
  return nativeAgentModule;
};

const record = (value: unknown, field: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new NativeAgentContractError(`${field} doit être un objet.`);
  }
  return value as Record<string, unknown>;
};

const requiredString = (
  value: unknown,
  field: string,
  maximumBytes = 16_000,
): string => {
  if (typeof value !== 'string') {
    throw new NativeAgentContractError(`${field} doit être une chaîne.`);
  }
  const trimmed = value.trim();
  if (!trimmed || utf8ByteLength(trimmed) > maximumBytes) {
    throw new NativeAgentContractError(`${field} est vide ou trop long.`);
  }
  return trimmed;
};

const integer = (
  value: unknown,
  field: string,
  minimum: number,
  maximum: number,
): number => {
  if (
    typeof value !== 'number' ||
    !Number.isInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new NativeAgentContractError(`${field} est hors limites.`);
  }
  return value;
};

const boolean = (value: unknown, field: string): boolean => {
  if (typeof value !== 'boolean') {
    throw new NativeAgentContractError(`${field} doit être booléen.`);
  }
  return value;
};

const uuid = (value: unknown, field: string): string => {
  const output = requiredString(value, field, 64);
  if (!UUID_PATTERN.test(output)) {
    throw new NativeAgentContractError(`${field} doit être un UUID.`);
  }
  return output.toLowerCase();
};

const toolId = (value: unknown, field: string): string => {
  const output = requiredString(value, field, 128);
  if (!TOOL_IDS.has(output)) {
    throw new NativeAgentContractError(`${field} est inconnu.`);
  }
  return output;
};

const permission = (value: unknown, field: string): AgentPermission => {
  const output = requiredString(value, field, 64);
  if (!PERMISSIONS.has(output as AgentPermission)) {
    throw new NativeAgentContractError(`${field} est inconnu.`);
  }
  return output as AgentPermission;
};

const normalizeOwnerId = (value: unknown, field: string): string => {
  if (
    typeof value !== 'string' ||
    [...value].some((character) => {
      const codePoint = character.codePointAt(0) ?? 0;
      return (
        codePoint <= 31 ||
        (codePoint >= 127 && codePoint <= 159) ||
        codePoint === 8_232 ||
        codePoint === 8_233
      );
    })
  ) {
    throw new NativeAgentContractError(
      `${field} contient des caractères de contrôle ou de saut de ligne.`,
    );
  }
  const output = requiredString(value, field, 256);
  return output;
};

const jsonValue = (value: unknown, field: string, depth = 0): JSONValue => {
  if (depth > 8) {
    throw new NativeAgentContractError(`${field} est trop imbriqué.`);
  }
  if (value === null || typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new NativeAgentContractError(
        `${field} contient un nombre invalide.`,
      );
    }
    return value;
  }
  if (typeof value === 'string') {
    if (utf8ByteLength(value) > 32_000) {
      throw new NativeAgentContractError(
        `${field} contient une chaîne trop longue.`,
      );
    }
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length > 256) {
      throw new NativeAgentContractError(`${field} contient trop d'éléments.`);
    }
    return value.map((item, index) =>
      jsonValue(item, `${field}[${index}]`, depth + 1),
    );
  }
  const source = record(value, field);
  const entries = Object.entries(source);
  if (entries.length > 128) {
    throw new NativeAgentContractError(`${field} contient trop de champs.`);
  }
  return Object.fromEntries(
    entries.map(([key, item]) => {
      if (!key || key.length > 128) {
        throw new NativeAgentContractError(
          `${field} contient une clé invalide.`,
        );
      }
      return [key, jsonValue(item, `${field}.${key}`, depth + 1)];
    }),
  );
};

const jsonObject = (value: unknown, field: string): JSONObject => {
  const normalized = jsonValue(value, field);
  if (
    !normalized ||
    Array.isArray(normalized) ||
    typeof normalized !== 'object'
  ) {
    throw new NativeAgentContractError(`${field} doit être un objet JSON.`);
  }
  return normalized;
};

const isoDate = (value: unknown, field: string): string => {
  const output = requiredString(value, field, 64);
  const parsed = new Date(output);
  if (Number.isNaN(parsed.getTime())) {
    throw new NativeAgentContractError(`${field} doit être une date ISO.`);
  }
  return parsed.toISOString();
};

const normalizeEvent = (value: unknown, index: number): NativeAgentEvent => {
  const source = record(value, `events[${index}]`);
  const type = requiredString(source.type, `events[${index}].type`, 64);
  if (!EVENT_TYPES.has(type as NativeAgentEvent['type'])) {
    throw new NativeAgentContractError(`events[${index}].type est inconnu.`);
  }
  const event: NativeAgentEvent = { type: type as NativeAgentEvent['type'] };
  if (source.toolId !== undefined) {
    event.toolId = toolId(source.toolId, `events[${index}].toolId`);
  }
  if (source.status !== undefined) {
    event.status = requiredString(source.status, `events[${index}].status`, 64);
  }
  if (source.stepIndex !== undefined) {
    event.stepIndex = integer(
      source.stepIndex,
      `events[${index}].stepIndex`,
      0,
      7,
    );
  }
  if (source.message !== undefined) {
    event.message = requiredString(
      source.message,
      `events[${index}].message`,
      1_000,
    );
  }
  return event;
};

const normalizeApproval = (value: unknown): NativeAgentApproval => {
  const source = record(value, 'approval');
  const risk = requiredString(source.risk, 'approval.risk', 16);
  if (!['low', 'moderate', 'high', 'critical'].includes(risk)) {
    throw new NativeAgentContractError('approval.risk est invalide.');
  }
  return {
    recordId: uuid(source.recordId, 'approval.recordId'),
    toolId: toolId(source.toolId, 'approval.toolId'),
    arguments: jsonObject(source.arguments, 'approval.arguments'),
    displayName: requiredString(
      source.displayName,
      'approval.displayName',
      256,
    ),
    risk: risk as NativeAgentApproval['risk'],
    expiresAt: isoDate(source.expiresAt, 'approval.expiresAt'),
  };
};

const normalizeRunResult = (
  value: unknown,
  expectedRunId: string,
): NativeAgentRunResult => {
  const source = record(value, 'result');
  const runId = uuid(source.runId, 'result.runId');
  if (runId !== expectedRunId.toLowerCase()) {
    throw new NativeAgentContractError(
      'result.runId ne correspond pas à la requête.',
    );
  }
  const intent = requiredString(source.intent, 'result.intent', 64);
  if (!INTENTS.has(intent as AgentIntent)) {
    throw new NativeAgentContractError('result.intent est inconnu.');
  }
  if (!Array.isArray(source.events) || source.events.length > 256) {
    throw new NativeAgentContractError('result.events est invalide.');
  }
  const base: NativeAgentRunBase = {
    runId,
    intent: intent as AgentIntent,
    events: source.events.map(normalizeEvent),
    executedToolCount: integer(
      source.executedToolCount,
      'result.executedToolCount',
      0,
      8,
    ),
    modelTurnCount: integer(
      source.modelTurnCount,
      'result.modelTurnCount',
      0,
      16,
    ),
    usedRepairAttempt: boolean(
      source.usedRepairAttempt,
      'result.usedRepairAttempt',
    ),
  };
  const status = requiredString(source.status, 'result.status', 64);
  switch (status) {
    case 'final':
    case 'clarification':
    case 'unavailable':
      return {
        ...base,
        status,
        message: requiredString(source.message, 'result.message', 16_000),
      };
    case 'approval_required':
      return { ...base, status, approval: normalizeApproval(source.approval) };
    case 'permission_required':
      return {
        ...base,
        status,
        permission: permission(source.permission, 'result.permission'),
        message: requiredString(source.message, 'result.message', 1_000),
      };
    case 'failed':
    case 'cancelled':
      return {
        ...base,
        status,
        code: requiredString(source.code, 'result.code', 128),
        message: requiredString(source.message, 'result.message', 2_000),
      };
    default:
      throw new NativeAgentContractError('result.status est inconnu.');
  }
};

const assertRunRequest = (
  request: NativeAgentRunRequest,
): NativeAgentRunRequest => {
  const runId = uuid(request.runId, 'request.runId');
  const ownerId = normalizeOwnerId(request.ownerId, 'request.ownerId');
  const prompt = requiredString(request.prompt, 'request.prompt', 128_000);
  if (!Array.isArray(request.history) || request.history.length > 50) {
    throw new NativeAgentContractError('request.history est invalide.');
  }
  const history = request.history.map((message, index) => {
    if (!message || (message.role !== 'user' && message.role !== 'assistant')) {
      throw new NativeAgentContractError(
        `request.history[${index}].role est invalide.`,
      );
    }
    return {
      role: message.role,
      content: requiredString(
        message.content,
        `request.history[${index}].content`,
        64_000,
      ),
    };
  });
  const approvalRecordId =
    request.approvalRecordId === undefined
      ? undefined
      : uuid(request.approvalRecordId, 'request.approvalRecordId');
  const requestedIntent = request.requestedIntent;
  const rawAllowedToolIds = request.allowedToolIds;
  if ((requestedIntent === undefined) !== (rawAllowedToolIds === undefined)) {
    throw new NativeAgentContractError(
      'request.requestedIntent et request.allowedToolIds doivent être fournis ensemble.',
    );
  }
  let allowedToolIds: string[] | undefined;
  if (requestedIntent !== undefined && rawAllowedToolIds !== undefined) {
    if (
      !INTENTS.has(requestedIntent) ||
      ['chat', 'unknown'].includes(requestedIntent)
    ) {
      throw new NativeAgentContractError(
        'request.requestedIntent ne peut pas exécuter cet outil.',
      );
    }
    if (
      !Array.isArray(rawAllowedToolIds) ||
      rawAllowedToolIds.length < 1 ||
      rawAllowedToolIds.length > TOOL_IDS.size
    ) {
      throw new NativeAgentContractError(
        'request.allowedToolIds est invalide.',
      );
    }
    allowedToolIds = rawAllowedToolIds.map((value, index) =>
      toolId(value, `request.allowedToolIds[${index}]`),
    );
    if (new Set(allowedToolIds).size !== allowedToolIds.length) {
      throw new NativeAgentContractError(
        'request.allowedToolIds contient des doublons.',
      );
    }
    const intentToolIds = new Set(INTENT_TOOL_IDS[requestedIntent]);
    if (allowedToolIds.some((tool) => !intentToolIds.has(tool))) {
      throw new NativeAgentContractError(
        "request.allowedToolIds est incompatible avec l'intention.",
      );
    }
    if (approvalRecordId) {
      throw new NativeAgentContractError(
        'une reprise approuvée ne peut pas modifier sa portée d outils.',
      );
    }
  }
  const maxSteps =
    request.maxSteps === undefined
      ? undefined
      : integer(request.maxSteps, 'request.maxSteps', 1, 8);
  return {
    runId,
    ownerId,
    prompt,
    history,
    ...(requestedIntent ? { requestedIntent } : {}),
    ...(allowedToolIds ? { allowedToolIds } : {}),
    ...(approvalRecordId ? { approvalRecordId } : {}),
    ...(maxSteps ? { maxSteps } : {}),
  };
};

const assertBinding = (
  binding: NativeAgentApprovalBinding,
): NativeAgentApprovalBinding => {
  const ownerId = normalizeOwnerId(binding.ownerId, 'binding.ownerId');
  return {
    recordId: uuid(binding.recordId, 'binding.recordId'),
    ownerId,
    prompt: requiredString(binding.prompt, 'binding.prompt', 128_000),
    toolId: toolId(binding.toolId, 'binding.toolId'),
    arguments: jsonObject(binding.arguments, 'binding.arguments'),
    expiresAt: isoDate(binding.expiresAt, 'binding.expiresAt'),
  };
};

const normalizeApprovalResult = (
  value: unknown,
  expectedRecordId: string,
): NativeAgentApprovalResult => {
  const source = record(value, 'approvalResult');
  const recordId = uuid(source.recordId, 'approvalResult.recordId');
  if (recordId !== expectedRecordId.toLowerCase()) {
    throw new NativeAgentContractError(
      'approvalResult.recordId ne correspond pas.',
    );
  }
  const status = requiredString(source.status, 'approvalResult.status', 32);
  if (!['approved', 'rejected', 'expired'].includes(status)) {
    throw new NativeAgentContractError('approvalResult.status est invalide.');
  }
  return { recordId, status: status as NativeAgentApprovalResult['status'] };
};

export const createNativeAgentRunId = (): string =>
  'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (token) => {
    const random = Math.floor(Math.random() * 16);
    const value = token === 'x' ? random : (random % 4) + 8;
    return value.toString(16);
  });

export const getNativeAgentCapabilities = async (
  ownerId: string,
): Promise<NativeAgentCapabilities> => {
  const normalizedOwner = normalizeOwnerId(ownerId, 'ownerId');
  const source = record(
    await requireModule().getAgentCapabilities(normalizedOwner),
    'capabilities',
  );
  if (!Array.isArray(source.toolIds) || source.toolIds.length > TOOL_IDS.size) {
    throw new NativeAgentContractError('capabilities.toolIds est invalide.');
  }
  const toolIds = source.toolIds.map((value, index) =>
    toolId(value, `capabilities.toolIds[${index}]`),
  );
  if (new Set(toolIds).size !== toolIds.length) {
    throw new NativeAgentContractError(
      'capabilities.toolIds contient des doublons.',
    );
  }
  const toolCount = integer(source.toolCount, 'capabilities.toolCount', 0, 53);
  if (toolCount !== toolIds.length) {
    throw new NativeAgentContractError(
      'capabilities.toolCount ne correspond pas.',
    );
  }
  if (source.available !== true || source.supportsApprovals !== true) {
    throw new NativeAgentContractError(
      'capabilities annonce un moteur incomplet.',
    );
  }
  return {
    available: true,
    toolIds,
    toolCount,
    supportsApprovals: true,
    maximumSteps: integer(
      source.maximumSteps,
      'capabilities.maximumSteps',
      1,
      8,
    ),
  };
};

export const runNativeAgent = async (
  request: NativeAgentRunRequest,
): Promise<NativeAgentRunResult> => {
  const normalized = assertRunRequest(request);
  return normalizeRunResult(
    await requireModule().runAgent(normalized),
    normalized.runId,
  );
};

export const requestNativeAgentPermission = async (
  requestedPermission: AgentPermission,
): Promise<NativeAgentPermissionResult> => {
  const normalizedPermission = permission(requestedPermission, 'permission');
  const source = record(
    await requireModule().requestAgentPermission(normalizedPermission),
    'permissionResult',
  );
  const returnedPermission = permission(
    source.permission,
    'permissionResult.permission',
  );
  if (returnedPermission !== normalizedPermission) {
    throw new NativeAgentContractError(
      'permissionResult.permission ne correspond pas.',
    );
  }
  const state = requiredString(source.state, 'permissionResult.state', 32);
  if (
    ![
      'granted',
      'limited',
      'notDetermined',
      'denied',
      'restricted',
      'unavailable',
    ].includes(state)
  ) {
    throw new NativeAgentContractError('permissionResult.state est invalide.');
  }
  return {
    permission: returnedPermission,
    state: state as NativeAgentPermissionResult['state'],
  };
};

export const getPendingNativeAgentTrigger = async (
  rawOwnerId: string,
): Promise<NativeAgentTriggerHandoff | null> => {
  const normalizedOwnerId = normalizeOwnerId(rawOwnerId, 'ownerId');
  const value = await requireModule().getPendingAgentTrigger(normalizedOwnerId);
  if (value === null || value === undefined) {
    return null;
  }
  const source = record(value, 'triggerHandoff');
  return {
    id: uuid(source.id, 'triggerHandoff.id'),
    title: requiredString(source.title, 'triggerHandoff.title', 1_000),
    prompt: requiredString(source.prompt, 'triggerHandoff.prompt', 32_000),
    repeats: boolean(source.repeats, 'triggerHandoff.repeats'),
  };
};

export const normalizeNativeAgentTriggerSignal = (
  value: unknown,
): NativeAgentTriggerSignal => {
  const source = record(value, 'triggerSignal');
  return {
    id: uuid(source.id, 'triggerSignal.id'),
    tappedAt: isoDate(source.tappedAt, 'triggerSignal.tappedAt'),
  };
};

export const subscribeNativeAgentTriggerHandoff = (
  listener: (signal: NativeAgentTriggerSignal) => void,
): (() => void) => {
  const module = requireModule();
  nativeAgentEventEmitter ??= new NativeEventEmitter(module);
  const subscription = nativeAgentEventEmitter.addListener(
    'onAgentTriggerHandoff',
    (value: unknown) => {
      try {
        listener(normalizeNativeAgentTriggerSignal(value));
      } catch (error) {
        console.debug('[nativeAgent] invalid trigger signal ignored', error);
      }
    },
  );
  return () => subscription.remove();
};

export const acknowledgePendingNativeAgentTrigger = async (
  rawOwnerId: string,
  rawId: string,
): Promise<boolean> => {
  const normalizedOwnerId = normalizeOwnerId(rawOwnerId, 'ownerId');
  const normalizedId = uuid(rawId, 'id');
  const source = record(
    await requireModule().acknowledgePendingAgentTrigger({
      ownerId: normalizedOwnerId,
      id: normalizedId,
    }),
    'triggerAcknowledgement',
  );
  const returnedId = uuid(source.id, 'triggerAcknowledgement.id');
  if (returnedId !== normalizedId) {
    throw new NativeAgentContractError(
      'triggerAcknowledgement.id ne correspond pas.',
    );
  }
  return source.acknowledged === true;
};

export const approveNativeAgent = async (
  binding: NativeAgentApprovalBinding,
): Promise<NativeAgentApprovalResult> => {
  const normalized = assertBinding(binding);
  return normalizeApprovalResult(
    await requireModule().approveAgent(normalized),
    normalized.recordId,
  );
};

export const rejectNativeAgent = async (
  binding: NativeAgentApprovalBinding,
): Promise<NativeAgentApprovalResult> => {
  const normalized = assertBinding(binding);
  return normalizeApprovalResult(
    await requireModule().rejectAgent(normalized),
    normalized.recordId,
  );
};

export const cancelNativeAgent = async (runId: string): Promise<boolean> => {
  const normalized = uuid(runId, 'runId');
  const source = record(
    await requireModule().cancelAgent(normalized),
    'cancel',
  );
  return source.cancelled === true;
};
