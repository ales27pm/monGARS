import type { JSONObject, JSONValue } from './types';

export type AgentApprovalArgumentSummary = {
  key: string;
  value: string;
};

const SECRET_KEY = /(authorization|password|secret|token|credential|api.?key)/i;
const PRIORITY = [
  'id',
  'to',
  'recipient',
  'number',
  'destination',
  'title',
  'subject',
  'startsInMinutes',
  'inMinutes',
  'timestamp',
  'durationSeconds',
  'schedule',
];

const redactNested = (value: JSONValue): JSONValue => {
  if (Array.isArray(value)) {
    return value.map(redactNested);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        SECRET_KEY.test(key) ? '[masqué]' : redactNested(item),
      ]),
    );
  }
  return value;
};

const displayValue = (key: string, value: JSONValue): string => {
  if (SECRET_KEY.test(key)) {
    return '[masqué]';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (value === null) {
    return 'null';
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(redactNested(value));
};

/** Full local-only action detail; credential-shaped fields remain redacted. */
export const summarizeAgentApprovalArguments = (
  argumentsObject: JSONObject,
): AgentApprovalArgumentSummary[] =>
  Object.entries(argumentsObject)
    .sort(([left], [right]) => {
      const leftPriority = PRIORITY.indexOf(left);
      const rightPriority = PRIORITY.indexOf(right);
      if (leftPriority === -1 && rightPriority === -1) {
        return left.localeCompare(right);
      }
      if (leftPriority === -1) {
        return 1;
      }
      if (rightPriority === -1) {
        return -1;
      }
      return leftPriority - rightPriority;
    })
    .map(([key, value]) => ({ key, value: displayValue(key, value) }));
