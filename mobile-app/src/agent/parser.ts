import type { AgentTurn, JSONObject } from './types';

export type AgentTurnParseErrorCode =
  | 'output_too_large'
  | 'invalid_fence'
  | 'invalid_json'
  | 'duplicate_key'
  | 'root_not_object'
  | 'extra_top_level_keys'
  | 'invalid_thought'
  | 'missing_decision'
  | 'mutually_exclusive_decision'
  | 'action_not_object'
  | 'extra_action_keys'
  | 'missing_tool'
  | 'invalid_tool'
  | 'missing_args'
  | 'args_not_object'
  | 'invalid_final';

export class AgentTurnParseError extends Error {
  constructor(
    readonly code: AgentTurnParseErrorCode,
    message: string,
  ) {
    super(message);
    this.name = 'AgentTurnParseError';
  }
}

const MAXIMUM_MODEL_OUTPUT_CHARACTERS = 64_000;

const isObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const extractJSON = (raw: string): string => {
  const trimmed = raw.trim();
  if (trimmed.length > MAXIMUM_MODEL_OUTPUT_CHARACTERS) {
    throw new AgentTurnParseError(
      'output_too_large',
      'Agent output exceeds the size limit.',
    );
  }
  if (!trimmed.startsWith('```')) {
    return trimmed;
  }
  const match = /^```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```$/i.exec(trimmed);
  if (!match) {
    throw new AgentTurnParseError(
      'invalid_fence',
      'A fenced response must contain exactly one JSON block and no prose.',
    );
  }
  return match[1].trim();
};

/** JSON.parse accepts duplicate keys; this scanner rejects them before policy evaluation. */
class DuplicateKeyScanner {
  private index = 0;

  constructor(private readonly source: string) {}

  scan(): void {
    this.skipWhitespace();
    this.scanValue();
    this.skipWhitespace();
    if (this.index !== this.source.length) {
      throw new Error('trailing input');
    }
  }

  private scanValue(): void {
    this.skipWhitespace();
    const character = this.source[this.index];
    if (character === '{') {
      this.scanObject();
    } else if (character === '[') {
      this.scanArray();
    } else if (character === '"') {
      this.scanString();
    } else if (character === 't') {
      this.consumeLiteral('true');
    } else if (character === 'f') {
      this.consumeLiteral('false');
    } else if (character === 'n') {
      this.consumeLiteral('null');
    } else {
      this.scanNumber();
    }
  }

  private scanObject(): void {
    this.index += 1;
    this.skipWhitespace();
    if (this.source[this.index] === '}') {
      this.index += 1;
      return;
    }
    const keys = new Set<string>();
    while (this.index < this.source.length) {
      this.skipWhitespace();
      const key = this.scanString();
      if (keys.has(key)) {
        throw new AgentTurnParseError(
          'duplicate_key',
          `Duplicate JSON key: ${key}.`,
        );
      }
      keys.add(key);
      this.skipWhitespace();
      this.expect(':');
      this.scanValue();
      this.skipWhitespace();
      const next = this.source[this.index];
      this.index += 1;
      if (next === '}') {
        return;
      }
      if (next !== ',') {
        throw new Error('invalid object');
      }
    }
    throw new Error('unterminated object');
  }

  private scanArray(): void {
    this.index += 1;
    this.skipWhitespace();
    if (this.source[this.index] === ']') {
      this.index += 1;
      return;
    }
    while (this.index < this.source.length) {
      this.scanValue();
      this.skipWhitespace();
      const next = this.source[this.index];
      this.index += 1;
      if (next === ']') {
        return;
      }
      if (next !== ',') {
        throw new Error('invalid array');
      }
    }
    throw new Error('unterminated array');
  }

  private scanString(): string {
    const start = this.index;
    this.expect('"');
    while (this.index < this.source.length) {
      const character = this.source[this.index];
      this.index += 1;
      if (character === '\\') {
        this.index += 1;
      } else if (character === '"') {
        return JSON.parse(this.source.slice(start, this.index)) as string;
      }
    }
    throw new Error('unterminated string');
  }

  private scanNumber(): void {
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(
      this.source.slice(this.index),
    );
    if (!match) {
      throw new Error('invalid number');
    }
    this.index += match[0].length;
  }

  private consumeLiteral(literal: string): void {
    if (
      this.source.slice(this.index, this.index + literal.length) !== literal
    ) {
      throw new Error('invalid literal');
    }
    this.index += literal.length;
  }

  private skipWhitespace(): void {
    while (/\s/.test(this.source[this.index] ?? '')) {
      this.index += 1;
    }
  }

  private expect(character: string): void {
    if (this.source[this.index] !== character) {
      throw new Error(`expected ${character}`);
    }
    this.index += 1;
  }
}

export const parseAgentTurn = (raw: string): AgentTurn => {
  const source = extractJSON(raw);
  let decoded: unknown;
  try {
    decoded = JSON.parse(source);
    new DuplicateKeyScanner(source).scan();
  } catch (error) {
    if (error instanceof AgentTurnParseError) {
      throw error;
    }
    throw new AgentTurnParseError(
      'invalid_json',
      'Output is not one complete JSON value.',
    );
  }
  if (!isObject(decoded)) {
    throw new AgentTurnParseError(
      'root_not_object',
      'Agent output root must be an object.',
    );
  }

  const extraTopLevel = Object.keys(decoded)
    .filter((key) => !['thought', 'action', 'final'].includes(key))
    .sort();
  if (extraTopLevel.length > 0) {
    throw new AgentTurnParseError(
      'extra_top_level_keys',
      `Unexpected top-level keys: ${extraTopLevel.join(', ')}.`,
    );
  }
  if ('thought' in decoded && typeof decoded.thought !== 'string') {
    throw new AgentTurnParseError(
      'invalid_thought',
      'Optional thought must be a string.',
    );
  }

  const hasAction = Object.prototype.hasOwnProperty.call(decoded, 'action');
  const hasFinal = Object.prototype.hasOwnProperty.call(decoded, 'final');
  if (!hasAction && !hasFinal) {
    throw new AgentTurnParseError(
      'missing_decision',
      'Output must contain exactly one action or final field.',
    );
  }
  if (hasAction && hasFinal) {
    throw new AgentTurnParseError(
      'mutually_exclusive_decision',
      'Output cannot contain both action and final.',
    );
  }

  if (hasAction) {
    if (!isObject(decoded.action)) {
      throw new AgentTurnParseError(
        'action_not_object',
        'Action must be an object.',
      );
    }
    const extraAction = Object.keys(decoded.action)
      .filter((key) => !['tool', 'args'].includes(key))
      .sort();
    if (extraAction.length > 0) {
      throw new AgentTurnParseError(
        'extra_action_keys',
        `Unexpected action keys: ${extraAction.join(', ')}.`,
      );
    }
    if (!Object.prototype.hasOwnProperty.call(decoded.action, 'tool')) {
      throw new AgentTurnParseError('missing_tool', 'Action is missing tool.');
    }
    if (
      typeof decoded.action.tool !== 'string' ||
      decoded.action.tool.trim() === ''
    ) {
      throw new AgentTurnParseError(
        'invalid_tool',
        'Action tool must be a non-empty string.',
      );
    }
    if (!Object.prototype.hasOwnProperty.call(decoded.action, 'args')) {
      throw new AgentTurnParseError('missing_args', 'Action is missing args.');
    }
    if (!isObject(decoded.action.args)) {
      throw new AgentTurnParseError(
        'args_not_object',
        'Action args must be an object.',
      );
    }
    return {
      kind: 'action',
      action: {
        tool: decoded.action.tool.trim(),
        args: decoded.action.args as JSONObject,
      },
    };
  }

  if (typeof decoded.final !== 'string' || decoded.final.trim() === '') {
    throw new AgentTurnParseError(
      'invalid_final',
      'Final must be a non-empty string.',
    );
  }
  return { kind: 'final', final: decoded.final };
};
