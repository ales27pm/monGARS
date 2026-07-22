import {
  AGENT_TOOL_CATALOG,
  AgentToolValidationError,
  canonicalToolId,
  duplicateCallKey,
  validateToolCall,
} from './catalog';
import { parseAgentTurn, AgentTurnParseError } from './parser';
import { buildAgentSystemPrompt } from './prompt';
import { routeAgentIntent, unavailableIntentMessage } from './routing';
import {
  sanitizeAgentText,
  sanitizeError,
  stringifyObservation,
} from './sanitizer';
import type {
  AgentIntent,
  AgentToolDefinition,
  JSONObject,
  JSONValue,
  ValidatedAgentToolCall,
} from './types';

export interface AgentObservation {
  readonly tool: string;
  readonly output: string;
  readonly succeeded: boolean;
}

export interface AgentModelRequest {
  readonly systemPrompt: string;
  readonly userInput: string;
  readonly observations: readonly AgentObservation[];
  readonly repair?: string;
  readonly decisionNumber: number;
}

export type AgentModelCallback = (
  request: AgentModelRequest,
) => Promise<string>;
export type AgentToolCallback = (
  call: ValidatedAgentToolCall,
) => Promise<JSONValue>;

interface BaseResult {
  readonly intent: AgentIntent;
  readonly decisionCount: number;
}

export type AgentRunResult =
  | (BaseResult & { readonly status: 'final'; readonly message: string })
  | (BaseResult & {
      readonly status: 'clarification';
      readonly message: string;
    })
  | (BaseResult & {
      readonly status: 'pendingApproval';
      readonly approval: {
        readonly tool: string;
        readonly args: JSONObject;
        readonly displayName: string;
        readonly risk: AgentToolDefinition['risk'];
      };
    })
  | (BaseResult & {
      readonly status: 'error';
      readonly code:
        | 'tools_unavailable'
        | 'model_failure'
        | 'invalid_decision'
        | 'duplicate_call'
        | 'background_policy'
        | 'step_limit';
      readonly message: string;
    });

export interface RunAgentOptions {
  readonly userInput: string;
  readonly model: AgentModelCallback;
  readonly executeTool: AgentToolCallback;
  readonly availableToolIds?: ReadonlySet<string>;
  readonly maximumDecisions?: number;
  readonly mode?: 'foreground' | 'background';
}

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

const filteredDefinitions = (
  routedIds: ReadonlySet<string>,
  availableIds: ReadonlySet<string>,
): AgentToolDefinition[] => {
  const available = new Set([...availableIds].map(canonicalToolId));
  return AGENT_TOOL_CATALOG.filter(
    (definition) =>
      routedIds.has(definition.id) && available.has(definition.id),
  );
};

const policyError = (
  definition: AgentToolDefinition,
  mode: 'foreground' | 'background',
): string | undefined => {
  if (mode !== 'background') {
    return undefined;
  }
  if (!definition.supportsBackgroundExecution) {
    return `Tool cannot run in the background: ${definition.id}.`;
  }
  if (definition.requiresApproval) {
    return `Approval-requiring tool cannot run in the background: ${definition.id}.`;
  }
  if (definition.risk === 'high' || definition.risk === 'critical') {
    return `Tool risk is too high for background execution: ${definition.id}.`;
  }
  return undefined;
};

const repairDiagnostic = (error: unknown): string => {
  if (
    error instanceof AgentTurnParseError ||
    error instanceof AgentToolValidationError
  ) {
    return sanitizeAgentText(error.message, 500);
  }
  return 'The prior decision violated the agent contract.';
};

/**
 * Bounded policy kernel. It has no side effects except through the injected,
 * validated tool callback and always stops before approval-requiring actions.
 */
export const runAgent = async (
  options: RunAgentOptions,
): Promise<AgentRunResult> => {
  const route = routeAgentIntent(options.userInput);
  if (route.clarification) {
    return {
      status: 'clarification',
      intent: route.intent,
      message: route.clarification,
      decisionCount: 0,
    };
  }
  if (route.requiresTool && utf8ByteLength(options.userInput) > 512) {
    return {
      status: 'clarification',
      intent: route.intent,
      message:
        "This on-device tool request is too long for the pinned model's strict JSON output budget. Shorten it to 512 UTF-8 bytes or fewer; no tool was executed.",
      decisionCount: 0,
    };
  }

  const availableToolIds =
    options.availableToolIds ??
    new Set(AGENT_TOOL_CATALOG.map((definition) => definition.id));
  const definitions = filteredDefinitions(
    route.allowedToolIds,
    availableToolIds,
  );
  const availableFulfillmentIds = new Set(
    definitions
      .map((definition) => definition.id)
      .filter((id) => route.fulfillmentToolIds.has(id)),
  );
  if (route.requiresTool && availableFulfillmentIds.size === 0) {
    return {
      status: 'error',
      code: 'tools_unavailable',
      intent: route.intent,
      message: unavailableIntentMessage(route.intent),
      decisionCount: 0,
    };
  }

  const routedAvailableIds = new Set(
    definitions.map((definition) => definition.id),
  );
  const systemPrompt = buildAgentSystemPrompt(definitions, route.requiresTool);
  const maximumDecisions = Math.min(
    8,
    Math.max(1, options.maximumDecisions ?? 4),
  );
  const observations: AgentObservation[] = [];
  const executedCalls = new Set<string>();
  let decisionCount = 0;
  let repairUsed = false;
  let pendingRepair: string | undefined;
  let fulfilledRoute = false;

  while (decisionCount < maximumDecisions) {
    decisionCount += 1;
    let raw: string;
    try {
      raw = await options.model({
        systemPrompt,
        userInput: options.userInput,
        observations: [...observations],
        repair: pendingRepair,
        decisionNumber: decisionCount,
      });
    } catch (error) {
      return {
        status: 'error',
        code: 'model_failure',
        intent: route.intent,
        message: `Local model failed: ${sanitizeError(error)}.`,
        decisionCount,
      };
    }
    pendingRepair = undefined;

    try {
      const turn = parseAgentTurn(raw);
      if (turn.kind === 'final') {
        if (route.requiresTool && !fulfilledRoute) {
          throw new AgentTurnParseError(
            'missing_decision',
            'A routed tool action is required before final.',
          );
        }
        const message = sanitizeAgentText(turn.final, 4_000);
        if (!message) {
          throw new AgentTurnParseError(
            'invalid_final',
            'Final is empty after sanitization.',
          );
        }
        return {
          status: 'final',
          intent: route.intent,
          message,
          decisionCount,
        };
      }

      const call = validateToolCall(
        turn.action.tool,
        turn.action.args,
        routedAvailableIds,
      );
      const duplicateKey = duplicateCallKey(call);
      if (executedCalls.has(duplicateKey)) {
        return {
          status: 'error',
          code: 'duplicate_call',
          intent: route.intent,
          message: `Duplicate tool call blocked: ${call.tool}.`,
          decisionCount,
        };
      }

      const backgroundDenial = policyError(
        call.definition,
        options.mode ?? 'foreground',
      );
      if (backgroundDenial) {
        return {
          status: 'error',
          code: 'background_policy',
          intent: route.intent,
          message: backgroundDenial,
          decisionCount,
        };
      }
      if (call.definition.requiresApproval) {
        return {
          status: 'pendingApproval',
          intent: route.intent,
          decisionCount,
          approval: {
            tool: call.tool,
            args: call.args,
            displayName: call.definition.displayName,
            risk: call.definition.risk,
          },
        };
      }

      executedCalls.add(duplicateKey);
      try {
        const result = await options.executeTool(call);
        observations.push({
          tool: call.tool,
          output: sanitizeAgentText(
            stringifyObservation(result),
            call.definition.maximumOutputCharacters,
          ),
          succeeded: true,
        });
        if (route.fulfillmentToolIds.has(call.tool)) {
          fulfilledRoute = true;
        }
      } catch (error) {
        observations.push({
          tool: call.tool,
          output: `Tool failed safely: ${sanitizeError(error, call.definition.maximumOutputCharacters)}`,
          succeeded: false,
        });
      }
    } catch (error) {
      if (!repairUsed && decisionCount < maximumDecisions) {
        repairUsed = true;
        pendingRepair = `${repairDiagnostic(error)} Return one corrected JSON decision only.`;
        continue;
      }
      return {
        status: 'error',
        code: 'invalid_decision',
        intent: route.intent,
        message: `Agent decision rejected: ${repairDiagnostic(error)}`,
        decisionCount,
      };
    }
  }

  return {
    status: 'error',
    code: 'step_limit',
    intent: route.intent,
    message: 'Agent stopped at the decision limit.',
    decisionCount,
  };
};
