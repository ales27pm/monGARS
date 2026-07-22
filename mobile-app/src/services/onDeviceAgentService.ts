import { routeAgentIntent } from '../agent/routing';
import type { AgentIntent } from '../agent/types';
import {
  createNativeAgentRunId,
  runNativeAgent,
  type NativeAgentHistoryMessage,
  type NativeAgentRunResult,
} from '../native/agent';

export type OnDeviceAgentContext = {
  ownerId: string;
  prompt: string;
  history: NativeAgentHistoryMessage[];
  requestedIntent?: AgentIntent;
  allowedToolIds?: string[];
  approvalRecordId?: string;
};

/**
 * Keeps ordinary conversation on the existing streaming generator while all
 * deterministic tool and clarification routes stay inside the native agent.
 */
export const shouldUseNativeAgent = (prompt: string): boolean => {
  const route = routeAgentIntent(prompt);
  return route.requiresTool || Boolean(route.clarification);
};

export const executeNativeAgent = async (
  context: OnDeviceAgentContext,
  runId = createNativeAgentRunId(),
): Promise<NativeAgentRunResult> =>
  runNativeAgent({
    runId,
    ownerId: context.ownerId,
    prompt: context.prompt,
    history: context.history,
    ...(context.requestedIntent && context.allowedToolIds
      ? {
          requestedIntent: context.requestedIntent,
          allowedToolIds: context.allowedToolIds,
        }
      : {}),
    ...(context.approvalRecordId
      ? { approvalRecordId: context.approvalRecordId }
      : {}),
  });
