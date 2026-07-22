export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONValue[] | JSONObject;
export type JSONObject = { [key: string]: JSONValue };

export type AgentArgumentType =
  | 'string'
  | 'number'
  | 'bool'
  | 'array'
  | 'object'
  | 'enum';

export type AgentToolCategory =
  | 'productivity'
  | 'communication'
  | 'location'
  | 'media'
  | 'health'
  | 'knowledge';

export type AgentPermission =
  | 'calendar'
  | 'reminders'
  | 'contacts'
  | 'location'
  | 'photos'
  | 'camera'
  | 'health'
  | 'motion'
  | 'alarms'
  | 'notifications';

export type AgentToolRisk = 'low' | 'moderate' | 'high' | 'critical';

export interface AgentArgumentSchema {
  readonly name: string;
  readonly type: AgentArgumentType;
  readonly required: boolean;
  readonly allowedValues?: readonly string[];
}

export interface AgentToolDefinition {
  readonly id: string;
  readonly displayName: string;
  readonly description: string;
  readonly category: AgentToolCategory;
  readonly arguments: readonly AgentArgumentSchema[];
  readonly permission?: AgentPermission;
  readonly risk: AgentToolRisk;
  readonly requiresApproval: boolean;
  readonly supportsBackgroundExecution: boolean;
  readonly maximumOutputCharacters: number;
}

export interface AgentToolCall {
  readonly tool: string;
  readonly args: JSONObject;
}

/** Private model reasoning is deliberately absent from this public type. */
export type AgentTurn =
  | { readonly kind: 'action'; readonly action: AgentToolCall }
  | { readonly kind: 'final'; readonly final: string };

export type AgentIntent =
  | 'weather'
  | 'webSearch'
  | 'emailDraft'
  | 'messageDraft'
  | 'phoneCall'
  | 'contactSearch'
  | 'calendar'
  | 'reminder'
  | 'maps'
  | 'photos'
  | 'camera'
  | 'health'
  | 'motion'
  | 'files'
  | 'memory'
  | 'rag'
  | 'trigger'
  | 'alarm'
  | 'outlook'
  | 'note'
  | 'chat'
  | 'unknown';

export interface AgentIntentRoute {
  readonly intent: AgentIntent;
  readonly allowedToolIds: ReadonlySet<string>;
  readonly fulfillmentToolIds: ReadonlySet<string>;
  readonly clarification?: string;
  readonly requiresTool: boolean;
}

export interface ValidatedAgentToolCall {
  readonly tool: string;
  readonly args: JSONObject;
  readonly definition: AgentToolDefinition;
}
