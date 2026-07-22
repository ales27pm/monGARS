import type { AgentToolDefinition } from './types';

const schemaFor = (definition: AgentToolDefinition): string => {
  if (definition.arguments.length === 0) {
    return '{}';
  }
  const fields = definition.arguments.map((argument) => {
    const type =
      argument.type === 'enum'
        ? (argument.allowedValues?.join('|') ?? 'string')
        : argument.type;
    return `${argument.name}${argument.required ? '' : '?'}:${type}`;
  });
  return `{${fields.join(',')}}`;
};

/** Compact enough for the 2,048-token CoreML model while preserving exact schemas. */
export const buildAgentSystemPrompt = (
  definitions: readonly AgentToolDefinition[],
  requiresTool: boolean,
): string => {
  const tools = definitions
    .map(
      (definition) =>
        `- ${definition.id}${schemaFor(definition)}${definition.requiresApproval ? ' [approval]' : ''}: ${definition.description}`,
    )
    .join('\n');
  const toolInstruction = requiresTool
    ? 'This request requires a tool action before final.'
    : 'A final answer is allowed without a tool.';
  return [
    'You are the monGARS local agent. Return exactly one JSON object, with no prose or markdown.',
    'Use {"action":{"tool":"id","args":{...}}} or {"final":"user-facing answer"}.',
    'Never expose private reasoning. Use only listed tool IDs and exact argument types; do not invent fields.',
    toolInstruction,
    definitions.length > 0 ? `Tools:\n${tools}` : 'Tools: none.',
    'After a tool observation, return another action only if needed; otherwise return final.',
  ].join('\n');
};
