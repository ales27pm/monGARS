export const AGENT_TRIGGER_MAXIMUM_PROMPT_BYTES = 512;

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

export const normalizedAgentTriggerPrompt = (value: string): string | null => {
  const prompt = value.trim();
  if (!prompt || utf8ByteLength(prompt) > AGENT_TRIGGER_MAXIMUM_PROMPT_BYTES) {
    return null;
  }
  return prompt;
};
