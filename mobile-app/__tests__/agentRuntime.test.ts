import {
  buildAgentSystemPrompt,
  runAgent,
  routeAgentIntent,
  AGENT_TOOL_CATALOG,
  type AgentModelCallback,
  type AgentToolCallback,
} from '../src/agent';

const queuedModel = (
  outputs: readonly string[],
  requests: Parameters<AgentModelCallback>[0][] = [],
): AgentModelCallback => {
  let index = 0;
  return async (request) => {
    requests.push(request);
    const output = outputs[index];
    index += 1;
    if (output === undefined) {
      throw new Error('No queued model output.');
    }
    return output;
  };
};

describe('agent runtime', () => {
  it('repairs one invalid decision, executes a validated action, and returns final', async () => {
    const requests: Parameters<AgentModelCallback>[0][] = [];
    const model = queuedModel(
      [
        'I should search the web.',
        '{"action":{"tool":"search-web","args":{"q":"Swift 6"}}}',
        '{"final":"I found the Swift result."}',
      ],
      requests,
    );
    const executeTool: AgentToolCallback = jest.fn(async (call) => ({
      title: 'Swift',
      canonicalTool: call.tool,
    }));

    const result = await runAgent({
      userInput: 'Search web for Swift 6',
      model,
      executeTool,
    });

    expect(result).toMatchObject({
      status: 'final',
      message: 'I found the Swift result.',
      decisionCount: 3,
    });
    expect(executeTool).toHaveBeenCalledTimes(1);
    expect(executeTool).toHaveBeenCalledWith(
      expect.objectContaining({
        tool: 'web.search',
        args: { query: 'Swift 6' },
      }),
    );
    expect(requests[1].repair).toContain('corrected JSON decision');
    expect(requests[2].observations).toEqual([
      expect.objectContaining({ tool: 'web.search', succeeded: true }),
    ]);
  });

  it('prevents a duplicate tool call', async () => {
    const action = '{"action":{"tool":"web.search","args":{"query":"Swift"}}}';
    const executeTool: AgentToolCallback = jest.fn(async () => 'result');
    const result = await runAgent({
      userInput: 'Search web for Swift',
      model: queuedModel([action, action]),
      executeTool,
    });

    expect(result).toMatchObject({ status: 'error', code: 'duplicate_call' });
    expect(executeTool).toHaveBeenCalledTimes(1);
  });

  it('returns a pending approval without executing the action', async () => {
    const executeTool: AgentToolCallback = jest.fn(async () => 'captured');
    const result = await runAgent({
      userInput: 'Take a photo',
      model: queuedModel(['{"action":{"tool":"camera.capture","args":{}}}']),
      executeTool,
    });

    expect(result).toMatchObject({
      status: 'pendingApproval',
      approval: { tool: 'camera.capture', args: {}, risk: 'high' },
    });
    expect(executeTool).not.toHaveBeenCalled();
  });

  it('returns a direct, sanitized final for chat', async () => {
    const executeTool: AgentToolCallback = jest.fn();
    const result = await runAgent({
      userInput: 'Hello there',
      model: queuedModel(['{"final":"<|assistant|>Hello!\\u0000"}']),
      executeTool,
    });

    expect(result).toEqual({
      status: 'final',
      intent: 'chat',
      message: 'Hello!',
      decisionCount: 1,
    });
    expect(executeTool).not.toHaveBeenCalled();
  });

  it('returns clarification before invoking either callback', async () => {
    const model: AgentModelCallback = jest.fn();
    const executeTool: AgentToolCallback = jest.fn();
    const result = await runAgent({
      userInput: 'Set alarm',
      model,
      executeTool,
    });

    expect(result).toMatchObject({ status: 'clarification', decisionCount: 0 });
    expect(model).not.toHaveBeenCalled();
    expect(executeTool).not.toHaveBeenCalled();
  });

  it('rejects an oversized tool request before model generation', async () => {
    const model: AgentModelCallback = jest.fn();
    const executeTool: AgentToolCallback = jest.fn();
    const result = await runAgent({
      userInput: `Search web for ${'a'.repeat(600)}`,
      model,
      executeTool,
    });

    expect(result).toMatchObject({
      status: 'clarification',
      decisionCount: 0,
      message: expect.stringContaining('512 UTF-8 bytes'),
    });
    expect(model).not.toHaveBeenCalled();
    expect(executeTool).not.toHaveBeenCalled();
  });

  it('requires a fulfilling tool after a supporting observation', async () => {
    const executeTool: AgentToolCallback = jest.fn(async (call) => ({
      tool: call.tool,
      value: call.tool === 'weather' ? '18 C' : 'Toronto',
    }));
    const result = await runAgent({
      userInput: 'Weather in Toronto',
      model: queuedModel([
        '{"action":{"tool":"location.current","args":{}}}',
        '{"final":"It is sunny."}',
        '{"action":{"tool":"weather","args":{"location":"Toronto"}}}',
        '{"final":"It is 18 C."}',
      ]),
      executeTool,
    });

    expect(result).toMatchObject({
      status: 'final',
      message: 'It is 18 C.',
    });
    expect(executeTool).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ tool: 'location.current' }),
    );
    expect(executeTool).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ tool: 'weather' }),
    );
  });

  it('builds a compact prompt containing only routed tools', () => {
    const route = routeAgentIntent('Search web for Swift');
    const definitions = AGENT_TOOL_CATALOG.filter((tool) =>
      route.allowedToolIds.has(tool.id),
    );
    const prompt = buildAgentSystemPrompt(definitions, route.requiresTool);

    expect(prompt).toContain('web.search');
    expect(prompt).toContain('web.fetch');
    expect(prompt).not.toContain('calendar.create');
    expect(prompt).not.toContain('outlook.mail.send');
    expect(prompt.length).toBeLessThan(2_000);
  });
});
