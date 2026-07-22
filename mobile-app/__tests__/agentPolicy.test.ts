import {
  AGENT_TOOL_CATALOG,
  AgentToolValidationError,
  canonicalToolId,
  parseAgentTurn,
  routeAgentIntent,
  validateToolCall,
} from '../src/agent';

const EXPECTED_TOOL_IDS = [
  'calendar.create',
  'calendar.list',
  'reminders.create',
  'reminders.list',
  'contacts.search',
  'messages.draft',
  'mail.draft',
  'outlook.status',
  'outlook.folders.list',
  'outlook.messages.list',
  'outlook.messages.search',
  'outlook.message.read',
  'outlook.attachments.list',
  'outlook.draft.create',
  'outlook.mail.send',
  'outlook.message.mark_read',
  'outlook.message.mark_unread',
  'outlook.message.move',
  'outlook.message.archive',
  'outlook.message.delete',
  'outlook.message.reply',
  'outlook.message.reply_all',
  'outlook.message.forward',
  'phone.call',
  'location.current',
  'weather',
  'maps.directions',
  'maps.search',
  'photos.search',
  'camera.capture',
  'health.summary',
  'motion.activity',
  'web.search',
  'web.fetch',
  'files.read',
  'memory.save',
  'memory.recall',
  'rag.search',
  'rag.index_files',
  'rag.index_photos',
  'trigger.create',
  'trigger.list',
  'trigger.cancel',
  'alarm.authorization_status',
  'alarm.request_authorization',
  'alarm.schedule',
  'alarm.countdown',
  'alarm.list',
  'alarm.pause',
  'alarm.resume',
  'alarm.stop',
  'alarm.snooze',
  'alarm.cancel',
] as const;

describe('agent policy catalog', () => {
  it('contains the exact 53 canonical tools and 26 approval boundaries', () => {
    expect(AGENT_TOOL_CATALOG.map((tool) => tool.id)).toEqual(
      EXPECTED_TOOL_IDS,
    );
    expect(new Set(EXPECTED_TOOL_IDS).size).toBe(53);
    expect(
      AGENT_TOOL_CATALOG.filter((tool) => tool.requiresApproval),
    ).toHaveLength(26);
  });

  it('normalizes only declared tool and argument aliases', () => {
    expect(canonicalToolId(' Weather-Current ')).toBe('weather');
    expect(canonicalToolId('outlook-message-reply-all')).toBe(
      'outlook.message.reply_all',
    );
    expect(canonicalToolId('open.url')).toBe('open.url');

    const call = validateToolCall(
      'sms',
      { recipient: '+15145550123', text: 'Bonjour' },
      new Set(['messages.draft']),
    );
    expect(call).toMatchObject({
      tool: 'messages.draft',
      args: { to: '+15145550123', body: 'Bonjour' },
    });
  });

  it.each([
    ['missing required args', 'web.search', {}, 'missing_argument'],
    ['wrong types', 'web.search', { query: 4 }, 'wrong_type'],
    [
      'extra fields',
      'web.search',
      { query: 'swift', limit: 2 },
      'extra_arguments',
    ],
    [
      'invalid enum values',
      'rag.search',
      { query: 'swift', sourceScope: 'internet' },
      'invalid_enum',
    ],
    [
      'invalid trigger schedule combinations',
      'trigger.create',
      {
        title: 'Run',
        prompt: 'Summarize',
        schedule: 'relative',
        atTime: '09:00',
      },
      'invalid_arguments',
    ],
    [
      'missing alarm schedule selector',
      'alarm.schedule',
      { title: 'Wake' },
      'invalid_arguments',
    ],
  ])('rejects %s', (_label, tool, args, code) => {
    try {
      validateToolCall(tool, args, new Set([tool]));
      throw new Error('Expected validation to fail.');
    } catch (error) {
      expect(error).toBeInstanceOf(AgentToolValidationError);
      expect((error as AgentToolValidationError).code).toBe(code);
    }
  });

  it('rejects unavailable tools and conflicting aliases', () => {
    expect(() =>
      validateToolCall('web.search', { query: 'swift' }, new Set(['weather'])),
    ).toThrow('not available');
    expect(() =>
      validateToolCall(
        'weather',
        { location: 'Montréal', city: 'Toronto' },
        new Set(['weather']),
      ),
    ).toThrow('Conflicting values');
  });

  it('normalizes enum arguments case-insensitively to canonical values', () => {
    const call = validateToolCall(
      'rag.search',
      { query: 'notes', sourceScope: 'PHOTOS' },
      new Set(['rag.search']),
    );

    expect(call.args.sourceScope).toBe('photos');
  });

  it('canonicalizes Outlook destination/comment aliases with conflict checks', () => {
    expect(
      validateToolCall(
        'outlook.message.move',
        { messageId: 'm1', destinationId: 'inbox' },
        new Set(['outlook.message.move']),
      ).args,
    ).toEqual({ messageId: 'm1', destination: 'inbox' });
    expect(
      validateToolCall(
        'outlook.message.reply',
        { messageId: 'm1', comment: 'Merci' },
        new Set(['outlook.message.reply']),
      ).args,
    ).toEqual({ messageId: 'm1', body: 'Merci' });
    expect(() =>
      validateToolCall(
        'outlook.message.reply_all',
        { messageId: 'm1', body: 'Oui', comment: 'Non' },
        new Set(['outlook.message.reply_all']),
      ),
    ).toThrow('Conflicting values');
  });

  it('validates every trigger schedule shape and exact cancel selector', () => {
    const available = new Set(['trigger.create', 'trigger.cancel']);
    expect(
      validateToolCall(
        'trigger.create',
        {
          title: 'Daily',
          prompt: 'Prepare summary',
          schedule: 'ABSOLUTE',
          atTime: '09:05',
        },
        available,
      ).args.schedule,
    ).toBe('absolute');
    expect(
      validateToolCall(
        'trigger.create',
        {
          title: 'Prepare',
          prompt: 'Summarize meeting',
          schedule: 'before_next_event',
          beforeMinutes: 15,
        },
        available,
      ).args.beforeMinutes,
    ).toBe(15);
    expect(
      validateToolCall(
        'trigger.cancel',
        { title: 'Morning summary' },
        available,
      ).args,
    ).toEqual({ title: 'Morning summary' });
    expect(() =>
      validateToolCall(
        'trigger.cancel',
        {
          id: '11111111-2222-4333-8444-555555555555',
          title: 'Morning summary',
        },
        available,
      ),
    ).toThrow('exactly one');
  });

  it('binds the default AlarmKit snooze into the validated approval payload', () => {
    expect(
      validateToolCall(
        'alarm.schedule',
        { title: 'Wake', inMinutes: 5 },
        new Set(['alarm.schedule']),
      ).args,
    ).toEqual({ title: 'Wake', inMinutes: 5, snoozeMinutes: 5 });
    expect(
      validateToolCall(
        'alarm.schedule',
        { title: 'Wake', inMinutes: 5, repeats: false },
        new Set(['alarm.schedule']),
      ).args,
    ).toEqual({
      title: 'Wake',
      inMinutes: 5,
      repeats: false,
      snoozeMinutes: 5,
    });
    expect(() =>
      validateToolCall(
        'alarm.schedule',
        { title: 'Wake', inMinutes: 5, repeats: true },
        new Set(['alarm.schedule']),
      ),
    ).toThrow('one-shot alarms only');
  });
});

describe('agent turn parser', () => {
  it('accepts raw or solely fenced JSON and discards private thought', () => {
    expect(parseAgentTurn('{"thought":"private","final":"Bonjour"}')).toEqual({
      kind: 'final',
      final: 'Bonjour',
    });
    expect(
      parseAgentTurn(
        '```json\n{"action":{"tool":"web.search","args":{"query":"Swift"}}}\n```',
      ),
    ).toEqual({
      kind: 'action',
      action: { tool: 'web.search', args: { query: 'Swift' } },
    });
  });

  it.each([
    'Here is JSON: {"final":"no"}',
    '```json\n{"final":"no"}\n```\nextra prose',
    '{"final":"one","final":"two"}',
    '{"action":{"tool":"web.search","args":{},"extra":true}}',
    '{"action":{"tool":"web.search","args":{}},"final":"both"}',
  ])('fails closed for invalid output: %s', (output) => {
    expect(() => parseAgentTurn(output)).toThrow();
  });
});

describe('agent intent router', () => {
  it('isolates weather from unrelated capabilities', () => {
    const route = routeAgentIntent('What is the weather in Toronto?');
    expect(route.intent).toBe('weather');
    expect([...route.allowedToolIds]).toEqual(['weather', 'location.current']);
    expect(route.allowedToolIds.has('web.search')).toBe(false);
    expect(route.allowedToolIds.has('calendar.create')).toBe(false);
  });

  it('asks deterministic clarification for underspecified actions', () => {
    expect(routeAgentIntent('Start a timer').clarification).toBe(
      'What duration should I use for the timer?',
    );
    expect(routeAgentIntent('draft email').clarification).toBe(
      'Who should I send it to, and what should it say?',
    );
    expect(routeAgentIntent('meeting').clarification).toContain(
      'calendar event',
    );
  });

  it('accepts an explicit alarm title after the clarification turn', () => {
    expect(routeAgentIntent('Cancel alarm').clarification).toBe(
      'Which alarm should I cancel?',
    );
    const resumed = routeAgentIntent('Cancel alarm Morning');
    expect(resumed.intent).toBe('alarm');
    expect(resumed.clarification).toBeUndefined();
  });
});
