import type { AgentIntent, AgentIntentRoute } from './types';

export const INTENT_TOOL_IDS: Readonly<Record<AgentIntent, readonly string[]>> =
  {
    weather: ['weather', 'location.current'],
    webSearch: ['web.search', 'web.fetch'],
    emailDraft: ['mail.draft', 'contacts.search'],
    messageDraft: ['messages.draft', 'contacts.search'],
    phoneCall: ['phone.call', 'contacts.search'],
    contactSearch: ['contacts.search'],
    calendar: ['calendar.create', 'calendar.list'],
    reminder: ['reminders.create', 'reminders.list'],
    maps: ['maps.search', 'maps.directions', 'location.current'],
    photos: ['photos.search'],
    camera: ['camera.capture'],
    health: ['health.summary'],
    motion: ['motion.activity'],
    files: ['files.read'],
    memory: ['memory.save', 'memory.recall'],
    note: ['memory.save', 'memory.recall'],
    rag: [
      'rag.search',
      'rag.index_files',
      'rag.index_photos',
      'files.read',
      'photos.search',
    ],
    trigger: ['trigger.create', 'trigger.list', 'trigger.cancel'],
    alarm: [
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
    ],
    outlook: [
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
      'contacts.search',
    ],
    chat: [],
    unknown: [],
  };

const makeRoute = (
  intent: AgentIntent,
  clarification?: string,
  allowedToolIds: readonly string[] = INTENT_TOOL_IDS[intent],
): AgentIntentRoute => {
  const allowed = new Set(allowedToolIds);
  const fulfillmentToolIds = (() => {
    switch (intent) {
      case 'weather':
        return new Set(['weather']);
      case 'emailDraft':
        return new Set(['mail.draft']);
      case 'messageDraft':
        return new Set(['messages.draft']);
      case 'phoneCall':
        return new Set(['phone.call']);
      case 'maps':
        return allowed.size === 1 && allowed.has('location.current')
          ? new Set(['location.current'])
          : new Set(['maps.search', 'maps.directions']);
      case 'rag':
        return new Set(['rag.search', 'rag.index_files', 'rag.index_photos']);
      case 'outlook':
        return new Set([...allowed].filter((id) => id.startsWith('outlook.')));
      case 'chat':
      case 'unknown':
        return new Set<string>();
      default:
        return new Set(allowed);
    }
  })();
  return {
    intent,
    allowedToolIds: allowed,
    fulfillmentToolIds,
    clarification,
    requiresTool: intent !== 'chat' && intent !== 'unknown',
  };
};

const containsAny = (text: string, needles: readonly string[]): boolean =>
  needles.some((needle) => text.includes(needle));

const hasContentAfterAction = (
  text: string,
  actions: readonly string[],
): boolean =>
  actions.some((action) => {
    const index = text.indexOf(action);
    return index >= 0 && text.slice(index + action.length).trim().length > 0;
  });

const containsMessageReference = (text: string): boolean =>
  /\b[0-9a-f]{6,}\b|\b(first|second|third|latest|last)\b/.test(text);

const containsTimeExpression = (text: string): boolean =>
  containsAny(text, [
    'today',
    'tomorrow',
    'tonight',
    'morning',
    'afternoon',
    'evening',
    ' in ',
  ]) || /\b\d{1,2}(?::\d{2})?\s*(am|pm)?\b/.test(text);

const containsDuration = (text: string): boolean =>
  /\b\d+(?:\.\d+)?\s*(seconds?|minutes?|hours?|months?)\b/.test(text);

const communicationRoute = (
  intent: 'emailDraft' | 'messageDraft',
  text: string,
  recipientPrompt: string,
  contentPrompt: string,
  combinedPrompt: string,
): AgentIntentRoute => {
  const hasRecipient = /\bto\s+[^\s]+/.test(text) || text.includes('@');
  const hasContent = containsAny(text, [
    ' saying ',
    ' say ',
    ' body ',
    ' that ',
    ' about ',
  ]);
  if (!hasRecipient && !hasContent) {
    return makeRoute(intent, combinedPrompt);
  }
  if (!hasRecipient) {
    return makeRoute(intent, recipientPrompt);
  }
  if (!hasContent) {
    return makeRoute(intent, contentPrompt);
  }
  return makeRoute(intent);
};

/** Deterministic, offline intent routing. It never expands beyond the manifest matrix. */
export const routeAgentIntent = (userInput: string): AgentIntentRoute => {
  const text = userInput
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .join(' ');
  if (!text) {
    return makeRoute('chat');
  }

  if (['meeting', 'find a meeting', 'show meetings'].includes(text)) {
    return makeRoute(
      'unknown',
      'Do you mean a calendar event or a nearby meeting location?',
    );
  }

  if (containsAny(text, ['outlook', 'hotmail', 'microsoft graph'])) {
    if (['outlook', 'hotmail', 'open outlook'].includes(text)) {
      return makeRoute('outlook', 'What would you like to do in Outlook?');
    }
    if (
      containsAny(text, ['search outlook', 'search hotmail', 'find email']) &&
      !hasContentAfterAction(text, ['search', 'find'])
    ) {
      return makeRoute('outlook', 'What should I search for in Outlook?');
    }
    if (
      containsAny(text, ['read outlook', 'read email']) &&
      !containsMessageReference(text)
    ) {
      return makeRoute('outlook', 'Which Outlook message should I read?');
    }
    if (text.includes('attachment') && !containsMessageReference(text)) {
      return makeRoute(
        'outlook',
        'Which Outlook message should I inspect for attachments?',
      );
    }
    return makeRoute('outlook');
  }

  if (
    containsAny(text, ['alarm', 'timer', 'countdown', 'wake me', 'wake us'])
  ) {
    if (
      containsAny(text, [
        'set alarm',
        'schedule alarm',
        'wake me',
        'wake us',
      ]) &&
      !containsTimeExpression(text)
    ) {
      return makeRoute('alarm', 'What time should I use for the alarm?');
    }
    if (containsAny(text, ['timer', 'countdown']) && !containsDuration(text)) {
      return makeRoute('alarm', 'What duration should I use for the timer?');
    }
    const alarmReferences: ReadonlyArray<readonly [string, string]> = [
      ['cancel', 'Which alarm should I cancel?'],
      ['pause', 'Which alarm should I pause?'],
      ['resume', 'Which alarm should I resume?'],
      ['stop', 'Which alarm should I stop?'],
      ['snooze', 'Which alarm should I snooze?'],
    ];
    for (const [verb, prompt] of alarmReferences) {
      if (
        text.includes(`${verb} alarm`) &&
        !hasContentAfterAction(text, [`${verb} alarm`])
      ) {
        return makeRoute('alarm', prompt);
      }
    }
    return makeRoute('alarm');
  }

  if (
    containsAny(text, [
      'agent run',
      'background agent',
      'trigger',
      'scheduled run',
    ])
  ) {
    if (
      containsAny(text, [
        'create trigger',
        'schedule agent',
        'scheduled run',
      ]) &&
      !containsAny(text, [' to ', ' saying ', ' prompt ', ' about '])
    ) {
      return makeRoute('trigger', 'What should the scheduled agent run do?');
    }
    return makeRoute('trigger');
  }

  if (
    containsAny(text, ['reindex photos', 'index photos']) &&
    !containsDuration(text)
  ) {
    return makeRoute('rag', 'How many months of photos should I index?');
  }
  if (
    containsAny(text, [
      'weather',
      'forecast',
      'temperature',
      'rain',
      'snow',
      'wind outside',
    ])
  ) {
    return makeRoute('weather');
  }
  if (
    containsAny(text, [
      'search web',
      'web search',
      'search online',
      'look up',
      'find online',
      'http://',
      'https://',
    ])
  ) {
    if (
      ['search web', 'web search', 'look it up', 'search online'].includes(text)
    ) {
      return makeRoute('webSearch', 'What should I search for?');
    }
    return makeRoute('webSearch');
  }
  if (
    containsAny(text, [
      'draft email',
      'write email',
      'compose email',
      'email to',
      'send email',
    ])
  ) {
    return communicationRoute(
      'emailDraft',
      text,
      'Who should I send it to?',
      'What should the email say?',
      'Who should I send it to, and what should it say?',
    );
  }
  if (
    containsAny(text, [
      'draft message',
      'write message',
      'compose message',
      'text message',
      'sms',
      'imessage',
      'send a text',
    ])
  ) {
    return communicationRoute(
      'messageDraft',
      text,
      'Who should I message?',
      'What should the message say?',
      'Who should I message, and what should it say?',
    );
  }
  if (/\b(call|dial|phone)\b/.test(text)) {
    return ['call', 'phone', 'make a call', 'start a call'].includes(text)
      ? makeRoute('phoneCall', 'Who should I call?')
      : makeRoute('phoneCall');
  }
  if (
    containsAny(text, [
      'find contact',
      'search contacts',
      'address book',
      'phone number for',
      'email address for',
    ])
  ) {
    return ['find contact', 'search contacts', 'address book'].includes(text)
      ? makeRoute('contactSearch', 'Which contact should I look up?')
      : makeRoute('contactSearch');
  }
  if (containsAny(text, ['calendar', 'event', 'appointments'])) {
    if (
      containsAny(text, ['create event', 'add event', 'schedule event']) &&
      !containsAny(text, [' called ', ' titled ', ' for ', ' about '])
    ) {
      return makeRoute('calendar', 'What should the calendar event be?');
    }
    return makeRoute('calendar');
  }
  if (
    containsAny(text, [
      'remind me',
      'reminder',
      'todo',
      'to do',
      'pending reminders',
    ])
  ) {
    return [
      'remind me',
      'create reminder',
      'add reminder',
      'reminder',
    ].includes(text)
      ? makeRoute('reminder', 'What should I remind you about?')
      : makeRoute('reminder');
  }
  if (
    containsAny(text, [
      'current location',
      'where am i',
      'my gps location',
      'where are we',
    ])
  ) {
    return makeRoute('maps', undefined, ['location.current']);
  }
  if (
    containsAny(text, [
      'maps',
      'directions',
      'navigate',
      'route to',
      'near me',
      'nearby',
      'closest',
      'nearest',
      'show me on map',
    ])
  ) {
    return ['maps', 'directions', 'navigate', 'nearby'].includes(text)
      ? makeRoute('maps', 'What place or destination should I look for?')
      : makeRoute('maps');
  }
  if (
    containsAny(text, [
      'search photos',
      'find photos',
      'photo library',
      'find pictures',
      'latest photo',
      'latest selfie',
    ])
  ) {
    return [
      'search photos',
      'find photos',
      'find pictures',
      'photo library',
    ].includes(text)
      ? makeRoute('photos', 'Which photos should I look for?')
      : makeRoute('photos');
  }
  if (
    containsAny(text, [
      'take a photo',
      'capture image',
      'open camera',
      'take picture',
    ])
  ) {
    return makeRoute('camera');
  }
  if (
    containsAny(text, [
      'health summary',
      'heart rate',
      'health data',
      'sleep data',
      'active energy',
      'walking distance',
    ])
  ) {
    return makeRoute('health');
  }
  if (
    containsAny(text, [
      'motion activity',
      'am i walking',
      'am i running',
      'device motion',
      'recent activity',
    ])
  ) {
    return makeRoute('motion');
  }
  if (
    containsAny(text, [
      'remember',
      'memory',
      'save this fact',
      'what do you remember',
      'keep this in mind',
    ])
  ) {
    return [
      'remember',
      'memory',
      'save memory',
      'recall memory',
      'note',
    ].includes(text)
      ? makeRoute('memory', 'What should I save or recall?')
      : makeRoute('memory');
  }
  if (
    containsAny(text, [
      'rag search',
      'search my files',
      'search my documents',
      'search my notes',
      'search personal data',
      'reindex files',
      'index files',
      'architecture notes',
    ])
  ) {
    return ['rag search', 'search personal data'].includes(text)
      ? makeRoute('rag', 'What should I search for?')
      : makeRoute('rag');
  }
  if (
    containsAny(text, [
      'read file',
      'open file',
      'read document',
      'imported file',
      'local document',
    ])
  ) {
    return [
      'read file',
      'open file',
      'read document',
      'imported file',
      'local document',
    ].includes(text)
      ? makeRoute('files', 'Which file should I read?')
      : makeRoute('files');
  }
  if (text.startsWith('note ') || text.startsWith('save this ')) {
    return makeRoute('note');
  }
  return makeRoute('chat');
};

export const unavailableIntentMessage = (intent: AgentIntent): string =>
  ({
    weather: 'Weather and location tools are unavailable.',
    webSearch: 'Web tools are unavailable.',
    emailDraft: 'Email drafting tools are unavailable.',
    messageDraft: 'Message drafting tools are unavailable.',
    phoneCall: 'Phone and contact tools are unavailable.',
    contactSearch: 'Contact tools are unavailable.',
    calendar: 'Calendar tools are unavailable.',
    reminder: 'Reminder tools are unavailable.',
    maps: 'Maps and location tools are unavailable.',
    photos: 'Photo tools are unavailable.',
    camera: 'Camera tools are unavailable.',
    health: 'Health tools are unavailable.',
    motion: 'Motion tools are unavailable.',
    files: 'File tools are unavailable.',
    memory: 'Memory tools are unavailable.',
    note: 'Memory tools are unavailable.',
    rag: 'Local retrieval tools are unavailable.',
    trigger: 'Scheduled trigger tools are unavailable.',
    alarm: 'Alarm tools are unavailable.',
    outlook: 'Outlook tools are unavailable.',
    chat: 'No matching tool is available.',
    unknown: 'No matching tool is available.',
  })[intent];
