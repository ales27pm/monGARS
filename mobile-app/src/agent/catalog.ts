import type {
  AgentArgumentSchema,
  AgentArgumentType,
  AgentPermission,
  AgentToolCategory,
  AgentToolDefinition,
  AgentToolRisk,
  JSONObject,
  JSONValue,
  ValidatedAgentToolCall,
} from './types';

const arg = (
  name: string,
  type: AgentArgumentType = 'string',
  required = true,
  allowedValues?: readonly string[],
): AgentArgumentSchema => ({ name, type, required, allowedValues });

const tool = (
  id: string,
  displayName: string,
  description: string,
  category: AgentToolCategory,
  args: readonly AgentArgumentSchema[],
  permission: AgentPermission | undefined,
  risk: AgentToolRisk,
  requiresApproval: boolean,
  supportsBackgroundExecution: boolean,
  maximumOutputCharacters = 2_400,
): AgentToolDefinition => ({
  id,
  displayName,
  description,
  category,
  arguments: args,
  permission,
  risk,
  requiresApproval,
  supportsBackgroundExecution,
  maximumOutputCharacters: Math.max(256, maximumOutputCharacters),
});

/** The 53 canonical Lumen capabilities. Host code supplies their implementations. */
export const AGENT_TOOL_CATALOG: readonly AgentToolDefinition[] = [
  tool(
    'calendar.create',
    'Create Event',
    'Add an event to the calendar.',
    'productivity',
    [arg('title'), arg('startsInMinutes', 'number')],
    'calendar',
    'high',
    true,
    false,
  ),
  tool(
    'calendar.list',
    'List Events',
    'Read upcoming calendar events.',
    'productivity',
    [],
    'calendar',
    'moderate',
    false,
    true,
  ),
  tool(
    'reminders.create',
    'Add Reminder',
    'Create a reminder.',
    'productivity',
    [arg('title')],
    'reminders',
    'high',
    true,
    false,
  ),
  tool(
    'reminders.list',
    'List Reminders',
    'Read pending reminders.',
    'productivity',
    [],
    'reminders',
    'moderate',
    false,
    true,
  ),
  tool(
    'contacts.search',
    'Search Contacts',
    'Find a contact by name.',
    'communication',
    [arg('query')],
    'contacts',
    'moderate',
    false,
    false,
  ),
  tool(
    'messages.draft',
    'Draft Message',
    'Compose an iMessage or SMS draft.',
    'communication',
    [
      arg('to'),
      arg('body'),
      arg('recipient', 'string', false),
      arg('number', 'string', false),
      arg('message', 'string', false),
      arg('text', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'mail.draft',
    'Draft Email',
    'Compose a system email draft.',
    'communication',
    [
      arg('to'),
      arg('subject', 'string', false),
      arg('body'),
      arg('recipient', 'string', false),
      arg('email', 'string', false),
      arg('message', 'string', false),
      arg('text', 'string', false),
      arg('title', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),

  tool(
    'outlook.status',
    'Outlook Status',
    'Check Outlook sign-in status.',
    'communication',
    [],
    undefined,
    'low',
    false,
    true,
  ),
  tool(
    'outlook.folders.list',
    'List Outlook Folders',
    'List Outlook mail folders.',
    'communication',
    [arg('includeHidden', 'bool', false)],
    undefined,
    'moderate',
    false,
    true,
  ),
  tool(
    'outlook.messages.list',
    'List Outlook Messages',
    'List recent Outlook messages.',
    'communication',
    [
      arg('folder', 'string', false),
      arg('folderId', 'string', false),
      arg('limit', 'number', false),
      arg('unreadOnly', 'bool', false),
    ],
    undefined,
    'moderate',
    false,
    true,
    4_000,
  ),
  tool(
    'outlook.messages.search',
    'Search Outlook Messages',
    'Search Outlook mail.',
    'communication',
    [
      arg('query'),
      arg('folder', 'string', false),
      arg('folderId', 'string', false),
      arg('limit', 'number', false),
    ],
    undefined,
    'moderate',
    false,
    true,
    4_000,
  ),
  tool(
    'outlook.message.read',
    'Read Outlook Message',
    'Read one Outlook message.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'moderate',
    false,
    true,
    6_000,
  ),
  tool(
    'outlook.attachments.list',
    'List Outlook Attachments',
    'List attachment metadata.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'moderate',
    false,
    true,
  ),
  tool(
    'outlook.draft.create',
    'Create Outlook Draft',
    'Create a saved Outlook draft.',
    'communication',
    [arg('to'), arg('subject'), arg('body')],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.mail.send',
    'Send Outlook Email',
    'Send mail through Outlook.',
    'communication',
    [arg('to'), arg('subject'), arg('body')],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.mark_read',
    'Mark Outlook Read',
    'Mark an Outlook message read.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.mark_unread',
    'Mark Outlook Unread',
    'Mark an Outlook message unread.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.move',
    'Move Outlook Message',
    'Move an Outlook message.',
    'communication',
    [
      arg('messageId'),
      arg('destination', 'string', false),
      arg('id', 'string', false),
      arg('destinationId', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.archive',
    'Archive Outlook Message',
    'Archive an Outlook message.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'critical',
    true,
    false,
  ),
  tool(
    'outlook.message.delete',
    'Delete Outlook Message',
    'Delete an Outlook message.',
    'communication',
    [arg('messageId'), arg('id', 'string', false)],
    undefined,
    'critical',
    true,
    false,
  ),
  tool(
    'outlook.message.reply',
    'Reply Outlook Message',
    'Reply to an Outlook message.',
    'communication',
    [
      arg('messageId'),
      arg('body', 'string', false),
      arg('id', 'string', false),
      arg('comment', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.reply_all',
    'Reply All Outlook Message',
    'Reply all to an Outlook message.',
    'communication',
    [
      arg('messageId'),
      arg('body', 'string', false),
      arg('id', 'string', false),
      arg('comment', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'outlook.message.forward',
    'Forward Outlook Message',
    'Forward an Outlook message.',
    'communication',
    [
      arg('messageId'),
      arg('to'),
      arg('id', 'string', false),
      arg('body', 'string', false),
      arg('comment', 'string', false),
    ],
    undefined,
    'high',
    true,
    false,
  ),
  tool(
    'phone.call',
    'Start Call',
    'Open the phone dialer.',
    'communication',
    [arg('number')],
    undefined,
    'high',
    true,
    false,
  ),

  tool(
    'location.current',
    'Current Location',
    'Read the current GPS location.',
    'location',
    [],
    'location',
    'moderate',
    false,
    false,
  ),
  tool(
    'weather',
    'Current Weather',
    'Get current weather from a city or location.',
    'location',
    [arg('location', 'string', false), arg('city', 'string', false)],
    'location',
    'low',
    false,
    true,
    4_000,
  ),
  tool(
    'maps.directions',
    'Get Directions',
    'Get directions to a destination.',
    'location',
    [arg('destination')],
    undefined,
    'moderate',
    false,
    false,
  ),
  tool(
    'maps.search',
    'Search Nearby',
    'Search for nearby places.',
    'location',
    [arg('query')],
    'location',
    'moderate',
    false,
    false,
  ),
  tool(
    'photos.search',
    'Search Photos',
    'Search the local photo library.',
    'media',
    [arg('query')],
    'photos',
    'moderate',
    false,
    false,
  ),
  tool(
    'camera.capture',
    'Capture Image',
    'Capture a device image.',
    'media',
    [],
    'camera',
    'high',
    true,
    false,
  ),
  tool(
    'health.summary',
    'Health Summary',
    'Read a local health summary.',
    'health',
    [],
    'health',
    'moderate',
    false,
    false,
  ),
  tool(
    'motion.activity',
    'Motion Activity',
    'Read recent motion activity.',
    'health',
    [],
    'motion',
    'moderate',
    false,
    true,
  ),

  tool(
    'web.search',
    'Web Search',
    'Search the public web.',
    'knowledge',
    [arg('query')],
    undefined,
    'low',
    false,
    false,
    4_000,
  ),
  tool(
    'web.fetch',
    'Fetch URL',
    'Fetch a specific web page.',
    'knowledge',
    [arg('url')],
    undefined,
    'low',
    false,
    false,
    4_000,
  ),
  tool(
    'files.read',
    'Read File',
    'Read a previously imported local document.',
    'knowledge',
    [arg('name')],
    undefined,
    'moderate',
    false,
    true,
  ),
  tool(
    'memory.save',
    'Save Memory',
    'Store a user fact or preference.',
    'knowledge',
    [arg('content'), arg('kind')],
    undefined,
    'moderate',
    false,
    false,
  ),
  tool(
    'memory.recall',
    'Recall Memory',
    'Search stored memories.',
    'knowledge',
    [arg('query')],
    undefined,
    'moderate',
    false,
    true,
  ),
  tool(
    'rag.search',
    'Search Local Knowledge',
    'Search indexed local content.',
    'knowledge',
    [
      arg('query'),
      arg('limit', 'number', false),
      arg('sourceScope', 'enum', false, [
        'all',
        'documents',
        'notes',
        'photos',
      ]),
    ],
    undefined,
    'moderate',
    false,
    true,
    3_000,
  ),
  tool(
    'rag.index_files',
    'Index Files',
    'Index imported local files.',
    'knowledge',
    [],
    undefined,
    'moderate',
    false,
    false,
  ),
  tool(
    'rag.index_photos',
    'Index Photos',
    'Index local photo metadata.',
    'knowledge',
    [arg('months', 'number')],
    'photos',
    'moderate',
    false,
    false,
  ),

  tool(
    'trigger.create',
    'Create Trigger',
    'Create a scheduled agent trigger.',
    'productivity',
    [
      arg('title'),
      arg('prompt'),
      arg('schedule', 'enum', true, [
        'absolute',
        'before_next_event',
        'interval',
        'relative',
      ]),
      arg('inMinutes', 'number', false),
      arg('atTime', 'string', false),
      arg('intervalSeconds', 'number', false),
      arg('beforeMinutes', 'number', false),
    ],
    'notifications',
    'high',
    true,
    false,
  ),
  tool(
    'trigger.list',
    'List Triggers',
    'List scheduled agent triggers.',
    'productivity',
    [],
    'notifications',
    'low',
    false,
    true,
  ),
  tool(
    'trigger.cancel',
    'Cancel Trigger',
    'Cancel a scheduled agent trigger by UUID or exact title.',
    'productivity',
    [arg('id', 'string', false), arg('title', 'string', false)],
    'notifications',
    'critical',
    true,
    false,
  ),

  tool(
    'alarm.authorization_status',
    'Alarm Authorization',
    'Read AlarmKit authorization status.',
    'productivity',
    [],
    'alarms',
    'low',
    false,
    true,
  ),
  tool(
    'alarm.request_authorization',
    'Request Alarm Access',
    'Request AlarmKit authorization.',
    'productivity',
    [],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.schedule',
    'Schedule Alarm',
    'Schedule an alarm with a five-minute snooze by default.',
    'productivity',
    [
      arg('title'),
      arg('inMinutes', 'number', false),
      arg('timestamp', 'string', false),
      arg('repeats', 'bool', false),
      arg('snoozeMinutes', 'number', false),
    ],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.countdown',
    'Start Countdown',
    'Create a countdown alarm.',
    'productivity',
    [arg('title'), arg('durationSeconds', 'number')],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.list',
    'List Alarms',
    'List active alarms.',
    'productivity',
    [],
    'alarms',
    'moderate',
    false,
    true,
  ),
  tool(
    'alarm.pause',
    'Pause Alarm',
    'Pause an alarm.',
    'productivity',
    [arg('id')],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.resume',
    'Resume Alarm',
    'Resume a paused alarm.',
    'productivity',
    [arg('id')],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.stop',
    'Stop Alarm',
    'Stop an alerting alarm.',
    'productivity',
    [arg('id')],
    'alarms',
    'critical',
    true,
    false,
  ),
  tool(
    'alarm.snooze',
    'Snooze Alarm',
    'Snooze an alerting alarm.',
    'productivity',
    [arg('id')],
    'alarms',
    'high',
    true,
    false,
  ),
  tool(
    'alarm.cancel',
    'Cancel Alarm',
    'Cancel a scheduled alarm.',
    'productivity',
    [arg('id')],
    'alarms',
    'critical',
    true,
    false,
  ),
] as const;

export const CANONICAL_TOOL_IDS: ReadonlySet<string> = new Set(
  AGENT_TOOL_CATALOG.map((definition) => definition.id),
);

const DEFINITIONS_BY_ID = new Map(
  AGENT_TOOL_CATALOG.map((definition) => [definition.id, definition] as const),
);

const TOOL_ALIASES: Readonly<Record<string, string>> = {
  'weather.current': 'weather',
  'current.weather': 'weather',
  'forecast.current': 'weather',
  'weather.get': 'weather',
  'get.weather': 'weather',
  getweather: 'weather',
  currentweather: 'weather',
  search: 'web.search',
  'internet.search': 'web.search',
  web: 'web.search',
  websearch: 'web.search',
  'browser.search': 'web.search',
  'google.search': 'web.search',
  google: 'web.search',
  'search.web': 'web.search',
  searchweb: 'web.search',
  fetch: 'web.fetch',
  'browser.fetch': 'web.fetch',
  'url.fetch': 'web.fetch',
  'fetch.url': 'web.fetch',
  'read.url': 'web.fetch',
  'read.website': 'web.fetch',
  maps: 'maps.search',
  map: 'maps.search',
  'map.search': 'maps.search',
  'nearby.search': 'maps.search',
  'local.search': 'maps.search',
  'places.search': 'maps.search',
  'place.search': 'maps.search',
  'google.maps': 'maps.search',
  'google.maps.api': 'maps.search',
  googlemaps: 'maps.search',
  googlemapsapi: 'maps.search',
  'maps.api': 'maps.search',
  mapsapi: 'maps.search',
  'nearest.place': 'maps.search',
  'find.nearby': 'maps.search',
  'map.directions': 'maps.directions',
  directions: 'maps.directions',
  navigation: 'maps.directions',
  navigate: 'maps.directions',
  route: 'maps.directions',
  'route.to': 'maps.directions',
  'open.maps': 'maps.directions',
  location: 'location.current',
  gps: 'location.current',
  'current.location': 'location.current',
  'location.get': 'location.current',
  'get.location': 'location.current',
  currentlocation: 'location.current',
  'location.snapshot': 'location.current',
  calendar: 'calendar.create',
  'create.event': 'calendar.create',
  'event.create': 'calendar.create',
  'schedule.event': 'calendar.create',
  'calendar.read': 'calendar.list',
  'list.events': 'calendar.list',
  'events.list': 'calendar.list',
  reminder: 'reminders.create',
  'reminder.create': 'reminders.create',
  'create.reminder': 'reminders.create',
  'reminder.list': 'reminders.list',
  'list.reminders': 'reminders.list',
  mail: 'mail.draft',
  email: 'mail.draft',
  'email.draft': 'mail.draft',
  'compose.email': 'mail.draft',
  message: 'messages.draft',
  sms: 'messages.draft',
  'sms.draft': 'messages.draft',
  'compose.message': 'messages.draft',
  imessage: 'messages.draft',
  phone: 'phone.call',
  call: 'phone.call',
  dial: 'phone.call',
  contacts: 'contacts.search',
  'contact.search': 'contacts.search',
  'search.contacts': 'contacts.search',
  'contacts.lookup': 'contacts.search',
  'memory.search': 'memory.recall',
  'rag.search.secure': 'rag.search',
  outlook: 'outlook.status',
  'microsoft.outlook.status': 'outlook.status',
  'hotmail.status': 'outlook.status',
  'graph.status': 'outlook.status',
  'outlook.folders': 'outlook.folders.list',
  'outlook.folder.list': 'outlook.folders.list',
  'hotmail.folders': 'outlook.folders.list',
  'mail.folders.list': 'outlook.folders.list',
  'outlook.messages': 'outlook.messages.list',
  'outlook.inbox': 'outlook.messages.list',
  'outlook.mail.list': 'outlook.messages.list',
  'hotmail.inbox': 'outlook.messages.list',
  'hotmail.messages': 'outlook.messages.list',
  'graph.mail.list': 'outlook.messages.list',
  'outlook.search': 'outlook.messages.search',
  'outlook.mail.search': 'outlook.messages.search',
  'hotmail.search': 'outlook.messages.search',
  'search.outlook': 'outlook.messages.search',
  'search.email': 'outlook.messages.search',
  'email.search': 'outlook.messages.search',
  'outlook.read': 'outlook.message.read',
  'outlook.mail.read': 'outlook.message.read',
  'read.outlook': 'outlook.message.read',
  'read.email': 'outlook.message.read',
  'outlook.attachments': 'outlook.attachments.list',
  'outlook.message.attachments': 'outlook.attachments.list',
  'email.attachments': 'outlook.attachments.list',
  'outlook.draft': 'outlook.draft.create',
  'outlook.create.draft': 'outlook.draft.create',
  'outlook.mail.draft': 'outlook.draft.create',
  'hotmail.draft': 'outlook.draft.create',
  'outlook.send': 'outlook.mail.send',
  'hotmail.send': 'outlook.mail.send',
  'send.outlook': 'outlook.mail.send',
  'send.email.graph': 'outlook.mail.send',
  'outlook.mark.read': 'outlook.message.mark_read',
  'outlook.message.mark.read': 'outlook.message.mark_read',
  'email.mark.read': 'outlook.message.mark_read',
  'outlook.mark.unread': 'outlook.message.mark_unread',
  'outlook.message.mark.unread': 'outlook.message.mark_unread',
  'email.mark.unread': 'outlook.message.mark_unread',
  'outlook.move': 'outlook.message.move',
  'email.move': 'outlook.message.move',
  'outlook.archive': 'outlook.message.archive',
  'email.archive': 'outlook.message.archive',
  'outlook.delete': 'outlook.message.delete',
  'email.delete': 'outlook.message.delete',
  'outlook.reply': 'outlook.message.reply',
  'email.reply': 'outlook.message.reply',
  'outlook.reply.all': 'outlook.message.reply_all',
  'outlook.replyall': 'outlook.message.reply_all',
  'outlook.message.reply.all': 'outlook.message.reply_all',
  'email.reply.all': 'outlook.message.reply_all',
  'outlook.forward': 'outlook.message.forward',
  'email.forward': 'outlook.message.forward',
  'alarm.auth.status': 'alarm.authorization_status',
  'alarm.authorization': 'alarm.authorization_status',
  'alarm.authorization.status': 'alarm.authorization_status',
  'alarm.status': 'alarm.authorization_status',
  'alarm.permission.status': 'alarm.authorization_status',
  'alarm.request.auth': 'alarm.request_authorization',
  'alarm.request.authorization': 'alarm.request_authorization',
  'request.alarm.authorization': 'alarm.request_authorization',
  'request.alarm.permission': 'alarm.request_authorization',
  'schedule.alarm': 'alarm.schedule',
  'create.alarm': 'alarm.schedule',
  'set.alarm': 'alarm.schedule',
  'alarm.create': 'alarm.schedule',
  'countdown.alarm': 'alarm.countdown',
  'start.countdown': 'alarm.countdown',
  'timer.start': 'alarm.countdown',
  'start.timer': 'alarm.countdown',
  'list.alarms': 'alarm.list',
  'alarms.list': 'alarm.list',
  'show.alarms': 'alarm.list',
  'pause.alarm': 'alarm.pause',
  'resume.alarm': 'alarm.resume',
  'stop.alarm': 'alarm.stop',
  'snooze.alarm': 'alarm.snooze',
  'cancel.alarm': 'alarm.cancel',
  'delete.alarm': 'alarm.cancel',
};

const ARGUMENT_ALIASES: Readonly<
  Record<string, Readonly<Record<string, string>>>
> = {
  'messages.draft': {
    recipient: 'to',
    number: 'to',
    message: 'body',
    text: 'body',
  },
  'mail.draft': {
    recipient: 'to',
    email: 'to',
    title: 'subject',
    message: 'body',
    text: 'body',
  },
  'maps.search': {
    location: 'query',
    destination: 'query',
    place: 'query',
    nearby: 'query',
  },
  'maps.directions': {
    query: 'destination',
    location: 'destination',
    place: 'destination',
  },
  weather: { query: 'location', city: 'location' },
  'web.search': { q: 'query', term: 'query', search: 'query' },
  'web.fetch': { uri: 'url', link: 'url', query: 'url' },
  'outlook.messages.search': {
    q: 'query',
    term: 'query',
    search: 'query',
    subject: 'query',
    from: 'query',
  },
  'outlook.message.read': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.attachments.list': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.message.mark_read': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.message.mark_unread': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.message.move': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
    destinationId: 'destination',
  },
  'outlook.message.archive': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.message.delete': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
  },
  'outlook.message.reply': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
    comment: 'body',
  },
  'outlook.message.reply_all': {
    id: 'messageId',
    messageID: 'messageId',
    message: 'messageId',
    comment: 'body',
  },
  'outlook.draft.create': {
    recipient: 'to',
    recipients: 'to',
    email: 'to',
    message: 'body',
    text: 'body',
    content: 'body',
    comment: 'body',
  },
  'outlook.mail.send': {
    recipient: 'to',
    recipients: 'to',
    email: 'to',
    message: 'body',
    text: 'body',
    content: 'body',
    comment: 'body',
  },
  'outlook.message.forward': {
    id: 'messageId',
    messageID: 'messageId',
    recipient: 'to',
    recipients: 'to',
    email: 'to',
    text: 'body',
    content: 'body',
    comment: 'body',
  },
};

export class AgentToolValidationError extends Error {
  constructor(
    readonly code:
      | 'unknown_tool'
      | 'unavailable_tool'
      | 'missing_argument'
      | 'empty_argument'
      | 'wrong_type'
      | 'invalid_enum'
      | 'extra_arguments'
      | 'conflicting_alias'
      | 'invalid_arguments',
    message: string,
  ) {
    super(message);
    this.name = 'AgentToolValidationError';
  }
}

const normalizeIdentifier = (raw: string): string =>
  raw
    .trim()
    .toLowerCase()
    .replace(/[.\-\s]+/g, '.')
    .replace(/^\.|\.$/g, '');

export const canonicalToolId = (raw: string): string => {
  const normalized = normalizeIdentifier(raw);
  return CANONICAL_TOOL_IDS.has(normalized)
    ? normalized
    : (TOOL_ALIASES[normalized] ?? normalized);
};

export const toolDefinition = (
  rawToolId: string,
): AgentToolDefinition | undefined =>
  DEFINITIONS_BY_ID.get(canonicalToolId(rawToolId));

const isJSONObject = (value: unknown): value is JSONObject =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const canonicalJSON = (value: JSONValue): string => {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJSON).join(',')}]`;
  }
  if (isJSONObject(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJSON(value[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
};

const valuesEqual = (left: JSONValue, right: JSONValue): boolean =>
  canonicalJSON(left) === canonicalJSON(right);

const valueMatches = (
  value: JSONValue,
  expected: AgentArgumentType,
): boolean => {
  switch (expected) {
    case 'string':
    case 'enum':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'bool':
      return typeof value === 'boolean';
    case 'array':
      return Array.isArray(value);
    case 'object':
      return isJSONObject(value);
  }
};

const integerInRange = (
  value: JSONValue | undefined,
  minimum: number,
  maximum: number,
): boolean =>
  typeof value === 'number' &&
  Number.isInteger(value) &&
  value >= minimum &&
  value <= maximum;

const invalidArgumentRelationship = (
  id: string,
  args: JSONObject,
): string | undefined => {
  if (id === 'outlook.message.move') {
    if (
      typeof args.destination !== 'string' ||
      args.destination.trim() === ''
    ) {
      return 'provide one non-empty destination or destinationId.';
    }
  }

  if (id === 'outlook.message.reply' || id === 'outlook.message.reply_all') {
    if (typeof args.body !== 'string' || args.body.trim() === '') {
      return 'provide one non-empty body or comment.';
    }
  }

  if (id === 'trigger.create') {
    const schedule = args.schedule;
    const scheduleFields = [
      'inMinutes',
      'atTime',
      'intervalSeconds',
      'beforeMinutes',
    ] as const;
    const supplied = scheduleFields.filter(
      (field) => args[field] !== undefined,
    );
    if (
      schedule === 'relative' &&
      (supplied.length !== 1 ||
        supplied[0] !== 'inMinutes' ||
        !integerInRange(args.inMinutes, 1, 366 * 24 * 60))
    ) {
      return 'relative requires only integer inMinutes from 1 through 527040.';
    }
    if (
      schedule === 'absolute' &&
      (supplied.length !== 1 ||
        supplied[0] !== 'atTime' ||
        typeof args.atTime !== 'string' ||
        !/^(?:[01]\d|2[0-3]):[0-5]\d$/.test(args.atTime))
    ) {
      return 'absolute requires only atTime in strict HH:mm format.';
    }
    if (
      schedule === 'interval' &&
      (supplied.length !== 1 ||
        supplied[0] !== 'intervalSeconds' ||
        !integerInRange(args.intervalSeconds, 60, 31 * 86_400))
    ) {
      return 'interval requires only integer intervalSeconds from 60 through 2678400.';
    }
    if (
      schedule === 'before_next_event' &&
      (supplied.some((field) => field !== 'beforeMinutes') ||
        (args.beforeMinutes !== undefined &&
          !integerInRange(args.beforeMinutes, 1, 24 * 60)))
    ) {
      return 'before_next_event accepts only optional integer beforeMinutes from 1 through 1440.';
    }
  }

  if (id === 'trigger.cancel') {
    const triggerId = typeof args.id === 'string' ? args.id.trim() : '';
    const title = typeof args.title === 'string' ? args.title.trim() : '';
    if (triggerId.length > 0 === title.length > 0) {
      return 'provide exactly one non-empty id or title.';
    }
    if (
      triggerId.length > 0 &&
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
        triggerId,
      )
    ) {
      return 'id must be a UUID.';
    }
  }

  if (id === 'alarm.schedule') {
    if (args.repeats === true) {
      return 'repeats=true is unsupported; alarm.schedule creates one-shot alarms only.';
    }
    const supplied = ['inMinutes', 'timestamp'].filter(
      (field) => args[field] !== undefined,
    );
    if (supplied.length !== 1) {
      return 'provide exactly one of inMinutes or timestamp.';
    }
    if (
      args.inMinutes !== undefined &&
      !integerInRange(args.inMinutes, 1, 366 * 24 * 60)
    ) {
      return 'inMinutes must be an integer from 1 through 527040.';
    }
    if (args.timestamp !== undefined) {
      const normalized =
        typeof args.timestamp === 'string' ? args.timestamp.trim() : '';
      if (normalized.length === 0 || !Number.isFinite(Number(normalized))) {
        return 'timestamp must contain finite Unix seconds.';
      }
    }
    if (
      args.snoozeMinutes !== undefined &&
      !integerInRange(args.snoozeMinutes, 1, 24 * 60)
    ) {
      return 'snoozeMinutes must be an integer from 1 through 1440.';
    }
  }

  if (
    id === 'alarm.countdown' &&
    !integerInRange(args.durationSeconds, 1, 366 * 24 * 60 * 60)
  ) {
    return 'durationSeconds must be a positive bounded integer.';
  }

  return undefined;
};

export const validateToolCall = (
  rawToolId: string,
  rawArguments: JSONObject,
  availableToolIds: ReadonlySet<string>,
): ValidatedAgentToolCall => {
  const id = canonicalToolId(rawToolId);
  const definition = DEFINITIONS_BY_ID.get(id);
  if (!definition) {
    throw new AgentToolValidationError(
      'unknown_tool',
      `Unknown tool: ${rawToolId}.`,
    );
  }
  const canonicalAvailable = new Set(
    [...availableToolIds].map(canonicalToolId),
  );
  if (!canonicalAvailable.has(id)) {
    throw new AgentToolValidationError(
      'unavailable_tool',
      `Tool is not available in this run: ${id}.`,
    );
  }

  const aliases = ARGUMENT_ALIASES[id] ?? {};
  const declaredNames = new Set(
    definition.arguments.map((schema) => schema.name),
  );
  const acceptedNames = new Set([...declaredNames, ...Object.keys(aliases)]);
  const extra = Object.keys(rawArguments)
    .filter((key) => !acceptedNames.has(key))
    .sort();
  if (extra.length > 0) {
    throw new AgentToolValidationError(
      'extra_arguments',
      `Unexpected arguments for ${id}: ${extra.join(', ')}.`,
    );
  }

  const args: JSONObject = { ...rawArguments };
  const aliasesByCanonical = new Map<string, string[]>();
  Object.entries(aliases).forEach(([alias, canonical]) => {
    aliasesByCanonical.set(canonical, [
      ...(aliasesByCanonical.get(canonical) ?? []),
      alias,
    ]);
  });
  [...aliasesByCanonical]
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([canonical, aliasNames]) => {
      const supplied = aliasNames
        .sort()
        .filter((alias) => rawArguments[alias] !== undefined);
      if (supplied.length === 0) {
        return;
      }
      const reference = args[canonical] ?? rawArguments[supplied[0]];
      supplied.forEach((alias) => {
        if (!valuesEqual(reference, rawArguments[alias])) {
          throw new AgentToolValidationError(
            'conflicting_alias',
            `Conflicting values for ${id}.${canonical} and alias ${alias}.`,
          );
        }
        delete args[alias];
      });
      args[canonical] = reference;
    });

  for (const schema of definition.arguments) {
    const value = args[schema.name];
    if (value === undefined) {
      if (schema.required) {
        throw new AgentToolValidationError(
          'missing_argument',
          `Missing required argument ${schema.name} for ${id}.`,
        );
      }
      continue;
    }
    if (!valueMatches(value, schema.type)) {
      throw new AgentToolValidationError(
        'wrong_type',
        `Invalid type for ${id}.${schema.name}: expected ${schema.type}.`,
      );
    }
    if (schema.required && typeof value === 'string' && value.trim() === '') {
      throw new AgentToolValidationError(
        'empty_argument',
        `Required argument ${schema.name} for ${id} must not be empty.`,
      );
    }
    if (schema.type === 'enum') {
      const canonicalValue = schema.allowedValues?.find(
        (allowed) =>
          allowed.toLocaleLowerCase('en-US') ===
          (value as string).toLocaleLowerCase('en-US'),
      );
      if (!canonicalValue) {
        throw new AgentToolValidationError(
          'invalid_enum',
          `Invalid value for ${id}.${schema.name}; allowed: ${schema.allowedValues?.join(', ')}.`,
        );
      }
      args[schema.name] = canonicalValue;
    }
  }

  if (id === 'alarm.schedule' && args.snoozeMinutes === undefined) {
    // Canonicalize Lumen's implicit post-alert countdown before approval so
    // the displayed and payload-bound arguments match execution exactly.
    args.snoozeMinutes = 5;
  }

  const normalizedExtra = Object.keys(args)
    .filter((key) => !declaredNames.has(key))
    .sort();
  if (normalizedExtra.length > 0) {
    throw new AgentToolValidationError(
      'extra_arguments',
      `Unexpected arguments for ${id}: ${normalizedExtra.join(', ')}.`,
    );
  }
  const relationshipError = invalidArgumentRelationship(id, args);
  if (relationshipError) {
    throw new AgentToolValidationError(
      'invalid_arguments',
      `Invalid arguments for ${id}: ${relationshipError}`,
    );
  }
  return { tool: id, args, definition };
};

export const duplicateCallKey = (
  call: Pick<ValidatedAgentToolCall, 'tool' | 'args'>,
): string => `${call.tool}:${canonicalJSON(call.args)}`;
