import Foundation

public struct AgentToolID: RawRepresentable, Hashable, Sendable, Codable,
  Comparable, CustomStringConvertible, ExpressibleByStringLiteral
{
  public let rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: String) {
    rawValue = value
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    rawValue = try container.decode(String.self)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(rawValue)
  }

  public var description: String { rawValue }

  public static func < (lhs: AgentToolID, rhs: AgentToolID) -> Bool {
    lhs.rawValue < rhs.rawValue
  }
}

public enum AgentToolArgumentType: String, Codable, Sendable, Equatable {
  case string
  case number
  case boolean = "bool"
  case array
  case object
  case enumeration = "enum"
}

public struct AgentToolArgumentSchema: Codable, Sendable, Equatable {
  public let name: String
  public let type: AgentToolArgumentType
  public let required: Bool
  public let allowedValues: [String]?

  public init(
    name: String,
    type: AgentToolArgumentType = .string,
    required: Bool = true,
    allowedValues: Set<String>? = nil
  ) {
    self.name = name
    self.type = type
    self.required = required
    self.allowedValues = allowedValues?.sorted()
  }
}

public enum AgentToolCategory: String, Codable, Sendable, Equatable {
  case productivity
  case communication
  case location
  case media
  case health
  case knowledge
}

public enum AgentPermission: String, Codable, Sendable, Equatable, Hashable, CaseIterable {
  case calendar
  case reminders
  case contacts
  case location
  case photos
  case camera
  case health
  case motion
  case alarms
  case notifications
}

public enum AgentToolRisk: Int, Codable, Sendable, Comparable {
  case low
  case moderate
  case high
  case critical

  public static func < (lhs: AgentToolRisk, rhs: AgentToolRisk) -> Bool {
    lhs.rawValue < rhs.rawValue
  }
}

public struct AgentToolDefinition: Codable, Sendable, Equatable {
  public let id: AgentToolID
  public let displayName: String
  public let description: String
  public let category: AgentToolCategory
  public let arguments: [AgentToolArgumentSchema]
  public let permission: AgentPermission?
  public let risk: AgentToolRisk
  public let requiresApproval: Bool
  public let supportsBackgroundExecution: Bool
  public let maximumOutputCharacters: Int

  public init(
    id: AgentToolID,
    displayName: String,
    description: String,
    category: AgentToolCategory,
    arguments: [AgentToolArgumentSchema],
    permission: AgentPermission?,
    risk: AgentToolRisk,
    requiresApproval: Bool,
    supportsBackgroundExecution: Bool,
    maximumOutputCharacters: Int = 2_400
  ) {
    self.id = id
    self.displayName = displayName
    self.description = description
    self.category = category
    self.arguments = arguments
    self.permission = permission
    self.risk = risk
    self.requiresApproval = requiresApproval
    self.supportsBackgroundExecution = supportsBackgroundExecution
    self.maximumOutputCharacters = max(256, maximumOutputCharacters)
  }
}

/// Canonical, platform-neutral contracts. Implementations are intentionally
/// supplied by the host application through `AgentToolExecuting`.
public enum AgentToolCatalog {
  public static let all: [AgentToolDefinition] = [
    tool("calendar.create", "Create Event", "Add an event to the calendar.", .productivity,
         [arg("title"), arg("startsInMinutes", .number)], .calendar, .high, true, false),
    tool("calendar.list", "List Events", "Read upcoming calendar events.", .productivity,
         [], .calendar, .moderate, false, true),
    tool("reminders.create", "Add Reminder", "Create a reminder.", .productivity,
         [arg("title")], .reminders, .high, true, false),
    tool("reminders.list", "List Reminders", "Read pending reminders.", .productivity,
         [], .reminders, .moderate, false, true),
    tool("contacts.search", "Search Contacts", "Find a contact by name.", .communication,
         [arg("query")], .contacts, .moderate, false, false),
    tool("messages.draft", "Draft Message", "Compose an iMessage or SMS draft.", .communication,
         [arg("to"), arg("body"), arg("recipient", required: false),
          arg("number", required: false), arg("message", required: false),
          arg("text", required: false)], nil, .high, true, false),
    tool("mail.draft", "Draft Email", "Compose a system email draft.", .communication,
         [arg("to"), arg("subject", required: false), arg("body"),
          arg("recipient", required: false), arg("email", required: false),
          arg("message", required: false), arg("text", required: false),
          arg("title", required: false)], nil, .high, true, false),

    tool("outlook.status", "Outlook Status", "Check Outlook sign-in status.", .communication,
         [], nil, .low, false, true),
    tool("outlook.folders.list", "List Outlook Folders", "List Outlook mail folders.", .communication,
         [arg("includeHidden", .boolean, required: false)], nil, .moderate, false, true),
    tool("outlook.messages.list", "List Outlook Messages", "List recent Outlook messages.", .communication,
         [arg("folder", required: false), arg("folderId", required: false),
          arg("limit", .number, required: false), arg("unreadOnly", .boolean, required: false)],
         nil, .moderate, false, true, 4_000),
    tool("outlook.messages.search", "Search Outlook Messages", "Search Outlook mail.", .communication,
         [arg("query"), arg("folder", required: false), arg("folderId", required: false),
          arg("limit", .number, required: false)], nil, .moderate, false, true, 4_000),
    tool("outlook.message.read", "Read Outlook Message", "Read one Outlook message.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .moderate, false, true, 6_000),
    tool("outlook.attachments.list", "List Outlook Attachments", "List attachment metadata.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .moderate, false, true),
    tool("outlook.draft.create", "Create Outlook Draft", "Create a saved Outlook draft.", .communication,
         [arg("to"), arg("subject"), arg("body")], nil, .high, true, false),
    tool("outlook.mail.send", "Send Outlook Email", "Send mail through Outlook.", .communication,
         [arg("to"), arg("subject"), arg("body")], nil, .high, true, false),
    tool("outlook.message.mark_read", "Mark Outlook Read", "Mark an Outlook message read.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .high, true, false),
    tool("outlook.message.mark_unread", "Mark Outlook Unread", "Mark an Outlook message unread.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .high, true, false),
    tool("outlook.message.move", "Move Outlook Message", "Move an Outlook message.", .communication,
         [arg("messageId"), arg("destination", required: false), arg("id", required: false),
          arg("destinationId", required: false)], nil, .high, true, false),
    tool("outlook.message.archive", "Archive Outlook Message", "Archive an Outlook message.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .critical, true, false),
    tool("outlook.message.delete", "Delete Outlook Message", "Delete an Outlook message.", .communication,
         [arg("messageId"), arg("id", required: false)], nil, .critical, true, false),
    tool("outlook.message.reply", "Reply Outlook Message", "Reply to an Outlook message.", .communication,
         [arg("messageId"), arg("body", required: false), arg("id", required: false),
          arg("comment", required: false)], nil, .high, true, false),
    tool("outlook.message.reply_all", "Reply All Outlook Message", "Reply all to an Outlook message.", .communication,
         [arg("messageId"), arg("body", required: false), arg("id", required: false),
          arg("comment", required: false)], nil, .high, true, false),
    tool("outlook.message.forward", "Forward Outlook Message", "Forward an Outlook message.", .communication,
         [arg("messageId"), arg("to"), arg("id", required: false),
          arg("body", required: false), arg("comment", required: false)],
         nil, .high, true, false),
    tool("phone.call", "Start Call", "Open the phone dialer.", .communication,
         [arg("number")], nil, .high, true, false),

    tool("location.current", "Current Location", "Read the current GPS location.", .location,
         [], .location, .moderate, false, false),
    tool("weather", "Current Weather", "Get current weather from a city or location.", .location,
         [arg("location", required: false), arg("city", required: false)],
         .location, .low, false, true, 4_000),
    tool("maps.directions", "Get Directions", "Get directions to a destination.", .location,
         [arg("destination")], nil, .moderate, false, false),
    tool("maps.search", "Search Nearby", "Search for nearby places.", .location,
         [arg("query")], .location, .moderate, false, false),
    tool("photos.search", "Search Photos", "Search the local photo library.", .media,
         [arg("query")], .photos, .moderate, false, false),
    tool("camera.capture", "Capture Image", "Capture a device image.", .media,
         [], .camera, .high, true, false),
    tool("health.summary", "Health Summary", "Read a local health summary.", .health,
         [], .health, .moderate, false, false),
    tool("motion.activity", "Motion Activity", "Read recent motion activity.", .health,
         [], .motion, .moderate, false, true),

    tool("web.search", "Web Search", "Search the public web.", .knowledge,
         [arg("query")], nil, .low, false, false, 4_000),
    tool("web.fetch", "Fetch URL", "Fetch a specific web page.", .knowledge,
         [arg("url")], nil, .low, false, false, 4_000),
    tool("files.read", "Read File", "Read a previously imported local document.", .knowledge,
         [arg("name")], nil, .moderate, false, true),
    tool("memory.save", "Save Memory", "Store a user fact or preference.", .knowledge,
         [arg("content"), arg("kind")], nil, .moderate, false, false),
    tool("memory.recall", "Recall Memory", "Search stored memories.", .knowledge,
         [arg("query")], nil, .moderate, false, true),
    tool("rag.search", "Search Personal Data", "Search indexed local content.", .knowledge,
         [arg("query"), arg("limit", .number, required: false),
          arg("sourceScope", .enumeration, required: false,
              allowed: ["all", "documents", "notes", "photos"])],
         nil, .moderate, false, true, 3_000),
    tool("rag.index_files", "Reindex Files", "Index imported local files.", .knowledge,
         [], nil, .moderate, false, false),
    tool("rag.index_photos", "Reindex Photos", "Index local photo metadata.", .knowledge,
         [arg("months", .number)], .photos, .moderate, false, false),

    tool("trigger.create", "Schedule Agent Run", "Create a scheduled agent trigger.", .productivity,
         [arg("title"), arg("prompt"),
          arg("schedule", .enumeration,
              allowed: ["absolute", "before_next_event", "interval", "relative"]),
          arg("inMinutes", .number, required: false), arg("atTime", required: false),
          arg("intervalSeconds", .number, required: false),
          arg("beforeMinutes", .number, required: false)],
         .notifications, .high, true, false),
    tool("trigger.list", "List Triggers", "List scheduled agent triggers.", .productivity,
         [], .notifications, .low, false, true),
    tool("trigger.cancel", "Cancel Trigger", "Cancel a scheduled agent trigger by UUID or exact title.", .productivity,
         [arg("id", required: false), arg("title", required: false)],
         .notifications, .critical, true, false),

    tool("alarm.authorization_status", "Alarm Auth Status", "Read AlarmKit authorization status.", .productivity,
         [], .alarms, .low, false, true),
    tool("alarm.request_authorization", "Request Alarm Auth", "Request AlarmKit authorization.", .productivity,
         [], .alarms, .high, true, false),
    tool("alarm.schedule", "Schedule Alarm", "Schedule an alarm with a five-minute snooze by default.", .productivity,
         [arg("title"), arg("inMinutes", .number, required: false),
          arg("timestamp", required: false),
          arg("repeats", .boolean, required: false),
          arg("snoozeMinutes", .number, required: false)],
         .alarms, .high, true, false),
    tool("alarm.countdown", "Start Countdown", "Create a countdown alarm.", .productivity,
         [arg("title"), arg("durationSeconds", .number)], .alarms, .high, true, false),
    tool("alarm.list", "List Alarms", "List active alarms.", .productivity,
         [], .alarms, .moderate, false, true),
    tool("alarm.pause", "Pause Alarm", "Pause an alarm.", .productivity,
         [arg("id")], .alarms, .high, true, false),
    tool("alarm.resume", "Resume Alarm", "Resume an alarm.", .productivity,
         [arg("id")], .alarms, .high, true, false),
    tool("alarm.stop", "Stop Alarm", "Stop an alerting alarm.", .productivity,
         [arg("id")], .alarms, .critical, true, false),
    tool("alarm.snooze", "Snooze Alarm", "Snooze an alerting alarm.", .productivity,
         [arg("id")], .alarms, .high, true, false),
    tool("alarm.cancel", "Cancel Alarm", "Cancel a scheduled alarm.", .productivity,
         [arg("id")], .alarms, .critical, true, false),
  ]

  public static let canonicalIDs: Set<AgentToolID> = Set(all.map(\.id))

  private static let definitionsByID = Dictionary(
    uniqueKeysWithValues: all.map { ($0.id, $0) }
  )

  public static func definition(for rawToolID: String) -> AgentToolDefinition? {
    definitionsByID[AgentToolNormalizer.canonicalToolID(rawToolID)]
  }

  private static func arg(
    _ name: String,
    _ type: AgentToolArgumentType = .string,
    required: Bool = true,
    allowed: Set<String>? = nil
  ) -> AgentToolArgumentSchema {
    .init(name: name, type: type, required: required, allowedValues: allowed)
  }

  private static func tool(
    _ id: AgentToolID,
    _ displayName: String,
    _ description: String,
    _ category: AgentToolCategory,
    _ arguments: [AgentToolArgumentSchema],
    _ permission: AgentPermission?,
    _ risk: AgentToolRisk,
    _ requiresApproval: Bool,
    _ supportsBackgroundExecution: Bool,
    _ maximumOutputCharacters: Int = 2_400
  ) -> AgentToolDefinition {
    .init(
      id: id,
      displayName: displayName,
      description: description,
      category: category,
      arguments: arguments,
      permission: permission,
      risk: risk,
      requiresApproval: requiresApproval,
      supportsBackgroundExecution: supportsBackgroundExecution,
      maximumOutputCharacters: maximumOutputCharacters
    )
  }
}
