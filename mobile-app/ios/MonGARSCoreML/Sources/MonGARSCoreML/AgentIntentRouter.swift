import Foundation

public enum AgentIntent: String, Codable, Sendable, CaseIterable, Equatable, Hashable {
  case weather
  case webSearch
  case emailDraft
  case messageDraft
  case phoneCall
  case contactSearch
  case calendar
  case reminder
  case maps
  case photos
  case camera
  case health
  case motion
  case files
  case memory
  case rag
  case trigger
  case alarm
  case outlook
  case note
  case chat
  case unknown
}

public struct AgentIntentRoute: Sendable, Equatable {
  public let intent: AgentIntent
  public let allowedToolIDs: Set<AgentToolID>
  public let clarification: String?

  public init(
    intent: AgentIntent,
    allowedToolIDs: Set<AgentToolID>,
    clarification: String? = nil
  ) {
    self.intent = intent
    self.allowedToolIDs = allowedToolIDs
    self.clarification = clarification
  }

  public var requiresClarification: Bool {
    clarification?.isEmpty == false
  }

  public var requiresTool: Bool {
    ![.chat, .unknown].contains(intent)
  }

  /// Tools that can actually satisfy this route. Other allowed tools (for
  /// example Contacts before drafting mail, or Location before Weather) are
  /// supporting reads and must not make a premature final answer acceptable.
  public var fulfillmentToolIDs: Set<AgentToolID> {
    switch intent {
    case .weather: return ["weather"]
    case .webSearch: return ["web.search", "web.fetch"]
    case .emailDraft: return ["mail.draft"]
    case .messageDraft: return ["messages.draft"]
    case .phoneCall: return ["phone.call"]
    case .contactSearch: return ["contacts.search"]
    case .calendar: return ["calendar.create", "calendar.list"]
    case .reminder: return ["reminders.create", "reminders.list"]
    case .maps:
      return allowedToolIDs == ["location.current"]
        ? ["location.current"]
        : ["maps.search", "maps.directions"]
    case .photos: return ["photos.search"]
    case .camera: return ["camera.capture"]
    case .health: return ["health.summary"]
    case .motion: return ["motion.activity"]
    case .files: return ["files.read"]
    case .memory, .note: return ["memory.save", "memory.recall"]
    case .rag: return ["rag.search", "rag.index_files", "rag.index_photos"]
    case .trigger: return ["trigger.create", "trigger.list", "trigger.cancel"]
    case .alarm:
      return Set(allowedToolIDs.filter { $0.rawValue.hasPrefix("alarm.") })
    case .outlook:
      return Set(allowedToolIDs.filter { $0.rawValue.hasPrefix("outlook.") })
    case .chat, .unknown: return []
    }
  }
}

public enum AgentIntentRouter {
  public static func route(intent: AgentIntent) -> AgentIntentRoute {
    .init(intent: intent, allowedToolIDs: allowedToolIDs(for: intent))
  }

  public static func route(_ userInput: String) -> AgentIntentRoute {
    let text = normalized(userInput)
    guard !text.isEmpty else { return route(intent: .chat) }

    if isAmbiguousMeeting(text) {
      return clarified(
        .unknown,
        "Do you mean a calendar event or a nearby meeting location?"
      )
    }

    if containsAny(text, ["outlook", "hotmail", "microsoft graph"]) {
      if ["outlook", "hotmail", "open outlook"].contains(text) {
        return clarified(.outlook, "What would you like to do in Outlook?")
      }
      if containsAny(text, ["search outlook", "search hotmail", "find email"]),
         !hasContentAfterAction(text, actions: ["search", "find"]) {
        return clarified(.outlook, "What should I search for in Outlook?")
      }
      if containsAny(text, ["read outlook", "read email"]), !containsMessageReference(text) {
        return clarified(.outlook, "Which Outlook message should I read?")
      }
      if text.contains("attachment"), !containsMessageReference(text) {
        return clarified(.outlook, "Which Outlook message should I inspect for attachments?")
      }
      return route(intent: .outlook)
    }

    if containsAny(text, ["alarm", "timer", "countdown", "wake me", "wake us"]) {
      if containsAny(text, ["set alarm", "schedule alarm", "wake me", "wake us"]),
         !containsTimeExpression(text) {
        return clarified(.alarm, "What time should I use for the alarm?")
      }
      if containsAny(text, ["timer", "countdown"]), !containsDuration(text) {
        return clarified(.alarm, "What duration should I use for the timer?")
      }
      for (verb, prompt) in [
        ("cancel", "Which alarm should I cancel?"),
        ("pause", "Which alarm should I pause?"),
        ("resume", "Which alarm should I resume?"),
        ("stop", "Which alarm should I stop?"),
        ("snooze", "Which alarm should I snooze?"),
      ] where text.contains("\(verb) alarm")
        && !hasContentAfterAction(text, actions: ["\(verb) alarm"]) {
        return clarified(.alarm, prompt)
      }
      return route(intent: .alarm)
    }

    if containsAny(text, ["agent run", "background agent", "trigger", "scheduled run"]) {
      if containsAny(text, ["create trigger", "schedule agent", "scheduled run"]),
         !containsAny(text, [" to ", " saying ", " prompt ", " about "]) {
        return clarified(.trigger, "What should the scheduled agent run do?")
      }
      return route(intent: .trigger)
    }

    if containsAny(text, ["reindex photos", "index photos"]), !containsDuration(text) {
      return clarified(.rag, "How many months of photos should I index?")
    }

    if containsAny(text, ["weather", "forecast", "temperature", "rain", "snow", "wind outside"]) {
      return route(intent: .weather)
    }

    if isWebSearch(text) {
      if ["search web", "web search", "look it up", "search online"].contains(text) {
        return clarified(.webSearch, "What should I search for?")
      }
      return route(intent: .webSearch)
    }

    if containsAny(text, ["draft email", "write email", "compose email", "email to", "send email"]) {
      return communicationRoute(
        intent: .emailDraft,
        text: text,
        recipientPrompt: "Who should I send it to?",
        contentPrompt: "What should the email say?",
        combinedPrompt: "Who should I send it to, and what should it say?"
      )
    }

    if containsAny(text, ["draft message", "write message", "compose message", "text message", "sms", "imessage", "send a text"]) {
      return communicationRoute(
        intent: .messageDraft,
        text: text,
        recipientPrompt: "Who should I message?",
        contentPrompt: "What should the message say?",
        combinedPrompt: "Who should I message, and what should it say?"
      )
    }

    if isPhoneCall(text) {
      let bare = ["call", "phone", "make a call", "start a call"].contains(text)
      return bare ? clarified(.phoneCall, "Who should I call?") : route(intent: .phoneCall)
    }

    if containsAny(text, ["find contact", "search contacts", "address book", "phone number for", "email address for"]) {
      let bare = ["find contact", "search contacts", "address book"].contains(text)
      return bare ? clarified(.contactSearch, "Which contact should I look up?") : route(intent: .contactSearch)
    }

    if containsAny(text, ["calendar", "event", "appointments"]) {
      if containsAny(text, ["create event", "add event", "schedule event"]),
         !containsAny(text, [" called ", " titled ", " for ", " about "]) {
        return clarified(.calendar, "What should the calendar event be?")
      }
      return route(intent: .calendar)
    }

    if containsAny(text, ["remind me", "reminder", "todo", "to do", "pending reminders"]) {
      if ["remind me", "create reminder", "add reminder", "reminder"].contains(text) {
        return clarified(.reminder, "What should I remind you about?")
      }
      return route(intent: .reminder)
    }

    if isCurrentLocation(text) {
      return .init(intent: .maps, allowedToolIDs: ["location.current"])
    }
    if containsAny(text, ["maps", "directions", "navigate", "route to", "near me", "nearby", "closest", "nearest", "show me on map"]) {
      if ["maps", "directions", "navigate", "nearby"].contains(text) {
        return clarified(.maps, "What place or destination should I look for?")
      }
      return route(intent: .maps)
    }

    if containsAny(text, ["search photos", "find photos", "photo library", "find pictures", "latest photo", "latest selfie"]) {
      let bare = ["search photos", "find photos", "find pictures", "photo library"].contains(text)
      return bare ? clarified(.photos, "Which photos should I look for?") : route(intent: .photos)
    }
    if containsAny(text, ["take a photo", "capture image", "open camera", "take picture"]) {
      return route(intent: .camera)
    }
    if containsAny(text, ["health summary", "heart rate", "health data", "sleep data", "active energy", "walking distance"]) {
      return route(intent: .health)
    }
    if containsAny(text, ["motion activity", "am i walking", "am i running", "device motion", "recent activity"]) {
      return route(intent: .motion)
    }

    if containsAny(text, ["remember", "memory", "save this fact", "what do you remember", "keep this in mind"]) {
      if ["remember", "memory", "save memory", "recall memory", "note"].contains(text) {
        return clarified(.memory, "What should I save or recall?")
      }
      return route(intent: .memory)
    }

    if containsAny(text, ["rag search", "search my files", "search my documents", "search my notes", "search personal data", "reindex files", "index files", "architecture notes"]) {
      if ["rag search", "search personal data"].contains(text) {
        return clarified(.rag, "What should I search for?")
      }
      return route(intent: .rag)
    }

    if containsAny(text, ["read file", "open file", "read document", "imported file", "local document"]) {
      let bare = ["read file", "open file", "read document", "imported file", "local document"].contains(text)
      return bare ? clarified(.files, "Which file should I read?") : route(intent: .files)
    }

    if text.hasPrefix("note ") || text.hasPrefix("save this ") {
      return route(intent: .note)
    }
    return route(intent: .chat)
  }

  public static func allowedToolIDs(for intent: AgentIntent) -> Set<AgentToolID> {
    switch intent {
    case .weather: return ["weather", "location.current"]
    case .webSearch: return ["web.search", "web.fetch"]
    case .emailDraft: return ["mail.draft", "contacts.search"]
    case .messageDraft: return ["messages.draft", "contacts.search"]
    case .phoneCall: return ["phone.call", "contacts.search"]
    case .contactSearch: return ["contacts.search"]
    case .calendar: return ["calendar.create", "calendar.list"]
    case .reminder: return ["reminders.create", "reminders.list"]
    case .maps: return ["maps.search", "maps.directions", "location.current"]
    case .photos: return ["photos.search"]
    case .camera: return ["camera.capture"]
    case .health: return ["health.summary"]
    case .motion: return ["motion.activity"]
    case .files: return ["files.read"]
    case .memory, .note: return ["memory.save", "memory.recall"]
    case .rag: return ["rag.search", "rag.index_files", "rag.index_photos", "files.read", "photos.search"]
    case .trigger: return ["trigger.create", "trigger.list", "trigger.cancel"]
    case .alarm:
      return [
        "alarm.authorization_status", "alarm.request_authorization", "alarm.schedule",
        "alarm.countdown", "alarm.list", "alarm.pause", "alarm.resume", "alarm.stop",
        "alarm.snooze", "alarm.cancel",
      ]
    case .outlook:
      return [
        "outlook.status", "outlook.folders.list", "outlook.messages.list",
        "outlook.messages.search", "outlook.message.read", "outlook.attachments.list",
        "outlook.draft.create", "outlook.mail.send", "outlook.message.mark_read",
        "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive",
        "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all",
        "outlook.message.forward", "contacts.search",
      ]
    case .chat, .unknown: return []
    }
  }

  public static func unavailableMessage(for intent: AgentIntent) -> String {
    switch intent {
    case .weather: return "Weather and location tools are unavailable."
    case .webSearch: return "Web tools are unavailable."
    case .emailDraft: return "Email drafting tools are unavailable."
    case .messageDraft: return "Message drafting tools are unavailable."
    case .phoneCall: return "Phone and contact tools are unavailable."
    case .contactSearch: return "Contact tools are unavailable."
    case .calendar: return "Calendar tools are unavailable."
    case .reminder: return "Reminder tools are unavailable."
    case .maps: return "Maps and location tools are unavailable."
    case .photos: return "Photo tools are unavailable."
    case .camera: return "Camera tools are unavailable."
    case .health: return "Health tools are unavailable."
    case .motion: return "Motion tools are unavailable."
    case .files: return "File tools are unavailable."
    case .memory, .note: return "Memory tools are unavailable."
    case .rag: return "Local retrieval tools are unavailable."
    case .trigger: return "Scheduled trigger tools are unavailable."
    case .alarm: return "Alarm tools are unavailable."
    case .outlook: return "Outlook tools are unavailable."
    case .chat, .unknown: return "No matching tool is available."
    }
  }

  private static func clarified(_ intent: AgentIntent, _ prompt: String) -> AgentIntentRoute {
    .init(intent: intent, allowedToolIDs: allowedToolIDs(for: intent), clarification: prompt)
  }

  private static func communicationRoute(
    intent: AgentIntent,
    text: String,
    recipientPrompt: String,
    contentPrompt: String,
    combinedPrompt: String
  ) -> AgentIntentRoute {
    let hasRecipient = text.range(
      of: #"\bto\s+[^\s]+"#,
      options: .regularExpression
    ) != nil || text.contains("@")
    let hasContent = containsAny(text, [" saying ", " say ", " body ", " that ", " about "])
    if !hasRecipient, !hasContent { return clarified(intent, combinedPrompt) }
    if !hasRecipient { return clarified(intent, recipientPrompt) }
    if !hasContent { return clarified(intent, contentPrompt) }
    return route(intent: intent)
  }

  private static func normalized(_ text: String) -> String {
    text.trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
  }

  private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
    needles.contains { text.contains($0) }
  }

  private static func isAmbiguousMeeting(_ text: String) -> Bool {
    ["meeting", "find a meeting", "show meetings"].contains(text)
  }

  private static func isWebSearch(_ text: String) -> Bool {
    containsAny(text, ["search web", "web search", "search online", "look up", "find online", "http://", "https://"])
  }

  private static func isPhoneCall(_ text: String) -> Bool {
    text.range(of: #"\b(call|dial|phone)\b"#, options: .regularExpression) != nil
  }

  private static func isCurrentLocation(_ text: String) -> Bool {
    containsAny(text, ["current location", "where am i", "my gps location", "where are we"])
  }

  private static func containsMessageReference(_ text: String) -> Bool {
    text.range(of: #"\b[0-9a-f]{6,}\b|\b(first|second|third|latest|last)\b"#, options: .regularExpression) != nil
  }

  private static func containsTimeExpression(_ text: String) -> Bool {
    containsAny(text, ["today", "tomorrow", "tonight", "morning", "afternoon", "evening", " in "])
      || text.range(of: #"\b\d{1,2}(?::\d{2})?\s*(am|pm)?\b"#, options: .regularExpression) != nil
  }

  private static func containsDuration(_ text: String) -> Bool {
    text.range(of: #"\b\d+(?:\.\d+)?\s*(seconds?|minutes?|hours?|months?)\b"#, options: .regularExpression) != nil
  }

  private static func hasContentAfterAction(_ text: String, actions: [String]) -> Bool {
    actions.contains { action in
      guard let range = text.range(of: action) else { return false }
      return !String(text[range.upperBound...])
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .isEmpty
    }
  }
}
