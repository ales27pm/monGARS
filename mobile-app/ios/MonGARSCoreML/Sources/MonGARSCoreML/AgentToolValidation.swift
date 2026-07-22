import Foundation

public enum AgentToolNormalizer {
  /// Canonicalizes only semantics-preserving aliases. In particular, `open.url`
  /// is not an alias for `web.fetch`: opening and reading a URL have different
  /// user-visible and approval semantics.
  public static func canonicalToolID(_ raw: String) -> AgentToolID {
    let normalized = normalizedIdentifier(raw)
    if AgentToolCatalog.canonicalIDs.contains(.init(rawValue: normalized)) {
      return .init(rawValue: normalized)
    }
    return .init(rawValue: toolAliases[normalized] ?? normalized)
  }

  public static func normalizedArguments(
    for toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) throws -> AgentJSONArguments {
    let aliases = argumentAliases[toolID] ?? [:]
    var output = arguments

    let aliasesByCanonical = Dictionary(grouping: aliases.keys, by: { aliases[$0] ?? $0 })
    for canonicalName in aliasesByCanonical.keys.sorted() {
      let aliasNames = aliasesByCanonical[canonicalName, default: []].sorted()
      let suppliedAliases = aliasNames.compactMap { alias -> (String, AgentJSONValue)? in
        guard let value = arguments[alias] else { return nil }
        return (alias, value)
      }
      guard !suppliedAliases.isEmpty else { continue }

      let reference = output[canonicalName] ?? suppliedAliases[0].1
      for (alias, value) in suppliedAliases where value != reference {
        throw AgentToolValidationError.conflictingAlias(
          tool: toolID,
          canonicalArgument: canonicalName,
          alias: alias
        )
      }
      output[canonicalName] = reference
      for (alias, _) in suppliedAliases {
        output.removeValue(forKey: alias)
      }
    }
    return output
  }

  public static func acceptedAliasNames(for toolID: AgentToolID) -> Set<String> {
    Set((argumentAliases[toolID] ?? [:]).keys)
  }

  private static func normalizedIdentifier(_ raw: String) -> String {
    var result = ""
    var previousWasSeparator = false
    for scalar in raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased().unicodeScalars {
      let isSeparator = scalar == "-" || scalar == "."
        || CharacterSet.whitespacesAndNewlines.contains(scalar)
      if isSeparator {
        if !result.isEmpty, !previousWasSeparator {
          result.append(".")
        }
        previousWasSeparator = true
      } else {
        result.unicodeScalars.append(scalar)
        previousWasSeparator = false
      }
    }
    while result.last == "." { result.removeLast() }
    return result
  }

  private static let toolAliases: [String: String] = [
    "weather.current": "weather", "current.weather": "weather",
    "forecast.current": "weather", "weather.get": "weather",
    "get.weather": "weather", "getweather": "weather", "currentweather": "weather",
    "search": "web.search", "internet.search": "web.search", "web": "web.search",
    "websearch": "web.search", "browser.search": "web.search",
    "google.search": "web.search", "google": "web.search",
    "search.web": "web.search", "searchweb": "web.search",
    "fetch": "web.fetch", "browser.fetch": "web.fetch", "url.fetch": "web.fetch",
    "fetch.url": "web.fetch", "read.url": "web.fetch", "read.website": "web.fetch",
    "maps": "maps.search", "map": "maps.search", "map.search": "maps.search",
    "nearby.search": "maps.search", "local.search": "maps.search",
    "places.search": "maps.search", "place.search": "maps.search",
    "google.maps": "maps.search", "google.maps.api": "maps.search",
    "googlemaps": "maps.search", "googlemapsapi": "maps.search",
    "maps.api": "maps.search", "mapsapi": "maps.search",
    "nearest.place": "maps.search", "find.nearby": "maps.search",
    "map.directions": "maps.directions", "directions": "maps.directions",
    "navigation": "maps.directions", "navigate": "maps.directions",
    "route": "maps.directions", "route.to": "maps.directions", "open.maps": "maps.directions",
    "location": "location.current", "gps": "location.current",
    "current.location": "location.current", "location.get": "location.current",
    "get.location": "location.current", "currentlocation": "location.current",
    "location.snapshot": "location.current",
    "calendar": "calendar.create", "create.event": "calendar.create",
    "event.create": "calendar.create", "schedule.event": "calendar.create",
    "calendar.read": "calendar.list", "list.events": "calendar.list", "events.list": "calendar.list",
    "reminder": "reminders.create", "reminder.create": "reminders.create",
    "create.reminder": "reminders.create", "reminder.list": "reminders.list",
    "list.reminders": "reminders.list",
    "mail": "mail.draft", "email": "mail.draft", "email.draft": "mail.draft",
    "compose.email": "mail.draft", "message": "messages.draft", "sms": "messages.draft",
    "sms.draft": "messages.draft", "compose.message": "messages.draft", "imessage": "messages.draft",
    "phone": "phone.call", "call": "phone.call", "dial": "phone.call",
    "contacts": "contacts.search", "contact.search": "contacts.search",
    "search.contacts": "contacts.search", "contacts.lookup": "contacts.search",
    "memory.search": "memory.recall", "rag.search.secure": "rag.search",
    "outlook": "outlook.status", "microsoft.outlook.status": "outlook.status",
    "hotmail.status": "outlook.status", "graph.status": "outlook.status",
    "outlook.folders": "outlook.folders.list", "outlook.folder.list": "outlook.folders.list",
    "hotmail.folders": "outlook.folders.list", "mail.folders.list": "outlook.folders.list",
    "outlook.messages": "outlook.messages.list", "outlook.inbox": "outlook.messages.list",
    "outlook.mail.list": "outlook.messages.list", "hotmail.inbox": "outlook.messages.list",
    "hotmail.messages": "outlook.messages.list", "graph.mail.list": "outlook.messages.list",
    "outlook.search": "outlook.messages.search", "outlook.mail.search": "outlook.messages.search",
    "hotmail.search": "outlook.messages.search", "search.outlook": "outlook.messages.search",
    "search.email": "outlook.messages.search", "email.search": "outlook.messages.search",
    "outlook.read": "outlook.message.read", "outlook.mail.read": "outlook.message.read",
    "read.outlook": "outlook.message.read", "read.email": "outlook.message.read",
    "outlook.attachments": "outlook.attachments.list",
    "outlook.message.attachments": "outlook.attachments.list", "email.attachments": "outlook.attachments.list",
    "outlook.draft": "outlook.draft.create", "outlook.create.draft": "outlook.draft.create",
    "outlook.mail.draft": "outlook.draft.create", "hotmail.draft": "outlook.draft.create",
    "outlook.send": "outlook.mail.send", "hotmail.send": "outlook.mail.send",
    "send.outlook": "outlook.mail.send", "send.email.graph": "outlook.mail.send",
    "outlook.mark.read": "outlook.message.mark_read",
    "outlook.message.mark.read": "outlook.message.mark_read", "email.mark.read": "outlook.message.mark_read",
    "outlook.mark.unread": "outlook.message.mark_unread",
    "outlook.message.mark.unread": "outlook.message.mark_unread",
    "email.mark.unread": "outlook.message.mark_unread",
    "outlook.move": "outlook.message.move", "email.move": "outlook.message.move",
    "outlook.archive": "outlook.message.archive", "email.archive": "outlook.message.archive",
    "outlook.delete": "outlook.message.delete", "email.delete": "outlook.message.delete",
    "outlook.reply": "outlook.message.reply", "email.reply": "outlook.message.reply",
    "outlook.reply.all": "outlook.message.reply_all", "outlook.replyall": "outlook.message.reply_all",
    "outlook.message.reply.all": "outlook.message.reply_all", "email.reply.all": "outlook.message.reply_all",
    "outlook.forward": "outlook.message.forward", "email.forward": "outlook.message.forward",
    "alarm.auth.status": "alarm.authorization_status",
    "alarm.authorization": "alarm.authorization_status",
    "alarm.authorization.status": "alarm.authorization_status",
    "alarm.status": "alarm.authorization_status", "alarm.permission.status": "alarm.authorization_status",
    "alarm.request.auth": "alarm.request_authorization",
    "alarm.request.authorization": "alarm.request_authorization",
    "request.alarm.authorization": "alarm.request_authorization",
    "request.alarm.permission": "alarm.request_authorization",
    "schedule.alarm": "alarm.schedule", "create.alarm": "alarm.schedule",
    "set.alarm": "alarm.schedule", "alarm.create": "alarm.schedule",
    "countdown.alarm": "alarm.countdown", "start.countdown": "alarm.countdown",
    "timer.start": "alarm.countdown", "start.timer": "alarm.countdown",
    "list.alarms": "alarm.list", "alarms.list": "alarm.list", "show.alarms": "alarm.list",
    "pause.alarm": "alarm.pause", "resume.alarm": "alarm.resume",
    "stop.alarm": "alarm.stop", "snooze.alarm": "alarm.snooze",
    "cancel.alarm": "alarm.cancel", "delete.alarm": "alarm.cancel",
  ]

  private static let argumentAliases: [AgentToolID: [String: String]] = [
    "messages.draft": ["recipient": "to", "number": "to", "message": "body", "text": "body"],
    "mail.draft": ["recipient": "to", "email": "to", "title": "subject",
                   "message": "body", "text": "body"],
    "maps.search": ["location": "query", "destination": "query", "place": "query", "nearby": "query"],
    "maps.directions": ["query": "destination", "location": "destination", "place": "destination"],
    "weather": ["query": "location", "city": "location"],
    "web.search": ["q": "query", "term": "query", "search": "query"],
    "web.fetch": ["uri": "url", "link": "url", "query": "url"],
    "outlook.messages.search": ["q": "query", "term": "query", "search": "query",
                                "subject": "query", "from": "query"],
    "outlook.message.read": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.attachments.list": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.message.mark_read": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.message.mark_unread": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.message.move": ["id": "messageId", "messageID": "messageId", "message": "messageId",
                              "destinationId": "destination"],
    "outlook.message.archive": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.message.delete": ["id": "messageId", "messageID": "messageId", "message": "messageId"],
    "outlook.message.reply": ["id": "messageId", "messageID": "messageId", "message": "messageId",
                               "comment": "body"],
    "outlook.message.reply_all": ["id": "messageId", "messageID": "messageId", "message": "messageId",
                                   "comment": "body"],
    "outlook.draft.create": ["recipient": "to", "recipients": "to", "email": "to",
                              "message": "body", "text": "body", "content": "body", "comment": "body"],
    "outlook.mail.send": ["recipient": "to", "recipients": "to", "email": "to",
                           "message": "body", "text": "body", "content": "body", "comment": "body"],
    "outlook.message.forward": ["id": "messageId", "messageID": "messageId",
                                 "recipient": "to", "recipients": "to", "email": "to",
                                 "text": "body", "content": "body", "comment": "body"],
  ]
}

public enum AgentJSONKind: String, Sendable, Equatable {
  case null
  case boolean
  case number
  case string
  case array
  case object

  init(_ value: AgentJSONValue) {
    switch value {
    case .null: self = .null
    case .bool: self = .boolean
    case .number: self = .number
    case .string: self = .string
    case .array: self = .array
    case .object: self = .object
    }
  }
}

public enum AgentToolValidationError: Error, Sendable, Equatable {
  case unknownTool(String)
  case unavailableTool(AgentToolID)
  case missingRequiredArgument(tool: AgentToolID, argument: String)
  case emptyRequiredArgument(tool: AgentToolID, argument: String)
  case invalidArgumentType(
    tool: AgentToolID,
    argument: String,
    expected: AgentToolArgumentType,
    actual: AgentJSONKind
  )
  case invalidEnumValue(tool: AgentToolID, argument: String, allowed: [String])
  case extraArguments(tool: AgentToolID, arguments: [String])
  case conflictingAlias(tool: AgentToolID, canonicalArgument: String, alias: String)
  case invalidArgumentCombination(tool: AgentToolID, reason: String)

  public var diagnostic: String {
    switch self {
    case let .unknownTool(tool):
      return "Unknown tool: \(tool)."
    case let .unavailableTool(tool):
      return "Tool is not available in this run: \(tool.rawValue)."
    case let .missingRequiredArgument(tool, argument):
      return "Missing required argument \(argument) for \(tool.rawValue)."
    case let .emptyRequiredArgument(tool, argument):
      return "Required argument \(argument) for \(tool.rawValue) must not be empty."
    case let .invalidArgumentType(tool, argument, expected, actual):
      return "Invalid type for \(tool.rawValue).\(argument): expected \(expected.rawValue), got \(actual.rawValue)."
    case let .invalidEnumValue(tool, argument, allowed):
      return "Invalid value for \(tool.rawValue).\(argument); allowed: \(allowed.joined(separator: ", "))."
    case let .extraArguments(tool, arguments):
      return "Unexpected arguments for \(tool.rawValue): \(arguments.joined(separator: ", "))."
    case let .conflictingAlias(tool, canonicalArgument, alias):
      return "Conflicting values for \(tool.rawValue).\(canonicalArgument) and alias \(alias)."
    case let .invalidArgumentCombination(tool, reason):
      return "Invalid arguments for \(tool.rawValue): \(reason)"
    }
  }
}

public struct AgentValidatedToolCall: Sendable, Equatable {
  public let toolID: AgentToolID
  public let arguments: AgentJSONArguments
  public let definition: AgentToolDefinition

  public init(
    toolID: AgentToolID,
    arguments: AgentJSONArguments,
    definition: AgentToolDefinition
  ) {
    self.toolID = toolID
    self.arguments = arguments
    self.definition = definition
  }

  public func duplicateKey() throws -> String {
    let encodedArguments = try AgentJSONValue.object(arguments).canonicalJSONString()
    return "\(toolID.rawValue):\(encodedArguments)"
  }
}

public enum AgentToolValidator {
  public static func validate(
    rawToolID: String,
    arguments rawArguments: AgentJSONArguments,
    availableToolIDs: Set<AgentToolID>
  ) -> Result<AgentValidatedToolCall, AgentToolValidationError> {
    let toolID = AgentToolNormalizer.canonicalToolID(rawToolID)
    guard let definition = AgentToolCatalog.definition(for: toolID.rawValue) else {
      return .failure(.unknownTool(rawToolID))
    }
    let canonicalAvailable = Set(availableToolIDs.map {
      AgentToolNormalizer.canonicalToolID($0.rawValue)
    })
    guard canonicalAvailable.contains(toolID) else {
      return .failure(.unavailableTool(toolID))
    }

    let declaredNames = Set(definition.arguments.map(\.name))
    let acceptedNames = declaredNames.union(
      AgentToolNormalizer.acceptedAliasNames(for: toolID)
    )
    let extra = Set(rawArguments.keys).subtracting(acceptedNames)
    guard extra.isEmpty else {
      return .failure(.extraArguments(tool: toolID, arguments: extra.sorted()))
    }

    let arguments: AgentJSONArguments
    do {
      arguments = try AgentToolNormalizer.normalizedArguments(
        for: toolID,
        arguments: rawArguments
      )
    } catch let error as AgentToolValidationError {
      return .failure(error)
    } catch {
      return .failure(.unknownTool(rawToolID))
    }

    var validatedArguments = arguments
    for schema in definition.arguments {
      guard let value = validatedArguments[schema.name] else {
        if schema.required {
          return .failure(.missingRequiredArgument(tool: toolID, argument: schema.name))
        }
        continue
      }
      guard matches(value, schema.type) else {
        return .failure(.invalidArgumentType(
          tool: toolID,
          argument: schema.name,
          expected: schema.type,
          actual: AgentJSONKind(value)
        ))
      }
      if schema.required, case let .string(string) = value,
         string.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return .failure(.emptyRequiredArgument(tool: toolID, argument: schema.name))
      }
      if schema.type == .enumeration,
         let allowed = schema.allowedValues,
         let enumValue = value.stringValue {
        guard let canonicalValue = allowed.first(where: {
          $0.caseInsensitiveCompare(enumValue) == .orderedSame
        }) else {
          return .failure(.invalidEnumValue(
            tool: toolID,
            argument: schema.name,
            allowed: allowed.sorted()
          ))
        }
        validatedArguments[schema.name] = .string(canonicalValue)
      }
    }

    if toolID == "alarm.schedule", validatedArguments["snoozeMinutes"] == nil {
      // Lumen's fixed-alarm contract includes a five-minute post-alert
      // countdown. Canonicalizing the default here keeps approval binding and
      // the host invocation on the exact same payload.
      validatedArguments["snoozeMinutes"] = 5
    }

    let normalizedExtra = Set(validatedArguments.keys).subtracting(declaredNames)
    guard normalizedExtra.isEmpty else {
      return .failure(.extraArguments(tool: toolID, arguments: normalizedExtra.sorted()))
    }
    if let relationshipError = validateArgumentRelationships(
      toolID: toolID,
      arguments: validatedArguments
    ) {
      return .failure(relationshipError)
    }
    return .success(.init(
      toolID: toolID,
      arguments: validatedArguments,
      definition: definition
    ))
  }

  private static func matches(
    _ value: AgentJSONValue,
    _ expected: AgentToolArgumentType
  ) -> Bool {
    switch (value, expected) {
    case (.string, .string), (.string, .enumeration), (.number, .number),
         (.bool, .boolean), (.array, .array), (.object, .object):
      return true
    default:
      return false
    }
  }

  private static func validateArgumentRelationships(
    toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) -> AgentToolValidationError? {
    func invalid(_ reason: String) -> AgentToolValidationError {
      .invalidArgumentCombination(tool: toolID, reason: reason)
    }

    switch toolID.rawValue {
    case "outlook.message.move":
      guard let destination = arguments["destination"]?.stringValue,
        !destination.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return invalid("provide one non-empty destination or destinationId.")
      }

    case "outlook.message.reply", "outlook.message.reply_all":
      guard let body = arguments["body"]?.stringValue,
        !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return invalid("provide one non-empty body or comment.")
      }

    case "trigger.create":
      guard let schedule = arguments["schedule"]?.stringValue else {
        return invalid("schedule is required.")
      }
      let scheduleFields: Set<String> = [
        "inMinutes", "atTime", "intervalSeconds", "beforeMinutes",
      ]
      let supplied = Set(arguments.keys).intersection(scheduleFields)
      switch schedule {
      case "relative":
        guard supplied == ["inMinutes"],
          isInteger(arguments["inMinutes"], in: 1...(366 * 24 * 60)) else {
          return invalid("relative requires only integer inMinutes from 1 through 527040.")
        }
      case "absolute":
        guard supplied == ["atTime"],
          isStrictTimeOfDay(arguments["atTime"]?.stringValue) else {
          return invalid("absolute requires only atTime in strict HH:mm format.")
        }
      case "interval":
        guard supplied == ["intervalSeconds"],
          isInteger(arguments["intervalSeconds"], in: 60...(31 * 86_400)) else {
          return invalid("interval requires only integer intervalSeconds from 60 through 2678400.")
        }
      case "before_next_event":
        guard supplied.isSubset(of: ["beforeMinutes"]),
          arguments["beforeMinutes"] == nil
            || isInteger(arguments["beforeMinutes"], in: 1...(24 * 60)) else {
          return invalid("before_next_event accepts only optional integer beforeMinutes from 1 through 1440.")
        }
      default:
        return invalid("schedule is unsupported.")
      }

    case "trigger.cancel":
      let id = (arguments["id"]?.stringValue ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
      let title = (arguments["title"]?.stringValue ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
      guard id.isEmpty != title.isEmpty else {
        return invalid("provide exactly one non-empty id or title.")
      }
      if !id.isEmpty, UUID(uuidString: id) == nil {
        return invalid("id must be a UUID.")
      }

    case "alarm.schedule":
      if arguments["repeats"]?.boolValue == true {
        return invalid("repeats=true is unsupported; alarm.schedule creates one-shot alarms only.")
      }
      let supplied = ["inMinutes", "timestamp"].filter { arguments[$0] != nil }
      guard supplied.count == 1 else {
        return invalid("provide exactly one of inMinutes or timestamp.")
      }
      if arguments["inMinutes"] != nil,
        !isInteger(arguments["inMinutes"], in: 1...(366 * 24 * 60)) {
        return invalid("inMinutes must be an integer from 1 through 527040.")
      }
      if let timestamp = arguments["timestamp"]?.stringValue {
        let normalized = timestamp.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty,
          let value = TimeInterval(normalized), value.isFinite else {
          return invalid("timestamp must contain finite Unix seconds.")
        }
      }
      if arguments["snoozeMinutes"] != nil,
        !isInteger(arguments["snoozeMinutes"], in: 1...(24 * 60)) {
        return invalid("snoozeMinutes must be an integer from 1 through 1440.")
      }

    case "alarm.countdown":
      guard isInteger(
        arguments["durationSeconds"],
        in: 1...(366 * 24 * 60 * 60)
      ) else {
        return invalid("durationSeconds must be a positive bounded integer.")
      }

    default:
      break
    }
    return nil
  }

  private static func isInteger(
    _ value: AgentJSONValue?,
    in range: ClosedRange<Int>
  ) -> Bool {
    guard let number = value?.numberValue,
      number.isFinite,
      number.rounded(.towardZero) == number else { return false }
    return Double(range.lowerBound) <= number && number <= Double(range.upperBound)
  }

  private static func isStrictTimeOfDay(_ value: String?) -> Bool {
    guard let value else { return false }
    let components = value.split(separator: ":", omittingEmptySubsequences: false)
    guard components.count == 2,
      components[0].count == 2,
      components[1].count == 2,
      let hour = Int(components[0]),
      let minute = Int(components[1]) else { return false }
    return (0...23).contains(hour) && (0...59).contains(minute)
  }
}
