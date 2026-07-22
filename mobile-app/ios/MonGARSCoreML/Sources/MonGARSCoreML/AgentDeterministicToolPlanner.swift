import Foundation

/// Produces only explicit multi-action plans. Single actions remain model
/// generated, while every planned action is still validated and authorized by
/// `AgentExecutor` before execution.
public enum AgentDeterministicToolPlanner {
  public static let maximumPlannedActions = 2

  public static func plan(
    route: AgentIntentRoute,
    prompt: String,
    availableToolIDs: Set<AgentToolID>
  ) -> [AgentToolAction] {
    guard !route.requiresClarification else { return [] }
    let available = Set(availableToolIDs.map {
      AgentToolNormalizer.canonicalToolID($0.rawValue)
    })

    let candidate: [AgentToolAction]
    switch route.intent {
    case .memory, .note:
      candidate = memorySaveThenRecall(prompt: prompt, available: available)
    case .maps:
      candidate = nearbySearch(prompt: prompt, available: available)
    case .weather:
      candidate = currentLocationWeather(prompt: prompt, available: available)
    case .calendar:
      candidate = calendarReadThenCreate(prompt: prompt, available: available)
    default:
      candidate = []
    }

    guard candidate.count == maximumPlannedActions else { return [] }
    return candidate
  }

  private static func memorySaveThenRecall(
    prompt: String,
    available: Set<AgentToolID>
  ) -> [AgentToolAction] {
    guard available.isSuperset(of: ["memory.save", "memory.recall"]),
          let captures = captures(
            in: prompt,
            pattern: #"(?is)^\s*(?:please\s+)?(?:remember|save(?:\s+this)?(?:\s+fact)?|note)\s+(?:that\s+)?(.+?)\s+(?:,?\s*(?:and\s+)?then)\s+(?:recall|search\s+(?:my\s+)?memor(?:y|ies)(?:\s+for)?|tell\s+me\s+what\s+you\s+remember\s+about)\s+(.+?)\s*[.!?]?\s*$"#,
            count: 2
          ),
          let content = boundedArgument(captures[0], maximumBytes: 320),
          let query = boundedArgument(captures[1], maximumBytes: 240)
    else { return [] }

    return [
      .init(tool: "memory.save", arguments: [
        "content": .string(content),
        "kind": .string("fact"),
      ]),
      .init(tool: "memory.recall", arguments: ["query": .string(query)]),
    ]
  }

  private static func nearbySearch(
    prompt: String,
    available: Set<AgentToolID>
  ) -> [AgentToolAction] {
    guard available.isSuperset(of: ["location.current", "maps.search"]) else {
      return []
    }
    let patterns = [
      #"(?is)^\s*(?:please\s+)?(?:find|search(?:\s+for)?|show\s+me|locate)\s+(.+?)\s+(?:near\s+me|nearby)\s*[.!?]?\s*$"#,
      #"(?is)^\s*(?:please\s+)?(?:find|show\s+me|search(?:\s+for)?)?\s*(?:the\s+)?(?:nearest|closest|nearby)\s+(.+?)\s*[.!?]?\s*$"#,
    ]
    let rawQuery = patterns.lazy.compactMap {
      captures(in: prompt, pattern: $0, count: 1)?.first
    }.first
    guard let rawQuery,
          let query = boundedArgument(rawQuery, maximumBytes: 160),
          !isUnresolvedReference(query) else { return [] }
    return [
      .init(tool: "location.current", arguments: [:]),
      .init(tool: "maps.search", arguments: ["query": .string(query)]),
    ]
  }

  private static func currentLocationWeather(
    prompt: String,
    available: Set<AgentToolID>
  ) -> [AgentToolAction] {
    guard available.isSuperset(of: ["location.current", "weather"]) else {
      return []
    }
    let text = normalized(prompt)
    guard text.contains("weather") || text.contains("forecast")
      || text.contains("temperature") else { return [] }
    guard [
      "weather here", "forecast here", "temperature here", "where i am",
      "current location", "my location",
    ].contains(where: { text.contains($0) }) else { return [] }
    return [
      .init(tool: "location.current", arguments: [:]),
      .init(tool: "weather", arguments: [:]),
    ]
  }

  private static func calendarReadThenCreate(
    prompt: String,
    available: Set<AgentToolID>
  ) -> [AgentToolAction] {
    guard available.isSuperset(of: ["calendar.list", "calendar.create"]),
          let values = captures(
            in: prompt,
            pattern: #"(?is)^\s*(?:please\s+)?(?:list|show)\s+(?:my\s+)?(?:calendar|events?)\s*,?\s*(?:and\s+)?then\s+(?:create|add|schedule)\s+(?:an?\s+)?event\s+(?:called|titled|named)\s+(.+?)\s+in\s+(\d{1,6})\s+minutes?\s*[.!?]?\s*$"#,
            count: 2
          ),
          let title = boundedArgument(values[0], maximumBytes: 160),
          let minutes = Int(values[1]), (1...(366 * 24 * 60)).contains(minutes)
    else { return [] }
    return [
      .init(tool: "calendar.list", arguments: [:]),
      .init(tool: "calendar.create", arguments: [
        "title": .string(title),
        "startsInMinutes": .number(Double(minutes)),
      ]),
    ]
  }

  private static func captures(
    in text: String,
    pattern: String,
    count: Int
  ) -> [String]? {
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
    let ns = text as NSString
    guard let match = regex.firstMatch(
      in: text,
      range: NSRange(location: 0, length: ns.length)
    ), match.numberOfRanges == count + 1 else { return nil }
    var values: [String] = []
    for index in 1...count {
      let range = match.range(at: index)
      guard range.location != NSNotFound else { return nil }
      values.append(ns.substring(with: range))
    }
    return values
  }

  private static func boundedArgument(
    _ raw: String,
    maximumBytes: Int
  ) -> String? {
    let value = raw.trimmingCharacters(
      in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "\"'.,!?"))
    )
    guard !value.isEmpty, value.utf8.count <= maximumBytes,
          !value.contains("\n"), !value.contains("\r") else { return nil }
    return value
  }

  private static func isUnresolvedReference(_ value: String) -> Bool {
    ["it", "that", "this", "them", "one", "something", "anything"]
      .contains(normalized(value))
  }

  private static func normalized(_ value: String) -> String {
    value.lowercased()
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
  }
}
