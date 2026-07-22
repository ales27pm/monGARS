import Foundation

public enum AgentReferenceResolutionKind: String, Sendable, Equatable {
  case none
  case clarificationAnswer
  case explicitHistoryEntity
}

/// A bounded rewrite of the current user message using only text that the user
/// supplied in the current message or in recent conversation history.
public struct AgentReferenceResolution: Sendable, Equatable {
  public let originalInput: String
  public let rewrittenInput: String
  public let kind: AgentReferenceResolutionKind

  public init(
    originalInput: String,
    rewrittenInput: String,
    kind: AgentReferenceResolutionKind
  ) {
    self.originalInput = originalInput
    self.rewrittenInput = rewrittenInput
    self.kind = kind
  }

  public var didRewrite: Bool { originalInput != rewrittenInput }
}

/// Resolves only high-confidence conversational references. It deliberately
/// does not infer entities from assistant prose or tool observations.
public enum AgentReferenceResolver {
  public static let maximumHistoryMessages = 6
  public static let maximumHistoryMessageBytes = 1_024
  public static let maximumInputBytes = AgentPromptComposer.maximumToolUserInputBytes

  public static func resolve(
    userInput: String,
    history: [AgentConversationMessage]
  ) -> AgentReferenceResolution {
    let original = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !original.isEmpty, original.utf8.count <= maximumInputBytes else {
      return unchanged(userInput)
    }

    let recent = Array(history.suffix(maximumHistoryMessages))
    if let rewritten = clarificationRewrite(answer: original, history: recent) {
      return .init(
        originalInput: userInput,
        rewrittenInput: rewritten,
        kind: .clarificationAnswer
      )
    }

    if containsPersonPronoun(original),
       let candidate = uniqueExplicitPersonCandidate(in: recent),
       let rewritten = replacingPersonPronoun(in: original, with: candidate),
       rewritten.utf8.count <= maximumInputBytes {
      return .init(
        originalInput: userInput,
        rewrittenInput: rewritten,
        kind: .explicitHistoryEntity
      )
    }

    return unchanged(userInput)
  }

  private static func clarificationRewrite(
    answer: String,
    history: [AgentConversationMessage]
  ) -> String? {
    guard let latest = history.last,
          latest.role == .assistant,
          latest.content.utf8.count <= maximumHistoryMessageBytes else {
      return nil
    }
    let assistantIndex = history.index(before: history.endIndex)
    let clarification = normalized(history[assistantIndex].content)
    guard let previousUser = history[..<assistantIndex].last(where: {
      $0.role == .user && $0.content.utf8.count <= maximumHistoryMessageBytes
    }) else { return nil }

    // Treat assistant prose as a control signal only when it exactly matches
    // the deterministic clarification that routing the preceding user message
    // would emit. A model response that merely contains one of these phrases
    // cannot turn the next user message into a tool request.
    let previousRoute = AgentIntentRouter.route(previousUser.content)
    guard previousRoute.requiresClarification,
          let expectedClarification = previousRoute.clarification,
          normalized(expectedClarification) == clarification else {
      return nil
    }

    if clarification == "who should i call?"
      || clarification == "which contact or phone number should i call?" {
      guard let target = safeRecipient(answer) else { return nil }
      return bounded("Call \(target)")
    }
    if clarification == "who should i message?" {
      guard let target = safeRecipient(answer) else { return nil }
      return bounded("\(previousUser.content) to \(target)")
    }
    if clarification == "who should i send it to?" {
      guard let target = safeRecipient(answer) else { return nil }
      return bounded("\(previousUser.content) to \(target)")
    }
    if clarification == "which contact should i look up?" {
      guard let target = safePerson(answer) else { return nil }
      return bounded("Find contact \(target)")
    }
    if clarification == "which file should i read?" {
      guard let file = safeFileName(answer) else { return nil }
      return bounded("Read file \(file)")
    }
    if clarification == "what place or destination should i look for?" {
      guard let destination = safeFreeText(answer, maximumBytes: 160) else { return nil }
      return bounded("Directions to \(destination)")
    }
    if clarification == "what should i search for?"
      || clarification == "what should i search for in outlook?" {
      guard let query = safeFreeText(answer, maximumBytes: 240) else { return nil }
      switch previousRoute.intent {
      case .outlook:
        return bounded("Search Outlook for \(query)")
      case .rag:
        return bounded("Search personal data for \(query)")
      case .webSearch:
        return bounded("Search web for \(query)")
      default:
        return nil
      }
    }
    if clarification == "what should i save or recall?" {
      guard let fact = safeFreeText(answer, maximumBytes: 320) else { return nil }
      let previous = normalized(previousUser.content)
      if previous.contains("remember") || previous.contains("save")
        || previous.contains("note") {
        return bounded("Remember \(fact)")
      }
      if previous.contains("recall") || previous.contains("what do you remember") {
        return bounded("What do you remember about \(fact)")
      }
    }

    guard let detail = safeFreeText(answer, maximumBytes: 320) else { return nil }
    switch clarification {
    case "what should the email say?", "what should the message say?":
      return bounded("\(previousUser.content) saying \(detail)")
    case "who should i send it to, and what should it say?",
      "who should i message, and what should it say?":
      // The answer must itself contain the missing recipient/content markers;
      // otherwise the router will safely ask again.
      return bounded("\(previousUser.content) \(detail)")
    case "what should the calendar event be?":
      return bounded("Create event called \(detail)")
    case "what should i remind you about?":
      return bounded("Remind me to \(detail)")
    case "which photos should i look for?":
      return bounded("Search photos for \(detail)")
    case "what time should i use for the alarm?":
      return bounded("\(previousUser.content) at \(detail)")
    case "what duration should i use for the timer?":
      return bounded("\(previousUser.content) for \(detail)")
    case "which alarm should i cancel?", "which alarm should i pause?",
      "which alarm should i resume?", "which alarm should i stop?",
      "which alarm should i snooze?":
      return bounded("\(previousUser.content) \(detail)")
    case "what should the scheduled agent run do?":
      return bounded("\(previousUser.content) to \(detail)")
    case "how many months of photos should i index?":
      return bounded("\(previousUser.content) for \(detail)")
    case "what would you like to do in outlook?":
      return bounded("Outlook \(detail)")
    case "which outlook message should i read?":
      return bounded("Read Outlook message \(detail)")
    case "which outlook message should i inspect for attachments?":
      return bounded("List Outlook attachments for message \(detail)")
    case "do you mean a calendar event or a nearby meeting location?":
      let choice = normalized(detail)
      if choice == "calendar event" { return "Show my calendar" }
      if choice == "nearby meeting location" { return "Find a meeting location nearby" }
      return nil
    default:
      return nil
    }
  }

  private static func uniqueExplicitPersonCandidate(
    in history: [AgentConversationMessage]
  ) -> String? {
    var candidates: [String: String] = [:]
    let patterns = [
      #"(?i)\b(?:find|search for|look up)\s+(?:the\s+)?contact\s+(?:named\s+)?([\p{L}\p{M}][\p{L}\p{M}'’.-]*(?:\s+[\p{L}\p{M}][\p{L}\p{M}'’.-]*){0,3})(?=\s*[?.!,]?\s*$)"#,
      #"(?i)\b(?:call|dial|phone|message|text|email)\s+([\p{L}\p{M}][\p{L}\p{M}'’.-]*(?:\s+[\p{L}\p{M}][\p{L}\p{M}'’.-]*){0,3})(?=\s+(?:saying|about|that)\b|\s*[?.!,]?\s*$)"#,
    ]

    for message in history where message.role == .user {
      guard message.content.utf8.count <= maximumHistoryMessageBytes else { continue }
      for pattern in patterns {
        guard let candidate = firstCapture(in: message.content, pattern: pattern),
              let safe = safePerson(candidate) else { continue }
        candidates[normalized(safe)] = safe
      }
    }
    guard candidates.count == 1 else { return nil }
    return candidates.values.first
  }

  private static func replacingPersonPronoun(
    in input: String,
    with candidate: String
  ) -> String? {
    let pattern = #"(?i)\b(call|dial|phone|message|text|email)\s+(her|him|them)\b"#
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
    let ns = input as NSString
    guard let match = regex.firstMatch(
      in: input,
      range: NSRange(location: 0, length: ns.length)
    ), let verbRange = Range(match.range(at: 1), in: input),
      let fullRange = Range(match.range(at: 0), in: input) else { return nil }
    let verb = String(input[verbRange])
    return input.replacingCharacters(in: fullRange, with: "\(verb) \(candidate)")
  }

  private static func containsPersonPronoun(_ input: String) -> Bool {
    input.range(
      of: #"(?i)\b(?:call|dial|phone|message|text|email)\s+(?:her|him|them)\b"#,
      options: .regularExpression
    ) != nil
  }

  private static func safeRecipient(_ raw: String) -> String? {
    if let person = safePerson(raw) { return person }
    let value = trimmedTerminalPunctuation(raw)
    let isEmail = value.range(
      of: #"^[^\s@]+@[^\s@]+\.[^\s@]+$"#,
      options: .regularExpression
    ) != nil
    let isPhone = value.range(
      of: #"^\+?[0-9][0-9 ()-]{2,30}$"#,
      options: .regularExpression
    ) != nil
    guard value.utf8.count <= 160,
          isEmail || isPhone else { return nil }
    return value
  }

  private static func safePerson(_ raw: String) -> String? {
    let value = trimmedTerminalPunctuation(raw)
    let forbidden = ["her", "him", "them", "it", "that", "this", "someone", "anyone"]
    guard value.utf8.count <= 120,
          !forbidden.contains(normalized(value)),
          value.range(
            of: #"^[\p{L}\p{M}][\p{L}\p{M}'’.-]*(?:\s+[\p{L}\p{M}][\p{L}\p{M}'’.-]*){0,3}$"#,
            options: .regularExpression
          ) != nil else { return nil }
    return value
  }

  private static func safeFileName(_ raw: String) -> String? {
    let value = trimmedTerminalPunctuation(raw)
    guard value.utf8.count <= 160,
          value != ".", value != "..",
          !value.contains("/"), !value.contains("\\"), !value.contains(":"),
          value.range(of: #"^[\p{L}\p{M}0-9 _().-]+$"#, options: .regularExpression) != nil
    else { return nil }
    return value
  }

  private static func safeFreeText(_ raw: String, maximumBytes: Int) -> String? {
    let value = trimmedTerminalPunctuation(raw)
    guard !value.isEmpty, value.utf8.count <= maximumBytes,
          !value.contains("\n"), !value.contains("\r"),
          !["it", "that", "this", "them", "anything", "something"]
            .contains(normalized(value)) else { return nil }
    return value
  }

  private static func bounded(_ value: String) -> String? {
    value.utf8.count <= maximumInputBytes ? value : nil
  }

  private static func trimmedTerminalPunctuation(_ raw: String) -> String {
    raw.trimmingCharacters(
      in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "\"'?!,"))
    )
  }

  private static func firstCapture(in text: String, pattern: String) -> String? {
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
    let ns = text as NSString
    guard let match = regex.firstMatch(
      in: text,
      range: NSRange(location: 0, length: ns.length)
    ), match.numberOfRanges > 1, match.range(at: 1).location != NSNotFound else { return nil }
    return ns.substring(with: match.range(at: 1))
  }

  private static func normalized(_ text: String) -> String {
    text.lowercased()
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
  }

  private static func unchanged(_ input: String) -> AgentReferenceResolution {
    .init(originalInput: input, rewrittenInput: input, kind: .none)
  }
}
