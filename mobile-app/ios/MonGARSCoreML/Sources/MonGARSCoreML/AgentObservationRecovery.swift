import Foundation

struct AgentDegradedToolFailure: Sendable, Equatable {
  let toolID: AgentToolID
  let status: AgentToolResultStatus
  let errorCode: String?
  let text: String

  var runFailure: AgentRunFailure {
    .toolExecutionFailed(tool: toolID, status: status, errorCode: errorCode)
  }
}

enum AgentObservationRecoveryPolicy {
  /// Explicit rather than risk-derived: some local writes intentionally do not
  /// require approval, so `requiresApproval == false` is not a read-only proof.
  static let readOnlyToolIDs: Set<AgentToolID> = [
    "alarm.authorization_status", "alarm.list", "calendar.list",
    "contacts.search", "files.read", "health.summary", "location.current",
    "memory.recall", "motion.activity", "outlook.attachments.list",
    "outlook.folders.list", "outlook.message.read", "outlook.messages.list",
    "outlook.messages.search", "outlook.status", "photos.search",
    "rag.search", "reminders.list", "trigger.list", "weather", "web.fetch",
    "web.search", "maps.search",
  ]

  static func isMutating(_ definition: AgentToolDefinition) -> Bool {
    let localMutationIDs: Set<AgentToolID> = [
      "memory.save", "rag.index_files", "rag.index_photos",
    ]
    return definition.requiresApproval || localMutationIDs.contains(definition.id)
  }

  static func canRecover(
    failure: AgentDegradedToolFailure,
    definition: AgentToolDefinition,
    route: AgentIntentRoute,
    prompt: String,
    availableToolIDs: Set<AgentToolID>,
    stepIndex: Int,
    maxSteps: Int,
    alreadyRecovering: Bool
  ) -> Bool {
    guard !alreadyRecovering,
          stepIndex < maxSteps - 1,
          [.failed, .unavailable].contains(failure.status),
          readOnlyToolIDs.contains(definition.id) else { return false }
    let available = Set(availableToolIDs.map {
      AgentToolNormalizer.canonicalToolID($0.rawValue)
    })
    return !allowedAlternateToolIDs(
      after: failure,
      route: route,
      prompt: prompt
    ).intersection(available).isEmpty
  }

  static func isSafeAlternate(
    _ call: AgentValidatedToolCall,
    after failure: AgentDegradedToolFailure,
    route: AgentIntentRoute,
    prompt: String
  ) -> Bool {
    guard call.toolID != failure.toolID,
          readOnlyToolIDs.contains(call.toolID),
          allowedAlternateToolIDs(
            after: failure,
            route: route,
            prompt: prompt
          ).contains(call.toolID) else { return false }
    if call.toolID == "web.fetch" {
      guard let url = call.arguments["url"]?.stringValue else { return false }
      return explicitWebURLs(in: prompt).contains(url)
    }
    return true
  }

  /// Recovery pairs are semantic contracts, not merely tools sharing a broad
  /// intent. Outlook and alarm reads intentionally have no automatic
  /// substitutes because status/folder/list results cannot satisfy one another.
  private static func allowedAlternateToolIDs(
    after failure: AgentDegradedToolFailure,
    route: AgentIntentRoute,
    prompt: String
  ) -> Set<AgentToolID> {
    switch (failure.toolID.rawValue, route.intent) {
    case ("location.current", .weather):
      return ["weather"]
    case ("location.current", .maps) where isNearbyMapPrompt(prompt):
      return ["maps.search"]
    case ("web.search", .webSearch) where !explicitWebURLs(in: prompt).isEmpty:
      return ["web.fetch"]
    default:
      return []
    }
  }

  private static func isNearbyMapPrompt(_ prompt: String) -> Bool {
    let normalized = prompt.lowercased()
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
    guard !["directions", "navigate", "route to"].contains(where: {
      normalized.contains($0)
    }) else { return false }
    return ["near me", "nearby", "nearest", "closest"].contains(where: {
      normalized.contains($0)
    })
  }

  private static func explicitWebURLs(in prompt: String) -> Set<String> {
    guard prompt.utf8.count <= AgentPromptComposer.maximumToolUserInputBytes,
          let regex = try? NSRegularExpression(
            pattern: #"https?://[^\s<>{}\[\]\"']+"#,
            options: [.caseInsensitive]
          ) else { return [] }
    let ns = prompt as NSString
    return Set(regex.matches(
      in: prompt,
      range: NSRange(location: 0, length: ns.length)
    ).compactMap { match in
      guard match.range.location != NSNotFound else { return nil }
      return ns.substring(with: match.range)
        .trimmingCharacters(in: CharacterSet(charactersIn: ".,;:!?"))
    })
  }

  static func recoveredMessage(
    from failure: AgentDegradedToolFailure,
    alternateToolID: AgentToolID,
    alternateObservation: String,
    maximumCharacters: Int
  ) -> String {
    let code = failure.errorCode.map { " (\($0))" } ?? ""
    return AgentOutputSanitizer.sanitizeFinal(
      "The original \(failure.toolID.rawValue) attempt ended with \(failure.status.rawValue)\(code). "
        + "The alternate \(alternateToolID.rawValue) request succeeded. Result: \(alternateObservation)",
      maximumCharacters: maximumCharacters
    )
  }
}
