import Foundation
import MonGARSAgentTools

#if canImport(AppIntents)
import AppIntents

private enum MonGARSIntentHandoffSubmission {
  static func enqueue(
    kind: MonGARSAppIntentHandoffKind,
    input: String? = nil
  ) async -> String {
    guard let store = MonGARSAppIntentHandoffStore.shared else {
      return "monGARS cannot open this request because its protected handoff store is unavailable."
    }
    do {
      let record = try await store.enqueue(kind: kind, input: input)
      NotificationCenter.default.post(
        name: Notification.Name("MonGARSAppIntentHandoffAvailable"),
        object: nil,
        userInfo: [
          "id": record.id.uuidString.lowercased(),
          "createdAt": record.createdAt,
        ]
      )
      return confirmation(for: kind)
    } catch MonGARSAppIntentHandoffStoreError.invalidInput {
      return invalidInputMessage(for: kind)
    } catch {
      return "monGARS could not store this request securely. Nothing was executed."
    }
  }

  private static func confirmation(for kind: MonGARSAppIntentHandoffKind) -> String {
    switch kind {
    case .ask:
      return "Question ready in monGARS. Confirm it in the foreground to run it."
    case .memorySearch:
      return "Local memory search ready in monGARS. Confirm it to view private results."
    case .memoryAdd:
      return "Memory text ready in monGARS. Confirm it before it is saved."
    case .runTrigger:
      return "Stored trigger ready in monGARS. Confirm it before the agent runs."
    case .diagnostics:
      return "monGARS diagnostics are ready to open. No capture was started."
    case .masked:
      return "The protected request could not be opened. Nothing was executed."
    }
  }

  private static func invalidInputMessage(for kind: MonGARSAppIntentHandoffKind) -> String {
    switch kind {
    case .ask:
      return "The question must be non-empty and no larger than 512 UTF-8 bytes. Nothing was executed."
    case .memorySearch:
      return "The search must be non-empty and no larger than 192 UTF-8 bytes. Nothing was executed."
    case .memoryAdd:
      return "The memory text must be non-empty and no larger than 186 UTF-8 bytes. Nothing was executed."
    case .runTrigger:
      return "The trigger name or UUID must be non-empty and no larger than 512 bytes."
    case .diagnostics:
      return "The diagnostics request was invalid. Nothing was started."
    case .masked:
      return "The protected request was invalid. Nothing was executed."
    }
  }
}

@available(iOS 18.0, *)
struct MonGARSAskIntent: AppIntent {
  static let title: LocalizedStringResource = "Ask monGARS"
  static let description = IntentDescription(
    "Open monGARS with a question for explicit foreground confirmation."
  )
  static let openAppWhenRun = true

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  static var supportedModes: IntentModes { [.foreground(.immediate)] }
#endif

  @Parameter(title: "Question")
  var question: String

  func perform() async -> some IntentResult & ReturnsValue<String> {
    .result(value: await MonGARSIntentHandoffSubmission.enqueue(kind: .ask, input: question))
  }
}

@available(iOS 18.0, *)
struct MonGARSSearchMemoryIntent: AppIntent {
  static let title: LocalizedStringResource = "Search monGARS Memory"
  static let description = IntentDescription(
    "Open monGARS before searching private local memories."
  )
  static let openAppWhenRun = true

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  static var supportedModes: IntentModes { [.foreground(.immediate)] }
#endif

  @Parameter(title: "Search")
  var query: String

  func perform() async -> some IntentResult & ReturnsValue<String> {
    .result(
      value: await MonGARSIntentHandoffSubmission.enqueue(kind: .memorySearch, input: query)
    )
  }
}

@available(iOS 18.0, *)
struct MonGARSAddMemoryIntent: AppIntent {
  static let title: LocalizedStringResource = "Add monGARS Memory"
  static let description = IntentDescription(
    "Open monGARS to review text before saving it to local memory."
  )
  static let openAppWhenRun = true

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  static var supportedModes: IntentModes { [.foreground(.immediate)] }
#endif

  @Parameter(title: "Memory Text")
  var text: String

  func perform() async -> some IntentResult & ReturnsValue<String> {
    .result(
      value: await MonGARSIntentHandoffSubmission.enqueue(kind: .memoryAdd, input: text)
    )
  }
}

@available(iOS 18.0, *)
struct MonGARSRunTriggerIntent: AppIntent {
  static let title: LocalizedStringResource = "Run monGARS Trigger"
  static let description = IntentDescription(
    "Open monGARS to resolve and confirm a stored trigger for the active profile."
  )
  static let openAppWhenRun = true

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  static var supportedModes: IntentModes { [.foreground(.immediate)] }
#endif

  @Parameter(title: "Trigger Name or UUID")
  var trigger: String

  func perform() async -> some IntentResult & ReturnsValue<String> {
    .result(
      value: await MonGARSIntentHandoffSubmission.enqueue(kind: .runTrigger, input: trigger)
    )
  }
}

@available(iOS 18.0, *)
struct MonGARSDiagnosticsIntent: AppIntent {
  static let title: LocalizedStringResource = "Open monGARS Diagnostics"
  static let description = IntentDescription(
    "Open passive diagnostics without starting a capture or diagnostic campaign."
  )
  static let openAppWhenRun = true

#if compiler(>=6.2)
  @available(iOS 26.0, *)
  static var supportedModes: IntentModes { [.foreground(.immediate)] }
#endif

  func perform() async -> some IntentResult & ReturnsValue<String> {
    .result(value: await MonGARSIntentHandoffSubmission.enqueue(kind: .diagnostics))
  }
}
#endif
