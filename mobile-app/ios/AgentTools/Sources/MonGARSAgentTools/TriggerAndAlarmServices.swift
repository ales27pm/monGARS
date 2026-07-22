import Foundation
import MonGARSAlarmSupport
import MonGARSCoreML

#if os(iOS)
import EventKit
import UserNotifications
#if canImport(AlarmKit)
import AlarmKit
import SwiftUI
#endif
#endif

private struct StoredAgentTrigger: Codable, Sendable, Equatable {
  let id: UUID
  let scope: String
  let title: String
  let prompt: String
  let schedule: String
  let createdAt: Date
  var nextFireAt: Date?
  let repeats: Bool
  let intervalSeconds: TimeInterval?
  let timeOfDayMinutes: Int?
  let beforeMinutes: Int?
}

public enum AgentTriggerPromptContract {
  public static let maximumUTF8Bytes = AgentPromptComposer.maximumToolUserInputBytes

  public static func normalized(_ prompt: String) -> String? {
    let value = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty, value.utf8.count <= maximumUTF8Bytes else { return nil }
    return value
  }
}

public enum AgentTriggerHandoffDefaultsKeys {
  public static let identifier = "MonGARS.PendingAgentTriggerHandoffID"
  public static let receivedAt = "MonGARS.PendingAgentTriggerHandoffDate"
}

/// One-time, expiring registry for the opaque ID written by AppDelegate after
/// a notification tap. Prompt content is never stored in UserDefaults.
public actor AgentTriggerHandoffRegistry {
  private let defaults: UserDefaults
  private let now: @Sendable () -> Date
  private let timeToLive: TimeInterval

  public init(
    defaults: UserDefaults = .standard,
    timeToLive: TimeInterval = 10 * 60,
    now: @escaping @Sendable () -> Date = { Date() }
  ) {
    self.defaults = defaults
    self.timeToLive = min(max(timeToLive, 30), 60 * 60)
    self.now = now
  }

  public func pendingID() -> UUID? {
    guard let rawID = defaults.string(forKey: AgentTriggerHandoffDefaultsKeys.identifier),
      let id = UUID(uuidString: rawID),
      let receivedAt = defaults.object(forKey: AgentTriggerHandoffDefaultsKeys.receivedAt) as? Date else {
      clear()
      return nil
    }
    let age = now().timeIntervalSince(receivedAt)
    guard age >= -60, age <= timeToLive else {
      clear()
      return nil
    }
    return id
  }

  public func consume(expectedID: UUID) -> Bool {
    guard pendingID() == expectedID else { return false }
    clear()
    return true
  }

  public func clear() {
    defaults.removeObject(forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.removeObject(forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
  }
}

#if os(iOS)
public actor LocalNotificationAgentTriggerScheduler: AgentTriggerScheduling {
  private let stateURL: URL
  private let center: UNUserNotificationCenter
  private let handoffRegistry: AgentTriggerHandoffRegistry
  private let eventStore: EKEventStore
  private let now: @Sendable () -> Date
  private var cachedTriggers: [StoredAgentTrigger]?

  public init(
    stateURL: URL,
    center: UNUserNotificationCenter = .current(),
    defaults: UserDefaults = .standard,
    eventStore: EKEventStore = EKEventStore(),
    now: @escaping @Sendable () -> Date = { Date() }
  ) {
    self.stateURL = stateURL
    self.center = center
    self.handoffRegistry = .init(defaults: defaults, now: now)
    self.eventStore = eventStore
    self.now = now
  }

  public func create(arguments: AgentJSONArguments, scope: String) async -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope),
      let title = AgentToolInput.requiredString("title", in: arguments, maximumBytes: 500),
      let prompt = AgentToolInput.requiredString(
        "prompt",
        in: arguments,
        maximumBytes: AgentTriggerPromptContract.maximumUTF8Bytes
      ),
      let schedule = AgentToolInput.requiredString("schedule", in: arguments, maximumBytes: 32) else {
      return .failed("The trigger arguments are invalid.", code: "trigger_invalid_arguments")
    }
    let settings = await center.notificationSettings()
    guard [.authorized, .provisional, .ephemeral].contains(settings.authorizationStatus) else {
      return .denied("Notification permission is required for scheduled handoffs.", code: "trigger_notifications_denied")
    }

    let referenceDate = now()
    let parsed: ParsedSchedule
    switch schedule.lowercased() {
    case "absolute", "daily":
      guard let raw = AgentToolInput.requiredString("atTime", in: arguments, maximumBytes: 5),
        let minutes = AgentTriggerScheduleCalculator.timeOfDayMinutes(raw),
        let date = AgentTriggerScheduleCalculator.nextDaily(
          after: referenceDate,
          timeOfDayMinutes: minutes,
          calendar: .autoupdatingCurrent
        ) else {
        return .failed("Daily triggers require atTime in HH:mm format.", code: "trigger_invalid_daily_time")
      }
      let components = DateComponents(hour: minutes / 60, minute: minutes % 60)
      parsed = .init(
        notification: UNCalendarNotificationTrigger(dateMatching: components, repeats: true),
        nextFireAt: date,
        repeats: true,
        schedule: "daily",
        intervalSeconds: nil,
        timeOfDayMinutes: minutes,
        beforeMinutes: nil
      )
    case "relative":
      guard let minutes = AgentToolInput.integer("inMinutes", in: arguments, range: 1...(366 * 24 * 60)) else {
        return .failed("The relative trigger delay is invalid.", code: "trigger_invalid_delay")
      }
      let seconds = TimeInterval(minutes * 60)
      parsed = .init(
        notification: UNTimeIntervalNotificationTrigger(timeInterval: seconds, repeats: false),
        nextFireAt: referenceDate.addingTimeInterval(seconds),
        repeats: false,
        schedule: "relative",
        intervalSeconds: nil,
        timeOfDayMinutes: nil,
        beforeMinutes: nil
      )
    case "interval":
      guard let seconds = AgentToolInput.integer("intervalSeconds", in: arguments, range: 60...(31 * 86_400)) else {
        return .failed("The repeating trigger interval is invalid.", code: "trigger_invalid_interval")
      }
      let interval = TimeInterval(seconds)
      parsed = .init(
        notification: UNTimeIntervalNotificationTrigger(timeInterval: interval, repeats: true),
        nextFireAt: referenceDate.addingTimeInterval(interval),
        repeats: true,
        schedule: "interval",
        intervalSeconds: interval,
        timeOfDayMinutes: nil,
        beforeMinutes: nil
      )
    case "beforenextevent", "before_next_event", "before-next-event":
      let beforeMinutes: Int
      if arguments["beforeMinutes"] == nil {
        beforeMinutes = 15
      } else if let value = AgentToolInput.integer(
        "beforeMinutes",
        in: arguments,
        range: 1...(24 * 60)
      ) {
        beforeMinutes = value
      } else {
        return .failed("The before-event lead time is invalid.", code: "trigger_invalid_before_minutes")
      }
      switch nextEventStart(
        after: referenceDate.addingTimeInterval(TimeInterval(beforeMinutes * 60)),
        through: referenceDate.addingTimeInterval(366 * 86_400)
      ) {
      case let .found(eventStart):
        guard let fireDate = AgentTriggerScheduleCalculator.fireDate(
          before: eventStart,
          minutes: beforeMinutes,
          now: referenceDate
        ) else {
          return .failed("The next event is too close to schedule this lead time.", code: "trigger_event_too_close")
        }
        let components = Calendar.autoupdatingCurrent.dateComponents(
          [.year, .month, .day, .hour, .minute, .second],
          from: fireDate
        )
        parsed = .init(
          notification: UNCalendarNotificationTrigger(dateMatching: components, repeats: false),
          nextFireAt: fireDate,
          repeats: true,
          schedule: "before_next_event",
          intervalSeconds: nil,
          timeOfDayMinutes: nil,
          beforeMinutes: beforeMinutes
        )
      case .permissionDenied:
        return .denied(
          "Full calendar access is required for a before-next-event trigger.",
          code: "trigger_calendar_permission_denied"
        )
      case .notFound:
        return .failed("No upcoming calendar event was found.", code: "trigger_no_upcoming_event")
      }
    default:
      return .failed("The trigger schedule type is invalid.", code: "trigger_invalid_schedule")
    }

    var triggers = loadTriggers()
    guard triggers.filter({ $0.scope == scope }).count < 128 else {
      return .failed("The active profile has reached its trigger limit.", code: "trigger_capacity_reached")
    }
    let trigger = StoredAgentTrigger(
      id: UUID(),
      scope: scope,
      title: title,
      prompt: prompt,
      schedule: parsed.schedule,
      createdAt: referenceDate,
      nextFireAt: parsed.nextFireAt,
      repeats: parsed.repeats,
      intervalSeconds: parsed.intervalSeconds,
      timeOfDayMinutes: parsed.timeOfDayMinutes,
      beforeMinutes: parsed.beforeMinutes
    )
    let content = Self.notificationContent(for: trigger)
    // The prompt stays in the protected local store; notification metadata
    // carries only an opaque identifier for the foreground handoff.
    do {
      try await center.add(.init(
        identifier: Self.notificationIdentifier(trigger.id),
        content: content,
        trigger: parsed.notification
      ))
      triggers.append(trigger)
      guard persist(triggers) else {
        center.removePendingNotificationRequests(withIdentifiers: [Self.notificationIdentifier(trigger.id)])
        return .failed("The trigger could not be stored securely.", code: "trigger_persist_failed")
      }
      return .success(
        "Scheduled a notification handoff for \(Self.iso8601.string(from: parsed.nextFireAt)). The agent runs after the user opens MonGARS; iOS does not guarantee unattended model execution.",
        payload: [
          "id": .string(trigger.id.uuidString),
          "nextFireAt": .string(Self.iso8601.string(from: parsed.nextFireAt)),
          "repeats": .bool(trigger.repeats),
          "executionMode": .string("notification_foreground_handoff"),
        ]
      )
    } catch {
      return .failed("The notification trigger could not be scheduled.", code: "trigger_schedule_failed")
    }
  }

  public func list(scope: String) async -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active trigger scope is invalid.", code: "trigger_scope_invalid")
    }
    var triggers = loadTriggers()
    let referenceDate = now()
    for index in triggers.indices where triggers[index].repeats {
      if let nextFireAt = triggers[index].nextFireAt,
        nextFireAt > referenceDate { continue }
      var trigger = triggers[index]
      _ = await advanceRepeatingTrigger(&trigger, after: referenceDate)
      triggers[index] = trigger
    }
    // Retain fired one-shot prompts long enough for a later notification tap.
    // The prompt never enters UserDefaults or the notification payload.
    triggers.removeAll {
      !$0.repeats
        && ($0.nextFireAt ?? .distantPast) < referenceDate.addingTimeInterval(-30 * 86_400)
    }
    _ = persist(triggers)
    let scoped = triggers.filter { $0.scope == scope }.sorted {
      ($0.nextFireAt ?? .distantFuture) < ($1.nextFireAt ?? .distantFuture)
    }
    let values: [AgentJSONValue] = scoped.map {
      var value: AgentJSONArguments = [
        "id": .string($0.id.uuidString),
        "title": .string($0.title),
        "schedule": .string($0.schedule),
        "repeats": .bool($0.repeats),
        "executionMode": .string("notification_foreground_handoff"),
      ]
      if let nextFireAt = $0.nextFireAt {
        value["nextFireAt"] = .string(Self.iso8601.string(from: nextFireAt))
      }
      if let interval = $0.intervalSeconds { value["intervalSeconds"] = .number(interval) }
      if let minutes = $0.timeOfDayMinutes { value["timeOfDayMinutes"] = .number(Double(minutes)) }
      if let minutes = $0.beforeMinutes { value["beforeMinutes"] = .number(Double(minutes)) }
      return .object(value)
    }
    let text = scoped.isEmpty
      ? "No scheduled triggers were found."
      : scoped.map {
        let next = $0.nextFireAt.map(Self.iso8601.string) ?? "waiting for a future event"
        return "- \($0.title) at \(next) [\($0.id.uuidString)]"
      }
        .joined(separator: "\n")
    return .success(text, payload: ["triggers": .array(values)])
  }

  public func cancel(id: String?, title: String?, scope: String) async -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active trigger scope is invalid.", code: "trigger_scope_invalid")
    }
    var triggers = loadTriggers()
    let candidates = triggers.filter { $0.scope == scope }.map {
      AgentTriggerCancellationCandidate(id: $0.id, title: $0.title)
    }
    let resolvedID: UUID
    switch AgentTriggerCancellationResolver.resolve(
      id: id,
      title: title,
      candidates: candidates
    ) {
    case let .match(id):
      resolvedID = id
    case .invalidSelector:
      return .failed("Provide exactly one trigger UUID or exact title.", code: "trigger_invalid_identifier")
    case .invalidUUID:
      return .failed("The trigger UUID is invalid.", code: "trigger_invalid_id")
    case .ambiguousTitle:
      return .failed(
        "Multiple triggers match that exact title. Use the UUID from trigger.list.",
        code: "trigger_ambiguous_title"
      )
    case .notFound:
      return .failed("The trigger was not found in the active profile.", code: "trigger_not_found")
    }
    guard let index = triggers.firstIndex(where: { $0.id == resolvedID && $0.scope == scope }) else {
      return .failed("The trigger was not found in the active profile.", code: "trigger_not_found")
    }
    let removed = triggers.remove(at: index)
    guard persist(triggers) else {
      return .failed("The trigger cancellation could not be persisted.", code: "trigger_persist_failed")
    }
    let notificationID = Self.notificationIdentifier(removed.id)
    center.removePendingNotificationRequests(withIdentifiers: [notificationID])
    center.removeDeliveredNotifications(withIdentifiers: [notificationID])
    return .success(
      "Cancelled the scheduled trigger \"\(removed.title)\".",
      payload: [
        "id": .string(removed.id.uuidString),
        "title": .string(removed.title),
      ]
    )
  }

  public func pendingHandoff(scope: String) async -> AgentPendingTriggerHandoff? {
    guard let scope = AgentToolInput.validatedScope(scope),
      let id = await handoffRegistry.pendingID() else { return nil }
    guard let trigger = loadTriggers().first(where: { $0.id == id && $0.scope == scope }),
      let prompt = AgentTriggerPromptContract.normalized(trigger.prompt)
    else { return nil }
    return .init(
      id: trigger.id,
      title: trigger.title,
      prompt: prompt,
      repeats: trigger.repeats
    )
  }

  public func resolveHandoff(
    selector: String,
    scope: String
  ) async -> AgentPendingTriggerHandoff? {
    guard let scope = AgentToolInput.validatedScope(scope) else { return nil }
    let value = selector.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty, value.utf8.count <= 512 else { return nil }
    let triggers = loadTriggers().filter { $0.scope == scope }
    let candidates = triggers.map {
      AgentTriggerCancellationCandidate(id: $0.id, title: $0.title)
    }
    let id = UUID(uuidString: value).map { _ in value }
    let title = id == nil ? value : nil
    guard case let .match(resolvedID) = AgentTriggerCancellationResolver.resolve(
      id: id,
      title: title,
      candidates: candidates
    ), let trigger = triggers.first(where: { $0.id == resolvedID }),
      let prompt = AgentTriggerPromptContract.normalized(trigger.prompt) else {
      return nil
    }
    return .init(
      id: trigger.id,
      title: trigger.title,
      prompt: prompt,
      repeats: trigger.repeats
    )
  }

  public func acknowledgePendingHandoff(id: UUID, scope: String) async -> Bool {
    guard let scope = AgentToolInput.validatedScope(scope),
      await handoffRegistry.pendingID() == id else { return false }
    let original = loadTriggers()
    guard let index = original.firstIndex(where: { $0.id == id && $0.scope == scope }) else {
      // A different signed-in profile cannot acknowledge this handoff.
      return false
    }
    let trigger = original[index]
    // Validate before any one-shot removal or repeating-trigger advancement.
    // Oversized prompts written by an older app remain stored and cancellable.
    guard AgentTriggerPromptContract.normalized(trigger.prompt) != nil else { return false }
    var scheduledReplacement = false
    if !trigger.repeats {
      var updated = original
      updated.remove(at: index)
      guard persist(updated) else { return false }
    } else {
      var updated = original
      var repeatingTrigger = updated[index]
      let advancement = await advanceRepeatingTrigger(&repeatingTrigger, after: now())
      updated[index] = repeatingTrigger
      scheduledReplacement = advancement.scheduledNotification
      guard persist(updated) else {
        if scheduledReplacement {
          center.removePendingNotificationRequests(withIdentifiers: [Self.notificationIdentifier(id)])
        }
        return false
      }
    }
    guard await handoffRegistry.consume(expectedID: id) else {
      _ = persist(original)
      if scheduledReplacement {
        center.removePendingNotificationRequests(withIdentifiers: [Self.notificationIdentifier(id)])
      }
      return false
    }
    return true
  }

  private func advanceRepeatingTrigger(
    _ trigger: inout StoredAgentTrigger,
    after referenceDate: Date
  ) async -> (changed: Bool, scheduledNotification: Bool) {
    switch trigger.schedule {
    case "daily":
      guard let minutes = trigger.timeOfDayMinutes,
        let next = AgentTriggerScheduleCalculator.nextDaily(
          after: referenceDate,
          timeOfDayMinutes: minutes,
          calendar: .autoupdatingCurrent
        ) else { return (false, false) }
      trigger.nextFireAt = next
      return (true, false)
    case "interval":
      guard let interval = trigger.intervalSeconds,
        let next = AgentTriggerScheduleCalculator.nextInterval(
          after: referenceDate,
          previousFireAt: trigger.nextFireAt,
          intervalSeconds: interval
        ) else { return (false, false) }
      trigger.nextFireAt = next
      return (true, false)
    case "before_next_event":
      guard let beforeMinutes = trigger.beforeMinutes else { return (false, false) }
      switch nextEventStart(
        after: referenceDate.addingTimeInterval(TimeInterval(beforeMinutes * 60) + 1),
        through: referenceDate.addingTimeInterval(366 * 86_400)
      ) {
      case let .found(eventStart):
        guard let fireDate = AgentTriggerScheduleCalculator.fireDate(
          before: eventStart,
          minutes: beforeMinutes,
          now: referenceDate
        ) else {
          trigger.nextFireAt = nil
          return (true, false)
        }
        let scheduled = await scheduleBeforeEventNotification(trigger: trigger, at: fireDate)
        guard scheduled else { return (false, false) }
        trigger.nextFireAt = fireDate
        return (true, true)
      case .permissionDenied, .notFound:
        trigger.nextFireAt = nil
        return (true, false)
      }
    default:
      return (false, false)
    }
  }

  private func scheduleBeforeEventNotification(
    trigger: StoredAgentTrigger,
    at fireDate: Date
  ) async -> Bool {
    let components = Calendar.autoupdatingCurrent.dateComponents(
      [.year, .month, .day, .hour, .minute, .second],
      from: fireDate
    )
    do {
      try await center.add(.init(
        identifier: Self.notificationIdentifier(trigger.id),
        content: Self.notificationContent(for: trigger),
        trigger: UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
      ))
      return true
    } catch {
      return false
    }
  }

  private func nextEventStart(after start: Date, through end: Date) -> EventLookup {
    guard EKEventStore.authorizationStatus(for: .event) == .fullAccess else {
      return .permissionDenied
    }
    let predicate = eventStore.predicateForEvents(
      withStart: start,
      end: end,
      calendars: nil
    )
    let eligibleStarts = eventStore.events(matching: predicate)
      .filter { $0.status != .canceled }
      .map(\.startDate)
    guard let eventStart = AgentTriggerScheduleCalculator.nextEventStart(
      in: eligibleStarts,
      after: start
    ) else { return .notFound }
    return .found(eventStart)
  }

  private enum EventLookup {
    case found(Date)
    case permissionDenied
    case notFound
  }

  private struct ParsedSchedule {
    let notification: UNNotificationTrigger
    let nextFireAt: Date
    let repeats: Bool
    let schedule: String
    let intervalSeconds: TimeInterval?
    let timeOfDayMinutes: Int?
    let beforeMinutes: Int?
  }

  private func loadTriggers() -> [StoredAgentTrigger] {
    if let cachedTriggers { return cachedTriggers }
    guard let data = try? Data(contentsOf: stateURL),
      let decoded = try? Self.decoder.decode([StoredAgentTrigger].self, from: data) else {
      cachedTriggers = []
      return []
    }
    cachedTriggers = decoded
    return decoded
  }

  private func persist(_ triggers: [StoredAgentTrigger]) -> Bool {
    let existing = loadTriggers()
    // New and changed records must be directly runnable. Preserve unchanged
    // legacy records so an upgrade never silently deletes an oversized prompt.
    guard triggers.allSatisfy({ trigger in
      if AgentTriggerPromptContract.normalized(trigger.prompt) == trigger.prompt {
        return true
      }
      return existing.contains { $0.id == trigger.id && $0.prompt == trigger.prompt }
    }) else { return false }
    do {
      try FileManager.default.createDirectory(
        at: stateURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )
      let data = try Self.encoder.encode(triggers)
      try data.write(to: stateURL, options: [.atomic, .completeFileProtection])
      try FileManager.default.setAttributes(
        [.protectionKey: FileProtectionType.complete],
        ofItemAtPath: stateURL.path
      )
      cachedTriggers = triggers
      return true
    } catch {
      return false
    }
  }

  private static func notificationIdentifier(_ id: UUID) -> String {
    "com.mongars.agent.trigger.\(id.uuidString)"
  }

  private static func notificationContent(for trigger: StoredAgentTrigger) -> UNNotificationContent {
    let content = UNMutableNotificationContent()
    content.title = trigger.title
    content.body = "Open MonGARS to run the scheduled request."
    content.sound = .default
    content.userInfo = ["monGARSAgentTriggerID": trigger.id.uuidString]
    return content
  }

  private static let iso8601 = ISO8601DateFormatter()
  private static let encoder: JSONEncoder = {
    let value = JSONEncoder()
    value.outputFormatting = [.sortedKeys]
    value.dateEncodingStrategy = .iso8601
    return value
  }()
  private static let decoder: JSONDecoder = {
    let value = JSONDecoder()
    value.dateDecodingStrategy = .iso8601
    return value
  }()
}
#else
public actor LocalNotificationAgentTriggerScheduler: AgentTriggerScheduling {
  public init(stateURL: URL) {}
  public func create(arguments: AgentJSONArguments, scope: String) async -> AgentServiceResponse {
    .unavailable("Scheduled triggers require iOS notifications.", code: "trigger_unavailable")
  }
  public func list(scope: String) async -> AgentServiceResponse {
    .unavailable("Scheduled triggers require iOS notifications.", code: "trigger_unavailable")
  }
  public func cancel(id: String?, title: String?, scope: String) async -> AgentServiceResponse {
    .unavailable("Scheduled triggers require iOS notifications.", code: "trigger_unavailable")
  }
  public func resolveHandoff(
    selector: String,
    scope: String
  ) async -> AgentPendingTriggerHandoff? { nil }
  public func pendingHandoff(scope: String) async -> AgentPendingTriggerHandoff? { nil }
  public func acknowledgePendingHandoff(id: UUID, scope: String) async -> Bool { false }
}
#endif

public struct IOSAlarmService: AgentAlarmServing, Sendable {
  public init() {}

  @MainActor
  public func execute(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments
  ) async -> AgentServiceResponse {
#if os(iOS) && canImport(AlarmKit)
    if #available(iOS 26.0, *) {
      guard Self.hasUsageDescription else {
        return .unavailable("AlarmKit is unavailable because NSAlarmKitUsageDescription is missing.", code: "alarm_usage_description_missing")
      }
      if Self.requiresLiveActivity(operation: operation, arguments: arguments),
        !Self.hasLiveActivityConfiguration {
        return .unavailable(
          "This AlarmKit operation requires the embedded MonGARS alarm Live Activity.",
          code: "alarm_live_activity_missing"
        )
      }
      switch operation {
      case .alarmAuthorizationStatus:
        return .success(
          "Alarm authorization status: \(String(describing: AlarmManager.shared.authorizationState)).",
          payload: ["status": .string(String(describing: AlarmManager.shared.authorizationState))]
        )
      case .alarmRequestAuthorization:
        do {
          let state = try await AlarmManager.shared.requestAuthorization()
          return .success(
            "Alarm authorization result: \(String(describing: state)).",
            payload: ["status": .string(String(describing: state))]
          )
        } catch {
          return .denied("Alarm authorization was not granted.", code: "alarm_authorization_failed")
        }
      case .alarmSchedule:
        return await schedule(arguments)
      case .alarmCountdown:
        return await countdown(arguments)
      case .alarmList:
        do {
          let alarms = try AlarmManager.shared.alarms
          let values: [AgentJSONValue] = alarms.prefix(100).map {
            [
              "id": .string($0.id.uuidString),
              "state": .string(String(describing: $0.state)),
            ]
          }
          let text = alarms.isEmpty
            ? "No active alarms were found."
            : alarms.map { "- \($0.id.uuidString): \(String(describing: $0.state))" }.joined(separator: "\n")
          return .success(text, payload: ["alarms": .array(values)])
        } catch {
          return .failed("Active alarms could not be read.", code: "alarm_read_failed")
        }
      case .alarmPause: return mutate(arguments, action: "pause") { try AlarmManager.shared.pause(id: $0) }
      case .alarmResume: return mutate(arguments, action: "resume") { try AlarmManager.shared.resume(id: $0) }
      case .alarmStop: return mutate(arguments, action: "stop") { try AlarmManager.shared.stop(id: $0) }
      case .alarmSnooze: return mutate(arguments, action: "snooze") { try AlarmManager.shared.countdown(id: $0) }
      case .alarmCancel: return mutate(arguments, action: "cancel") { try AlarmManager.shared.cancel(id: $0) }
      default: return .failed("Unsupported AlarmKit operation.", code: "alarm_unsupported_operation")
      }
    }
#endif
    return .unavailable("AlarmKit requires an iOS 26 or newer AlarmKit-capable runtime.", code: "alarmkit_unavailable")
  }

#if os(iOS) && canImport(AlarmKit)
  @MainActor
  @available(iOS 26.0, *)
  private func schedule(_ arguments: AgentJSONArguments) async -> AgentServiceResponse {
    guard let title = AgentToolInput.requiredString("title", in: arguments, maximumBytes: 500) else {
      return .failed("The alarm title is invalid.", code: "alarm_invalid_title")
    }
    if arguments["repeats"] != nil {
      guard let repeats = AgentToolInput.bool("repeats", in: arguments) else {
        return .failed("The repeats flag must be boolean.", code: "alarm_invalid_repeats")
      }
      guard !repeats else {
        return .failed("Repeating alarms are not supported by this tool path.", code: "alarm_repeats_unsupported")
      }
    }
    let snoozeMinutes: Int
    if arguments["snoozeMinutes"] != nil {
      guard let validated = AgentToolInput.integer(
        "snoozeMinutes",
        in: arguments,
        range: 1...(24 * 60)
      ) else {
        return .failed("The snooze duration is invalid.", code: "alarm_invalid_snooze")
      }
      snoozeMinutes = validated
    } else {
      snoozeMinutes = 5
    }

    let hasRelativeTime = arguments["inMinutes"] != nil
    let hasTimestamp = arguments["timestamp"] != nil
    guard hasRelativeTime != hasTimestamp else {
      return .failed(
        "Provide exactly one of inMinutes or a Unix-seconds timestamp for the alarm.",
        code: "alarm_invalid_schedule"
      )
    }

    let fireDate: Date
    if hasRelativeTime {
      guard let inMinutes = AgentToolInput.integer(
        "inMinutes",
        in: arguments,
        range: 1...(366 * 24 * 60)
      ) else {
        return .failed("The relative alarm time is invalid.", code: "alarm_invalid_schedule")
      }
      fireDate = Date().addingTimeInterval(TimeInterval(inMinutes * 60))
    } else if let rawTimestamp = AgentToolInput.optionalString(
        "timestamp",
        in: arguments,
        maximumBytes: 64
      ), let timestamp = TimeInterval(rawTimestamp), timestamp.isFinite {
      fireDate = Date(timeIntervalSince1970: timestamp)
    } else {
      return .failed("The alarm timestamp is invalid.", code: "alarm_invalid_schedule")
    }
    guard fireDate > Date() else {
      return .failed("The alarm time must be in the future.", code: "alarm_time_in_past")
    }

    do {
      let id = UUID()
      let configuration = AlarmManager.AlarmConfiguration<MonGARSAlarmMetadata>(
        countdownDuration: Alarm.CountdownDuration(
          preAlert: nil,
          postAlert: TimeInterval(snoozeMinutes * 60)
        ),
        schedule: .fixed(fireDate),
        attributes: alarmAttributes(title: title)
      )
      let alarm = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
      return .success(
        "Alarm scheduled for \(fireDate.formatted(date: .abbreviated, time: .shortened)).",
        payload: [
          "id": .string(alarm.id.uuidString),
          "title": .string(title),
          "fireDate": .string(Self.iso8601.string(from: fireDate)),
          "state": .string(String(describing: alarm.state)),
        ]
      )
    } catch {
      return .failed("The alarm could not be scheduled.", code: "alarm_schedule_failed")
    }
  }

  @MainActor
  @available(iOS 26.0, *)
  private func countdown(_ arguments: AgentJSONArguments) async -> AgentServiceResponse {
    guard let title = AgentToolInput.requiredString("title", in: arguments, maximumBytes: 500),
      let durationSeconds = AgentToolInput.integer(
        "durationSeconds",
        in: arguments,
        range: 1...(366 * 24 * 60 * 60)
      ) else {
      return .failed("The countdown arguments are invalid.", code: "alarm_invalid_countdown")
    }
    do {
      let id = UUID()
      let configuration = AlarmManager.AlarmConfiguration<MonGARSAlarmMetadata>.timer(
        duration: TimeInterval(durationSeconds),
        attributes: alarmAttributes(title: title)
      )
      let alarm = try await AlarmManager.shared.schedule(id: id, configuration: configuration)
      return .success(
        "Alarm countdown scheduled for \(durationSeconds) seconds.",
        payload: [
          "id": .string(alarm.id.uuidString),
          "title": .string(title),
          "durationSeconds": .number(Double(durationSeconds)),
          "state": .string(String(describing: alarm.state)),
        ]
      )
    } catch {
      return .failed("The alarm countdown could not be scheduled.", code: "alarm_countdown_failed")
    }
  }

  @MainActor
  @available(iOS 26.0, *)
  private func mutate(
    _ arguments: AgentJSONArguments,
    action: String,
    operation: (UUID) throws -> Void
  ) -> AgentServiceResponse {
    guard let rawID = AgentToolInput.requiredString("id", in: arguments, maximumBytes: 64),
      let id = UUID(uuidString: rawID) else {
      return .failed("The alarm identifier is invalid.", code: "alarm_invalid_id")
    }
    do {
      try operation(id)
      return .success(
        "Alarm \(action) completed.",
        payload: ["id": .string(id.uuidString), "action": .string(action)]
      )
    } catch {
      return .failed("The alarm \(action) operation failed.", code: "alarm_\(action)_failed")
    }
  }

  @MainActor
  @available(iOS 26.0, *)
  private func alarmAttributes(title: String) -> AlarmAttributes<MonGARSAlarmMetadata> {
    AlarmAttributes(
      presentation: alarmPresentation(title: title),
      metadata: MonGARSAlarmMetadata(title: title),
      tintColor: .orange
    )
  }

  @MainActor
  @available(iOS 26.0, *)
  private func alarmPresentation(title: String) -> AlarmPresentation {
    let displayTitle = title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      ? "Alarm"
      : title
    let localizedTitle = LocalizedStringResource(stringLiteral: displayTitle)
    let pauseButton = AlarmButton(
      text: "Pause",
      textColor: .orange,
      systemImageName: "pause.fill"
    )
    let resumeButton = AlarmButton(
      text: "Resume",
      textColor: .orange,
      systemImageName: "play.fill"
    )
    if #available(iOS 26.1, *) {
      let snoozeButton = AlarmButton(
        text: "Snooze",
        textColor: .orange,
        systemImageName: "zzz"
      )
      return AlarmPresentation(
        alert: .init(
          title: localizedTitle,
          secondaryButton: snoozeButton,
          secondaryButtonBehavior: .countdown
        ),
        countdown: .init(title: localizedTitle, pauseButton: pauseButton),
        paused: .init(title: localizedTitle, resumeButton: resumeButton)
      )
    }
    let stopButton = AlarmButton(
      text: "Stop",
      textColor: .orange,
      systemImageName: "stop.fill"
    )
    return AlarmPresentation(
      alert: .init(title: localizedTitle, stopButton: stopButton),
      countdown: .init(title: localizedTitle, pauseButton: pauseButton),
      paused: .init(title: localizedTitle, resumeButton: resumeButton)
    )
  }

  static var hasUsageDescription: Bool {
    guard let value = Bundle.main.object(forInfoDictionaryKey: "NSAlarmKitUsageDescription") as? String else {
      return false
    }
    return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  static var hasLiveActivityConfiguration: Bool {
    guard Bundle.main.object(forInfoDictionaryKey: "NSSupportsLiveActivities") as? Bool == true,
      let plugInsURL = Bundle.main.builtInPlugInsURL,
      let plugInURLs = try? FileManager.default.contentsOfDirectory(
        at: plugInsURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
      ) else { return false }
    return plugInURLs.contains { url in
      guard url.lastPathComponent == "MonGARSAlarmWidget.appex",
        let bundle = Bundle(url: url),
        bundle.object(forInfoDictionaryKey: "NSSupportsLiveActivities") as? Bool == true,
        let extensionInfo = bundle.object(forInfoDictionaryKey: "NSExtension") as? [String: Any]
      else { return false }
      return extensionInfo["NSExtensionPointIdentifier"] as? String
        == "com.apple.widgetkit-extension"
    }
  }

  private static func requiresLiveActivity(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments
  ) -> Bool {
    switch operation {
    case .alarmCountdown, .alarmPause, .alarmResume, .alarmSnooze:
      return true
    case .alarmSchedule:
      return true
    default:
      return false
    }
  }

  private static let iso8601 = ISO8601DateFormatter()
#endif

}
