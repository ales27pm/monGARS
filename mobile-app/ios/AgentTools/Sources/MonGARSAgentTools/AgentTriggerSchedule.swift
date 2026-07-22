import Foundation

enum AgentTriggerScheduleCalculator {
  static func timeOfDayMinutes(_ raw: String) -> Int? {
    let components = raw.split(separator: ":", omittingEmptySubsequences: false)
    guard components.count == 2,
      components[0].count == 2,
      components[1].count == 2,
      let hour = Int(components[0]),
      let minute = Int(components[1]),
      (0...23).contains(hour),
      (0...59).contains(minute) else { return nil }
    return hour * 60 + minute
  }

  static func nextDaily(
    after now: Date,
    timeOfDayMinutes: Int,
    calendar: Calendar
  ) -> Date? {
    guard (0..<(24 * 60)).contains(timeOfDayMinutes) else { return nil }
    var components = calendar.dateComponents([.year, .month, .day], from: now)
    components.hour = timeOfDayMinutes / 60
    components.minute = timeOfDayMinutes % 60
    components.second = 0
    guard var candidate = calendar.date(from: components) else { return nil }
    if candidate <= now {
      candidate = calendar.date(byAdding: .day, value: 1, to: candidate)
        ?? candidate.addingTimeInterval(86_400)
    }
    return candidate
  }

  static func nextInterval(
    after now: Date,
    previousFireAt: Date?,
    intervalSeconds: TimeInterval
  ) -> Date? {
    guard intervalSeconds >= 60, intervalSeconds.isFinite else { return nil }
    guard let previousFireAt else { return now.addingTimeInterval(intervalSeconds) }
    guard previousFireAt <= now else { return previousFireAt }
    let elapsed = max(0, now.timeIntervalSince(previousFireAt))
    let intervals = floor(elapsed / intervalSeconds) + 1
    return previousFireAt.addingTimeInterval(intervals * intervalSeconds)
  }

  static func fireDate(
    before eventStart: Date,
    minutes: Int,
    now: Date
  ) -> Date? {
    guard (1...(24 * 60)).contains(minutes) else { return nil }
    let fireDate = eventStart.addingTimeInterval(-TimeInterval(minutes * 60))
    return fireDate > now ? fireDate : nil
  }

  static func nextEventStart(in eventStarts: [Date], after earliestStart: Date) -> Date? {
    eventStarts.lazy.filter { $0 >= earliestStart }.min()
  }
}

struct AgentTriggerCancellationCandidate: Equatable {
  let id: UUID
  let title: String
}

enum AgentTriggerCancellationResolution: Equatable {
  case match(UUID)
  case invalidSelector
  case invalidUUID
  case notFound
  case ambiguousTitle
}

enum AgentTriggerCancellationResolver {
  static func resolve(
    id: String?,
    title: String?,
    candidates: [AgentTriggerCancellationCandidate]
  ) -> AgentTriggerCancellationResolution {
    let normalizedID = id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let normalizedTitle = title?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard normalizedID.isEmpty != normalizedTitle.isEmpty else { return .invalidSelector }

    if !normalizedID.isEmpty {
      guard let uuid = UUID(uuidString: normalizedID) else { return .invalidUUID }
      return candidates.contains(where: { $0.id == uuid }) ? .match(uuid) : .notFound
    }

    let exact = candidates.filter { $0.title == normalizedTitle }
    if exact.count == 1, let candidate = exact.first { return .match(candidate.id) }
    if exact.count > 1 { return .ambiguousTitle }
    let caseInsensitive = candidates.filter {
      $0.title.caseInsensitiveCompare(normalizedTitle) == .orderedSame
    }
    if caseInsensitive.count == 1, let candidate = caseInsensitive.first {
      return .match(candidate.id)
    }
    return caseInsensitive.isEmpty ? .notFound : .ambiguousTitle
  }
}
