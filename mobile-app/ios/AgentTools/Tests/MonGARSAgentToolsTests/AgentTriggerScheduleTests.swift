@testable import MonGARSAgentTools
import Foundation
import XCTest

final class AgentTriggerScheduleTests: XCTestCase {
  func testTriggerPromptContractAccepts512UTF8BytesAndRejects513() {
    let maximumASCII = String(repeating: "a", count: 512)
    let maximumMultibyte = String(repeating: "é", count: 256)

    XCTAssertEqual(AgentTriggerPromptContract.maximumUTF8Bytes, 512)
    XCTAssertEqual(AgentTriggerPromptContract.normalized(maximumASCII), maximumASCII)
    XCTAssertEqual(AgentTriggerPromptContract.normalized(maximumMultibyte), maximumMultibyte)
    XCTAssertNil(AgentTriggerPromptContract.normalized(maximumASCII + "a"))
    XCTAssertNil(AgentTriggerPromptContract.normalized(maximumMultibyte + "a"))
  }

  func testAbsoluteTimeIsStrictHHMMDailyTime() {
    XCTAssertEqual(AgentTriggerScheduleCalculator.timeOfDayMinutes("09:05"), 545)
    XCTAssertEqual(AgentTriggerScheduleCalculator.timeOfDayMinutes("23:59"), 1_439)
    XCTAssertNil(AgentTriggerScheduleCalculator.timeOfDayMinutes("9:05"))
    XCTAssertNil(AgentTriggerScheduleCalculator.timeOfDayMinutes("24:00"))
    XCTAssertNil(AgentTriggerScheduleCalculator.timeOfDayMinutes("2026-08-01T09:05:00Z"))
  }

  func testDailyScheduleAdvancesToTomorrowAfterTodaysTime() throws {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = try XCTUnwrap(TimeZone(secondsFromGMT: 0))
    let now = try XCTUnwrap(ISO8601DateFormatter().date(from: "2026-07-21T10:00:00Z"))
    let next = AgentTriggerScheduleCalculator.nextDaily(
      after: now,
      timeOfDayMinutes: 9 * 60,
      calendar: calendar
    )
    XCTAssertEqual(next, ISO8601DateFormatter().date(from: "2026-07-22T09:00:00Z"))
  }

  func testIntervalAdvancesPastEveryMissedOccurrence() throws {
    let formatter = ISO8601DateFormatter()
    let previous = try XCTUnwrap(formatter.date(from: "2026-07-21T09:00:00Z"))
    let now = try XCTUnwrap(formatter.date(from: "2026-07-21T12:30:00Z"))
    XCTAssertEqual(
      AgentTriggerScheduleCalculator.nextInterval(
        after: now,
        previousFireAt: previous,
        intervalSeconds: 3_600
      ),
      formatter.date(from: "2026-07-21T13:00:00Z")
    )
  }

  func testBeforeEventRequiresFutureBoundedLeadTime() throws {
    let formatter = ISO8601DateFormatter()
    let now = try XCTUnwrap(formatter.date(from: "2026-07-21T10:00:00Z"))
    let event = try XCTUnwrap(formatter.date(from: "2026-07-21T11:00:00Z"))
    XCTAssertEqual(
      AgentTriggerScheduleCalculator.fireDate(before: event, minutes: 15, now: now),
      formatter.date(from: "2026-07-21T10:45:00Z")
    )
    XCTAssertNil(AgentTriggerScheduleCalculator.fireDate(before: event, minutes: 90, now: now))
  }

  func testNextEventSkipsAnOverlappingCurrentEvent() throws {
    let formatter = ISO8601DateFormatter()
    let earliest = try XCTUnwrap(formatter.date(from: "2026-07-21T10:15:00Z"))
    let ongoingStart = try XCTUnwrap(formatter.date(from: "2026-07-21T09:30:00Z"))
    let futureStart = try XCTUnwrap(formatter.date(from: "2026-07-21T11:00:00Z"))

    XCTAssertEqual(
      AgentTriggerScheduleCalculator.nextEventStart(
        in: [ongoingStart, futureStart],
        after: earliest
      ),
      futureStart
    )
  }

  func testCancellationResolverPrefersCaseSensitiveExactTitleAndRejectsAmbiguity() throws {
    let first = try XCTUnwrap(UUID(uuidString: "11111111-2222-4333-8444-555555555555"))
    let second = try XCTUnwrap(UUID(uuidString: "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"))
    let candidates = [
      AgentTriggerCancellationCandidate(id: first, title: "Morning Summary"),
      AgentTriggerCancellationCandidate(id: second, title: "morning summary"),
    ]

    XCTAssertEqual(
      AgentTriggerCancellationResolver.resolve(
        id: nil,
        title: "Morning Summary",
        candidates: candidates
      ),
      .match(first)
    )
    XCTAssertEqual(
      AgentTriggerCancellationResolver.resolve(
        id: nil,
        title: "MORNING SUMMARY",
        candidates: candidates
      ),
      .ambiguousTitle
    )
    XCTAssertEqual(
      AgentTriggerCancellationResolver.resolve(
        id: first.uuidString,
        title: nil,
        candidates: candidates
      ),
      .match(first)
    )
    XCTAssertEqual(
      AgentTriggerCancellationResolver.resolve(
        id: first.uuidString,
        title: "Morning Summary",
        candidates: candidates
      ),
      .invalidSelector
    )
  }
}
