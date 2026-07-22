import Foundation
import MonGARSAgentTools
import XCTest

final class TriggerHandoffRegistryTests: XCTestCase {
#if os(iOS)
  func testOversizedLegacyOneShotIsNotAcknowledgedOrConsumed() async throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("MonGARSAgentToolsTests.\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let stateURL = root.appendingPathComponent("triggers.json")
    let id = UUID()
    let now = Date(timeIntervalSince1970: 10_000)
    let formatter = ISO8601DateFormatter()
    let legacyTrigger: [String: Any] = [
      "id": id.uuidString,
      "scope": "profile.test",
      "title": "Legacy oversized trigger",
      "prompt": String(repeating: "a", count: 513),
      "schedule": "relative",
      "createdAt": formatter.string(from: now.addingTimeInterval(-60)),
      "nextFireAt": formatter.string(from: now),
      "repeats": false,
    ]
    let encoded = try JSONSerialization.data(withJSONObject: [legacyTrigger])
    try encoded.write(to: stateURL)

    let suite = "MonGARSAgentToolsTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    defaults.set(id.uuidString, forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.set(now, forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
    let scheduler = LocalNotificationAgentTriggerScheduler(
      stateURL: stateURL,
      defaults: defaults,
      now: { now }
    )

    let acknowledged = await scheduler.acknowledgePendingHandoff(
      id: id,
      scope: "profile.test"
    )

    XCTAssertFalse(acknowledged)
    XCTAssertEqual(try Data(contentsOf: stateURL), encoded)
    XCTAssertEqual(
      defaults.string(forKey: AgentTriggerHandoffDefaultsKeys.identifier),
      id.uuidString
    )
  }
#endif

  func testExpectedIDConsumesOnceAndMismatchDoesNotConsume() async throws {
    let suite = "MonGARSAgentToolsTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let now = Date(timeIntervalSince1970: 1_000)
    let expected = UUID()
    defaults.set(expected.uuidString, forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.set(now, forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
    let registry = AgentTriggerHandoffRegistry(defaults: defaults, now: { now })

    let mismatch = await registry.consume(expectedID: UUID())
    let stillPending = await registry.pendingID()
    let firstConsume = await registry.consume(expectedID: expected)
    let afterConsume = await registry.pendingID()
    let replay = await registry.consume(expectedID: expected)
    XCTAssertFalse(mismatch)
    XCTAssertEqual(stillPending, expected)
    XCTAssertTrue(firstConsume)
    XCTAssertNil(afterConsume)
    XCTAssertFalse(replay)
  }

  func testExpiredOrFutureHandoffFailsClosed() async throws {
    let suite = "MonGARSAgentToolsTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let now = Date(timeIntervalSince1970: 10_000)
    let registry = AgentTriggerHandoffRegistry(defaults: defaults, timeToLive: 600, now: { now })

    defaults.set(UUID().uuidString, forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.set(now.addingTimeInterval(-601), forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
    let expired = await registry.pendingID()
    XCTAssertNil(expired)

    defaults.set(UUID().uuidString, forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.set(now.addingTimeInterval(61), forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
    let future = await registry.pendingID()
    XCTAssertNil(future)
  }

  func testMalformedIdentifierIsCleared() async throws {
    let suite = "MonGARSAgentToolsTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    defaults.set("not-a-uuid", forKey: AgentTriggerHandoffDefaultsKeys.identifier)
    defaults.set(Date(), forKey: AgentTriggerHandoffDefaultsKeys.receivedAt)
    let registry = AgentTriggerHandoffRegistry(defaults: defaults)

    let pending = await registry.pendingID()
    XCTAssertNil(pending)
    XCTAssertNil(defaults.object(forKey: AgentTriggerHandoffDefaultsKeys.identifier))
    XCTAssertNil(defaults.object(forKey: AgentTriggerHandoffDefaultsKeys.receivedAt))
  }
}
