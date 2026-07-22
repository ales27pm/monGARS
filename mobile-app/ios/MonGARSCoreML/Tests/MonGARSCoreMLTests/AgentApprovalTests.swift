import Foundation
import XCTest
@testable import MonGARSCoreML

final class AgentApprovalTests: XCTestCase {
  func testApprovalIsPayloadBoundAndConsumedExactlyOnce() async {
    let store = AgentApprovalStore()
    let arguments: AgentJSONArguments = ["title": "Review", "startsInMinutes": 30]
    let requested = await store.requestApproval(toolID: "calendar.create", arguments: arguments)
    guard case let .success(record) = requested else {
      return XCTFail("Expected approval record")
    }
    guard case .success = await store.approve(id: record.id) else {
      return XCTFail("Expected pending approval to approve")
    }

    let mismatched = await store.consumeApproval(
      id: record.id,
      toolID: "calendar.create",
      arguments: ["title": "Different", "startsInMinutes": 30]
    )
    XCTAssertEqual(mismatched, .failure(.bindingMismatch))
    guard case let .success(consumed) = await store.consumeApproval(
      id: record.id,
      toolID: "calendar.create",
      arguments: arguments
    ) else {
      return XCTFail("Expected exact payload to consume")
    }
    XCTAssertEqual(consumed.status, .consumed)
    let secondConsume = await store.consumeApproval(
      id: record.id,
      toolID: "calendar.create",
      arguments: arguments
    )
    XCTAssertEqual(secondConsume, .failure(.alreadyConsumed))
  }

  func testPendingAndApprovedRecordsExpire() async {
    let clock = LockedTestClock(Date(timeIntervalSince1970: 1_000))
    let store = AgentApprovalStore(
      defaultTTL: 60,
      maximumTTL: 60,
      now: { clock.now() }
    )
    guard case let .success(pending) = await store.requestApproval(
      toolID: "phone.call",
      arguments: ["number": "+15551234567"]
    ) else { return XCTFail("Expected pending record") }
    clock.advance(by: 61)

    let lateApproval = await store.approve(id: pending.id)
    let expiredRecord = await store.record(id: pending.id)
    XCTAssertEqual(lateApproval, .failure(.expired))
    XCTAssertEqual(expiredRecord?.status, .expired)

    guard case let .success(second) = await store.requestApproval(
      toolID: "phone.call",
      arguments: ["number": "+15557654321"]
    ) else { return XCTFail("Expected second record") }
    guard case .success = await store.approve(id: second.id) else {
      return XCTFail("Expected approval")
    }
    clock.advance(by: 61)
    let expiredConsume = await store.consumeApproval(
      id: second.id,
      toolID: "phone.call",
      arguments: ["number": "+15557654321"]
    )
    XCTAssertEqual(expiredConsume, .failure(.expired))
  }

  func testRejectedApprovalCannotBeConsumed() async {
    let store = AgentApprovalStore()
    guard case let .success(record) = await store.requestApproval(
      toolID: "alarm.cancel",
      arguments: ["id": "alarm-id"]
    ) else { return XCTFail("Expected record") }
    guard case let .success(rejected) = await store.reject(id: record.id) else {
      return XCTFail("Expected rejection")
    }

    XCTAssertEqual(rejected.status, .rejected)
    let rejectedConsume = await store.consumeApproval(
      id: record.id,
      toolID: "alarm.cancel",
      arguments: ["id": "alarm-id"]
    )
    XCTAssertEqual(rejectedConsume, .failure(.rejected))
  }

  func testCapacityNeverSilentlyEvictsPendingRecords() async {
    let store = AgentApprovalStore(maximumRecords: 1)
    guard case let .success(first) = await store.requestApproval(
      toolID: "phone.call",
      arguments: ["number": "1"]
    ) else { return XCTFail("Expected first record") }

    let fullRequest = await store.requestApproval(
      toolID: "phone.call",
      arguments: ["number": "2"]
    )
    XCTAssertEqual(fullRequest, .failure(.capacityReached))
    guard case .success = await store.reject(id: first.id) else {
      return XCTFail("Expected rejection")
    }
    guard case .success = await store.requestApproval(
      toolID: "phone.call",
      arguments: ["number": "2"]
    ) else {
      return XCTFail("A terminal record should be safely evicted")
    }
  }

  func testPolicyRequestsForegroundPermissionBeforeApproval() {
    let policy = AgentApprovalPolicy()
    let calendar = requireDefinition("calendar.create")

    XCTAssertEqual(
      policy.evaluate(
        definition: calendar,
        arguments: ["title": "Review", "startsInMinutes": 15],
        permissionState: .notDetermined,
        mode: .foreground
      ),
      .permissionRequestRequired(.calendar)
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: calendar,
        arguments: ["title": "Review", "startsInMinutes": 15],
        permissionState: .granted,
        mode: .foreground
      ),
      .approvalRequired
    )
  }

  func testPolicyAllowsCityWeatherWithoutLocationPermission() {
    let policy = AgentApprovalPolicy()
    let weather = requireDefinition("weather")

    XCTAssertEqual(
      policy.evaluate(
        definition: weather,
        arguments: ["location": "Toronto"],
        permissionState: .denied,
        mode: .foreground
      ),
      .allowed
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: weather,
        arguments: ["location": "current location"],
        permissionState: .denied,
        mode: .foreground
      ),
      .denied(.permissionDenied(.location))
    )
  }

  func testBeforeNextEventRequiresCalendarAfterNotifications() {
    let policy = AgentApprovalPolicy()
    let trigger = requireDefinition("trigger.create")
    let beforeEvent: AgentJSONArguments = [
      "title": "Prepare",
      "prompt": "Summarize the next meeting",
      "schedule": "before_next_event",
      "beforeMinutes": 15,
    ]

    XCTAssertEqual(
      policy.additionalPermissions(definition: trigger, arguments: beforeEvent),
      [.calendar]
    )
    XCTAssertEqual(
      policy.additionalPermissions(
        definition: trigger,
        arguments: [
          "title": "Daily",
          "prompt": "Prepare my summary",
          "schedule": "absolute",
          "atTime": "08:00",
        ]
      ),
      []
    )
    XCTAssertEqual(
      policy.evaluate(
        permission: .calendar,
        permissionState: .notDetermined,
        mode: .foreground,
        acceptsLimited: false
      ),
      .permissionRequestRequired(.calendar)
    )
    XCTAssertEqual(
      policy.evaluate(
        permission: .calendar,
        permissionState: .limited,
        mode: .foreground,
        acceptsLimited: false
      ),
      .permissionRequestRequired(.calendar)
    )
  }

  func testReadToolsRequireFullEventKitAccessWhileCreatesAcceptWriteOnly() {
    let policy = AgentApprovalPolicy()
    XCTAssertEqual(
      policy.evaluate(
        definition: requireDefinition("calendar.list"),
        arguments: [:],
        permissionState: .limited,
        mode: .foreground
      ),
      .permissionRequestRequired(.calendar)
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: requireDefinition("reminders.list"),
        arguments: [:],
        permissionState: .limited,
        mode: .foreground
      ),
      .permissionRequestRequired(.reminders)
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: requireDefinition("calendar.create"),
        arguments: ["title": "Review", "startsInMinutes": 15],
        permissionState: .limited,
        mode: .foreground
      ),
      .approvalRequired
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: requireDefinition("calendar.list"),
        arguments: [:],
        permissionState: .limited,
        mode: .background
      ),
      .denied(.permissionPromptRequiresForeground(.calendar))
    )
  }

  func testPolicyFailsClosedInBackground() {
    let policy = AgentApprovalPolicy()
    let memorySave = requireDefinition("memory.save")
    let calendarCreate = requireDefinition("calendar.create")
    let memoryRecall = requireDefinition("memory.recall")

    XCTAssertEqual(
      policy.evaluate(
        definition: memorySave,
        arguments: ["content": "x", "kind": "fact"],
        permissionState: nil,
        mode: .background
      ),
      .denied(.backgroundExecutionUnsupported("memory.save"))
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: calendarCreate,
        arguments: ["title": "x", "startsInMinutes": 10],
        permissionState: .granted,
        mode: .background
      ),
      .denied(.backgroundExecutionUnsupported("calendar.create"))
    )
    XCTAssertEqual(
      policy.evaluate(
        definition: memoryRecall,
        arguments: ["query": "x"],
        permissionState: nil,
        mode: .background
      ),
      .allowed
    )
  }

  private func requireDefinition(_ id: String) -> AgentToolDefinition {
    guard let definition = AgentToolCatalog.definition(for: id) else {
      fatalError("Missing test tool \(id)")
    }
    return definition
  }
}

private final class LockedTestClock: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Date

  init(_ value: Date) {
    self.value = value
  }

  func now() -> Date {
    lock.lock()
    defer { lock.unlock() }
    return value
  }

  func advance(by interval: TimeInterval) {
    lock.lock()
    value = value.addingTimeInterval(interval)
    lock.unlock()
  }
}
