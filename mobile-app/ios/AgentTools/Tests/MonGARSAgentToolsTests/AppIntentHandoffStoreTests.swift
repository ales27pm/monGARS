import Foundation
import XCTest
@testable import MonGARSAgentTools

final class AppIntentHandoffStoreTests: XCTestCase {
  private var directoryURL: URL!
  private var defaults: UserDefaults!
  private var suiteName: String!

  override func setUpWithError() throws {
    try super.setUpWithError()
    directoryURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("MonGARSAppIntentTests-\(UUID().uuidString)", isDirectory: true)
    suiteName = "MonGARSAppIntentTests.\(UUID().uuidString)"
    defaults = try XCTUnwrap(UserDefaults(suiteName: suiteName))
    defaults.removePersistentDomain(forName: suiteName)
  }

  override func tearDownWithError() throws {
    if let directoryURL {
      try? FileManager.default.removeItem(at: directoryURL)
    }
    if let suiteName {
      defaults?.removePersistentDomain(forName: suiteName)
    }
    directoryURL = nil
    defaults = nil
    suiteName = nil
    try super.tearDownWithError()
  }

  func testStoresContentOnlyInProtectedPayloadAndConsumesExactIdentifier() async throws {
    let now = Date(timeIntervalSince1970: 1_753_099_200)
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      now: { now }
    )

    let record = try await store.enqueue(kind: .memoryAdd, input: "  secret local fact  ")
    XCTAssertEqual(record.input, "secret local fact")
    let firstPending = await store.pending()
    XCTAssertEqual(firstPending, record)

    let defaultsSnapshot = defaults.dictionaryRepresentation()
    XCTAssertFalse(defaultsSnapshot.values.contains { String(describing: $0).contains("secret") })
    let substitutedAcknowledgement = await store.acknowledge(expectedID: UUID())
    XCTAssertFalse(substitutedAcknowledgement)
    let pendingAfterSubstitution = await store.pending()
    XCTAssertEqual(pendingAfterSubstitution, record)
    let acknowledgement = await store.acknowledge(expectedID: record.id)
    XCTAssertTrue(acknowledgement)
    let consumed = await store.pending()
    XCTAssertNil(consumed)
  }

  func testNewHandoffReplacesThePreviousSingleSlot() async throws {
    let now = Date(timeIntervalSince1970: 1_753_099_200)
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      now: { now }
    )
    let first = try await store.enqueue(kind: .ask, input: "First")
    let second = try await store.enqueue(kind: .memorySearch, input: "Second")

    XCTAssertNotEqual(first.id, second.id)
    let pending = await store.pending()
    XCTAssertEqual(pending, second)
    let staleAcknowledgement = await store.acknowledge(expectedID: first.id)
    XCTAssertFalse(staleAcknowledgement)
  }

  func testExpiredOrFutureDatedPointerFailsClosed() async throws {
    final class Clock: @unchecked Sendable {
      var date = Date(timeIntervalSince1970: 1_753_099_200)
    }
    let clock = Clock()
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      lifetime: 60,
      now: { clock.date }
    )
    _ = try await store.enqueue(kind: .runTrigger, input: "Morning")
    clock.date = clock.date.addingTimeInterval(61)

    let pending = await store.pending()
    XCTAssertNil(pending)
  }

  func testExpiredOrphanPayloadIsSweptAfterItsTTL() async throws {
    final class Clock: @unchecked Sendable {
      var date = Date(timeIntervalSince1970: 1_753_099_200)
    }
    let clock = Clock()
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      lifetime: 60,
      now: { clock.date }
    )
    let record = try await store.enqueue(kind: .ask, input: "Orphan after crash")
    let payloadURL = directoryURL
      .appendingPathComponent("\(record.id.uuidString.lowercased()).json")
    defaults.removePersistentDomain(forName: suiteName)
    _ = defaults.synchronize()
    clock.date = clock.date.addingTimeInterval(61)

    XCTAssertTrue(FileManager.default.fileExists(atPath: payloadURL.path))
    let pending = await store.pending()

    XCTAssertNil(pending)
    XCTAssertFalse(FileManager.default.fileExists(atPath: payloadURL.path))
  }

  func testRecentUnpointedPayloadIsConservativelyRetainedUntilTTL() async throws {
    final class Clock: @unchecked Sendable {
      var date = Date(timeIntervalSince1970: 1_753_099_200)
    }
    let clock = Clock()
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      lifetime: 60,
      now: { clock.date }
    )
    let record = try await store.enqueue(kind: .memorySearch, input: "Recent")
    let payloadURL = directoryURL
      .appendingPathComponent("\(record.id.uuidString.lowercased()).json")
    defaults.removePersistentDomain(forName: suiteName)
    _ = defaults.synchronize()
    clock.date = clock.date.addingTimeInterval(30)

    // The pointer may have been lost during publication, so a recent valid
    // payload is retained conservatively until its embedded expiry.
    let pending = await store.pending()

    XCTAssertNil(pending)
    XCTAssertTrue(FileManager.default.fileExists(atPath: payloadURL.path))
  }

  func testSweepNeverDeletesFilesOutsideCanonicalPayloadPattern() async throws {
    let now = Date(timeIntervalSince1970: 1_753_099_200)
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults,
      lifetime: 60,
      now: { now }
    )
    try FileManager.default.createDirectory(
      at: directoryURL,
      withIntermediateDirectories: true
    )
    let nonUUIDJSON = directoryURL.appendingPathComponent("notes.json")
    let wrongExtension = directoryURL
      .appendingPathComponent("\(UUID().uuidString.lowercased()).txt")
    try Data("private".utf8).write(to: nonUUIDJSON)
    try Data("private".utf8).write(to: wrongExtension)
    let oldDate = now.addingTimeInterval(-120)
    try FileManager.default.setAttributes(
      [.modificationDate: oldDate],
      ofItemAtPath: nonUUIDJSON.path
    )
    try FileManager.default.setAttributes(
      [.modificationDate: oldDate],
      ofItemAtPath: wrongExtension.path
    )

    let pending = await store.pending()

    XCTAssertNil(pending)
    XCTAssertTrue(FileManager.default.fileExists(atPath: nonUUIDJSON.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: wrongExtension.path))
  }

  func testValidatesKindSpecificInputBounds() async throws {
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults
    )

    await XCTAssertThrowsErrorAsync {
      _ = try await store.enqueue(kind: .memorySearch, input: String(repeating: "x", count: 193))
    }
    await XCTAssertThrowsErrorAsync {
      _ = try await store.enqueue(kind: .memoryAdd, input: String(repeating: "x", count: 187))
    }
    await XCTAssertThrowsErrorAsync {
      _ = try await store.enqueue(kind: .ask, input: String(repeating: "x", count: 513))
    }
    await XCTAssertThrowsErrorAsync {
      _ = try await store.enqueue(kind: .diagnostics, input: "unexpected")
    }
    await XCTAssertThrowsErrorAsync {
      _ = try await store.enqueue(kind: .masked, input: nil)
    }
    let diagnostics = try await store.enqueue(kind: .diagnostics, input: nil)
    XCTAssertNil(diagnostics.input)
  }

  func testProfileBindingIsCapturedAndOwnerMismatchFailsClosed() async throws {
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults
    )
    let activatedAlice = await store.setActiveProfile(rawOwnerID: "account:alice")
    XCTAssertTrue(activatedAlice)
    let record = try await store.enqueue(kind: .memorySearch, input: "cedar")
    let activatedBob = await store.setActiveProfile(rawOwnerID: "account:bob")
    XCTAssertTrue(activatedBob)

    let pendingForAlice = await store.pending(rawOwnerID: "account:alice")
    let pendingForBob = await store.pending(rawOwnerID: "account:bob")
    let aliceLookup = try XCTUnwrap(pendingForAlice)
    XCTAssertTrue(aliceLookup.profileMatches)
    XCTAssertEqual(aliceLookup.handoff, record)
    let bobLookup = try XCTUnwrap(pendingForBob)
    XCTAssertFalse(bobLookup.profileMatches)
    XCTAssertEqual(bobLookup.handoff.id, record.id)
    XCTAssertEqual(bobLookup.handoff.kind, .masked)
    XCTAssertNil(bobLookup.handoff.input)
    let wrongOwnerAcknowledgement = await store.acknowledge(
      expectedID: record.id,
      rawOwnerID: "account:bob"
    )
    XCTAssertFalse(wrongOwnerAcknowledgement)
    let stillPending = await store.pending()
    XCTAssertNotNil(stillPending)
    let correctOwnerAcknowledgement = await store.acknowledge(
      expectedID: record.id,
      rawOwnerID: "account:alice"
    )
    XCTAssertTrue(correctOwnerAcknowledgement)
    let pendingAfterAcknowledgement = await store.pending()
    XCTAssertNil(pendingAfterAcknowledgement)
  }

  func testExactMemoryConsumptionRejectsSubstitutionAndReplay() async throws {
    let store = MonGARSAppIntentHandoffStore(
      directoryURL: directoryURL,
      defaults: defaults
    )
    let activated = await store.setActiveProfile(rawOwnerID: "account:alice")
    XCTAssertTrue(activated)
    let record = try await store.enqueue(kind: .memoryAdd, input: "cedar fact")

    let wrongID = await store.consumeExactMemoryAction(
      expectedID: UUID(),
      rawOwnerID: "account:alice",
      expectedKind: .memoryAdd,
      expectedInput: "cedar fact"
    )
    XCTAssertNil(wrongID)
    let wrongOwner = await store.consumeExactMemoryAction(
      expectedID: record.id,
      rawOwnerID: "account:bob",
      expectedKind: .memoryAdd,
      expectedInput: "cedar fact"
    )
    XCTAssertNil(wrongOwner)
    let wrongKind = await store.consumeExactMemoryAction(
      expectedID: record.id,
      rawOwnerID: "account:alice",
      expectedKind: .memorySearch,
      expectedInput: "cedar fact"
    )
    XCTAssertNil(wrongKind)
    let wrongInput = await store.consumeExactMemoryAction(
      expectedID: record.id,
      rawOwnerID: "account:alice",
      expectedKind: .memoryAdd,
      expectedInput: "different fact"
    )
    XCTAssertNil(wrongInput)

    let consumed = await store.consumeExactMemoryAction(
      expectedID: record.id,
      rawOwnerID: "account:alice",
      expectedKind: .memoryAdd,
      expectedInput: "cedar fact"
    )
    XCTAssertEqual(consumed, record)
    let replay = await store.consumeExactMemoryAction(
      expectedID: record.id,
      rawOwnerID: "account:alice",
      expectedKind: .memoryAdd,
      expectedInput: "cedar fact"
    )
    XCTAssertNil(replay)
  }
}

private func XCTAssertThrowsErrorAsync(
  _ expression: () async throws -> Void,
  file: StaticString = #filePath,
  line: UInt = #line
) async {
  do {
    try await expression()
    XCTFail("Expected expression to throw.", file: file, line: line)
  } catch {
    XCTAssertEqual(
      error as? MonGARSAppIntentHandoffStoreError,
      .invalidInput,
      file: file,
      line: line
    )
  }
}
