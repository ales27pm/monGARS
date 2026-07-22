@testable import MonGARSAgentTools
import MonGARSCoreML
import XCTest

final class AgentHostCatalogTests: XCTestCase {
  private struct OwnerScopedGraphService: MicrosoftGraphServing {
    let connectedScope: String

    func isConfigured() async -> Bool { false }
    func isConfigured(profileScope: String?) async -> Bool {
      profileScope == connectedScope
    }
    func perform(
      operation: AgentHostOperation,
      arguments: AgentJSONArguments
    ) async -> AgentServiceResponse {
      .unavailable("Unscoped test call")
    }
  }

  func testDispatchTableCoversCanonicalCatalogExactly() {
    XCTAssertEqual(AgentHostOperation.canonicalDispatchTable.count, 53)
    XCTAssertEqual(
      Set(AgentHostOperation.canonicalDispatchTable.keys),
      AgentToolCatalog.canonicalIDs
    )
    XCTAssertEqual(Set(AgentHostOperation.canonicalDispatchTable.values), Set(AgentHostOperation.allCases))
  }

  func testAlarmSurfaceIncludesEveryLumenAlarmKitOperation() {
    let ids = Set(
      AgentHostOperation.canonicalDispatchTable.compactMap { id, operation in
        AgentHostOperation.alarmOperations.contains(operation) ? id.rawValue : nil
      }
    )
    XCTAssertEqual(ids, [
      "alarm.authorization_status",
      "alarm.request_authorization",
      "alarm.schedule",
      "alarm.countdown",
      "alarm.list",
      "alarm.pause",
      "alarm.resume",
      "alarm.stop",
      "alarm.snooze",
      "alarm.cancel",
    ])
  }

  func testBackgroundUIInvocationFailsClosed() async {
    let executor = IOSAgentToolExecutor(
      graphService: nil,
      webService: nil,
      importedFilesRoot: FileManager.default.temporaryDirectory,
      protectedStateRoot: FileManager.default.temporaryDirectory,
      presenter: UnavailableAgentForegroundPresenter(),
      photoProvider: nil
    )
    let invocation = AgentToolInvocation(
      runID: UUID(),
      stepIndex: 0,
      toolID: "camera.capture",
      arguments: [:],
      mode: .background
    )
    let result = await executor.execute(invocation: invocation)
    XCTAssertEqual(result.invocationID, invocation.id)
    XCTAssertEqual(result.status, .denied)
    XCTAssertEqual(result.errorCode, "background_execution_denied")
  }

  func testUnknownToolNeverFallsThrough() async {
    let executor = IOSAgentToolExecutor(
      graphService: nil,
      webService: nil,
      importedFilesRoot: FileManager.default.temporaryDirectory,
      protectedStateRoot: FileManager.default.temporaryDirectory,
      presenter: nil,
      photoProvider: nil
    )
    let invocation = AgentToolInvocation(
      runID: UUID(),
      stepIndex: 0,
      toolID: "calendar.create_and_send",
      arguments: [:],
      mode: .foreground
    )
    let result = await executor.execute(invocation: invocation)
    XCTAssertEqual(result.status, .unavailable)
    XCTAssertEqual(result.errorCode, "tool_unavailable")
  }

  func testUnavailableIntegrationsAreNotAdvertised() async {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("MonGARSAgentToolsTests-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let executor = IOSAgentToolExecutor(
      importedFilesRoot: root.appendingPathComponent("imports"),
      protectedStateRoot: root.appendingPathComponent("state"),
      presenter: nil,
      photoProvider: nil
    )
    let available = await executor.availableToolIDs()
    XCTAssertFalse(available.contains("outlook.status"))
    XCTAssertFalse(available.contains("outlook.mail.send"))
    XCTAssertFalse(available.contains("web.fetch"))
    XCTAssertFalse(available.contains("files.read"))
    XCTAssertFalse(available.contains("rag.index_files"))
  }

  func testOutlookStatusIsOwnerScopedAndAuthenticatedToolsNeedExactSession() async throws {
    let alice = try XCTUnwrap(AgentOpaqueProfileScope.make(rawOwnerID: "account:alice"))
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("MonGARSOwnerCapabilities-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let executor = IOSAgentToolExecutor(
      graphService: OwnerScopedGraphService(connectedScope: alice),
      webService: nil,
      importedFilesRoot: root.appendingPathComponent("imports"),
      protectedStateRoot: root.appendingPathComponent("state"),
      presenter: nil,
      photoProvider: nil
    )

    let aliceTools = await ScopedIOSAgentToolExecutor(
      base: executor,
      rawOwnerID: "account:alice"
    ).availableToolIDs()
    let bobTools = await ScopedIOSAgentToolExecutor(
      base: executor,
      rawOwnerID: "account:bob"
    ).availableToolIDs()

    XCTAssertTrue(aliceTools.contains("outlook.status"))
    XCTAssertTrue(aliceTools.contains("outlook.mail.send"))
    XCTAssertTrue(bobTools.contains("outlook.status"))
    XCTAssertFalse(bobTools.contains("outlook.mail.send"))
    let unscopedTools = await executor.availableToolIDs()
    XCTAssertFalse(unscopedTools.contains("outlook.status"))
  }
}
