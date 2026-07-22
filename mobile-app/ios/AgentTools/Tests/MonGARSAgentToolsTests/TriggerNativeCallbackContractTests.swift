import Foundation
import XCTest

final class TriggerNativeCallbackContractTests: XCTestCase {
  func testNotificationTapPersistsAndEmitsOnlyOpaqueHandoffMetadata() throws {
    let appDelegate = try source("MonGARSMobile/AppDelegate.mm")
    XCTAssertTrue(appDelegate.contains("MonGARSAgentTriggerHandoffAvailable"))
    XCTAssertTrue(appDelegate.contains("MonGARS.PendingAgentTriggerHandoffID"))
    XCTAssertTrue(appDelegate.contains("MonGARS.PendingAgentTriggerHandoffDate"))
    XCTAssertTrue(appDelegate.contains("@\"id\": triggerID, @\"tappedAt\": tappedAt"))
    XCTAssertFalse(appDelegate.contains("monGARSAgentTriggerPrompt"))

    let bridge = try source("CoreMLInference/CoreMLInferenceModule.swift")
    XCTAssertTrue(bridge.contains("case agentTriggerHandoff = \"onAgentTriggerHandoff\""))
    XCTAssertTrue(bridge.contains("MonGARSAgentTriggerHandoffAvailable"))
    XCTAssertTrue(bridge.contains("UUID(uuidString: identifier) != nil"))
    XCTAssertTrue(bridge.contains("\"tappedAt\": ISO8601DateFormatter().string(from: tappedAt)"))
  }

  private func source(_ relativePath: String) throws -> String {
    try String(contentsOf: iosRoot.appendingPathComponent(relativePath), encoding: .utf8)
  }

  private var iosRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }
}
