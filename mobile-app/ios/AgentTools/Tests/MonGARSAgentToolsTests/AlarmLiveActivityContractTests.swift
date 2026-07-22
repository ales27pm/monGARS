import Foundation
import XCTest

final class AlarmLiveActivityContractTests: XCTestCase {
  func testMetadataIdentityAndAllAlarmKitModesAreShared() throws {
    let metadata = try source("Sources/MonGARSAlarmSupport/MonGARSAlarmMetadata.swift")
    XCTAssertTrue(metadata.contains("public struct MonGARSAlarmMetadata: AlarmMetadata"))
    XCTAssertTrue(metadata.contains("Codable"))

    let widget = try iosSource("MonGARSAlarmWidget/MonGARSAlarmWidgetBundle.swift")
    XCTAssertTrue(widget.contains("ActivityConfiguration(for: AlarmAttributes<MonGARSAlarmMetadata>.self)"))
    XCTAssertTrue(widget.contains("if #available(iOS 26.0, *)"))
    XCTAssertTrue(widget.contains("MonGARSAlarmAvailabilityWidget()"))
    XCTAssertTrue(widget.contains("case .countdown"))
    XCTAssertTrue(widget.contains("case .paused"))
    XCTAssertTrue(widget.contains("case .alert"))
    XCTAssertTrue(widget.contains("DynamicIslandExpandedRegion"))
    XCTAssertTrue(widget.contains("compactLeading:"))
    XCTAssertTrue(widget.contains("compactTrailing:"))
    XCTAssertTrue(widget.contains("minimal:"))
  }

  func testLiveActivityPlistsAndEmbeddingContract() throws {
    let appInfo = try plist("MonGARSMobile/Info.plist")
    let widgetInfo = try plist("MonGARSAlarmWidget/Info.plist")
    XCTAssertEqual(appInfo["NSSupportsLiveActivities"] as? Bool, true)
    XCTAssertEqual(widgetInfo["NSSupportsLiveActivities"] as? Bool, true)
    let extensionInfo = try XCTUnwrap(widgetInfo["NSExtension"] as? [String: Any])
    XCTAssertEqual(
      extensionInfo["NSExtensionPointIdentifier"] as? String,
      "com.apple.widgetkit-extension"
    )

    let project = try iosSource("MonGARSMobile.xcodeproj/project.pbxproj")
    XCTAssertTrue(project.contains("MonGARSAlarmWidget.appex in Embed App Extensions"))
    XCTAssertTrue(project.contains("MonGARSAlarmSupport in Frameworks"))
    XCTAssertTrue(project.contains("APPLICATION_EXTENSION_API_ONLY = YES"))
    XCTAssertFalse(project.contains("IPHONEOS_DEPLOYMENT_TARGET = 26.0"))

    let entitlements = try plist("MonGARSMobile/MonGARSMobile.entitlements")
    XCTAssertNil(
      entitlements["com.apple.developer.alarm"],
      "AlarmKit is configured by usage description and authorization, not an entitlement"
    )
  }

  func testAlarmServiceDefaultsFixedAlarmsToLumenFiveMinuteSnooze() throws {
    let service = try source("Sources/MonGARSAgentTools/TriggerAndAlarmServices.swift")
    XCTAssertTrue(service.contains("schedule: .fixed(fireDate)"))
    XCTAssertTrue(service.contains("snoozeMinutes = 5"))
    XCTAssertTrue(service.contains("guard hasRelativeTime != hasTimestamp"))
    XCTAssertTrue(service.contains("guard !repeats"))
    XCTAssertTrue(service.contains("postAlert: TimeInterval(snoozeMinutes * 60)"))
    XCTAssertTrue(service.contains("Alarm.CountdownDuration"))
    XCTAssertTrue(service.contains(".timer("))
    XCTAssertTrue(service.contains("hasLiveActivityConfiguration"))

    let executor = try source("Sources/MonGARSAgentTools/IOSAgentToolExecutor.swift")
    XCTAssertTrue(
      executor.contains(
        ".alarmSchedule, .alarmCountdown, .alarmPause, .alarmResume, .alarmSnooze"
      ),
      "alarm.schedule must not be advertised when its required Live Activity is absent"
    )
  }

  private func source(_ relativePath: String) throws -> String {
    try String(contentsOf: packageRoot.appendingPathComponent(relativePath), encoding: .utf8)
  }

  private func iosSource(_ relativePath: String) throws -> String {
    try String(contentsOf: iosRoot.appendingPathComponent(relativePath), encoding: .utf8)
  }

  private func plist(_ relativePath: String) throws -> [String: Any] {
    let data = try Data(contentsOf: iosRoot.appendingPathComponent(relativePath))
    return try XCTUnwrap(
      PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any]
    )
  }

  private var packageRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  private var iosRoot: URL {
    packageRoot.deletingLastPathComponent()
  }
}
