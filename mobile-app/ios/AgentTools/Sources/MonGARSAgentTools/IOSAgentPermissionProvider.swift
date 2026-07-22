import Foundation
import MonGARSCoreML

#if os(iOS)
import AVFoundation
import Contacts
import CoreLocation
import CoreMotion
import EventKit
import HealthKit
import Photos
import UIKit
import UserNotifications
#if canImport(AlarmKit)
import AlarmKit
#endif
#endif

public final class IOSAgentPermissionProvider: AgentPermissionProviding, @unchecked Sendable {
  public static let shared = IOSAgentPermissionProvider()

  public init() {}

  public func state(for permission: AgentPermission) async -> AgentPermissionState {
#if os(iOS)
    switch permission {
    case .calendar:
      guard Self.hasUsageDescription("NSCalendarsFullAccessUsageDescription") else { return .unavailable }
      return Self.mapEventKit(EKEventStore.authorizationStatus(for: .event))
    case .reminders:
      guard Self.hasUsageDescription("NSRemindersFullAccessUsageDescription") else { return .unavailable }
      return Self.mapEventKit(EKEventStore.authorizationStatus(for: .reminder))
    case .contacts:
      guard Self.hasUsageDescription("NSContactsUsageDescription") else { return .unavailable }
      switch CNContactStore.authorizationStatus(for: .contacts) {
      case .authorized: return .granted
      case .limited: return .limited
      case .notDetermined: return .notDetermined
      case .denied: return .denied
      case .restricted: return .restricted
      @unknown default: return .unavailable
      }
    case .location:
      guard Self.hasUsageDescription("NSLocationWhenInUseUsageDescription"),
        CLLocationManager.locationServicesEnabled() else { return .unavailable }
      return Self.mapLocation(CLLocationManager.authorizationStatus())
    case .photos:
      guard Self.hasUsageDescription("NSPhotoLibraryUsageDescription") else { return .unavailable }
      switch PHPhotoLibrary.authorizationStatus(for: .readWrite) {
      case .authorized: return .granted
      case .limited: return .limited
      case .notDetermined: return .notDetermined
      case .denied: return .denied
      case .restricted: return .restricted
      @unknown default: return .unavailable
      }
    case .camera:
      guard Self.hasUsageDescription("NSCameraUsageDescription") else { return .unavailable }
      let cameraAvailable = await MainActor.run {
        UIImagePickerController.isSourceTypeAvailable(.camera)
      }
      guard cameraAvailable else { return .unavailable }
      switch AVCaptureDevice.authorizationStatus(for: .video) {
      case .authorized: return .granted
      case .notDetermined: return .notDetermined
      case .denied: return .denied
      case .restricted: return .restricted
      @unknown default: return .unavailable
      }
    case .health:
      guard Self.hasUsageDescription("NSHealthShareUsageDescription"),
        HKHealthStore.isHealthDataAvailable() else { return .unavailable }
      let store = HKHealthStore()
      let status: HKAuthorizationRequestStatus? = await withCheckedContinuation { continuation in
        store.getRequestStatusForAuthorization(toShare: [], read: Self.healthReadTypes) { status, error in
          continuation.resume(returning: error == nil ? status : nil)
        }
      }
      switch status {
      case .shouldRequest: return .notDetermined
      case .unnecessary: return .granted
      case .unknown, .none: return .unavailable
      @unknown default: return .unavailable
      }
    case .motion:
      guard Self.hasUsageDescription("NSMotionUsageDescription"),
        CMMotionActivityManager.isActivityAvailable() else { return .unavailable }
      switch CMMotionActivityManager.authorizationStatus() {
      case .authorized: return .granted
      case .notDetermined: return .notDetermined
      case .denied: return .denied
      case .restricted: return .restricted
      @unknown default: return .unavailable
      }
    case .notifications:
      let settings = await UNUserNotificationCenter.current().notificationSettings()
      switch settings.authorizationStatus {
      case .authorized, .provisional, .ephemeral: return .granted
      case .notDetermined: return .notDetermined
      case .denied: return .denied
      @unknown default: return .unavailable
      }
    case .alarms:
      guard Self.hasUsageDescription("NSAlarmKitUsageDescription") else { return .unavailable }
#if canImport(AlarmKit)
      if #available(iOS 26.0, *) {
        switch AlarmManager.shared.authorizationState {
        case .authorized: return .granted
        case .notDetermined: return .notDetermined
        case .denied: return .denied
        @unknown default: return .unavailable
        }
      }
#endif
      return .unavailable
    }
#else
    return .unavailable
#endif
  }

  /// Requests one permission from a foreground UI flow. Agent execution itself
  /// never prompts implicitly; callers resume the approved run afterwards.
  public func request(_ permission: AgentPermission) async -> AgentPermissionState {
#if os(iOS)
    switch permission {
    case .calendar:
      guard Self.hasUsageDescription("NSCalendarsFullAccessUsageDescription") else { return .unavailable }
      _ = try? await EKEventStore().requestFullAccessToEvents()
    case .reminders:
      guard Self.hasUsageDescription("NSRemindersFullAccessUsageDescription") else { return .unavailable }
      _ = try? await EKEventStore().requestFullAccessToReminders()
    case .contacts:
      guard Self.hasUsageDescription("NSContactsUsageDescription") else { return .unavailable }
      _ = try? await CNContactStore().requestAccess(for: .contacts)
    case .location:
      guard Self.hasUsageDescription("NSLocationWhenInUseUsageDescription") else { return .unavailable }
      await IOSOneShotLocationProvider.shared.requestWhenInUseAuthorization()
    case .photos:
      guard Self.hasUsageDescription("NSPhotoLibraryUsageDescription") else { return .unavailable }
      _ = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    case .camera:
      guard Self.hasUsageDescription("NSCameraUsageDescription") else { return .unavailable }
      _ = await AVCaptureDevice.requestAccess(for: .video)
    case .health:
      guard Self.hasUsageDescription("NSHealthShareUsageDescription"),
        HKHealthStore.isHealthDataAvailable() else { return .unavailable }
      try? await HKHealthStore().requestAuthorization(toShare: [], read: Self.healthReadTypes)
    case .motion:
      guard Self.hasUsageDescription("NSMotionUsageDescription"),
        CMMotionActivityManager.isActivityAvailable() else { return .unavailable }
      let end = Date()
      let manager = CMMotionActivityManager()
      await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        manager.queryActivityStarting(
          from: end.addingTimeInterval(-60),
          to: end,
          to: .main,
          withHandler: { _, _ in continuation.resume() }
        )
      }
    case .notifications:
      _ = try? await UNUserNotificationCenter.current()
        .requestAuthorization(options: [.alert, .sound, .badge])
    case .alarms:
      guard Self.hasUsageDescription("NSAlarmKitUsageDescription") else { return .unavailable }
#if canImport(AlarmKit)
      if #available(iOS 26.0, *) {
        _ = try? await AlarmManager.shared.requestAuthorization()
      }
#endif
    }
    return await state(for: permission)
#else
    return .unavailable
#endif
  }

#if os(iOS)
  private static func hasUsageDescription(_ key: String) -> Bool {
    guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String else { return false }
    return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  private static func mapEventKit(_ status: EKAuthorizationStatus) -> AgentPermissionState {
    switch status {
    case .fullAccess: return .granted
    case .writeOnly: return .limited
    case .notDetermined: return .notDetermined
    case .denied: return .denied
    case .restricted: return .restricted
    @unknown default: return .unavailable
    }
  }

  private static func mapLocation(_ status: CLAuthorizationStatus) -> AgentPermissionState {
    switch status {
    case .authorizedAlways, .authorizedWhenInUse: return .granted
    case .notDetermined: return .notDetermined
    case .denied: return .denied
    case .restricted: return .restricted
    @unknown default: return .unavailable
    }
  }

  private static let healthReadTypes: Set<HKObjectType> = [
    HKQuantityType(.stepCount),
    HKQuantityType(.heartRate),
    HKCategoryType(.sleepAnalysis),
    HKQuantityType(.activeEnergyBurned),
    HKQuantityType(.distanceWalkingRunning),
  ]
#endif
}
