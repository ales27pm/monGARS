import Foundation
import MonGARSCoreML

#if os(iOS)
import AVFoundation
import Contacts
import CoreLocation
import CoreMotion
import EventKit
import HealthKit
import MapKit
import MessageUI
import Photos
import UIKit
#if canImport(WeatherKit)
import WeatherKit
#endif
#endif

public struct AgentCoordinate: Sendable, Equatable {
  public let latitude: Double
  public let longitude: Double

  public init(latitude: Double, longitude: Double) {
    self.latitude = latitude
    self.longitude = longitude
  }
}

public protocol AgentLocationProviding: Sendable {
  func currentCoordinate() async -> Result<AgentCoordinate, AgentLocationFailure>
}

public enum AgentLocationFailure: Error, Sendable, Equatable {
  case unavailable
  case permissionNotDetermined
  case denied
  case restricted
  case busy
  case timedOut
  case providerFailed
}

#if os(iOS)
@MainActor
public final class IOSOneShotLocationProvider: NSObject, AgentLocationProviding,
  CLLocationManagerDelegate, @unchecked Sendable
{
  public static let shared = IOSOneShotLocationProvider()

  private let manager = CLLocationManager()
  private var locationContinuation: CheckedContinuation<Result<AgentCoordinate, AgentLocationFailure>, Never>?
  private var authorizationContinuation: CheckedContinuation<Void, Never>?
  private var timeoutTask: Task<Void, Never>?

  public override init() {
    super.init()
    manager.delegate = self
    manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
  }

  public func currentCoordinate() async -> Result<AgentCoordinate, AgentLocationFailure> {
    guard CLLocationManager.locationServicesEnabled() else { return .failure(.unavailable) }
    switch manager.authorizationStatus {
    case .authorizedAlways, .authorizedWhenInUse: break
    case .notDetermined: return .failure(.permissionNotDetermined)
    case .denied: return .failure(.denied)
    case .restricted: return .failure(.restricted)
    @unknown default: return .failure(.unavailable)
    }
    guard locationContinuation == nil else { return .failure(.busy) }
    return await withCheckedContinuation { continuation in
      locationContinuation = continuation
      manager.requestLocation()
      timeoutTask?.cancel()
      timeoutTask = Task { [weak self] in
        try? await Task.sleep(nanoseconds: 10_000_000_000)
        guard !Task.isCancelled else { return }
        await self?.finishLocation(.failure(.timedOut))
      }
    }
  }

  public func requestWhenInUseAuthorization() async {
    guard manager.authorizationStatus == .notDetermined,
      authorizationContinuation == nil else { return }
    await withCheckedContinuation { continuation in
      authorizationContinuation = continuation
      manager.requestWhenInUseAuthorization()
      Task { [weak self] in
        try? await Task.sleep(nanoseconds: 30_000_000_000)
        await self?.finishAuthorization()
      }
    }
  }

  public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
    guard let location = locations.last else {
      finishLocation(.failure(.providerFailed))
      return
    }
    finishLocation(.success(.init(
      latitude: location.coordinate.latitude,
      longitude: location.coordinate.longitude
    )))
  }

  public func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
    finishLocation(.failure(.providerFailed))
  }

  public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
    guard manager.authorizationStatus != .notDetermined else { return }
    finishAuthorization()
  }

  private func finishLocation(_ result: Result<AgentCoordinate, AgentLocationFailure>) {
    timeoutTask?.cancel()
    timeoutTask = nil
    let continuation = locationContinuation
    locationContinuation = nil
    continuation?.resume(returning: result)
  }

  private func finishAuthorization() {
    let continuation = authorizationContinuation
    authorizationContinuation = nil
    continuation?.resume()
  }
}

public struct IOSLocationProvider: AgentLocationProviding, Sendable {
  public init() {}

  public func currentCoordinate() async -> Result<AgentCoordinate, AgentLocationFailure> {
    await IOSOneShotLocationProvider.shared.currentCoordinate()
  }
}
#else
public struct IOSOneShotLocationProvider: AgentLocationProviding, Sendable {
  public static let shared = IOSOneShotLocationProvider()
  public init() {}
  public func currentCoordinate() async -> Result<AgentCoordinate, AgentLocationFailure> {
    .failure(.unavailable)
  }
  public func requestWhenInUseAuthorization() async {}
}

public typealias IOSLocationProvider = IOSOneShotLocationProvider
#endif

#if os(iOS)
public actor AppleProductivityService {
  private let eventStore = EKEventStore()

  public init() {}

  public func createCalendarEvent(arguments: AgentJSONArguments) -> AgentServiceResponse {
    guard let title = AgentToolInput.requiredString("title", in: arguments, maximumBytes: 1_000),
      let minutes = AgentToolInput.integer("startsInMinutes", in: arguments, range: 0...(366 * 24 * 60)) else {
      return .failed("Calendar title or start offset is invalid.", code: "calendar_invalid_arguments")
    }
    let status = EKEventStore.authorizationStatus(for: .event)
    guard status == .fullAccess || status == .writeOnly else {
      return Self.permissionResponse(status, noun: "calendar", requiresRead: false)
    }
    guard let calendar = eventStore.defaultCalendarForNewEvents else {
      return .unavailable("No writable calendar is available.", code: "calendar_unavailable")
    }
    let event = EKEvent(eventStore: eventStore)
    event.title = title
    event.startDate = Date().addingTimeInterval(TimeInterval(minutes * 60))
    event.endDate = event.startDate.addingTimeInterval(3_600)
    event.calendar = calendar
    do {
      try eventStore.save(event, span: .thisEvent, commit: true)
      return .success(
        "Created calendar event \"\(title)\".",
        payload: [
          "id": .string(event.eventIdentifier ?? ""),
          "title": .string(title),
          "startsAt": .string(Self.iso8601.string(from: event.startDate)),
          "calendar": .string(calendar.title),
        ]
      )
    } catch {
      return .failed("The calendar event could not be saved.", code: "calendar_save_failed")
    }
  }

  public func listCalendarEvents() -> AgentServiceResponse {
    let status = EKEventStore.authorizationStatus(for: .event)
    guard status == .fullAccess else {
      return Self.permissionResponse(status, noun: "calendar", requiresRead: true)
    }
    let start = Date()
    let end = start.addingTimeInterval(7 * 86_400)
    let predicate = eventStore.predicateForEvents(withStart: start, end: end, calendars: nil)
    let events = eventStore.events(matching: predicate)
      .sorted { $0.startDate < $1.startDate }
      .prefix(20)
    let values: [AgentJSONValue] = events.map { event in
      [
        "id": .string(event.eventIdentifier ?? ""),
        "title": .string(event.title ?? "Untitled"),
        "startsAt": .string(Self.iso8601.string(from: event.startDate)),
        "endsAt": .string(Self.iso8601.string(from: event.endDate)),
        "calendar": .string(event.calendar.title),
      ]
    }
    let text = events.isEmpty
      ? "No upcoming calendar events were found."
      : events.map { "- \($0.title ?? "Untitled") at \(Self.iso8601.string(from: $0.startDate))" }
        .joined(separator: "\n")
    return .success(text, payload: ["events": .array(values)])
  }

  public func createReminder(arguments: AgentJSONArguments) -> AgentServiceResponse {
    guard let title = AgentToolInput.requiredString("title", in: arguments, maximumBytes: 1_000) else {
      return .failed("Reminder title is invalid.", code: "reminder_invalid_arguments")
    }
    let status = EKEventStore.authorizationStatus(for: .reminder)
    guard status == .fullAccess else {
      return Self.permissionResponse(status, noun: "reminders", requiresRead: false)
    }
    guard let calendar = eventStore.defaultCalendarForNewReminders() else {
      return .unavailable("No writable reminder list is available.", code: "reminders_unavailable")
    }
    let reminder = EKReminder(eventStore: eventStore)
    reminder.title = title
    reminder.calendar = calendar
    do {
      try eventStore.save(reminder, commit: true)
      return .success(
        "Added reminder \"\(title)\".",
        payload: [
          "id": .string(reminder.calendarItemIdentifier),
          "title": .string(title),
          "list": .string(calendar.title),
        ]
      )
    } catch {
      return .failed("The reminder could not be saved.", code: "reminder_save_failed")
    }
  }

  public func listReminders() async -> AgentServiceResponse {
    let status = EKEventStore.authorizationStatus(for: .reminder)
    guard status == .fullAccess else {
      return Self.permissionResponse(status, noun: "reminders", requiresRead: true)
    }
    let predicate = eventStore.predicateForIncompleteReminders(
      withDueDateStarting: nil,
      ending: nil,
      calendars: nil
    )
    let reminders: [EKReminder] = await withCheckedContinuation { continuation in
      eventStore.fetchReminders(matching: predicate) { values in
        continuation.resume(returning: Array((values ?? []).prefix(20)))
      }
    }
    let values: [AgentJSONValue] = reminders.map {
      [
        "id": .string($0.calendarItemIdentifier),
        "title": .string($0.title ?? "Untitled"),
        "list": .string($0.calendar.title),
      ]
    }
    let text = reminders.isEmpty
      ? "No pending reminders were found."
      : reminders.map { "- \($0.title ?? "Untitled")" }.joined(separator: "\n")
    return .success(text, payload: ["reminders": .array(values)])
  }

  private static func permissionResponse(
    _ status: EKAuthorizationStatus,
    noun: String,
    requiresRead: Bool
  ) -> AgentServiceResponse {
    switch status {
    case .denied: return .denied("Access to \(noun) is denied.", code: "\(noun)_permission_denied")
    case .restricted: return .denied("Access to \(noun) is restricted.", code: "\(noun)_permission_restricted")
    case .notDetermined: return .denied("Access to \(noun) has not been requested.", code: "\(noun)_permission_not_determined")
    case .writeOnly where requiresRead:
      return .denied("Read access to \(noun) is unavailable with write-only permission.", code: "\(noun)_read_denied")
    case .writeOnly, .fullAccess:
      return .failed("The \(noun) request could not be completed.", code: "\(noun)_provider_failed")
    @unknown default: return .unavailable("\(noun.capitalized) are unavailable.", code: "\(noun)_unavailable")
    }
  }

  private static let iso8601 = ISO8601DateFormatter()
}

public actor AppleContactsService {
  private let store = CNContactStore()

  public init() {}

  public func search(arguments: AgentJSONArguments) -> AgentServiceResponse {
    guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 500) else {
      return .failed("The contact search query is invalid.", code: "contacts_invalid_query")
    }
    let authorization = CNContactStore.authorizationStatus(for: .contacts)
    guard authorization == .authorized || authorization == .limited else {
      return .denied("Contacts access is not authorized.", code: "contacts_permission_denied")
    }
    let keys: [CNKeyDescriptor] = [
      CNContactIdentifierKey as CNKeyDescriptor,
      CNContactFormatter.descriptorForRequiredKeys(for: .fullName),
      CNContactPhoneNumbersKey as CNKeyDescriptor,
      CNContactEmailAddressesKey as CNKeyDescriptor,
    ]
    do {
      let contacts = try store.unifiedContacts(
        matching: CNContact.predicateForContacts(matchingName: query),
        keysToFetch: keys
      ).prefix(10)
      let values: [AgentJSONValue] = contacts.map { contact in
        let name = CNContactFormatter.string(from: contact, style: .fullName) ?? "Unnamed"
        return [
          "id": .string(contact.identifier),
          "name": .string(name),
          "phoneNumbers": .array(contact.phoneNumbers.prefix(4).map { .string($0.value.stringValue) }),
          "emailAddresses": .array(contact.emailAddresses.prefix(4).map { .string(String($0.value)) }),
        ]
      }
      let text = contacts.isEmpty
        ? "No matching contacts were found."
        : contacts.map { CNContactFormatter.string(from: $0, style: .fullName) ?? "Unnamed" }
          .map { "- \($0)" }.joined(separator: "\n")
      return .success(text, payload: ["contacts": .array(values)])
    } catch {
      return .failed("Contacts could not be searched.", code: "contacts_search_failed")
    }
  }
}

public final class IOSPhotoMetadataProvider: AgentPhotoMetadataProviding, @unchecked Sendable {
  public init() {}

  public func searchMetadata(query: String, limit: Int) async throws -> [AgentPhotoMetadata] {
    guard Self.isAuthorized else { throw IOSPhotoError.permissionDenied }
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let intent = Self.searchIntent(query: query, now: Date(), calendar: .current)
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = min(max(limit * 20, 100), 2_000)
    let assets = PHAsset.fetchAssets(with: options)
    let selfieIDs = intent.selfies ? Self.selfieIdentifiers(limit: 5_000) : []
    var output: [AgentPhotoMetadata] = []
    assets.enumerateObjects { asset, _, stop in
      let resource = PHAssetResource.assetResources(for: asset).first
      let filename = resource?.originalFilename
      guard Self.matches(
        asset: asset,
        filename: filename,
        intent: intent,
        selfieIDs: selfieIDs
      ) else { return }
      output.append(Self.metadata(asset: asset, filename: filename, queryMatched: query))
      if output.count >= min(max(limit, 1), 50) { stop.pointee = true }
    }
    return output
  }

  public func metadataSince(_ startDate: Date, limit: Int) async throws -> [AgentPhotoMetadata] {
    guard Self.isAuthorized else { throw IOSPhotoError.permissionDenied }
    let options = PHFetchOptions()
    options.predicate = NSPredicate(format: "creationDate >= %@", startDate as NSDate)
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = min(max(limit, 1), 5_000)
    let assets = PHAsset.fetchAssets(with: .image, options: options)
    var output: [AgentPhotoMetadata] = []
    assets.enumerateObjects { asset, _, _ in
      let filename = PHAssetResource.assetResources(for: asset).first?.originalFilename
      output.append(Self.metadata(asset: asset, filename: filename, queryMatched: nil))
    }
    return output
  }

  private static var isAuthorized: Bool {
    let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    return status == .authorized || status == .limited
  }

  private static func metadata(
    asset: PHAsset,
    filename: String?,
    queryMatched: String?
  ) -> AgentPhotoMetadata {
    .init(
      localIdentifier: asset.localIdentifier,
      filename: filename,
      createdAt: asset.creationDate,
      latitude: asset.location?.coordinate.latitude,
      longitude: asset.location?.coordinate.longitude,
      mediaType: mediaTypeName(asset.mediaType),
      mediaSubtypes: mediaSubtypeNames(asset.mediaSubtypes),
      isFavorite: asset.isFavorite,
      pixelWidth: asset.pixelWidth,
      pixelHeight: asset.pixelHeight,
      displayToken: "photos://asset/\(asset.localIdentifier)",
      queryMatched: queryMatched
    )
  }

  private struct SearchIntent {
    let dateRange: Range<Date>?
    let favorites: Bool
    let selfies: Bool
    let videos: Bool
    let screenshots: Bool
    let livePhotos: Bool
    let portraits: Bool
    let remainingTerms: [String]
  }

  private static func searchIntent(
    query: String,
    now: Date,
    calendar: Calendar
  ) -> SearchIntent {
    let words = query.split { !$0.isLetter && !$0.isNumber }.map(String.init)
    let wordSet = Set(words)
    let dateRange: Range<Date>?
    if wordSet.contains("today") {
      dateRange = calendar.startOfDay(for: now)..<now.addingTimeInterval(0.001)
    } else if wordSet.contains("yesterday") {
      let today = calendar.startOfDay(for: now)
      let yesterday = calendar.date(byAdding: .day, value: -1, to: today)
        ?? today.addingTimeInterval(-86_400)
      dateRange = yesterday..<today
    } else if wordSet.contains("week") {
      dateRange = (calendar.date(byAdding: .day, value: -7, to: now)
        ?? now.addingTimeInterval(-7 * 86_400))..<now.addingTimeInterval(0.001)
    } else if wordSet.contains("month") {
      dateRange = (calendar.date(byAdding: .month, value: -1, to: now)
        ?? now.addingTimeInterval(-31 * 86_400))..<now.addingTimeInterval(0.001)
    } else if wordSet.contains("year") {
      dateRange = (calendar.date(byAdding: .year, value: -1, to: now)
        ?? now.addingTimeInterval(-366 * 86_400))..<now.addingTimeInterval(0.001)
    } else {
      dateRange = nil
    }
    let ignored: Set<String> = [
      "today", "yesterday", "week", "month", "year", "favorite", "favorites",
      "favourite", "favourites", "selfie", "selfies", "video", "videos",
      "screenshot", "screenshots", "live", "photo", "photos", "portrait",
      "portraits", "latest", "newest", "recent", "find", "show", "from",
      "my", "the", "a", "an", "in", "of",
    ]
    return .init(
      dateRange: dateRange,
      favorites: !wordSet.isDisjoint(with: ["favorite", "favorites", "favourite", "favourites"]),
      selfies: !wordSet.isDisjoint(with: ["selfie", "selfies"]),
      videos: !wordSet.isDisjoint(with: ["video", "videos"]),
      screenshots: !wordSet.isDisjoint(with: ["screenshot", "screenshots"]),
      livePhotos: wordSet.contains("live"),
      portraits: !wordSet.isDisjoint(with: ["portrait", "portraits"]),
      remainingTerms: words.filter { !ignored.contains($0) }
    )
  }

  private static func matches(
    asset: PHAsset,
    filename: String?,
    intent: SearchIntent,
    selfieIDs: Set<String>
  ) -> Bool {
    if let range = intent.dateRange {
      guard let created = asset.creationDate, range.contains(created) else { return false }
    }
    if intent.favorites, !asset.isFavorite { return false }
    if intent.selfies, !selfieIDs.contains(asset.localIdentifier) { return false }
    if intent.videos, asset.mediaType != .video { return false }
    if !intent.videos, asset.mediaType != .image { return false }
    if intent.screenshots, !asset.mediaSubtypes.contains(.photoScreenshot) { return false }
    if intent.livePhotos, !asset.mediaSubtypes.contains(.photoLive) { return false }
    if intent.portraits, !asset.mediaSubtypes.contains(.photoDepthEffect) { return false }
    if !intent.remainingTerms.isEmpty {
      let searchable = [
        filename?.lowercased() ?? "",
        asset.creationDate.map { dateFormatter.string(from: $0).lowercased() } ?? "",
      ].joined(separator: " ")
      guard intent.remainingTerms.allSatisfy(searchable.contains) else { return false }
    }
    return true
  }

  private static func selfieIdentifiers(limit: Int) -> Set<String> {
    let collections = PHAssetCollection.fetchAssetCollections(
      with: .smartAlbum,
      subtype: .smartAlbumSelfPortraits,
      options: nil
    )
    var identifiers: Set<String> = []
    collections.enumerateObjects { collection, _, stopCollections in
      var reachedLimit = false
      PHAsset.fetchAssets(in: collection, options: nil).enumerateObjects { asset, _, stopAssets in
        identifiers.insert(asset.localIdentifier)
        if identifiers.count >= limit {
          stopAssets.pointee = true
          reachedLimit = true
        }
      }
      if reachedLimit { stopCollections.pointee = true }
    }
    return identifiers
  }

  private static func mediaTypeName(_ type: PHAssetMediaType) -> String {
    switch type {
    case .image: return "image"
    case .video: return "video"
    case .audio: return "audio"
    case .unknown: return "unknown"
    @unknown default: return "unknown"
    }
  }

  private static func mediaSubtypeNames(_ subtypes: PHAssetMediaSubtype) -> [String] {
    var names: [String] = []
    if subtypes.contains(.photoPanorama) { names.append("photoPanorama") }
    if subtypes.contains(.photoHDR) { names.append("photoHDR") }
    if subtypes.contains(.photoScreenshot) { names.append("photoScreenshot") }
    if subtypes.contains(.photoLive) { names.append("photoLive") }
    if subtypes.contains(.photoDepthEffect) { names.append("photoDepthEffect") }
    if subtypes.contains(.videoHighFrameRate) { names.append("videoHighFrameRate") }
    if subtypes.contains(.videoTimelapse) { names.append("videoTimelapse") }
    return names
  }

  private static let dateFormatter: DateFormatter = {
    let value = DateFormatter()
    value.dateStyle = .long
    value.timeStyle = .none
    return value
  }()
}

public enum IOSPhotoError: Error, Sendable, Equatable {
  case permissionDenied
}

@MainActor
public final class UIApplicationForegroundPresenter: NSObject, AgentForegroundPresenting,
  MFMessageComposeViewControllerDelegate, MFMailComposeViewControllerDelegate,
  @unchecked Sendable
{
  public static let shared = UIApplicationForegroundPresenter()

  public override init() { super.init() }

  public func presentMessageDraft(to: String, body: String) async -> Bool {
    guard isActive, MFMessageComposeViewController.canSendText(),
      let presenter = topViewController() else { return false }
    let controller = MFMessageComposeViewController()
    controller.messageComposeDelegate = self
    controller.recipients = [to]
    controller.body = body
    presenter.present(controller, animated: true)
    return true
  }

  public func presentMailDraft(to: String, subject: String, body: String) async -> Bool {
    guard isActive, MFMailComposeViewController.canSendMail(),
      let presenter = topViewController() else { return false }
    let controller = MFMailComposeViewController()
    controller.mailComposeDelegate = self
    controller.setToRecipients([to])
    controller.setSubject(subject)
    controller.setMessageBody(body, isHTML: false)
    presenter.present(controller, animated: true)
    return true
  }

  public func openPhone(number: String) async -> Bool {
    guard isActive else { return false }
    let allowed = CharacterSet(charactersIn: "+0123456789,;#*")
    let normalized = number.unicodeScalars.filter(allowed.contains).map(String.init).joined()
    guard !normalized.isEmpty, normalized.utf8.count <= 64,
      let url = URL(string: "tel:\(normalized)"),
      UIApplication.shared.canOpenURL(url) else { return false }
    return await withCheckedContinuation { continuation in
      UIApplication.shared.open(url, options: [:]) { continuation.resume(returning: $0) }
    }
  }

  public func openDirections(destination: String) async -> Bool {
    guard isActive else { return false }
    do {
      let placemarks = try await CLGeocoder().geocodeAddressString(destination)
      guard let placemark = placemarks.first else { return false }
      let item = MKMapItem(placemark: MKPlacemark(placemark: placemark))
      item.name = destination
      return item.openInMaps(launchOptions: [
        MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving,
      ])
    } catch {
      return false
    }
  }

  public func captureCameraImage() async -> AgentCameraCaptureResult {
    guard isActive else { return .unavailable }
    return await IOSCameraCaptureController.shared.capture()
  }

  public func messageComposeViewController(
    _ controller: MFMessageComposeViewController,
    didFinishWith result: MessageComposeResult
  ) {
    controller.dismiss(animated: true)
  }

  public func mailComposeController(
    _ controller: MFMailComposeViewController,
    didFinishWith result: MFMailComposeResult,
    error: Error?
  ) {
    controller.dismiss(animated: true)
  }

  private var isActive: Bool { UIApplication.shared.applicationState == .active }

  private func topViewController() -> UIViewController? {
    let scene = UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }
      .first { $0.activationState == .foregroundActive }
    let root = scene?.windows.first(where: \.isKeyWindow)?.rootViewController
    return top(from: root)
  }

  private func top(from controller: UIViewController?) -> UIViewController? {
    if let presented = controller?.presentedViewController { return top(from: presented) }
    if let navigation = controller as? UINavigationController { return top(from: navigation.visibleViewController) }
    if let tabs = controller as? UITabBarController { return top(from: tabs.selectedViewController) }
    return controller
  }
}

public struct IOSForegroundPresenter: AgentForegroundPresenting, Sendable {
  public init() {}

  @MainActor
  public func presentMessageDraft(to: String, body: String) async -> Bool {
    await UIApplicationForegroundPresenter.shared.presentMessageDraft(to: to, body: body)
  }

  @MainActor
  public func presentMailDraft(to: String, subject: String, body: String) async -> Bool {
    await UIApplicationForegroundPresenter.shared.presentMailDraft(to: to, subject: subject, body: body)
  }

  @MainActor
  public func openPhone(number: String) async -> Bool {
    await UIApplicationForegroundPresenter.shared.openPhone(number: number)
  }

  @MainActor
  public func openDirections(destination: String) async -> Bool {
    await UIApplicationForegroundPresenter.shared.openDirections(destination: destination)
  }

  @MainActor
  public func captureCameraImage() async -> AgentCameraCaptureResult {
    await UIApplicationForegroundPresenter.shared.captureCameraImage()
  }
}

@MainActor
private final class IOSCameraCaptureController: NSObject {
  static let shared = IOSCameraCaptureController()

  private var session: AVCaptureSession?
  private var photoOutput: AVCapturePhotoOutput?
  private var photoDelegate: IOSCameraPhotoDelegate?
  private var captureTask: Task<Void, Never>?
  private var timeoutTask: Task<Void, Never>?
  private var continuation: CheckedContinuation<AgentCameraCaptureResult, Never>?

  func capture() async -> AgentCameraCaptureResult {
#if targetEnvironment(simulator)
    return .unavailable
#else
    guard continuation == nil else { return .failed }
    let status = AVCaptureDevice.authorizationStatus(for: .video)
    let authorized: Bool
    switch status {
    case .authorized:
      authorized = true
    case .notDetermined:
      authorized = await AVCaptureDevice.requestAccess(for: .video)
    default:
      authorized = false
    }
    guard authorized else { return .permissionDenied }
    return await withCheckedContinuation { continuation in
      self.continuation = continuation
      performCapture()
    }
#endif
  }

  private func performCapture() {
    let session = AVCaptureSession()
    session.sessionPreset = .photo
    guard let device = AVCaptureDevice.default(
      .builtInWideAngleCamera,
      for: .video,
      position: .back
    ) ?? AVCaptureDevice.default(for: .video),
      let input = try? AVCaptureDeviceInput(device: device) else {
      finish(.unavailable)
      return
    }

    session.beginConfiguration()
    guard session.canAddInput(input) else {
      session.commitConfiguration()
      finish(.unavailable)
      return
    }
    session.addInput(input)
    let output = AVCapturePhotoOutput()
    guard session.canAddOutput(output) else {
      session.commitConfiguration()
      finish(.unavailable)
      return
    }
    session.addOutput(output)
    session.commitConfiguration()

    let delegate = IOSCameraPhotoDelegate(owner: self)
    self.session = session
    self.photoOutput = output
    self.photoDelegate = delegate
    session.startRunning()
    captureTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 600_000_000)
      guard !Task.isCancelled,
        let self,
        let output = self.photoOutput,
        let delegate = self.photoDelegate else { return }
      output.capturePhoto(with: AVCapturePhotoSettings(), delegate: delegate)
    }
    timeoutTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 15_000_000_000)
      guard !Task.isCancelled else { return }
      self?.finish(.failed)
    }
  }

  fileprivate func didCapture(data: Data?, error: Error?) {
    guard error == nil, let data, let image = UIImage(data: data) else {
      finish(.failed)
      return
    }
    let width = image.cgImage?.width ?? Int(image.size.width * image.scale)
    let height = image.cgImage?.height ?? Int(image.size.height * image.scale)
    finish(.captured(pixelWidth: width, pixelHeight: height, bytes: data.count))
  }

  private func finish(_ result: AgentCameraCaptureResult) {
    captureTask?.cancel()
    captureTask = nil
    timeoutTask?.cancel()
    timeoutTask = nil
    session?.stopRunning()
    session = nil
    photoOutput = nil
    photoDelegate = nil
    continuation?.resume(returning: result)
    continuation = nil
  }
}

@MainActor
private final class IOSCameraPhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
  private unowned let owner: IOSCameraCaptureController

  init(owner: IOSCameraCaptureController) {
    self.owner = owner
  }

  nonisolated func photoOutput(
    _ output: AVCapturePhotoOutput,
    didFinishProcessingPhoto photo: AVCapturePhoto,
    error: Error?
  ) {
    let data = photo.fileDataRepresentation()
    Task { @MainActor [owner] in
      owner.didCapture(data: data, error: error)
    }
  }
}

public actor AppleLocationAndMapService {
  private let locationProvider: any AgentLocationProviding

  public init(locationProvider: any AgentLocationProviding = IOSLocationProvider()) {
    self.locationProvider = locationProvider
  }

  public func currentLocation() async -> AgentServiceResponse {
    switch await locationProvider.currentCoordinate() {
    case let .success(coordinate):
      return .success(
        "Current location resolved.",
        payload: [
          "latitude": .number(coordinate.latitude),
          "longitude": .number(coordinate.longitude),
          "accuracy": .string("approximately-100m"),
        ]
      )
    case let .failure(failure): return Self.locationFailure(failure)
    }
  }

  public func search(arguments: AgentJSONArguments) async -> AgentServiceResponse {
    guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 1_000) else {
      return .failed("The map search query is invalid.", code: "maps_invalid_query")
    }
    let request = MKLocalSearch.Request()
    request.naturalLanguageQuery = query
    if case let .success(coordinate) = await locationProvider.currentCoordinate() {
      request.region = MKCoordinateRegion(
        center: .init(latitude: coordinate.latitude, longitude: coordinate.longitude),
        latitudinalMeters: 25_000,
        longitudinalMeters: 25_000
      )
    }
    do {
      let response = try await MKLocalSearch(request: request).start()
      let items = response.mapItems.prefix(10)
      let values: [AgentJSONValue] = items.map { item in
        [
          "name": .string(item.name ?? "Unnamed"),
          "latitude": .number(item.placemark.coordinate.latitude),
          "longitude": .number(item.placemark.coordinate.longitude),
          "address": .string(item.placemark.title ?? ""),
        ]
      }
      let text = items.isEmpty
        ? "No nearby places were found."
        : items.map { "- \($0.name ?? "Unnamed"): \($0.placemark.title ?? "")" }.joined(separator: "\n")
      return .success(text, payload: ["places": .array(values)])
    } catch {
      return .failed("Maps search could not be completed.", code: "maps_search_failed")
    }
  }

  public func weather(arguments: AgentJSONArguments) async -> AgentServiceResponse {
#if canImport(WeatherKit)
    let requested = AgentToolInput.optionalString("location", in: arguments, maximumBytes: 500)
      ?? AgentToolInput.optionalString("city", in: arguments, maximumBytes: 500)
      ?? ""
    let location: CLLocation
    if requested.isEmpty || ["here", "current", "current location"].contains(requested.lowercased()) {
      switch await locationProvider.currentCoordinate() {
      case let .success(coordinate):
        location = CLLocation(latitude: coordinate.latitude, longitude: coordinate.longitude)
      case let .failure(failure): return Self.locationFailure(failure)
      }
    } else {
      do {
        guard let resolved = try await CLGeocoder().geocodeAddressString(requested).first?.location else {
          return .failed("Weather location could not be resolved.", code: "weather_geocode_failed")
        }
        location = resolved
      } catch {
        return .failed("Weather location could not be resolved.", code: "weather_geocode_failed")
      }
    }
    do {
      let weather = try await WeatherService.shared.weather(for: location)
      let current = weather.currentWeather
      let temperature = current.temperature.converted(to: .celsius).value
      let apparent = current.apparentTemperature.converted(to: .celsius).value
      let wind = current.wind.speed.converted(to: .kilometersPerHour).value
      return .success(
        String(format: "%@ · %.0f°C · feels like %.0f°C · humidity %.0f%% · wind %.0f km/h", current.condition.description, temperature, apparent, current.humidity * 100, wind),
        payload: [
          "condition": .string(current.condition.description),
          "temperatureCelsius": .number(temperature),
          "apparentTemperatureCelsius": .number(apparent),
          "humidityPercent": .number(current.humidity * 100),
          "windKilometersPerHour": .number(wind),
          "provider": .string("WeatherKit"),
        ]
      )
    } catch {
      return .unavailable("WeatherKit is unavailable for this build, account, or location.", code: "weatherkit_unavailable")
    }
#else
    return .unavailable("WeatherKit is unavailable in this SDK.", code: "weatherkit_unavailable")
#endif
  }

  private static func locationFailure(_ failure: AgentLocationFailure) -> AgentServiceResponse {
    switch failure {
    case .permissionNotDetermined:
      return .denied("Location permission has not been requested.", code: "location_permission_not_determined")
    case .denied: return .denied("Location permission is denied.", code: "location_permission_denied")
    case .restricted: return .denied("Location permission is restricted.", code: "location_permission_restricted")
    case .timedOut: return .failed("Location lookup timed out.", code: "location_timed_out")
    case .busy: return .failed("Another location lookup is in progress.", code: "location_busy")
    case .unavailable: return .unavailable("Location services are unavailable.", code: "location_unavailable")
    case .providerFailed: return .failed("Location services could not resolve a coordinate.", code: "location_failed")
    }
  }
}

public actor AppleHealthService {
  private let store = HKHealthStore()

  public init() {}

  public func summary() async -> AgentServiceResponse {
    guard HKHealthStore.isHealthDataAvailable() else {
      return .unavailable("Health data is unavailable on this device.", code: "health_unavailable")
    }
    let now = Date()
    let start = Calendar.current.startOfDay(for: now)
    let stepType = HKQuantityType(.stepCount)
    let distanceType = HKQuantityType(.distanceWalkingRunning)
    let energyType = HKQuantityType(.activeEnergyBurned)
    let heartType = HKQuantityType(.heartRate)
    let sleepType = HKCategoryType(.sleepAnalysis)

    async let steps = sum(stepType, unit: .count(), start: start, end: now)
    async let distance = sum(distanceType, unit: .meter(), start: start, end: now)
    async let energy = sum(energyType, unit: .kilocalorie(), start: start, end: now)
    async let heart = average(
      heartType,
      unit: HKUnit.count().unitDivided(by: .minute()),
      start: now.addingTimeInterval(-86_400),
      end: now
    )
    async let sleep = sleepHours(
      sleepType,
      start: now.addingTimeInterval(-36 * 60 * 60),
      end: now
    )
    let values = await (steps, distance, energy, heart, sleep)
    guard values.0 != nil || values.1 != nil || values.2 != nil || values.3 != nil
      || values.4 != nil else {
      return .denied("No authorized Health summary data is available.", code: "health_data_unavailable")
    }
    var payload: AgentJSONArguments = ["period": "today"]
    var text: [String] = []
    if let value = values.0 { payload["steps"] = .number(value); text.append("\(Int(value)) steps") }
    if let value = values.1 { payload["distanceMeters"] = .number(value); text.append(String(format: "%.2f km", value / 1_000)) }
    if let value = values.2 { payload["activeKilocalories"] = .number(value); text.append("\(Int(value)) kcal") }
    if let value = values.3 { payload["averageHeartRate"] = .number(value); text.append("\(Int(value)) bpm average") }
    if let value = values.4 {
      payload["sleepHours"] = .number(value)
      text.append(String(format: "%.1f h sleep", value))
    }
    return .success(text.joined(separator: " · "), payload: .object(payload))
  }

  private func sum(_ type: HKQuantityType, unit: HKUnit, start: Date, end: Date) async -> Double? {
    await withCheckedContinuation { continuation in
      let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
      let query = HKStatisticsQuery(
        quantityType: type,
        quantitySamplePredicate: predicate,
        options: .cumulativeSum
      ) { _, statistics, _ in
        continuation.resume(returning: statistics?.sumQuantity()?.doubleValue(for: unit))
      }
      store.execute(query)
    }
  }

  private func average(_ type: HKQuantityType, unit: HKUnit, start: Date, end: Date) async -> Double? {
    await withCheckedContinuation { continuation in
      let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: .strictStartDate)
      let query = HKStatisticsQuery(
        quantityType: type,
        quantitySamplePredicate: predicate,
        options: .discreteAverage
      ) { _, statistics, _ in
        continuation.resume(returning: statistics?.averageQuantity()?.doubleValue(for: unit))
      }
      store.execute(query)
    }
  }

  private func sleepHours(
    _ type: HKCategoryType,
    start: Date,
    end: Date
  ) async -> Double? {
    await withCheckedContinuation { continuation in
      let predicate = HKQuery.predicateForSamples(
        withStart: start,
        end: end,
        options: .strictEndDate
      )
      let query = HKSampleQuery(
        sampleType: type,
        predicate: predicate,
        limit: 512,
        sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]
      ) { _, samples, _ in
        let asleepValues: Set<Int> = [
          HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue,
          HKCategoryValueSleepAnalysis.asleepCore.rawValue,
          HKCategoryValueSleepAnalysis.asleepDeep.rawValue,
          HKCategoryValueSleepAnalysis.asleepREM.rawValue,
        ]
        let intervals = (samples as? [HKCategorySample] ?? [])
          .filter { asleepValues.contains($0.value) }
          .map { (max(start, $0.startDate), min(end, $0.endDate)) }
          .filter { $0.1 > $0.0 }
          .sorted { $0.0 < $1.0 }
        var merged: [(Date, Date)] = []
        for interval in intervals {
          if let last = merged.last, interval.0 <= last.1 {
            _ = merged.removeLast()
            merged.append((last.0, max(last.1, interval.1)))
          } else {
            merged.append(interval)
          }
        }
        let seconds = merged.reduce(0) { $0 + $1.1.timeIntervalSince($1.0) }
        continuation.resume(returning: seconds > 0 ? seconds / 3_600 : nil)
      }
      store.execute(query)
    }
  }
}

public final class AppleMotionService: @unchecked Sendable {
  private let pedometer = CMPedometer()
  private let activityManager = CMMotionActivityManager()

  public init() {}

  public func activity() async -> AgentServiceResponse {
    guard CMMotionActivityManager.isActivityAvailable() else {
      return .unavailable("Motion activity is unavailable on this device.", code: "motion_unavailable")
    }
    let status = CMMotionActivityManager.authorizationStatus()
    guard status != .denied, status != .restricted else {
      return .denied("Motion access is denied or restricted.", code: "motion_permission_denied")
    }
    let end = Date()
    let start = Calendar.current.startOfDay(for: end)
    async let steps = pedometerData(start: start, end: end)
    async let activities = activityData(start: start, end: end)
    let values = await (steps, activities)
    guard values.0 != nil || !values.1.isEmpty else {
      return .success("No motion activity was recorded today.", payload: ["activities": []])
    }
    var payload: AgentJSONArguments = [:]
    var text: [String] = []
    if let pedometer = values.0 {
      payload["steps"] = .number(Double(pedometer.steps))
      text.append("\(pedometer.steps) steps")
      if let distance = pedometer.distance {
        payload["distanceMeters"] = .number(distance)
        text.append(String(format: "%.2f km", distance / 1_000))
      }
      if let floors = pedometer.floors {
        payload["floorsAscended"] = .number(Double(floors))
        text.append("\(floors) floors")
      }
    }
    payload["activities"] = .array(values.1.map {
      ["type": .string($0.label), "minutes": .number(Double($0.minutes))]
    })
    if !values.1.isEmpty {
      text.append(values.1.map { "\($0.minutes)m \($0.label)" }.joined(separator: ", "))
    }
    return .success(text.joined(separator: " · "), payload: .object(payload))
  }

  private func pedometerData(
    start: Date,
    end: Date
  ) async -> (steps: Int, distance: Double?, floors: Int?)? {
    guard CMPedometer.isStepCountingAvailable() else { return nil }
    return await withCheckedContinuation { continuation in
      pedometer.queryPedometerData(from: start, to: end) { data, _ in
        guard let data else { continuation.resume(returning: nil); return }
        continuation.resume(returning: (
          data.numberOfSteps.intValue,
          data.distance?.doubleValue,
          data.floorsAscended?.intValue
        ))
      }
    }
  }

  private func activityData(start: Date, end: Date) async -> [(label: String, minutes: Int)] {
    await withCheckedContinuation { continuation in
      activityManager.queryActivityStarting(from: start, to: end, to: .main) { activities, _ in
        guard let activities else { continuation.resume(returning: []); return }
        var totals: [String: TimeInterval] = [:]
        for (index, activity) in activities.enumerated() {
          let next = index + 1 < activities.count ? activities[index + 1].startDate : end
          let duration = max(0, next.timeIntervalSince(activity.startDate))
          let label: String?
          if activity.walking { label = "walking" }
          else if activity.running { label = "running" }
          else if activity.cycling { label = "cycling" }
          else if activity.automotive { label = "driving" }
          else if activity.stationary { label = "stationary" }
          else { label = nil }
          if let label { totals[label, default: 0] += duration }
        }
        continuation.resume(returning: totals.map {
          (label: $0.key, minutes: Int($0.value / 60))
        }.filter { $0.minutes > 0 }.sorted { $0.minutes > $1.minutes })
      }
    }
  }
}
#endif
