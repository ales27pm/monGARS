import CryptoKit
import Foundation
import MonGARSCoreML

enum AgentOpaqueProfileScope {
  static func make(rawOwnerID: String) -> String? {
    let value = rawOwnerID.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty, value.utf8.count <= 4_096 else { return nil }
    let digest = SHA256.hash(data: Data(value.utf8))
      .map { String(format: "%02x", $0) }
      .joined()
    return "profile.\(digest)"
  }
}

public struct ScopedIOSAgentToolExecutor: AgentToolExecuting, Sendable {
  private let base: IOSAgentToolExecutor
  private let opaqueScope: String

  /// The raw owner identifier is hashed before it reaches storage, so profile
  /// separation does not persist an email, username, or account token.
  public init(base: IOSAgentToolExecutor = .shared, rawOwnerID: String) {
    self.base = base
    self.opaqueScope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID)
      ?? "invalid.\(UUID().uuidString.lowercased())"
  }

  public func availableToolIDs() async -> Set<AgentToolID> {
    await base.availableToolIDs(profileScope: opaqueScope)
  }

  /// Recovers one notification-tap handoff for the signed-in owner. This is a
  /// foreground lookup only; it does not start model execution.
  public func pendingTrigger() async -> AgentPendingTriggerHandoff? {
    await base.pendingTrigger(profileScope: opaqueScope)
  }

  public func acknowledgePendingTrigger(id: UUID) async -> Bool {
    await base.acknowledgePendingTrigger(id: id, profileScope: opaqueScope)
  }

  /// Resolves an exact UUID or exact title for the active profile without
  /// executing or mutating the stored trigger.
  public func resolveStoredTrigger(_ selector: String) async -> AgentPendingTriggerHandoff? {
    await base.resolveStoredTrigger(selector: selector, profileScope: opaqueScope)
  }

  public func execute(invocation: AgentToolInvocation) async -> AgentToolResult {
    await base.execute(invocation: invocation, profileScope: opaqueScope)
  }
}

public actor IOSAgentToolExecutor: AgentToolExecuting {
  public static let shared = IOSAgentToolExecutor()

  private let graphService: (any MicrosoftGraphServing)?
  private let webService: (any AgentWebServing)?
  private let fileService: SafeLocalFileService
  private let knowledgeStore: AgentLocalKnowledgeStore
  private let scopeProvider: any AgentMemoryScopeProviding
  private let triggerScheduler: any AgentTriggerScheduling
  private let alarmService: any AgentAlarmServing
  private let presenter: (any AgentForegroundPresenting)?
  private let photoProvider: (any AgentPhotoMetadataProviding)?

#if os(iOS)
  private let productivityService: AppleProductivityService
  private let contactsService: AppleContactsService
  private let locationAndMapService: AppleLocationAndMapService
  private let healthService: AppleHealthService
  private let motionService: AppleMotionService
#endif

  public init(
    graphService: (any MicrosoftGraphServing)? = URLSessionMicrosoftGraphService(
      tokenProvider: MicrosoftGraphOAuthTokenProvider.shared
    ),
    webService: (any AgentWebServing)? = PublicWebToolService(),
    scopeProvider: any AgentMemoryScopeProviding = StaticAgentMemoryScopeProvider(),
    importedFilesRoot: URL? = nil,
    protectedStateRoot: URL? = nil,
    presenter: (any AgentForegroundPresenting)? = IOSAgentToolExecutor.defaultPresenter,
    photoProvider: (any AgentPhotoMetadataProviding)? = IOSAgentToolExecutor.defaultPhotoProvider,
    triggerScheduler: (any AgentTriggerScheduling)? = nil,
    alarmService: any AgentAlarmServing = IOSAlarmService()
  ) {
    let directories = Self.defaultDirectories()
    let stateRoot = protectedStateRoot ?? directories.state
    self.graphService = graphService
    self.webService = webService
    self.scopeProvider = scopeProvider
    self.fileService = .init(rootDirectory: importedFilesRoot ?? directories.importedFiles)
    self.knowledgeStore = .init(stateURL: stateRoot.appendingPathComponent("knowledge.json"))
    self.presenter = presenter
    self.photoProvider = photoProvider
    self.triggerScheduler = triggerScheduler ?? LocalNotificationAgentTriggerScheduler(
      stateURL: stateRoot.appendingPathComponent("triggers.json")
    )
    self.alarmService = alarmService
#if os(iOS)
    self.productivityService = .init()
    self.contactsService = .init()
    self.locationAndMapService = .init()
    self.healthService = .init()
    self.motionService = .init()
#endif
  }

  /// Returns tools backed by an implementation on this SDK/runtime. Permission,
  /// authentication, and approval state remain separate runtime checks.
  public func availableToolIDs() async -> Set<AgentToolID> {
    await availableToolIDs(profileScope: nil)
  }

  func availableToolIDs(profileScope: String?) async -> Set<AgentToolID> {
    var result = Set(AgentHostOperation.canonicalDispatchTable.keys)
    if profileScope == nil || graphService == nil {
      result.subtract(Self.ids(for: AgentHostOperation.outlookOperations))
    } else if let graphService,
      !(await graphService.isConfigured(profileScope: profileScope)) {
      let authenticatedOperations = AgentHostOperation.outlookOperations.filter {
        $0 != .outlookStatus
      }
      result.subtract(Self.ids(for: authenticatedOperations))
    }
    if webService == nil {
      result.remove("web.search")
      result.remove("web.fetch")
    } else if webService?.supportsPublicFetch != true {
      result.remove("web.fetch")
    }
    if presenter == nil {
      result.subtract(Self.ids(for: AgentHostOperation.foregroundUIOperations))
    }
    if photoProvider == nil {
      result.remove("photos.search")
      result.remove("rag.index_photos")
    }
    if !fileService.hasReadableDocuments() {
      result.remove("files.read")
      result.remove("rag.index_files")
    }
#if os(iOS)
#if !canImport(WeatherKit)
    result.remove("weather")
#endif
#if canImport(AlarmKit)
    if #available(iOS 26.0, *) {
      if !IOSAlarmService.hasUsageDescription {
        result.subtract(Self.ids(for: AgentHostOperation.alarmOperations))
      } else if !IOSAlarmService.hasLiveActivityConfiguration {
        result.subtract(Self.ids(for: [
          .alarmSchedule, .alarmCountdown, .alarmPause, .alarmResume, .alarmSnooze,
        ]))
      }
    } else {
      result.subtract(Self.ids(for: AgentHostOperation.alarmOperations))
    }
#else
    result.subtract(Self.ids(for: AgentHostOperation.alarmOperations))
#endif
#else
    let appleOnly: Set<AgentHostOperation> = [
      .calendarCreate, .calendarList, .remindersCreate, .remindersList, .contactsSearch,
      .messagesDraft, .mailDraft, .phoneCall, .locationCurrent, .weather, .mapsDirections,
      .mapsSearch, .photosSearch, .cameraCapture, .healthSummary, .motionActivity,
      .ragIndexPhotos, .triggerCreate, .triggerList, .triggerCancel,
    ].union(AgentHostOperation.alarmOperations)
    result.subtract(Self.ids(for: appleOnly))
#endif
    return result
  }

  /// Peeks at one notification-tap handoff for the signed-in owner. The prompt
  /// remains protected until the UI explicitly runs or dismisses it.
  public func pendingTrigger(rawOwnerID: String) async -> AgentPendingTriggerHandoff? {
    guard let scope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else { return nil }
    return await triggerScheduler.pendingHandoff(scope: scope)
  }

  public func acknowledgePendingTrigger(
    rawOwnerID: String,
    id: UUID
  ) async -> Bool {
    guard let scope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else { return false }
    return await triggerScheduler.acknowledgePendingHandoff(id: id, scope: scope)
  }

  public func resolveStoredTrigger(
    rawOwnerID: String,
    selector: String
  ) async -> AgentPendingTriggerHandoff? {
    guard let scope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else { return nil }
    return await triggerScheduler.resolveHandoff(selector: selector, scope: scope)
  }

  func pendingTrigger(profileScope: String) async -> AgentPendingTriggerHandoff? {
    await triggerScheduler.pendingHandoff(scope: profileScope)
  }

  func acknowledgePendingTrigger(id: UUID, profileScope: String) async -> Bool {
    await triggerScheduler.acknowledgePendingHandoff(id: id, scope: profileScope)
  }

  func resolveStoredTrigger(
    selector: String,
    profileScope: String
  ) async -> AgentPendingTriggerHandoff? {
    await triggerScheduler.resolveHandoff(selector: selector, scope: profileScope)
  }

  public func execute(invocation: AgentToolInvocation) async -> AgentToolResult {
    await execute(invocation: invocation, profileScope: nil)
  }

  public func execute(
    invocation: AgentToolInvocation,
    profileScope: String?
  ) async -> AgentToolResult {
    if Task.isCancelled {
      return AgentToolResultFactory.make(
        invocation: invocation,
        response: .init(status: .cancelled, text: "Tool execution was cancelled.", errorCode: "tool_cancelled")
      )
    }
    let canonicalID = AgentToolNormalizer.canonicalToolID(invocation.toolID.rawValue)
    guard let operation = AgentHostOperation.canonicalDispatchTable[canonicalID] else {
      return AgentToolResultFactory.make(
        invocation: invocation,
        response: .unavailable("No host implementation is registered for \(canonicalID.rawValue).", code: "tool_unavailable")
      )
    }
    guard let definition = AgentToolCatalog.definition(for: canonicalID.rawValue) else {
      return AgentToolResultFactory.make(
        invocation: invocation,
        response: .unavailable("The requested tool is not in the canonical catalog.", code: "tool_not_catalogued")
      )
    }
    if invocation.mode == .background, !definition.supportsBackgroundExecution {
      return AgentToolResultFactory.make(
        invocation: invocation,
        response: .denied("This tool cannot execute in the background.", code: "background_execution_denied")
      )
    }
    let arguments: AgentJSONArguments
    do {
      arguments = try AgentToolNormalizer.normalizedArguments(
        for: canonicalID,
        arguments: invocation.arguments
      )
    } catch {
      return AgentToolResultFactory.make(
        invocation: invocation,
        response: .failed("The tool arguments contain conflicting aliases.", code: "tool_argument_alias_conflict")
      )
    }

    let response = await dispatch(
      operation: operation,
      arguments: arguments,
      mode: invocation.mode,
      profileScope: profileScope
    )
    return AgentToolResultFactory.make(invocation: invocation, response: response)
  }

  private func dispatch(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments,
    mode: AgentExecutionMode,
    profileScope: String?
  ) async -> AgentServiceResponse {
    let activeScope: String
    if let profileScope {
      activeScope = profileScope
    } else {
      activeScope = await scopeProvider.currentScope()
    }
    switch operation {
    case .calendarCreate:
#if os(iOS)
      return await productivityService.createCalendarEvent(arguments: arguments)
#else
      return Self.iOSUnavailable(operation)
#endif
    case .calendarList:
#if os(iOS)
      return await productivityService.listCalendarEvents()
#else
      return Self.iOSUnavailable(operation)
#endif
    case .remindersCreate:
#if os(iOS)
      return await productivityService.createReminder(arguments: arguments)
#else
      return Self.iOSUnavailable(operation)
#endif
    case .remindersList:
#if os(iOS)
      return await productivityService.listReminders()
#else
      return Self.iOSUnavailable(operation)
#endif
    case .contactsSearch:
#if os(iOS)
      return await contactsService.search(arguments: arguments)
#else
      return Self.iOSUnavailable(operation)
#endif
    case .messagesDraft:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      guard let to = AgentToolInput.requiredString("to", in: arguments, maximumBytes: 500),
        let body = AgentToolInput.requiredString("body", in: arguments, maximumBytes: 16_000) else {
        return .failed("The message recipient or body is invalid.", code: "message_invalid_arguments")
      }
      guard let presenter, await presenter.presentMessageDraft(to: to, body: body) else {
        return .unavailable("The system message composer could not be presented.", code: "message_composer_unavailable")
      }
      return .success("The system message draft was presented for user review.", payload: ["presented": true])
    case .mailDraft:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      guard let to = AgentToolInput.requiredString("to", in: arguments, maximumBytes: 320),
        let body = AgentToolInput.requiredString("body", in: arguments, maximumBytes: 100_000) else {
        return .failed("The email recipient or body is invalid.", code: "mail_invalid_arguments")
      }
      let subject = AgentToolInput.optionalString("subject", in: arguments, maximumBytes: 1_000) ?? ""
      guard let presenter, await presenter.presentMailDraft(to: to, subject: subject, body: body) else {
        return .unavailable("The system mail composer could not be presented.", code: "mail_composer_unavailable")
      }
      return .success("The system email draft was presented for user review.", payload: ["presented": true])
    case .outlookStatus, .outlookFoldersList, .outlookMessagesList, .outlookMessagesSearch,
      .outlookMessageRead, .outlookAttachmentsList, .outlookDraftCreate, .outlookMailSend,
      .outlookMessageMarkRead, .outlookMessageMarkUnread, .outlookMessageMove,
      .outlookMessageArchive, .outlookMessageDelete, .outlookMessageReply,
      .outlookMessageReplyAll, .outlookMessageForward:
      guard let graphService else {
        return .unavailable("Outlook is unavailable because no Microsoft Graph service is configured.", code: "outlook_service_unavailable")
      }
      return await graphService.perform(
        operation: operation,
        arguments: arguments,
        profileScope: activeScope
      )
    case .phoneCall:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      guard let number = AgentToolInput.requiredString("number", in: arguments, maximumBytes: 64) else {
        return .failed("The phone number is invalid.", code: "phone_invalid_number")
      }
      guard let presenter, await presenter.openPhone(number: number) else {
        return .unavailable("The phone dialer could not be opened.", code: "phone_unavailable")
      }
      return .success("The phone dialer was opened for user confirmation.", payload: ["opened": true])
    case .locationCurrent:
#if os(iOS)
      return await locationAndMapService.currentLocation()
#else
      return Self.iOSUnavailable(operation)
#endif
    case .weather:
#if os(iOS)
      return await locationAndMapService.weather(arguments: arguments)
#else
      return Self.iOSUnavailable(operation)
#endif
    case .mapsDirections:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      guard let destination = AgentToolInput.requiredString("destination", in: arguments, maximumBytes: 1_000) else {
        return .failed("The directions destination is invalid.", code: "maps_invalid_destination")
      }
      guard let presenter, await presenter.openDirections(destination: destination) else {
        return .unavailable("Apple Maps could not open directions for that destination.", code: "maps_directions_unavailable")
      }
      return .success("Apple Maps directions were opened.", payload: ["opened": true])
    case .mapsSearch:
#if os(iOS)
      return await locationAndMapService.search(arguments: arguments)
#else
      return Self.iOSUnavailable(operation)
#endif
    case .photosSearch:
#if os(iOS)
      guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 500) else {
        return .failed("The photo search query is invalid.", code: "photos_invalid_query")
      }
      guard let photoProvider else {
        return .unavailable("Photo metadata search is not configured.", code: "photos_unavailable")
      }
      do {
        let photos = try await photoProvider.searchMetadata(query: query, limit: 20)
        let values = Self.photoPayload(photos)
        let text = photos.isEmpty
          ? "No matching photo metadata was found."
          : photos.map { "- \($0.filename ?? "Photo") · \($0.createdAt.map(Self.iso8601.string) ?? "unknown date")" }
            .joined(separator: "\n")
        return .success(text, payload: ["photos": .array(values)])
      } catch {
        return .denied("Photo metadata access is unavailable or denied.", code: "photos_permission_denied")
      }
#else
      return Self.iOSUnavailable(operation)
#endif
    case .cameraCapture:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      guard let presenter else {
        return .unavailable("Camera capture is not configured.", code: "camera_unavailable")
      }
      switch await presenter.captureCameraImage() {
      case let .captured(pixelWidth, pixelHeight, bytes):
        return .success(
          "Captured an image with the device camera.",
          payload: [
            "captured": true,
            "pixelWidth": .number(Double(pixelWidth)),
            "pixelHeight": .number(Double(pixelHeight)),
            "bytes": .number(Double(bytes)),
          ]
        )
      case .permissionDenied:
        return .denied("Camera permission is denied.", code: "camera_permission_denied")
      case .unavailable:
        return .unavailable("No camera device is available.", code: "camera_unavailable")
      case .failed:
        return .failed("The camera could not capture an image.", code: "camera_capture_failed")
      }
    case .healthSummary:
#if os(iOS)
      return await healthService.summary()
#else
      return Self.iOSUnavailable(operation)
#endif
    case .motionActivity:
#if os(iOS)
      return await motionService.activity()
#else
      return Self.iOSUnavailable(operation)
#endif
    case .webSearch:
      guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 1_000) else {
        return .failed("The web search query is invalid.", code: "web_invalid_query")
      }
      guard let webService else {
        return .unavailable("Public web search is not configured.", code: "web_unavailable")
      }
      return await webService.search(query: query)
    case .webFetch:
      guard let url = AgentToolInput.requiredString("url", in: arguments, maximumBytes: 4_096) else {
        return .failed("The web URL is invalid.", code: "web_invalid_url")
      }
      guard let webService else {
        return .unavailable("Public web fetch is not configured.", code: "web_unavailable")
      }
      return await webService.fetch(url: url)
    case .filesRead:
      guard let name = AgentToolInput.requiredString("name", in: arguments, maximumBytes: 1_024) else {
        return .failed("The imported file name is invalid.", code: "file_invalid_name")
      }
      return fileService.read(name: name)
    case .memorySave:
      guard let content = AgentToolInput.requiredString("content", in: arguments),
        let kind = AgentToolInput.requiredString("kind", in: arguments, maximumBytes: 64) else {
        return .failed("The memory content or kind is invalid.", code: "memory_invalid_arguments")
      }
      return await knowledgeStore.saveMemory(
        content: content,
        kind: kind,
        scope: activeScope
      )
    case .memoryRecall:
      guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 4_000) else {
        return .failed("The memory query is invalid.", code: "memory_invalid_query")
      }
      return await knowledgeStore.recallMemory(
        query: query,
        scope: activeScope
      )
    case .ragSearch:
      guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 4_000) else {
        return .failed("The local search query is invalid.", code: "rag_invalid_query")
      }
      return await knowledgeStore.search(
        query: query,
        sourceScope: AgentToolInput.optionalString("sourceScope", in: arguments, maximumBytes: 32) ?? "all",
        scope: activeScope,
        limit: AgentToolInput.integer("limit", in: arguments, range: 1...20) ?? 8
      )
    case .ragIndexFiles:
      switch fileService.documentsForIndexing() {
      case let .success(documents):
        return await knowledgeStore.indexDocuments(documents, scope: activeScope)
      case .failure(.rootUnavailable):
        return .unavailable(
          "The imported document directory is unavailable.",
          code: "file_import_root_unavailable"
        )
      case .failure(.enumerationFailed):
        return .failed(
          "The imported document directory could not be enumerated safely.",
          code: "file_enumeration_failed"
        )
      }
    case .ragIndexPhotos:
#if os(iOS)
      guard let months = AgentToolInput.integer("months", in: arguments, range: 1...120),
        let start = Calendar.current.date(byAdding: .month, value: -months, to: Date()) else {
        return .failed("The photo index range is invalid.", code: "rag_invalid_photo_range")
      }
      guard let photoProvider else {
        return .unavailable("Photo metadata indexing is not configured.", code: "photos_unavailable")
      }
      do {
        let photos = try await photoProvider.metadataSince(start, limit: 5_000)
        return await knowledgeStore.indexPhotos(
          photos,
          scope: activeScope
        )
      } catch {
        return .denied("Photo metadata access is unavailable or denied.", code: "photos_permission_denied")
      }
#else
      return Self.iOSUnavailable(operation)
#endif
    case .triggerCreate:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      return await triggerScheduler.create(
        arguments: arguments,
        scope: activeScope
      )
    case .triggerList:
      return await triggerScheduler.list(scope: activeScope)
    case .triggerCancel:
      guard mode == .foreground else { return Self.backgroundUIDenial(operation) }
      return await triggerScheduler.cancel(
        id: AgentToolInput.optionalString("id", in: arguments, maximumBytes: 64),
        title: AgentToolInput.optionalString("title", in: arguments, maximumBytes: 500),
        scope: activeScope
      )
    case .alarmAuthorizationStatus, .alarmRequestAuthorization, .alarmSchedule,
      .alarmCountdown, .alarmList, .alarmPause, .alarmResume, .alarmStop,
      .alarmSnooze, .alarmCancel:
      return await alarmService.execute(operation: operation, arguments: arguments)
    }
  }

  private static func ids(for operations: Set<AgentHostOperation>) -> Set<AgentToolID> {
    Set(AgentHostOperation.canonicalDispatchTable.compactMap { key, value in
      operations.contains(value) ? key : nil
    })
  }

  private static func defaultDirectories() -> (state: URL, importedFiles: URL) {
    let applicationSupport = FileManager.default.urls(
      for: .applicationSupportDirectory,
      in: .userDomainMask
    ).first ?? FileManager.default.temporaryDirectory
    let documents = FileManager.default.urls(
      for: .documentDirectory,
      in: .userDomainMask
    ).first ?? applicationSupport
    return (
      applicationSupport.appendingPathComponent("MonGARSAgentTools", isDirectory: true),
      documents.appendingPathComponent("ImportedDocuments", isDirectory: true)
    )
  }

  private static var defaultPresenter: (any AgentForegroundPresenting)? {
#if os(iOS)
    return IOSForegroundPresenter()
#else
    return nil
#endif
  }

  private static var defaultPhotoProvider: (any AgentPhotoMetadataProviding)? {
#if os(iOS)
    return IOSPhotoMetadataProvider()
#else
    return nil
#endif
  }

  private static func iOSUnavailable(_ operation: AgentHostOperation) -> AgentServiceResponse {
    .unavailable("\(operation.rawValue) requires an iOS host runtime.", code: "ios_runtime_unavailable")
  }

  private static func backgroundUIDenial(_ operation: AgentHostOperation) -> AgentServiceResponse {
    .denied("\(operation.rawValue) requires an active foreground scene.", code: "background_ui_denied")
  }

  private static let iso8601 = ISO8601DateFormatter()

  private static func photoPayload(_ photos: [AgentPhotoMetadata]) -> [AgentJSONValue] {
    photos.map { photo in
      var item: AgentJSONArguments = ["localIdentifier": .string(photo.localIdentifier)]
      if let filename = photo.filename { item["filename"] = .string(filename) }
      if let date = photo.createdAt { item["createdAt"] = .string(iso8601.string(from: date)) }
      if let latitude = photo.latitude { item["latitude"] = .number(latitude) }
      if let longitude = photo.longitude { item["longitude"] = .number(longitude) }
      if let mediaType = photo.mediaType { item["mediaType"] = .string(mediaType) }
      if !photo.mediaSubtypes.isEmpty {
        item["mediaSubtypes"] = .array(photo.mediaSubtypes.map(AgentJSONValue.string))
      }
      if let isFavorite = photo.isFavorite { item["isFavorite"] = .bool(isFavorite) }
      if let width = photo.pixelWidth { item["pixelWidth"] = .number(Double(width)) }
      if let height = photo.pixelHeight { item["pixelHeight"] = .number(Double(height)) }
      if let token = photo.displayToken { item["displayToken"] = .string(token) }
      if let query = photo.queryMatched { item["queryMatched"] = .string(query) }
      return .object(item)
    }
  }
}
