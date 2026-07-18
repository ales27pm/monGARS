import Foundation
import MonGARSCoreML
import os.log
import React
import UIKit

private enum CoreMLBridgeEvent: String, CaseIterable {
  case status = "onCoreMLStatus"
  case downloadProgress = "onCoreMLDownloadProgress"
  case generation = "onCoreMLGeneration"
  case complete = "onCoreMLComplete"
  case error = "onCoreMLError"
}

private enum CoreMLBridgeOperation: String, Sendable {
  case prepare
  case generate
  case unload
  case delete
  case lifecycle

  var canBeCancelledByControlOperation: Bool {
    self == .prepare || self == .generate
  }
}

private final class CoreMLBridgePromise: @unchecked Sendable {
  private let resolveBlock: RCTPromiseResolveBlock
  private let rejectBlock: RCTPromiseRejectBlock

  init(
    resolve: @escaping RCTPromiseResolveBlock,
    reject: @escaping RCTPromiseRejectBlock
  ) {
    resolveBlock = resolve
    rejectBlock = reject
  }

  func resolve(_ value: Any?) {
    resolveBlock(value)
  }

  func reject(code: String, message: String, error: Error? = nil) {
    rejectBlock(code, message, error)
  }
}

private final class CoreMLEventEnvelope: @unchecked Sendable {
  let event: CoreMLBridgeEvent
  let body: [String: Any]

  init(event: CoreMLBridgeEvent, body: [String: Any]) {
    self.event = event
    self.body = body
  }
}

private actor GenerationAccumulator {
  private var text = ""
  private var generatedTokens = 0
  private var tokensPerSecond = 0.0

  func record(_ update: GenerationUpdate) {
    text = update.text
    generatedTokens = update.generatedTokens
    tokensPerSecond = update.tokensPerSecond
  }

  func snapshot() -> (text: String, generatedTokens: Int, tokensPerSecond: Double) {
    (text, generatedTokens, tokensPerSecond)
  }
}

@available(iOS 18.0, *)
@objc(CoreMLInferenceModule)
final class CoreMLInferenceModule: RCTEventEmitter, @unchecked Sendable {
  private struct ScheduledOperation: Sendable {
    let id: String
    let kind: CoreMLBridgeOperation
    let priority: TaskPriority
    let run: @Sendable () async -> Void
    let rejectWhenBusy: @Sendable (_ code: String, _ message: String) -> Void
  }

  private struct ActiveOperation {
    let id: String
    let kind: CoreMLBridgeOperation
    let task: Task<Void, Never>
  }

  private let coordinator = InferenceCoordinator()
  private let operationStateQueue = DispatchQueue(
    label: "com.mongars.mobile.coreml.operation-state"
  )
  private let lifecycleObserverLock = NSLock()
  private let logger = Logger(
    subsystem: "com.mongars.mobile",
    category: "CoreMLInference"
  )

  // Access to these properties is confined to operationStateQueue.
  private var activeOperation: ActiveOperation?
  private var pendingOperation: ScheduledOperation?
  private var hasEventListeners = false
  private var isInvalidated = false

  // Access is protected because React Native invalidation and UIKit notifications
  // are not guaranteed to arrive on the same queue.
  private var lifecycleObservers: [NSObjectProtocol] = []

  override init() {
    super.init()
    installLifecycleObservers()
  }

  deinit {
    removeLifecycleObservers()
  }

  static func moduleName() -> String! {
    "CoreMLInferenceModule"
  }

  static func requiresMainQueueSetup() -> Bool {
    false
  }

  override func supportedEvents() -> [String]! {
    CoreMLBridgeEvent.allCases.map(\.rawValue)
  }

  override func startObserving() {
    super.startObserving()
    operationStateQueue.async { [weak self] in
      self?.hasEventListeners = true
    }
  }

  override func stopObserving() {
    operationStateQueue.async { [weak self] in
      self?.hasEventListeners = false
    }
    super.stopObserving()
  }

  override func invalidate() {
    removeLifecycleObservers()
    enqueueInvalidationCleanup()
    super.invalidate()
  }

  @objc func getModelStatus(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    operationStateQueue.async { [weak self] in
      guard let self else {
        promise.reject(
          code: "coreml_unavailable",
          message: "Le module Core ML n'est plus disponible."
        )
        return
      }
      guard !self.isInvalidated else {
        promise.reject(
          code: "coreml_invalidated",
          message: "Le pont React Native a ete invalide."
        )
        return
      }

      Task { [weak self] in
        guard let self else { return }
        let status = await self.coordinator.status()
        promise.resolve(self.statusPayload(status))
      }
    }
  }

  @objc func prepareModel(
    _ options: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    // Reserved for consent/network policy options. Model selection stays pinned
    // to MonGARSModelManifest and is never accepted from JavaScript.
    _ = options
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .prepare,
      priority: .utility,
      run: { [weak self] in
        await self?.performPrepare(promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func generate(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let requestID = UUID().uuidString

    let parsed: (messages: [ChatMessage], options: GenerationOptions)
    do {
      parsed = try parseGenerationRequest(request)
    } catch {
      let message = error.localizedDescription
      promise.reject(code: "coreml_invalid_request", message: message, error: error)
      emit(
        .error,
        body: errorPayload(
          requestID: requestID,
          operation: .generate,
          code: "coreml_invalid_request",
          message: message,
          recoverable: true
        )
      )
      return
    }

    let operation = ScheduledOperation(
      id: requestID,
      kind: .generate,
      priority: .userInitiated,
      run: { [weak self] in
        await self?.performGeneration(
          requestID: requestID,
          messages: parsed.messages,
          options: parsed.options
        )
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )

    enqueue(
      operation,
      cancellingCurrentOperation: false,
      onStarted: {
        promise.resolve(["requestId": requestID])
      }
    )
  }

  @objc func cancelGeneration(
    _ requestID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let identifier = requestID as String

    operationStateQueue.async { [weak self] in
      guard let self else {
        promise.resolve(["requestId": identifier, "cancelled": false])
        return
      }
      guard
        let active = self.activeOperation,
        active.kind == .generate,
        active.id == identifier
      else {
        promise.resolve(["requestId": identifier, "cancelled": false])
        return
      }

      active.task.cancel()
      promise.resolve(["requestId": identifier, "cancelled": true])
    }
  }

  @objc func unloadModel(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .unload,
      priority: .utility,
      run: { [weak self] in
        await self?.performUnload(promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: true)
  }

  @objc func deleteModel(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .delete,
      priority: .utility,
      run: { [weak self] in
        await self?.performDelete(promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: true)
  }

  private func performPrepare(promise: CoreMLBridgePromise) async {
    emit(
      .status,
      body: transientStatusPayload(
        phase: .downloading,
        detail: "Telechargement du modele Hugging Face"
      )
    )

    do {
      let status = try await coordinator.prepareModel { [weak self] progress in
        guard let self else { return }
        self.emit(.downloadProgress, body: self.progressPayload(progress))
      }
      let payload = statusPayload(status)
      emit(.status, body: payload)
      promise.resolve(payload)
    } catch {
      await publishCurrentStatus()
      if isCancellation(error) {
        promise.reject(
          code: "coreml_cancelled",
          message: "Preparation du modele annulee.",
          error: error
        )
        return
      }

      let code = bridgeErrorCode(error)
      let message = error.localizedDescription
      logger.error("Model preparation failed: \(message, privacy: .public)")
      emit(
        .error,
        body: errorPayload(
          requestID: nil,
          operation: .prepare,
          code: code,
          message: message,
          recoverable: isRecoverable(error)
        )
      )
      promise.reject(code: code, message: message, error: error)
    }
  }

  private func performGeneration(
    requestID: String,
    messages: [ChatMessage],
    options: GenerationOptions
  ) async {
    let accumulator = GenerationAccumulator()
    let startedAt = Date()
    emit(
      .status,
      body: transientStatusPayload(
        phase: .generating,
        detail: "Generation locale en cours",
        installedBytes: MonGARSModelManifest.installedBytes
      )
    )

    do {
      let result = try await coordinator.generate(
        messages: messages,
        options: options
      ) { [weak self] update in
        guard let self else { return }
        await accumulator.record(update)
        self.emit(
          .generation,
          body: self.generationPayload(
            requestID: requestID,
            sequence: update.generatedTokens,
            update: update
          )
        )
      }

      emit(
        .complete,
        body: completionPayload(
          requestID: requestID,
          sequence: result.generatedTokens + 1,
          result: result
        )
      )
      await publishCurrentStatus()
    } catch {
      if isCancellation(error) {
        let latest = await accumulator.snapshot()
        emit(
          .complete,
          body: [
            "requestId": requestID,
            "sequence": latest.generatedTokens + 1,
            "text": latest.text,
            "promptTokens": NSNull(),
            "generatedTokens": latest.generatedTokens,
            "duration": max(Date().timeIntervalSince(startedAt), 0),
            "tokensPerSecond": latest.tokensPerSecond,
            "finishReason": "cancelled",
            "modelId": MonGARSModelManifest.modelID,
          ]
        )
        await publishCurrentStatus()
        return
      }

      let code = bridgeErrorCode(error)
      let message = error.localizedDescription
      logger.error("Local generation failed: \(message, privacy: .public)")
      emit(
        .error,
        body: errorPayload(
          requestID: requestID,
          operation: .generate,
          code: code,
          message: message,
          recoverable: isRecoverable(error)
        )
      )
      await publishCurrentStatus()
    }
  }

  private func performUnload(promise: CoreMLBridgePromise) async {
    await coordinator.unloadModel()
    let status = await coordinator.status()
    let payload = statusPayload(status)
    emit(.status, body: payload)
    promise.resolve(payload)
  }

  private func performDelete(promise: CoreMLBridgePromise) async {
    do {
      let status = try await coordinator.deleteModel()
      let payload = statusPayload(status)
      emit(.status, body: payload)
      promise.resolve(payload)
    } catch {
      let code = bridgeErrorCode(error)
      let message = error.localizedDescription
      emit(
        .error,
        body: errorPayload(
          requestID: nil,
          operation: .delete,
          code: code,
          message: message,
          recoverable: isRecoverable(error)
        )
      )
      promise.reject(code: code, message: message, error: error)
    }
  }

  private func enqueue(
    _ operation: ScheduledOperation,
    cancellingCurrentOperation: Bool,
    onStarted: (@Sendable () -> Void)? = nil
  ) {
    operationStateQueue.async { [weak self] in
      guard let self else {
        operation.rejectWhenBusy(
          "coreml_unavailable",
          "Le module Core ML n'est plus disponible."
        )
        return
      }
      guard !self.isInvalidated else {
        operation.rejectWhenBusy(
          "coreml_invalidated",
          "Le pont React Native a ete invalide."
        )
        return
      }

      guard let active = self.activeOperation else {
        guard self.pendingOperation == nil else {
          operation.rejectWhenBusy(
            "coreml_busy",
            "Une operation Core ML est deja en attente."
          )
          return
        }
        self.launchLocked(operation, onStarted: onStarted)
        return
      }

      guard
        cancellingCurrentOperation,
        active.kind.canBeCancelledByControlOperation,
        self.pendingOperation == nil
      else {
        operation.rejectWhenBusy(
          "coreml_busy",
          "Une operation Core ML est deja en cours."
        )
        return
      }

      self.pendingOperation = operation
      active.task.cancel()
    }
  }

  private func launchLocked(
    _ operation: ScheduledOperation,
    onStarted: (@Sendable () -> Void)? = nil
  ) {
    dispatchPrecondition(condition: .onQueue(operationStateQueue))
    precondition(activeOperation == nil)

    let task = Task(priority: operation.priority) { [weak self] in
      await operation.run()
      self?.finishOperation(identifier: operation.id)
    }
    activeOperation = ActiveOperation(
      id: operation.id,
      kind: operation.kind,
      task: task
    )
    onStarted?()
  }

  private func finishOperation(identifier: String) {
    operationStateQueue.async { [self] in
      guard activeOperation?.id == identifier else { return }
      activeOperation = nil

      guard let pending = pendingOperation else { return }
      pendingOperation = nil
      launchLocked(pending)
    }
  }

  private func installLifecycleObservers() {
    let center = NotificationCenter.default
    let observers = [
      center.addObserver(
        forName: UIApplication.didEnterBackgroundNotification,
        object: nil,
        queue: nil
      ) { [weak self] _ in
        self?.enqueueLifecycleCleanup(reason: "background")
      },
      center.addObserver(
        forName: UIApplication.didReceiveMemoryWarningNotification,
        object: nil,
        queue: nil
      ) { [weak self] _ in
        self?.enqueueLifecycleCleanup(reason: "memory-warning")
      },
      center.addObserver(
        forName: ProcessInfo.thermalStateDidChangeNotification,
        object: nil,
        queue: nil
      ) { [weak self] _ in
        guard ProcessInfo.processInfo.thermalState == .critical else { return }
        self?.enqueueLifecycleCleanup(reason: "thermal-critical")
      },
    ]

    lifecycleObserverLock.lock()
    lifecycleObservers.append(contentsOf: observers)
    lifecycleObserverLock.unlock()
  }

  private func removeLifecycleObservers() {
    lifecycleObserverLock.lock()
    let observers = lifecycleObservers
    lifecycleObservers.removeAll()
    lifecycleObserverLock.unlock()

    for observer in observers {
      NotificationCenter.default.removeObserver(observer)
    }
  }

  private func enqueueLifecycleCleanup(reason: String) {
    let cleanup = ScheduledOperation(
      id: "lifecycle-\(UUID().uuidString)",
      kind: .lifecycle,
      priority: .utility,
      run: { [weak self] in
        guard let self else { return }
        await self.coordinator.unloadModel()
        await self.publishCurrentStatus()
      },
      rejectWhenBusy: { _, _ in }
    )

    operationStateQueue.async { [weak self] in
      guard let self, !self.isInvalidated else { return }
      self.scheduleCleanupLocked(cleanup, reason: reason)
    }
  }

  private func enqueueInvalidationCleanup() {
    let cleanup = ScheduledOperation(
      id: "invalidate-\(UUID().uuidString)",
      kind: .lifecycle,
      priority: .utility,
      run: { [coordinator] in
        await coordinator.unloadModel()
      },
      rejectWhenBusy: { _, _ in }
    )

    operationStateQueue.sync {
      guard !isInvalidated else { return }
      isInvalidated = true
      hasEventListeners = false
      scheduleCleanupLocked(cleanup, reason: "react-native-invalidate")
    }
  }

  private func scheduleCleanupLocked(
    _ cleanup: ScheduledOperation,
    reason: String
  ) {
    dispatchPrecondition(condition: .onQueue(operationStateQueue))
    logger.info("Scheduling Core ML cleanup: \(reason, privacy: .public)")

    guard let active = activeOperation else {
      if pendingOperation == nil {
        launchLocked(cleanup)
      }
      return
    }

    // A background URLSession can continue the large model transfer while the
    // app is suspended. Keep preparation alive; verification resumes when the
    // process is scheduled again. Generation is still cancelled immediately.
    if reason == "background", active.kind == .prepare {
      return
    }

    if active.kind == .lifecycle || active.kind == .unload || active.kind == .delete {
      return
    }

    active.task.cancel()
    if pendingOperation == nil {
      pendingOperation = cleanup
    }
  }

  private func emit(_ event: CoreMLBridgeEvent, body: [String: Any]) {
    let envelope = CoreMLEventEnvelope(event: event, body: body)
    operationStateQueue.async { [weak self] in
      guard let self, self.hasEventListeners, !self.isInvalidated else { return }
      DispatchQueue.main.async { [weak self] in
        guard let self, self.canEmitEvents else { return }
        self.sendEvent(withName: envelope.event.rawValue, body: envelope.body)
      }
    }
  }

  private var canEmitEvents: Bool {
    operationStateQueue.sync {
      hasEventListeners && !isInvalidated
    }
  }

  private func publishCurrentStatus() async {
    let status = await coordinator.status()
    emit(.status, body: statusPayload(status))
  }

  private func parseGenerationRequest(
    _ request: NSDictionary?
  ) throws -> (messages: [ChatMessage], options: GenerationOptions) {
    guard let request else {
      throw CoreMLBridgeRequestError.missingRequest
    }
    guard let rawMessages = request["messages"] as? NSArray else {
      throw CoreMLBridgeRequestError.missingMessages
    }
    guard !rawMessages.isEmpty, rawMessages.count <= 100 else {
      throw CoreMLBridgeRequestError.invalidMessageCount
    }

    var messages: [ChatMessage] = []
    messages.reserveCapacity(rawMessages.count)
    for (index, value) in rawMessages.enumerated() {
      guard
        let dictionary = value as? NSDictionary,
        let role = dictionary["role"] as? String,
        let content = dictionary["content"] as? String,
        role == "user" || role == "assistant"
      else {
        throw CoreMLBridgeRequestError.invalidMessage(index: index)
      }

      let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmed.isEmpty, trimmed.utf8.count <= 128_000 else {
        throw CoreMLBridgeRequestError.invalidMessage(index: index)
      }
      messages.append(ChatMessage(role: role, content: trimmed))
    }

    let rawOptions = request["options"] as? NSDictionary
    let defaults = GenerationOptions()
    let options = GenerationOptions(
      maxNewTokens: integer(rawOptions?["maxNewTokens"]) ?? defaults.maxNewTokens,
      temperature: float(rawOptions?["temperature"]) ?? defaults.temperature,
      topK: integer(rawOptions?["topK"]) ?? defaults.topK,
      topP: float(rawOptions?["topP"]) ?? defaults.topP,
      repetitionPenalty: float(rawOptions?["repetitionPenalty"])
        ?? defaults.repetitionPenalty,
      doSample: boolean(rawOptions?["doSample"]) ?? defaults.doSample
    )
    return (messages, options)
  }

  private func integer(_ value: Any?) -> Int? {
    (value as? NSNumber)?.intValue
  }

  private func float(_ value: Any?) -> Float? {
    (value as? NSNumber)?.floatValue
  }

  private func boolean(_ value: Any?) -> Bool? {
    (value as? NSNumber)?.boolValue
  }

  private func statusPayload(_ status: ModelStatus) -> [String: Any] {
    [
      "phase": status.phase.rawValue,
      "modelId": status.modelID,
      "displayName": status.displayName,
      "revision": status.revision,
      "installedBytes": status.installedBytes,
      "contextLength": status.contextLength,
      "minimumIOSVersion": status.minimumIOSVersion,
      "detail": nullable(status.detail),
    ]
  }

  private func transientStatusPayload(
    phase: InferencePhase,
    detail: String,
    installedBytes: Int64 = 0
  ) -> [String: Any] {
    [
      "phase": phase.rawValue,
      "modelId": MonGARSModelManifest.modelID,
      "displayName": MonGARSModelManifest.displayName,
      "revision": MonGARSModelManifest.revision,
      "installedBytes": installedBytes,
      "contextLength": MonGARSModelManifest.contextLength,
      "minimumIOSVersion": 18,
      "detail": detail,
    ]
  }

  private func progressPayload(_ progress: ModelProgress) -> [String: Any] {
    [
      "phase": progress.phase.rawValue,
      "fractionCompleted": progress.fractionCompleted,
      "bytesPerSecond": progress.bytesPerSecond.map { $0 as Any } ?? NSNull(),
      "detail": progress.detail,
    ]
  }

  private func generationPayload(
    requestID: String,
    sequence: Int,
    update: GenerationUpdate
  ) -> [String: Any] {
    [
      "requestId": requestID,
      "sequence": sequence,
      "text": update.text,
      "generatedTokens": update.generatedTokens,
      "tokensPerSecond": update.tokensPerSecond,
    ]
  }

  private func completionPayload(
    requestID: String,
    sequence: Int,
    result: GenerationResult
  ) -> [String: Any] {
    [
      "requestId": requestID,
      "sequence": sequence,
      "text": result.text,
      "promptTokens": result.promptTokens,
      "generatedTokens": result.generatedTokens,
      "duration": result.duration,
      "tokensPerSecond": result.tokensPerSecond,
      "finishReason": result.finishReason,
      "modelId": result.modelID,
    ]
  }

  private func errorPayload(
    requestID: String?,
    operation: CoreMLBridgeOperation,
    code: String,
    message: String,
    recoverable: Bool
  ) -> [String: Any] {
    [
      "requestId": nullable(requestID),
      "operation": operation.rawValue,
      "code": code,
      "message": message,
      "recoverable": recoverable,
    ]
  }

  private func nullable(_ value: String?) -> Any {
    guard let value else { return NSNull() }
    return value
  }

  private func isCancellation(_ error: Error) -> Bool {
    if error is CancellationError { return true }
    guard let inferenceError = error as? InferenceError else { return false }
    if case .preparationCancelled = inferenceError { return true }
    if case .generationCancelled = inferenceError { return true }
    return false
  }

  private func bridgeErrorCode(_ error: Error) -> String {
    guard let inferenceError = error as? InferenceError else {
      return "coreml_error"
    }
    switch inferenceError {
    case .unsupportedOS:
      return "coreml_unsupported_os"
    case .simulatorUnsupported:
      return "coreml_simulator_unsupported"
    case .insufficientDisk:
      return "coreml_insufficient_disk"
    case .modelNotInstalled:
      return "coreml_model_not_installed"
    case .invalidModel:
      return "coreml_invalid_model"
    case .integrityFailure:
      return "coreml_integrity_failure"
    case .thermalCritical:
      return "coreml_thermal_critical"
    case .emptyPrompt:
      return "coreml_empty_prompt"
    case .promptTooLong:
      return "coreml_prompt_too_long"
    case .operationInProgress:
      return "coreml_busy"
    case .preparationCancelled:
      return "coreml_cancelled"
    case .generationCancelled:
      return "coreml_cancelled"
    }
  }

  private func isRecoverable(_ error: Error) -> Bool {
    guard let inferenceError = error as? InferenceError else { return true }
    switch inferenceError {
    case .unsupportedOS, .simulatorUnsupported, .invalidModel, .integrityFailure:
      return false
    case .insufficientDisk, .modelNotInstalled, .thermalCritical, .emptyPrompt,
      .promptTooLong, .operationInProgress, .preparationCancelled,
      .generationCancelled:
      return true
    }
  }
}

private enum CoreMLBridgeRequestError: LocalizedError {
  case missingRequest
  case missingMessages
  case invalidMessageCount
  case invalidMessage(index: Int)

  var errorDescription: String? {
    switch self {
    case .missingRequest:
      return "La requete de generation est absente."
    case .missingMessages:
      return "Le tableau messages est absent."
    case .invalidMessageCount:
      return "La requete doit contenir entre 1 et 100 messages."
    case let .invalidMessage(index):
      return "Le message \(index) doit avoir un role user/assistant et un contenu valide."
    }
  }
}
