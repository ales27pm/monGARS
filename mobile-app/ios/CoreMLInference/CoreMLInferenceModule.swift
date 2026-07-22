import Foundation
import MonGARSAgentTools
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
  case agentTriggerHandoff = "onAgentTriggerHandoff"
  case appIntentHandoff = "onAppIntentHandoff"
}

private enum CoreMLBridgeOperation: String, Sendable {
  case prepare
  case generate
  case agent
  case permission
  case oauth
  case unload
  case delete
  case lifecycle

  var canBeCancelledByControlOperation: Bool {
    self == .prepare || self == .generate || self == .agent || self == .oauth
  }
}

private struct AgentBridgeRunRequest: Sendable {
  let runID: UUID
  let ownerID: String
  let prompt: String
  let history: [AgentConversationMessage]
  let requestedIntent: AgentIntent?
  let allowedToolIDs: Set<AgentToolID>?
  let approvalRecordID: UUID?
  let maxSteps: Int
}

private struct AppIntentMemoryBridgeRequest: Sendable {
  let id: UUID
  let ownerID: String
  let kind: MonGARSAppIntentHandoffKind
  let input: String
}

private struct AgentBridgeApprovalBinding: Sendable, Equatable {
  let recordID: UUID
  let ownerID: String
  let prompt: String
  let toolID: AgentToolID
  let arguments: AgentJSONArguments
  let expiresAt: Date
}

private actor AgentBridgeApprovalBindings {
  private var bindings: [UUID: AgentBridgeApprovalBinding] = [:]

  func register(
    record: AgentApprovalRecord,
    ownerID: String,
    prompt: String
  ) {
    removeExpired()
    bindings[record.id] = .init(
      recordID: record.id,
      ownerID: ownerID,
      prompt: prompt,
      toolID: record.toolID,
      arguments: record.arguments,
      expiresAt: record.expiresAt
    )
  }

  func matchesRun(recordID: UUID, ownerID: String, prompt: String) -> Bool {
    removeExpired()
    guard let binding = bindings[recordID] else { return false }
    return binding.ownerID == ownerID && binding.prompt == prompt
  }

  func matches(_ candidate: AgentBridgeApprovalBinding) -> Bool {
    removeExpired()
    guard let binding = bindings[candidate.recordID] else { return false }
    return binding.ownerID == candidate.ownerID
      && binding.prompt == candidate.prompt
      && binding.toolID == candidate.toolID
      && binding.arguments == candidate.arguments
  }

  func remove(recordID: UUID) {
    bindings.removeValue(forKey: recordID)
  }

  func removeAll() {
    bindings.removeAll()
  }

  private func removeExpired() {
    let now = Date()
    bindings = bindings.filter { $0.value.expiresAt > now }
  }
}

// One authority per application process. Individual AgentExecutor values are
// intentionally short-lived, but approvals and their contextual bindings are
// retained across runs and React Native module reconstruction until expiry,
// rejection, consumption, or bridge invalidation.
private let processAgentApprovalStore = AgentApprovalStore()
private let processAgentApprovalBindings = AgentBridgeApprovalBindings()

@available(iOS 18.0, *)
private struct CoreMLAgentModelAdapter: AgentModelGenerating {
  let coordinator: InferenceCoordinator

  func generate(request: AgentModelRequest) async throws -> String {
    var messages = request.history.map {
      ChatMessage(role: $0.role.rawValue, content: $0.content)
    }
    messages.append(
      ChatMessage(
        role: "user",
        content: AgentPromptComposer.modelTurnContent(request)
      )
    )

    let systemPrompt = """
    \(request.systemPrompt)
    Required response JSON schema:
    \(request.responseJSONSchema)
    """
    let result = try await coordinator.generate(
      messages: messages,
      options: .init(
        maxNewTokens: MonGARSModelManifest.maximumNewTokens,
        temperature: 0.05,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1.05,
        doSample: false
      ),
      systemPrompt: systemPrompt,
      onUpdate: { _ in
        // Raw structured decisions may contain a private `thought` field.
        // They are intentionally never emitted across the React Native bridge.
      }
    )
    return result.text
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

  private final class ActiveOperation {
    let id: String
    let kind: CoreMLBridgeOperation
    var task: Task<Void, Never>?

    init(id: String, kind: CoreMLBridgeOperation) {
      self.id = id
      self.kind = kind
    }
  }

  private final class EventTarget: @unchecked Sendable {
    weak var module: CoreMLInferenceModule?

    func emit(_ event: CoreMLBridgeEvent, body: [String: Any]) {
      module?.emit(event, body: body)
    }
  }

  private final class OperationWorker: @unchecked Sendable {
    private let coordinator: InferenceCoordinator
    private let approvalStore: AgentApprovalStore
    private let approvalBindings: AgentBridgeApprovalBindings
    private let logger: Logger
    private let emit: @Sendable (CoreMLBridgeEvent, [String: Any]) -> Void

    init(
      coordinator: InferenceCoordinator,
      approvalStore: AgentApprovalStore,
      approvalBindings: AgentBridgeApprovalBindings,
      logger: Logger,
      emit: @escaping @Sendable (CoreMLBridgeEvent, [String: Any]) -> Void
    ) {
      self.coordinator = coordinator
      self.approvalStore = approvalStore
      self.approvalBindings = approvalBindings
      self.logger = logger
      self.emit = emit
    }

    func status(promise: CoreMLBridgePromise) async {
      let status = await coordinator.status()
      promise.resolve(CoreMLInferenceModule.statusPayload(status))
    }

    func prepare(promise: CoreMLBridgePromise) async {
      emit(
        .status,
        CoreMLInferenceModule.transientStatusPayload(
          phase: .downloading,
          detail: "Telechargement du modele Hugging Face"
        )
      )

      do {
        let status = try await coordinator.prepareModel { [emit = self.emit] progress in
          emit(
            .downloadProgress,
            CoreMLInferenceModule.progressPayload(progress)
          )
        }
        let payload = CoreMLInferenceModule.statusPayload(status)
        emit(.status, payload)
        promise.resolve(payload)
      } catch {
        await publishCurrentStatus()
        if InferenceError.isCancellation(error) {
          promise.reject(
            code: "coreml_cancelled",
            message: "Preparation du modele annulee.",
            error: error
          )
          return
        }

        let code = CoreMLInferenceModule.bridgeErrorCode(error)
        let message = error.localizedDescription
        logger.error("Model preparation failed: \(message, privacy: .public)")
        emit(
          .error,
          CoreMLInferenceModule.errorPayload(
            requestID: nil,
            operation: .prepare,
            code: code,
            message: message,
            recoverable: CoreMLInferenceModule.isRecoverable(error)
          )
        )
        promise.reject(code: code, message: message, error: error)
      }
    }

    func generate(
      requestID: String,
      messages: [ChatMessage],
      options: GenerationOptions,
      systemPrompt: String?
    ) async {
      let accumulator = GenerationAccumulator()
      let startedAt = Date()
      let currentStatus = await coordinator.status()
      emit(
        .status,
        CoreMLInferenceModule.transientStatusPayload(
          phase: .generating,
          detail: "Generation locale en cours",
          installedBytes: currentStatus.installedBytes
        )
      )

      do {
        let result = try await coordinator.generate(
          messages: messages,
          options: options,
          systemPrompt: systemPrompt,
          progress: { [emit = self.emit] progress in
            emit(
              .downloadProgress,
              CoreMLInferenceModule.progressPayload(progress)
            )
          },
          onUpdate: { [emit = self.emit] update in
            await accumulator.record(update)
            emit(
              .generation,
              CoreMLInferenceModule.generationPayload(
                requestID: requestID,
                sequence: update.generatedTokens,
                update: update
              )
            )
          }
        )

        emit(
          .complete,
          CoreMLInferenceModule.completionPayload(
            requestID: requestID,
            sequence: result.generatedTokens + 1,
            result: result
          )
        )
        await publishCurrentStatus()
      } catch {
        if InferenceError.isCancellation(error) {
          let latest = await accumulator.snapshot()
          emit(
            .complete,
            [
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

        let code = CoreMLInferenceModule.bridgeErrorCode(error)
        let message = error.localizedDescription
        logger.error("Local generation failed: \(message, privacy: .public)")
        emit(
          .error,
          CoreMLInferenceModule.errorPayload(
            requestID: requestID,
            operation: .generate,
            code: code,
            message: message,
            recoverable: CoreMLInferenceModule.isRecoverable(error)
          )
        )
        await publishCurrentStatus()
      }
    }

    func agentCapabilities(
      ownerID: String,
      promise: CoreMLBridgePromise
    ) async {
      let available = await ScopedIOSAgentToolExecutor(rawOwnerID: ownerID)
        .availableToolIDs()
        .intersection(AgentToolCatalog.canonicalIDs)
        .sorted()
      promise.resolve([
        "available": true,
        "toolIds": available.map(\.rawValue),
        "toolCount": available.count,
        "supportsApprovals": true,
        "maximumSteps": 8,
      ])
    }

    func requestAgentPermission(
      _ permission: AgentPermission,
      promise: CoreMLBridgePromise
    ) async {
      let state = await IOSAgentPermissionProvider.shared.request(permission)
      promise.resolve([
        "permission": permission.rawValue,
        "state": state.rawValue,
      ])
    }

    func outlookConnectionStatus(
      ownerID: String,
      promise: CoreMLBridgePromise
    ) async {
      let status = await MicrosoftGraphOAuthTokenProvider.shared.status(rawOwnerID: ownerID)
      promise.resolve(CoreMLInferenceModule.outlookConnectionPayload(status))
    }

    func configureOutlook(
      ownerID: String,
      clientID: String,
      promise: CoreMLBridgePromise
    ) async {
      do {
        try MicrosoftGraphOAuthConfiguration.configureRuntimeClientID(clientID)
        let status = await MicrosoftGraphOAuthTokenProvider.shared.status(rawOwnerID: ownerID)
        promise.resolve(CoreMLInferenceModule.outlookConnectionPayload(status))
      } catch {
        let code = CoreMLInferenceModule.outlookOAuthErrorCode(error)
        let message = CoreMLInferenceModule.outlookOAuthErrorMessage(error)
        logger.error("Microsoft OAuth configuration failed: \(code, privacy: .public)")
        promise.reject(code: code, message: message, error: error)
      }
    }

    func connectOutlook(ownerID: String, promise: CoreMLBridgePromise) async {
      do {
        let status = try await MicrosoftGraphOAuthTokenProvider.shared.connect(
          rawOwnerID: ownerID
        )
        promise.resolve(CoreMLInferenceModule.outlookConnectionPayload(status))
      } catch is CancellationError {
        promise.reject(
          code: "outlook_sign_in_cancelled",
          message: "La connexion Microsoft a été annulée."
        )
      } catch {
        let code = CoreMLInferenceModule.outlookOAuthErrorCode(error)
        let message = CoreMLInferenceModule.outlookOAuthErrorMessage(error)
        logger.error("Microsoft OAuth failed: \(code, privacy: .public)")
        promise.reject(code: code, message: message, error: error)
      }
    }

    func disconnectOutlook(ownerID: String, promise: CoreMLBridgePromise) async {
      do {
        let status = try await MicrosoftGraphOAuthTokenProvider.shared.disconnect(
          rawOwnerID: ownerID
        )
        promise.resolve(CoreMLInferenceModule.outlookConnectionPayload(status))
      } catch {
        let code = CoreMLInferenceModule.outlookOAuthErrorCode(error)
        let message = CoreMLInferenceModule.outlookOAuthErrorMessage(error)
        logger.error("Microsoft OAuth disconnect failed: \(code, privacy: .public)")
        promise.reject(code: code, message: message, error: error)
      }
    }

    func getPendingAgentTrigger(
      ownerID: String,
      promise: CoreMLBridgePromise
    ) async {
      let handoff = await IOSAgentToolExecutor.shared.pendingTrigger(
        rawOwnerID: ownerID
      )
      guard let handoff else {
        promise.resolve(NSNull())
        return
      }
      promise.resolve([
        "id": handoff.id.uuidString.lowercased(),
        "title": handoff.title,
        "prompt": handoff.prompt,
        "repeats": handoff.repeats,
      ])
    }

    func acknowledgePendingAgentTrigger(
      ownerID: String,
      id: UUID,
      promise: CoreMLBridgePromise
    ) async {
      let acknowledged = await IOSAgentToolExecutor.shared.acknowledgePendingTrigger(
        rawOwnerID: ownerID,
        id: id
      )
      promise.resolve([
        "id": id.uuidString.lowercased(),
        "acknowledged": acknowledged,
      ])
    }

    func setActiveAppIntentProfile(
      ownerID: String,
      promise: CoreMLBridgePromise
    ) async {
      guard let store = MonGARSAppIntentHandoffStore.shared else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent protégé n'est pas disponible."
        )
        return
      }
      guard await store.setActiveProfile(rawOwnerID: ownerID) else {
        promise.reject(
          code: "app_intent_profile_unavailable",
          message: "Le profil App Intent n'a pas pu être protégé."
        )
        return
      }
      promise.resolve(["active": true])
    }

    func getPendingAppIntentHandoff(
      ownerID: String,
      promise: CoreMLBridgePromise
    ) async {
      guard let store = MonGARSAppIntentHandoffStore.shared else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent protégé n'est pas disponible."
        )
        return
      }
      guard let lookup = await store.pending(rawOwnerID: ownerID) else {
        promise.resolve(NSNull())
        return
      }
      let handoff = lookup.handoff
      var payload: [String: Any] = [
        "id": handoff.id.uuidString.lowercased(),
        "kind": lookup.profileMatches ? handoff.kind.rawValue : "masked",
        "createdAt": ISO8601DateFormatter().string(from: handoff.createdAt),
        "expiresAt": ISO8601DateFormatter().string(from: handoff.expiresAt),
        "profileMatches": lookup.profileMatches,
      ]
      if lookup.profileMatches, let input = handoff.input {
        payload["input"] = input
      }
      promise.resolve(payload)
    }

    func acknowledgeAppIntentHandoff(
      ownerID: String,
      id: UUID,
      promise: CoreMLBridgePromise
    ) async {
      guard let store = MonGARSAppIntentHandoffStore.shared else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent protégé n'est pas disponible."
        )
        return
      }
      let acknowledged = await store.acknowledge(
        expectedID: id,
        rawOwnerID: ownerID
      )
      promise.resolve([
        "id": id.uuidString.lowercased(),
        "acknowledged": acknowledged,
      ])
    }

    func discardAppIntentHandoff(id: UUID, promise: CoreMLBridgePromise) async {
      guard let store = MonGARSAppIntentHandoffStore.shared else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent protégé n'est pas disponible."
        )
        return
      }
      let discarded = await store.acknowledge(expectedID: id)
      promise.resolve([
        "id": id.uuidString.lowercased(),
        "discarded": discarded,
      ])
    }

    func executeAppIntentMemoryAction(
      request: AppIntentMemoryBridgeRequest,
      promise: CoreMLBridgePromise
    ) async {
      guard let store = MonGARSAppIntentHandoffStore.shared else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent protégé n'est pas disponible."
        )
        return
      }
      let tools = ScopedIOSAgentToolExecutor(rawOwnerID: request.ownerID)
      let expectedToolID: AgentToolID = request.kind == .memorySearch
        ? "memory.recall" : "memory.save"
      let availableToolIDs = await tools.availableToolIDs()
      guard availableToolIDs.contains(expectedToolID) else {
        promise.reject(
          code: "app_intent_memory_unavailable",
          message: "La mémoire locale n'est pas disponible pour ce profil."
        )
        return
      }
      guard let consumed = await store.consumeExactMemoryAction(
        expectedID: request.id,
        rawOwnerID: request.ownerID,
        expectedKind: request.kind,
        expectedInput: request.input
      ), let protectedInput = consumed.input else {
        promise.reject(
          code: "app_intent_handoff_mismatch",
          message: "L'action mémoire a expiré, changé ou appartient à un autre profil."
        )
        return
      }
      let arguments: AgentJSONArguments = consumed.kind == .memorySearch
        ? ["query": .string(protectedInput)]
        : ["content": .string(protectedInput), "kind": .string("fact")]
      guard case let .success(call) = AgentToolValidator.validate(
        rawToolID: expectedToolID.rawValue,
        arguments: arguments,
        availableToolIDs: [expectedToolID]
      ) else {
        promise.reject(
          code: "app_intent_memory_invalid",
          message: "Les arguments mémoire protégés sont invalides."
        )
        return
      }
      let invocation = AgentToolInvocation(
        runID: UUID(),
        stepIndex: 0,
        toolID: call.toolID,
        arguments: call.arguments,
        mode: .foreground
      )
      let rawResult = await tools.execute(invocation: invocation)

      if consumed.kind == .memoryAdd {
        let invocationMatches = rawResult.invocationID == invocation.id
        if rawResult.status == .success && invocationMatches {
          promise.resolve([
            "id": consumed.id.uuidString.lowercased(),
            "toolId": call.toolID.rawValue,
            "status": AgentToolResultStatus.success.rawValue,
            "message": "L'information a été ajoutée à la mémoire locale.",
            "errorCode": NSNull(),
          ])
          return
        }
        var failureMessage = AgentOutputSanitizer.sanitizeToolOutput(
          rawResult.text,
          maximumCharacters: call.definition.maximumOutputCharacters
        )
        if !invocationMatches {
          failureMessage = "L'ajout mémoire peut avoir réussi; vérifiez la mémoire avant toute relance."
        } else if failureMessage.isEmpty {
          failureMessage = "L'ajout mémoire n'a pas été confirmé; vérifiez la mémoire avant toute relance."
        }
        let errorCode = invocationMatches
          ? (rawResult.errorCode.map {
              AgentOutputSanitizer.sanitizeToolOutput($0, maximumCharacters: 120)
            } ?? "app_intent_memory_add_unconfirmed")
          : "app_intent_memory_add_commit_uncertain"
        promise.resolve([
          "id": consumed.id.uuidString.lowercased(),
          "toolId": call.toolID.rawValue,
          "status": AgentToolResultStatus.failed.rawValue,
          "message": failureMessage,
          "errorCode": errorCode,
        ])
        return
      }

      guard rawResult.invocationID == invocation.id else {
        promise.reject(
          code: "app_intent_memory_result_mismatch",
          message: "La mémoire locale a retourné un résultat incohérent."
        )
        return
      }
      var message = AgentOutputSanitizer.sanitizeToolOutput(
        rawResult.text,
        maximumCharacters: call.definition.maximumOutputCharacters
      )
      if message.isEmpty, let payload = rawResult.payload,
        let encoded = try? payload.canonicalJSONString() {
        message = AgentOutputSanitizer.sanitizeToolOutput(
          encoded,
          maximumCharacters: call.definition.maximumOutputCharacters
        )
      }
      let status = rawResult.status == .success && message.isEmpty
        ? AgentToolResultStatus.failed : rawResult.status
      if message.isEmpty {
        message = "La mémoire locale n'a retourné aucun résultat utilisable."
      }
      promise.resolve([
        "id": consumed.id.uuidString.lowercased(),
        "toolId": call.toolID.rawValue,
        "status": status.rawValue,
        "message": message,
        "errorCode": rawResult.errorCode.map {
          AgentOutputSanitizer.sanitizeToolOutput($0, maximumCharacters: 120) as Any
        } ?? NSNull(),
      ])
    }

    func resolveStoredAgentTrigger(
      ownerID: String,
      selector: String,
      promise: CoreMLBridgePromise
    ) async {
      let handoff = await IOSAgentToolExecutor.shared.resolveStoredTrigger(
        rawOwnerID: ownerID,
        selector: selector
      )
      guard let handoff else {
        promise.resolve(NSNull())
        return
      }
      promise.resolve([
        "id": handoff.id.uuidString.lowercased(),
        "title": handoff.title,
        "prompt": handoff.prompt,
        "repeats": handoff.repeats,
      ])
    }

    func runAgent(
      request: AgentBridgeRunRequest,
      promise: CoreMLBridgePromise
    ) async {
      if let approvalRecordID = request.approvalRecordID {
        let matches = await approvalBindings.matchesRun(
          recordID: approvalRecordID,
          ownerID: request.ownerID,
          prompt: request.prompt
        )
        guard matches else {
          promise.reject(
            code: "agent_approval_binding_mismatch",
            message: "L'approbation ne correspond pas à cette demande locale."
          )
          return
        }
      }

      let scopedTools = ScopedIOSAgentToolExecutor(rawOwnerID: request.ownerID)
      let hostAvailableToolIDs = await scopedTools.availableToolIDs()
        .intersection(AgentToolCatalog.canonicalIDs)
      let availableToolIDs = request.allowedToolIDs.map {
        hostAvailableToolIDs.intersection($0)
      } ?? hostAvailableToolIDs
      let executor = AgentExecutor(
        model: CoreMLAgentModelAdapter(coordinator: coordinator),
        toolExecutor: scopedTools,
        permissionProvider: IOSAgentPermissionProvider.shared,
        approvalAuthorizer: approvalStore
      )
      let result = await executor.run(.init(
        runID: request.runID,
        userInput: request.prompt,
        history: request.history,
        requestedIntent: request.requestedIntent,
        availableToolIDs: availableToolIDs,
        mode: .foreground,
        approvalRecordID: request.approvalRecordID,
        options: .init(maxSteps: request.maxSteps)
      ))

      if let consumedRecordID = request.approvalRecordID {
        await approvalBindings.remove(recordID: consumedRecordID)
      }
      if case let .approvalRequired(record) = result.outcome {
        await approvalBindings.register(
          record: record,
          ownerID: request.ownerID,
          prompt: request.prompt
        )
      }
      promise.resolve(CoreMLInferenceModule.agentResultPayload(result))
    }

    func approveAgent(
      binding: AgentBridgeApprovalBinding,
      promise: CoreMLBridgePromise
    ) async {
      guard await approvalBindings.matches(binding) else {
        promise.reject(
          code: "agent_approval_binding_mismatch",
          message: "L'approbation ne correspond pas à l'action proposée."
        )
        return
      }
      switch await approvalStore.approve(id: binding.recordID) {
      case let .success(record):
        promise.resolve([
          "recordId": record.id.uuidString.lowercased(),
          "status": record.status.rawValue,
        ])
      case .failure(.expired):
        await approvalBindings.remove(recordID: binding.recordID)
        promise.resolve([
          "recordId": binding.recordID.uuidString.lowercased(),
          "status": "expired",
        ])
      case let .failure(error):
        promise.reject(
          code: "agent_approval_rejected",
          message: CoreMLInferenceModule.approvalErrorMessage(error)
        )
      }
    }

    func rejectAgent(
      binding: AgentBridgeApprovalBinding,
      promise: CoreMLBridgePromise
    ) async {
      guard await approvalBindings.matches(binding) else {
        promise.reject(
          code: "agent_approval_binding_mismatch",
          message: "L'approbation ne correspond pas à l'action proposée."
        )
        return
      }
      switch await approvalStore.reject(id: binding.recordID) {
      case let .success(record):
        await approvalBindings.remove(recordID: binding.recordID)
        promise.resolve([
          "recordId": record.id.uuidString.lowercased(),
          "status": record.status.rawValue,
        ])
      case .failure(.expired):
        await approvalBindings.remove(recordID: binding.recordID)
        promise.resolve([
          "recordId": binding.recordID.uuidString.lowercased(),
          "status": "expired",
        ])
      case .failure(.notPending):
        // If JS approved the record but the subsequent model run failed before
        // consuming it, atomically burn the exact payload-bound approval. No
        // tool is invoked, and the record cannot be replayed.
        if case .success = await approvalStore.consumeApproval(
          id: binding.recordID,
          toolID: binding.toolID,
          arguments: binding.arguments
        ) {
          await approvalBindings.remove(recordID: binding.recordID)
          promise.resolve([
            "recordId": binding.recordID.uuidString.lowercased(),
            "status": "rejected",
          ])
          return
        }
        promise.reject(
          code: "agent_rejection_failed",
          message: "L'approbation n'est plus révocable."
        )
      case let .failure(error):
        promise.reject(
          code: "agent_rejection_failed",
          message: CoreMLInferenceModule.approvalErrorMessage(error)
        )
      }
    }

    func unload(promise: CoreMLBridgePromise? = nil) async {
      await coordinator.unloadModel()
      let status = await coordinator.status()
      let payload = CoreMLInferenceModule.statusPayload(status)
      emit(.status, payload)
      promise?.resolve(payload)
    }

    func delete(promise: CoreMLBridgePromise) async {
      do {
        let status = try await coordinator.deleteModel()
        let payload = CoreMLInferenceModule.statusPayload(status)
        emit(.status, payload)
        promise.resolve(payload)
      } catch {
        let code = CoreMLInferenceModule.bridgeErrorCode(error)
        let message = error.localizedDescription
        emit(
          .error,
          CoreMLInferenceModule.errorPayload(
            requestID: nil,
            operation: .delete,
            code: code,
            message: message,
            recoverable: CoreMLInferenceModule.isRecoverable(error)
          )
        )
        promise.reject(code: code, message: message, error: error)
      }
    }

    private func publishCurrentStatus() async {
      let status = await coordinator.status()
      emit(.status, CoreMLInferenceModule.statusPayload(status))
    }
  }

  private let coordinator: InferenceCoordinator
  private let approvalStore: AgentApprovalStore
  private let approvalBindings: AgentBridgeApprovalBindings
  private let operationStateQueue = DispatchQueue(
    label: "com.mongars.mobile.coreml.operation-state"
  )
  private let lifecycleObserverLock = NSLock()
  private let logger: Logger
  private let operationWorker: OperationWorker

  // Access to these properties is confined to operationStateQueue.
  private var activeOperation: ActiveOperation?
  private var pendingOperation: ScheduledOperation?
  private var hasEventListeners = false
  private var isInvalidated = false

  // Access is protected because React Native invalidation and UIKit notifications
  // are not guaranteed to arrive on the same queue.
  private var lifecycleObservers: [NSObjectProtocol] = []

  override init() {
    let coordinator = InferenceCoordinator()
    let approvalStore = processAgentApprovalStore
    let approvalBindings = processAgentApprovalBindings
    let logger = Logger(
      subsystem: "com.mongars.mobile",
      category: "CoreMLInference"
    )
    let eventTarget = EventTarget()
    self.coordinator = coordinator
    self.approvalStore = approvalStore
    self.approvalBindings = approvalBindings
    self.logger = logger
    self.operationWorker = OperationWorker(
      coordinator: coordinator,
      approvalStore: approvalStore,
      approvalBindings: approvalBindings,
      logger: logger,
      emit: { [eventTarget] event, body in
        eventTarget.emit(event, body: body)
      }
    )
    super.init()
    eventTarget.module = self
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

      let worker = self.operationWorker
      Task { [worker] in
        await worker.status(promise: promise)
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
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .prepare,
      priority: .utility,
      run: { [worker] in
        await worker.prepare(promise: promise)
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

    let parsed: (
      messages: [ChatMessage],
      options: GenerationOptions,
      systemPrompt: String?
    )
    do {
      parsed = try parseGenerationRequest(request)
    } catch {
      let message = error.localizedDescription
      promise.reject(code: "coreml_invalid_request", message: message, error: error)
      emit(
        .error,
        body: Self.errorPayload(
          requestID: requestID,
          operation: .generate,
          code: "coreml_invalid_request",
          message: message,
          recoverable: true
        )
      )
      return
    }

    let worker = operationWorker
    let operation = ScheduledOperation(
      id: requestID,
      kind: .generate,
      priority: .userInitiated,
      run: { [worker] in
        await worker.generate(
          requestID: requestID,
          messages: parsed.messages,
          options: parsed.options,
          systemPrompt: parsed.systemPrompt
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

  @objc func getAgentCapabilities(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "agent_invalid_owner",
        message: "Une session monGARS valide est requise pour les capacités locales."
      )
      return
    }
    operationStateQueue.async { [weak self] in
      guard let self, !self.isInvalidated else {
        promise.reject(
          code: "agent_unavailable",
          message: "Le moteur d'outils local n'est plus disponible."
        )
        return
      }
      let worker = self.operationWorker
      Task { [worker] in
        await worker.agentCapabilities(ownerID: ownerID, promise: promise)
      }
    }
  }

  @objc func runAgent(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let parsed: AgentBridgeRunRequest
    do {
      parsed = try parseAgentRunRequest(request)
    } catch {
      promise.reject(
        code: "agent_invalid_request",
        message: error.localizedDescription,
        error: error
      )
      return
    }

    let worker = operationWorker
    let operation = ScheduledOperation(
      id: parsed.runID.uuidString.lowercased(),
      kind: .agent,
      priority: .userInitiated,
      run: { [worker] in
        await worker.runAgent(request: parsed, promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func requestAgentPermission(
    _ rawPermission: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let value = (rawPermission as String)
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard let permission = AgentPermission(rawValue: value) else {
      promise.reject(
        code: "agent_invalid_permission",
        message: "L'autorisation demandée est inconnue."
      )
      return
    }
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: "permission-\(UUID().uuidString.lowercased())",
      kind: .permission,
      priority: .userInitiated,
      run: { [worker] in
        await worker.requestAgentPermission(permission, promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func getOutlookConnectionStatus(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "outlook_invalid_owner",
        message: "Une session monGARS valide est requise pour accéder à Outlook."
      )
      return
    }
    operationStateQueue.async { [weak self] in
      guard let self, !self.isInvalidated else {
        promise.reject(
          code: "outlook_unavailable",
          message: "La connexion Outlook native n'est plus disponible."
        )
        return
      }
      let worker = self.operationWorker
      Task { [worker] in
        await worker.outlookConnectionStatus(ownerID: ownerID, promise: promise)
      }
    }
  }

  @objc func connectOutlook(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "outlook_invalid_owner",
        message: "Une session monGARS valide est requise pour connecter Outlook."
      )
      return
    }
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: "outlook-connect-\(UUID().uuidString.lowercased())",
      kind: .oauth,
      priority: .userInitiated,
      run: { [worker] in
        await worker.connectOutlook(ownerID: ownerID, promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func configureOutlookClientID(
    _ rawOwnerID: NSString,
    clientID rawClientID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "outlook_invalid_owner",
        message: "Une session monGARS valide est requise pour configurer Outlook."
      )
      return
    }
    let clientIDValue = (rawClientID as String)
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    guard let parsedClientID = UUID(uuidString: clientIDValue) else {
      promise.reject(
        code: "outlook_invalid_client_id",
        message: "L'identifiant d'application Microsoft doit être un UUID valide."
      )
      return
    }
    let clientID = parsedClientID.uuidString.lowercased()
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: "outlook-configure-\(UUID().uuidString.lowercased())",
      kind: .oauth,
      priority: .userInitiated,
      run: { [worker] in
        await worker.configureOutlook(
          ownerID: ownerID,
          clientID: clientID,
          promise: promise
        )
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func disconnectOutlook(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "outlook_invalid_owner",
        message: "Une session monGARS valide est requise pour déconnecter Outlook."
      )
      return
    }
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: "outlook-disconnect-\(UUID().uuidString.lowercased())",
      kind: .oauth,
      priority: .userInitiated,
      run: { [worker] in
        await worker.disconnectOutlook(ownerID: ownerID, promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }

  @objc func getPendingAgentTrigger(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "agent_invalid_owner",
        message: "Le propriétaire local est invalide."
      )
      return
    }
    operationStateQueue.async { [weak self] in
      guard let self, !self.isInvalidated else {
        promise.reject(
          code: "agent_unavailable",
          message: "Le moteur d'outils local n'est plus disponible."
        )
        return
      }
      let worker = self.operationWorker
      Task { [worker] in
        await worker.getPendingAgentTrigger(
          ownerID: ownerID,
          promise: promise
        )
      }
    }
  }

  @objc func acknowledgePendingAgentTrigger(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let request else {
      promise.reject(
        code: "agent_invalid_trigger",
        message: "La confirmation du déclencheur est absente."
      )
      return
    }
    do {
      try requireOnlyKeys(request, allowed: ["ownerId", "id"])
    } catch {
      promise.reject(
        code: "agent_invalid_trigger",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    guard let ownerID = boundedAgentOwnerID(request["ownerId"]),
      let idText = boundedString(request["id"], maximumBytes: 64),
      let id = UUID(uuidString: idText) else {
      promise.reject(
        code: "agent_invalid_trigger",
        message: "Le propriétaire ou l'identifiant du déclencheur est invalide."
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.acknowledgePendingAgentTrigger(
        ownerID: ownerID,
        id: id,
        promise: promise
      )
    }
  }


  @objc func setActiveAppIntentProfile(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(code: "app_intent_profile_invalid", message: "Le profil App Intent est invalide.")
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.setActiveAppIntentProfile(ownerID: ownerID, promise: promise)
    }
  }

  @objc func getPendingAppIntentHandoff(
    _ rawOwnerID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let ownerID = boundedAgentOwnerID(rawOwnerID as String) else {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: "Le profil App Intent est invalide."
      )
      return
    }
    operationStateQueue.async { [weak self] in
      guard let self, !self.isInvalidated else {
        promise.reject(
          code: "app_intent_handoff_unavailable",
          message: "Le transfert App Intent n'est plus disponible."
        )
        return
      }
      let worker = self.operationWorker
      Task { [worker] in
        await worker.getPendingAppIntentHandoff(
          ownerID: ownerID,
          promise: promise
        )
      }
    }
  }

  @objc func acknowledgeAppIntentHandoff(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let request else {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: "La confirmation App Intent est absente."
      )
      return
    }
    do {
      try requireOnlyKeys(request, allowed: ["ownerId", "id"])
    } catch {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    guard
      let ownerID = boundedAgentOwnerID(request["ownerId"]),
      let idText = boundedString(request["id"], maximumBytes: 64),
      let id = UUID(uuidString: idText)
    else {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: "Le profil ou l'identifiant App Intent est invalide."
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.acknowledgeAppIntentHandoff(
        ownerID: ownerID,
        id: id,
        promise: promise
      )
    }
  }

  @objc func discardAppIntentHandoff(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let request else {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: "L'identifiant App Intent à ignorer est absent."
      )
      return
    }
    do {
      try requireOnlyKeys(request, allowed: ["id"])
    } catch {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    guard
      let idText = boundedString(request["id"], maximumBytes: 64),
      let id = UUID(uuidString: idText)
    else {
      promise.reject(
        code: "app_intent_handoff_invalid",
        message: "L'identifiant App Intent à ignorer est invalide."
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.discardAppIntentHandoff(id: id, promise: promise)
    }
  }

  @objc func executeAppIntentMemoryAction(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let request else {
      promise.reject(code: "app_intent_memory_invalid", message: "L'action mémoire App Intent est absente.")
      return
    }
    do {
      try requireOnlyKeys(request, allowed: ["ownerId", "id", "kind", "input"])
    } catch {
      promise.reject(code: "app_intent_memory_invalid", message: error.localizedDescription, error: error)
      return
    }
    guard let ownerID = boundedAgentOwnerID(request["ownerId"]),
      let idText = boundedString(request["id"], maximumBytes: 64),
      let id = UUID(uuidString: idText),
      let kindText = boundedString(request["kind"], maximumBytes: 32),
      let kind = MonGARSAppIntentHandoffKind(rawValue: kindText),
      kind == .memorySearch || kind == .memoryAdd,
      let input = boundedString(
        request["input"],
        maximumBytes: kind == .memorySearch ? 192 : 186
      ) else {
      promise.reject(
        code: "app_intent_memory_invalid",
        message: "Le profil, l'identifiant, le type ou le contenu mémoire est invalide."
      )
      return
    }
    let parsed = AppIntentMemoryBridgeRequest(
      id: id,
      ownerID: ownerID,
      kind: kind,
      input: input
    )
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: "app-intent-memory-\(id.uuidString.lowercased())",
      kind: .agent,
      priority: .userInitiated,
      run: { [worker] in
        await worker.executeAppIntentMemoryAction(request: parsed, promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: false)
  }


  @objc func resolveStoredAgentTrigger(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    guard let request else {
      promise.reject(
        code: "agent_invalid_trigger",
        message: "Le sélecteur du déclencheur est absent."
      )
      return
    }
    do {
      try requireOnlyKeys(request, allowed: ["ownerId", "selector"])
    } catch {
      promise.reject(
        code: "agent_invalid_trigger",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    guard
      let ownerID = boundedAgentOwnerID(request["ownerId"]),
      let selector = boundedString(request["selector"], maximumBytes: 512)
    else {
      promise.reject(
        code: "agent_invalid_trigger",
        message: "Le propriétaire ou le sélecteur du déclencheur est invalide."
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.resolveStoredAgentTrigger(
        ownerID: ownerID,
        selector: selector,
        promise: promise
      )
    }
  }

  @objc func approveAgent(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let binding: AgentBridgeApprovalBinding
    do {
      binding = try parseAgentApprovalBinding(request)
    } catch {
      promise.reject(
        code: "agent_invalid_approval",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.approveAgent(binding: binding, promise: promise)
    }
  }

  @objc func rejectAgent(
    _ request: NSDictionary?,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let binding: AgentBridgeApprovalBinding
    do {
      binding = try parseAgentApprovalBinding(request)
    } catch {
      promise.reject(
        code: "agent_invalid_rejection",
        message: error.localizedDescription,
        error: error
      )
      return
    }
    let worker = operationWorker
    Task { [worker] in
      await worker.rejectAgent(binding: binding, promise: promise)
    }
  }

  @objc func cancelAgent(
    _ runID: NSString,
    resolver resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let identifier = (runID as String).lowercased()
    guard UUID(uuidString: identifier) != nil else {
      promise.reject(
        code: "agent_invalid_request",
        message: "L'identifiant d'exécution de l'agent est invalide."
      )
      return
    }
    operationStateQueue.async { [weak self] in
      guard let self,
        let active = self.activeOperation,
        active.kind == .agent,
        active.id == identifier else {
        promise.resolve(["runId": identifier, "cancelled": false])
        return
      }
      active.task?.cancel()
      promise.resolve(["runId": identifier, "cancelled": true])
    }
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

      active.task?.cancel()
      promise.resolve(["requestId": identifier, "cancelled": true])
    }
  }

  @objc func unloadModel(
    _ resolve: @escaping RCTPromiseResolveBlock,
    rejecter reject: @escaping RCTPromiseRejectBlock
  ) {
    let promise = CoreMLBridgePromise(resolve: resolve, reject: reject)
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .unload,
      priority: .utility,
      run: { [worker] in
        await worker.unload(promise: promise)
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
    let worker = operationWorker
    let operation = ScheduledOperation(
      id: UUID().uuidString,
      kind: .delete,
      priority: .utility,
      run: { [worker] in
        await worker.delete(promise: promise)
      },
      rejectWhenBusy: { code, message in
        promise.reject(code: code, message: message)
      }
    )
    enqueue(operation, cancellingCurrentOperation: true)
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
      active.task?.cancel()
    }
  }

  private func launchLocked(
    _ operation: ScheduledOperation,
    onStarted: (@Sendable () -> Void)? = nil
  ) {
    dispatchPrecondition(condition: .onQueue(operationStateQueue))
    precondition(activeOperation == nil)

    let active = ActiveOperation(id: operation.id, kind: operation.kind)
    activeOperation = active
    onStarted?()

    let task = Task(priority: operation.priority) { [weak self] in
      await operation.run()
      self?.finishOperation(identifier: operation.id)
    }
    active.task = task
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
      center.addObserver(
        forName: Notification.Name("MonGARSAgentTriggerHandoffAvailable"),
        object: nil,
        queue: nil
      ) { [weak self] notification in
        guard
          let identifier = notification.userInfo?["id"] as? String,
          UUID(uuidString: identifier) != nil,
          let tappedAt = notification.userInfo?["tappedAt"] as? Date
        else { return }
        self?.emit(
          .agentTriggerHandoff,
          body: [
            "id": identifier,
            "tappedAt": ISO8601DateFormatter().string(from: tappedAt),
          ]
        )
      },
      center.addObserver(
        forName: Notification.Name("MonGARSAppIntentHandoffAvailable"),
        object: nil,
        queue: nil
      ) { [weak self] notification in
        guard
          let identifier = notification.userInfo?["id"] as? String,
          UUID(uuidString: identifier) != nil,
          let createdAt = notification.userInfo?["createdAt"] as? Date
        else { return }
        self?.emit(
          .appIntentHandoff,
          body: [
            "id": identifier.lowercased(),
            "createdAt": ISO8601DateFormatter().string(from: createdAt),
          ]
        )
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
    let worker = operationWorker
    let cleanup = ScheduledOperation(
      id: "lifecycle-\(UUID().uuidString)",
      kind: .lifecycle,
      priority: .utility,
      run: { [worker] in
        await worker.unload()
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
      run: { [coordinator, approvalBindings] in
        await coordinator.unloadModel()
        await approvalBindings.removeAll()
      },
      rejectWhenBusy: { _, _ in }
    )

    operationStateQueue.sync {
      guard !isInvalidated else { return }
      isInvalidated = true
      hasEventListeners = false
      logger.info("Scheduling Core ML cleanup: react-native-invalidate")

      if let pendingOperation {
        pendingOperation.rejectWhenBusy(
          "coreml_invalidated",
          "Le pont React Native a ete invalide."
        )
        self.pendingOperation = nil
      }

      guard let activeOperation else {
        launchLocked(cleanup)
        return
      }

      // Keep the interrupted operation registered until its task exits. Its
      // finish callback will then launch cleanup, preserving the same
      // serialization guarantee used by unload/delete control operations.
      pendingOperation = cleanup
      activeOperation.task?.cancel()
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

    active.task?.cancel()
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

  private func parseGenerationRequest(
    _ request: NSDictionary?
  ) throws -> (
    messages: [ChatMessage],
    options: GenerationOptions,
    systemPrompt: String?
  ) {
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
    let systemPrompt: String?
    if let rawSystemPrompt = request["systemPrompt"] {
      guard let value = rawSystemPrompt as? String else {
        throw CoreMLBridgeRequestError.invalidSystemPrompt
      }
      let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmed.isEmpty, trimmed.utf8.count <= 32_000 else {
        throw CoreMLBridgeRequestError.invalidSystemPrompt
      }
      systemPrompt = trimmed
    } else {
      systemPrompt = nil
    }
    return (messages, options, systemPrompt)
  }

  private func parseAgentRunRequest(
    _ request: NSDictionary?
  ) throws -> AgentBridgeRunRequest {
    guard let request else { throw AgentBridgeRequestError.missingRequest }
    try requireOnlyKeys(
      request,
      allowed: [
        "runId", "ownerId", "prompt", "history", "requestedIntent",
        "allowedToolIds", "approvalRecordId", "maxSteps",
      ]
    )
    guard let runIDText = boundedString(request["runId"], maximumBytes: 64),
      let runID = UUID(uuidString: runIDText) else {
      throw AgentBridgeRequestError.invalidRunID
    }
    guard let ownerID = boundedAgentOwnerID(request["ownerId"]) else {
      throw AgentBridgeRequestError.invalidOwner
    }
    guard let prompt = boundedString(request["prompt"], maximumBytes: 128_000) else {
      throw AgentBridgeRequestError.invalidPrompt
    }
    guard let rawHistory = request["history"] as? NSArray,
      rawHistory.count <= 50 else {
      throw AgentBridgeRequestError.invalidHistory
    }
    let history = try rawHistory.enumerated().map { index, value in
      guard let item = value as? NSDictionary else {
        throw AgentBridgeRequestError.invalidHistoryMessage(index)
      }
      try requireOnlyKeys(item, allowed: ["role", "content"])
      guard let rawRole = item["role"] as? String,
        let role = AgentMessageRole(rawValue: rawRole),
        let content = boundedString(item["content"], maximumBytes: 64_000) else {
        throw AgentBridgeRequestError.invalidHistoryMessage(index)
      }
      return AgentConversationMessage(role: role, content: content)
    }
    let approvalRecordID: UUID?
    if let rawApprovalRecordID = request["approvalRecordId"] {
      guard let text = boundedString(rawApprovalRecordID, maximumBytes: 64),
        let identifier = UUID(uuidString: text) else {
        throw AgentBridgeRequestError.invalidApprovalRecordID
      }
      approvalRecordID = identifier
    } else {
      approvalRecordID = nil
    }
    let rawRequestedIntent = request["requestedIntent"]
    let rawAllowedToolIDs = request["allowedToolIds"]
    guard (rawRequestedIntent == nil) == (rawAllowedToolIDs == nil) else {
      throw AgentBridgeRequestError.invalidToolScope
    }
    let requestedIntent: AgentIntent?
    let allowedToolIDs: Set<AgentToolID>?
    if let rawRequestedIntent, let rawAllowedToolIDs {
      guard approvalRecordID == nil,
        let intentText = boundedString(rawRequestedIntent, maximumBytes: 64),
        let intent = AgentIntent(rawValue: intentText),
        ![AgentIntent.chat, .unknown].contains(intent),
        let rawToolIDs = rawAllowedToolIDs as? NSArray,
        !rawToolIDs.isEmpty,
        rawToolIDs.count <= AgentToolCatalog.canonicalIDs.count else {
        throw AgentBridgeRequestError.invalidToolScope
      }
      let parsedToolIDs = try rawToolIDs.map { value -> AgentToolID in
        guard let rawToolID = boundedString(value, maximumBytes: 128),
          let definition = AgentToolCatalog.definition(for: rawToolID),
          definition.id.rawValue == rawToolID else {
          throw AgentBridgeRequestError.invalidToolScope
        }
        return definition.id
      }
      let scopedToolIDs = Set(parsedToolIDs)
      let route = AgentIntentRouter.route(intent: intent)
      guard scopedToolIDs.count == parsedToolIDs.count,
        scopedToolIDs.isSubset(of: route.allowedToolIDs),
        !scopedToolIDs.intersection(route.fulfillmentToolIDs).isEmpty else {
        throw AgentBridgeRequestError.invalidToolScope
      }
      requestedIntent = intent
      allowedToolIDs = scopedToolIDs
    } else {
      requestedIntent = nil
      allowedToolIDs = nil
    }
    let maxSteps: Int
    if let rawMaxSteps = request["maxSteps"] {
      guard let value = strictInteger(rawMaxSteps), (1...8).contains(value) else {
        throw AgentBridgeRequestError.invalidMaxSteps
      }
      maxSteps = value
    } else {
      maxSteps = 4
    }
    return .init(
      runID: runID,
      ownerID: ownerID,
      prompt: prompt,
      history: history,
      requestedIntent: requestedIntent,
      allowedToolIDs: allowedToolIDs,
      approvalRecordID: approvalRecordID,
      maxSteps: maxSteps
    )
  }

  private func parseAgentApprovalBinding(
    _ request: NSDictionary?
  ) throws -> AgentBridgeApprovalBinding {
    guard let request else { throw AgentBridgeRequestError.missingRequest }
    try requireOnlyKeys(
      request,
      allowed: ["recordId", "ownerId", "prompt", "toolId", "arguments", "expiresAt"]
    )
    guard let recordText = boundedString(request["recordId"], maximumBytes: 64),
      let recordID = UUID(uuidString: recordText) else {
      throw AgentBridgeRequestError.invalidApprovalRecordID
    }
    guard let ownerID = boundedAgentOwnerID(request["ownerId"]) else {
      throw AgentBridgeRequestError.invalidOwner
    }
    guard let prompt = boundedString(request["prompt"], maximumBytes: 128_000) else {
      throw AgentBridgeRequestError.invalidPrompt
    }
    guard let rawToolID = boundedString(request["toolId"], maximumBytes: 128),
      let definition = AgentToolCatalog.definition(for: rawToolID),
      definition.id.rawValue == rawToolID else {
      throw AgentBridgeRequestError.invalidToolID
    }
    guard let rawArguments = request["arguments"] as? NSDictionary else {
      throw AgentBridgeRequestError.invalidArguments
    }
    let value = try agentJSONValue(rawArguments, depth: 0)
    guard case let .object(arguments) = value else {
      throw AgentBridgeRequestError.invalidArguments
    }
    guard let expiresAtText = boundedString(request["expiresAt"], maximumBytes: 64),
      let expiresAt = Self.parseISO8601(expiresAtText) else {
      throw AgentBridgeRequestError.invalidExpiry
    }
    return .init(
      recordID: recordID,
      ownerID: ownerID,
      prompt: prompt,
      toolID: definition.id,
      arguments: arguments,
      expiresAt: expiresAt
    )
  }

  private func requireOnlyKeys(
    _ dictionary: NSDictionary,
    allowed: Set<String>
  ) throws {
    let keys = dictionary.allKeys.compactMap { $0 as? String }
    guard keys.count == dictionary.count,
      Set(keys).subtracting(allowed).isEmpty else {
      throw AgentBridgeRequestError.unexpectedFields
    }
  }

  private func boundedString(_ value: Any?, maximumBytes: Int) -> String? {
    guard let raw = value as? String else { return nil }
    let output = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !output.isEmpty, output.utf8.count <= maximumBytes else { return nil }
    return output
  }

  private func boundedAgentOwnerID(_ value: Any?) -> String? {
    guard let raw = value as? String,
      !raw.unicodeScalars.contains(where: {
        CharacterSet.controlCharacters.union(.newlines).contains($0)
      })
    else { return nil }
    return boundedString(raw, maximumBytes: 256)
  }

  private func strictInteger(_ value: Any?) -> Int? {
    guard let number = value as? NSNumber else { return nil }
    let double = number.doubleValue
    guard double.isFinite,
      double.rounded(.towardZero) == double,
      double >= Double(Int.min),
      double <= Double(Int.max) else { return nil }
    return Int(double)
  }

  private func agentJSONValue(_ value: Any, depth: Int) throws -> AgentJSONValue {
    guard depth == 0 else { throw AgentBridgeRequestError.invalidArguments }
    do {
      return try AgentFoundationJSON.decode(
        value,
        limits: .init(
          maximumDepth: 8,
          maximumArrayCount: 256,
          maximumObjectCount: 128,
          maximumKeyBytes: 128,
          maximumStringBytes: 32_000
        )
      )
    } catch {
      throw AgentBridgeRequestError.invalidArguments
    }
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

  private static func agentResultPayload(_ result: AgentRunResult) -> [String: Any] {
    var payload: [String: Any] = [
      "runId": result.runID.uuidString.lowercased(),
      "intent": result.route.intent.rawValue,
      "events": result.events.map(agentEventPayload),
      "executedToolCount": result.executedToolCount,
      "modelTurnCount": result.modelTurnCount,
      "usedRepairAttempt": result.usedRepairAttempt,
    ]
    switch result.outcome {
    case let .final(message):
      payload["status"] = "final"
      payload["message"] = message
    case let .clarification(message):
      payload["status"] = "clarification"
      payload["message"] = message
    case let .approvalRequired(record):
      let definition = AgentToolCatalog.definition(for: record.toolID.rawValue)
      payload["status"] = "approval_required"
      payload["approval"] = [
        "recordId": record.id.uuidString.lowercased(),
        "toolId": record.toolID.rawValue,
        "arguments": foundationJSON(.object(record.arguments)),
        "displayName": definition?.displayName ?? record.toolID.rawValue,
        "risk": riskName(definition?.risk ?? .critical),
        "expiresAt": iso8601(record.expiresAt),
      ]
    case let .permissionRequired(permission):
      payload["status"] = "permission_required"
      payload["permission"] = permission.rawValue
      payload["message"] = "L'autorisation \(permission.rawValue) est requise avant cette action."
    case let .unavailable(message):
      payload["status"] = "unavailable"
      payload["message"] = message
    case let .failed(failure):
      payload["status"] = failure == .cancelled ? "cancelled" : "failed"
      payload["code"] = agentFailureCode(failure)
      payload["message"] = failure.message
    }
    return payload
  }

  private static func agentEventPayload(_ event: AgentEvent) -> [String: Any] {
    switch event {
    case .started:
      return ["type": "started"]
    case .routed:
      return ["type": "routed"]
    case let .modelTurnStarted(stepIndex, isRepair):
      return [
        "type": "model_turn",
        "stepIndex": stepIndex,
        "status": isRepair ? "repair" : "initial",
      ]
    case let .repairRequested(stepIndex, _):
      return ["type": "repair_requested", "stepIndex": stepIndex]
    case let .actionValidated(toolID, _):
      return ["type": "action_validated", "toolId": toolID.rawValue]
    case let .duplicateCallBlocked(toolID):
      return ["type": "failure", "toolId": toolID.rawValue, "status": "duplicate"]
    case let .permissionRequired(permission):
      return ["type": "permission_required", "status": permission.rawValue]
    case let .approvalRequired(record):
      return ["type": "approval_required", "toolId": record.toolID.rawValue]
    case .policyDenied:
      return ["type": "failure", "status": "policy_denied"]
    case let .toolInvocation(invocation):
      return [
        "type": "tool_started",
        "toolId": invocation.toolID.rawValue,
        "stepIndex": invocation.stepIndex,
      ]
    case let .toolResult(toolID, result):
      return [
        "type": "tool_finished",
        "toolId": toolID.rawValue,
        "status": result.status.rawValue,
      ]
    case .final:
      return ["type": "final"]
    case let .failure(failure):
      return ["type": "failure", "status": agentFailureCode(failure)]
    case .completed:
      return ["type": "completed"]
    }
  }

  private static func agentFailureCode(_ failure: AgentRunFailure) -> String {
    switch failure {
    case .cancelled: return "agent_cancelled"
    case .cancelledAfterToolExecution: return "cancelled_after_tool_execution"
    case .failureAfterCommittedMutation: return "failure_after_committed_mutation"
    case .modelGenerationFailed: return "model_generation_failed"
    case .malformedModelOutput: return "malformed_model_output"
    case .invalidToolCall: return "invalid_tool_call"
    case .duplicateToolCall: return "duplicate_tool_call"
    case .permissionDenied: return "permission_denied"
    case .approvalFailed: return "approval_failed"
    case .toolResultInvocationMismatch: return "tool_result_mismatch"
    case .toolExecutionFailed: return "tool_execution_failed"
    case .requiredToolActionMissing: return "required_tool_action_missing"
    case .approvedActionMissing: return "approved_action_missing"
    case .emptySanitizedFinal: return "empty_final"
    case .stepLimitReached: return "step_limit"
    case .internalEncodingFailure: return "internal_encoding_failure"
    }
  }

  private static func foundationJSON(_ value: AgentJSONValue) -> Any {
    switch value {
    case .null:
      return NSNull()
    case let .bool(value):
      return value
    case let .number(value):
      return value
    case let .string(value):
      return value
    case let .array(values):
      return values.map(foundationJSON)
    case let .object(values):
      return values.mapValues(foundationJSON)
    }
  }

  private static func riskName(_ risk: AgentToolRisk) -> String {
    switch risk {
    case .low: return "low"
    case .moderate: return "moderate"
    case .high: return "high"
    case .critical: return "critical"
    }
  }

  private static func approvalErrorMessage(_ error: AgentApprovalError) -> String {
    switch error {
    case .capacityReached: return "La file d'approbation est pleine."
    case .notFound: return "L'approbation locale est introuvable."
    case .expired: return "L'approbation locale a expiré."
    case .notPending: return "L'approbation n'est plus en attente."
    case .notApproved: return "L'action n'a pas été approuvée."
    case .rejected: return "L'action a été rejetée."
    case .alreadyConsumed: return "L'approbation a déjà été utilisée."
    case .bindingMismatch: return "L'approbation ne correspond pas à l'action."
    }
  }

  private static func iso8601(_ date: Date) -> String {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter.string(from: date)
  }

  private static func outlookConnectionPayload(
    _ status: MicrosoftGraphOAuthStatus
  ) -> [String: Any] {
    [
      "configured": status.configured,
      "connected": status.connected,
      "account": nullable(status.account),
      "expiresAt": status.expiresAt.map(iso8601) ?? NSNull(),
      "requiredScopes": status.requiredScopes,
      "redirectUri": status.redirectURI,
      "detail": status.detail,
    ]
  }

  private static func outlookOAuthErrorCode(_ error: Error) -> String {
    (error as? MicrosoftGraphOAuthError)?.bridgeCode ?? "outlook_auth_failed"
  }

  private static func outlookOAuthErrorMessage(_ error: Error) -> String {
    (error as? MicrosoftGraphOAuthError)?.localizedDescription
      ?? "La connexion Outlook n'a pas pu être mise à jour."
  }

  private static func parseISO8601(_ value: String) -> Date? {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = formatter.date(from: value) { return date }
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.date(from: value)
  }

  private static func statusPayload(_ status: ModelStatus) -> [String: Any] {
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

  private static func transientStatusPayload(
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

  private static func progressPayload(_ progress: ModelProgress) -> [String: Any] {
    [
      "phase": progress.phase.rawValue,
      "fractionCompleted": progress.fractionCompleted,
      "bytesPerSecond": progress.bytesPerSecond.map { $0 as Any } ?? NSNull(),
      "detail": progress.detail,
    ]
  }

  private static func generationPayload(
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

  private static func completionPayload(
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

  private static func errorPayload(
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

  private static func nullable(_ value: String?) -> Any {
    guard let value else { return NSNull() }
    return value
  }

  private static func bridgeErrorCode(_ error: Error) -> String {
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
    case .invalidGenerationOptions:
      return "coreml_invalid_options"
    case .operationInProgress:
      return "coreml_busy"
    case .preparationCancelled:
      return "coreml_cancelled"
    case .generationCancelled:
      return "coreml_cancelled"
    }
  }

  private static func isRecoverable(_ error: Error) -> Bool {
    guard let inferenceError = error as? InferenceError else { return true }
    return inferenceError.isRecoverable
  }
}

private enum CoreMLBridgeRequestError: LocalizedError {
  case missingRequest
  case missingMessages
  case invalidMessageCount
  case invalidMessage(index: Int)
  case invalidSystemPrompt

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
    case .invalidSystemPrompt:
      return "Le prompt systeme doit etre une chaine non vide de 32 000 octets ou moins."
    }
  }
}

private enum AgentBridgeRequestError: LocalizedError {
  case missingRequest
  case unexpectedFields
  case invalidRunID
  case invalidOwner
  case invalidPrompt
  case invalidHistory
  case invalidHistoryMessage(Int)
  case invalidApprovalRecordID
  case invalidToolScope
  case invalidMaxSteps
  case invalidToolID
  case invalidArguments
  case invalidExpiry

  var errorDescription: String? {
    switch self {
    case .missingRequest:
      return "La requête locale de l'agent est absente."
    case .unexpectedFields:
      return "La requête locale contient des champs inattendus."
    case .invalidRunID:
      return "L'identifiant d'exécution doit être un UUID."
    case .invalidOwner:
      return "Le propriétaire local est invalide."
    case .invalidPrompt:
      return "Le prompt local est vide ou trop long."
    case .invalidHistory:
      return "L'historique local doit contenir au plus 50 messages."
    case let .invalidHistoryMessage(index):
      return "Le message local \(index) est invalide."
    case .invalidApprovalRecordID:
      return "L'identifiant d'approbation est invalide."
    case .invalidToolScope:
      return "La portée d'outils locale est invalide ou incompatible avec l'intention."
    case .invalidMaxSteps:
      return "Le nombre d'étapes doit être compris entre 1 et 8."
    case .invalidToolID:
      return "L'outil proposé est inconnu ou non canonique."
    case .invalidArguments:
      return "Les arguments d'outil ne sont pas du JSON local valide."
    case .invalidExpiry:
      return "L'expiration de l'approbation est invalide."
    }
  }
}
