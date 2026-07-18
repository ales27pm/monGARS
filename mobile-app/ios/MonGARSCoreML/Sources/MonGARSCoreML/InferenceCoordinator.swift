import Foundation
import Tokenizers

#if canImport(CoreML)
import CoreML

@available(iOS 18.0, macOS 15.0, *)
public actor InferenceCoordinator {
  private enum Operation {
    case prepare
    case generate
  }

  private let store: ModelStore
  private var runner: StatefulCoreMLRunner?
  private var tokenizer: (any Tokenizer)?
  private var phase: InferencePhase = .notDownloaded
  private var lastError: String?
  private var activeOperation: Operation?
  private var shouldUnloadWhenIdle = false

  public init() {
    store = ModelStore()
  }

  public func status() -> ModelStatus {
    #if targetEnvironment(simulator)
    return ModelStatus(
      phase: .unavailable,
      detail: InferenceError.simulatorUnsupported.localizedDescription
    )
    #else
    let installed = store.isInstalled()
    if installed, phase == .notDownloaded {
      phase = .ready
    } else if !installed, phase == .ready {
      phase = .notDownloaded
    }
    return ModelStatus(
      phase: phase,
      detail: lastError,
      installedBytes: installed ? MonGARSModelManifest.installedBytes : 0
    )
    #endif
  }

  @discardableResult
  public func prepareModel(
    progress: @escaping @Sendable (ModelProgress) -> Void
  ) async throws -> ModelStatus {
    try ensurePhysicalDevice()
    try begin(.prepare)
    defer { finishOperation() }
    phase = .downloading
    lastError = nil

    do {
      _ = try await store.prepare { update in
        progress(update)
      }
      try Task.checkCancellation()
      phase = .loading
      progress(
        ModelProgress(
          phase: .loading,
          fractionCompleted: 1,
          detail: "Chargement sur le Neural Engine"
        )
      )
      try await loadIfNeeded()
      phase = .ready
      return status()
    } catch {
      if isCancellation(error) {
        lastError = nil
        phase = store.isInstalled() ? .ready : .notDownloaded
        throw InferenceError.preparationCancelled
      }
      phase = .error
      lastError = error.localizedDescription
      throw error
    }
  }

  public func generate(
    messages: [ChatMessage],
    options: GenerationOptions,
    onUpdate: @escaping @Sendable (GenerationUpdate) async -> Void
  ) async throws -> GenerationResult {
    try ensurePhysicalDevice()
    try begin(.generate)
    defer { finishOperation() }
    lastError = nil

    do {
      try Task.checkCancellation()
      guard store.isInstalled() else { throw InferenceError.modelNotInstalled }
      try await loadIfNeeded()
      guard let runner, let tokenizer else {
        throw InferenceError.invalidModel("Moteur ou tokenizer absent.")
      }

      let prompt = try PromptBuilder.build(
        messages: messages,
        tokenizer: tokenizer,
        maxNewTokens: options.maxNewTokens
      )
      phase = .generating
      lastError = nil
      let result = try await runner.generate(
        promptTokens: prompt,
        requestedOptions: options,
        onUpdate: onUpdate
      )
      phase = .ready
      return result
    } catch {
      if isCancellation(error) {
        runner = nil
        lastError = nil
        phase = store.isInstalled() ? .ready : .notDownloaded
        throw InferenceError.generationCancelled
      }

      if let inferenceError = error as? InferenceError {
        lastError = inferenceError.localizedDescription
        switch inferenceError {
        case .emptyPrompt, .promptTooLong, .invalidGenerationOptions:
          // Invalid user input must not make a healthy, verified model unusable.
          phase = .ready
        case .thermalCritical:
          runner = nil
          phase = .ready
        default:
          runner = nil
          phase = .error
        }
        throw inferenceError
      }

      runner = nil
      phase = .error
      lastError = error.localizedDescription
      throw error
    }
  }

  public func unloadModel() {
    guard activeOperation == nil else {
      // Actor methods may interleave at every await. Defer an unload requested
      // by a direct package client instead of invalidating a runner in use.
      shouldUnloadWhenIdle = true
      return
    }
    unloadNow()
  }

  public func deleteModel() throws -> ModelStatus {
    guard activeOperation == nil else { throw InferenceError.operationInProgress }
    shouldUnloadWhenIdle = false
    runner = nil
    tokenizer = nil
    try store.deleteModel()
    phase = .notDownloaded
    lastError = nil
    return status()
  }

  private func loadIfNeeded() async throws {
    if runner != nil, tokenizer != nil { return }
    phase = .loading
    try Task.checkCancellation()
    try await store.ensureVerified()
    try Task.checkCancellation()
    let loadedTokenizer = try await AutoTokenizer.from(
      modelFolder: store.tokenizerDirectory
    )
    try Task.checkCancellation()
    let loadedRunner = try StatefulCoreMLRunner(
      modelURL: store.modelDirectory,
      tokenizer: loadedTokenizer
    )
    try Task.checkCancellation()
    tokenizer = loadedTokenizer
    runner = loadedRunner
  }

  private func unloadNow() {
    runner = nil
    tokenizer = nil
    phase = store.isInstalled() ? .ready : .notDownloaded
    lastError = nil
  }

  private func finishOperation() {
    activeOperation = nil
    guard shouldUnloadWhenIdle else { return }
    shouldUnloadWhenIdle = false
    unloadNow()
  }

  private func isCancellation(_ error: Error) -> Bool {
    if error is CancellationError || Task.isCancelled { return true }
    guard let inferenceError = error as? InferenceError else { return false }
    switch inferenceError {
    case .preparationCancelled, .generationCancelled:
      return true
    default:
      return false
    }
  }

  private func ensurePhysicalDevice() throws {
    #if targetEnvironment(simulator)
    throw InferenceError.simulatorUnsupported
    #endif
  }

  private func begin(_ operation: Operation) throws {
    guard activeOperation == nil else { throw InferenceError.operationInProgress }
    activeOperation = operation
  }
}
#endif
