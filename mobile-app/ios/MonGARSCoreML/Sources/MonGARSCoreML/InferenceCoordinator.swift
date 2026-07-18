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
    defer { activeOperation = nil }
    phase = .downloading
    lastError = nil

    do {
      _ = try await store.prepare { update in
        progress(update)
      }
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
    } catch is CancellationError {
      phase = store.isInstalled() ? .ready : .notDownloaded
      throw InferenceError.preparationCancelled
    } catch {
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
    defer { activeOperation = nil }
    guard store.isInstalled() else { throw InferenceError.modelNotInstalled }

    do {
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
    } catch is CancellationError {
      runner = nil
      phase = .ready
      throw InferenceError.generationCancelled
    } catch let error as InferenceError {
      lastError = error.localizedDescription
      switch error {
      case .emptyPrompt, .promptTooLong:
        // Invalid user input must not make a healthy, verified model unusable.
        phase = .ready
      case .thermalCritical:
        runner = nil
        phase = .ready
      default:
        runner = nil
        phase = .error
      }
      throw error
    } catch {
      runner = nil
      phase = .error
      lastError = error.localizedDescription
      throw error
    }
  }

  public func unloadModel() {
    runner = nil
    tokenizer = nil
    phase = store.isInstalled() ? .ready : .notDownloaded
  }

  public func deleteModel() throws -> ModelStatus {
    guard activeOperation == nil else { throw InferenceError.operationInProgress }
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
    try await store.ensureVerified()
    let loadedTokenizer = try await AutoTokenizer.from(
      modelFolder: store.tokenizerDirectory
    )
    let loadedRunner = try StatefulCoreMLRunner(
      modelURL: store.modelDirectory,
      tokenizer: loadedTokenizer
    )
    tokenizer = loadedTokenizer
    runner = loadedRunner
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
