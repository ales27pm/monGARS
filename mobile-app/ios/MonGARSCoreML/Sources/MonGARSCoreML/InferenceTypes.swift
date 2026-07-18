import Foundation

public enum InferencePhase: String, Sendable {
  case unavailable
  case notDownloaded = "not-downloaded"
  case downloading
  case verifying
  case compiling
  case loading
  case ready
  case generating
  case error
}

public struct ModelStatus: Sendable, Equatable {
  public let phase: InferencePhase
  public let modelID: String
  public let displayName: String
  public let revision: String
  public let installedBytes: Int64
  public let contextLength: Int
  public let minimumIOSVersion: Int
  public let detail: String?

  public init(
    phase: InferencePhase,
    detail: String? = nil,
    installedBytes: Int64 = 0
  ) {
    self.phase = phase
    modelID = MonGARSModelManifest.modelID
    displayName = MonGARSModelManifest.displayName
    revision = MonGARSModelManifest.revision
    self.installedBytes = installedBytes
    contextLength = MonGARSModelManifest.contextLength
    minimumIOSVersion = 18
    self.detail = detail
  }
}

public struct ModelProgress: Sendable, Equatable {
  public let phase: InferencePhase
  public let fractionCompleted: Double
  public let bytesPerSecond: Double?
  public let detail: String

  public init(
    phase: InferencePhase,
    fractionCompleted: Double,
    bytesPerSecond: Double? = nil,
    detail: String
  ) {
    self.phase = phase
    self.fractionCompleted = min(max(fractionCompleted, 0), 1)
    self.bytesPerSecond = bytesPerSecond
    self.detail = detail
  }
}

public struct ChatMessage: Sendable, Equatable {
  public let role: String
  public let content: String

  public init(role: String, content: String) {
    self.role = role
    self.content = content
  }
}

public struct GenerationOptions: Sendable, Equatable {
  public var maxNewTokens: Int
  public var temperature: Float
  public var topK: Int
  public var topP: Float
  public var repetitionPenalty: Float
  public var doSample: Bool

  public init(
    maxNewTokens: Int = MonGARSModelManifest.defaultMaxNewTokens,
    temperature: Float = 0.6,
    topK: Int = 20,
    topP: Float = 0.95,
    repetitionPenalty: Float = 1.08,
    doSample: Bool = true
  ) {
    let finiteTemperature = temperature.isFinite ? temperature : 0.6
    let finiteTopP = topP.isFinite ? topP : 0.95
    let finiteRepetitionPenalty = repetitionPenalty.isFinite
      ? repetitionPenalty
      : 1.08
    self.maxNewTokens = min(
      max(maxNewTokens, 1),
      MonGARSModelManifest.maximumNewTokens
    )
    self.temperature = min(max(finiteTemperature, 0.05), 2)
    self.topK = min(max(topK, 1), 200)
    self.topP = min(max(finiteTopP, 0.05), 1)
    self.repetitionPenalty = min(max(finiteRepetitionPenalty, 1), 2)
    self.doSample = doSample
  }

  func validate() throws {
    guard (1...MonGARSModelManifest.maximumNewTokens).contains(maxNewTokens) else {
      throw InferenceError.invalidGenerationOptions(
        "maxNewTokens doit respecter les limites du modele."
      )
    }
    guard temperature.isFinite, (0.05...2).contains(temperature) else {
      throw InferenceError.invalidGenerationOptions(
        "temperature doit etre finie et comprise entre 0,05 et 2."
      )
    }
    guard (1...200).contains(topK) else {
      throw InferenceError.invalidGenerationOptions(
        "topK doit etre compris entre 1 et 200."
      )
    }
    guard topP.isFinite, (0.05...1).contains(topP) else {
      throw InferenceError.invalidGenerationOptions(
        "topP doit etre fini et compris entre 0,05 et 1."
      )
    }
    guard
      repetitionPenalty.isFinite,
      (1...2).contains(repetitionPenalty)
    else {
      throw InferenceError.invalidGenerationOptions(
        "repetitionPenalty doit etre fini et compris entre 1 et 2."
      )
    }
  }
}

public struct GenerationUpdate: Sendable, Equatable {
  public let text: String
  public let generatedTokens: Int
  public let tokensPerSecond: Double

  public init(text: String, generatedTokens: Int, tokensPerSecond: Double) {
    self.text = text
    self.generatedTokens = generatedTokens
    self.tokensPerSecond = tokensPerSecond
  }
}

public struct GenerationResult: Sendable, Equatable {
  public let text: String
  public let promptTokens: Int
  public let generatedTokens: Int
  public let duration: Double
  public let tokensPerSecond: Double
  public let finishReason: String
  public let modelID: String

  public init(
    text: String,
    promptTokens: Int,
    generatedTokens: Int,
    duration: Double,
    tokensPerSecond: Double,
    finishReason: String
  ) {
    self.text = text
    self.promptTokens = promptTokens
    self.generatedTokens = generatedTokens
    self.duration = duration
    self.tokensPerSecond = tokensPerSecond
    self.finishReason = finishReason
    modelID = MonGARSModelManifest.modelID
  }
}

public enum InferenceError: LocalizedError, Sendable {
  case unsupportedOS
  case simulatorUnsupported
  case insufficientDisk(required: Int64, available: Int64)
  case modelNotInstalled
  case invalidModel(String)
  case integrityFailure(String)
  case thermalCritical
  case emptyPrompt
  case promptTooLong
  case invalidGenerationOptions(String)
  case operationInProgress
  case preparationCancelled
  case generationCancelled

  public static func isCancellation(_ error: Error) -> Bool {
    // A task's cancellation flag is sticky and can coexist with a concrete
    // failure. Classify the thrown error so the concrete failure is preserved.
    if error is CancellationError { return true }
    guard let inferenceError = error as? InferenceError else { return false }
    switch inferenceError {
    case .preparationCancelled, .generationCancelled:
      return true
    default:
      return false
    }
  }

  public var errorDescription: String? {
    switch self {
    case .unsupportedOS:
      return "L'inference Core ML locale exige iOS 18 ou une version plus recente."
    case .simulatorUnsupported:
      return "Ce modele doit etre execute sur un iPhone reel."
    case let .insufficientDisk(required, available):
      return "Espace insuffisant: \(required) octets requis, \(available) disponibles."
    case .modelNotInstalled:
      return "Le modele local n'est pas installe."
    case let .invalidModel(detail):
      return "Contrat Core ML invalide: \(detail)"
    case let .integrityFailure(path):
      return "La verification du modele a echoue pour \(path)."
    case .thermalCritical:
      return "Generation interrompue: l'iPhone est trop chaud."
    case .emptyPrompt:
      return "Le prompt local est vide."
    case .promptTooLong:
      return "Le prompt local depasse le contexte disponible."
    case let .invalidGenerationOptions(detail):
      return "Options de generation locale invalides: \(detail)"
    case .operationInProgress:
      return "Une operation Core ML est deja en cours."
    case .preparationCancelled:
      return "Preparation du modele local annulee."
    case .generationCancelled:
      return "Generation locale annulee."
    }
  }
}
