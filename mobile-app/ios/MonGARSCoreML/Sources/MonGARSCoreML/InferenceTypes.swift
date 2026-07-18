import Foundation

public enum InferencePhase: String, Sendable {
  case unavailable
  case notDownloaded = "not-downloaded"
  case downloading
  case verifying
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
    self.maxNewTokens = min(
      max(maxNewTokens, 1),
      MonGARSModelManifest.maximumNewTokens
    )
    self.temperature = min(max(temperature, 0.05), 2)
    self.topK = min(max(topK, 1), 200)
    self.topP = min(max(topP, 0.05), 1)
    self.repetitionPenalty = min(max(repetitionPenalty, 1), 2)
    self.doSample = doSample
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
  case operationInProgress
  case preparationCancelled
  case generationCancelled

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
    case .operationInProgress:
      return "Une operation Core ML est deja en cours."
    case .preparationCancelled:
      return "Preparation du modele local annulee."
    case .generationCancelled:
      return "Generation locale annulee."
    }
  }
}
