import Foundation
import MonGARSCoreML

public protocol AgentMemoryScopeProviding: Sendable {
  func currentScope() async -> String
}

public struct StaticAgentMemoryScopeProvider: AgentMemoryScopeProviding, Sendable {
  private let scope: String

  public init(scope: String = "local.default") {
    self.scope = AgentToolInput.validatedScope(scope) ?? "local.default"
  }

  public func currentScope() async -> String { scope }
}

public protocol MicrosoftGraphAccessTokenProviding: Sendable {
  func accessToken() async throws -> String?
  func hasUsableSession() async -> Bool
  func accessToken(profileScope: String, forceRefresh: Bool) async throws -> String?
  func hasUsableSession(profileScope: String) async -> Bool
  func outlookStatus(profileScope: String) async -> MicrosoftGraphOAuthStatus?
}

public extension MicrosoftGraphAccessTokenProviding {
  func hasUsableSession() async -> Bool {
    do {
      guard let token = try await accessToken()?
        .trimmingCharacters(in: .whitespacesAndNewlines) else { return false }
      return !token.isEmpty && token.utf8.count <= 16_384
        && !token.contains("\r") && !token.contains("\n")
    } catch {
      return false
    }
  }

  /// Compatibility defaults keep injected test providers source-compatible.
  /// The production OAuth provider overrides these methods and refuses all
  /// unscoped access so one monGARS owner can never inherit another's session.
  func accessToken(profileScope: String, forceRefresh: Bool) async throws -> String? {
    try await accessToken()
  }

  func hasUsableSession(profileScope: String) async -> Bool {
    await hasUsableSession()
  }

  func outlookStatus(profileScope: String) async -> MicrosoftGraphOAuthStatus? { nil }
}

public struct UnavailableMicrosoftGraphTokenProvider: MicrosoftGraphAccessTokenProviding, Sendable {
  public init() {}
  public func accessToken() async throws -> String? { nil }
  public func hasUsableSession() async -> Bool { false }
  public func accessToken(profileScope: String, forceRefresh: Bool) async throws -> String? { nil }
  public func hasUsableSession(profileScope: String) async -> Bool { false }
  public func outlookStatus(profileScope: String) async -> MicrosoftGraphOAuthStatus? { nil }
}

/// A narrow UI boundary. Implementations may present system composers or open
/// system URLs only while the application is active in the foreground.
public protocol AgentForegroundPresenting: Sendable {
  @MainActor
  func presentMessageDraft(to: String, body: String) async -> Bool
  @MainActor
  func presentMailDraft(to: String, subject: String, body: String) async -> Bool
  @MainActor
  func openPhone(number: String) async -> Bool
  @MainActor
  func openDirections(destination: String) async -> Bool
  @MainActor
  func captureCameraImage() async -> AgentCameraCaptureResult
}

public enum AgentCameraCaptureResult: Sendable, Equatable {
  case captured(pixelWidth: Int, pixelHeight: Int, bytes: Int)
  case permissionDenied
  case unavailable
  case failed
}

public struct UnavailableAgentForegroundPresenter: AgentForegroundPresenting, Sendable {
  public init() {}
  @MainActor public func presentMessageDraft(to: String, body: String) async -> Bool { false }
  @MainActor public func presentMailDraft(to: String, subject: String, body: String) async -> Bool { false }
  @MainActor public func openPhone(number: String) async -> Bool { false }
  @MainActor public func openDirections(destination: String) async -> Bool { false }
  @MainActor public func captureCameraImage() async -> AgentCameraCaptureResult { .unavailable }
}

public protocol AgentPhotoMetadataProviding: Sendable {
  func searchMetadata(query: String, limit: Int) async throws -> [AgentPhotoMetadata]
  func metadataSince(_ startDate: Date, limit: Int) async throws -> [AgentPhotoMetadata]
}

public struct UnavailableAgentPhotoMetadataProvider: AgentPhotoMetadataProviding, Sendable {
  public init() {}
  public func searchMetadata(query: String, limit: Int) async throws -> [AgentPhotoMetadata] { [] }
  public func metadataSince(_ startDate: Date, limit: Int) async throws -> [AgentPhotoMetadata] { [] }
}

public struct AgentPhotoMetadata: Codable, Equatable, Sendable {
  public let localIdentifier: String
  public let filename: String?
  public let createdAt: Date?
  public let latitude: Double?
  public let longitude: Double?
  public let mediaType: String?
  public let mediaSubtypes: [String]
  public let isFavorite: Bool?
  public let pixelWidth: Int?
  public let pixelHeight: Int?
  public let displayToken: String?
  public let queryMatched: String?

  public init(
    localIdentifier: String,
    filename: String?,
    createdAt: Date?,
    latitude: Double?,
    longitude: Double?,
    mediaType: String? = nil,
    mediaSubtypes: [String] = [],
    isFavorite: Bool? = nil,
    pixelWidth: Int? = nil,
    pixelHeight: Int? = nil,
    displayToken: String? = nil,
    queryMatched: String? = nil
  ) {
    self.localIdentifier = localIdentifier
    self.filename = filename
    self.createdAt = createdAt
    self.latitude = latitude
    self.longitude = longitude
    self.mediaType = mediaType
    self.mediaSubtypes = mediaSubtypes
    self.isFavorite = isFavorite
    self.pixelWidth = pixelWidth
    self.pixelHeight = pixelHeight
    self.displayToken = displayToken
    self.queryMatched = queryMatched
  }
}

public protocol AgentTriggerScheduling: Sendable {
  func create(arguments: AgentJSONArguments, scope: String) async -> AgentServiceResponse
  func list(scope: String) async -> AgentServiceResponse
  func cancel(id: String?, title: String?, scope: String) async -> AgentServiceResponse
  func resolveHandoff(selector: String, scope: String) async -> AgentPendingTriggerHandoff?
  func pendingHandoff(scope: String) async -> AgentPendingTriggerHandoff?
  func acknowledgePendingHandoff(id: UUID, scope: String) async -> Bool
}

public struct AgentPendingTriggerHandoff: Codable, Sendable, Equatable {
  public let id: UUID
  public let title: String
  public let prompt: String
  public let repeats: Bool

  public init(id: UUID, title: String, prompt: String, repeats: Bool) {
    self.id = id
    self.title = title
    self.prompt = prompt
    self.repeats = repeats
  }
}

public protocol AgentAlarmServing: Sendable {
  @MainActor
  func execute(operation: AgentHostOperation, arguments: AgentJSONArguments) async -> AgentServiceResponse
}

public struct AgentServiceResponse: Sendable, Equatable {
  public let status: AgentToolResultStatus
  public let text: String
  public let payload: AgentJSONValue?
  public let errorCode: String?

  public init(
    status: AgentToolResultStatus,
    text: String,
    payload: AgentJSONValue? = nil,
    errorCode: String? = nil
  ) {
    self.status = status
    self.text = text
    self.payload = payload
    self.errorCode = errorCode
  }

  public static func success(_ text: String, payload: AgentJSONValue? = nil) -> Self {
    .init(status: .success, text: text, payload: payload)
  }

  public static func unavailable(_ text: String, code: String = "tool_unavailable") -> Self {
    .init(status: .unavailable, text: text, errorCode: code)
  }

  public static func denied(_ text: String, code: String) -> Self {
    .init(status: .denied, text: text, errorCode: code)
  }

  public static func failed(_ text: String, code: String) -> Self {
    .init(status: .failed, text: text, errorCode: code)
  }
}

enum AgentToolInput {
  static func requiredString(
    _ key: String,
    in arguments: AgentJSONArguments,
    maximumBytes: Int = 16_000
  ) -> String? {
    guard let value = arguments[key]?.stringValue?
      .trimmingCharacters(in: .whitespacesAndNewlines),
      !value.isEmpty,
      value.utf8.count <= maximumBytes else { return nil }
    return value
  }

  static func optionalString(
    _ key: String,
    in arguments: AgentJSONArguments,
    maximumBytes: Int = 16_000
  ) -> String? {
    guard let raw = arguments[key]?.stringValue else { return nil }
    let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty, value.utf8.count <= maximumBytes else { return nil }
    return value
  }

  static func integer(
    _ key: String,
    in arguments: AgentJSONArguments,
    range: ClosedRange<Int>
  ) -> Int? {
    guard let number = arguments[key]?.numberValue,
      number.isFinite,
      number.rounded(.towardZero) == number,
      number >= Double(range.lowerBound),
      number <= Double(range.upperBound) else { return nil }
    return Int(number)
  }

  static func bool(_ key: String, in arguments: AgentJSONArguments) -> Bool? {
    arguments[key]?.boolValue
  }

  static func validatedScope(_ raw: String) -> String? {
    let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty, value.utf8.count <= 128 else { return nil }
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-:"))
    guard value.unicodeScalars.allSatisfy(allowed.contains) else { return nil }
    return value
  }
}

enum AgentToolResultFactory {
  static func make(
    invocation: AgentToolInvocation,
    response: AgentServiceResponse
  ) -> AgentToolResult {
    let maximum = AgentToolCatalog.definition(for: invocation.toolID.rawValue)?
      .maximumOutputCharacters ?? 2_400
    return .init(
      invocationID: invocation.id,
      status: response.status,
      text: bounded(response.text, maximumCharacters: maximum),
      payload: boundedPayload(response.payload, maximumBytes: maximum * 3),
      errorCode: response.errorCode
    )
  }

  static func bounded(_ text: String, maximumCharacters: Int) -> String {
    let normalized = text
      .replacingOccurrences(of: "\u{0000}", with: "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard normalized.count > maximumCharacters else { return normalized }
    let end = normalized.index(normalized.startIndex, offsetBy: maximumCharacters)
    return String(normalized[..<end]) + "…"
  }

  private static func boundedPayload(
    _ payload: AgentJSONValue?,
    maximumBytes: Int
  ) -> AgentJSONValue? {
    guard let payload,
      let encoded = try? payload.canonicalJSONString(),
      encoded.utf8.count <= maximumBytes else { return nil }
    return payload
  }
}

extension AgentJSONValue {
  static func fromFoundation(_ value: Any) -> AgentJSONValue? {
    try? AgentFoundationJSON.decode(value)
  }
}

public enum SafeAgentFilePath {
  /// Resolves a user-supplied relative path under a fixed import root. Both the
  /// root and candidate are standardized and symlinks are resolved before the
  /// containment check.
  public static func resolve(name: String, under root: URL) -> URL? {
    let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty,
      trimmed.utf8.count <= 1_024,
      !trimmed.hasPrefix("/"),
      !trimmed.contains("\u{0000}") else { return nil }

    let canonicalRoot = root.standardizedFileURL.resolvingSymlinksInPath()
    let candidate = canonicalRoot
      .appendingPathComponent(trimmed, isDirectory: false)
      .standardizedFileURL
      .resolvingSymlinksInPath()
    let rootPath = canonicalRoot.path.hasSuffix("/")
      ? canonicalRoot.path
      : canonicalRoot.path + "/"
    guard candidate.path.hasPrefix(rootPath), candidate.path != canonicalRoot.path else {
      return nil
    }
    return candidate
  }
}
