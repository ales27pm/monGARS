import CryptoKit
import Foundation
import Security

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

#if os(iOS)
import AuthenticationServices
import UIKit
#endif

public enum MicrosoftGraphOAuthScopes {
  /// The single delegated grant used by all 16 canonical Outlook tools.
  /// `Mail.ReadWrite` covers mailbox reads and mutations, `Mail.Send` covers
  /// sends/replies/forwards, `User.Read` resolves the connected account, and
  /// `offline_access` permits proactive refresh without a client secret.
  public static let outlookTools = [
    "User.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "offline_access",
  ]

  private static let grantOnlyScopes = Set(["offline_access"])

  public static func grantedScopesSatisfy(_ grantedScopes: String?) -> Bool {
    let required = Set(
      outlookTools
        .map { $0.lowercased() }
        .filter { !grantOnlyScopes.contains($0) }
    )
    let granted = Set(
      (grantedScopes ?? "")
        .split(whereSeparator: { $0.isWhitespace })
        .map { $0.lowercased() }
    )
    return required.isSubset(of: granted)
  }

  static var authorizationValue: String {
    outlookTools.sorted().joined(separator: " ")
  }
}

public enum MicrosoftGraphOAuthError: LocalizedError, Sendable, Equatable {
  case notConfigured
  case redirectSchemeMissing
  case interactiveSignInUnavailable
  case signInAlreadyRunning
  case signInCancelled
  case invalidAuthorizationResponse
  case invalidState
  case consentRequired
  case interactionRequired
  case invalidGrant
  case invalidScope
  case tokenEndpointThrottled
  case tokenEndpointUnavailable
  case invalidTokenResponse
  case accountLookupFailed
  case keychainFailure(Int32)

  public var errorDescription: String? {
    switch self {
    case .notConfigured:
      return "Microsoft Outlook n'est pas configuré pour cette version de l'app."
    case .redirectSchemeMissing:
      return "Le schéma de redirection Microsoft n'est pas enregistré dans cette version de l'app."
    case .interactiveSignInUnavailable:
      return "La connexion Microsoft interactive exige l'app iOS au premier plan."
    case .signInAlreadyRunning:
      return "Une connexion Microsoft est déjà en cours."
    case .signInCancelled:
      return "La connexion Microsoft a été annulée."
    case .invalidAuthorizationResponse, .invalidState:
      return "La réponse de connexion Microsoft n'a pas pu être validée."
    case .consentRequired:
      return "Microsoft exige votre consentement pour les autorisations Outlook demandées."
    case .interactionRequired:
      return "Microsoft exige une nouvelle connexion interactive."
    case .invalidGrant:
      return "La session Outlook a expiré ou a été révoquée; reconnectez le compte."
    case .invalidScope:
      return "Microsoft n'a pas accordé toutes les autorisations Outlook requises."
    case .tokenEndpointThrottled:
      return "Microsoft limite temporairement les demandes de connexion."
    case .tokenEndpointUnavailable:
      return "Le service de connexion Microsoft est temporairement indisponible."
    case .invalidTokenResponse:
      return "Microsoft a retourné une session illisible."
    case .accountLookupFailed:
      return "Le compte Microsoft connecté n'a pas pu être vérifié."
    case .keychainFailure:
      return "La session Outlook n'a pas pu être protégée dans le trousseau iOS."
    }
  }

  public var bridgeCode: String {
    switch self {
    case .notConfigured: return "outlook_not_configured"
    case .redirectSchemeMissing: return "outlook_redirect_not_configured"
    case .interactiveSignInUnavailable: return "outlook_interactive_unavailable"
    case .signInAlreadyRunning: return "outlook_sign_in_in_progress"
    case .signInCancelled: return "outlook_sign_in_cancelled"
    case .invalidAuthorizationResponse: return "outlook_invalid_authorization_response"
    case .invalidState: return "outlook_invalid_oauth_state"
    case .consentRequired: return "outlook_consent_required"
    case .interactionRequired: return "outlook_interaction_required"
    case .invalidGrant: return "outlook_invalid_grant"
    case .invalidScope: return "outlook_invalid_scope"
    case .tokenEndpointThrottled: return "outlook_auth_throttled"
    case .tokenEndpointUnavailable: return "outlook_auth_unavailable"
    case .invalidTokenResponse: return "outlook_invalid_token_response"
    case .accountLookupFailed: return "outlook_account_lookup_failed"
    case .keychainFailure: return "outlook_keychain_failure"
    }
  }
}

public struct MicrosoftGraphOAuthConfiguration: Sendable, Equatable {
  public static let clientIDInfoKey = "MONGARSMicrosoftClientID"
  public static let runtimeClientIDDefaultsKey = "MONGARSMicrosoftClientIDOverride"

  public let clientID: String
  public let bundleIdentifier: String
  public let callbackScheme: String
  public let redirectURI: String

  public static func load(
    bundle: Bundle = .main,
    userDefaults: UserDefaults = .standard
  ) throws -> Self {
    let clientID = resolvedClientID(
      bundled: bundle.object(forInfoDictionaryKey: clientIDInfoKey) as? String,
      runtime: userDefaults.string(forKey: runtimeClientIDDefaultsKey)
    )
    let bundleIdentifier = bundle.bundleIdentifier ?? ""
    let registeredSchemes = Set(
      ((bundle.object(forInfoDictionaryKey: "CFBundleURLTypes") as? [[String: Any]]) ?? [])
        .flatMap { ($0["CFBundleURLSchemes"] as? [String]) ?? [] }
    )
    return try validated(
      clientID: clientID,
      bundleIdentifier: bundleIdentifier,
      registeredSchemes: registeredSchemes
    )
  }

  /// Stores only the public Entra application identifier. A valid build-time
  /// Info.plist value remains authoritative; this fallback lets an unbranded
  /// build be configured from the app's Settings surface without accepting a
  /// client secret or exposing OAuth tokens to React Native.
  @discardableResult
  public static func configureRuntimeClientID(
    _ rawClientID: String,
    bundle: Bundle = .main,
    userDefaults: UserDefaults = .standard
  ) throws -> Self {
    let bundleIdentifier = bundle.bundleIdentifier ?? ""
    let registeredSchemes = Set(
      ((bundle.object(forInfoDictionaryKey: "CFBundleURLTypes") as? [[String: Any]]) ?? [])
        .flatMap { ($0["CFBundleURLSchemes"] as? [String]) ?? [] }
    )
    let configuration = try validated(
      clientID: rawClientID,
      bundleIdentifier: bundleIdentifier,
      registeredSchemes: registeredSchemes
    )
    userDefaults.set(configuration.clientID, forKey: runtimeClientIDDefaultsKey)
    return configuration
  }

  static func resolvedClientID(bundled: String?, runtime: String?) -> String? {
    if normalizedClientID(bundled) != nil { return bundled }
    if normalizedClientID(runtime) != nil { return runtime }
    return nil
  }

  static func validated(
    clientID rawClientID: String?,
    bundleIdentifier rawBundleIdentifier: String,
    registeredSchemes: Set<String>
  ) throws -> Self {
    let clientIDValue = rawClientID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let bundleIdentifier = rawBundleIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !clientIDValue.isEmpty,
      !clientIDValue.contains("$("),
      let parsedClientID = UUID(uuidString: clientIDValue),
      !bundleIdentifier.isEmpty,
      bundleIdentifier.utf8.count <= 255,
      !bundleIdentifier.unicodeScalars.contains(where: {
        CharacterSet.controlCharacters.union(.whitespacesAndNewlines).contains($0)
      }) else {
      throw MicrosoftGraphOAuthError.notConfigured
    }

    let callbackScheme = "msauth.\(bundleIdentifier)"
    guard registeredSchemes.contains(where: {
      $0.caseInsensitiveCompare(callbackScheme) == .orderedSame
    }) else {
      throw MicrosoftGraphOAuthError.redirectSchemeMissing
    }
    return .init(
      clientID: parsedClientID.uuidString.lowercased(),
      bundleIdentifier: bundleIdentifier,
      callbackScheme: callbackScheme,
      redirectURI: "\(callbackScheme)://auth"
    )
  }

  private static func normalizedClientID(_ rawValue: String?) -> String? {
    let value = rawValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !value.isEmpty, !value.contains("$("), let parsed = UUID(uuidString: value) else {
      return nil
    }
    return parsed.uuidString.lowercased()
  }

  static func expectedRedirectURI(bundle: Bundle = .main) -> String {
    let bundleIdentifier = bundle.bundleIdentifier?.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let bundleIdentifier, !bundleIdentifier.isEmpty else {
      return "msauth.<bundle-id>://auth"
    }
    return "msauth.\(bundleIdentifier)://auth"
  }
}

public struct MicrosoftGraphOAuthStatus: Sendable, Equatable {
  public let configured: Bool
  public let connected: Bool
  public let account: String?
  public let expiresAt: Date?
  public let requiredScopes: [String]
  public let redirectURI: String
  public let detail: String
}

private struct MicrosoftGraphOAuthTokenSet: Codable, Sendable, Equatable {
  let accessToken: String
  let refreshToken: String?
  let expiresAt: Date
  let grantedScopes: String
  let tokenType: String

  var needsProactiveRefresh: Bool {
    expiresAt.timeIntervalSinceNow < 300
  }

  var hasUnexpiredAccessToken: Bool {
    expiresAt > Date()
      && Self.validCredential(accessToken)
      && tokenType.caseInsensitiveCompare("Bearer") == .orderedSame
  }

  var hasRefreshToken: Bool {
    guard let refreshToken else { return false }
    return Self.validCredential(refreshToken)
  }

  static func validCredential(_ value: String) -> Bool {
    !value.isEmpty && value.utf8.count <= 16_384
      && !value.unicodeScalars.contains(where: {
        CharacterSet.controlCharacters.union(.newlines).contains($0)
      })
  }
}

private struct MicrosoftGraphOAuthAccount: Codable, Sendable, Equatable {
  let id: String
  let displayName: String?
  let userPrincipalName: String?
  let mail: String?

  var displayAccount: String? {
    [mail, userPrincipalName, displayName]
      .compactMap { value in
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty, trimmed.utf8.count <= 512 else { return nil }
        return trimmed
      }
      .first
  }
}

private struct MicrosoftGraphOAuthSession: Codable, Sendable, Equatable {
  let profileScope: String
  let clientID: String
  let account: MicrosoftGraphOAuthAccount
  let token: MicrosoftGraphOAuthTokenSet
}

private enum MicrosoftGraphOAuthKeychainStore {
  static func load(
    bundleIdentifier: String,
    profileScope: String
  ) throws -> MicrosoftGraphOAuthSession? {
    var query = baseQuery(bundleIdentifier: bundleIdentifier, profileScope: profileScope)
    query[kSecReturnData as String] = true
    query[kSecMatchLimit as String] = kSecMatchLimitOne
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    if status == errSecItemNotFound { return nil }
    guard status == errSecSuccess, let data = item as? Data else {
      throw MicrosoftGraphOAuthError.keychainFailure(status)
    }
    guard let session = try? JSONDecoder().decode(MicrosoftGraphOAuthSession.self, from: data) else {
      try? clear(bundleIdentifier: bundleIdentifier, profileScope: profileScope)
      return nil
    }
    guard session.profileScope == profileScope else {
      try? clear(bundleIdentifier: bundleIdentifier, profileScope: profileScope)
      return nil
    }
    return session
  }

  static func save(
    _ session: MicrosoftGraphOAuthSession,
    bundleIdentifier: String,
    profileScope: String
  ) throws {
    guard session.profileScope == profileScope else {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    let data: Data
    do {
      data = try JSONEncoder().encode(session)
    } catch {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    let base = baseQuery(bundleIdentifier: bundleIdentifier, profileScope: profileScope)
    let update: [String: Any] = [
      kSecValueData as String: data,
      kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
    ]
    let updateStatus = SecItemUpdate(base as CFDictionary, update as CFDictionary)
    if updateStatus == errSecSuccess { return }
    guard updateStatus == errSecItemNotFound else {
      throw MicrosoftGraphOAuthError.keychainFailure(updateStatus)
    }
    var addition = base
    addition.merge(update) { _, new in new }
    let addStatus = SecItemAdd(addition as CFDictionary, nil)
    guard addStatus == errSecSuccess else {
      throw MicrosoftGraphOAuthError.keychainFailure(addStatus)
    }
  }

  static func clear(bundleIdentifier: String, profileScope: String) throws {
    let status = SecItemDelete(
      baseQuery(bundleIdentifier: bundleIdentifier, profileScope: profileScope) as CFDictionary
    )
    guard status == errSecSuccess || status == errSecItemNotFound else {
      throw MicrosoftGraphOAuthError.keychainFailure(status)
    }
  }

  private static func baseQuery(
    bundleIdentifier: String,
    profileScope: String
  ) -> [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: "\(bundleIdentifier).mongars.microsoftgraph.oauth",
      // profileScope is already a SHA-256-derived opaque identifier.
      kSecAttrAccount as String: profileScope,
    ]
  }
}

private struct MicrosoftGraphOAuthTokenResponse: Decodable {
  let tokenType: String
  let scope: String?
  let expiresIn: Int
  let accessToken: String
  let refreshToken: String?

  enum CodingKeys: String, CodingKey {
    case tokenType = "token_type"
    case scope
    case expiresIn = "expires_in"
    case accessToken = "access_token"
    case refreshToken = "refresh_token"
  }
}

private struct MicrosoftGraphOAuthErrorResponse: Decodable {
  let error: String?
  let errorDescription: String?
  let suberror: String?

  enum CodingKeys: String, CodingKey {
    case error
    case errorDescription = "error_description"
    case suberror
  }
}

private final class MicrosoftGraphOAuthNoRedirectDelegate: NSObject,
  URLSessionTaskDelegate, @unchecked Sendable
{
  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    completionHandler(nil)
  }
}

private final class MicrosoftGraphOAuthHTTPClient: @unchecked Sendable {
  private let session: URLSession

  init() {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = 20
    configuration.timeoutIntervalForResource = 30
    configuration.httpCookieStorage = nil
    configuration.urlCache = nil
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    session = URLSession(
      configuration: configuration,
      delegate: MicrosoftGraphOAuthNoRedirectDelegate(),
      delegateQueue: nil
    )
  }

  func send(_ request: URLRequest, maximumBytes: Int = 256 * 1_024) async throws
    -> (Data, HTTPURLResponse)
  {
    let (data, response) = try await session.data(for: request)
    guard data.count <= maximumBytes,
      let http = response as? HTTPURLResponse else {
      throw MicrosoftGraphOAuthError.tokenEndpointUnavailable
    }
    return (data, http)
  }
}

public actor MicrosoftGraphOAuthTokenProvider: MicrosoftGraphAccessTokenProviding {
  public static let shared = MicrosoftGraphOAuthTokenProvider()

  private static let authorizeURL = URL(
    string: "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
  )!
  private static let tokenURL = URL(
    string: "https://login.microsoftonline.com/common/oauth2/v2.0/token"
  )!
  private static let accountURL = URL(
    string: "https://graph.microsoft.com/v1.0/me?$select=id,displayName,userPrincipalName,mail"
  )!

  private struct RefreshWork: Sendable {
    let id: UUID
    let generation: UInt64
    let task: Task<MicrosoftGraphOAuthTokenSet, Error>
  }

  private let httpClient = MicrosoftGraphOAuthHTTPClient()
  private var connectingScopes = Set<String>()
  private var refreshWorkByScope: [String: RefreshWork] = [:]
  private var generationByScope: [String: UInt64] = [:]

  public init() {}

  /// OAuth sessions are never available through the legacy app-global API.
  /// Callers must present the opaque scope derived from the active monGARS owner.
  public func hasUsableSession() async -> Bool {
    false
  }

  public func accessToken() async throws -> String? {
    nil
  }

  public func hasUsableSession(profileScope: String) async -> Bool {
    guard Self.isValidProfileScope(profileScope),
      let configuration = try? MicrosoftGraphOAuthConfiguration.load(),
      let session = try? MicrosoftGraphOAuthKeychainStore.load(
        bundleIdentifier: configuration.bundleIdentifier,
        profileScope: profileScope
      ),
      session.profileScope == profileScope,
      session.clientID.caseInsensitiveCompare(configuration.clientID) == .orderedSame,
      MicrosoftGraphOAuthScopes.grantedScopesSatisfy(session.token.grantedScopes)
    else { return false }
    return session.token.hasUnexpiredAccessToken || session.token.hasRefreshToken
  }

  public func accessToken(
    profileScope: String,
    forceRefresh: Bool
  ) async throws -> String? {
    guard Self.isValidProfileScope(profileScope) else { return nil }
    // Do not refresh an old cached account while an interactive reconnect for
    // the same owner is in flight; its completion could otherwise race the
    // new account commit.
    guard !connectingScopes.contains(profileScope) else {
      throw MicrosoftGraphOAuthError.signInAlreadyRunning
    }
    let configuration = try MicrosoftGraphOAuthConfiguration.load()
    guard let cached = try MicrosoftGraphOAuthKeychainStore.load(
        bundleIdentifier: configuration.bundleIdentifier,
        profileScope: profileScope
      ), cached.profileScope == profileScope,
      cached.clientID.caseInsensitiveCompare(configuration.clientID) == .orderedSame else {
      return nil
    }
    guard MicrosoftGraphOAuthScopes.grantedScopesSatisfy(cached.token.grantedScopes) else {
      throw MicrosoftGraphOAuthError.invalidScope
    }
    if !forceRefresh,
      cached.token.hasUnexpiredAccessToken,
      !cached.token.needsProactiveRefresh {
      return cached.token.accessToken
    }
    guard let refreshToken = cached.token.refreshToken,
      MicrosoftGraphOAuthTokenSet.validCredential(refreshToken) else {
      if !forceRefresh, cached.token.hasUnexpiredAccessToken {
        return cached.token.accessToken
      }
      throw MicrosoftGraphOAuthError.interactionRequired
    }

    let generation = currentGeneration(for: profileScope)
    do {
      let refreshed = try await refresh(
        configuration: configuration,
        cached: cached,
        refreshToken: refreshToken,
        profileScope: profileScope,
        generation: generation
      )
      return refreshed.accessToken
    } catch let error as MicrosoftGraphOAuthError
      where error == .invalidGrant || error == .interactionRequired
        || error == .consentRequired || error == .invalidScope
    {
      // A stale refresh must never clear a newer connect result.
      if currentGeneration(for: profileScope) == generation {
        try? MicrosoftGraphOAuthKeychainStore.clear(
          bundleIdentifier: configuration.bundleIdentifier,
          profileScope: profileScope
        )
      }
      throw error
    }
  }

  public func status(rawOwnerID: String) async -> MicrosoftGraphOAuthStatus {
    guard let profileScope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else {
      return Self.invalidOwnerStatus()
    }
    return await outlookStatus(profileScope: profileScope) ?? Self.invalidOwnerStatus()
  }

  public func outlookStatus(profileScope: String) async -> MicrosoftGraphOAuthStatus? {
    guard Self.isValidProfileScope(profileScope) else { return Self.invalidOwnerStatus() }
    let redirectURI = MicrosoftGraphOAuthConfiguration.expectedRedirectURI()
    let configuration: MicrosoftGraphOAuthConfiguration
    do {
      configuration = try MicrosoftGraphOAuthConfiguration.load()
    } catch let error as MicrosoftGraphOAuthError {
      return .init(
        configured: false,
        connected: false,
        account: nil,
        expiresAt: nil,
        requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
        redirectURI: redirectURI,
        detail: error.localizedDescription
      )
    } catch {
      return .init(
        configured: false,
        connected: false,
        account: nil,
        expiresAt: nil,
        requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
        redirectURI: redirectURI,
        detail: MicrosoftGraphOAuthError.notConfigured.localizedDescription
      )
    }

    do {
      guard let session = try MicrosoftGraphOAuthKeychainStore.load(
        bundleIdentifier: configuration.bundleIdentifier,
        profileScope: profileScope
      ) else {
        return Self.disconnectedStatus(configuration: configuration)
      }
      guard session.profileScope == profileScope,
        session.clientID.caseInsensitiveCompare(configuration.clientID) == .orderedSame,
        MicrosoftGraphOAuthScopes.grantedScopesSatisfy(session.token.grantedScopes),
        session.token.hasUnexpiredAccessToken || session.token.hasRefreshToken else {
        return .init(
          configured: true,
          connected: false,
          account: nil,
          expiresAt: nil,
          requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
          redirectURI: configuration.redirectURI,
          detail: "La session Outlook enregistrée doit être reconnectée."
        )
      }
      return .init(
        configured: true,
        connected: true,
        account: session.account.displayAccount ?? session.account.id,
        expiresAt: session.token.expiresAt,
        requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
        redirectURI: configuration.redirectURI,
        detail: "Compte Outlook connecté; les jetons restent dans le trousseau iOS."
      )
    } catch {
      return .init(
        configured: true,
        connected: false,
        account: nil,
        expiresAt: nil,
        requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
        redirectURI: configuration.redirectURI,
        detail: MicrosoftGraphOAuthError.keychainFailure(errSecInteractionNotAllowed)
          .localizedDescription
      )
    }
  }

  public func connect(rawOwnerID: String) async throws -> MicrosoftGraphOAuthStatus {
    guard let profileScope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else {
      throw MicrosoftGraphOAuthError.interactiveSignInUnavailable
    }
    return try await connect(profileScope: profileScope)
  }

  private func connect(profileScope: String) async throws -> MicrosoftGraphOAuthStatus {
    guard Self.isValidProfileScope(profileScope) else {
      throw MicrosoftGraphOAuthError.interactiveSignInUnavailable
    }
    guard !connectingScopes.contains(profileScope) else {
      throw MicrosoftGraphOAuthError.signInAlreadyRunning
    }
    connectingScopes.insert(profileScope)
    defer { connectingScopes.remove(profileScope) }

    let configuration = try MicrosoftGraphOAuthConfiguration.load()
    let generation = advanceGeneration(for: profileScope)
    if let refreshWork = refreshWorkByScope.removeValue(forKey: profileScope) {
      refreshWork.task.cancel()
      _ = await refreshWork.task.result
    }
    try Task.checkCancellation()
    guard currentGeneration(for: profileScope) == generation else {
      throw CancellationError()
    }
#if os(iOS)
    let verifier = try Self.makeCodeVerifier()
    let state = UUID().uuidString.lowercased()
    let authorizationURL = try Self.authorizationURL(
      configuration: configuration,
      state: state,
      codeChallenge: Self.codeChallenge(verifier: verifier)
    )
    let webSession = await MainActor.run { MicrosoftGraphOAuthWebSession() }
    let callback = try await webSession.authenticate(
      url: authorizationURL,
      callbackScheme: configuration.callbackScheme
    )
    try Task.checkCancellation()
    guard currentGeneration(for: profileScope) == generation else {
      throw CancellationError()
    }
    let code = try Self.authorizationCode(
      from: callback,
      configuration: configuration,
      expectedState: state
    )
    let token = try await requestToken(
      configuration: configuration,
      form: [
        "client_id": configuration.clientID,
        "scope": MicrosoftGraphOAuthScopes.authorizationValue,
        "code": code,
        "redirect_uri": configuration.redirectURI,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
      ],
      existingRefreshToken: nil,
      existingScopes: nil
    )
    guard MicrosoftGraphOAuthScopes.grantedScopesSatisfy(token.grantedScopes) else {
      throw MicrosoftGraphOAuthError.invalidScope
    }
    let account = try await fetchAccount(accessToken: token.accessToken)
    try Task.checkCancellation()
    // There is deliberately no suspension between the epoch check and save.
    guard currentGeneration(for: profileScope) == generation else {
      throw CancellationError()
    }
    try MicrosoftGraphOAuthKeychainStore.save(
      .init(
        profileScope: profileScope,
        clientID: configuration.clientID,
        account: account,
        token: token
      ),
      bundleIdentifier: configuration.bundleIdentifier,
      profileScope: profileScope
    )
    return await outlookStatus(profileScope: profileScope) ?? Self.invalidOwnerStatus()
#else
    throw MicrosoftGraphOAuthError.interactiveSignInUnavailable
#endif
  }

  public func disconnect(rawOwnerID: String) async throws -> MicrosoftGraphOAuthStatus {
    guard let profileScope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else {
      throw MicrosoftGraphOAuthError.interactiveSignInUnavailable
    }
    return try await disconnect(profileScope: profileScope)
  }

  private func disconnect(profileScope: String) async throws -> MicrosoftGraphOAuthStatus {
    guard Self.isValidProfileScope(profileScope),
      let bundleIdentifier = Bundle.main.bundleIdentifier,
      !bundleIdentifier.isEmpty else {
      throw MicrosoftGraphOAuthError.notConfigured
    }
    let generation = advanceGeneration(for: profileScope)
    if let refreshWork = refreshWorkByScope.removeValue(forKey: profileScope) {
      refreshWork.task.cancel()
      _ = await refreshWork.task.result
    }
    // If a newer connect started while cancellation completed, it owns the
    // scope now and this stale disconnect must not erase its result.
    guard currentGeneration(for: profileScope) == generation else {
      return await outlookStatus(profileScope: profileScope) ?? Self.invalidOwnerStatus()
    }
    try MicrosoftGraphOAuthKeychainStore.clear(
      bundleIdentifier: bundleIdentifier,
      profileScope: profileScope
    )
    return await outlookStatus(profileScope: profileScope) ?? Self.invalidOwnerStatus()
  }

  private func refresh(
    configuration: MicrosoftGraphOAuthConfiguration,
    cached: MicrosoftGraphOAuthSession,
    refreshToken: String,
    profileScope: String,
    generation: UInt64
  ) async throws -> MicrosoftGraphOAuthTokenSet {
    guard currentGeneration(for: profileScope) == generation else {
      throw CancellationError()
    }
    let work: RefreshWork
    if let current = refreshWorkByScope[profileScope],
      current.generation == generation {
      work = current
    } else {
      let task = Task { [httpClient] in
        try await Self.requestToken(
          httpClient: httpClient,
          configuration: configuration,
          form: [
            "client_id": configuration.clientID,
            "scope": MicrosoftGraphOAuthScopes.authorizationValue,
            "refresh_token": refreshToken,
            "grant_type": "refresh_token",
          ],
          existingRefreshToken: refreshToken,
          existingScopes: cached.token.grantedScopes
        )
      }
      work = .init(id: UUID(), generation: generation, task: task)
      refreshWorkByScope[profileScope] = work
    }

    let token: MicrosoftGraphOAuthTokenSet
    do {
      token = try await work.task.value
    } catch {
      if refreshWorkByScope[profileScope]?.id == work.id {
        refreshWorkByScope.removeValue(forKey: profileScope)
      }
      throw error
    }
    if refreshWorkByScope[profileScope]?.id == work.id {
      refreshWorkByScope.removeValue(forKey: profileScope)
    }
    try Task.checkCancellation()
    guard currentGeneration(for: profileScope) == generation else {
      throw CancellationError()
    }
    guard MicrosoftGraphOAuthScopes.grantedScopesSatisfy(token.grantedScopes) else {
      throw MicrosoftGraphOAuthError.invalidScope
    }
    // There is deliberately no suspension between the epoch check and save.
    try MicrosoftGraphOAuthKeychainStore.save(
      .init(
        profileScope: profileScope,
        clientID: configuration.clientID,
        account: cached.account,
        token: token
      ),
      bundleIdentifier: configuration.bundleIdentifier,
      profileScope: profileScope
    )
    return token
  }

  private func currentGeneration(for profileScope: String) -> UInt64 {
    generationByScope[profileScope] ?? 0
  }

  @discardableResult
  private func advanceGeneration(for profileScope: String) -> UInt64 {
    let next = currentGeneration(for: profileScope) &+ 1
    generationByScope[profileScope] = next
    return next
  }

  static func isValidProfileScope(_ profileScope: String) -> Bool {
    guard profileScope.count == 72, profileScope.hasPrefix("profile.") else { return false }
    let hexadecimal = Set("0123456789abcdef")
    return profileScope.dropFirst(8).allSatisfy { character in
      hexadecimal.contains(character)
    }
  }

  private static func invalidOwnerStatus() -> MicrosoftGraphOAuthStatus {
    .init(
      configured: false,
      connected: false,
      account: nil,
      expiresAt: nil,
      requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
      redirectURI: MicrosoftGraphOAuthConfiguration.expectedRedirectURI(),
      detail: "Connectez-vous à monGARS avant de connecter Outlook."
    )
  }

  /*
   * The code below contains the fixed token exchange, account lookup, and
   * pure PKCE/callback helpers shared by connect and refresh.
   */

  private func requestToken(
    configuration: MicrosoftGraphOAuthConfiguration,
    form: [String: String],
    existingRefreshToken: String?,
    existingScopes: String?
  ) async throws -> MicrosoftGraphOAuthTokenSet {
    try await Self.requestToken(
      httpClient: httpClient,
      configuration: configuration,
      form: form,
      existingRefreshToken: existingRefreshToken,
      existingScopes: existingScopes
    )
  }

  private static func requestToken(
    httpClient: MicrosoftGraphOAuthHTTPClient,
    configuration: MicrosoftGraphOAuthConfiguration,
    form: [String: String],
    existingRefreshToken: String?,
    existingScopes: String?
  ) async throws -> MicrosoftGraphOAuthTokenSet {
    var request = URLRequest(url: tokenURL)
    request.httpMethod = "POST"
    request.timeoutInterval = 20
    request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.httpBody = formEncoded(form)

    let data: Data
    let response: HTTPURLResponse
    do {
      (data, response) = try await httpClient.send(request)
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      throw MicrosoftGraphOAuthError.tokenEndpointUnavailable
    }
    guard (200...299).contains(response.statusCode) else {
      let decoded = try? JSONDecoder().decode(MicrosoftGraphOAuthErrorResponse.self, from: data)
      throw classifyTokenError(
        code: decoded?.error,
        suberror: decoded?.suberror,
        description: decoded?.errorDescription,
        status: response.statusCode
      )
    }

    guard let decoded = try? JSONDecoder().decode(
      MicrosoftGraphOAuthTokenResponse.self,
      from: data
    ) else {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    let grantedScopes = resolvedGrantedScopes(
      responseScope: decoded.scope,
      existingScopes: existingScopes,
      requestedScopes: form["scope"]
    )
    let refreshToken = decoded.refreshToken?.trimmingCharacters(in: .whitespacesAndNewlines)
    let retainedRefreshToken = refreshToken?.isEmpty == false
      ? refreshToken
      : existingRefreshToken
    guard let expiresAt = validatedExpiry(expiresIn: decoded.expiresIn) else {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    let token = MicrosoftGraphOAuthTokenSet(
      accessToken: decoded.accessToken,
      refreshToken: retainedRefreshToken,
      expiresAt: expiresAt,
      grantedScopes: grantedScopes,
      tokenType: decoded.tokenType
    )
    guard token.hasUnexpiredAccessToken,
      MicrosoftGraphOAuthScopes.grantedScopesSatisfy(token.grantedScopes) else {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    return token
  }

  static func resolvedGrantedScopes(
    responseScope: String?,
    existingScopes: String?,
    requestedScopes: String?
  ) -> String {
    for candidate in [responseScope, existingScopes, requestedScopes] {
      let trimmed = candidate?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      if !trimmed.isEmpty { return trimmed }
    }
    return ""
  }

  static func validatedExpiry(
    expiresIn: Int,
    now: Date = Date()
  ) -> Date? {
    // Delegated Microsoft access tokens are short lived. Reject zero,
    // negative, and implausibly large lifetimes instead of extending them.
    guard (1...86_400).contains(expiresIn) else { return nil }
    let expiry = now.addingTimeInterval(TimeInterval(expiresIn))
    return expiry > now ? expiry : nil
  }

  private func fetchAccount(accessToken: String) async throws
    -> MicrosoftGraphOAuthAccount
  {
    var request = URLRequest(url: Self.accountURL)
    request.httpMethod = "GET"
    request.timeoutInterval = 20
    request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    do {
      let (data, response) = try await httpClient.send(request)
      guard (200...299).contains(response.statusCode),
        let account = try? JSONDecoder().decode(MicrosoftGraphOAuthAccount.self, from: data),
        !account.id.isEmpty,
        account.id.utf8.count <= 512 else {
        throw MicrosoftGraphOAuthError.accountLookupFailed
      }
      return account
    } catch let error as MicrosoftGraphOAuthError {
      throw error
    } catch {
      throw MicrosoftGraphOAuthError.accountLookupFailed
    }
  }

  private static func disconnectedStatus(
    configuration: MicrosoftGraphOAuthConfiguration
  ) -> MicrosoftGraphOAuthStatus {
    .init(
      configured: true,
      connected: false,
      account: nil,
      expiresAt: nil,
      requiredScopes: MicrosoftGraphOAuthScopes.outlookTools,
      redirectURI: configuration.redirectURI,
      detail: "Outlook est configuré, mais aucun compte Microsoft n'est connecté."
    )
  }

  static func authorizationURL(
    configuration: MicrosoftGraphOAuthConfiguration,
    state: String,
    codeChallenge: String
  ) throws -> URL {
    guard !state.isEmpty, state.utf8.count <= 256,
      !codeChallenge.isEmpty, codeChallenge.utf8.count <= 128,
      var components = URLComponents(url: authorizeURL, resolvingAgainstBaseURL: false)
    else { throw MicrosoftGraphOAuthError.notConfigured }
    components.queryItems = [
      .init(name: "client_id", value: configuration.clientID),
      .init(name: "response_type", value: "code"),
      .init(name: "redirect_uri", value: configuration.redirectURI),
      .init(name: "response_mode", value: "query"),
      .init(name: "scope", value: MicrosoftGraphOAuthScopes.authorizationValue),
      .init(name: "state", value: state),
      .init(name: "prompt", value: "select_account"),
      .init(name: "code_challenge", value: codeChallenge),
      .init(name: "code_challenge_method", value: "S256"),
    ]
    guard let url = components.url,
      url.scheme == "https",
      url.host == "login.microsoftonline.com" else {
      throw MicrosoftGraphOAuthError.notConfigured
    }
    return url
  }

  static func authorizationCode(
    from url: URL,
    configuration: MicrosoftGraphOAuthConfiguration,
    expectedState: String
  ) throws -> String {
    guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
      components.scheme?.caseInsensitiveCompare(configuration.callbackScheme) == .orderedSame,
      components.host?.caseInsensitiveCompare("auth") == .orderedSame,
      components.user == nil,
      components.password == nil,
      components.port == nil,
      components.path.isEmpty,
      components.fragment == nil
    else { throw MicrosoftGraphOAuthError.invalidAuthorizationResponse }
    let items = components.queryItems ?? []
    let states = items.filter { $0.name == "state" }
    let codes = items.filter { $0.name == "code" }
    let errors = items.filter { $0.name == "error" }
    guard states.count == 1 else {
      throw MicrosoftGraphOAuthError.invalidState
    }
    // Validate the anti-CSRF binding before interpreting provider-controlled
    // success or error fields, including error callbacks.
    guard states[0].value == expectedState else {
      throw MicrosoftGraphOAuthError.invalidState
    }
    guard codes.count <= 1, errors.count <= 1, codes.isEmpty != errors.isEmpty else {
      throw MicrosoftGraphOAuthError.invalidAuthorizationResponse
    }
    if let error = errors.first?.value {
      throw classifyAuthorizationError(error)
    }
    guard let code = codes.first?.value,
      !code.isEmpty, code.utf8.count <= 16_384,
      !code.unicodeScalars.contains(where: {
        CharacterSet.controlCharacters.union(.newlines).contains($0)
      }) else {
      throw MicrosoftGraphOAuthError.invalidAuthorizationResponse
    }
    return code
  }

  static func codeChallenge(verifier: String) -> String {
    Data(SHA256.hash(data: Data(verifier.utf8))).base64URLEncodedString()
  }

  static func formEncoded(_ form: [String: String]) -> Data {
    form.keys.sorted().map { key in
      "\(percentEncode(key))=\(percentEncode(form[key] ?? ""))"
    }
    .joined(separator: "&")
    .data(using: .utf8) ?? Data()
  }

  static func classifyTokenError(
    code: String?,
    suberror: String?,
    description: String?,
    status: Int
  ) -> MicrosoftGraphOAuthError {
    if status == 429 { return .tokenEndpointThrottled }
    if (500...599).contains(status) { return .tokenEndpointUnavailable }
    let normalizedCode = code?.trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased() ?? ""
    let detail = [suberror, description]
      .compactMap { $0?.lowercased() }
      .joined(separator: " ")
    if detail.contains("consent_required") || detail.contains("aadsts65001") {
      return .consentRequired
    }
    if detail.contains("interaction_required") || detail.contains("aadsts50076") {
      return .interactionRequired
    }
    if detail.contains("invalid_scope") || detail.contains("aadsts70011") {
      return .invalidScope
    }
    switch normalizedCode {
    case "invalid_grant": return .invalidGrant
    case "interaction_required", "login_required", "account_selection_required":
      return .interactionRequired
    case "consent_required", "access_denied": return .consentRequired
    case "invalid_scope", "insufficient_scope": return .invalidScope
    case "temporarily_unavailable", "server_error": return .tokenEndpointUnavailable
    case "too_many_requests", "throttled": return .tokenEndpointThrottled
    default: return .tokenEndpointUnavailable
    }
  }

  private static func classifyAuthorizationError(
    _ code: String
  ) -> MicrosoftGraphOAuthError {
    switch code.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "access_denied", "consent_required": return .consentRequired
    case "invalid_scope": return .invalidScope
    case "temporarily_unavailable", "server_error": return .tokenEndpointUnavailable
    default: return .interactionRequired
    }
  }

  private static func makeCodeVerifier() throws -> String {
    var bytes = [UInt8](repeating: 0, count: 32)
    guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
      throw MicrosoftGraphOAuthError.interactiveSignInUnavailable
    }
    return Data(bytes).base64URLEncodedString()
  }

  private static func percentEncode(_ value: String) -> String {
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-._~")
    return value.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
  }
}

#if os(iOS)
@MainActor
private final class MicrosoftGraphOAuthWebSession: NSObject,
  ASWebAuthenticationPresentationContextProviding
{
  private var activeSession: ASWebAuthenticationSession?

  func authenticate(url: URL, callbackScheme: String) async throws -> URL {
    try Task.checkCancellation()
    let callback = try await withTaskCancellationHandler {
      try Task.checkCancellation()
      return try await withCheckedThrowingContinuation { continuation in
        let session = ASWebAuthenticationSession(
          url: url,
          callbackURLScheme: callbackScheme
        ) { [weak self] callbackURL, error in
          Task { @MainActor in
            self?.activeSession = nil
            if let callbackURL {
              continuation.resume(returning: callbackURL)
            } else if let webError = error as? ASWebAuthenticationSessionError,
              webError.code == .canceledLogin {
              continuation.resume(throwing: MicrosoftGraphOAuthError.signInCancelled)
            } else {
              continuation.resume(throwing: MicrosoftGraphOAuthError.interactionRequired)
            }
          }
        }
        session.presentationContextProvider = self
        session.prefersEphemeralWebBrowserSession = false
        activeSession = session
        guard session.start() else {
          activeSession = nil
          continuation.resume(
            throwing: MicrosoftGraphOAuthError.interactiveSignInUnavailable
          )
          return
        }
        // Cancellation can race with session creation/start. The outer
        // handler covers later cancellation; this check closes that window.
        if Task.isCancelled {
          session.cancel()
        }
      }
    } onCancel: {
      Task { @MainActor [weak self] in
        self?.activeSession?.cancel()
        self?.activeSession = nil
      }
    }
    try Task.checkCancellation()
    return callback
  }

  func presentationAnchor(for session: ASWebAuthenticationSession)
    -> ASPresentationAnchor
  {
    let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
    let active = scenes.first(where: { $0.activationState == .foregroundActive })
      ?? scenes.first
    return active?.windows.first(where: { $0.isKeyWindow })
      ?? active?.windows.first
      ?? ASPresentationAnchor()
  }
}
#endif

private extension Data {
  func base64URLEncodedString() -> String {
    base64EncodedString()
      .replacingOccurrences(of: "+", with: "-")
      .replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "=", with: "")
  }
}
