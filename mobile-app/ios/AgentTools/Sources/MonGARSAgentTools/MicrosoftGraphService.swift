import CoreFoundation
import Foundation
import MonGARSCoreML

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public enum MicrosoftGraphHTTPMethod: String, Sendable, Equatable {
  case get = "GET"
  case post = "POST"
  case patch = "PATCH"
  case delete = "DELETE"
}

public struct MicrosoftGraphRequest: Sendable, Equatable {
  public let method: MicrosoftGraphHTTPMethod
  public let pathComponents: [String]
  public let queryItems: [URLQueryItem]
  public let body: AgentJSONValue?
  public let headers: [String: String]

  public init(
    method: MicrosoftGraphHTTPMethod,
    pathComponents: [String],
    queryItems: [URLQueryItem] = [],
    body: AgentJSONValue? = nil,
    headers: [String: String] = [:]
  ) {
    self.method = method
    self.pathComponents = pathComponents
    self.queryItems = queryItems
    self.body = body
    self.headers = headers
  }

  public func urlRequest(
    baseURL: URL = URL(string: "https://graph.microsoft.com/v1.0")!
  ) throws -> URLRequest {
    guard baseURL.scheme == "https", baseURL.host == "graph.microsoft.com" else {
      throw MicrosoftGraphRequestError.invalidBaseURL
    }
    var url = baseURL
    for component in pathComponents {
      guard MicrosoftGraphRequestBuilder.isSafePathComponent(component) else {
        throw MicrosoftGraphRequestError.invalidIdentifier
      }
      url.appendPathComponent(component)
    }
    guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
      throw MicrosoftGraphRequestError.invalidURL
    }
    components.queryItems = queryItems.isEmpty ? nil : queryItems
    guard let resolvedURL = components.url,
      resolvedURL.scheme == "https",
      resolvedURL.host == "graph.microsoft.com" else {
      throw MicrosoftGraphRequestError.invalidURL
    }

    var request = URLRequest(url: resolvedURL)
    request.httpMethod = method.rawValue
    request.timeoutInterval = 15
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    for (name, value) in headers {
      guard !name.contains("\r"), !name.contains("\n"),
        !value.contains("\r"), !value.contains("\n") else {
        throw MicrosoftGraphRequestError.invalidHeader
      }
      request.setValue(value, forHTTPHeaderField: name)
    }
    if let body {
      request.httpBody = try JSONEncoder().encode(body)
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    }
    return request
  }
}

public enum MicrosoftGraphRequestError: Error, Sendable, Equatable {
  case unsupportedOperation
  case invalidBaseURL
  case invalidURL
  case invalidHeader
  case invalidIdentifier
  case invalidArguments(String)
}

public enum MicrosoftGraphRequestBuilder {
  private static let messageFields = "id,subject,from,toRecipients,receivedDateTime,isRead,hasAttachments,bodyPreview"

  public static func build(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments
  ) throws -> MicrosoftGraphRequest {
    switch operation {
    case .outlookStatus:
      return .init(
        method: .get,
        pathComponents: ["me"],
        queryItems: [.init(name: "$select", value: "id,displayName")]
      )

    case .outlookFoldersList:
      var items = [
        URLQueryItem(name: "$select", value: "id,displayName,parentFolderId,childFolderCount,totalItemCount,unreadItemCount"),
        URLQueryItem(name: "$top", value: "50"),
      ]
      if AgentToolInput.bool("includeHidden", in: arguments) == true {
        items.append(.init(name: "includeHiddenFolders", value: "true"))
      }
      return .init(method: .get, pathComponents: ["me", "mailFolders"], queryItems: items)

    case .outlookMessagesList:
      let limit = AgentToolInput.integer("limit", in: arguments, range: 1...50) ?? 20
      let folder = AgentToolInput.optionalString("folderId", in: arguments, maximumBytes: 512)
        ?? AgentToolInput.optionalString("folder", in: arguments, maximumBytes: 512)
      var path = ["me"]
      if let folder {
        let normalized = normalizedFolderIdentifier(folder)
        guard isSafePathComponent(normalized) else { throw MicrosoftGraphRequestError.invalidIdentifier }
        path += ["mailFolders", normalized]
      }
      path.append("messages")
      var items = listQueryItems(limit: limit)
      if AgentToolInput.bool("unreadOnly", in: arguments) == true {
        items.append(.init(name: "$filter", value: "isRead eq false"))
      }
      return .init(method: .get, pathComponents: path, queryItems: items)

    case .outlookMessagesSearch:
      guard let query = AgentToolInput.requiredString("query", in: arguments, maximumBytes: 500) else {
        throw MicrosoftGraphRequestError.invalidArguments("query")
      }
      let limit = AgentToolInput.integer("limit", in: arguments, range: 1...50) ?? 20
      let folder = AgentToolInput.optionalString("folderId", in: arguments, maximumBytes: 512)
        ?? AgentToolInput.optionalString("folder", in: arguments, maximumBytes: 512)
      var path = ["me"]
      if let folder {
        let normalized = normalizedFolderIdentifier(folder)
        guard isSafePathComponent(normalized) else { throw MicrosoftGraphRequestError.invalidIdentifier }
        path += ["mailFolders", normalized]
      }
      path.append("messages")
      let quoted = query
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
      var items = searchQueryItems(limit: limit)
      items.append(.init(name: "$search", value: "\"\(quoted)\""))
      return .init(
        method: .get,
        pathComponents: path,
        queryItems: items,
        headers: ["ConsistencyLevel": "eventual"]
      )

    case .outlookMessageRead:
      return .init(
        method: .get,
        pathComponents: try messagePath(arguments),
        queryItems: [.init(name: "$select", value: messageFields + ",body,ccRecipients,bccRecipients,sentDateTime")]
      )

    case .outlookAttachmentsList:
      return .init(
        method: .get,
        pathComponents: try messagePath(arguments) + ["attachments"],
        queryItems: [.init(name: "$select", value: "id,name,contentType,size,isInline,lastModifiedDateTime")]
      )

    case .outlookDraftCreate:
      return .init(
        method: .post,
        pathComponents: ["me", "messages"],
        body: try messageBody(arguments)
      )

    case .outlookMailSend:
      return .init(
        method: .post,
        pathComponents: ["me", "sendMail"],
        body: .object([
          "message": try messageBody(arguments),
          "saveToSentItems": .bool(true),
        ])
      )

    case .outlookMessageMarkRead, .outlookMessageMarkUnread:
      return .init(
        method: .patch,
        pathComponents: try messagePath(arguments),
        body: .object(["isRead": .bool(operation == .outlookMessageMarkRead)])
      )

    case .outlookMessageMove:
      guard let rawDestination = AgentToolInput.optionalString("destinationId", in: arguments, maximumBytes: 512)
        ?? AgentToolInput.requiredString("destination", in: arguments, maximumBytes: 512),
        !rawDestination.isEmpty else {
        throw MicrosoftGraphRequestError.invalidArguments("destination")
      }
      let destination = normalizedFolderIdentifier(rawDestination)
      guard
        isSafePathComponent(destination) else {
        throw MicrosoftGraphRequestError.invalidArguments("destination")
      }
      return .init(
        method: .post,
        pathComponents: try messagePath(arguments) + ["move"],
        body: .object(["destinationId": .string(destination)])
      )

    case .outlookMessageArchive:
      return .init(
        method: .post,
        pathComponents: try messagePath(arguments) + ["move"],
        body: .object(["destinationId": .string("archive")])
      )

    case .outlookMessageDelete:
      return .init(method: .delete, pathComponents: try messagePath(arguments))

    case .outlookMessageReply, .outlookMessageReplyAll:
      guard let body = AgentToolInput.requiredString("body", in: arguments)
        ?? AgentToolInput.optionalString("comment", in: arguments) else {
        throw MicrosoftGraphRequestError.invalidArguments("body")
      }
      let action = operation == .outlookMessageReply ? "reply" : "replyAll"
      return .init(
        method: .post,
        pathComponents: try messagePath(arguments) + [action],
        body: .object(["comment": .string(body)])
      )

    case .outlookMessageForward:
      guard let rawRecipients = AgentToolInput.requiredString("to", in: arguments, maximumBytes: 4_096),
        let recipientAddresses = parseRecipients(rawRecipients) else {
        throw MicrosoftGraphRequestError.invalidArguments("to")
      }
      let comment = AgentToolInput.optionalString("comment", in: arguments)
        ?? AgentToolInput.optionalString("body", in: arguments)
        ?? ""
      return .init(
        method: .post,
        pathComponents: try messagePath(arguments) + ["forward"],
        body: .object([
          "comment": .string(comment),
          "toRecipients": recipients(recipientAddresses),
        ])
      )

    default:
      throw MicrosoftGraphRequestError.unsupportedOperation
    }
  }

  public static func isSafePathComponent(_ value: String) -> Bool {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty, trimmed.utf8.count <= 512,
      trimmed != ".", trimmed != ".." else { return false }
    return !trimmed.unicodeScalars.contains { scalar in
      CharacterSet.controlCharacters.contains(scalar) || scalar == "/" || scalar == "\\"
    }
  }

  private static func messagePath(_ arguments: AgentJSONArguments) throws -> [String] {
    guard let id = AgentToolInput.optionalString("id", in: arguments, maximumBytes: 512)
      ?? AgentToolInput.requiredString("messageId", in: arguments, maximumBytes: 512),
      isSafePathComponent(id) else {
      throw MicrosoftGraphRequestError.invalidArguments("messageId")
    }
    return ["me", "messages", id]
  }

  private static func listQueryItems(limit: Int) -> [URLQueryItem] {
    [
      .init(name: "$select", value: messageFields),
      .init(name: "$orderby", value: "receivedDateTime desc"),
      .init(name: "$top", value: String(limit)),
    ]
  }

  private static func searchQueryItems(limit: Int) -> [URLQueryItem] {
    [
      .init(name: "$select", value: messageFields),
      .init(name: "$top", value: String(limit)),
    ]
  }

  private static func messageBody(_ arguments: AgentJSONArguments) throws -> AgentJSONValue {
    guard let to = AgentToolInput.requiredString("to", in: arguments, maximumBytes: 4_096),
      let recipientAddresses = parseRecipients(to),
      let subject = AgentToolInput.requiredString("subject", in: arguments, maximumBytes: 1_000),
      let body = AgentToolInput.requiredString("body", in: arguments, maximumBytes: 100_000) else {
      throw MicrosoftGraphRequestError.invalidArguments("message")
    }
    return .object([
      "subject": .string(subject),
      "body": .object([
        "contentType": .string("Text"),
        "content": .string(body),
      ]),
      "toRecipients": recipients(recipientAddresses),
    ])
  }

  private static func recipients(_ addresses: [String]) -> AgentJSONValue {
    .array(addresses.map { address in
      .object([
        "emailAddress": .object(["address": .string(address)]),
      ])
    })
  }

  static func parseRecipients(_ value: String) -> [String]? {
    let separators = CharacterSet(charactersIn: ",;\n\r")
    let addresses = value.components(separatedBy: separators)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
    guard !addresses.isEmpty, addresses.count <= 50,
      addresses.allSatisfy(isPlausibleEmailAddress) else { return nil }
    return addresses
  }

  static func normalizedFolderIdentifier(_ value: String) -> String {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    switch trimmed.lowercased() {
    case "deleted", "deleteditems", "trash": return "deleteditems"
    case "junk", "junkemail", "spam": return "junkemail"
    case "sent", "sentitems": return "sentitems"
    case "draft", "drafts": return "drafts"
    case "inbox": return "inbox"
    case "archive": return "archive"
    default: return trimmed
    }
  }

  private static func isPlausibleEmailAddress(_ value: String) -> Bool {
    guard value.utf8.count <= 320,
      let at = value.firstIndex(of: "@"),
      at != value.startIndex,
      value.index(after: at) != value.endIndex,
      value[value.index(after: at)...].contains("."),
      !value.contains(where: { $0.isWhitespace || $0.isNewline }) else { return false }
    return true
  }
}

public protocol MicrosoftGraphServing: Sendable {
  func isConfigured() async -> Bool
  func isConfigured(profileScope: String?) async -> Bool
  func perform(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments
  ) async -> AgentServiceResponse
  func perform(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments,
    profileScope: String?
  ) async -> AgentServiceResponse
}

public extension MicrosoftGraphServing {
  func isConfigured() async -> Bool { true }
  func isConfigured(profileScope: String?) async -> Bool { await isConfigured() }
  func perform(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments,
    profileScope: String?
  ) async -> AgentServiceResponse {
    await perform(operation: operation, arguments: arguments)
  }
}

public final class URLSessionMicrosoftGraphService: MicrosoftGraphServing, @unchecked Sendable {
  typealias RequestLoader = @Sendable (URLRequest) async throws -> BoundedHTTPSLoader.Response

  private let tokenProvider: any MicrosoftGraphAccessTokenProviding
  private let requestLoader: RequestLoader

  public init(
    tokenProvider: any MicrosoftGraphAccessTokenProviding,
    loader: BoundedHTTPSLoader = .init(maximumBytes: 512 * 1_024, timeout: 15)
  ) {
    self.tokenProvider = tokenProvider
    self.requestLoader = { request in
      try await loader.load(request, redirectPolicy: .sameOrigin)
    }
  }

  init(
    tokenProvider: any MicrosoftGraphAccessTokenProviding,
    requestLoader: @escaping RequestLoader
  ) {
    self.tokenProvider = tokenProvider
    self.requestLoader = requestLoader
  }

  public func isConfigured() async -> Bool {
    await tokenProvider.hasUsableSession()
  }

  public func isConfigured(profileScope: String?) async -> Bool {
    guard let profileScope else { return false }
    return await tokenProvider.hasUsableSession(profileScope: profileScope)
  }

  public func perform(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments
  ) async -> AgentServiceResponse {
    await perform(operation: operation, arguments: arguments, profileScope: nil)
  }

  public func perform(
    operation: AgentHostOperation,
    arguments: AgentJSONArguments,
    profileScope: String?
  ) async -> AgentServiceResponse {
    guard let profileScope else {
      return .unavailable(
        "Outlook is unavailable until a Microsoft account is connected.",
        code: "outlook_not_authenticated"
      )
    }
    if operation == .outlookStatus {
      return await localStatus(profileScope: profileScope)
    }

    let token: String
    do {
      guard let supplied = try await scopedToken(
        profileScope: profileScope,
        forceRefresh: false
      ) else {
        return .unavailable("Outlook is unavailable until a Microsoft account is connected.", code: "outlook_not_authenticated")
      }
      token = supplied
    } catch MicrosoftGraphOAuthError.invalidTokenResponse {
      return .denied("The Outlook credential was rejected.", code: "outlook_invalid_credential")
    } catch {
      return .unavailable("Outlook authentication is temporarily unavailable.", code: "outlook_token_unavailable")
    }

    let graphRequest: MicrosoftGraphRequest
    do {
      graphRequest = try MicrosoftGraphRequestBuilder.build(operation: operation, arguments: arguments)
    } catch {
      return .failed("The Outlook request arguments are invalid.", code: "outlook_invalid_arguments")
    }

    do {
      let first = try await load(graphRequest: graphRequest, token: token)
      let loaded: BoundedHTTPSLoader.Response
      if first.response.statusCode == 401 {
        // A resource server can reject a token before its local expiry. Force
        // one refresh and replay the same fixed Graph request exactly once.
        let refreshed: String
        do {
          guard let value = try await scopedToken(
            profileScope: profileScope,
            forceRefresh: true
          ) else {
            return .denied("Outlook authorization is missing or expired.", code: "outlook_authorization_failed")
          }
          refreshed = value
        } catch {
          return .denied("Outlook authorization is missing or expired.", code: "outlook_authorization_failed")
        }
        loaded = try await load(graphRequest: graphRequest, token: refreshed)
      } else {
        loaded = first
      }
      switch loaded.response.statusCode {
      case 200..<300:
        return Self.successResponse(operation: operation, data: loaded.data)
      case 401, 403:
        return .denied("Outlook authorization is missing or expired.", code: "outlook_authorization_failed")
      case 404:
        return .failed("The requested Outlook item was not found.", code: "outlook_not_found")
      case 429:
        return .failed("Outlook is rate limiting requests. Try again later.", code: "outlook_rate_limited")
      default:
        return .failed("Outlook could not complete the request.", code: "outlook_http_\(loaded.response.statusCode)")
      }
    } catch is CancellationError {
      return .init(status: .cancelled, text: "The Outlook request was cancelled.", errorCode: "outlook_cancelled")
    } catch AgentWebError.responseTooLarge {
      return .failed("The Outlook response exceeded the safe size limit.", code: "outlook_response_too_large")
    } catch AgentWebError.disallowedURL {
      return .denied("The Outlook endpoint or redirect was blocked.", code: "outlook_endpoint_denied")
    } catch AgentWebError.disallowedRedirect {
      return .denied("The Outlook endpoint or redirect was blocked.", code: "outlook_endpoint_denied")
    } catch {
      return .failed("Outlook could not be reached.", code: "outlook_network_failure")
    }
  }

  private func scopedToken(
    profileScope: String,
    forceRefresh: Bool
  ) async throws -> String? {
    guard let supplied = try await tokenProvider.accessToken(
      profileScope: profileScope,
      forceRefresh: forceRefresh
    )?.trimmingCharacters(in: .whitespacesAndNewlines),
      !supplied.isEmpty else { return nil }
    guard supplied.utf8.count <= 16_384,
      !supplied.contains("\r"),
      !supplied.contains("\n") else {
      throw MicrosoftGraphOAuthError.invalidTokenResponse
    }
    return supplied
  }

  private func localStatus(profileScope: String) async -> AgentServiceResponse {
    if let status = await tokenProvider.outlookStatus(profileScope: profileScope) {
      let expiresAt: AgentJSONValue = status.expiresAt.map {
        .string(ISO8601DateFormatter().string(from: $0))
      } ?? .null
      let account: AgentJSONValue = status.account.map(AgentJSONValue.string) ?? .null
      return .success(
        status.detail,
        payload: .object([
          "configured": .bool(status.configured),
          "connected": .bool(status.connected),
          "account": account,
          "expiresAt": expiresAt,
          "requiredScopes": .array(status.requiredScopes.map(AgentJSONValue.string)),
          "redirectUri": .string(status.redirectURI),
        ])
      )
    }
    let connected = await tokenProvider.hasUsableSession(profileScope: profileScope)
    return .success(
      connected ? "Outlook is connected." : "Outlook is not configured or connected.",
      payload: .object([
        "configured": .bool(connected),
        "connected": .bool(connected),
      ])
    )
  }

  private func load(
    graphRequest: MicrosoftGraphRequest,
    token: String
  ) async throws -> BoundedHTTPSLoader.Response {
    var request = try graphRequest.urlRequest()
    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    // The loader enforces the limit while bytes arrive and permits only
    // same-origin redirects, preventing bearer-token forwarding.
    return try await requestLoader(request)
  }

  static func successResponse(
    operation: AgentHostOperation,
    data: Data
  ) -> AgentServiceResponse {
    guard !data.isEmpty else {
      return .success(successText(for: operation), payload: ["completed": true])
    }
    guard let object = try? JSONSerialization.jsonObject(with: data),
      let scrubbed = scrub(object) else {
      return .failed("Outlook returned unreadable data.", code: "outlook_decode_failed")
    }
    let projection = projectedResponse(operation: operation, value: scrubbed)
    guard let payload = AgentJSONValue.fromFoundation(projection.payload) else {
      return .failed("Outlook returned unreadable data.", code: "outlook_decode_failed")
    }
    return .success(projection.text, payload: payload)
  }

  private static func projectedResponse(
    operation: AgentHostOperation,
    value: Any
  ) -> (text: String, payload: Any) {
    switch operation {
    case .outlookMessagesList, .outlookMessagesSearch:
      guard let object = value as? [String: Any],
        let rawMessages = object["value"] as? [Any] else {
        return (successText(for: operation), ["messages": []])
      }
      let messages = rawMessages.prefix(8).compactMap { compactMessage($0) }
      let lines = messages.enumerated().map { index, message in
        let id = message["id"] as? String ?? "unknown-id"
        let subject = message["subject"] as? String ?? "(no subject)"
        let sender = message["from"] as? String ?? "unknown sender"
        let preview = message["bodyPreview"] as? String ?? ""
        return "\(index + 1). [\(id)] \(subject) — \(sender)\(preview.isEmpty ? "" : "\n   \(preview)")"
      }
      let text = lines.isEmpty
        ? "No Outlook messages matched."
        : lines.joined(separator: "\n")
      return (
        text,
        [
          "messages": messages,
          "returned": messages.count,
          "truncated": rawMessages.count > messages.count,
        ]
      )

    case .outlookMessageRead:
      guard let message = compactMessage(value, includeBody: true) else {
        return (successText(for: operation), ["message": [:]])
      }
      let id = message["id"] as? String ?? "unknown-id"
      let subject = message["subject"] as? String ?? "(no subject)"
      let sender = message["from"] as? String ?? "unknown sender"
      let body = message["body"] as? String ?? message["bodyPreview"] as? String ?? ""
      return (
        "[\(id)] \(subject)\nFrom: \(sender)\n\(body)",
        ["message": message]
      )

    case .outlookFoldersList:
      let values = (value as? [String: Any])?["value"] as? [Any] ?? []
      let folders = values.prefix(12).compactMap { item -> [String: Any]? in
        guard let source = item as? [String: Any],
          let id = boundedGraphString(source["id"], maximumCharacters: 256) else { return nil }
        var output: [String: Any] = ["id": id]
        copyBoundedString("displayName", from: source, into: &output, maximum: 200)
        copyNumber("totalItemCount", from: source, into: &output)
        copyNumber("unreadItemCount", from: source, into: &output)
        return output
      }
      let lines = folders.map {
        "[\($0["id"] as? String ?? "unknown-id")] \($0["displayName"] as? String ?? "Folder")"
      }
      return (
        lines.isEmpty ? "No Outlook folders were returned." : lines.joined(separator: "\n"),
        ["folders": folders, "truncated": values.count > folders.count]
      )

    case .outlookAttachmentsList:
      let values = (value as? [String: Any])?["value"] as? [Any] ?? []
      let attachments = values.prefix(8).compactMap { item -> [String: Any]? in
        guard let source = item as? [String: Any],
          let id = boundedGraphString(source["id"], maximumCharacters: 256) else { return nil }
        var output: [String: Any] = ["id": id]
        copyBoundedString("name", from: source, into: &output, maximum: 240)
        copyBoundedString("contentType", from: source, into: &output, maximum: 200)
        copyNumber("size", from: source, into: &output)
        copyBool("isInline", from: source, into: &output)
        return output
      }
      let lines = attachments.map {
        "[\($0["id"] as? String ?? "unknown-id")] \($0["name"] as? String ?? "Attachment")"
      }
      return (
        lines.isEmpty ? "No Outlook attachments were returned." : lines.joined(separator: "\n"),
        ["attachments": attachments, "truncated": values.count > attachments.count]
      )

    default:
      if let encoded = try? JSONSerialization.data(withJSONObject: value),
        encoded.count <= 10_000 {
        return (successText(for: operation), value)
      }
      let source = value as? [String: Any] ?? [:]
      var compact: [String: Any] = ["truncated": true]
      for key in ["id", "displayName", "subject", "status"] {
        copyBoundedString(key, from: source, into: &compact, maximum: 512)
      }
      return (successText(for: operation), compact)
    }
  }

  private static func compactMessage(
    _ value: Any,
    includeBody: Bool = false
  ) -> [String: Any]? {
    guard let source = value as? [String: Any],
      let id = boundedGraphString(
        source["id"],
        maximumCharacters: includeBody ? 512 : 256
      ) else { return nil }
    var output: [String: Any] = ["id": id]
    copyBoundedString(
      "subject",
      from: source,
      into: &output,
      maximum: includeBody ? 500 : 240
    )
    copyBoundedString("receivedDateTime", from: source, into: &output, maximum: 64)
    copyBoundedString("sentDateTime", from: source, into: &output, maximum: 64)
    copyBool("isRead", from: source, into: &output)
    copyBool("hasAttachments", from: source, into: &output)
    if let sender = compactEmail(source["from"]) {
      output["from"] = AgentToolResultFactory.bounded(
        sender,
        maximumCharacters: includeBody ? 620 : 320
      )
    }
    let previewMaximum = includeBody ? 500 : 240
    if let preview = boundedGraphString(
      source["bodyPreview"],
      maximumCharacters: previewMaximum
    ) {
      output["bodyPreview"] = stripHTML(preview, maximumCharacters: previewMaximum)
    }
    if includeBody,
      let bodyObject = source["body"] as? [String: Any],
      let content = boundedGraphString(bodyObject["content"], maximumCharacters: 20_000) {
      output["body"] = stripHTML(content, maximumCharacters: 4_000)
    }
    if includeBody {
      for key in ["toRecipients", "ccRecipients", "bccRecipients"] {
        let recipients = (source[key] as? [Any] ?? [])
          .prefix(20)
          .compactMap(compactEmail)
        if !recipients.isEmpty { output[key] = recipients }
      }
    }
    return output
  }

  private static func compactEmail(_ value: Any?) -> String? {
    guard let source = value as? [String: Any] else { return nil }
    let email = source["emailAddress"] as? [String: Any] ?? source
    let address = boundedGraphString(email["address"], maximumCharacters: 320)
    let name = boundedGraphString(email["name"], maximumCharacters: 300)
    switch (name, address) {
    case let (name?, address?): return "\(name) <\(address)>"
    case let (nil, address?): return address
    case let (name?, nil): return name
    default: return nil
    }
  }

  private static func boundedGraphString(
    _ value: Any?,
    maximumCharacters: Int
  ) -> String? {
    guard let value = value as? String else { return nil }
    let normalized = value
      .replacingOccurrences(of: "\u{0000}", with: "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalized.isEmpty else { return nil }
    return AgentToolResultFactory.bounded(normalized, maximumCharacters: maximumCharacters)
  }

  private static func stripHTML(
    _ value: String,
    maximumCharacters: Int
  ) -> String {
    var output = value
    if let unsafeBlocks = try? NSRegularExpression(
      pattern: "(?is)<(script|style)[^>]*>.*?</\\1>"
    ) {
      output = unsafeBlocks.stringByReplacingMatches(
        in: output,
        range: NSRange(output.startIndex..., in: output),
        withTemplate: " "
      )
    }
    if let tags = try? NSRegularExpression(pattern: "(?s)<[^>]+>") {
      output = tags.stringByReplacingMatches(
        in: output,
        range: NSRange(output.startIndex..., in: output),
        withTemplate: " "
      )
    }
    output = output
      .replacingOccurrences(of: "&nbsp;", with: " ")
      .replacingOccurrences(of: "&amp;", with: "&")
      .replacingOccurrences(of: "&lt;", with: "<")
      .replacingOccurrences(of: "&gt;", with: ">")
      .replacingOccurrences(of: "&quot;", with: "\"")
    let collapsed = output
      .split(whereSeparator: { $0.isWhitespace })
      .joined(separator: " ")
    return AgentToolResultFactory.bounded(collapsed, maximumCharacters: maximumCharacters)
  }

  private static func copyBoundedString(
    _ key: String,
    from source: [String: Any],
    into output: inout [String: Any],
    maximum: Int
  ) {
    if let value = boundedGraphString(source[key], maximumCharacters: maximum) {
      output[key] = value
    }
  }

  private static func copyNumber(
    _ key: String,
    from source: [String: Any],
    into output: inout [String: Any]
  ) {
    guard let value = source[key] as? NSNumber,
      CFGetTypeID(value) != CFBooleanGetTypeID(),
      value.doubleValue.isFinite else { return }
    output[key] = value
  }

  private static func copyBool(
    _ key: String,
    from source: [String: Any],
    into output: inout [String: Any]
  ) {
    guard let value = source[key] as? NSNumber,
      CFGetTypeID(value) == CFBooleanGetTypeID() else { return }
    output[key] = value.boolValue
  }

  private static func successText(for operation: AgentHostOperation) -> String {
    switch operation {
    case .outlookStatus: return "Outlook is connected."
    case .outlookFoldersList: return "Outlook folders loaded."
    case .outlookMessagesList: return "Outlook messages loaded."
    case .outlookMessagesSearch: return "Outlook search completed."
    case .outlookMessageRead: return "Outlook message loaded."
    case .outlookAttachmentsList: return "Outlook attachment metadata loaded."
    case .outlookDraftCreate: return "Outlook draft created."
    case .outlookMailSend: return "Outlook message sent."
    case .outlookMessageMarkRead: return "Outlook message marked read."
    case .outlookMessageMarkUnread: return "Outlook message marked unread."
    case .outlookMessageMove: return "Outlook message moved."
    case .outlookMessageArchive: return "Outlook message archived."
    case .outlookMessageDelete: return "Outlook message deleted."
    case .outlookMessageReply: return "Outlook reply sent."
    case .outlookMessageReplyAll: return "Outlook reply-all sent."
    case .outlookMessageForward: return "Outlook message forwarded."
    default: return "Outlook request completed."
    }
  }

  private static func scrub(_ value: Any) -> Any? {
    if let object = value as? [String: Any] {
      return object.reduce(into: [String: Any]()) { output, item in
        let normalized = item.key.lowercased()
        guard !normalized.contains("token"), normalized != "@odata.nextlink" else { return }
        output[item.key] = scrub(item.value)
      }
    }
    if let array = value as? [Any] {
      return array.compactMap(scrub)
    }
    if value is NSNull || value is String || value is NSNumber || value is Bool {
      return value
    }
    return nil
  }
}
