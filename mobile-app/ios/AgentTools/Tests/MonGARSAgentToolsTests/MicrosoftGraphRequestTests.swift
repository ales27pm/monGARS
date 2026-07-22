import Foundation
@testable import MonGARSAgentTools
import MonGARSCoreML
import XCTest

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

final class MicrosoftGraphRequestTests: XCTestCase {
  private actor RecordingTokenProvider: MicrosoftGraphAccessTokenProviding {
    private(set) var forcedRefreshes: [Bool] = []

    func accessToken() async throws -> String? { nil }
    func hasUsableSession() async -> Bool { false }
    func accessToken(profileScope: String, forceRefresh: Bool) async throws -> String? {
      forcedRefreshes.append(forceRefresh)
      return forceRefresh ? "fresh-token" : "stale-token"
    }
    func hasUsableSession(profileScope: String) async -> Bool { true }
    func refreshCalls() -> [Bool] { forcedRefreshes }
  }

  private actor RecordingGraphLoader {
    private var statusCodes: [Int]
    private(set) var authorizationHeaders: [String?] = []

    init(statusCodes: [Int]) {
      self.statusCodes = statusCodes
    }

    func load(_ request: URLRequest) throws -> BoundedHTTPSLoader.Response {
      authorizationHeaders.append(request.value(forHTTPHeaderField: "Authorization"))
      let status = statusCodes.isEmpty ? 500 : statusCodes.removeFirst()
      let response = try XCTUnwrap(HTTPURLResponse(
        url: try XCTUnwrap(request.url),
        statusCode: status,
        httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": "application/json"]
      ))
      let data = status == 200 ? Data("{\"value\":[]}".utf8) : Data()
      return .init(data: data, response: response)
    }

    func headers() -> [String?] { authorizationHeaders }
  }

  func testListMessagesBuildsFixedGraphEndpoint() throws {
    let request = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMessagesList,
      arguments: [
        "folderId": "inbox",
        "limit": 12,
        "unreadOnly": true,
      ]
    )
    let urlRequest = try request.urlRequest()

    XCTAssertEqual(request.method, .get)
    XCTAssertEqual(urlRequest.url?.scheme, "https")
    XCTAssertEqual(urlRequest.url?.host, "graph.microsoft.com")
    XCTAssertEqual(urlRequest.url?.path, "/v1.0/me/mailFolders/inbox/messages")
    let components = URLComponents(url: try XCTUnwrap(urlRequest.url), resolvingAgainstBaseURL: false)
    XCTAssertEqual(components?.queryItems?.first(where: { $0.name == "$top" })?.value, "12")
    XCTAssertEqual(components?.queryItems?.first(where: { $0.name == "$filter" })?.value, "isRead eq false")
    XCTAssertNil(urlRequest.value(forHTTPHeaderField: "Authorization"))
  }

  func testSendMailUsesPOSTAndStructuredBody() throws {
    let request = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMailSend,
      arguments: [
        "to": "person@example.com",
        "subject": "Review",
        "body": "Please review this.",
      ]
    )
    let urlRequest = try request.urlRequest()

    XCTAssertEqual(request.method, .post)
    XCTAssertEqual(urlRequest.url?.path, "/v1.0/me/sendMail")
    let body = try XCTUnwrap(urlRequest.httpBody)
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
    XCTAssertNotNil(json["message"])
    XCTAssertEqual(json["saveToSentItems"] as? Bool, true)
  }

  func testSendAndForwardSplitBoundedRecipientLists() throws {
    let send = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMailSend,
      arguments: [
        "to": "one@example.com; two@example.com,three@example.com\nfour@example.com",
        "subject": "Review",
        "body": "Please review this.",
      ]
    )
    let sendJSON = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(send.urlRequest().httpBody))
        as? [String: Any]
    )
    let message = try XCTUnwrap(sendJSON["message"] as? [String: Any])
    let recipients = try XCTUnwrap(message["toRecipients"] as? [[String: Any]])
    XCTAssertEqual(recipients.count, 4)

    let forward = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMessageForward,
      arguments: [
        "messageId": "message-1",
        "to": "one@example.com;two@example.com",
      ]
    )
    let forwardJSON = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(forward.urlRequest().httpBody))
        as? [String: Any]
    )
    XCTAssertEqual((forwardJSON["toRecipients"] as? [[String: Any]])?.count, 2)
    XCTAssertNil(MicrosoftGraphRequestBuilder.parseRecipients("valid@example.com;not-an-email"))
  }

  func testSearchOmitsOrderByAndFolderAliasesAreCanonical() throws {
    let search = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMessagesSearch,
      arguments: ["query": "invoice", "folder": "Spam", "limit": 10]
    )
    let request = try search.urlRequest()
    let components = try XCTUnwrap(
      URLComponents(url: try XCTUnwrap(request.url), resolvingAgainstBaseURL: false)
    )
    XCTAssertEqual(request.url?.path, "/v1.0/me/mailFolders/junkemail/messages")
    XCTAssertNotNil(components.queryItems?.first(where: { $0.name == "$search" }))
    XCTAssertNil(components.queryItems?.first(where: { $0.name == "$orderby" }))

    let move = try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMessageMove,
      arguments: ["messageId": "message-1", "destination": "Trash"]
    )
    let moveJSON = try XCTUnwrap(
      JSONSerialization.jsonObject(with: try XCTUnwrap(move.urlRequest().httpBody))
        as? [String: Any]
    )
    XCTAssertEqual(moveJSON["destinationId"] as? String, "deleteditems")
  }

  func testIdentifierCannotInjectPathOrQuery() {
    XCTAssertThrowsError(try MicrosoftGraphRequestBuilder.build(
      operation: .outlookMessageRead,
      arguments: ["messageId": "../../me?$select=password"]
    ))
    XCTAssertFalse(MicrosoftGraphRequestBuilder.isSafePathComponent("message/id"))
    XCTAssertFalse(MicrosoftGraphRequestBuilder.isSafePathComponent(".."))
  }

  func testMissingTokenReturnsUnavailableWithoutNetworkRequest() async {
    let service = URLSessionMicrosoftGraphService(
      tokenProvider: UnavailableMicrosoftGraphTokenProvider()
    )
    let response = await service.perform(operation: .outlookStatus, arguments: [:])
    XCTAssertEqual(response.status, .unavailable)
    XCTAssertEqual(response.errorCode, "outlook_not_authenticated")
  }

  func testGraph401ForcesOneRefreshAndRetriesExactRequestOnce() async throws {
    let tokens = RecordingTokenProvider()
    let loader = RecordingGraphLoader(statusCodes: [401, 200])
    let service = URLSessionMicrosoftGraphService(
      tokenProvider: tokens,
      requestLoader: { request in try await loader.load(request) }
    )

    let response = await service.perform(
      operation: .outlookFoldersList,
      arguments: [:],
      profileScope: String(repeating: "a", count: 72)
    )

    let refreshCalls = await tokens.refreshCalls()
    let authorizationHeaders = await loader.headers().compactMap { $0 }
    XCTAssertEqual(response.status, .success)
    XCTAssertEqual(refreshCalls, [false, true])
    XCTAssertEqual(
      authorizationHeaders,
      ["Bearer stale-token", "Bearer fresh-token"]
    )
  }

  func testGraph403DoesNotRefresh() async throws {
    let tokens = RecordingTokenProvider()
    let loader = RecordingGraphLoader(statusCodes: [403])
    let service = URLSessionMicrosoftGraphService(
      tokenProvider: tokens,
      requestLoader: { request in try await loader.load(request) }
    )

    let response = await service.perform(
      operation: .outlookFoldersList,
      arguments: [:],
      profileScope: String(repeating: "b", count: 72)
    )

    let refreshCalls = await tokens.refreshCalls()
    let headerCount = await loader.headers().count
    XCTAssertEqual(response.status, .denied)
    XCTAssertEqual(response.errorCode, "outlook_authorization_failed")
    XCTAssertEqual(refreshCalls, [false])
    XCTAssertEqual(headerCount, 1)
  }

  func testOutlookStatusWorksWithoutAccessToken() async {
    let service = URLSessionMicrosoftGraphService(
      tokenProvider: UnavailableMicrosoftGraphTokenProvider()
    )

    let response = await service.perform(
      operation: .outlookStatus,
      arguments: [:],
      profileScope: String(repeating: "c", count: 72)
    )

    XCTAssertEqual(response.status, .success)
    guard case let .object(payload)? = response.payload else {
      return XCTFail("Expected a structured local Outlook status")
    }
    XCTAssertEqual(payload["connected"]?.boolValue, false)
  }

  func testOversizedMessageListRetainsBoundedIDsAndScrubsTokens() throws {
    let messages: [[String: Any]] = (0..<20).map { index in
      [
        "id": "message-\(index)",
        "subject": "Subject \(index) " + String(repeating: "S", count: 500),
        "from": [
          "emailAddress": ["name": "Sender", "address": "sender@example.com"],
        ],
        "bodyPreview": "Preview \(index) " + String(repeating: "P", count: 1_000),
        "accessToken": "must-not-survive",
      ]
    }
    let data = try JSONSerialization.data(withJSONObject: [
      "value": messages,
      "refresh_token": "must-not-survive",
    ])

    let response = URLSessionMicrosoftGraphService.successResponse(
      operation: .outlookMessagesList,
      data: data
    )
    let payload = try XCTUnwrap(response.payload?.canonicalJSONString())

    XCTAssertEqual(response.status, .success)
    XCTAssertTrue(response.text.contains("message-0"))
    XCTAssertTrue(response.text.contains("Subject 0"))
    XCTAssertTrue(payload.contains("message-0"))
    XCTAssertTrue(payload.contains("\"truncated\":true"))
    XCTAssertFalse(response.text.lowercased().contains("token"))
    XCTAssertFalse(payload.lowercased().contains("token"))
    XCTAssertLessThan(payload.utf8.count, 12_000)
  }

  func testOversizedReadRetainsPlainBodyAndScrubsHTMLSecrets() throws {
    let data = try JSONSerialization.data(withJSONObject: [
      "id": "message-read-1",
      "subject": "Important update",
      "from": [
        "emailAddress": ["name": "Alice", "address": "alice@example.com"],
      ],
      "body": [
        "contentType": "html",
        "content": "<script>accessToken=secret</script><p>Hello <b>team</b>.</p>"
          + String(repeating: "Useful details. ", count: 1_000),
      ],
      "refreshToken": "must-not-survive",
    ])

    let response = URLSessionMicrosoftGraphService.successResponse(
      operation: .outlookMessageRead,
      data: data
    )
    let payload = try XCTUnwrap(response.payload?.canonicalJSONString())

    XCTAssertTrue(response.text.contains("message-read-1"))
    XCTAssertTrue(response.text.contains("Hello team"))
    XCTAssertTrue(response.text.contains("Useful details"))
    XCTAssertFalse(response.text.contains("<script>"))
    XCTAssertFalse(response.text.contains("accessToken"))
    XCTAssertFalse(payload.lowercased().contains("refreshtoken"))
    XCTAssertLessThan(payload.utf8.count, 18_000)
  }

  func testGraphProjectionKeepsNumbersAndBooleansDistinct() throws {
    let data = try JSONSerialization.data(withJSONObject: [
      "value": [
        ["id": "valid", "size": 1, "isInline": true],
        ["id": "wrong-types", "size": true, "isInline": 1],
      ],
    ])

    let response = URLSessionMicrosoftGraphService.successResponse(
      operation: .outlookAttachmentsList,
      data: data
    )
    guard case let .object(payload)? = response.payload,
      case let .array(attachments)? = payload["attachments"],
      attachments.count == 2,
      case let .object(valid) = attachments[0],
      case let .object(wrongTypes) = attachments[1] else {
      return XCTFail("Expected two projected attachments")
    }

    XCTAssertEqual(valid["size"]?.numberValue, 1)
    XCTAssertEqual(valid["isInline"]?.boolValue, true)
    XCTAssertNil(valid["size"]?.boolValue)
    XCTAssertNil(valid["isInline"]?.numberValue)
    XCTAssertNil(wrongTypes["size"])
    XCTAssertNil(wrongTypes["isInline"])
  }
}
