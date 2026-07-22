import Foundation
import MonGARSCoreML

#if canImport(Darwin)
import Darwin
#endif
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public protocol AgentWebServing: Sendable {
  /// True only for a host implementation that pins or allowlists the network
  /// destination through connection establishment (not merely DNS preflight).
  var supportsPublicFetch: Bool { get }
  func search(query: String) async -> AgentServiceResponse
  func fetch(url: String) async -> AgentServiceResponse
}

public extension AgentWebServing {
  var supportsPublicFetch: Bool { false }
}

public enum AgentResolvedHostAddress: Sendable, Equatable {
  /// Network-byte-order bytes.
  case ipv4([UInt8])
  case ipv6([UInt8])
}

public protocol AgentHostResolving: Sendable {
  /// Returns nil when resolution fails. An empty answer is also denied.
  func resolve(host: String) -> [AgentResolvedHostAddress]?
}

public struct SystemAgentHostResolver: AgentHostResolving, Sendable {
  public init() {}

  public func resolve(host: String) -> [AgentResolvedHostAddress]? {
#if canImport(Darwin)
    var hints = addrinfo(
      ai_flags: AI_ADDRCONFIG,
      ai_family: AF_UNSPEC,
      ai_socktype: Int32(SOCK_STREAM.rawValue),
      ai_protocol: Int32(IPPROTO_TCP),
      ai_addrlen: 0,
      ai_canonname: nil,
      ai_addr: nil,
      ai_next: nil
    )
    var head: UnsafeMutablePointer<addrinfo>?
    guard getaddrinfo(host, nil, &hints, &head) == 0, let first = head else { return nil }
    defer { freeaddrinfo(first) }
    var output: [AgentResolvedHostAddress] = []
    var cursor: UnsafeMutablePointer<addrinfo>? = first
    while let item = cursor?.pointee {
      if item.ai_family == AF_INET, let address = item.ai_addr {
        let ipv4 = address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee.sin_addr }
        var mutable = ipv4
        output.append(.ipv4(withUnsafeBytes(of: &mutable) { Array($0) }))
      } else if item.ai_family == AF_INET6, let address = item.ai_addr {
        let ipv6 = address.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { $0.pointee.sin6_addr }
        var mutable = ipv6
        output.append(.ipv6(withUnsafeBytes(of: &mutable) { Array($0) }))
      }
      cursor = item.ai_next
    }
    return output
#else
    return nil
#endif
  }
}

public enum PublicHTTPSURLPolicy {
  public static func validateSyntax(_ url: URL) -> Bool {
    guard url.scheme?.lowercased() == "https",
      url.user == nil,
      url.password == nil,
      url.port == nil || url.port == 443,
      let host = url.host?.lowercased(),
      !host.isEmpty,
      host.utf8.count <= 253 else { return false }

    if host == "localhost" || host.hasSuffix(".localhost")
      || host.hasSuffix(".local") || host.hasSuffix(".internal")
      || host.hasSuffix(".home") || host.hasSuffix(".arpa") {
      return false
    }
    if let literal = parsedAddress(host) { return isPublic(literal) }
    return true
  }

  public static func validate(
    _ url: URL,
    resolver: any AgentHostResolving = SystemAgentHostResolver()
  ) -> Bool {
    guard validateSyntax(url), let host = url.host?.lowercased() else { return false }
    if let literal = parsedAddress(host) {
      return isPublic(literal)
    }
    guard let addresses = resolver.resolve(host: host), !addresses.isEmpty else { return false }
    // A mixed public/private answer is denied so DNS rebinding and split-horizon
    // hostnames cannot reach loopback, link-local, or RFC1918 services.
    return addresses.allSatisfy(isPublic)
  }

  public static func redirectAllowed(
    from initialURL: URL,
    to redirectedURL: URL,
    policy: AgentRedirectPolicy,
    resolver: any AgentHostResolving = SystemAgentHostResolver()
  ) -> Bool {
    guard validate(redirectedURL, resolver: resolver) else { return false }
    switch policy {
    case .deny: return false
    case .publicHTTPS: return true
    case .sameOrigin:
      return redirectedURL.scheme?.lowercased() == initialURL.scheme?.lowercased()
        && redirectedURL.host?.lowercased() == initialURL.host?.lowercased()
        && (redirectedURL.port ?? 443) == (initialURL.port ?? 443)
    }
  }

  public static func isPublic(_ address: AgentResolvedHostAddress) -> Bool {
    switch address {
    case let .ipv4(bytes):
      guard bytes.count == 4 else { return false }
      let first = bytes[0], second = bytes[1], third = bytes[2]
      if first == 0 || first == 10 || first == 127 || first >= 224 { return false }
      if first == 100 && (64...127).contains(second) { return false }
      if first == 169 && second == 254 { return false }
      if first == 172 && (16...31).contains(second) { return false }
      if first == 192 && second == 168 { return false }
      if first == 192 && second == 0 && third == 0 { return false }
      if first == 192 && second == 0 && third == 2 { return false }
      if first == 192 && second == 88 && third == 99 { return false }
      if first == 198 && (second == 18 || second == 19) { return false }
      if first == 198 && second == 51 && third == 100 { return false }
      if first == 203 && second == 0 && third == 113 { return false }
      return true
    case let .ipv6(bytes):
      guard bytes.count == 16 else { return false }
      if bytes.allSatisfy({ $0 == 0 }) { return false }
      if bytes.dropLast().allSatisfy({ $0 == 0 }) && bytes.last == 1 { return false }
      if bytes[0] == 0xfc || bytes[0] == 0xfd { return false }
      if bytes[0] == 0xfe && (bytes[1] & 0xc0) == 0x80 { return false }
      if bytes[0] == 0xff { return false }
      if bytes[0] == 0x01 && bytes.dropFirst().allSatisfy({ $0 == 0 }) { return false }
      // Documentation and benchmarking ranges.
      if bytes[0] == 0x20 && bytes[1] == 0x01 && bytes[2] == 0x0d && bytes[3] == 0xb8 { return false }
      if bytes[0] == 0x20 && bytes[1] == 0x01 && bytes[2] == 0x00 && bytes[3] == 0x02 { return false }
      // IPv4-mapped addresses retain the IPv4 safety decision.
      if bytes.prefix(10).allSatisfy({ $0 == 0 }), bytes[10] == 0xff, bytes[11] == 0xff {
        return isPublic(.ipv4(Array(bytes.suffix(4))))
      }
      return true
    }
  }

  private static func parsedAddress(_ host: String) -> AgentResolvedHostAddress? {
#if canImport(Darwin)
    var ipv4 = in_addr()
    if inet_pton(AF_INET, host, &ipv4) == 1 {
      var value = ipv4
      return .ipv4(withUnsafeBytes(of: &value) { Array($0) })
    }

    let unbracketed = host.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
    var ipv6 = in6_addr()
    if inet_pton(AF_INET6, unbracketed, &ipv6) == 1 {
      return .ipv6(withUnsafeBytes(of: &ipv6) { Array($0) })
    }
#endif
    return nil
  }
}

public enum AgentRedirectPolicy: Sendable, Equatable {
  case publicHTTPS
  case sameOrigin
  case deny
}

public final class BoundedHTTPSLoader: @unchecked Sendable {
  public struct Response: Sendable {
    public let data: Data
    public let response: HTTPURLResponse
  }

  private let maximumBytes: Int
  private let timeout: TimeInterval
  private let resolver: any AgentHostResolving

  public init(
    maximumBytes: Int = 512 * 1_024,
    timeout: TimeInterval = 12,
    resolver: any AgentHostResolving = SystemAgentHostResolver()
  ) {
    self.maximumBytes = min(max(maximumBytes, 4 * 1_024), 2 * 1_024 * 1_024)
    self.timeout = min(max(timeout, 2), 30)
    self.resolver = resolver
  }

  public func load(
    _ request: URLRequest,
    redirectPolicy: AgentRedirectPolicy = .publicHTTPS
  ) async throws -> Response {
    guard let url = request.url, PublicHTTPSURLPolicy.validate(url, resolver: resolver) else {
      throw AgentWebError.disallowedURL
    }
    let cancellation = RequestCancellationBox()
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        let delegate = SingleRequestDelegate(
          maximumBytes: maximumBytes,
          initialURL: url,
          redirectPolicy: redirectPolicy,
          resolver: resolver,
          continuation: continuation
        )
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.waitsForConnectivity = false
        configuration.httpCookieStorage = nil
        configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: queue)
        delegate.session = session
        let task = session.dataTask(with: request)
        delegate.task = task
        cancellation.register(task)
        task.resume()
      }
    } onCancel: {
      cancellation.cancel()
    }
  }
}

private final class RequestCancellationBox: @unchecked Sendable {
  private let lock = NSLock()
  private var task: URLSessionTask?
  private var isCancelled = false

  func register(_ task: URLSessionTask) {
    lock.lock()
    if isCancelled {
      lock.unlock()
      task.cancel()
      return
    }
    self.task = task
    lock.unlock()
  }

  func cancel() {
    lock.lock()
    isCancelled = true
    let task = self.task
    self.task = nil
    lock.unlock()
    task?.cancel()
  }
}

private final class SingleRequestDelegate: NSObject, URLSessionDataDelegate, URLSessionTaskDelegate,
  @unchecked Sendable
{
  private let maximumBytes: Int
  private let initialURL: URL
  private let redirectPolicy: AgentRedirectPolicy
  private let resolver: any AgentHostResolving
  private var continuation: CheckedContinuation<BoundedHTTPSLoader.Response, Error>?
  private var data = Data()
  private var httpResponse: HTTPURLResponse?
  var session: URLSession?
  weak var task: URLSessionDataTask?

  init(
    maximumBytes: Int,
    initialURL: URL,
    redirectPolicy: AgentRedirectPolicy,
    resolver: any AgentHostResolving,
    continuation: CheckedContinuation<BoundedHTTPSLoader.Response, Error>
  ) {
    self.maximumBytes = maximumBytes
    self.initialURL = initialURL
    self.redirectPolicy = redirectPolicy
    self.resolver = resolver
    self.continuation = continuation
  }

  func urlSession(
    _ session: URLSession,
    dataTask: URLSessionDataTask,
    didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
  ) {
    guard let http = response as? HTTPURLResponse else {
      completionHandler(.cancel)
      finish(.failure(AgentWebError.invalidResponse))
      return
    }
    if response.expectedContentLength > Int64(maximumBytes) {
      completionHandler(.cancel)
      finish(.failure(AgentWebError.responseTooLarge))
      return
    }
    httpResponse = http
    completionHandler(.allow)
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive chunk: Data) {
    guard data.count + chunk.count <= maximumBytes else {
      dataTask.cancel()
      finish(.failure(AgentWebError.responseTooLarge))
      return
    }
    data.append(chunk)
  }

  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    guard let url = request.url,
      PublicHTTPSURLPolicy.redirectAllowed(
        from: initialURL,
        to: url,
        policy: redirectPolicy,
        resolver: resolver
      ) else {
      completionHandler(nil)
      finish(.failure(AgentWebError.disallowedRedirect))
      return
    }
    completionHandler(request)
  }

  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    if let error {
      let nsError = error as NSError
      if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
        finish(.failure(CancellationError()))
        return
      }
      finish(.failure(error))
      return
    }
    guard let httpResponse else {
      finish(.failure(AgentWebError.invalidResponse))
      return
    }
    finish(.success(.init(data: data, response: httpResponse)))
  }

  private func finish(_ result: Result<BoundedHTTPSLoader.Response, Error>) {
    guard let continuation else { return }
    self.continuation = nil
    continuation.resume(with: result)
    session?.finishTasksAndInvalidate()
    session = nil
  }
}

public enum AgentWebError: Error, Sendable, Equatable {
  case disallowedURL
  case disallowedRedirect
  case invalidResponse
  case responseTooLarge
  case unsupportedContentType
}

public final class PublicWebToolService: AgentWebServing, @unchecked Sendable {
  private let loader: BoundedHTTPSLoader

  /// Arbitrary fetch is intentionally disabled: DNS preflight cannot pin the
  /// address URLSession ultimately connects to. Inject a separate pinned or
  /// allowlisted `AgentWebServing` implementation to advertise web.fetch.
  public let supportsPublicFetch = false

  public init(loader: BoundedHTTPSLoader = .init()) {
    self.loader = loader
  }

  public func search(query: String) async -> AgentServiceResponse {
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty, query.utf8.count <= 1_000 else {
      return .failed("The web search query is invalid.", code: "web_invalid_query")
    }
    var components = URLComponents(string: "https://html.duckduckgo.com/html/")
    components?.queryItems = [.init(name: "q", value: query)]
    guard let url = components?.url else {
      return .failed("The web search request is invalid.", code: "web_invalid_url")
    }
    var request = URLRequest(url: url)
    request.setValue("MonGARS-iOS/1.0", forHTTPHeaderField: "User-Agent")
    request.setValue("text/html", forHTTPHeaderField: "Accept")
    let response = await loadText(request)
    guard response.status == .success,
      let html = response.payload?.objectValue?["content"]?.stringValue else { return response }

    let results = Self.extractSearchResults(html).prefix(8)
    guard !results.isEmpty else {
      return .success("No public web results were found.", payload: ["results": []])
    }
    let payloadResults: [AgentJSONValue] = results.map { item in
      ["title": .string(item.title), "url": .string(item.url)]
    }
    let text = results.map { "- \($0.title): \($0.url)" }.joined(separator: "\n")
    return .success(text, payload: ["results": .array(payloadResults)])
  }

  public func fetch(url rawURL: String) async -> AgentServiceResponse {
    guard supportsPublicFetch else {
      return .unavailable(
        "Arbitrary URL fetch is disabled until a destination-pinned transport is configured.",
        code: "web_fetch_requires_pinned_transport"
      )
    }
    guard rawURL.utf8.count <= 4_096,
      let url = URL(string: rawURL),
      PublicHTTPSURLPolicy.validate(url) else {
      return .denied("Only public HTTPS URLs can be fetched.", code: "web_url_denied")
    }
    var request = URLRequest(url: url)
    request.setValue("MonGARS-iOS/1.0", forHTTPHeaderField: "User-Agent")
    request.setValue("text/html, text/plain, application/json", forHTTPHeaderField: "Accept")
    let response = await loadText(request)
    guard response.status == .success,
      let content = response.payload?.objectValue?["content"]?.stringValue else { return response }
    let readable = Self.readableText(content)
    return .success(
      readable,
      payload: [
        "url": .string(url.absoluteString),
        "content": .string(readable),
      ]
    )
  }

  private func loadText(_ request: URLRequest) async -> AgentServiceResponse {
    do {
      let loaded = try await loader.load(request)
      guard (200..<300).contains(loaded.response.statusCode) else {
        return .failed("The web server returned HTTP \(loaded.response.statusCode).", code: "web_http_\(loaded.response.statusCode)")
      }
      let type = loaded.response.value(forHTTPHeaderField: "Content-Type")?.lowercased() ?? ""
      guard type.isEmpty || type.contains("text/") || type.contains("application/json")
        || type.contains("application/xhtml") else {
        return .failed("The web response is not readable text.", code: "web_unsupported_content_type")
      }
      guard let text = String(data: loaded.data, encoding: .utf8) else {
        return .failed("The web response is not valid UTF-8 text.", code: "web_invalid_encoding")
      }
      return .success("Web content loaded.", payload: ["content": .string(text)])
    } catch AgentWebError.disallowedURL {
      return .denied("The web request was blocked by the public HTTPS policy.", code: "web_url_denied")
    } catch AgentWebError.disallowedRedirect {
      return .denied("The web request was blocked by the public HTTPS policy.", code: "web_url_denied")
    } catch AgentWebError.responseTooLarge {
      return .failed("The web response exceeded the safe size limit.", code: "web_response_too_large")
    } catch is CancellationError {
      return .init(status: .cancelled, text: "The web request was cancelled.", errorCode: "web_cancelled")
    } catch {
      return .failed("The public web service could not be reached.", code: "web_network_failure")
    }
  }

  private static func extractSearchResults(_ html: String) -> [(title: String, url: String)] {
    guard let regex = try? NSRegularExpression(
      pattern: #"<a[^>]+class=[\"'][^\"']*result__a[^\"']*[\"'][^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>"#,
      options: [.caseInsensitive, .dotMatchesLineSeparators]
    ) else { return [] }
    let range = NSRange(html.startIndex..., in: html)
    return regex.matches(in: html, range: range).compactMap { match in
      guard let urlRange = Range(match.range(at: 1), in: html),
        let titleRange = Range(match.range(at: 2), in: html) else { return nil }
      let title = readableText(String(html[titleRange]))
      let rawURL = decodeEntities(String(html[urlRange]))
      let resolvedURL: String
      if let components = URLComponents(string: rawURL),
        let redirect = components.queryItems?.first(where: { $0.name == "uddg" })?.value {
        resolvedURL = redirect
      } else {
        resolvedURL = rawURL
      }
      guard let url = URL(string: resolvedURL),
        PublicHTTPSURLPolicy.validateSyntax(url),
        !title.isEmpty else {
        return nil
      }
      return (AgentToolResultFactory.bounded(title, maximumCharacters: 240), url.absoluteString)
    }
  }

  private static func readableText(_ html: String) -> String {
    var text = html
    text = text.replacingOccurrences(
      of: #"<(script|style|noscript)[^>]*>.*?</\1>"#,
      with: " ",
      options: [.regularExpression, .caseInsensitive]
    )
    text = text.replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: .regularExpression)
    text = decodeEntities(text)
    text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    return AgentToolResultFactory.bounded(text, maximumCharacters: 12_000)
  }

  private static func decodeEntities(_ text: String) -> String {
    text
      .replacingOccurrences(of: "&amp;", with: "&")
      .replacingOccurrences(of: "&lt;", with: "<")
      .replacingOccurrences(of: "&gt;", with: ">")
      .replacingOccurrences(of: "&quot;", with: "\"")
      .replacingOccurrences(of: "&#39;", with: "'")
      .replacingOccurrences(of: "&nbsp;", with: " ")
  }
}
