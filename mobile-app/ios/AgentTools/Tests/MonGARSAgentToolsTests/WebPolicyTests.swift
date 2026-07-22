import Foundation
import MonGARSAgentTools
import XCTest

final class WebPolicyTests: XCTestCase {
  func testOnlyPublicHTTPSURLsAreAccepted() {
    let resolver = FixtureResolver(addresses: ["example.com": [.ipv4([93, 184, 216, 34])]])
    XCTAssertTrue(PublicHTTPSURLPolicy.validate(URL(string: "https://example.com/article")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "http://example.com")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "https://localhost/admin")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "https://127.0.0.1/admin")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "https://10.0.0.1/admin")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "https://[::1]/admin")!, resolver: resolver))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(URL(string: "https://user:pass@example.com")!, resolver: resolver))
  }

  func testHostnameResolvingToPrivateOrMixedAddressesIsRejected() {
    let privateResolver = FixtureResolver(addresses: ["metadata.example": [.ipv4([169, 254, 169, 254])]])
    let mixedResolver = FixtureResolver(addresses: [
      "rebinding.example": [.ipv4([93, 184, 216, 34]), .ipv4([192, 168, 1, 8])],
    ])
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(
      URL(string: "https://metadata.example/latest")!,
      resolver: privateResolver
    ))
    XCTAssertFalse(PublicHTTPSURLPolicy.validate(
      URL(string: "https://rebinding.example/latest")!,
      resolver: mixedResolver
    ))
  }

  func testReservedAndDocumentationRangesAreRejected() {
    for address: [UInt8] in [
      [192, 0, 0, 8], [192, 0, 2, 10], [198, 18, 0, 1],
      [198, 51, 100, 10], [203, 0, 113, 10], [224, 0, 0, 1], [240, 0, 0, 1],
    ] {
      XCTAssertFalse(PublicHTTPSURLPolicy.isPublic(.ipv4(address)))
    }
  }

  func testGraphStyleRedirectsStaySameOriginAndPublic() {
    let resolver = FixtureResolver(addresses: [
      "graph.microsoft.com": [.ipv4([20, 190, 128, 1])],
      "login.example": [.ipv4([93, 184, 216, 34])],
      "private.example": [.ipv4([10, 0, 0, 1])],
    ])
    let source = URL(string: "https://graph.microsoft.com/v1.0/me")!
    XCTAssertTrue(PublicHTTPSURLPolicy.redirectAllowed(
      from: source,
      to: URL(string: "https://graph.microsoft.com/v1.0/me/messages")!,
      policy: .sameOrigin,
      resolver: resolver
    ))
    XCTAssertFalse(PublicHTTPSURLPolicy.redirectAllowed(
      from: source,
      to: URL(string: "https://login.example/continue")!,
      policy: .sameOrigin,
      resolver: resolver
    ))
    XCTAssertFalse(PublicHTTPSURLPolicy.redirectAllowed(
      from: source,
      to: URL(string: "https://private.example/admin")!,
      policy: .publicHTTPS,
      resolver: resolver
    ))
  }
}

private struct FixtureResolver: AgentHostResolving {
  let addresses: [String: [AgentResolvedHostAddress]]
  func resolve(host: String) -> [AgentResolvedHostAddress]? { addresses[host] }
}
