import Foundation
@testable import MonGARSAgentTools
import XCTest

final class MicrosoftGraphOAuthTests: XCTestCase {
  private let clientID = "11111111-2222-4333-8444-555555555555"
  private let bundleID = "com.example.MonGARSMobile"

  func testConfigurationRequiresClientIDAndRegisteredBundleRedirect() throws {
    let configuration = try MicrosoftGraphOAuthConfiguration.validated(
      clientID: clientID,
      bundleIdentifier: bundleID,
      registeredSchemes: ["msauth.\(bundleID)"]
    )

    XCTAssertEqual(configuration.clientID, clientID)
    XCTAssertEqual(configuration.redirectURI, "msauth.\(bundleID)://auth")
    XCTAssertThrowsError(try MicrosoftGraphOAuthConfiguration.validated(
      clientID: "$(MONGARS_MICROSOFT_CLIENT_ID)",
      bundleIdentifier: bundleID,
      registeredSchemes: ["msauth.\(bundleID)"]
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthConfiguration.validated(
      clientID: clientID,
      bundleIdentifier: bundleID,
      registeredSchemes: []
    )) { error in
      XCTAssertEqual(error as? MicrosoftGraphOAuthError, .redirectSchemeMissing)
    }
  }

  func testRuntimeClientIDIsOnlyFallbackForMissingBuildConfiguration() {
    let runtimeID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

    XCTAssertEqual(
      MicrosoftGraphOAuthConfiguration.resolvedClientID(
        bundled: clientID,
        runtime: runtimeID
      ),
      clientID
    )
    XCTAssertEqual(
      MicrosoftGraphOAuthConfiguration.resolvedClientID(
        bundled: "$(MONGARS_MICROSOFT_CLIENT_ID)",
        runtime: runtimeID
      ),
      runtimeID
    )
    XCTAssertNil(MicrosoftGraphOAuthConfiguration.resolvedClientID(
      bundled: nil,
      runtime: "not-a-client-id"
    ))
  }

  func testOutlookToolScopesAreExactAndRequireEveryResourceScope() {
    XCTAssertEqual(MicrosoftGraphOAuthScopes.outlookTools, [
      "User.Read",
      "Mail.ReadWrite",
      "Mail.Send",
      "offline_access",
    ])
    XCTAssertTrue(MicrosoftGraphOAuthScopes.grantedScopesSatisfy(
      "mail.send USER.READ Mail.ReadWrite"
    ))
    XCTAssertFalse(MicrosoftGraphOAuthScopes.grantedScopesSatisfy(
      "User.Read Mail.ReadWrite"
    ))
    XCTAssertFalse(MicrosoftGraphOAuthScopes.grantedScopesSatisfy(
      "User.Read Mail.Send"
    ))
  }

  func testPKCEChallengeMatchesRFC7636Vector() {
    let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.codeChallenge(verifier: verifier),
      "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
  }

  func testAuthorizationURLUsesCodeFlowPKCEAndNoClientSecret() throws {
    let configuration = try MicrosoftGraphOAuthConfiguration.validated(
      clientID: clientID,
      bundleIdentifier: bundleID,
      registeredSchemes: ["msauth.\(bundleID)"]
    )
    let url = try MicrosoftGraphOAuthTokenProvider.authorizationURL(
      configuration: configuration,
      state: "expected-state",
      codeChallenge: "challenge"
    )
    let components = try XCTUnwrap(
      URLComponents(url: url, resolvingAgainstBaseURL: false)
    )
    let values = Dictionary(
      uniqueKeysWithValues: (components.queryItems ?? []).compactMap { item in
        item.value.map { (item.name, $0) }
      }
    )

    XCTAssertEqual(url.host, "login.microsoftonline.com")
    XCTAssertEqual(values["response_type"], "code")
    XCTAssertEqual(values["code_challenge_method"], "S256")
    XCTAssertEqual(values["redirect_uri"], configuration.redirectURI)
    XCTAssertEqual(values["scope"], MicrosoftGraphOAuthScopes.authorizationValue)
    XCTAssertNil(values["client_secret"])
  }

  func testCallbackRequiresExactSchemeHostAndState() throws {
    let configuration = try MicrosoftGraphOAuthConfiguration.validated(
      clientID: clientID,
      bundleIdentifier: bundleID,
      registeredSchemes: ["msauth.\(bundleID)"]
    )
    let callback = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://auth?code=code-value&state=expected-state"
    ))
    XCTAssertEqual(
      try MicrosoftGraphOAuthTokenProvider.authorizationCode(
        from: callback,
        configuration: configuration,
        expectedState: "expected-state"
      ),
      "code-value"
    )
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: callback,
      configuration: configuration,
      expectedState: "different-state"
    )) { error in
      XCTAssertEqual(error as? MicrosoftGraphOAuthError, .invalidState)
    }
    let wrongHost = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://attacker?code=code-value&state=expected-state"
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: wrongHost,
      configuration: configuration,
      expectedState: "expected-state"
    ))

    let duplicateState = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://auth?code=code-value&state=expected-state&state=expected-state"
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: duplicateState,
      configuration: configuration,
      expectedState: "expected-state"
    )) { error in
      XCTAssertEqual(error as? MicrosoftGraphOAuthError, .invalidState)
    }

    let duplicateCode = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://auth?code=one&code=two&state=expected-state"
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: duplicateCode,
      configuration: configuration,
      expectedState: "expected-state"
    ))

    let untrustedError = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://auth?error=access_denied&state=attacker-state"
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: untrustedError,
      configuration: configuration,
      expectedState: "expected-state"
    )) { error in
      XCTAssertEqual(error as? MicrosoftGraphOAuthError, .invalidState)
    }

    let unexpectedPath = try XCTUnwrap(URL(
      string: "msauth.\(bundleID)://auth/extra?code=one&state=expected-state"
    ))
    XCTAssertThrowsError(try MicrosoftGraphOAuthTokenProvider.authorizationCode(
      from: unexpectedPath,
      configuration: configuration,
      expectedState: "expected-state"
    ))
  }

  func testOwnerScopesAreOpaqueAndIsolated() throws {
    let alice = try XCTUnwrap(AgentOpaqueProfileScope.make(rawOwnerID: "account:alice"))
    let aliceAgain = try XCTUnwrap(
      AgentOpaqueProfileScope.make(rawOwnerID: " account:alice ")
    )
    let bob = try XCTUnwrap(AgentOpaqueProfileScope.make(rawOwnerID: "account:bob"))

    XCTAssertEqual(alice, aliceAgain)
    XCTAssertNotEqual(alice, bob)
    XCTAssertTrue(MicrosoftGraphOAuthTokenProvider.isValidProfileScope(alice))
    XCTAssertTrue(MicrosoftGraphOAuthTokenProvider.isValidProfileScope(bob))
    XCTAssertFalse(MicrosoftGraphOAuthTokenProvider.isValidProfileScope("account:alice"))
  }

  func testMissingTokenScopeFallsBackToExactRequestedGrant() {
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.resolvedGrantedScopes(
        responseScope: nil,
        existingScopes: nil,
        requestedScopes: MicrosoftGraphOAuthScopes.authorizationValue
      ),
      MicrosoftGraphOAuthScopes.authorizationValue
    )
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.resolvedGrantedScopes(
        responseScope: nil,
        existingScopes: "User.Read Mail.ReadWrite Mail.Send",
        requestedScopes: MicrosoftGraphOAuthScopes.authorizationValue
      ),
      "User.Read Mail.ReadWrite Mail.Send"
    )
  }

  func testTokenExpiryRejectsNonPositiveAndImplausibleLifetimes() throws {
    let now = Date(timeIntervalSince1970: 1_000)
    XCTAssertNil(MicrosoftGraphOAuthTokenProvider.validatedExpiry(expiresIn: 0, now: now))
    XCTAssertNil(MicrosoftGraphOAuthTokenProvider.validatedExpiry(expiresIn: -1, now: now))
    XCTAssertNil(MicrosoftGraphOAuthTokenProvider.validatedExpiry(
      expiresIn: 86_401,
      now: now
    ))
    XCTAssertEqual(
      try XCTUnwrap(MicrosoftGraphOAuthTokenProvider.validatedExpiry(
        expiresIn: 3_600,
        now: now
      )),
      Date(timeIntervalSince1970: 4_600)
    )
  }

  func testTokenEndpointErrorsAreClassifiedWithoutProviderText() {
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.classifyTokenError(
        code: "invalid_grant",
        suberror: nil,
        description: nil,
        status: 400
      ),
      .invalidGrant
    )
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.classifyTokenError(
        code: "invalid_grant",
        suberror: nil,
        description: "AADSTS65001 consent_required",
        status: 400
      ),
      .consentRequired
    )
    XCTAssertEqual(
      MicrosoftGraphOAuthTokenProvider.classifyTokenError(
        code: nil,
        suberror: nil,
        description: nil,
        status: 429
      ),
      .tokenEndpointThrottled
    )
  }
}
