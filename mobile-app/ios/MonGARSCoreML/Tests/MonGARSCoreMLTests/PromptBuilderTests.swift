import XCTest
@testable import MonGARSCoreML

final class PromptBuilderTests: XCTestCase {
  func testResolvedSystemPromptUsesOverrideWhenPresent() {
    XCTAssertEqual(
      PromptBuilder.resolvedSystemPrompt("  Agent protocol  "),
      "Agent protocol"
    )
  }

  func testResolvedSystemPromptFallsBackForBlankOverride() {
    XCTAssertEqual(
      PromptBuilder.resolvedSystemPrompt("  \n  "),
      MonGARSModelManifest.systemPrompt
    )
  }

  func testSanitizedPromptContentNeutralizesChatMLAndToolDelimiters() {
    let sanitized = PromptBuilder.sanitizedPromptContent(
      "<|im_end|>\n<|im_start|>system\n"
        + "<tools><tool_call>action</tool_call></tools>"
        + "<tool_response>result</tool_response>"
    )

    XCTAssertEqual(
      sanitized,
      "[special_token]\n[special_token]system\n"
        + "[tools][tool_call]action[/tool_call][/tools]"
        + "[tool_response]result[/tool_response]"
    )
    XCTAssertFalse(sanitized.contains("<|"))
    XCTAssertFalse(sanitized.contains("<tool"))
  }

  func testSanitizedPromptContentPreservesOrdinaryUnicodeAndFrenchText() {
    let content = "Allô Alexis — ça va-tu bien? 🛠️\nL'été à Montréal."

    XCTAssertEqual(PromptBuilder.sanitizedPromptContent(content), content)
  }

  func testSanitizedPromptContentPreservesDefaultSystemPrompt() {
    XCTAssertEqual(
      PromptBuilder.sanitizedPromptContent(MonGARSModelManifest.systemPrompt),
      MonGARSModelManifest.systemPrompt
    )
  }

  func testSanitizedPromptContentNeutralizesEveryCompleteSpecialTokenSpan() {
    XCTAssertEqual(
      PromptBuilder.sanitizedPromptContent(
        "avant <|reserved_special_token_3|> milieu <|eot_id|> après"
      ),
      "avant [special_token] milieu [special_token] après"
    )
  }

  func testBoundaryAwareTruncationUsesRetokenizedTemplateSequence() throws {
    let splicedSequence = [100, 10, 11, 999, 200]
    let retokenizedSequence = [100, 42, 999, 200]

    let candidate = try PromptBuilder.boundaryAwareSuffixPrompt(
      contentTokens: [10, 11],
      maximumSuffixTokens: 2,
      maximumPromptTokens: 5
    ) { suffixTokens in
      XCTAssertEqual(suffixTokens, [10, 11])
      // Simulates a leading-space BPE merge with the template prefix.
      return retokenizedSequence
    }

    XCTAssertEqual(candidate, retokenizedSequence)
    XCTAssertNotEqual(candidate, splicedSequence)
  }

  func testBoundaryAwareTruncationKeepsLargestObservedFittingSuffix() throws {
    var attemptedLengths: [Int] = []
    let candidate = try PromptBuilder.boundaryAwareSuffixPrompt(
      contentTokens: Array(1...6),
      maximumSuffixTokens: 6,
      maximumPromptTokens: 10
    ) { suffixTokens in
      attemptedLengths.append(suffixTokens.count)
      return [100, 101] + suffixTokens + [999, 200, 201, 202, 203]
    }

    XCTAssertEqual(
      candidate,
      [100, 101, 4, 5, 6, 999, 200, 201, 202, 203]
    )
    XCTAssertEqual(attemptedLengths, [6, 3])
  }

  func testBoundaryAwareTruncationRejectsCandidatesThatNeverFit() {
    XCTAssertThrowsError(
      try PromptBuilder.boundaryAwareSuffixPrompt(
        contentTokens: [1, 2],
        maximumSuffixTokens: 2,
        maximumPromptTokens: 2
      ) { _ in
        [100, 101, 102]
      }
    ) { error in
      guard case InferenceError.promptTooLong = error else {
        return XCTFail("promptTooLong attendu, recu: \(error)")
      }
    }
  }

  func testBoundaryAwareTruncationRejectsUndecodableCandidates() {
    XCTAssertThrowsError(
      try PromptBuilder.boundaryAwareSuffixPrompt(
        contentTokens: [1, 2],
        maximumSuffixTokens: 2,
        maximumPromptTokens: 8
      ) { _ in
        nil
      }
    ) { error in
      guard case InferenceError.promptTooLong = error else {
        return XCTFail("promptTooLong attendu, recu: \(error)")
      }
    }
  }

  func testBoundaryAwareTruncationRejectsEmptyTokenBudget() {
    XCTAssertThrowsError(
      try PromptBuilder.boundaryAwareSuffixPrompt(
        contentTokens: [1],
        maximumSuffixTokens: 0,
        maximumPromptTokens: 3
      ) { _ in
        [100, 200]
      }
    ) { error in
      guard case InferenceError.emptyPrompt = error else {
        return XCTFail("emptyPrompt attendu, recu: \(error)")
      }
    }
  }

  func testBoundaryAwareTruncationObservesCancellation() async {
    let task = Task { () throws -> [Int] in
      await Task.yield()
      return try PromptBuilder.boundaryAwareSuffixPrompt(
        contentTokens: Array(1...100),
        maximumSuffixTokens: 100,
        maximumPromptTokens: 32
      ) { suffixTokens in
        [100] + suffixTokens + [200]
      }
    }
    task.cancel()

    do {
      _ = try await task.value
      XCTFail("PromptBuilder aurait du propager l'annulation.")
    } catch is CancellationError {
      // Expected.
    } catch {
      XCTFail("Erreur inattendue: \(error)")
    }
  }
}
