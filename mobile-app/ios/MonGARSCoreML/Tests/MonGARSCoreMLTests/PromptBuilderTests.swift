import XCTest
@testable import MonGARSCoreML

final class PromptBuilderTests: XCTestCase {
  func testTokenLevelTruncationPreservesTemplateAndLargestFittingSuffix() throws {
    let candidate = try PromptBuilder.tokenLevelSuffixPrompt(
      contentTokens: Array(1...6),
      emptyPromptTokens: [100, 999, 101, 102, 999, 200, 201],
      maximumSuffixTokens: 6,
      maximumPromptTokens: 10,
      messageEndTokenID: 999
    )

    XCTAssertEqual(
      candidate,
      [100, 999, 101, 102, 4, 5, 6, 999, 200, 201]
    )
  }

  func testTokenLevelTruncationRejectsMissingMessageEndMarker() {
    XCTAssertThrowsError(
      try PromptBuilder.tokenLevelSuffixPrompt(
        contentTokens: [1, 2],
        emptyPromptTokens: [100, 101],
        maximumSuffixTokens: 2,
        maximumPromptTokens: 8,
        messageEndTokenID: 999
      )
    ) { error in
      guard case InferenceError.invalidModel = error else {
        return XCTFail("invalidModel attendu, recu: \(error)")
      }
    }
  }

  func testTokenLevelTruncationRejectsTemplateWithoutContentBudget() {
    XCTAssertThrowsError(
      try PromptBuilder.tokenLevelSuffixPrompt(
        contentTokens: [1],
        emptyPromptTokens: [100, 999, 200],
        maximumSuffixTokens: 1,
        maximumPromptTokens: 3,
        messageEndTokenID: 999
      )
    ) { error in
      guard case InferenceError.promptTooLong = error else {
        return XCTFail("promptTooLong attendu, recu: \(error)")
      }
    }
  }

  func testTokenLevelTruncationObservesCancellation() async {
    let task = Task { () throws -> [Int] in
      await Task.yield()
      return try PromptBuilder.tokenLevelSuffixPrompt(
        contentTokens: Array(1...100),
        emptyPromptTokens: [100, 999, 200],
        maximumSuffixTokens: 100,
        maximumPromptTokens: 32,
        messageEndTokenID: 999
      )
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
