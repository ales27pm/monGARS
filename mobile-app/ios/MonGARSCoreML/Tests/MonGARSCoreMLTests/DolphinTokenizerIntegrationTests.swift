import Foundation
import Tokenizers
import XCTest
@testable import MonGARSCoreML

final class DolphinTokenizerIntegrationTests: XCTestCase {
  func testPinnedDolphinTokenizerBuildsChatPrompt() async throws {
    guard
      let path = ProcessInfo.processInfo.environment["DOLPHIN_TOKENIZER_DIR"],
      !path.isEmpty
    else {
      throw XCTSkip("DOLPHIN_TOKENIZER_DIR absent; test Hub optionnel.")
    }

    let tokenizer = try await AutoTokenizer.from(
      modelFolder: URL(fileURLWithPath: path, isDirectory: true)
    )
    let prompt = try PromptBuilder.build(
      messages: [
        ChatMessage(role: "user", content: "Reponds bonjour en francais."),
      ],
      tokenizer: tokenizer,
      maxNewTokens: 32
    )

    XCTAssertFalse(prompt.isEmpty)
    XCTAssertLessThan(prompt.count, MonGARSModelManifest.contextLength)
    XCTAssertTrue(prompt.allSatisfy { token in
      token >= 0 && token < MonGARSModelManifest.vocabularySize
    })
    XCTAssertTrue(
      tokenizer.decode(tokens: prompt).contains("Reponds bonjour en francais.")
    )
  }
}
