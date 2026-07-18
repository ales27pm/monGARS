import Foundation
import Tokenizers

enum PromptBuilder {
  static func build(
    messages: [ChatMessage],
    tokenizer: any Tokenizer,
    maxNewTokens: Int
  ) throws -> [Int] {
    let maximumPromptTokens = max(
      1,
      MonGARSModelManifest.contextLength - maxNewTokens
    )
    let history = messages
      .filter { message in
        ["user", "assistant"].contains(message.role)
          && !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      }
      .suffix(10)

    guard !history.isEmpty else { throw InferenceError.emptyPrompt }

    let system = ChatMessage(
      role: "system",
      content: MonGARSModelManifest.systemPrompt
    )
    var selected = [system] + Array(history)
    var encoded = try encode(selected, tokenizer: tokenizer)

    while encoded.count > maximumPromptTokens, selected.count > 2 {
      selected.remove(at: 1)
      encoded = try encode(selected, tokenizer: tokenizer)
    }

    if encoded.count > maximumPromptTokens, let latest = selected.last {
      let contentTokens = tokenizer.encode(
        text: latest.content,
        addSpecialTokens: false
      )
      let maximumSuffixTokens = min(
        contentTokens.count,
        maximumPromptTokens
      )

      for length in stride(from: maximumSuffixTokens, through: 1, by: -1) {
        let shortened = ChatMessage(
          role: latest.role,
          content: tokenizer.decode(
            tokens: Array(contentTokens.suffix(length)),
            skipSpecialTokens: false
          )
        )
        let candidate = try encode([system, shortened], tokenizer: tokenizer)
        if candidate.count <= maximumPromptTokens {
          encoded = candidate
          break
        }
      }

      if encoded.count > maximumPromptTokens {
        throw InferenceError.promptTooLong
      }
    }

    guard !encoded.isEmpty else { throw InferenceError.emptyPrompt }
    return encoded
  }

  private static func encode(
    _ messages: [ChatMessage],
    tokenizer: any Tokenizer
  ) throws -> [Int] {
    let templateMessages: [Tokenizers.Message] = messages.map { message in
      ["role": message.role, "content": message.content]
    }
    return try tokenizer.applyChatTemplate(
      messages: templateMessages,
      tools: nil,
      additionalContext: ["enable_thinking": false]
    )
  }
}
