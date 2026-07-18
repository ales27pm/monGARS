import Foundation
import Tokenizers

enum PromptBuilder {
  static func build(
    messages: [ChatMessage],
    tokenizer: any Tokenizer,
    maxNewTokens: Int
  ) throws -> [Int] {
    try Task.checkCancellation()
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
      try Task.checkCancellation()
      let contentTokens = tokenizer.encode(
        text: latest.content,
        addSpecialTokens: false
      )
      try Task.checkCancellation()
      let maximumSuffixTokens = min(
        contentTokens.count,
        min(maximumPromptTokens, MonGARSModelManifest.contextLength)
      )

      if let candidate = try largestFittingEncodedSuffix(
        contentTokens: contentTokens,
        maximumSuffixTokens: maximumSuffixTokens,
        maximumPromptTokens: maximumPromptTokens,
        encodeCandidate: { suffix in
          let shortened = ChatMessage(
            role: latest.role,
            content: tokenizer.decode(
              tokens: suffix,
              skipSpecialTokens: false
            )
          )
          return try encode([system, shortened], tokenizer: tokenizer)
        }
      ) {
        encoded = candidate
      }

      if encoded.count > maximumPromptTokens {
        throw InferenceError.promptTooLong
      }
    }

    guard !encoded.isEmpty else { throw InferenceError.emptyPrompt }
    return encoded
  }

  static func largestFittingEncodedSuffix(
    contentTokens: [Int],
    maximumSuffixTokens: Int,
    maximumPromptTokens: Int,
    encodeCandidate: ([Int]) throws -> [Int]
  ) throws -> [Int]? {
    let upperBound = min(
      contentTokens.count,
      min(max(maximumSuffixTokens, 0), MonGARSModelManifest.contextLength)
    )
    guard upperBound > 0 else { return nil }

    // Re-encoding decoded BPE suffixes is not monotonic: prepending one token
    // can merge neighboring pieces and shorten the encoded prompt. Descending
    // exhaustive search is therefore the smallest exact strategy. The model
    // context bounds it to at most 512 attempts.
    for length in stride(from: upperBound, through: 1, by: -1) {
      try Task.checkCancellation()
      let candidate = try encodeCandidate(Array(contentTokens.suffix(length)))
      try Task.checkCancellation()
      if candidate.count <= maximumPromptTokens {
        return candidate
      }
    }
    return nil
  }

  private static func encode(
    _ messages: [ChatMessage],
    tokenizer: any Tokenizer
  ) throws -> [Int] {
    try Task.checkCancellation()
    let templateMessages: [Tokenizers.Message] = messages.map { message in
      ["role": message.role, "content": message.content]
    }
    let encoded = try tokenizer.applyChatTemplate(
      messages: templateMessages,
      tools: nil,
      additionalContext: ["enable_thinking": false]
    )
    try Task.checkCancellation()
    return encoded
  }
}
