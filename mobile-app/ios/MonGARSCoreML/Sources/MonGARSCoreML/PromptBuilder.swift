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
      encoded = try boundaryAwareSuffixPrompt(
        contentTokens: contentTokens,
        maximumSuffixTokens: maximumSuffixTokens,
        maximumPromptTokens: maximumPromptTokens
      ) { suffixTokens in
        try Task.checkCancellation()
        let suffix = tokenizer.decode(tokens: suffixTokens)
        guard !suffix.isEmpty, latest.content.hasSuffix(suffix) else {
          return nil
        }
        return try encode(
          [system, ChatMessage(role: latest.role, content: suffix)],
          tokenizer: tokenizer
        )
      }

      if encoded.count > maximumPromptTokens {
        throw InferenceError.promptTooLong
      }
    }

    guard !encoded.isEmpty else { throw InferenceError.emptyPrompt }
    return encoded
  }

  static func boundaryAwareSuffixPrompt(
    contentTokens: [Int],
    maximumSuffixTokens: Int,
    maximumPromptTokens: Int,
    encodeCandidate: ([Int]) throws -> [Int]?
  ) throws -> [Int] {
    try Task.checkCancellation()
    let maximumLength = min(
      contentTokens.count,
      max(maximumSuffixTokens, 0)
    )
    guard maximumLength > 0 else { throw InferenceError.emptyPrompt }

    var removedTokens = 0
    while removedTokens < maximumLength {
      try Task.checkCancellation()
      let suffixLength = maximumLength - removedTokens
      let suffixTokens = Array(contentTokens.suffix(suffixLength))
      let candidate = try encodeCandidate(suffixTokens)
      try Task.checkCancellation()

      if let candidate, !candidate.isEmpty,
        candidate.count <= maximumPromptTokens
      {
        return candidate
      }

      // BPE token counts are not monotonic across string boundaries. Use the
      // observed overflow and exponential retreat instead of binary search.
      let overflow = max(
        (candidate?.count ?? maximumPromptTokens + 1)
          - maximumPromptTokens,
        1
      )
      let nextByOverflow = removedTokens + overflow
      let nextByDoubling = removedTokens == 0 ? 1 : removedTokens * 2
      removedTokens = min(
        maximumLength - 1,
        max(nextByOverflow, nextByDoubling)
      )

      if suffixLength == 1 { break }
    }

    throw InferenceError.promptTooLong
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
