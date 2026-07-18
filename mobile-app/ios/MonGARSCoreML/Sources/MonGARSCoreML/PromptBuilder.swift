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
      let emptyLatest = ChatMessage(role: latest.role, content: "")
      let emptyPromptTokens = try encode(
        [system, emptyLatest],
        tokenizer: tokenizer
      )
      encoded = try tokenLevelSuffixPrompt(
        contentTokens: contentTokens,
        emptyPromptTokens: emptyPromptTokens,
        maximumSuffixTokens: maximumSuffixTokens,
        maximumPromptTokens: maximumPromptTokens,
        messageEndTokenID: MonGARSModelManifest.chatMessageEndTokenID
      )

      if encoded.count > maximumPromptTokens {
        throw InferenceError.promptTooLong
      }
    }

    guard !encoded.isEmpty else { throw InferenceError.emptyPrompt }
    return encoded
  }

  static func tokenLevelSuffixPrompt(
    contentTokens: [Int],
    emptyPromptTokens: [Int],
    maximumSuffixTokens: Int,
    maximumPromptTokens: Int,
    messageEndTokenID: Int
  ) throws -> [Int] {
    try Task.checkCancellation()
    guard let messageEndIndex = emptyPromptTokens.lastIndex(
      of: messageEndTokenID
    ) else {
      throw InferenceError.invalidModel(
        "Le gabarit de conversation ne contient pas le marqueur de fin attendu."
      )
    }

    let prefix = Array(emptyPromptTokens[..<messageEndIndex])
    let suffix = Array(emptyPromptTokens[messageEndIndex...])
    let availableContentTokens = maximumPromptTokens
      - prefix.count
      - suffix.count
    guard availableContentTokens > 0 else {
      throw InferenceError.promptTooLong
    }

    let suffixLength = min(
      contentTokens.count,
      min(max(maximumSuffixTokens, 0), availableContentTokens)
    )
    guard suffixLength > 0 else { throw InferenceError.emptyPrompt }

    let selectedContent = contentTokens.suffix(suffixLength)
    let result = prefix + selectedContent + suffix
    guard result.count <= maximumPromptTokens else {
      throw InferenceError.promptTooLong
    }
    try Task.checkCancellation()
    return result
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
