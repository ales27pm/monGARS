import Foundation

public struct AgentToolAction: Sendable, Equatable {
  public let tool: String
  public let arguments: AgentJSONArguments

  public init(tool: String, arguments: AgentJSONArguments) {
    self.tool = tool
    self.arguments = arguments
  }
}

public enum AgentTurn: Sendable, Equatable {
  case action(thought: String?, AgentToolAction)
  case final(thought: String?, String)

  public var thought: String? {
    switch self {
    case let .action(thought, _), let .final(thought, _): return thought
    }
  }
}

public enum AgentTurnParseError: Error, Sendable, Equatable {
  case outputTooLarge
  case invalidJSON
  case duplicateObjectKey(String)
  case rootMustBeObject
  case extraTopLevelKeys([String])
  case invalidThought
  case missingActionOrFinal
  case actionAndFinalAreMutuallyExclusive
  case actionMustBeObject
  case extraActionKeys([String])
  case missingTool
  case toolMustBeNonEmptyString
  case missingArguments
  case argumentsMustBeObject
  case finalMustBeNonEmptyString

  public var diagnostic: String {
    switch self {
    case .outputTooLarge: return "Agent output exceeds the size limit."
    case .invalidJSON: return "Output is not one complete JSON value."
    case let .duplicateObjectKey(key): return "Duplicate JSON object key: \(key)."
    case .rootMustBeObject: return "Agent output root must be an object."
    case let .extraTopLevelKeys(keys): return "Unexpected top-level keys: \(keys.joined(separator: ", "))."
    case .invalidThought: return "Optional thought must be a string."
    case .missingActionOrFinal: return "Output must contain exactly one action or final field."
    case .actionAndFinalAreMutuallyExclusive: return "Output cannot contain both action and final."
    case .actionMustBeObject: return "Action must be an object."
    case let .extraActionKeys(keys): return "Unexpected action keys: \(keys.joined(separator: ", "))."
    case .missingTool: return "Action is missing tool."
    case .toolMustBeNonEmptyString: return "Action tool must be a non-empty string."
    case .missingArguments: return "Action is missing args."
    case .argumentsMustBeObject: return "Action args must be an object."
    case .finalMustBeNonEmptyString: return "Final must be a non-empty string."
    }
  }
}

public enum AgentTurnParser {
  public static let maximumModelOutputBytes = 64_000
  public static let responseJSONSchema = #"{"type":"object","oneOf":[{"required":["action"],"properties":{"thought":{"type":"string"},"action":{"type":"object","required":["tool","args"],"properties":{"tool":{"type":"string"},"args":{"type":"object"}},"additionalProperties":false}},"additionalProperties":false},{"required":["final"],"properties":{"thought":{"type":"string"},"final":{"type":"string"}},"additionalProperties":false}]}"#

  public static func parse(_ raw: String) -> Result<AgentTurn, AgentTurnParseError> {
    guard let data = raw.data(using: .utf8) else {
      return .failure(.invalidJSON)
    }
    guard data.count <= maximumModelOutputBytes else {
      return .failure(.outputTooLarge)
    }
    guard let decoded = try? JSONDecoder().decode(AgentJSONValue.self, from: data) else {
      return .failure(.invalidJSON)
    }
    var duplicateScanner = AgentJSONDuplicateKeyScanner(data: data)
    if let duplicateKey = duplicateScanner.firstDuplicateKey() {
      return .failure(.duplicateObjectKey(duplicateKey))
    }
    guard case let .object(root) = decoded else {
      return .failure(.rootMustBeObject)
    }
    let extraRootKeys = Set(root.keys).subtracting(["thought", "action", "final"])
    guard extraRootKeys.isEmpty else {
      return .failure(.extraTopLevelKeys(extraRootKeys.sorted()))
    }

    let thought: String?
    if let value = root["thought"] {
      guard case let .string(string) = value else {
        return .failure(.invalidThought)
      }
      thought = string
    } else {
      thought = nil
    }

    let hasAction = root["action"] != nil
    let hasFinal = root["final"] != nil
    guard hasAction || hasFinal else { return .failure(.missingActionOrFinal) }
    guard !(hasAction && hasFinal) else {
      return .failure(.actionAndFinalAreMutuallyExclusive)
    }

    if let actionValue = root["action"] {
      guard case let .object(action) = actionValue else {
        return .failure(.actionMustBeObject)
      }
      let extraActionKeys = Set(action.keys).subtracting(["tool", "args"])
      guard extraActionKeys.isEmpty else {
        return .failure(.extraActionKeys(extraActionKeys.sorted()))
      }
      guard let toolValue = action["tool"] else { return .failure(.missingTool) }
      guard case let .string(rawTool) = toolValue else {
        return .failure(.toolMustBeNonEmptyString)
      }
      let tool = rawTool.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !tool.isEmpty else { return .failure(.toolMustBeNonEmptyString) }
      guard let argumentsValue = action["args"] else {
        return .failure(.missingArguments)
      }
      guard case let .object(arguments) = argumentsValue else {
        return .failure(.argumentsMustBeObject)
      }
      return .success(.action(
        thought: thought,
        .init(tool: tool, arguments: arguments)
      ))
    }

    guard case let .string(final)? = root["final"],
          !final.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      return .failure(.finalMustBeNonEmptyString)
    }
    return .success(.final(thought: thought, final))
  }
}

/// JSONDecoder accepts duplicate object keys using last-value-wins semantics.
/// Agent decisions reject those ambiguous payloads before schema validation.
private struct AgentJSONDuplicateKeyScanner {
  private let bytes: [UInt8]
  private var index = 0
  private var duplicateKey: String?

  init(data: Data) {
    bytes = Array(data)
  }

  mutating func firstDuplicateKey() -> String? {
    skipWhitespace()
    guard parseValue() else { return nil }
    skipWhitespace()
    guard index == bytes.count else { return nil }
    return duplicateKey
  }

  private mutating func parseValue() -> Bool {
    skipWhitespace()
    guard index < bytes.count else { return false }
    switch bytes[index] {
    case 0x7B: return parseObject()
    case 0x5B: return parseArray()
    case 0x22: return parseString() != nil
    default: return parsePrimitive()
    }
  }

  private mutating func parseObject() -> Bool {
    index += 1
    skipWhitespace()
    var keys: Set<String> = []
    if consume(0x7D) { return true }
    while index < bytes.count {
      guard let key = parseString() else { return false }
      if keys.contains(key), duplicateKey == nil {
        duplicateKey = key
      }
      keys.insert(key)
      skipWhitespace()
      guard consume(0x3A), parseValue() else { return false }
      skipWhitespace()
      if consume(0x7D) { return true }
      guard consume(0x2C) else { return false }
      skipWhitespace()
    }
    return false
  }

  private mutating func parseArray() -> Bool {
    index += 1
    skipWhitespace()
    if consume(0x5D) { return true }
    while index < bytes.count {
      guard parseValue() else { return false }
      skipWhitespace()
      if consume(0x5D) { return true }
      guard consume(0x2C) else { return false }
    }
    return false
  }

  private mutating func parseString() -> String? {
    guard index < bytes.count, bytes[index] == 0x22 else { return nil }
    let start = index
    index += 1
    var escaped = false
    while index < bytes.count {
      let byte = bytes[index]
      index += 1
      if escaped {
        escaped = false
      } else if byte == 0x5C {
        escaped = true
      } else if byte == 0x22 {
        let encoded = Data(bytes[start..<index])
        return try? JSONDecoder().decode(String.self, from: encoded)
      }
    }
    return nil
  }

  private mutating func parsePrimitive() -> Bool {
    let start = index
    while index < bytes.count {
      let byte = bytes[index]
      if byte == 0x2C || byte == 0x5D || byte == 0x7D || isWhitespace(byte) {
        break
      }
      index += 1
    }
    return index > start
  }

  private mutating func skipWhitespace() {
    while index < bytes.count, isWhitespace(bytes[index]) {
      index += 1
    }
  }

  private mutating func consume(_ byte: UInt8) -> Bool {
    guard index < bytes.count, bytes[index] == byte else { return false }
    index += 1
    return true
  }

  private func isWhitespace(_ byte: UInt8) -> Bool {
    byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D
  }
}

public enum AgentOutputSanitizer {
  public static func sanitizeFinal(
    _ raw: String,
    maximumCharacters: Int = 4_000
  ) -> String {
    sanitize(raw, maximumCharacters: maximumCharacters)
  }

  public static func sanitizeToolOutput(
    _ raw: String,
    maximumCharacters: Int
  ) -> String {
    sanitize(raw, maximumCharacters: maximumCharacters)
  }

  public static func sanitizeJSON(
    _ value: AgentJSONValue,
    maximumStringCharacters: Int
  ) -> AgentJSONValue {
    switch value {
    case .null, .bool, .number:
      return value
    case let .string(string):
      return .string(sanitize(string, maximumCharacters: maximumStringCharacters))
    case let .array(array):
      return .array(array.map {
        sanitizeJSON($0, maximumStringCharacters: maximumStringCharacters)
      })
    case let .object(object):
      return .object(object.mapValues {
        sanitizeJSON($0, maximumStringCharacters: maximumStringCharacters)
      })
    }
  }

  private static func sanitize(
    _ raw: String,
    maximumCharacters: Int
  ) -> String {
    var filtered = ""
    filtered.reserveCapacity(min(raw.count, max(1, maximumCharacters)))
    for scalar in raw.unicodeScalars {
      let allowedControl = scalar == "\n" || scalar == "\t"
      if allowedControl || (scalar.value >= 32 && scalar.value != 127) {
        filtered.unicodeScalars.append(scalar)
      }
    }
    for token in [
      "<|assistant|>", "<|user|>", "<|system|>", "<|eot_id|>",
      "<|start_header_id|>", "<|end_header_id|>",
    ] {
      filtered = filtered.replacingOccurrences(of: token, with: "")
    }
    filtered = filtered.replacingOccurrences(of: "\r\n", with: "\n")
      .replacingOccurrences(of: "\r", with: "\n")
    while filtered.contains("\n\n\n") {
      filtered = filtered.replacingOccurrences(of: "\n\n\n", with: "\n\n")
    }
    let trimmed = filtered.trimmingCharacters(in: .whitespacesAndNewlines)
    return String(trimmed.prefix(max(1, maximumCharacters)))
  }
}
