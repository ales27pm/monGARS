import CoreFoundation
import Foundation

/// A lossless, Sendable representation of values accepted by the agent protocol.
/// Numbers remain distinct from booleans and no string coercion is performed.
public indirect enum AgentJSONValue: Sendable, Equatable, Codable {
  case null
  case bool(Bool)
  case number(Double)
  case string(String)
  case array([AgentJSONValue])
  case object([String: AgentJSONValue])

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let value = try? container.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? container.decode(Double.self) {
      guard value.isFinite else {
        throw DecodingError.dataCorruptedError(
          in: container,
          debugDescription: "Agent JSON numbers must be finite."
        )
      }
      self = .number(value)
    } else if let value = try? container.decode(String.self) {
      self = .string(value)
    } else if let value = try? container.decode([AgentJSONValue].self) {
      self = .array(value)
    } else if let value = try? container.decode([String: AgentJSONValue].self) {
      self = .object(value)
    } else {
      throw DecodingError.typeMismatch(
        AgentJSONValue.self,
        .init(
          codingPath: decoder.codingPath,
          debugDescription: "Unsupported agent JSON value."
        )
      )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .null:
      try container.encodeNil()
    case let .bool(value):
      try container.encode(value)
    case let .number(value):
      guard value.isFinite else {
        throw EncodingError.invalidValue(
          value,
          .init(
            codingPath: encoder.codingPath,
            debugDescription: "Agent JSON numbers must be finite."
          )
        )
      }
      try container.encode(value)
    case let .string(value):
      try container.encode(value)
    case let .array(value):
      try container.encode(value)
    case let .object(value):
      try container.encode(value)
    }
  }

  public var stringValue: String? {
    guard case let .string(value) = self else { return nil }
    return value
  }

  public var numberValue: Double? {
    guard case let .number(value) = self else { return nil }
    return value
  }

  public var boolValue: Bool? {
    guard case let .bool(value) = self else { return nil }
    return value
  }

  public var arrayValue: [AgentJSONValue]? {
    guard case let .array(value) = self else { return nil }
    return value
  }

  public var objectValue: [String: AgentJSONValue]? {
    guard case let .object(value) = self else { return nil }
    return value
  }

  /// Stable JSON used for duplicate-call keys and approval payload binding.
  public func canonicalJSONString() throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(self)
    guard let value = String(data: data, encoding: .utf8) else {
      throw AgentJSONEncodingError.invalidUTF8
    }
    return value
  }
}

public typealias AgentJSONArguments = [String: AgentJSONValue]

public enum AgentJSONEncodingError: Error, Sendable, Equatable {
  case invalidUTF8
}

public struct AgentFoundationJSONLimits: Sendable, Equatable {
  public let maximumDepth: Int
  public let maximumArrayCount: Int
  public let maximumObjectCount: Int
  public let maximumKeyBytes: Int
  public let maximumStringBytes: Int

  public init(
    maximumDepth: Int = 16,
    maximumArrayCount: Int = 4_096,
    maximumObjectCount: Int = 4_096,
    maximumKeyBytes: Int = 1_024,
    maximumStringBytes: Int = 1_048_576
  ) {
    self.maximumDepth = max(0, maximumDepth)
    self.maximumArrayCount = max(0, maximumArrayCount)
    self.maximumObjectCount = max(0, maximumObjectCount)
    self.maximumKeyBytes = max(0, maximumKeyBytes)
    self.maximumStringBytes = max(0, maximumStringBytes)
  }
}

public enum AgentFoundationJSONError: Error, Sendable, Equatable {
  case depthExceeded
  case collectionTooLarge
  case invalidObjectKey
  case stringTooLarge
  case nonFiniteNumber
  case unsupportedValue
}

/// Converts values produced by JSONSerialization or React Native without
/// allowing NSNumber(0/1) to change type into Bool. CoreFoundation owns the
/// only reliable distinction because Swift's Objective-C casts bridge both.
public enum AgentFoundationJSON {
  public static func decode(
    _ value: Any,
    limits: AgentFoundationJSONLimits = .init()
  ) throws -> AgentJSONValue {
    try decode(value, limits: limits, depth: 0)
  }

  private static func decode(
    _ value: Any,
    limits: AgentFoundationJSONLimits,
    depth: Int
  ) throws -> AgentJSONValue {
    guard depth <= limits.maximumDepth else {
      throw AgentFoundationJSONError.depthExceeded
    }

    switch value {
    case is NSNull:
      return .null
    case let number as NSNumber:
      if CFGetTypeID(number) == CFBooleanGetTypeID() {
        return .bool(number.boolValue)
      }
      let output = number.doubleValue
      guard output.isFinite else {
        throw AgentFoundationJSONError.nonFiniteNumber
      }
      return .number(output)
    case let bool as Bool:
      // Covers a native Swift Bool should it not bridge through NSNumber.
      return .bool(bool)
    case let string as String:
      guard string.utf8.count <= limits.maximumStringBytes else {
        throw AgentFoundationJSONError.stringTooLarge
      }
      return .string(string)
    case let array as NSArray:
      guard array.count <= limits.maximumArrayCount else {
        throw AgentFoundationJSONError.collectionTooLarge
      }
      return .array(try array.map {
        try decode($0, limits: limits, depth: depth + 1)
      })
    case let dictionary as NSDictionary:
      guard dictionary.count <= limits.maximumObjectCount else {
        throw AgentFoundationJSONError.collectionTooLarge
      }
      var output: [String: AgentJSONValue] = [:]
      output.reserveCapacity(dictionary.count)
      for (rawKey, rawValue) in dictionary {
        guard let key = rawKey as? String,
          !key.isEmpty,
          key.utf8.count <= limits.maximumKeyBytes else {
          throw AgentFoundationJSONError.invalidObjectKey
        }
        output[key] = try decode(rawValue, limits: limits, depth: depth + 1)
      }
      return .object(output)
    default:
      throw AgentFoundationJSONError.unsupportedValue
    }
  }
}

extension AgentJSONValue: ExpressibleByNilLiteral {
  public init(nilLiteral: ()) {
    self = .null
  }
}

extension AgentJSONValue: ExpressibleByBooleanLiteral {
  public init(booleanLiteral value: Bool) {
    self = .bool(value)
  }
}

extension AgentJSONValue: ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Int) {
    self = .number(Double(value))
  }
}

extension AgentJSONValue: ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self = .number(value)
  }
}

extension AgentJSONValue: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self = .string(value)
  }
}

extension AgentJSONValue: ExpressibleByArrayLiteral {
  public init(arrayLiteral elements: AgentJSONValue...) {
    self = .array(elements)
  }
}

extension AgentJSONValue: ExpressibleByDictionaryLiteral {
  public init(dictionaryLiteral elements: (String, AgentJSONValue)...) {
    self = .object(Dictionary(uniqueKeysWithValues: elements))
  }
}
