import Foundation

#if canImport(UIKit)
import UIKit
#endif

public enum MonGARSAppIntentHandoffKind: String, Codable, CaseIterable, Sendable {
  case ask
  case memorySearch
  case memoryAdd
  case runTrigger
  case diagnostics
  /// Read-only placeholder returned for a handoff owned by another profile.
  /// This value is never accepted for persistence or execution.
  case masked

  fileprivate var maximumInputBytes: Int {
    switch self {
    case .ask, .runTrigger:
      return 512
    case .memorySearch:
      return 192
    case .memoryAdd:
      return 186
    case .diagnostics:
      return 0
    case .masked:
      return 0
    }
  }
}

public struct MonGARSAppIntentHandoff: Codable, Equatable, Sendable {
  public let id: UUID
  public let kind: MonGARSAppIntentHandoffKind
  public let input: String?
  public let createdAt: Date
  public let expiresAt: Date
  let profileScope: String

  public init(
    id: UUID,
    kind: MonGARSAppIntentHandoffKind,
    input: String?,
    createdAt: Date,
    expiresAt: Date,
    profileScope: String
  ) {
    self.id = id
    self.kind = kind
    self.input = input
    self.createdAt = createdAt
    self.expiresAt = expiresAt
    self.profileScope = profileScope
  }
}

/// A read-only view of a protected handoff. The record metadata is safe to
/// present so the user can discard an exact stale request, while `input` must
/// only cross the native bridge when `profileMatches` is true.
public struct MonGARSAppIntentHandoffLookup: Equatable, Sendable {
  public let handoff: MonGARSAppIntentHandoff
  public let profileMatches: Bool

  public init(handoff: MonGARSAppIntentHandoff, profileMatches: Bool) {
    self.handoff = handoff
    self.profileMatches = profileMatches
  }
}

public enum MonGARSAppIntentHandoffStoreError: LocalizedError, Equatable {
  case invalidInput
  case unavailable
  case persistenceFailed

  public var errorDescription: String? {
    switch self {
    case .invalidInput:
      return "The App Intent input is empty or exceeds its allowed size."
    case .unavailable:
      return "The protected App Intent handoff container is unavailable."
    case .persistenceFailed:
      return "The App Intent handoff could not be stored securely."
    }
  }
}

/// A single-slot, short-lived bridge between App Intents and the foreground
/// React Native application. User content lives in a protected file; shared
/// defaults contain only an opaque UUID, timestamp, and hashed profile scope
/// for cold-launch lookup.
public actor MonGARSAppIntentHandoffStore {
  public static let applicationGroup = "group.com.mongars.mobile"
  public static let defaultLifetime: TimeInterval = 10 * 60

  private enum Keys {
    static let identifier = "MonGARS.PendingAppIntentHandoffID"
    static let createdAt = "MonGARS.PendingAppIntentHandoffDate"
    static let activeProfileScope = "MonGARS.ActiveAppIntentProfileScope"
  }

  public static let shared: MonGARSAppIntentHandoffStore? = {
#if canImport(UIKit)
    guard
      let defaults = UserDefaults(suiteName: applicationGroup),
      let container = FileManager.default.containerURL(
        forSecurityApplicationGroupIdentifier: applicationGroup
      )
    else { return nil }
    let directory = container
      .appendingPathComponent("Library", isDirectory: true)
      .appendingPathComponent("Application Support", isDirectory: true)
      .appendingPathComponent("MonGARSAppIntentHandoffs", isDirectory: true)
    return .init(directoryURL: directory, defaults: defaults)
#else
    return nil
#endif
  }()

  private let directoryURL: URL
  private let defaults: UserDefaults
  private let now: @Sendable () -> Date
  private let lifetime: TimeInterval

  public init(
    directoryURL: URL,
    defaults: UserDefaults,
    lifetime: TimeInterval = MonGARSAppIntentHandoffStore.defaultLifetime,
    now: @escaping @Sendable () -> Date = Date.init
  ) {
    self.directoryURL = directoryURL
    self.defaults = defaults
    self.lifetime = max(1, min(lifetime, MonGARSAppIntentHandoffStore.defaultLifetime))
    self.now = now
  }

  @discardableResult
  public func enqueue(
    kind: MonGARSAppIntentHandoffKind,
    input rawInput: String?
  ) throws -> MonGARSAppIntentHandoff {
    guard kind != .masked else {
      throw MonGARSAppIntentHandoffStoreError.invalidInput
    }
    let input = try Self.validatedInput(rawInput, for: kind)
    let profileScope = activeProfileScope()
    let createdAt = now()
    let record = MonGARSAppIntentHandoff(
      id: UUID(),
      kind: kind,
      input: input,
      createdAt: createdAt,
      expiresAt: createdAt.addingTimeInterval(lifetime),
      profileScope: profileScope
    )

    do {
      try FileManager.default.createDirectory(
        at: directoryURL,
        withIntermediateDirectories: true
      )
      try Self.excludeFromBackup(directoryURL)
      try sweepExpiredPayloads(at: createdAt)
      try removeCurrentPayload()

      let fileURL = payloadURL(for: record.id)
      let data = try Self.encoder.encode(record)
#if canImport(UIKit)
      try data.write(to: fileURL, options: [.atomic, .completeFileProtection])
      try FileManager.default.setAttributes(
        [.protectionKey: FileProtectionType.complete],
        ofItemAtPath: fileURL.path
      )
#else
      try data.write(to: fileURL, options: .atomic)
#endif
      try Self.excludeFromBackup(fileURL)

      // Publish the pointer only after the complete protected payload exists.
      defaults.set(record.id.uuidString.lowercased(), forKey: Keys.identifier)
      defaults.set(record.createdAt, forKey: Keys.createdAt)
      guard defaults.synchronize() else {
        try? FileManager.default.removeItem(at: fileURL)
        clearPointer()
        throw MonGARSAppIntentHandoffStoreError.persistenceFailed
      }
      return record
    } catch let error as MonGARSAppIntentHandoffStoreError {
      throw error
    } catch {
      clearPointer()
      throw MonGARSAppIntentHandoffStoreError.persistenceFailed
    }
  }

  func pending() -> MonGARSAppIntentHandoff? {
    let current = now()
    // Cleanup is best-effort on reads so a transient filesystem error cannot
    // manufacture a pending request. Enqueue remains fail-closed on errors.
    try? sweepExpiredPayloads(at: current)
    guard
      let rawID = defaults.string(forKey: Keys.identifier),
      let id = UUID(uuidString: rawID),
      let pointerDate = defaults.object(forKey: Keys.createdAt) as? Date
    else {
      clearInvalidState()
      return nil
    }

    guard
      pointerDate <= current.addingTimeInterval(60),
      current.timeIntervalSince(pointerDate) <= lifetime,
      let data = try? Data(contentsOf: payloadURL(for: id)),
      data.count <= 8_192,
      let record = try? Self.decoder.decode(MonGARSAppIntentHandoff.self, from: data),
      record.id == id,
      abs(record.createdAt.timeIntervalSince(pointerDate)) < 1,
      record.createdAt <= current.addingTimeInterval(60),
      record.expiresAt > current,
      record.expiresAt.timeIntervalSince(record.createdAt) <= lifetime + 1,
      record.kind != .masked,
      Self.isOpaqueProfileScope(record.profileScope),
      (try? Self.validatedInput(record.input, for: record.kind)) == record.input
    else {
      clearInvalidState()
      return nil
    }
    return record
  }

  /// Returns bounded metadata for an exact discard flow. Callers must never
  /// reveal `handoff.input` unless `profileMatches` is true.
  public func pending(rawOwnerID: String) -> MonGARSAppIntentHandoffLookup? {
    guard let record = pending(), let scope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID)
    else { return nil }
    let profileMatches = record.profileScope == scope
    let visibleRecord = profileMatches ? record : MonGARSAppIntentHandoff(
      id: record.id,
      kind: .masked,
      input: nil,
      createdAt: record.createdAt,
      expiresAt: record.expiresAt,
      profileScope: record.profileScope
    )
    return .init(
      handoff: visibleRecord,
      profileMatches: profileMatches
    )
  }

  /// Updates the profile future App Intents bind to. Only the SHA-256-derived
  /// opaque scope is shared with the system-facing intent layer.
  @discardableResult
  public func setActiveProfile(rawOwnerID: String) -> Bool {
    guard let scope = AgentOpaqueProfileScope.make(rawOwnerID: rawOwnerID) else { return false }
    defaults.set(scope, forKey: Keys.activeProfileScope)
    return defaults.synchronize()
  }

  /// Consumes only the exact handoff that was shown to the foreground UI.
  /// A stale or substituted identifier never clears a newer request.
  public func acknowledge(expectedID: UUID) -> Bool {
    guard let record = pending(), record.id == expectedID else { return false }
    do {
      try FileManager.default.removeItem(at: payloadURL(for: record.id))
      clearPointer()
      return true
    } catch {
      return false
    }
  }

  public func acknowledge(expectedID: UUID, rawOwnerID: String) -> Bool {
    guard let lookup = pending(rawOwnerID: rawOwnerID),
      lookup.profileMatches,
      lookup.handoff.id == expectedID else {
      return false
    }
    return acknowledge(expectedID: expectedID)
  }

  public func consumeExactMemoryAction(
    expectedID: UUID,
    rawOwnerID: String,
    expectedKind: MonGARSAppIntentHandoffKind,
    expectedInput: String
  ) -> MonGARSAppIntentHandoff? {
    guard expectedKind == .memorySearch || expectedKind == .memoryAdd,
      let validatedInput = try? Self.validatedInput(expectedInput, for: expectedKind),
      let lookup = pending(rawOwnerID: rawOwnerID),
      lookup.profileMatches else { return nil }
    let record = lookup.handoff
    guard
      record.id == expectedID,
      record.kind == expectedKind,
      record.input == validatedInput,
      acknowledge(expectedID: expectedID) else { return nil }
    return record
  }

  private func payloadURL(for id: UUID) -> URL {
    directoryURL.appendingPathComponent("\(id.uuidString.lowercased()).json")
  }

  /// Removes only canonical payload files whose bounded record has expired.
  /// An unpointed but still-valid recent record is deliberately retained until
  /// its TTL: it may be from the crash window between atomic file persistence
  /// and publishing the opaque UserDefaults pointer.
  private func sweepExpiredPayloads(at current: Date) throws {
    guard FileManager.default.fileExists(atPath: directoryURL.path) else { return }
    let resourceKeys: Set<URLResourceKey> = [
      .contentModificationDateKey,
      .fileSizeKey,
      .isRegularFileKey,
      .isSymbolicLinkKey,
    ]
    let files = try FileManager.default.contentsOfDirectory(
      at: directoryURL,
      includingPropertiesForKeys: Array(resourceKeys),
      options: [.skipsHiddenFiles]
    )

    for fileURL in files {
      guard let expectedID = Self.canonicalPayloadIdentifier(for: fileURL) else {
        continue
      }
      guard let values = try? fileURL.resourceValues(forKeys: resourceKeys),
        values.isRegularFile == true,
        values.isSymbolicLink != true else { continue }

      let record: MonGARSAppIntentHandoff? = {
        guard let fileSize = values.fileSize, fileSize <= 8_192,
          let data = try? Data(contentsOf: fileURL),
          data.count <= 8_192 else { return nil }
        return try? Self.decoder.decode(MonGARSAppIntentHandoff.self, from: data)
      }()

      let shouldRemove: Bool
      if let record, isValidPayloadRecord(
        record,
        expectedID: expectedID,
        current: current
      ) {
        shouldRemove = record.expiresAt <= current
      } else if let modifiedAt = values.contentModificationDate {
        // Invalid canonical payloads cannot provide a trustworthy expiry.
        // Retain them for at most one TTL from their filesystem timestamp.
        shouldRemove = current.timeIntervalSince(modifiedAt) >= lifetime
      } else {
        shouldRemove = false
      }

      if shouldRemove {
        try FileManager.default.removeItem(at: fileURL)
      }
    }
  }

  private func isValidPayloadRecord(
    _ record: MonGARSAppIntentHandoff,
    expectedID: UUID,
    current: Date
  ) -> Bool {
    record.id == expectedID
      && record.expiresAt > record.createdAt
      && record.createdAt <= current.addingTimeInterval(60)
      && record.expiresAt.timeIntervalSince(record.createdAt) <= lifetime + 1
      && record.kind != .masked
      && Self.isOpaqueProfileScope(record.profileScope)
      && (try? Self.validatedInput(record.input, for: record.kind)) == record.input
  }

  private static func canonicalPayloadIdentifier(for fileURL: URL) -> UUID? {
    guard fileURL.pathExtension == "json" else { return nil }
    let stem = fileURL.deletingPathExtension().lastPathComponent
    guard let id = UUID(uuidString: stem), stem == id.uuidString.lowercased() else {
      return nil
    }
    return id
  }

  private func removeCurrentPayload() throws {
    guard
      let rawID = defaults.string(forKey: Keys.identifier),
      let id = UUID(uuidString: rawID)
    else {
      clearPointer()
      return
    }
    let url = payloadURL(for: id)
    if FileManager.default.fileExists(atPath: url.path) {
      try FileManager.default.removeItem(at: url)
    }
    clearPointer()
  }

  private func clearInvalidState() {
    if
      let rawID = defaults.string(forKey: Keys.identifier),
      let id = UUID(uuidString: rawID)
    {
      try? FileManager.default.removeItem(at: payloadURL(for: id))
    }
    clearPointer()
  }

  private func clearPointer() {
    defaults.removeObject(forKey: Keys.identifier)
    defaults.removeObject(forKey: Keys.createdAt)
    defaults.synchronize()
  }

  private func activeProfileScope() -> String {
    if let value = defaults.string(forKey: Keys.activeProfileScope),
      Self.isOpaqueProfileScope(value) {
      return value
    }
    return AgentOpaqueProfileScope.make(rawOwnerID: "guest")
      ?? "profile.invalid"
  }

  private static func isOpaqueProfileScope(_ value: String) -> Bool {
    guard value.hasPrefix("profile.") else { return false }
    let digest = value.dropFirst("profile.".count)
    return digest.count == 64 && digest.allSatisfy { $0.isHexDigit }
  }

  private static func validatedInput(
    _ rawInput: String?,
    for kind: MonGARSAppIntentHandoffKind
  ) throws -> String? {
    if kind == .diagnostics || kind == .masked {
      guard rawInput?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false else {
        throw MonGARSAppIntentHandoffStoreError.invalidInput
      }
      return nil
    }

    guard let rawInput else {
      throw MonGARSAppIntentHandoffStoreError.invalidInput
    }
    let input = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
    guard
      !input.isEmpty,
      input.utf8.count <= kind.maximumInputBytes,
      !input.unicodeScalars.contains(where: {
        ($0.value < 32 && $0.value != 10 && $0.value != 9)
          || ($0.value >= 127 && $0.value <= 159)
          || $0.value == 8_232
          || $0.value == 8_233
      })
    else {
      throw MonGARSAppIntentHandoffStoreError.invalidInput
    }
    return input
  }

  private static func excludeFromBackup(_ url: URL) throws {
#if canImport(Darwin)
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    var mutableURL = url
    try mutableURL.setResourceValues(values)
#else
    _ = url
#endif
  }

  private static let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.sortedKeys]
    return encoder
  }()

  private static let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return decoder
  }()
}
