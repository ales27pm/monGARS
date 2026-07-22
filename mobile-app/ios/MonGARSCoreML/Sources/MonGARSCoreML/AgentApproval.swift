import Foundation

public enum AgentPermissionState: String, Codable, Sendable, Equatable {
  case granted
  case limited
  case notDetermined
  case denied
  case restricted
  case unavailable
}

public enum AgentExecutionMode: String, Codable, Sendable, Equatable {
  case foreground
  case background
}

public protocol AgentPermissionProviding: Sendable {
  func state(for permission: AgentPermission) async -> AgentPermissionState
}

public struct AgentStaticPermissionProvider: AgentPermissionProviding, Sendable {
  private let states: [AgentPermission: AgentPermissionState]
  private let defaultState: AgentPermissionState

  public init(
    states: [AgentPermission: AgentPermissionState] = [:],
    defaultState: AgentPermissionState = .unavailable
  ) {
    self.states = states
    self.defaultState = defaultState
  }

  public func state(for permission: AgentPermission) async -> AgentPermissionState {
    states[permission] ?? defaultState
  }
}

public enum AgentPolicyDenial: Sendable, Equatable {
  case permissionDenied(AgentPermission)
  case permissionRestricted(AgentPermission)
  case permissionUnavailable(AgentPermission)
  case permissionPromptRequiresForeground(AgentPermission)
  case backgroundExecutionUnsupported(AgentToolID)
  case backgroundApprovalUnsupported(AgentToolID)
  case backgroundRiskTooHigh(AgentToolID)

  public var message: String {
    switch self {
    case let .permissionDenied(permission):
      return "Permission denied: \(permission.rawValue)."
    case let .permissionRestricted(permission):
      return "Permission restricted: \(permission.rawValue)."
    case let .permissionUnavailable(permission):
      return "Permission unavailable: \(permission.rawValue)."
    case let .permissionPromptRequiresForeground(permission):
      return "Permission \(permission.rawValue) must be requested in the foreground."
    case let .backgroundExecutionUnsupported(tool):
      return "Tool cannot run in the background: \(tool.rawValue)."
    case let .backgroundApprovalUnsupported(tool):
      return "Approval-requiring tool cannot run in the background: \(tool.rawValue)."
    case let .backgroundRiskTooHigh(tool):
      return "Tool risk is too high for background execution: \(tool.rawValue)."
    }
  }
}

public enum AgentPolicyDecision: Sendable, Equatable {
  case allowed
  case permissionRequestRequired(AgentPermission)
  case approvalRequired
  case denied(AgentPolicyDenial)
}

public struct AgentApprovalPolicy: Sendable {
  public init() {}

  public func evaluate(
    definition: AgentToolDefinition,
    arguments: AgentJSONArguments,
    permissionState: AgentPermissionState?,
    mode: AgentExecutionMode
  ) -> AgentPolicyDecision {
    if mode == .background {
      guard definition.supportsBackgroundExecution else {
        return .denied(.backgroundExecutionUnsupported(definition.id))
      }
      guard !definition.requiresApproval else {
        return .denied(.backgroundApprovalUnsupported(definition.id))
      }
      guard definition.risk < .high else {
        return .denied(.backgroundRiskTooHigh(definition.id))
      }
    }

    if let permission = effectivePermission(
      definition: definition,
      arguments: arguments
    ), let permissionDecision = evaluate(
      permission: permission,
      permissionState: permissionState ?? .unavailable,
      mode: mode,
      acceptsLimited: !requiresFullAccess(
        definition: definition,
        arguments: arguments,
        permission: permission
      )
    ) {
      return permissionDecision
    }

    return definition.requiresApproval ? .approvalRequired : .allowed
  }

  /// Some argument variants need an additional permission beyond the tool's
  /// primary catalog permission. The executor checks these requirements only
  /// after the primary permission succeeds, so the UI can request them in a
  /// deterministic foreground sequence.
  func additionalPermissions(
    definition: AgentToolDefinition,
    arguments: AgentJSONArguments
  ) -> [AgentPermission] {
    guard definition.id == "trigger.create" else { return [] }
    let schedule = (arguments["schedule"]?.stringValue ?? "")
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    return schedule == "before_next_event" ? [.calendar] : []
  }

  func evaluate(
    permission: AgentPermission,
    permissionState: AgentPermissionState,
    mode: AgentExecutionMode,
    acceptsLimited: Bool = true
  ) -> AgentPolicyDecision? {
    switch permissionState {
    case .granted:
      return nil
    case .limited where acceptsLimited:
      return nil
    case .limited where mode == .foreground:
      return .permissionRequestRequired(permission)
    case .limited:
      return .denied(.permissionPromptRequiresForeground(permission))
    case .notDetermined where mode == .foreground:
      return .permissionRequestRequired(permission)
    case .notDetermined:
      return .denied(.permissionPromptRequiresForeground(permission))
    case .denied:
      return .denied(.permissionDenied(permission))
    case .restricted:
      return .denied(.permissionRestricted(permission))
    case .unavailable:
      return .denied(.permissionUnavailable(permission))
    }
  }

  func requiresFullAccess(
    definition: AgentToolDefinition,
    arguments: AgentJSONArguments,
    permission: AgentPermission
  ) -> Bool {
    if definition.id == "calendar.list", permission == .calendar { return true }
    if definition.id == "reminders.list", permission == .reminders { return true }
    return definition.id == "trigger.create"
      && permission == .calendar
      && arguments["schedule"]?.stringValue == "before_next_event"
  }

  private func effectivePermission(
    definition: AgentToolDefinition,
    arguments: AgentJSONArguments
  ) -> AgentPermission? {
    // These tools inspect/request the permission state itself and must remain
    // callable before authorization has been granted.
    if definition.id == "alarm.authorization_status"
      || definition.id == "alarm.request_authorization" {
      return nil
    }
    // Supplying a concrete city avoids a GPS permission dependency.
    if definition.id == "weather" {
      let location = (arguments["location"]?.stringValue ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
      if !location.isEmpty,
         !["here", "current", "current location"].contains(location) {
        return nil
      }
    }
    return definition.permission
  }
}

public enum AgentApprovalStatus: String, Codable, Sendable, Equatable {
  case pending
  case approved
  case rejected
  case consumed
  case expired
}

public struct AgentApprovalRecord: Codable, Sendable, Equatable, Identifiable {
  public let id: UUID
  public let toolID: AgentToolID
  public let arguments: AgentJSONArguments
  public let createdAt: Date
  public let expiresAt: Date
  public fileprivate(set) var status: AgentApprovalStatus

  public init(
    id: UUID,
    toolID: AgentToolID,
    arguments: AgentJSONArguments,
    createdAt: Date,
    expiresAt: Date,
    status: AgentApprovalStatus
  ) {
    self.id = id
    self.toolID = toolID
    self.arguments = arguments
    self.createdAt = createdAt
    self.expiresAt = expiresAt
    self.status = status
  }
}

public enum AgentApprovalError: Error, Sendable, Equatable {
  case capacityReached
  case notFound
  case expired
  case notPending
  case notApproved
  case rejected
  case alreadyConsumed
  case bindingMismatch
}

public protocol AgentApprovalAuthorizing: Sendable {
  func requestApproval(
    toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) async -> Result<AgentApprovalRecord, AgentApprovalError>

  func approve(id: UUID) async -> Result<AgentApprovalRecord, AgentApprovalError>
  func reject(id: UUID) async -> Result<AgentApprovalRecord, AgentApprovalError>

  func consumeApproval(
    id: UUID,
    toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) async -> Result<AgentApprovalRecord, AgentApprovalError>

  func record(id: UUID) async -> AgentApprovalRecord?
}

/// In-memory authority suitable for one application process. Hosts that need
/// cross-process persistence can implement `AgentApprovalAuthorizing` while
/// retaining the same expiring, payload-bound, atomic consume semantics.
public actor AgentApprovalStore: AgentApprovalAuthorizing {
  private let maximumRecords: Int
  private let defaultTTL: TimeInterval
  private let maximumTTL: TimeInterval
  private let now: @Sendable () -> Date
  private var records: [UUID: AgentApprovalRecord] = [:]

  public init(
    maximumRecords: Int = 256,
    defaultTTL: TimeInterval = 600,
    maximumTTL: TimeInterval = 1_800,
    now: @escaping @Sendable () -> Date = { Date() }
  ) {
    self.maximumRecords = max(1, maximumRecords)
    self.maximumTTL = max(1, maximumTTL)
    self.defaultTTL = min(max(1, defaultTTL), max(1, maximumTTL))
    self.now = now
  }

  public func requestApproval(
    toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) -> Result<AgentApprovalRecord, AgentApprovalError> {
    expireRecords()
    evictTerminalRecordsIfNeeded()
    guard records.count < maximumRecords else {
      return .failure(.capacityReached)
    }
    let createdAt = now()
    let ttl = min(defaultTTL, maximumTTL)
    let record = AgentApprovalRecord(
      id: UUID(),
      toolID: AgentToolNormalizer.canonicalToolID(toolID.rawValue),
      arguments: arguments,
      createdAt: createdAt,
      expiresAt: createdAt.addingTimeInterval(ttl),
      status: .pending
    )
    records[record.id] = record
    return .success(record)
  }

  public func approve(id: UUID) -> Result<AgentApprovalRecord, AgentApprovalError> {
    guard var record = currentRecord(id: id) else { return .failure(.notFound) }
    guard record.status != .expired else { return .failure(.expired) }
    guard record.status == .pending else { return .failure(.notPending) }
    record.status = .approved
    records[id] = record
    return .success(record)
  }

  public func reject(id: UUID) -> Result<AgentApprovalRecord, AgentApprovalError> {
    guard var record = currentRecord(id: id) else { return .failure(.notFound) }
    guard record.status != .expired else { return .failure(.expired) }
    guard record.status == .pending else { return .failure(.notPending) }
    record.status = .rejected
    records[id] = record
    return .success(record)
  }

  public func consumeApproval(
    id: UUID,
    toolID: AgentToolID,
    arguments: AgentJSONArguments
  ) -> Result<AgentApprovalRecord, AgentApprovalError> {
    guard var record = currentRecord(id: id) else { return .failure(.notFound) }
    switch record.status {
    case .expired: return .failure(.expired)
    case .rejected: return .failure(.rejected)
    case .consumed: return .failure(.alreadyConsumed)
    case .pending: return .failure(.notApproved)
    case .approved: break
    }
    let canonicalToolID = AgentToolNormalizer.canonicalToolID(toolID.rawValue)
    guard record.toolID == canonicalToolID, record.arguments == arguments else {
      return .failure(.bindingMismatch)
    }
    record.status = .consumed
    records[id] = record
    return .success(record)
  }

  public func record(id: UUID) -> AgentApprovalRecord? {
    currentRecord(id: id)
  }

  public func allRecords() -> [AgentApprovalRecord] {
    expireRecords()
    return records.values.sorted { $0.createdAt < $1.createdAt }
  }

  private func currentRecord(id: UUID) -> AgentApprovalRecord? {
    expireRecord(id: id)
    return records[id]
  }

  private func expireRecords() {
    for id in Array(records.keys) {
      expireRecord(id: id)
    }
  }

  private func expireRecord(id: UUID) {
    guard var record = records[id],
          record.status == .pending || record.status == .approved,
          now() >= record.expiresAt else { return }
    record.status = .expired
    records[id] = record
  }

  private func evictTerminalRecordsIfNeeded() {
    guard records.count >= maximumRecords else { return }
    let terminal = records.values
      .filter { [.rejected, .consumed, .expired].contains($0.status) }
      .sorted { $0.createdAt < $1.createdAt }
    let required = records.count - maximumRecords + 1
    for record in terminal.prefix(required) {
      records.removeValue(forKey: record.id)
    }
  }
}
