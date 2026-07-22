import XCTest
@testable import MonGARSCoreML

final class AgentToolValidationTests: XCTestCase {
  func testFoundationJSONKeepsNumbersAndBooleansDistinctRecursively() throws {
    let foundation: NSDictionary = [
      "zero": NSNumber(value: 0),
      "one": NSNumber(value: 1),
      "two": NSNumber(value: 2),
      "true": NSNumber(value: true),
      "false": NSNumber(value: false),
      "nested": NSArray(array: [
        NSNumber(value: 0),
        NSNumber(value: true),
        ["value": NSNumber(value: 1)],
      ]),
    ]

    let decoded = try AgentFoundationJSON.decode(foundation)

    XCTAssertEqual(decoded, .object([
      "zero": .number(0),
      "one": .number(1),
      "two": .number(2),
      "true": .bool(true),
      "false": .bool(false),
      "nested": .array([
        .number(0),
        .bool(true),
        .object(["value": .number(1)]),
      ]),
    ]))
  }

  func testFoundationJSONRejectsLimitsInsteadOfDroppingValues() {
    XCTAssertThrowsError(try AgentFoundationJSON.decode(
      ["items": [1, 2]],
      limits: .init(maximumArrayCount: 1)
    )) { error in
      XCTAssertEqual(error as? AgentFoundationJSONError, .collectionTooLarge)
    }
  }

  func testAgentJSONRoundTripsWithoutBoolNumberCoercion() throws {
    let value: AgentJSONValue = [
      "bool": true,
      "number": 1,
      "array": ["text", nil, 2.5],
      "object": ["b": 2, "a": 1],
    ]
    let data = try JSONEncoder().encode(value)
    let decoded = try JSONDecoder().decode(AgentJSONValue.self, from: data)

    XCTAssertEqual(decoded, value)
    XCTAssertEqual(decoded.objectValue?["bool"], .bool(true))
    XCTAssertEqual(decoded.objectValue?["number"], .number(1))
  }

  func testCanonicalJSONSortsObjectKeys() throws {
    let first = try AgentJSONValue.object(["b": 2, "a": 1]).canonicalJSONString()
    let second = try AgentJSONValue.object(["a": 1, "b": 2]).canonicalJSONString()

    XCTAssertEqual(first, second)
    XCTAssertEqual(first, #"{"a":1,"b":2}"#)
  }

  func testCanonicalToolAliasesAreSemanticsPreserving() {
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID(" Weather-Current "), "weather")
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID("contacts.lookup"), "contacts.search")
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID("outlook.message.mark.read"), "outlook.message.mark_read")
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID("timer start"), "alarm.countdown")
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID("unknown-tool"), "unknown.tool")

    // Opening a URL is user-visible; it must not silently become a read-only fetch.
    XCTAssertEqual(AgentToolNormalizer.canonicalToolID("open.url"), "open.url")
    XCTAssertNil(AgentToolCatalog.definition(for: "open.url"))
  }

  func testArgumentAliasesNormalizeToCanonicalFields() {
    let result = AgentToolValidator.validate(
      rawToolID: "mail",
      arguments: ["recipient": "person@example.com", "title": "Hello", "text": "Body"],
      availableToolIDs: ["mail.draft"]
    )

    guard case let .success(call) = result else {
      return XCTFail("Expected valid alias normalization, got \(result)")
    }
    XCTAssertEqual(call.toolID, "mail.draft")
    XCTAssertEqual(call.arguments, [
      "to": "person@example.com",
      "subject": "Hello",
      "body": "Body",
    ])
  }

  func testUnknownToolIsRejected() {
    XCTAssertEqual(
      failure("shell.execute", [:], available: AgentToolCatalog.canonicalIDs),
      .unknownTool("shell.execute")
    )
  }

  func testKnownButUnavailableToolIsRejected() {
    XCTAssertEqual(
      failure("web.search", ["query": "Swift"], available: ["weather"]),
      .unavailableTool("web.search")
    )
  }

  func testMissingAndEmptyRequiredArgumentsAreRejected() {
    XCTAssertEqual(
      failure("web.search", [:], available: ["web.search"]),
      .missingRequiredArgument(tool: "web.search", argument: "query")
    )
    XCTAssertEqual(
      failure("web.search", ["query": "  \n"], available: ["web.search"]),
      .emptyRequiredArgument(tool: "web.search", argument: "query")
    )
  }

  func testArgumentTypesAreNotCoerced() {
    XCTAssertEqual(
      failure("rag.search", ["query": "notes", "limit": "5"], available: ["rag.search"]),
      .invalidArgumentType(
        tool: "rag.search",
        argument: "limit",
        expected: .number,
        actual: .string
      )
    )
    XCTAssertEqual(
      failure("outlook.folders.list", ["includeHidden": 1], available: ["outlook.folders.list"]),
      .invalidArgumentType(
        tool: "outlook.folders.list",
        argument: "includeHidden",
        expected: .boolean,
        actual: .number
      )
    )
  }

  func testInvalidEnumAndExtraArgumentsAreRejected() {
    XCTAssertEqual(
      failure(
        "rag.search",
        ["query": "notes", "sourceScope": "cloud"],
        available: ["rag.search"]
      ),
      .invalidEnumValue(
        tool: "rag.search",
        argument: "sourceScope",
        allowed: ["all", "documents", "notes", "photos"]
      )
    )
    XCTAssertEqual(
      failure(
        "web.search",
        ["query": "Swift", "execute": true],
        available: ["web.search"]
      ),
      .extraArguments(tool: "web.search", arguments: ["execute"])
    )
  }

  func testEnumArgumentsNormalizeCaseInsensitivelyToCanonicalValues() throws {
    let call = try valid(
      "rag.search",
      ["query": "notes", "sourceScope": "PHOTOS"],
      available: ["rag.search"]
    )

    XCTAssertEqual(call.arguments["sourceScope"], "photos")
  }

  func testConflictingCanonicalAndAliasArgumentsAreRejected() {
    XCTAssertEqual(
      failure(
        "web.search",
        ["query": "Swift", "q": "Kotlin"],
        available: ["web.search"]
      ),
      .conflictingAlias(tool: "web.search", canonicalArgument: "query", alias: "q")
    )
  }

  func testOutlookAliasesCanonicalizeBeforeRequiredRelationships() throws {
    let move = try valid(
      "outlook.message.move",
      ["messageId": "m1", "destinationId": "inbox"],
      available: ["outlook.message.move"]
    )
    XCTAssertEqual(move.arguments, ["messageId": "m1", "destination": "inbox"])

    let reply = try valid(
      "outlook.message.reply",
      ["messageId": "m1", "comment": "Merci"],
      available: ["outlook.message.reply"]
    )
    XCTAssertEqual(reply.arguments, ["messageId": "m1", "body": "Merci"])
    XCTAssertEqual(
      failure(
        "outlook.message.reply_all",
        ["messageId": "m1", "body": "Oui", "comment": "Non"],
        available: ["outlook.message.reply_all"]
      ),
      .conflictingAlias(
        tool: "outlook.message.reply_all",
        canonicalArgument: "body",
        alias: "comment"
      )
    )
  }

  func testNoArgumentToolRejectsEveryArgument() {
    XCTAssertEqual(
      failure("calendar.list", ["limit": 5], available: ["calendar.list"]),
      .extraArguments(tool: "calendar.list", arguments: ["limit"])
    )
    guard case .success = AgentToolValidator.validate(
      rawToolID: "calendar.list",
      arguments: [:],
      availableToolIDs: ["calendar.list"]
    ) else {
      return XCTFail("Expected empty argument list to validate")
    }
  }

  func testTriggerCreateValidatesScheduleSpecificArgumentsBeforeApproval() throws {
    let base: AgentJSONArguments = ["title": "Run", "prompt": "Summarize"]
    let validSchedules: [AgentJSONArguments] = [
      ["schedule": "relative", "inMinutes": 15],
      ["schedule": "absolute", "atTime": "09:05"],
      ["schedule": "interval", "intervalSeconds": 3_600],
      ["schedule": "before_next_event"],
      ["schedule": "before_next_event", "beforeMinutes": 15],
    ]
    for schedule in validSchedules {
      _ = try valid(
        "trigger.create",
        base.merging(schedule) { _, new in new },
        available: ["trigger.create"]
      )
    }

    XCTAssertEqual(
      failure(
        "trigger.create",
        base.merging(["schedule": "relative", "atTime": "09:00"]) { _, new in new },
        available: ["trigger.create"]
      ),
      .invalidArgumentCombination(
        tool: "trigger.create",
        reason: "relative requires only integer inMinutes from 1 through 527040."
      )
    )
    XCTAssertEqual(
      failure(
        "trigger.create",
        base.merging(["schedule": "absolute", "atTime": "9:00"]) { _, new in new },
        available: ["trigger.create"]
      ),
      .invalidArgumentCombination(
        tool: "trigger.create",
        reason: "absolute requires only atTime in strict HH:mm format."
      )
    )
    XCTAssertEqual(
      failure(
        "trigger.create",
        base.merging(["schedule": "interval", "intervalSeconds": 60.5]) { _, new in new },
        available: ["trigger.create"]
      ),
      .invalidArgumentCombination(
        tool: "trigger.create",
        reason: "interval requires only integer intervalSeconds from 60 through 2678400."
      )
    )
    XCTAssertEqual(
      failure(
        "trigger.create",
        base.merging([
          "schedule": "before_next_event",
          "beforeMinutes": 15,
          "inMinutes": 10,
        ]) { _, new in new },
        available: ["trigger.create"]
      ),
      .invalidArgumentCombination(
        tool: "trigger.create",
        reason: "before_next_event accepts only optional integer beforeMinutes from 1 through 1440."
      )
    )
  }

  func testTriggerCancelRequiresExactlyOneExactIdentifier() throws {
    let uuid = "11111111-2222-4333-8444-555555555555"
    _ = try valid("trigger.cancel", ["id": .string(uuid)], available: ["trigger.cancel"])
    _ = try valid("trigger.cancel", ["title": "Morning summary"], available: ["trigger.cancel"])

    let invalidSelectors: [AgentJSONArguments] = [
      [:],
      ["id": .string(uuid), "title": "Morning summary"],
      ["title": "   "],
    ]
    for arguments in invalidSelectors {
      XCTAssertEqual(
        failure("trigger.cancel", arguments, available: ["trigger.cancel"]),
        .invalidArgumentCombination(
          tool: "trigger.cancel",
          reason: "provide exactly one non-empty id or title."
        )
      )
    }
    XCTAssertEqual(
      failure("trigger.cancel", ["id": "not-a-uuid"], available: ["trigger.cancel"]),
      .invalidArgumentCombination(tool: "trigger.cancel", reason: "id must be a UUID.")
    )
  }

  func testAlarmScheduleRequiresOneValidatedTimeBeforeApproval() throws {
    let defaultSnooze = try valid(
      "alarm.schedule",
      ["title": "Wake", "inMinutes": 5],
      available: ["alarm.schedule"]
    )
    XCTAssertEqual(defaultSnooze.arguments["snoozeMinutes"], 5)
    let explicitOneShot = try valid(
      "alarm.schedule",
      ["title": "Wake", "inMinutes": 5, "repeats": false],
      available: ["alarm.schedule"]
    )
    XCTAssertEqual(explicitOneShot.arguments["repeats"], false)
    XCTAssertEqual(
      failure(
        "alarm.schedule",
        ["title": "Wake", "inMinutes": 5, "repeats": true],
        available: ["alarm.schedule"]
      ),
      .invalidArgumentCombination(
        tool: "alarm.schedule",
        reason: "repeats=true is unsupported; alarm.schedule creates one-shot alarms only."
      )
    )
    _ = try valid(
      "alarm.schedule",
      ["title": "Wake", "timestamp": "1784635200"],
      available: ["alarm.schedule"]
    )
    XCTAssertEqual(
      failure("alarm.schedule", ["title": "Wake"], available: ["alarm.schedule"]),
      .invalidArgumentCombination(
        tool: "alarm.schedule",
        reason: "provide exactly one of inMinutes or timestamp."
      )
    )
    XCTAssertEqual(
      failure(
        "alarm.schedule",
        ["title": "Wake", "inMinutes": 5, "timestamp": "1784635200"],
        available: ["alarm.schedule"]
      ),
      .invalidArgumentCombination(
        tool: "alarm.schedule",
        reason: "provide exactly one of inMinutes or timestamp."
      )
    )
  }

  func testDuplicateKeyIsStableAcrossArgumentOrderAndAliases() throws {
    let first = try valid(
      "web.search",
      ["query": "Swift"],
      available: ["web.search"]
    ).duplicateKey()
    let second = try valid(
      "search",
      ["q": "Swift"],
      available: ["web.search"]
    ).duplicateKey()

    XCTAssertEqual(first, second)
  }

  private func failure(
    _ tool: String,
    _ arguments: AgentJSONArguments,
    available: Set<AgentToolID>
  ) -> AgentToolValidationError? {
    guard case let .failure(error) = AgentToolValidator.validate(
      rawToolID: tool,
      arguments: arguments,
      availableToolIDs: available
    ) else { return nil }
    return error
  }

  private func valid(
    _ tool: String,
    _ arguments: AgentJSONArguments,
    available: Set<AgentToolID>
  ) throws -> AgentValidatedToolCall {
    switch AgentToolValidator.validate(
      rawToolID: tool,
      arguments: arguments,
      availableToolIDs: available
    ) {
    case let .success(call): return call
    case let .failure(error): throw error
    }
  }
}
