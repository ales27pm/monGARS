import XCTest
@testable import MonGARSCoreML

final class AgentRoutingAndParserTests: XCTestCase {
  func testAll22IntentToolScopesMatchCanonicalMatrix() {
    let expected: [AgentIntent: Set<AgentToolID>] = [
      .weather: ["weather", "location.current"],
      .webSearch: ["web.search", "web.fetch"],
      .emailDraft: ["mail.draft", "contacts.search"],
      .messageDraft: ["messages.draft", "contacts.search"],
      .phoneCall: ["phone.call", "contacts.search"],
      .contactSearch: ["contacts.search"],
      .calendar: ["calendar.create", "calendar.list"],
      .reminder: ["reminders.create", "reminders.list"],
      .maps: ["maps.search", "maps.directions", "location.current"],
      .photos: ["photos.search"],
      .camera: ["camera.capture"],
      .health: ["health.summary"],
      .motion: ["motion.activity"],
      .files: ["files.read"],
      .memory: ["memory.save", "memory.recall"],
      .rag: ["rag.search", "rag.index_files", "rag.index_photos", "files.read", "photos.search"],
      .trigger: ["trigger.create", "trigger.list", "trigger.cancel"],
      .alarm: [
        "alarm.authorization_status", "alarm.request_authorization", "alarm.schedule",
        "alarm.countdown", "alarm.list", "alarm.pause", "alarm.resume", "alarm.stop",
        "alarm.snooze", "alarm.cancel",
      ],
      .outlook: [
        "outlook.status", "outlook.folders.list", "outlook.messages.list",
        "outlook.messages.search", "outlook.message.read", "outlook.attachments.list",
        "outlook.draft.create", "outlook.mail.send", "outlook.message.mark_read",
        "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive",
        "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all",
        "outlook.message.forward", "contacts.search",
      ],
      .note: ["memory.save", "memory.recall"],
      .chat: [],
      .unknown: [],
    ]

    XCTAssertEqual(AgentIntent.allCases.count, 22)
    for intent in AgentIntent.allCases {
      XCTAssertEqual(
        AgentIntentRouter.allowedToolIDs(for: intent),
        expected[intent] ?? [],
        "Unexpected tool scope for \(intent.rawValue)"
      )
    }
  }

  func testRouterRequiresClarificationForUnderspecifiedSensitiveActions() {
    XCTAssertEqual(AgentIntentRouter.route("draft email").clarification,
                   "Who should I send it to, and what should it say?")
    XCTAssertEqual(AgentIntentRouter.route("call").clarification, "Who should I call?")
    XCTAssertEqual(AgentIntentRouter.route("set alarm").clarification,
                   "What time should I use for the alarm?")
    XCTAssertEqual(AgentIntentRouter.route("read file").clarification,
                   "Which file should I read?")
    XCTAssertEqual(AgentIntentRouter.route("memory").clarification,
                   "What should I save or recall?")
    XCTAssertEqual(AgentIntentRouter.route("meeting").clarification,
                   "Do you mean a calendar event or a nearby meeting location?")
  }

  func testRouterScopesConcreteRequests() {
    XCTAssertEqual(AgentIntentRouter.route("weather in Toronto").intent, .weather)
    XCTAssertEqual(AgentIntentRouter.route("search Outlook for invoices").intent, .outlook)
    XCTAssertEqual(AgentIntentRouter.route("find coffee near me").intent, .maps)
    XCTAssertEqual(AgentIntentRouter.route("remember that I prefer tea").intent, .memory)
    XCTAssertEqual(AgentIntentRouter.route("take a photo").intent, .camera)

    let location = AgentIntentRouter.route("where am I")
    XCTAssertEqual(location.intent, .maps)
    XCTAssertEqual(location.allowedToolIDs, ["location.current"])
  }

  func testParserAcceptsStrictActionAndFinalForms() {
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"action":{"tool":"web.search","args":{"query":"Swift"}}}"#),
      .success(.action(
        thought: nil,
        .init(tool: "web.search", arguments: ["query": "Swift"])
      ))
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"thought":"private","final":"Done"}"#),
      .success(.final(thought: "private", "Done"))
    )
  }

  func testParserRejectsNoiseMutualExclusionAndExtraKeys() {
    XCTAssertEqual(
      AgentTurnParser.parse("```json\n{\"final\":\"Done\"}\n```"),
      .failure(.invalidJSON)
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"action":{"tool":"web.search","args":{}},"final":"Done"}"#),
      .failure(.actionAndFinalAreMutuallyExclusive)
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"final":"Done","debug":true}"#),
      .failure(.extraTopLevelKeys(["debug"]))
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"action":{"tool":"web.search","args":{},"confirm":false}}"#),
      .failure(.extraActionKeys(["confirm"]))
    )
  }

  func testParserRejectsMissingAndWronglyTypedFields() {
    XCTAssertEqual(AgentTurnParser.parse("{}"), .failure(.missingActionOrFinal))
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"action":{"args":{}}}"#),
      .failure(.missingTool)
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"action":{"tool":"web.search","args":[]}}"#),
      .failure(.argumentsMustBeObject)
    )
    XCTAssertEqual(
      AgentTurnParser.parse(#"{"final":"   "}"#),
      .failure(.finalMustBeNonEmptyString)
    )
  }

  func testParserRejectsDuplicateKeysAtEveryObjectDepth() {
    for raw in [
      #"{"final":"one","final":"two"}"#,
      #"{"action":{"tool":"weather","tool":"web.search","args":{}}}"#,
      #"{"action":{"tool":"weather","args":{"city":"Montreal","city":"Quebec"}}}"#,
    ] {
      guard case .failure(.duplicateObjectKey) = AgentTurnParser.parse(raw) else {
        return XCTFail("Expected duplicate-key rejection for \(raw)")
      }
    }
  }

  func testParserRejectsOversizedModelOutputBeforeDecoding() {
    let oversized = String(repeating: "x", count: AgentTurnParser.maximumModelOutputBytes + 1)

    XCTAssertEqual(AgentTurnParser.parse(oversized), .failure(.outputTooLarge))
  }

  func testOutputSanitizerRemovesControlTokensAndBoundsText() {
    let sanitized = AgentOutputSanitizer.sanitizeFinal(
      "\u{0}<|assistant|> Hello\n\n\nworld <|eot_id|>",
      maximumCharacters: 11
    )

    XCTAssertEqual(sanitized, "Hello\n\nworl")
  }

  func testJSONOutputSanitizerRecursesThroughPayloads() {
    let payload: AgentJSONValue = [
      "nested": ["value": "<|assistant|>abcdef"],
      "array": ["\u{0}hello"],
    ]
    let sanitized = AgentOutputSanitizer.sanitizeJSON(
      payload,
      maximumStringCharacters: 4
    )

    XCTAssertEqual(sanitized, [
      "nested": ["value": "abcd"],
      "array": ["hell"],
    ])
  }
}
