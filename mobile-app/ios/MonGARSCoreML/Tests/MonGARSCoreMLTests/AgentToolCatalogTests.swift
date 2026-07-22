import XCTest
@testable import MonGARSCoreML

final class AgentToolCatalogTests: XCTestCase {
  func testCatalogContainsExactlyThe53CanonicalTools() {
    let expected: Set<String> = [
      "calendar.create", "calendar.list", "reminders.create", "reminders.list",
      "contacts.search", "messages.draft", "mail.draft", "outlook.status",
      "outlook.folders.list", "outlook.messages.list", "outlook.messages.search",
      "outlook.message.read", "outlook.attachments.list", "outlook.draft.create",
      "outlook.mail.send", "outlook.message.mark_read", "outlook.message.mark_unread",
      "outlook.message.move", "outlook.message.archive", "outlook.message.delete",
      "outlook.message.reply", "outlook.message.reply_all", "outlook.message.forward",
      "phone.call", "location.current", "weather", "maps.directions", "maps.search",
      "photos.search", "camera.capture", "health.summary", "motion.activity",
      "web.search", "web.fetch", "files.read", "memory.save", "memory.recall",
      "rag.search", "rag.index_files", "rag.index_photos", "trigger.create",
      "trigger.list", "trigger.cancel", "alarm.authorization_status",
      "alarm.request_authorization", "alarm.schedule", "alarm.countdown", "alarm.list",
      "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel",
    ]
    let actual = Set(AgentToolCatalog.all.map(\.id.rawValue))

    XCTAssertEqual(AgentToolCatalog.all.count, 53)
    XCTAssertEqual(actual.count, 53)
    XCTAssertEqual(actual, expected)
  }

  func testEveryCanonicalSchemaMatchesTheContract() {
    let expected: [String: String] = [
      "calendar.create": "title:string!,startsInMinutes:number!",
      "calendar.list": "",
      "reminders.create": "title:string!",
      "reminders.list": "",
      "contacts.search": "query:string!",
      "messages.draft": "to:string!,body:string!,recipient:string?,number:string?,message:string?,text:string?",
      "mail.draft": "to:string!,subject:string?,body:string!,recipient:string?,email:string?,message:string?,text:string?,title:string?",
      "outlook.status": "",
      "outlook.folders.list": "includeHidden:bool?",
      "outlook.messages.list": "folder:string?,folderId:string?,limit:number?,unreadOnly:bool?",
      "outlook.messages.search": "query:string!,folder:string?,folderId:string?,limit:number?",
      "outlook.message.read": "messageId:string!,id:string?",
      "outlook.attachments.list": "messageId:string!,id:string?",
      "outlook.draft.create": "to:string!,subject:string!,body:string!",
      "outlook.mail.send": "to:string!,subject:string!,body:string!",
      "outlook.message.mark_read": "messageId:string!,id:string?",
      "outlook.message.mark_unread": "messageId:string!,id:string?",
      "outlook.message.move": "messageId:string!,destination:string!,id:string?,destinationId:string?",
      "outlook.message.archive": "messageId:string!,id:string?",
      "outlook.message.delete": "messageId:string!,id:string?",
      "outlook.message.reply": "messageId:string!,body:string!,id:string?,comment:string?",
      "outlook.message.reply_all": "messageId:string!,body:string!,id:string?,comment:string?",
      "outlook.message.forward": "messageId:string!,to:string!,id:string?,body:string?,comment:string?",
      "phone.call": "number:string!",
      "location.current": "",
      "weather": "location:string?,city:string?",
      "maps.directions": "destination:string!",
      "maps.search": "query:string!",
      "photos.search": "query:string!",
      "camera.capture": "",
      "health.summary": "",
      "motion.activity": "",
      "web.search": "query:string!",
      "web.fetch": "url:string!",
      "files.read": "name:string!",
      "memory.save": "content:string!,kind:string!",
      "memory.recall": "query:string!",
      "rag.search": "query:string!,limit:number?,sourceScope:enum?{all|documents|notes|photos}",
      "rag.index_files": "",
      "rag.index_photos": "months:number!",
      "trigger.create": "title:string!,prompt:string!,schedule:enum!{absolute|before_next_event|interval|relative},inMinutes:number?,atTime:string?,intervalSeconds:number?,beforeMinutes:number?",
      "trigger.list": "",
      "trigger.cancel": "id:string?,title:string?",
      "alarm.authorization_status": "",
      "alarm.request_authorization": "",
      "alarm.schedule": "title:string!,inMinutes:number?,timestamp:string?,repeats:bool?,snoozeMinutes:number?",
      "alarm.countdown": "title:string!,durationSeconds:number!",
      "alarm.list": "",
      "alarm.pause": "id:string!",
      "alarm.resume": "id:string!",
      "alarm.stop": "id:string!",
      "alarm.snooze": "id:string!",
      "alarm.cancel": "id:string!",
    ]
    let actual = Dictionary(uniqueKeysWithValues: AgentToolCatalog.all.map {
      ($0.id.rawValue, Self.signature($0))
    })

    XCTAssertEqual(actual, expected)
  }

  func testApprovalSetMatchesCanonicalBoundary() {
    let expected: Set<String> = [
      "calendar.create", "reminders.create", "messages.draft", "mail.draft",
      "outlook.draft.create", "outlook.mail.send", "outlook.message.mark_read",
      "outlook.message.mark_unread", "outlook.message.move", "outlook.message.archive",
      "outlook.message.delete", "outlook.message.reply", "outlook.message.reply_all",
      "outlook.message.forward", "phone.call", "camera.capture", "trigger.create",
      "trigger.cancel", "alarm.request_authorization", "alarm.schedule", "alarm.countdown",
      "alarm.pause", "alarm.resume", "alarm.stop", "alarm.snooze", "alarm.cancel",
    ]
    let actual = Set(AgentToolCatalog.all.filter(\.requiresApproval).map(\.id.rawValue))

    XCTAssertEqual(actual, expected)
    XCTAssertEqual(actual.count, 26)
    XCTAssertEqual(AgentToolCatalog.all.filter { !$0.requiresApproval }.count, 27)
  }

  func testPermissionMetadataIncludesDerivedNotificationAndAlarmDomains() {
    XCTAssertEqual(AgentToolCatalog.definition(for: "calendar.list")?.permission, .calendar)
    XCTAssertEqual(AgentToolCatalog.definition(for: "trigger.create")?.permission, .notifications)
    XCTAssertEqual(AgentToolCatalog.definition(for: "trigger.list")?.permission, .notifications)
    XCTAssertEqual(AgentToolCatalog.definition(for: "alarm.schedule")?.permission, .alarms)
    XCTAssertEqual(AgentToolCatalog.definition(for: "web.search")?.permission, nil)
  }

  func testBackgroundMetadataIsConservativeForWritesAndNetwork() {
    XCTAssertEqual(AgentToolCatalog.definition(for: "memory.recall")?.supportsBackgroundExecution, true)
    XCTAssertEqual(AgentToolCatalog.definition(for: "memory.save")?.supportsBackgroundExecution, false)
    XCTAssertEqual(AgentToolCatalog.definition(for: "rag.index_files")?.supportsBackgroundExecution, false)
    XCTAssertEqual(AgentToolCatalog.definition(for: "web.search")?.supportsBackgroundExecution, false)
  }

  private static func signature(_ definition: AgentToolDefinition) -> String {
    definition.arguments.map { argument in
      let requirement = argument.required ? "!" : "?"
      let allowed = argument.allowedValues.map {
        "{\($0.sorted().joined(separator: "|"))}"
      } ?? ""
      return "\(argument.name):\(argument.type.rawValue)\(requirement)\(allowed)"
    }.joined(separator: ",")
  }
}
