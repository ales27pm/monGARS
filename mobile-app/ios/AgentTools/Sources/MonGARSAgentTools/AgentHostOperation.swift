import Foundation
import MonGARSCoreML

/// Host operations are deliberately one-to-one with the portable catalog.
/// This table is the only dispatch authority; unknown or aliased IDs never
/// fall through to a similarly named platform action.
public enum AgentHostOperation: String, CaseIterable, Sendable, Equatable, Hashable {
  case calendarCreate
  case calendarList
  case remindersCreate
  case remindersList
  case contactsSearch
  case messagesDraft
  case mailDraft
  case outlookStatus
  case outlookFoldersList
  case outlookMessagesList
  case outlookMessagesSearch
  case outlookMessageRead
  case outlookAttachmentsList
  case outlookDraftCreate
  case outlookMailSend
  case outlookMessageMarkRead
  case outlookMessageMarkUnread
  case outlookMessageMove
  case outlookMessageArchive
  case outlookMessageDelete
  case outlookMessageReply
  case outlookMessageReplyAll
  case outlookMessageForward
  case phoneCall
  case locationCurrent
  case weather
  case mapsDirections
  case mapsSearch
  case photosSearch
  case cameraCapture
  case healthSummary
  case motionActivity
  case webSearch
  case webFetch
  case filesRead
  case memorySave
  case memoryRecall
  case ragSearch
  case ragIndexFiles
  case ragIndexPhotos
  case triggerCreate
  case triggerList
  case triggerCancel
  case alarmAuthorizationStatus
  case alarmRequestAuthorization
  case alarmSchedule
  case alarmCountdown
  case alarmList
  case alarmPause
  case alarmResume
  case alarmStop
  case alarmSnooze
  case alarmCancel

  public static let canonicalDispatchTable: [AgentToolID: AgentHostOperation] = [
    "calendar.create": .calendarCreate,
    "calendar.list": .calendarList,
    "reminders.create": .remindersCreate,
    "reminders.list": .remindersList,
    "contacts.search": .contactsSearch,
    "messages.draft": .messagesDraft,
    "mail.draft": .mailDraft,
    "outlook.status": .outlookStatus,
    "outlook.folders.list": .outlookFoldersList,
    "outlook.messages.list": .outlookMessagesList,
    "outlook.messages.search": .outlookMessagesSearch,
    "outlook.message.read": .outlookMessageRead,
    "outlook.attachments.list": .outlookAttachmentsList,
    "outlook.draft.create": .outlookDraftCreate,
    "outlook.mail.send": .outlookMailSend,
    "outlook.message.mark_read": .outlookMessageMarkRead,
    "outlook.message.mark_unread": .outlookMessageMarkUnread,
    "outlook.message.move": .outlookMessageMove,
    "outlook.message.archive": .outlookMessageArchive,
    "outlook.message.delete": .outlookMessageDelete,
    "outlook.message.reply": .outlookMessageReply,
    "outlook.message.reply_all": .outlookMessageReplyAll,
    "outlook.message.forward": .outlookMessageForward,
    "phone.call": .phoneCall,
    "location.current": .locationCurrent,
    "weather": .weather,
    "maps.directions": .mapsDirections,
    "maps.search": .mapsSearch,
    "photos.search": .photosSearch,
    "camera.capture": .cameraCapture,
    "health.summary": .healthSummary,
    "motion.activity": .motionActivity,
    "web.search": .webSearch,
    "web.fetch": .webFetch,
    "files.read": .filesRead,
    "memory.save": .memorySave,
    "memory.recall": .memoryRecall,
    "rag.search": .ragSearch,
    "rag.index_files": .ragIndexFiles,
    "rag.index_photos": .ragIndexPhotos,
    "trigger.create": .triggerCreate,
    "trigger.list": .triggerList,
    "trigger.cancel": .triggerCancel,
    "alarm.authorization_status": .alarmAuthorizationStatus,
    "alarm.request_authorization": .alarmRequestAuthorization,
    "alarm.schedule": .alarmSchedule,
    "alarm.countdown": .alarmCountdown,
    "alarm.list": .alarmList,
    "alarm.pause": .alarmPause,
    "alarm.resume": .alarmResume,
    "alarm.stop": .alarmStop,
    "alarm.snooze": .alarmSnooze,
    "alarm.cancel": .alarmCancel,
  ]

  public static let outlookOperations: Set<AgentHostOperation> = [
    .outlookStatus, .outlookFoldersList, .outlookMessagesList, .outlookMessagesSearch,
    .outlookMessageRead, .outlookAttachmentsList, .outlookDraftCreate, .outlookMailSend,
    .outlookMessageMarkRead, .outlookMessageMarkUnread, .outlookMessageMove,
    .outlookMessageArchive, .outlookMessageDelete, .outlookMessageReply,
    .outlookMessageReplyAll, .outlookMessageForward,
  ]

  public static let alarmOperations: Set<AgentHostOperation> = [
    .alarmAuthorizationStatus, .alarmRequestAuthorization, .alarmSchedule, .alarmCountdown,
    .alarmList, .alarmPause, .alarmResume, .alarmStop, .alarmSnooze, .alarmCancel,
  ]

  public static let foregroundUIOperations: Set<AgentHostOperation> = [
    .messagesDraft, .mailDraft, .phoneCall, .mapsDirections, .cameraCapture,
  ]
}
