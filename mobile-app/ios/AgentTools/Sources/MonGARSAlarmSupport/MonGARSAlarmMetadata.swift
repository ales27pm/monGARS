import Foundation

#if os(iOS) && canImport(AlarmKit)
import AlarmKit

/// Metadata shared verbatim by AlarmKit scheduling and the Live Activity
/// widget. Keeping it in a tiny extension-safe product guarantees the generic
/// ActivityAttributes identity matches in both processes.
@available(iOS 26.0, *)
public struct MonGARSAlarmMetadata: AlarmMetadata, Codable, Hashable, Sendable {
  public let title: String

  public init(title: String) {
    self.title = title
  }
}
#endif
