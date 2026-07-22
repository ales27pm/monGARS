import ActivityKit
import AlarmKit
import MonGARSAlarmSupport
import SwiftUI
import WidgetKit

@main
struct MonGARSAlarmWidgetBundle: WidgetBundle {
  @WidgetBundleBuilder
  var body: some Widget {
    MonGARSAlarmAvailabilityWidget()
    if #available(iOS 26.0, *) {
      MonGARSAlarmLiveActivity()
    }
  }
}

private struct MonGARSAlarmStatusEntry: TimelineEntry {
  let date: Date
}

private struct MonGARSAlarmStatusProvider: TimelineProvider {
  func placeholder(in context: Context) -> MonGARSAlarmStatusEntry {
    .init(date: Date())
  }

  func getSnapshot(
    in context: Context,
    completion: @escaping (MonGARSAlarmStatusEntry) -> Void
  ) {
    completion(.init(date: Date()))
  }

  func getTimeline(
    in context: Context,
    completion: @escaping (Timeline<MonGARSAlarmStatusEntry>) -> Void
  ) {
    completion(.init(entries: [.init(date: Date())], policy: .never))
  }
}

private struct MonGARSAlarmAvailabilityWidget: Widget {
  let kind = "MonGARSAlarmAvailability"

  var body: some WidgetConfiguration {
    StaticConfiguration(kind: kind, provider: MonGARSAlarmStatusProvider()) { _ in
      VStack(spacing: 8) {
        Image(systemName: "alarm.fill")
          .font(.title)
          .foregroundStyle(.orange)
        Text("MonGARS Alarms")
          .font(.headline)
        Text("Alarm Live Activities require iOS 26.")
          .font(.caption)
          .foregroundStyle(.secondary)
          .multilineTextAlignment(.center)
      }
      .padding()
      .containerBackground(.fill.tertiary, for: .widget)
    }
    .configurationDisplayName("MonGARS Alarms")
    .description("Shows AlarmKit availability and provides alarm Live Activities on iOS 26.")
    .supportedFamilies([.systemSmall])
  }
}

@available(iOS 26.0, *)
struct MonGARSAlarmLiveActivity: Widget {
  var body: some WidgetConfiguration {
    ActivityConfiguration(for: AlarmAttributes<MonGARSAlarmMetadata>.self) { context in
      VStack(alignment: .leading, spacing: 12) {
        HStack(spacing: 8) {
          MonGARSAlarmModeIcon(state: context.state)
          Text(context.attributes.metadata?.title ?? "MonGARS Alarm")
            .font(.headline)
            .lineLimit(2)
        }
        MonGARSAlarmStateView(state: context.state)
          .font(.system(.title, design: .rounded, weight: .semibold))
      }
      .foregroundStyle(.white)
      .padding()
      .activityBackgroundTint(Color.black.opacity(0.88))
      .activitySystemActionForegroundColor(.orange)
    } dynamicIsland: { context in
      DynamicIsland {
        DynamicIslandExpandedRegion(.leading) {
          MonGARSAlarmModeIcon(state: context.state)
            .font(.title2)
            .foregroundStyle(.orange)
        }
        DynamicIslandExpandedRegion(.trailing) {
          MonGARSAlarmStateView(state: context.state)
            .font(.headline.monospacedDigit())
            .foregroundStyle(.orange)
        }
        DynamicIslandExpandedRegion(.bottom) {
          HStack {
            Text(context.attributes.metadata?.title ?? "MonGARS Alarm")
              .lineLimit(1)
            Spacer(minLength: 8)
            MonGARSAlarmModeLabel(state: context.state)
              .foregroundStyle(.secondary)
          }
        }
      } compactLeading: {
        MonGARSAlarmModeIcon(state: context.state)
          .foregroundStyle(.orange)
      } compactTrailing: {
        MonGARSAlarmStateView(state: context.state)
          .font(.caption2.monospacedDigit())
          .foregroundStyle(.orange)
      } minimal: {
        MonGARSAlarmModeIcon(state: context.state)
          .foregroundStyle(.orange)
      }
      .keylineTint(.orange)
    }
  }
}

@available(iOS 26.0, *)
private struct MonGARSAlarmStateView: View {
  let state: AlarmPresentationState

  @ViewBuilder
  var body: some View {
    switch state.mode {
    case .countdown(let countdown):
      let start = Date()
      Text(
        timerInterval: start...max(start, countdown.fireDate),
        countsDown: true
      )
      .monospacedDigit()
      .lineLimit(1)
    case .paused:
      Text("Paused")
    case .alert:
      Text("Alarm")
    }
  }
}

@available(iOS 26.0, *)
private struct MonGARSAlarmModeIcon: View {
  let state: AlarmPresentationState

  var body: some View {
    Image(systemName: symbolName)
  }

  private var symbolName: String {
    switch state.mode {
    case .countdown: return "timer"
    case .paused: return "pause.circle.fill"
    case .alert: return "alarm.fill"
    }
  }
}

@available(iOS 26.0, *)
private struct MonGARSAlarmModeLabel: View {
  let state: AlarmPresentationState

  var body: some View {
    Text(label)
      .font(.caption)
  }

  private var label: String {
    switch state.mode {
    case .countdown: return "Counting down"
    case .paused: return "Paused"
    case .alert: return "Alerting"
    }
  }
}
