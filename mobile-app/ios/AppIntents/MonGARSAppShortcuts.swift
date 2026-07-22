import Foundation

#if canImport(AppIntents)
import AppIntents

@available(iOS 18.0, *)
struct MonGARSAppShortcuts: AppShortcutsProvider {
  @AppShortcutsBuilder
  static var appShortcuts: [AppShortcut] {
    AppShortcut(
      intent: MonGARSAskIntent(),
      phrases: [
        "Ask \(.applicationName)",
        "Demander à \(.applicationName)",
      ],
      shortTitle: "Ask monGARS",
      systemImageName: "bubble.left.and.text.bubble.right"
    )
    AppShortcut(
      intent: MonGARSSearchMemoryIntent(),
      phrases: [
        "Search memory in \(.applicationName)",
        "Chercher dans la mémoire de \(.applicationName)",
      ],
      shortTitle: "Search Memory",
      systemImageName: "magnifyingglass"
    )
    AppShortcut(
      intent: MonGARSAddMemoryIntent(),
      phrases: [
        "Add memory in \(.applicationName)",
        "Ajouter un souvenir dans \(.applicationName)",
      ],
      shortTitle: "Add Memory",
      systemImageName: "brain.head.profile"
    )
    AppShortcut(
      intent: MonGARSRunTriggerIntent(),
      phrases: [
        "Run a trigger in \(.applicationName)",
        "Lancer un déclencheur dans \(.applicationName)",
      ],
      shortTitle: "Run Trigger",
      systemImageName: "bolt"
    )
    AppShortcut(
      intent: MonGARSDiagnosticsIntent(),
      phrases: [
        "Open diagnostics in \(.applicationName)",
        "Ouvrir les diagnostics de \(.applicationName)",
      ],
      shortTitle: "Diagnostics",
      systemImageName: "stethoscope"
    )
  }
}
#endif
