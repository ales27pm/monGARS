// swift-tools-version: 5.9

import PackageDescription

let package = Package(
  name: "MonGARSAgentTools",
  platforms: [
    .iOS("18.0"),
    .macOS("15.0"),
  ],
  products: [
    .library(name: "MonGARSAlarmSupport", targets: ["MonGARSAlarmSupport"]),
    .library(name: "MonGARSAgentTools", targets: ["MonGARSAgentTools"]),
  ],
  dependencies: [
    .package(path: "../MonGARSCoreML"),
  ],
  targets: [
    .target(name: "MonGARSAlarmSupport"),
    .target(
      name: "MonGARSAgentTools",
      dependencies: ["MonGARSCoreML", "MonGARSAlarmSupport"]
    ),
    .testTarget(
      name: "MonGARSAgentToolsTests",
      dependencies: ["MonGARSAgentTools", "MonGARSCoreML"]
    ),
  ]
)
