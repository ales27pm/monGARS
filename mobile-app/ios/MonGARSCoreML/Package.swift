// swift-tools-version: 5.9

import PackageDescription

let package = Package(
  name: "MonGARSCoreML",
  platforms: [
    .iOS("18.0"),
    .macOS("15.0"),
  ],
  products: [
    .library(name: "MonGARSCoreML", targets: ["MonGARSCoreML"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/huggingface/swift-transformers.git",
      exact: "1.3.3"
    ),
  ],
  targets: [
    .target(
      name: "MonGARSCoreML",
      dependencies: [
        .product(name: "Hub", package: "swift-transformers"),
        .product(name: "Tokenizers", package: "swift-transformers"),
      ]
    ),
    .testTarget(
      name: "MonGARSCoreMLTests",
      dependencies: ["MonGARSCoreML"]
    ),
  ]
)
