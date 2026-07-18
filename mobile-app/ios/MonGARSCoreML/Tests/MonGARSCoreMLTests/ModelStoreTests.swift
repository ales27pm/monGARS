import Foundation
import XCTest
@testable import MonGARSCoreML

final class ModelStoreTests: XCTestCase {
  func testVerificationSessionOnlyReusesExplicitCryptographicSuccess() {
    var session = ModelStore.VerificationSession()

    XCTAssertTrue(session.requiresCryptographicVerification)

    session.recordCryptographicVerification()
    XCTAssertFalse(session.requiresCryptographicVerification)

    session.invalidate()
    XCTAssertTrue(session.requiresCryptographicVerification)
  }

  #if canImport(CryptoKit)
  func testSHA256DetectsSameSizeSameModificationDateRewrite() async throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
      at: directory,
      withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: directory) }

    let fileURL = directory.appendingPathComponent("artifact.bin")
    try Data([0, 1, 2, 3]).write(to: fileURL)
    let fixedModificationDate = Date(timeIntervalSince1970: 1_700_000_000)
    try FileManager.default.setAttributes(
      [.modificationDate: fixedModificationDate],
      ofItemAtPath: fileURL.path
    )
    let originalHash = try await ModelStore.sha256(fileURL)

    let handle = try FileHandle(forWritingTo: fileURL)
    try handle.write(contentsOf: Data([4, 5, 6, 7]))
    try handle.synchronize()
    try handle.close()
    try FileManager.default.setAttributes(
      [.modificationDate: fixedModificationDate],
      ofItemAtPath: fileURL.path
    )
    let rewrittenHash = try await ModelStore.sha256(fileURL)

    let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
    XCTAssertEqual((attributes[.size] as? NSNumber)?.int64Value, 4)
    XCTAssertEqual(attributes[.modificationDate] as? Date, fixedModificationDate)
    XCTAssertNotEqual(originalHash, rewrittenHash)
  }
  #endif
}
