import Foundation
import MonGARSAgentTools
import MonGARSCoreML
import XCTest

final class SafeFilesAndMemoryTests: XCTestCase {
  func testImportedDocumentDirectoryIsCreated() {
    let root = temporaryDirectory().appendingPathComponent("ImportedDocuments", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }

    _ = SafeLocalFileService(rootDirectory: root)

    var isDirectory: ObjCBool = false
    XCTAssertTrue(FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory))
    XCTAssertTrue(isDirectory.boolValue)
  }

  func testPathTraversalAndAbsolutePathsAreRejected() throws {
    let root = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    XCTAssertNil(SafeAgentFilePath.resolve(name: "../secret.txt", under: root))
    XCTAssertNil(SafeAgentFilePath.resolve(name: "/etc/passwd", under: root))
    XCTAssertNil(SafeAgentFilePath.resolve(name: "sub/../../secret.txt", under: root))
    XCTAssertNotNil(SafeAgentFilePath.resolve(name: "notes/today.md", under: root))
  }

  func testSymlinkEscapeIsRejected() throws {
    let container = temporaryDirectory()
    let root = container.appendingPathComponent("root", isDirectory: true)
    let outside = container.appendingPathComponent("outside", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: container) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    let link = root.appendingPathComponent("link", isDirectory: true)
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

    XCTAssertNil(SafeAgentFilePath.resolve(name: "link/private.txt", under: root))
  }

  func testDocumentDiscoveryDoesNotCollapseUnavailableRootToEmpty() throws {
    let container = temporaryDirectory()
    let root = container.appendingPathComponent("not-a-directory")
    defer { try? FileManager.default.removeItem(at: container) }
    try FileManager.default.createDirectory(at: container, withIntermediateDirectories: true)
    try Data("not a directory".utf8).write(to: root)

    let service = SafeLocalFileService(rootDirectory: root)

    guard case .failure(.rootUnavailable) = service.documentsForIndexing() else {
      return XCTFail("An unavailable import root must not look like an empty directory")
    }
    XCTAssertFalse(service.hasReadableDocuments())
  }

  func testMemoryRecallIsStrictlyScoped() async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = AgentLocalKnowledgeStore(stateURL: directory.appendingPathComponent("knowledge.json"))

    let save = await store.saveMemory(
      content: "Prefers dark roast coffee",
      kind: "preference",
      scope: "profile.alice"
    )
    XCTAssertEqual(save.status, .success)

    let alice = await store.recallMemory(
      query: "coffee",
      scope: "profile.alice"
    )
    XCTAssertEqual(alice.status, .success)
    XCTAssertTrue(alice.text.contains("dark roast"))

    let bob = await store.recallMemory(
      query: "coffee",
      scope: "profile.bob"
    )
    XCTAssertEqual(bob.status, .success)
    XCTAssertEqual(bob.text, "No matching memories were found.")
  }

  func testOwnerScopedExecutorsDoNotShareMemory() async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let base = IOSAgentToolExecutor(
      graphService: nil,
      webService: nil,
      importedFilesRoot: directory.appendingPathComponent("imports"),
      protectedStateRoot: directory.appendingPathComponent("state"),
      presenter: nil,
      photoProvider: nil
    )
    let alice = ScopedIOSAgentToolExecutor(base: base, rawOwnerID: "alice@example.com")
    let bob = ScopedIOSAgentToolExecutor(base: base, rawOwnerID: "bob@example.com")
    let save = AgentToolInvocation(
      runID: UUID(),
      stepIndex: 0,
      toolID: "memory.save",
      arguments: ["content": "Likes oolong tea", "kind": "preference"],
      mode: .foreground
    )
    let saveResult = await alice.execute(invocation: save)
    XCTAssertEqual(saveResult.status, .success)

    let recall = AgentToolInvocation(
      runID: UUID(),
      stepIndex: 1,
      toolID: "memory.recall",
      arguments: ["query": "oolong"],
      mode: .foreground
    )
    let bobResult = await bob.execute(invocation: recall)
    XCTAssertEqual(bobResult.status, .success)
    XCTAssertEqual(bobResult.text, "No matching memories were found.")
  }

  func testCorruptKnowledgeStoreFailsClosedWithoutOverwrite() async throws {
    let directory = temporaryDirectory()
    let stateURL = directory.appendingPathComponent("knowledge.json")
    let corruptData = Data("{not-json".utf8)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try corruptData.write(to: stateURL)
    let store = AgentLocalKnowledgeStore(stateURL: stateURL)

    let recall = await store.recallMemory(query: "coffee", scope: "profile.alice")
    let save = await store.saveMemory(
      content: "Prefers dark roast coffee",
      kind: "preference",
      scope: "profile.alice"
    )

    XCTAssertEqual(recall.status, .failed)
    XCTAssertEqual(recall.errorCode, "knowledge_store_corrupt")
    XCTAssertEqual(save.status, .failed)
    XCTAssertEqual(save.errorCode, "knowledge_store_corrupt")
    XCTAssertEqual(try Data(contentsOf: stateURL), corruptData)
  }

#if os(iOS)
  func testImportedDocumentDirectoryUsesCompleteFileProtection() throws {
    let root = temporaryDirectory().appendingPathComponent("ImportedDocuments", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root.deletingLastPathComponent()) }

    _ = SafeLocalFileService(rootDirectory: root)

    let attributes = try FileManager.default.attributesOfItem(atPath: root.path)
    XCTAssertEqual(attributes[.protectionKey] as? FileProtectionType, .complete)
  }

  func testMemoryStateUsesCompleteFileProtection() async throws {
    let directory = temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let stateURL = directory.appendingPathComponent("knowledge.json")
    let store = AgentLocalKnowledgeStore(stateURL: stateURL)

    let result = await store.saveMemory(
      content: "Protected foreground-only memory",
      kind: "note",
      scope: "profile.protection"
    )

    XCTAssertEqual(result.status, .success)
    let attributes = try FileManager.default.attributesOfItem(atPath: stateURL.path)
    XCTAssertEqual(attributes[.protectionKey] as? FileProtectionType, .complete)
  }
#endif

  private func temporaryDirectory() -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("MonGARSAgentToolsTests-\(UUID().uuidString)", isDirectory: true)
  }
}
