import Foundation
import Hub

#if canImport(CryptoKit)
import CryptoKit
#endif

final class ModelStore {
  private struct FileSignature: Equatable {
    let bytes: Int64
    let modificationDate: Date
  }

  private struct VerificationMarker: Codable {
    let modelID: String
    let revision: String
    let manifestFingerprint: String
    let verifiedAt: Date
  }

  private let fileManager: FileManager
  private let rootDirectory: URL
  private let hub: HubApi
  private var verifiedFileSignatures: [String: FileSignature]?

  private static let manifestFingerprint = MonGARSModelManifest.expectedFiles
    .map { "\($0.path)|\($0.bytes)|\($0.sha256)" }
    .sorted()
    .joined(separator: "\n")

  init(
    fileManager: FileManager = .default,
    rootDirectory: URL? = nil
  ) {
    self.fileManager = fileManager
    let base = rootDirectory
      ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
      ?? fileManager.temporaryDirectory
    self.rootDirectory = base
      .appendingPathComponent("monGARS", isDirectory: true)
      .appendingPathComponent("Models", isDirectory: true)
    hub = HubApi(
      downloadBase: self.rootDirectory,
      cache: nil,
      useBackgroundSession: true
    )
  }

  var repositoryDirectory: URL {
    hub.localRepoLocation(Hub.Repo(id: MonGARSModelManifest.modelID))
  }

  var modelDirectory: URL {
    repositoryDirectory.appendingPathComponent(
      MonGARSModelManifest.compiledDirectory,
      isDirectory: true
    )
  }

  var tokenizerDirectory: URL { repositoryDirectory }

  private var markerURL: URL {
    repositoryDirectory.appendingPathComponent("mongars-verification.json")
  }

  func isInstalled() -> Bool {
    guard
      fileManager.fileExists(atPath: modelDirectory.path),
      fileManager.fileExists(
        atPath: repositoryDirectory.appendingPathComponent("tokenizer.json").path
      ),
      let data = try? Data(contentsOf: markerURL),
      let marker = try? JSONDecoder().decode(VerificationMarker.self, from: data)
    else {
      return false
    }
    guard
      marker.modelID == MonGARSModelManifest.modelID,
      marker.revision == MonGARSModelManifest.revision,
      marker.manifestFingerprint == Self.manifestFingerprint
    else {
      return false
    }

    return MonGARSModelManifest.expectedFiles.allSatisfy { expected in
      let url = repositoryDirectory.appendingPathComponent(expected.path)
      guard
        let attributes = try? fileManager.attributesOfItem(atPath: url.path),
        let size = attributes[.size] as? NSNumber
      else {
        return false
      }
      return size.int64Value == expected.bytes
    }
  }

  func ensureVerified(
    progress: @escaping @Sendable (ModelProgress) -> Void = { _ in }
  ) async throws {
    guard isInstalled() else { throw InferenceError.modelNotInstalled }
    try applyBackupExclusion(to: repositoryDirectory)
    let currentSignatures = try currentFileSignatures()
    if
      let verifiedFileSignatures,
      verifiedFileSignatures == currentSignatures
    {
      return
    }

    do {
      try await verify(snapshot: repositoryDirectory, progress: progress)
      try Task.checkCancellation()
      verifiedFileSignatures = try currentFileSignatures()
    } catch {
      if
        let inferenceError = error as? InferenceError,
        case let .integrityFailure(path) = inferenceError
      {
        try? purgeArtifact(at: path)
      }
      throw error
    }
  }

  func prepare(
    progress: @escaping @Sendable (ModelProgress) -> Void
  ) async throws -> URL {
    try createRootDirectory()

    if isInstalled() {
      do {
        try await ensureVerified(progress: progress)
        return repositoryDirectory
      } catch let error as InferenceError {
        guard case .integrityFailure(_) = error else { throw error }
      }
    }

    let available = try availableDiskCapacity()
    guard available >= MonGARSModelManifest.requiredFreeDiskBytes else {
      throw InferenceError.insufficientDisk(
        required: MonGARSModelManifest.requiredFreeDiskBytes,
        available: available
      )
    }

    for attempt in 0..<2 {
      progress(
        ModelProgress(
          phase: .downloading,
          fractionCompleted: 0,
          detail: attempt == 0
            ? "Telechargement du modele Hugging Face"
            : "Reparation du fichier corrompu"
        )
      )

      let snapshot = try await hub.snapshot(
        from: Hub.Repo(id: MonGARSModelManifest.modelID),
        revision: MonGARSModelManifest.revision,
        matching: MonGARSModelManifest.files
      ) { downloadProgress, speed in
        progress(
          ModelProgress(
            phase: .downloading,
            fractionCompleted: downloadProgress.fractionCompleted,
            bytesPerSecond: speed,
            detail: "Telechargement du modele Hugging Face"
          )
        )
      }

      try Task.checkCancellation()
      do {
        try await verify(snapshot: snapshot, progress: progress)
        try Task.checkCancellation()
        try applyBackupExclusion(to: snapshot)
        try writeVerificationMarker()
        verifiedFileSignatures = try currentFileSignatures()
        return snapshot
      } catch let error as InferenceError {
        guard case let .integrityFailure(path) = error else { throw error }
        try purgeArtifact(at: path)
        if attempt == 1 { throw error }
      }
    }

    throw InferenceError.integrityFailure("manifest")
  }

  func deleteModel() throws {
    verifiedFileSignatures = nil
    if fileManager.fileExists(atPath: repositoryDirectory.path) {
      try fileManager.removeItem(at: repositoryDirectory)
    }
  }

  private func createRootDirectory() throws {
    try fileManager.createDirectory(
      at: rootDirectory,
      withIntermediateDirectories: true
    )
    var excluded = rootDirectory
    var values = URLResourceValues()
    values.isExcludedFromBackup = true
    try excluded.setResourceValues(values)
  }

  private func applyBackupExclusion(to snapshot: URL) throws {
    var values = URLResourceValues()
    values.isExcludedFromBackup = true

    var excludedSnapshot = snapshot
    try excludedSnapshot.setResourceValues(values)

    for expected in MonGARSModelManifest.expectedFiles {
      var excludedArtifact = snapshot.appendingPathComponent(expected.path)
      try excludedArtifact.setResourceValues(values)
    }
  }

  private func availableDiskCapacity() throws -> Int64 {
    #if canImport(Darwin)
    let values = try rootDirectory.resourceValues(forKeys: [
      .volumeAvailableCapacityForImportantUsageKey,
      .volumeAvailableCapacityKey,
    ])
    if let capacity = values.volumeAvailableCapacityForImportantUsage {
      return Int64(capacity)
    }
    return Int64(values.volumeAvailableCapacity ?? 0)
    #else
    let attributes = try fileManager.attributesOfFileSystem(
      forPath: rootDirectory.path
    )
    return (attributes[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    #endif
  }

  private func verify(
    snapshot: URL,
    progress: @escaping @Sendable (ModelProgress) -> Void
  ) async throws {
    for (index, expected) in MonGARSModelManifest.expectedFiles.enumerated() {
      try Task.checkCancellation()
      let fileURL = snapshot.appendingPathComponent(expected.path)
      guard
        let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
        let size = (attributes[.size] as? NSNumber)?.int64Value
      else {
        throw InferenceError.integrityFailure(expected.path)
      }
      guard size == expected.bytes else {
        throw InferenceError.integrityFailure(expected.path)
      }

      let checksum = try await sha256(fileURL)
      guard checksum == expected.sha256 else {
        throw InferenceError.integrityFailure(expected.path)
      }

      progress(
        ModelProgress(
          phase: .verifying,
          fractionCompleted: Double(index + 1)
            / Double(MonGARSModelManifest.expectedFiles.count),
          detail: "Verification cryptographique"
        )
      )
    }
  }

  private func writeVerificationMarker() throws {
    let marker = VerificationMarker(
      modelID: MonGARSModelManifest.modelID,
      revision: MonGARSModelManifest.revision,
      manifestFingerprint: Self.manifestFingerprint,
      verifiedAt: Date()
    )
    let data = try JSONEncoder().encode(marker)
    try data.write(to: markerURL, options: .atomic)
  }

  private func currentFileSignatures() throws -> [String: FileSignature] {
    var signatures: [String: FileSignature] = [:]
    signatures.reserveCapacity(MonGARSModelManifest.expectedFiles.count)

    for expected in MonGARSModelManifest.expectedFiles {
      let fileURL = repositoryDirectory.appendingPathComponent(expected.path)
      guard
        let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
        let size = (attributes[.size] as? NSNumber)?.int64Value,
        size == expected.bytes,
        let modificationDate = attributes[.modificationDate] as? Date
      else {
        throw InferenceError.integrityFailure(expected.path)
      }
      signatures[expected.path] = FileSignature(
        bytes: size,
        modificationDate: modificationDate
      )
    }

    return signatures
  }

  private func sha256(_ url: URL) async throws -> String {
    let hashTask: Task<String, Error> = Task.detached(priority: .utility) {
      #if canImport(CryptoKit)
      let handle = try FileHandle(forReadingFrom: url)
      defer { try? handle.close() }
      var hasher = SHA256()
      while true {
        try Task.checkCancellation()
        guard let data = try handle.read(upToCount: 4 * 1_024 * 1_024), !data.isEmpty else {
          break
        }
        hasher.update(data: data)
      }
      return hasher.finalize().map { String(format: "%02x", $0) }.joined()
      #else
      throw InferenceError.integrityFailure(url.lastPathComponent)
      #endif
    }
    return try await withTaskCancellationHandler(
      operation: { try await hashTask.value },
      onCancel: { hashTask.cancel() }
    )
  }

  private func purgeArtifact(at relativePath: String) throws {
    verifiedFileSignatures = nil
    let artifact = repositoryDirectory.appendingPathComponent(relativePath)
    if fileManager.fileExists(atPath: artifact.path) {
      try fileManager.removeItem(at: artifact)
    }

    let metadataRoot = repositoryDirectory
      .appendingPathComponent(".cache", isDirectory: true)
      .appendingPathComponent("huggingface", isDirectory: true)
      .appendingPathComponent("download", isDirectory: true)
    let metadata = metadataRoot.appendingPathComponent(relativePath + ".metadata")
    if fileManager.fileExists(atPath: metadata.path) {
      try fileManager.removeItem(at: metadata)
    }

    let metadataDirectory = metadata.deletingLastPathComponent()
    if let entries = try? fileManager.contentsOfDirectory(
      at: metadataDirectory,
      includingPropertiesForKeys: nil
    ) {
      let prefix = artifact.lastPathComponent + "."
      for entry in entries
      where entry.lastPathComponent.hasPrefix(prefix)
        && entry.pathExtension == "incomplete"
      {
        try? fileManager.removeItem(at: entry)
      }
    }

    if fileManager.fileExists(atPath: markerURL.path) {
      try fileManager.removeItem(at: markerURL)
    }
  }
}
