import Foundation
import Hub

#if canImport(CoreML)
import CoreML
#endif

#if canImport(CryptoKit)
import CryptoKit
#endif

final class ModelStore {
  struct VerificationSession {
    private(set) var hasVerifiedRepository = false

    var requiresCryptographicVerification: Bool {
      !hasVerifiedRepository
    }

    mutating func recordCryptographicVerification() {
      hasVerifiedRepository = true
    }

    mutating func invalidate() {
      hasVerifiedRepository = false
    }
  }

  private struct VerificationMarker: Codable {
    let modelID: String
    let revision: String
    let manifestFingerprint: String
    let verifiedAt: Date
  }

  private struct CompilationMarker: Codable {
    let modelID: String
    let revision: String
    let manifestFingerprint: String
    let compiledAt: Date
  }

  private let fileManager: FileManager
  private let rootDirectory: URL
  private let hub: HubApi
  // This proof is deliberately process-local. It is established only by a
  // full manifest SHA-256 pass and is never reconstructed from file metadata.
  private var verificationSession = VerificationSession()

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

  var modelPackageDirectory: URL {
    repositoryDirectory.appendingPathComponent(
      MonGARSModelManifest.packageDirectory,
      isDirectory: true
    )
  }

  var compiledModelDirectory: URL {
    repositoryDirectory.appendingPathComponent(
      MonGARSModelManifest.compiledDirectory,
      isDirectory: true
    )
  }

  var tokenizerDirectory: URL { repositoryDirectory }

  private var markerURL: URL {
    repositoryDirectory.appendingPathComponent("mongars-verification.json")
  }

  private var compilationMarkerURL: URL {
    repositoryDirectory.appendingPathComponent("mongars-compilation.json")
  }

  func isInstalled() -> Bool {
    guard
      fileManager.fileExists(atPath: modelPackageDirectory.path),
      fileManager.fileExists(
        atPath: repositoryDirectory.appendingPathComponent("tokenizer.json").path
      ),
      let data = try? Data(contentsOf: markerURL),
      let marker = try? JSONDecoder().decode(VerificationMarker.self, from: data)
    else {
      verificationSession.invalidate()
      return false
    }
    guard
      marker.modelID == MonGARSModelManifest.modelID,
      marker.revision == MonGARSModelManifest.revision,
      marker.manifestFingerprint == Self.manifestFingerprint
    else {
      verificationSession.invalidate()
      return false
    }

    let hasExpectedArtifacts = MonGARSModelManifest.expectedFiles.allSatisfy { expected in
      let url = repositoryDirectory.appendingPathComponent(expected.path)
      guard
        let attributes = try? fileManager.attributesOfItem(atPath: url.path),
        let size = attributes[.size] as? NSNumber
      else {
        return false
      }
      return size.int64Value == expected.bytes
    }
    if !hasExpectedArtifacts {
      verificationSession.invalidate()
    }
    return hasExpectedArtifacts
  }

  func ensureVerified(
    progress: @escaping @Sendable (ModelProgress) -> Void = { _ in }
  ) async throws {
    guard isInstalled() else { throw InferenceError.modelNotInstalled }
    try applyBackupExclusion(to: repositoryDirectory)
    guard verificationSession.requiresCryptographicVerification else { return }

    do {
      try await verify(snapshot: repositoryDirectory, progress: progress)
      try Task.checkCancellation()
      verificationSession.recordCryptographicVerification()
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
      // Hub.snapshot may create or replace artifacts. Any previously verified
      // in-process repository therefore stops being trusted before it runs.
      verificationSession.invalidate()
      try removeCompiledModel()
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
        verificationSession.recordCryptographicVerification()
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
    verificationSession.invalidate()
    if fileManager.fileExists(atPath: repositoryDirectory.path) {
      try fileManager.removeItem(at: repositoryDirectory)
    }
  }

  #if canImport(CoreML)
  @available(iOS 18.0, macOS 15.0, *)
  func ensureCompiledModel(
    progress: @escaping @Sendable (ModelProgress) -> Void = { _ in }
  ) async throws -> URL {
    guard isInstalled() else { throw InferenceError.modelNotInstalled }
    try await ensureVerified()
    if isCompiledModelCurrent() {
      try applyBackupExclusion(to: compiledModelDirectory)
      return compiledModelDirectory
    }

    try removeCompiledModel()
    let available = try availableDiskCapacity()
    guard available >= MonGARSModelManifest.requiredCompilationFreeDiskBytes else {
      throw InferenceError.insufficientDisk(
        required: MonGARSModelManifest.requiredCompilationFreeDiskBytes,
        available: available
      )
    }

    progress(
      ModelProgress(
        phase: .compiling,
        fractionCompleted: 0,
        detail: "Compilation Core ML sur cet iPhone"
      )
    )
    try Task.checkCancellation()
    let temporaryCompiledURL = try await MLModel.compileModel(
      at: modelPackageDirectory
    )
    defer {
      if fileManager.fileExists(atPath: temporaryCompiledURL.path) {
        try? fileManager.removeItem(at: temporaryCompiledURL)
      }
    }
    try Task.checkCancellation()

    let stagingURL = repositoryDirectory.appendingPathComponent(
      ".mongars-\(UUID().uuidString).mlmodelc",
      isDirectory: true
    )
    do {
      try fileManager.moveItem(at: temporaryCompiledURL, to: stagingURL)
    } catch {
      if fileManager.fileExists(atPath: stagingURL.path) {
        try? fileManager.removeItem(at: stagingURL)
      }
      let copyBytes = directoryBytes(at: temporaryCompiledURL)
      let copyHeadroom = copyBytes + 500_000_000
      let copyCapacity = try availableDiskCapacity()
      guard copyCapacity >= copyHeadroom else {
        throw InferenceError.insufficientDisk(
          required: copyHeadroom,
          available: copyCapacity
        )
      }
      try fileManager.copyItem(at: temporaryCompiledURL, to: stagingURL)
    }

    do {
      try Task.checkCancellation()
      try fileManager.moveItem(at: stagingURL, to: compiledModelDirectory)
      try applyBackupExclusion(to: compiledModelDirectory)
      try writeCompilationMarker()
    } catch {
      if fileManager.fileExists(atPath: stagingURL.path) {
        try? fileManager.removeItem(at: stagingURL)
      }
      try? removeCompiledModel()
      throw error
    }

    progress(
      ModelProgress(
        phase: .compiling,
        fractionCompleted: 1,
        detail: "Compilation Core ML terminee"
      )
    )
    return compiledModelDirectory
  }
  #endif

  func invalidateCompiledModel() throws {
    try removeCompiledModel()
  }

  func installedDiskBytes() -> Int64 {
    guard isInstalled() else { return 0 }
    let compiledBytes = isCompiledModelCurrent()
      ? directoryBytes(at: compiledModelDirectory)
      : 0
    return MonGARSModelManifest.downloadBytes + compiledBytes
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

    guard let enumerator = fileManager.enumerator(
      at: snapshot,
      includingPropertiesForKeys: nil
    ) else {
      return
    }
    while let artifact = enumerator.nextObject() as? URL {
      var excludedArtifact = artifact
      try excludedArtifact.setResourceValues(values)
    }
  }

  private func isCompiledModelCurrent() -> Bool {
    guard
      fileManager.fileExists(atPath: compiledModelDirectory.path),
      let data = try? Data(contentsOf: compilationMarkerURL),
      let marker = try? JSONDecoder().decode(CompilationMarker.self, from: data)
    else {
      return false
    }
    return marker.modelID == MonGARSModelManifest.modelID
      && marker.revision == MonGARSModelManifest.revision
      && marker.manifestFingerprint == Self.manifestFingerprint
  }

  private func writeCompilationMarker() throws {
    let marker = CompilationMarker(
      modelID: MonGARSModelManifest.modelID,
      revision: MonGARSModelManifest.revision,
      manifestFingerprint: Self.manifestFingerprint,
      compiledAt: Date()
    )
    let data = try JSONEncoder().encode(marker)
    try data.write(to: compilationMarkerURL, options: .atomic)
  }

  private func removeCompiledModel() throws {
    if fileManager.fileExists(atPath: compiledModelDirectory.path) {
      try fileManager.removeItem(at: compiledModelDirectory)
    }
    if fileManager.fileExists(atPath: compilationMarkerURL.path) {
      try fileManager.removeItem(at: compilationMarkerURL)
    }
    if let entries = try? fileManager.contentsOfDirectory(
      at: repositoryDirectory,
      includingPropertiesForKeys: nil
    ) {
      for entry in entries
      where entry.lastPathComponent.hasPrefix(".mongars-")
        && entry.pathExtension == "mlmodelc"
      {
        try fileManager.removeItem(at: entry)
      }
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

  private func directoryBytes(at directory: URL) -> Int64 {
    guard fileManager.fileExists(atPath: directory.path) else { return 0 }
    guard let enumerator = fileManager.enumerator(
      at: directory,
      includingPropertiesForKeys: [.fileSizeKey],
      options: []
    ) else {
      return 0
    }
    var total: Int64 = 0
    while let file = enumerator.nextObject() as? URL {
      guard
        let values = try? file.resourceValues(
          forKeys: [.isRegularFileKey, .fileSizeKey]
        ),
        values.isRegularFile == true,
        let bytes = values.fileSize
      else {
        continue
      }
      total += Int64(bytes)
    }
    return total
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

      let checksum = try await Self.sha256(fileURL)
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

  static func sha256(_ url: URL) async throws -> String {
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
    verificationSession.invalidate()
    try removeCompiledModel()
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
