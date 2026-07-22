import CryptoKit
import Foundation
import MonGARSCoreML

public struct AgentImportedDocument: Sendable, Equatable {
  public let name: String
  public let content: String
  public let modifiedAt: Date?

  public init(name: String, content: String, modifiedAt: Date?) {
    self.name = name
    self.content = content
    self.modifiedAt = modifiedAt
  }
}

public enum AgentImportedDocumentDiscoveryError: Error, Sendable, Equatable {
  case rootUnavailable
  case enumerationFailed
}

public final class SafeLocalFileService: @unchecked Sendable {
  public let rootDirectory: URL
  private let fileManager: FileManager
  private let maximumReadBytes: Int
  private let permittedExtensions: Set<String>

  public init(
    rootDirectory: URL,
    fileManager: FileManager = .default,
    maximumReadBytes: Int = 512 * 1_024,
    permittedExtensions: Set<String> = [
      "txt", "md", "markdown", "json", "csv", "tsv", "log", "html", "htm",
      "xml", "yaml", "yml",
    ]
  ) {
    self.rootDirectory = rootDirectory.standardizedFileURL
    self.fileManager = fileManager
    self.maximumReadBytes = min(max(maximumReadBytes, 4 * 1_024), 2 * 1_024 * 1_024)
    self.permittedExtensions = permittedExtensions
    do {
      try fileManager.createDirectory(
        at: self.rootDirectory,
        withIntermediateDirectories: true
      )
#if os(iOS)
      try fileManager.setAttributes(
        [.protectionKey: FileProtectionType.complete],
        ofItemAtPath: self.rootDirectory.path
      )
#endif
    } catch {
      // A missing/unprotected import root simply leaves file tools
      // unadvertised; reads still fail closed below.
    }
  }

  public func read(name: String) -> AgentServiceResponse {
    guard let fileURL = SafeAgentFilePath.resolve(name: name, under: rootDirectory) else {
      return .denied("The requested file is outside the imported document directory.", code: "file_path_denied")
    }
    guard permittedExtensions.contains(fileURL.pathExtension.lowercased()) else {
      return .denied("That imported file type is not readable by the agent.", code: "file_type_denied")
    }
    do {
      let values = try fileURL.resourceValues(forKeys: [
        .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
      ])
      guard values.isRegularFile == true, values.isSymbolicLink != true else {
        return .denied("Only regular imported files can be read.", code: "file_not_regular")
      }
      guard let size = values.fileSize, size <= maximumReadBytes else {
        return .failed("The imported file exceeds the safe read limit.", code: "file_too_large")
      }
      let data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
      guard data.count <= maximumReadBytes,
        let content = String(data: data, encoding: .utf8) else {
        return .failed("The imported file is not safe UTF-8 text.", code: "file_invalid_text")
      }
      return .success(
        content,
        payload: [
          "source": .string(name),
          "bytes": .number(Double(data.count)),
          "content": .string(content),
        ]
      )
    } catch {
      return .failed("The imported file could not be read.", code: "file_read_failed")
    }
  }

  public func documentsForIndexing(
    maximumFiles: Int = 500
  ) -> Result<[AgentImportedDocument], AgentImportedDocumentDiscoveryError> {
    do {
      let rootValues = try rootDirectory.resourceValues(forKeys: [
        .isDirectoryKey, .isSymbolicLinkKey,
      ])
      guard rootValues.isDirectory == true, rootValues.isSymbolicLink != true else {
        return .failure(.rootUnavailable)
      }
    } catch {
      return .failure(.rootUnavailable)
    }

    var enumerationFailed = false
    guard let enumerator = fileManager.enumerator(
      at: rootDirectory,
      includingPropertiesForKeys: [
        .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey,
      ],
      options: [.skipsHiddenFiles, .skipsPackageDescendants],
      errorHandler: { _, _ in
        enumerationFailed = true
        return false
      }
    ) else { return .failure(.enumerationFailed) }

    var documents: [AgentImportedDocument] = []
    while let url = enumerator.nextObject() as? URL, documents.count < maximumFiles {
      guard permittedExtensions.contains(url.pathExtension.lowercased()),
        let relative = relativeName(for: url),
        let safeURL = SafeAgentFilePath.resolve(name: relative, under: rootDirectory),
        safeURL == url.standardizedFileURL.resolvingSymlinksInPath(),
        let values = try? safeURL.resourceValues(forKeys: [
          .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey,
        ]),
        values.isRegularFile == true,
        values.isSymbolicLink != true,
        let size = values.fileSize,
        size <= maximumReadBytes,
        let data = try? Data(contentsOf: safeURL, options: [.mappedIfSafe]),
        data.count <= maximumReadBytes,
        let content = String(data: data, encoding: .utf8) else { continue }
      documents.append(.init(name: relative, content: content, modifiedAt: values.contentModificationDate))
    }
    guard !enumerationFailed else { return .failure(.enumerationFailed) }
    return .success(
      documents.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    )
  }

  public func hasReadableDocuments() -> Bool {
    guard case let .success(documents) = documentsForIndexing(maximumFiles: 1) else {
      return false
    }
    return !documents.isEmpty
  }

  private func relativeName(for url: URL) -> String? {
    let root = rootDirectory.standardizedFileURL.resolvingSymlinksInPath().path
    let candidate = url.standardizedFileURL.path
    let prefix = root.hasSuffix("/") ? root : root + "/"
    guard candidate.hasPrefix(prefix) else { return nil }
    return String(candidate.dropFirst(prefix.count))
  }
}

private struct StoredMemory: Codable, Sendable, Equatable {
  let id: UUID
  let scope: String
  let kind: String
  let content: String
  let createdAt: Date
}

private struct StoredKnowledgeChunk: Codable, Sendable, Equatable {
  let id: String
  let scope: String
  let sourceType: String
  let provenance: String
  let content: String
  let modifiedAt: Date?
  let checksum: String
}

private struct StoredKnowledgeState: Codable, Sendable, Equatable {
  var memories: [StoredMemory] = []
  var chunks: [StoredKnowledgeChunk] = []
}

private enum StoredKnowledgeStateLoadError: Error, Sendable, Equatable {
  case corrupt
  case unavailable
}

public actor AgentLocalKnowledgeStore {
  private let stateURL: URL
  private let now: @Sendable () -> Date
  private var cachedState: StoredKnowledgeState?

  public init(
    stateURL: URL,
    now: @escaping @Sendable () -> Date = { Date() }
  ) {
    self.stateURL = stateURL
    self.now = now
  }

  public func saveMemory(content: String, kind: String, scope: String) -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active memory scope is invalid.", code: "memory_scope_invalid")
    }
    let content = content.trimmingCharacters(in: .whitespacesAndNewlines)
    let kind = kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !content.isEmpty, content.utf8.count <= 16_000,
      !kind.isEmpty, kind.utf8.count <= 64 else {
      return .failed("The memory content or kind is invalid.", code: "memory_invalid_arguments")
    }
    let stateResult = loadStateResult()
    guard case .success(var state) = stateResult else {
      return Self.loadFailureResponse(stateResult)
    }
    let memory = StoredMemory(
      id: UUID(),
      scope: scope,
      kind: kind,
      content: content,
      createdAt: now()
    )
    state.memories.append(memory)
    state.memories = Array(state.memories.suffix(2_000))
    state.chunks.removeAll { $0.id == "memory:\(memory.id.uuidString)" }
    state.chunks.append(.init(
      id: "memory:\(memory.id.uuidString)",
      scope: scope,
      sourceType: "notes",
      provenance: "memory:\(memory.id.uuidString)",
      content: content,
      modifiedAt: memory.createdAt,
      checksum: Self.checksum(content)
    ))
    guard persist(state) else {
      return .failed("The memory could not be stored securely.", code: "memory_persist_failed")
    }
    return .success(
      "Memory saved in the active profile.",
      payload: [
        "id": .string(memory.id.uuidString),
        "kind": .string(kind),
        "createdAt": .string(Self.iso8601.string(from: memory.createdAt)),
      ]
    )
  }

  public func recallMemory(query: String, scope: String, limit: Int = 8) -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active memory scope is invalid.", code: "memory_scope_invalid")
    }
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty, query.utf8.count <= 4_000 else {
      return .failed("The memory query is invalid.", code: "memory_invalid_query")
    }
    let stateResult = loadStateResult()
    guard case let .success(state) = stateResult else {
      return Self.loadFailureResponse(stateResult)
    }
    let matches = state.memories
      .filter { $0.scope == scope }
      .compactMap { memory -> (StoredMemory, Double)? in
        let score = Self.lexicalScore(query: query, content: memory.content)
        return score > 0 ? (memory, score) : nil
      }
      .sorted {
        if $0.1 != $1.1 { return $0.1 > $1.1 }
        return $0.0.createdAt > $1.0.createdAt
      }
      .prefix(min(max(limit, 1), 20))

    guard !matches.isEmpty else {
      return .success("No matching memories were found.", payload: ["matches": []])
    }
    let values: [AgentJSONValue] = matches.map { memory, score in
      [
        "id": .string(memory.id.uuidString),
        "kind": .string(memory.kind),
        "content": .string(memory.content),
        "createdAt": .string(Self.iso8601.string(from: memory.createdAt)),
        "score": .number(score),
      ]
    }
    let text = matches.map { "- [\($0.0.kind)] \($0.0.content)" }.joined(separator: "\n")
    return .success(text, payload: ["matches": .array(values)])
  }

  public func indexDocuments(
    _ documents: [AgentImportedDocument],
    scope: String
  ) -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active knowledge scope is invalid.", code: "rag_scope_invalid")
    }
    let stateResult = loadStateResult()
    guard case .success(var state) = stateResult else {
      return Self.loadFailureResponse(stateResult)
    }
    state.chunks.removeAll { $0.scope == scope && $0.sourceType == "documents" }
    var chunks: [StoredKnowledgeChunk] = []
    for document in documents.prefix(500) {
      for (index, content) in Self.chunk(document.content).enumerated() {
        chunks.append(.init(
          id: "document:\(Self.checksum(document.name)):\(index)",
          scope: scope,
          sourceType: "documents",
          provenance: "file:\(document.name)#chunk=\(index)",
          content: content,
          modifiedAt: document.modifiedAt,
          checksum: Self.checksum(content)
        ))
      }
    }
    state.chunks.append(contentsOf: chunks)
    state.chunks = Array(state.chunks.suffix(20_000))
    guard persist(state) else {
      return .failed("The local file index could not be saved.", code: "rag_persist_failed")
    }
    return .success(
      "Indexed \(documents.count) imported files into \(chunks.count) local chunks.",
      payload: [
        "documents": .number(Double(documents.count)),
        "chunks": .number(Double(chunks.count)),
        "provenance": .string("local-imported-files"),
      ]
    )
  }

  public func indexPhotos(
    _ photos: [AgentPhotoMetadata],
    scope: String
  ) -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active knowledge scope is invalid.", code: "rag_scope_invalid")
    }
    let stateResult = loadStateResult()
    guard case .success(var state) = stateResult else {
      return Self.loadFailureResponse(stateResult)
    }
    state.chunks.removeAll { $0.scope == scope && $0.sourceType == "photos" }
    let chunks: [StoredKnowledgeChunk] = photos.prefix(5_000).map { photo in
      let date = photo.createdAt.map(Self.iso8601.string) ?? "unknown date"
      let name = photo.filename ?? "photo"
      let type = photo.mediaType ?? "image"
      let subtype = photo.mediaSubtypes.isEmpty
        ? ""
        : ", subtypes \(photo.mediaSubtypes.joined(separator: ","))"
      let favorite = photo.isFavorite == true ? ", favorite" : ""
      let coordinate: String
      if let latitude = photo.latitude, let longitude = photo.longitude {
        coordinate = " at \(latitude),\(longitude)"
      } else {
        coordinate = ""
      }
      let content = "\(name), \(type), created \(date)\(coordinate)\(subtype)\(favorite)"
      return .init(
        id: "photo:\(Self.checksum(photo.localIdentifier))",
        scope: scope,
        sourceType: "photos",
        provenance: "photo:\(photo.localIdentifier)",
        content: content,
        modifiedAt: photo.createdAt,
        checksum: Self.checksum(content)
      )
    }
    state.chunks.append(contentsOf: chunks)
    state.chunks = Array(state.chunks.suffix(20_000))
    guard persist(state) else {
      return .failed("The local photo index could not be saved.", code: "rag_persist_failed")
    }
    return .success(
      "Indexed metadata for \(chunks.count) local photos.",
      payload: [
        "photos": .number(Double(chunks.count)),
        "provenance": .string("photo-library-metadata"),
      ]
    )
  }

  public func search(
    query: String,
    sourceScope: String,
    scope: String,
    limit: Int
  ) -> AgentServiceResponse {
    guard let scope = AgentToolInput.validatedScope(scope) else {
      return .denied("The active knowledge scope is invalid.", code: "rag_scope_invalid")
    }
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty, query.utf8.count <= 4_000 else {
      return .failed("The local search query is invalid.", code: "rag_invalid_query")
    }
    let allowedSources: Set<String>
    switch sourceScope {
    case "all": allowedSources = ["documents", "notes", "photos"]
    case "documents", "notes", "photos": allowedSources = [sourceScope]
    default: return .failed("The local source scope is invalid.", code: "rag_invalid_source_scope")
    }
    let stateResult = loadStateResult()
    guard case let .success(state) = stateResult else {
      return Self.loadFailureResponse(stateResult)
    }
    let matches = state.chunks
      .filter { $0.scope == scope && allowedSources.contains($0.sourceType) }
      .compactMap { chunk -> (StoredKnowledgeChunk, Double)? in
        let score = Self.lexicalScore(query: query, content: chunk.content)
        return score > 0 ? (chunk, score) : nil
      }
      .sorted {
        if $0.1 != $1.1 { return $0.1 > $1.1 }
        return $0.0.provenance < $1.0.provenance
      }
      .prefix(min(max(limit, 1), 20))

    guard !matches.isEmpty else {
      return .success("No matching local knowledge was found.", payload: ["matches": []])
    }
    let values: [AgentJSONValue] = matches.map { chunk, score in
      [
        "sourceType": .string(chunk.sourceType),
        "provenance": .string(chunk.provenance),
        "content": .string(chunk.content),
        "checksum": .string(chunk.checksum),
        "score": .number(score),
      ]
    }
    let text = matches.map {
      "- [\($0.0.provenance)] \($0.0.content)"
    }.joined(separator: "\n")
    return .success(text, payload: ["matches": .array(values)])
  }

  private func loadStateResult() -> Result<StoredKnowledgeState, StoredKnowledgeStateLoadError> {
    if let cachedState { return .success(cachedState) }
    let data: Data
    do {
      let values = try stateURL.resourceValues(forKeys: [
        .isRegularFileKey, .isSymbolicLinkKey,
      ])
      guard values.isRegularFile == true, values.isSymbolicLink != true else {
        return .failure(.unavailable)
      }
      data = try Data(contentsOf: stateURL, options: [.mappedIfSafe])
    } catch {
      let cocoaError = error as NSError
      if cocoaError.domain == NSCocoaErrorDomain,
        cocoaError.code == NSFileReadNoSuchFileError {
        let empty = StoredKnowledgeState()
        cachedState = empty
        return .success(empty)
      }
      return .failure(.unavailable)
    }
    guard let decoded = try? Self.decoder.decode(StoredKnowledgeState.self, from: data) else {
      return .failure(.corrupt)
    }
    cachedState = decoded
    return .success(decoded)
  }

  private static func loadFailureResponse(
    _ result: Result<StoredKnowledgeState, StoredKnowledgeStateLoadError>
  ) -> AgentServiceResponse {
    guard case let .failure(error) = result else {
      return .failed("The local knowledge store could not be loaded.", code: "knowledge_store_unavailable")
    }
    switch error {
    case .corrupt:
      return .failed(
        "The local knowledge store is corrupt and was left unchanged.",
        code: "knowledge_store_corrupt"
      )
    case .unavailable:
      return .failed(
        "The protected local knowledge store is unavailable.",
        code: "knowledge_store_unavailable"
      )
    }
  }

  private func persist(_ state: StoredKnowledgeState) -> Bool {
    do {
      let directory = stateURL.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      let data = try Self.encoder.encode(state)
      try data.write(to: stateURL, options: [.atomic, .agentCompleteFileProtection])
#if os(iOS)
      try FileManager.default.setAttributes(
        [.protectionKey: FileProtectionType.complete],
        ofItemAtPath: stateURL.path
      )
#endif
      cachedState = state
      return true
    } catch {
      return false
    }
  }

  private static func lexicalScore(query: String, content: String) -> Double {
    let queryTokens = Set(tokens(query))
    guard !queryTokens.isEmpty else { return 0 }
    let contentTokens = tokens(content)
    let contentSet = Set(contentTokens)
    let overlap = queryTokens.intersection(contentSet).count
    guard overlap > 0 else { return 0 }
    let coverage = Double(overlap) / Double(queryTokens.count)
    let density = Double(overlap) / Double(max(contentSet.count, 1))
    let phrase = content.localizedCaseInsensitiveContains(query) ? 0.3 : 0
    return min(1, coverage * 0.7 + density * 0.2 + phrase)
  }

  private static func tokens(_ text: String) -> [String] {
    text.lowercased().unicodeScalars
      .split { !CharacterSet.alphanumerics.contains($0) }
      .map(String.init)
      .filter { $0.count > 1 }
  }

  private static func chunk(_ text: String, size: Int = 700, overlap: Int = 80) -> [String] {
    let normalized = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalized.isEmpty else { return [] }
    var output: [String] = []
    var start = normalized.startIndex
    while start < normalized.endIndex {
      let end = normalized.index(start, offsetBy: size, limitedBy: normalized.endIndex) ?? normalized.endIndex
      output.append(String(normalized[start..<end]))
      guard end < normalized.endIndex else { break }
      start = normalized.index(end, offsetBy: -min(overlap, normalized.distance(from: start, to: end)))
    }
    return output
  }

  private static func checksum(_ text: String) -> String {
    SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
  }

  private static let iso8601 = ISO8601DateFormatter()
  private static let encoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    return encoder
  }()
  private static let decoder: JSONDecoder = {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    return decoder
  }()
}

private extension Data.WritingOptions {
  static var agentCompleteFileProtection: Data.WritingOptions {
#if os(iOS)
    return .completeFileProtection
#else
    return []
#endif
  }
}
