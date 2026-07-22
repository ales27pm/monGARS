import XCTest
@testable import MonGARSCoreML

final class ManifestAndSamplerTests: XCTestCase {
  func testDolphinVocabularyMatchesPinnedModel() {
    XCTAssertEqual(MonGARSModelManifest.vocabularySize, 128_258)
  }

  func testKVCacheMatchesPinnedStatefulContract() {
    XCTAssertEqual(
      MonGARSModelManifest.kvCacheShape,
      [28, 1, 8, 2_048, 128]
    )
    XCTAssertEqual(MonGARSModelManifest.maximumQueryLength, 512)
    XCTAssertEqual(MonGARSModelManifest.contextLength, 2_048)
  }

  func testManifestPinsEveryLargeRuntimeArtifact() {
    let lowercaseHex = Set("0123456789abcdef")
    let expectedPaths = Set(MonGARSModelManifest.expectedFiles.map(\.path))
    XCTAssertEqual(expectedPaths, Set(MonGARSModelManifest.files))
    XCTAssertEqual(
      expectedPaths.count,
      MonGARSModelManifest.expectedFiles.count,
      "Chaque chemin doit etre unique."
    )
    XCTAssertEqual(
      MonGARSModelManifest.expectedFiles.reduce(Int64(0)) { $0 + $1.bytes },
      MonGARSModelManifest.downloadBytes
    )
    XCTAssertTrue(expectedPaths.contains("tokenizer.json"))
    XCTAssertTrue(
      expectedPaths.contains(
        "\(MonGARSModelManifest.packageDirectory)"
          + "/Data/com.apple.CoreML/weights/weight.bin"
      )
    )
    XCTAssertTrue(
      MonGARSModelManifest.expectedFiles.allSatisfy { file in
        file.bytes > 0
          && file.sha256.count == 64
          && file.sha256.allSatisfy(lowercaseHex.contains)
      }
    )
  }

  func testManifestPinsDolphinHubAndSourceRevisions() {
    XCTAssertEqual(MonGARSModelManifest.modelID, "ales27pm/Dolphin3.0-CoreML")
    XCTAssertEqual(
      MonGARSModelManifest.revision,
      "95671cf9a2f56d2a381816ae264cd9aae335d96f"
    )
    XCTAssertEqual(
      MonGARSModelManifest.sourceRevision,
      "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"
    )
    XCTAssertEqual(
      MonGARSModelManifest.eosTokenIDs,
      Set([128_256, 128_001, 128_008, 128_009])
    )
  }

  func testGenerationOptionsAreBoundedByDeviceContract() {
    let options = GenerationOptions(
      maxNewTokens: 10_000,
      temperature: 10,
      topK: 50_000,
      topP: 4,
      repetitionPenalty: 5
    )
    XCTAssertEqual(options.maxNewTokens, MonGARSModelManifest.maximumNewTokens)
    XCTAssertEqual(options.temperature, 2)
    XCTAssertEqual(options.topK, 200)
    XCTAssertEqual(options.topP, 1)
    XCTAssertEqual(options.repetitionPenalty, 2)
  }

  func testGenerationDefaultsMatchPinnedGenerationConfig() {
    let options = GenerationOptions()

    XCTAssertEqual(options.temperature, 0.6)
    XCTAssertEqual(options.topP, 0.9)
    XCTAssertTrue(options.doSample)
  }

  func testGenerationOptionsReplaceNonFiniteValues() {
    let options = GenerationOptions(
      temperature: .nan,
      topP: .infinity,
      repetitionPenalty: -.infinity
    )
    XCTAssertEqual(options.temperature, 0.6)
    XCTAssertEqual(options.topP, 0.9)
    XCTAssertEqual(options.repetitionPenalty, 1.08)
  }

  func testGreedySamplerReturnsHighestFiniteScore() throws {
    let scores: [Float] = [-10, 0.2, -3, 3.5, 1.4]
    let token = try Sampler.select(
      vocabularySize: scores.count,
      generatedTokens: [],
      options: GenerationOptions(doSample: false)
    ) { scores[$0] }
    XCTAssertEqual(token, 3)
  }

  func testRepetitionPenaltyCanChangeGreedyChoice() throws {
    let scores: [Float] = [5, 4.9]
    let token = try Sampler.select(
      vocabularySize: scores.count,
      generatedTokens: [0],
      options: GenerationOptions(repetitionPenalty: 2, doSample: false)
    ) { scores[$0] }
    XCTAssertEqual(token, 1)
  }

  func testSamplerRejectsEmptyVocabulary() {
    XCTAssertThrowsError(
      try Sampler.select(
        vocabularySize: 0,
        generatedTokens: [],
        options: GenerationOptions(doSample: false)
      ) { _ in 0 }
    ) { error in
      assertInvalidModel(error)
    }
  }

  func testSamplerRejectsAllNonFiniteLogits() {
    XCTAssertThrowsError(
      try Sampler.select(
        vocabularySize: 3,
        generatedTokens: [],
        options: GenerationOptions(doSample: false)
      ) { _ in .nan }
    ) { error in
      assertInvalidModel(error)
    }
  }

  func testSamplerRejectsAnyNonFiniteLogit() {
    let scores: [Float] = [2, .nan, 1]
    XCTAssertThrowsError(
      try Sampler.select(
        vocabularySize: scores.count,
        generatedTokens: [],
        options: GenerationOptions(doSample: false)
      ) { scores[$0] }
    ) { error in
      assertInvalidModel(error)
    }
  }

  func testSamplerClassifiesMutatedZeroTopKAsInvalidGenerationOptions() {
    var options = GenerationOptions(doSample: true)
    options.topK = 0

    XCTAssertThrowsError(
      try Sampler.select(
        vocabularySize: 3,
        generatedTokens: [],
        options: options
      ) { Float($0) }
    ) { error in
      guard let inferenceError = error as? InferenceError else {
        XCTFail("InferenceError attendu, recu: \(error)")
        return
      }
      guard case .invalidGenerationOptions = inferenceError else {
        XCTFail("invalidGenerationOptions attendu, recu: \(error)")
        return
      }
    }
  }

  func testSamplerRejectsMutatedNonFiniteTemperature() {
    var options = GenerationOptions(doSample: true)
    options.temperature = .nan

    XCTAssertThrowsError(
      try Sampler.select(
        vocabularySize: 3,
        generatedTokens: [],
        options: options
      ) { Float($0) }
    ) { error in
      guard let inferenceError = error as? InferenceError else {
        XCTFail("InferenceError attendu, recu: \(error)")
        return
      }
      guard case .invalidGenerationOptions = inferenceError else {
        XCTFail("invalidGenerationOptions attendu, recu: \(error)")
        return
      }
    }
  }

  func testSamplerObservesTaskCancellation() async {
    let task = Task { () throws -> Int in
      await Task.yield()
      return try Sampler.select(
        vocabularySize: 100_000,
        generatedTokens: [],
        options: GenerationOptions(doSample: false)
      ) { Float($0) }
    }
    task.cancel()

    do {
      _ = try await task.value
      XCTFail("Le sampler aurait du propager l'annulation.")
    } catch is CancellationError {
      // Expected.
    } catch {
      XCTFail("Erreur inattendue: \(error)")
    }
  }

  private func assertInvalidModel(
    _ error: Error,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    guard let inferenceError = error as? InferenceError else {
      XCTFail("InferenceError attendu, recu: \(error)", file: file, line: line)
      return
    }
    guard case .invalidModel = inferenceError else {
      XCTFail("invalidModel attendu, recu: \(error)", file: file, line: line)
      return
    }
  }
}
