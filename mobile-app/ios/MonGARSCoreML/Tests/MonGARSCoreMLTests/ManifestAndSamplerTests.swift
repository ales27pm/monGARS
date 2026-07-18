import XCTest
@testable import MonGARSCoreML

final class ManifestAndSamplerTests: XCTestCase {
  func testSegmentedLogitsCoverEntireVocabulary() {
    XCTAssertEqual(
      MonGARSModelManifest.logitsChunkCount * MonGARSModelManifest.logitsChunkSize,
      MonGARSModelManifest.vocabularySize
    )
  }

  func testKVCacheMatchesPinnedStatefulContract() {
    XCTAssertEqual(MonGARSModelManifest.kvCacheShape, [56, 8, 512, 128])
  }

  func testManifestPinsEveryLargeRuntimeArtifact() {
    let expectedPaths = Set(MonGARSModelManifest.expectedFiles.map(\.path))
    XCTAssertEqual(expectedPaths, Set(MonGARSModelManifest.files))
    XCTAssertEqual(
      expectedPaths.count,
      MonGARSModelManifest.expectedFiles.count,
      "Chaque chemin doit etre unique."
    )
    XCTAssertEqual(
      MonGARSModelManifest.expectedFiles.reduce(Int64(0)) { $0 + $1.bytes },
      MonGARSModelManifest.installedBytes
    )
    XCTAssertTrue(expectedPaths.contains("tokenizer.json"))
    XCTAssertTrue(
      expectedPaths.contains(
        "\(MonGARSModelManifest.compiledDirectory)/weights/weight.bin"
      )
    )
    XCTAssertTrue(
      MonGARSModelManifest.expectedFiles.allSatisfy {
        $0.bytes > 0 && $0.sha256.count == 64
      }
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

  func testGreedySamplerReturnsHighestFiniteScore() {
    let scores: [Float] = [-10, 0.2, .nan, 3.5, 1.4]
    let token = Sampler.select(
      vocabularySize: scores.count,
      generatedTokens: [],
      options: GenerationOptions(doSample: false)
    ) { scores[$0] }
    XCTAssertEqual(token, 3)
  }

  func testRepetitionPenaltyCanChangeGreedyChoice() {
    let scores: [Float] = [5, 4.9]
    let token = Sampler.select(
      vocabularySize: scores.count,
      generatedTokens: [0],
      options: GenerationOptions(repetitionPenalty: 2, doSample: false)
    ) { scores[$0] }
    XCTAssertEqual(token, 1)
  }
}
