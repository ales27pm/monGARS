import Dispatch
import XCTest
@testable import MonGARSCoreML

final class PromptBuilderTests: XCTestCase {
  func testLargestFittingSuffixHandlesNonMonotonicEncodedLengths() throws {
    var attemptedLengths: [Int] = []
    let fittingLengths: Set<Int> = [4, 2, 1]

    let candidate = try PromptBuilder.largestFittingEncodedSuffix(
      contentTokens: Array(1...6),
      maximumSuffixTokens: 6,
      maximumPromptTokens: 10
    ) { suffix in
      attemptedLengths.append(suffix.count)
      let encodedCount = fittingLengths.contains(suffix.count) ? 10 : 11
      return Array(repeating: suffix.count, count: encodedCount)
    }

    XCTAssertEqual(attemptedLengths, [6, 5, 4])
    XCTAssertEqual(candidate?.first, 4)
  }

  func testLargestFittingSuffixCapsAttemptsAtModelContextLength() throws {
    let tokenCount = MonGARSModelManifest.contextLength + 100
    var attemptedLengths: [Int] = []

    let candidate = try PromptBuilder.largestFittingEncodedSuffix(
      contentTokens: Array(0..<tokenCount),
      maximumSuffixTokens: tokenCount,
      maximumPromptTokens: 1
    ) { suffix in
      attemptedLengths.append(suffix.count)
      return [suffix.count]
    }

    XCTAssertEqual(attemptedLengths, [MonGARSModelManifest.contextLength])
    XCTAssertEqual(candidate, [MonGARSModelManifest.contextLength])
  }

  func testLargestFittingSuffixObservesCancellationAfterEncodingAttempt() async {
    let encodingStarted = DispatchSemaphore(value: 0)
    let finishEncoding = DispatchSemaphore(value: 0)
    let task = Task.detached { () throws -> [Int]? in
      try PromptBuilder.largestFittingEncodedSuffix(
        contentTokens: Array(1...4),
        maximumSuffixTokens: 4,
        maximumPromptTokens: 1
      ) { _ in
        encodingStarted.signal()
        finishEncoding.wait()
        return [0, 1]
      }
    }

    XCTAssertEqual(encodingStarted.wait(timeout: .now() + 1), .success)
    task.cancel()
    finishEncoding.signal()

    do {
      _ = try await task.value
      XCTFail("PromptBuilder aurait du propager l'annulation.")
    } catch is CancellationError {
      // Expected.
    } catch {
      XCTFail("Erreur inattendue: \(error)")
    }
  }
}
