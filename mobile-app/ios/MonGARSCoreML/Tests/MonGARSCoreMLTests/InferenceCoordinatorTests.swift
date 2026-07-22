import XCTest
@testable import MonGARSCoreML

final class InferenceErrorTests: XCTestCase {
  func testCancellationClassifierIgnoresAmbientTaskCancellation() async {
    let task = Task { () -> (taskWasCancelled: Bool, classified: Bool) in
      withUnsafeCurrentTask { $0?.cancel() }
      return (
        Task.isCancelled,
        InferenceError.isCancellation(
          InferenceError.invalidModel("Echec Core ML reel.")
        )
      )
    }

    let result = await task.value
    XCTAssertTrue(result.taskWasCancelled)
    XCTAssertFalse(result.classified)
  }

  func testCancellationClassifierRecognizesCancellationErrors() {
    XCTAssertTrue(InferenceError.isCancellation(CancellationError()))
    XCTAssertTrue(
      InferenceError.isCancellation(InferenceError.preparationCancelled)
    )
    XCTAssertTrue(
      InferenceError.isCancellation(InferenceError.generationCancelled)
    )
  }

  func testIntegrityFailureIsRecoverableByModelPreparation() {
    XCTAssertTrue(InferenceError.integrityFailure("weight.bin").isRecoverable)
    XCTAssertFalse(InferenceError.invalidModel("schema").isRecoverable)
  }
}
