#if canImport(CoreML)
import XCTest
@testable import MonGARSCoreML

@available(iOS 18.0, macOS 15.0, *)
final class InferenceCoordinatorTests: XCTestCase {
  func testCancellationClassifierDoesNotMaskRealErrorInCancelledTask() async {
    let task = Task { () -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      return InferenceCoordinator.isCancellation(
        InferenceError.invalidModel("Echec Core ML reel.")
      )
    }

    let classifiedAsCancellation = await task.value
    XCTAssertFalse(classifiedAsCancellation)
  }

  func testCancellationClassifierRecognizesCancellationErrors() {
    XCTAssertTrue(InferenceCoordinator.isCancellation(CancellationError()))
    XCTAssertTrue(
      InferenceCoordinator.isCancellation(InferenceError.preparationCancelled)
    )
    XCTAssertTrue(
      InferenceCoordinator.isCancellation(InferenceError.generationCancelled)
    )
  }
}
#endif
