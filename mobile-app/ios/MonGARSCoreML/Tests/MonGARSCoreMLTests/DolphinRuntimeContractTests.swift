import XCTest
@testable import MonGARSCoreML

final class DolphinRuntimeContractTests: XCTestCase {
  func testPrefillSplitsAtPublishedMaximumQueryLength() {
    XCTAssertEqual(
      DolphinRuntimeContract.prefillRanges(tokenCount: 1_200),
      [0..<512, 512..<1_024, 1_024..<1_200]
    )
  }

  func testFirstPrefillChunkUsesCausalVisibility() {
    XCTAssertTrue(
      DolphinRuntimeContract.causalMaskAllows(
        row: 0,
        column: 0,
        queryLength: 3,
        endStep: 3
      )
    )
    XCTAssertFalse(
      DolphinRuntimeContract.causalMaskAllows(
        row: 0,
        column: 1,
        queryLength: 3,
        endStep: 3
      )
    )
    XCTAssertTrue(
      DolphinRuntimeContract.causalMaskAllows(
        row: 2,
        column: 2,
        queryLength: 3,
        endStep: 3
      )
    )
  }

  func testLaterPrefillChunkSeesCacheAndPriorRows() {
    XCTAssertTrue(
      DolphinRuntimeContract.causalMaskAllows(
        row: 0,
        column: 3,
        queryLength: 2,
        endStep: 5
      )
    )
    XCTAssertFalse(
      DolphinRuntimeContract.causalMaskAllows(
        row: 0,
        column: 4,
        queryLength: 2,
        endStep: 5
      )
    )
    XCTAssertTrue(
      DolphinRuntimeContract.causalMaskAllows(
        row: 1,
        column: 4,
        queryLength: 2,
        endStep: 5
      )
    )
  }

  func testDecodeTokenCanAttendThroughItsAbsolutePosition() {
    XCTAssertTrue(
      DolphinRuntimeContract.causalMaskAllows(
        row: 0,
        column: 512,
        queryLength: 1,
        endStep: 513
      )
    )
  }
}
