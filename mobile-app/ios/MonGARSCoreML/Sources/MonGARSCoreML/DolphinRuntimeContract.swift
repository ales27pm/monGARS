import Foundation

enum DolphinRuntimeContract {
  static func prefillRanges(tokenCount: Int) -> [Range<Int>] {
    guard tokenCount > 0 else { return [] }
    var ranges: [Range<Int>] = []
    var offset = 0
    while offset < tokenCount {
      let end = min(
        offset + MonGARSModelManifest.maximumQueryLength,
        tokenCount
      )
      ranges.append(offset..<end)
      offset = end
    }
    return ranges
  }

  static func causalMaskAllows(
    row: Int,
    column: Int,
    queryLength: Int,
    endStep: Int
  ) -> Bool {
    guard
      row >= 0,
      row < queryLength,
      column >= 0,
      column < endStep,
      queryLength > 0,
      endStep >= queryLength
    else {
      return false
    }
    let pastLength = endStep - queryLength
    return column <= pastLength + row
  }
}
