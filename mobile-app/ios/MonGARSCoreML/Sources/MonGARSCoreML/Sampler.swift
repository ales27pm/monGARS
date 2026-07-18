import Foundation

enum Sampler {
  private struct Candidate {
    let token: Int
    let score: Float
  }

  static func select(
    vocabularySize: Int,
    generatedTokens: Set<Int>,
    options: GenerationOptions,
    scoreAt: (Int) -> Float
  ) throws -> Int {
    guard vocabularySize > 0 else {
      throw InferenceError.invalidModel("Le vocabulaire de sortie est vide.")
    }
    guard !options.doSample || options.topK > 0 else {
      throw InferenceError.invalidGenerationOptions(
        "topK doit etre strictement positif pour l'echantillonnage."
      )
    }
    let candidateCount = options.doSample ? min(options.topK, vocabularySize) : 1
    var heap: [Candidate] = []
    heap.reserveCapacity(candidateCount)

    for token in 0..<vocabularySize {
      if token.isMultiple(of: 1_024) {
        try Task.checkCancellation()
      }
      var score = scoreAt(token)
      guard score.isFinite else { continue }

      if generatedTokens.contains(token), options.repetitionPenalty > 1 {
        score = score < 0
          ? score * options.repetitionPenalty
          : score / options.repetitionPenalty
      }

      if options.doSample {
        score /= options.temperature
      }

      push(
        Candidate(token: token, score: score),
        into: &heap,
        capacity: candidateCount
      )
    }
    try Task.checkCancellation()

    let sorted = heap.sorted { $0.score > $1.score }
    guard !sorted.isEmpty else {
      throw InferenceError.invalidModel("Aucun logit fini n'a ete produit.")
    }
    guard options.doSample, sorted.count > 1, let maximum = sorted.first?.score else {
      return sorted[0].token
    }

    let weights = sorted.map { exp(Double($0.score - maximum)) }
    let total = weights.reduce(0, +)
    guard total.isFinite, total > 0 else { return sorted[0].token }

    var retained: [(candidate: Candidate, probability: Double)] = []
    var cumulative = 0.0
    for (candidate, weight) in zip(sorted, weights) {
      let probability = weight / total
      retained.append((candidate, probability))
      cumulative += probability
      if cumulative >= Double(options.topP) { break }
    }

    let retainedTotal = retained.reduce(0) { $0 + $1.probability }
    var draw = Double.random(in: 0..<retainedTotal)
    for item in retained {
      draw -= item.probability
      if draw <= 0 { return item.candidate.token }
    }
    return retained.last?.candidate.token ?? sorted[0].token
  }

  private static func push(
    _ candidate: Candidate,
    into heap: inout [Candidate],
    capacity: Int
  ) {
    guard capacity > 0 else { return }
    if heap.count < capacity {
      heap.append(candidate)
      siftUp(&heap, from: heap.count - 1)
      return
    }
    guard let minimum = heap.first, candidate.score > minimum.score else { return }
    heap[0] = candidate
    siftDown(&heap, from: 0)
  }

  private static func siftUp(_ heap: inout [Candidate], from start: Int) {
    var index = start
    while index > 0 {
      let parent = (index - 1) / 2
      guard heap[index].score < heap[parent].score else { break }
      heap.swapAt(index, parent)
      index = parent
    }
  }

  private static func siftDown(_ heap: inout [Candidate], from start: Int) {
    var index = start
    while true {
      let left = index * 2 + 1
      let right = left + 1
      var smallest = index
      if left < heap.count, heap[left].score < heap[smallest].score {
        smallest = left
      }
      if right < heap.count, heap[right].score < heap[smallest].score {
        smallest = right
      }
      guard smallest != index else { return }
      heap.swapAt(index, smallest)
      index = smallest
    }
  }
}
