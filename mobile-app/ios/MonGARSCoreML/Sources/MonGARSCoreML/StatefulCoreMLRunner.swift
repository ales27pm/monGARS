#if canImport(CoreML)
import CoreML
import CoreVideo
import Foundation
import Tokenizers

@available(iOS 18.0, macOS 15.0, *)
final class StatefulCoreMLRunner {
  private let inferModel: MLModel
  private let prefillModel: MLModel
  private let tokenizer: any Tokenizer
  private let predictionQueue = DispatchQueue(
    label: "com.mongars.coreml.stateful-prediction"
  )
  private let inferOutputBackings: [[String: MLMultiArray]]
  private var nextBackingIndex = 0

  init(modelURL: URL, tokenizer: any Tokenizer) throws {
    let inferConfiguration = MLModelConfiguration()
    inferConfiguration.computeUnits = .cpuAndNeuralEngine
    inferConfiguration.functionName = "infer"
    inferModel = try MLModel(contentsOf: modelURL, configuration: inferConfiguration)

    let prefillConfiguration = MLModelConfiguration()
    prefillConfiguration.computeUnits = .cpuAndNeuralEngine
    prefillConfiguration.functionName = "prefill"
    prefillModel = try MLModel(contentsOf: modelURL, configuration: prefillConfiguration)
    self.tokenizer = tokenizer
    inferOutputBackings = try Self.makeInferOutputBackings(count: 16)

    try Self.validate(model: inferModel, function: "infer", sequenceLength: 1)
    try Self.validate(model: prefillModel, function: "prefill", sequenceLength: 64)
  }

  func generate(
    promptTokens: [Int],
    requestedOptions: GenerationOptions,
    onUpdate: @escaping @Sendable (GenerationUpdate) async -> Void
  ) async throws -> GenerationResult {
    guard !promptTokens.isEmpty else { throw InferenceError.emptyPrompt }
    guard promptTokens.count < MonGARSModelManifest.contextLength else {
      throw InferenceError.invalidModel("Le prompt remplit tout le contexte.")
    }

    if ProcessInfo.processInfo.thermalState == .critical {
      throw InferenceError.thermalCritical
    }

    var options = requestedOptions
    if ProcessInfo.processInfo.isLowPowerModeEnabled
      || ProcessInfo.processInfo.thermalState == .serious
    {
      options.maxNewTokens = min(options.maxNewTokens, 48)
      options.doSample = false
    }
    options.maxNewTokens = min(
      options.maxNewTokens,
      MonGARSModelManifest.contextLength - promptTokens.count
    )

    let state = inferModel.makeState()
    let startedAt = Date()
    var currentLogits = try await prefill(
      promptTokens: promptTokens,
      state: state
    )
    try Task.checkCancellation()

    var generated: [Int] = []
    var generatedSet = Set<Int>()
    var finishReason = "length"
    var lastEmission = Date.distantPast
    var emittedText = ""

    for _ in 0..<options.maxNewTokens {
      try Task.checkCancellation()
      if ProcessInfo.processInfo.thermalState == .critical {
        throw InferenceError.thermalCritical
      }
      let token = Sampler.select(
        vocabularySize: MonGARSModelManifest.vocabularySize,
        generatedTokens: generatedSet,
        options: options
      ) { tokenID in
        Self.score(tokenID: tokenID, chunks: currentLogits)
      }

      if MonGARSModelManifest.eosTokenIDs.contains(token) {
        finishReason = "eos"
        break
      }

      generated.append(token)
      generatedSet.insert(token)
      let text = clean(
        tokenizer.decode(tokens: generated, skipSpecialTokens: true)
      )
      let now = Date()
      let elapsed = max(now.timeIntervalSince(startedAt), 0.001)

      if now.timeIntervalSince(lastEmission) >= 0.05 {
        emittedText = text
        lastEmission = now
        await onUpdate(
          GenerationUpdate(
            text: text,
            generatedTokens: generated.count,
            tokensPerSecond: Double(generated.count) / elapsed
          )
        )
      }

      let position = promptTokens.count + generated.count - 1
      guard position < MonGARSModelManifest.contextLength else {
        finishReason = "context"
        break
      }
      currentLogits = try predict(
        token: token,
        position: position,
        state: state
      )
    }

    let finalText = clean(
      tokenizer.decode(tokens: generated, skipSpecialTokens: true)
    )
    let duration = max(Date().timeIntervalSince(startedAt), 0.001)
    if finalText != emittedText {
      await onUpdate(
        GenerationUpdate(
          text: finalText,
          generatedTokens: generated.count,
          tokensPerSecond: Double(generated.count) / duration
        )
      )
    }

    return GenerationResult(
      text: finalText,
      promptTokens: promptTokens.count,
      generatedTokens: generated.count,
      duration: duration,
      tokensPerSecond: Double(generated.count) / duration,
      finishReason: finishReason
    )
  }

  private func prefill(
    promptTokens: [Int],
    state: MLState
  ) async throws -> [MLMultiArray] {
    let batchSize = 64
    var position = 0

    while position + batchSize <= promptTokens.count {
      try Task.checkCancellation()
      let inputIDs = try MLMultiArray(shape: [1, 64], dataType: .int32)
      let positionIDs = try MLMultiArray(shape: [64], dataType: .int32)
      let currentPosition = try MLMultiArray(shape: [1], dataType: .int32)
      let causalMask = try MLMultiArray(
        shape: [1, 1, 64, NSNumber(value: MonGARSModelManifest.contextLength)],
        dataType: .float16
      )

      for index in 0..<batchSize {
        inputIDs[index] = NSNumber(value: promptTokens[position + index])
        positionIDs[index] = NSNumber(value: position + index)
      }
      currentPosition[0] = NSNumber(value: position)
      fill(mask: causalMask, batchPosition: position, batchSize: batchSize)

      let provider = try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": inputIDs,
        "position_ids": positionIDs,
        "causal_mask": causalMask,
        "current_pos": currentPosition,
      ])
      _ = try predictionQueue.sync {
        try prefillModel.prediction(
          from: provider,
          using: state,
          options: MLPredictionOptions()
        )
      }
      position += batchSize
      await Task.yield()
    }

    var logits: [MLMultiArray]?
    if position == promptTokens.count {
      // Prefill populates the state but its 64-row logits are intentionally not
      // retained. Replaying the final position gives the decode-shaped logits
      // while deterministically overwriting the same KV slot.
      logits = try predict(
        token: promptTokens[position - 1],
        position: position - 1,
        state: state
      )
    }
    while position < promptTokens.count {
      try Task.checkCancellation()
      logits = try predict(
        token: promptTokens[position],
        position: position,
        state: state
      )
      position += 1
      await Task.yield()
    }

    guard let logits else {
      throw InferenceError.invalidModel("Aucun logit produit apres le prefill.")
    }
    return logits
  }

  private func predict(
    token: Int,
    position: Int,
    state: MLState
  ) throws -> [MLMultiArray] {
    let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
    inputIDs[0] = NSNumber(value: token)

    let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
    positionIDs[0] = NSNumber(value: position)

    let currentPosition = try MLMultiArray(shape: [1], dataType: .int32)
    currentPosition[0] = NSNumber(value: position)

    let causalMask = try MLMultiArray(
      shape: [1, 1, 1, NSNumber(value: MonGARSModelManifest.contextLength)],
      dataType: .float16
    )
    fill(mask: causalMask, visibleThrough: position)

    let provider = try MLDictionaryFeatureProvider(dictionary: [
      "input_ids": inputIDs,
      "position_ids": positionIDs,
      "causal_mask": causalMask,
      "current_pos": currentPosition,
    ])
    let backings = inferOutputBackings[nextBackingIndex]
    nextBackingIndex = (nextBackingIndex + 1) % inferOutputBackings.count
    let predictionOptions = MLPredictionOptions()
    predictionOptions.outputBackings = backings
    _ = try predictionQueue.sync {
      try inferModel.prediction(
        from: provider,
        using: state,
        options: predictionOptions
      )
    }

    return try (1...MonGARSModelManifest.logitsChunkCount).map { index in
      guard let logits = backings["logits\(index)"] else {
        throw InferenceError.invalidModel("Sortie logits\(index) absente.")
      }
      return logits
    }
  }

  private func fill(mask: MLMultiArray, visibleThrough position: Int) {
    let pointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
    for index in 0..<MonGARSModelManifest.contextLength {
      pointer[index] = index <= position ? 0 : -Float16.infinity
    }
  }

  private func fill(mask: MLMultiArray, batchPosition: Int, batchSize: Int) {
    let pointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
    let context = MonGARSModelManifest.contextLength
    for row in 0..<batchSize {
      let visibleThrough = batchPosition + row
      for column in 0..<context {
        pointer[row * context + column] = column <= visibleThrough ? 0 : -Float16.infinity
      }
    }
  }

  private func clean(_ value: String) -> String {
    let thinkingPrefix = "<think>\n\n</think>\n\n"
    if value.hasPrefix(thinkingPrefix) {
      return String(value.dropFirst(thinkingPrefix.count))
    }
    return value
  }

  private static func score(tokenID: Int, chunks: [MLMultiArray]) -> Float {
    let chunkIndex = tokenID / MonGARSModelManifest.logitsChunkSize
    let localIndex = tokenID % MonGARSModelManifest.logitsChunkSize
    guard chunks.indices.contains(chunkIndex) else { return -Float.infinity }
    let logits = chunks[chunkIndex]
    guard localIndex < logits.count else { return -Float.infinity }

    switch logits.dataType {
    case .float16:
      let pointer = logits.dataPointer.assumingMemoryBound(to: Float16.self)
      return Float(pointer[localIndex])
    case .float32:
      let pointer = logits.dataPointer.assumingMemoryBound(to: Float.self)
      return pointer[localIndex]
    case .double:
      let pointer = logits.dataPointer.assumingMemoryBound(to: Double.self)
      return Float(pointer[localIndex])
    default:
      return logits[localIndex].floatValue
    }
  }

  private static func validate(
    model: MLModel,
    function: String,
    sequenceLength: Int
  ) throws {
    let description = model.modelDescription
    let expectedInputs: [String: (shape: [Int], type: MLMultiArrayDataType)] = [
      "input_ids": ([1, sequenceLength], .int32),
      "position_ids": ([sequenceLength], .int32),
      "causal_mask": (
        [1, 1, sequenceLength, MonGARSModelManifest.contextLength],
        .float16
      ),
      "current_pos": ([1], .int32),
    ]

    for (name, expected) in expectedInputs {
      guard
        let input = description.inputDescriptionsByName[name],
        let constraint = input.multiArrayConstraint,
        constraint.shape.map(\.intValue) == expected.shape,
        constraint.dataType == expected.type
      else {
        throw InferenceError.invalidModel(
          "\(function).\(name) ne respecte pas \(expected.shape)."
        )
      }
    }

    guard
      let state = description.stateDescriptionsByName["model_model_kv_cache_0"],
      let stateConstraint = state.stateConstraint,
      stateConstraint.bufferShape == MonGARSModelManifest.kvCacheShape,
      stateConstraint.dataType == .float16
    else {
      throw InferenceError.invalidModel("Etat KV absent ou invalide pour \(function).")
    }

    for index in 1...MonGARSModelManifest.logitsChunkCount {
      guard
        let output = description.outputDescriptionsByName["logits\(index)"],
        let constraint = output.multiArrayConstraint,
        constraint.dataType == .float16,
        constraint.shape.map(\.intValue) == [
          1,
          sequenceLength,
          MonGARSModelManifest.logitsChunkSize,
        ]
      else {
        throw InferenceError.invalidModel("Sortie logits\(index) invalide.")
      }
    }
  }

  private static func makeInferOutputBackings(
    count: Int
  ) throws -> [[String: MLMultiArray]] {
    try (0..<count).map { _ in
      var backings: [String: MLMultiArray] = [:]
      for index in 1...MonGARSModelManifest.logitsChunkCount {
        var pixelBuffer: CVPixelBuffer?
        let attributes: [String: Any] = [
          kCVPixelBufferMetalCompatibilityKey as String: true,
          kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        let status = CVPixelBufferCreate(
          kCFAllocatorDefault,
          MonGARSModelManifest.logitsChunkSize,
          1,
          kCVPixelFormatType_OneComponent16Half,
          attributes as CFDictionary,
          &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
          throw InferenceError.invalidModel(
            "Impossible d'allouer le backing IOSurface logits\(index)."
          )
        }
        backings["logits\(index)"] = MLMultiArray(
          pixelBuffer: pixelBuffer,
          shape: [1, 1, NSNumber(value: MonGARSModelManifest.logitsChunkSize)]
        )
      }
      return backings
    }
  }
}
#endif
