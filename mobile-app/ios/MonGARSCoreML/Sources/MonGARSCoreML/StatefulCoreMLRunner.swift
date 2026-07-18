#if canImport(CoreML)
import CoreML
import Foundation
import Tokenizers

@available(iOS 18.0, macOS 15.0, *)
final class StatefulCoreMLRunner {
  private struct LogitsView {
    let backing: MLMultiArray
    let values: UnsafeMutablePointer<Float16>
    let rowOffset: Int
    let tokenStride: Int
    let vocabularySize: Int

    init(_ logits: MLMultiArray) throws {
      let shape = logits.shape.map(\.intValue)
      let strides = logits.strides.map(\.intValue)
      guard
        logits.dataType == .float16,
        shape.count == 3,
        strides.count == 3,
        shape[1] > 0,
        shape[2] == MonGARSModelManifest.vocabularySize
      else {
        throw InferenceError.invalidModel("Vue logits Dolphin invalide.")
      }
      backing = logits
      values = logits.dataPointer.bindMemory(
        to: Float16.self,
        capacity: logits.count
      )
      rowOffset = (shape[1] - 1) * strides[1]
      tokenStride = strides[2]
      vocabularySize = shape[2]
    }

    func score(tokenID: Int) -> Float {
      guard tokenID >= 0, tokenID < vocabularySize else {
        return -Float.infinity
      }
      return Float(values[rowOffset + tokenID * tokenStride])
    }
  }

  private let model: MLModel
  private let tokenizer: any Tokenizer
  private let predictionQueue = DispatchQueue(
    label: "com.mongars.coreml.stateful-prediction"
  )

  init(modelURL: URL, tokenizer: any Tokenizer) async throws {
    let configuration = MLModelConfiguration()
    // This artifact's published Swift reference runtime is validated on
    // CPU/GPU. Core ML remains free to partition supported operations there.
    configuration.computeUnits = .cpuAndGPU
    model = try await MLModel.load(
      contentsOf: modelURL,
      configuration: configuration
    )
    self.tokenizer = tokenizer
    try Self.validate(model: model)
  }

  func generate(
    promptTokens: [Int],
    requestedOptions: GenerationOptions,
    onUpdate: @escaping @Sendable (GenerationUpdate) async -> Void
  ) async throws -> GenerationResult {
    try Task.checkCancellation()
    try requestedOptions.validate()
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
    try options.validate()

    let state = model.makeState()
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

      let logitsView = try LogitsView(currentLogits)
      let token = try Sampler.select(
        vocabularySize: MonGARSModelManifest.vocabularySize,
        generatedTokens: generatedSet,
        options: options
      ) { tokenID in
        logitsView.score(tokenID: tokenID)
      }
      try Task.checkCancellation()

      if MonGARSModelManifest.eosTokenIDs.contains(token) {
        finishReason = "eos"
        break
      }

      generated.append(token)
      generatedSet.insert(token)
      let text = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
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
      try Task.checkCancellation()

      // The logits produced for the final requested token would never be read.
      if generated.count == options.maxNewTokens { break }

      let endStep = promptTokens.count + generated.count
      guard endStep < MonGARSModelManifest.contextLength else {
        finishReason = "context"
        break
      }
      currentLogits = try predict(
        tokens: [token],
        endStep: endStep,
        state: state
      )
    }
    try Task.checkCancellation()

    let finalText = tokenizer.decode(
      tokens: generated,
      skipSpecialTokens: true
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
      try Task.checkCancellation()
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
  ) async throws -> MLMultiArray {
    var logits: MLMultiArray?

    for range in DolphinRuntimeContract.prefillRanges(
      tokenCount: promptTokens.count
    ) {
      try Task.checkCancellation()
      let nextLogits = try predict(
        tokens: Array(promptTokens[range]),
        endStep: range.upperBound,
        state: state
      )
      if range.upperBound == promptTokens.count {
        logits = nextLogits
      }
      await Task.yield()
      try Task.checkCancellation()
    }

    guard let logits else {
      throw InferenceError.invalidModel("Aucun logit produit apres le prefill.")
    }
    return logits
  }

  private func predict(
    tokens: [Int],
    endStep: Int,
    state: MLState
  ) throws -> MLMultiArray {
    try Task.checkCancellation()
    guard
      !tokens.isEmpty,
      tokens.count <= MonGARSModelManifest.maximumQueryLength,
      endStep >= tokens.count,
      endStep <= MonGARSModelManifest.contextLength
    else {
      throw InferenceError.invalidModel("Dimensions de requete Core ML invalides.")
    }

    let inputIDs = try makeInputIDs(tokens)
    let causalMask = try makeCausalMask(
      queryLength: tokens.count,
      endStep: endStep
    )
    let provider = try MLDictionaryFeatureProvider(dictionary: [
      "inputIds": inputIDs,
      "causalMask": causalMask,
    ])
    let options = MLPredictionOptions()

    let output = try runPrediction(
      provider: provider,
      state: state,
      options: options
    )
    guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
      throw InferenceError.invalidModel("Sortie logits absente.")
    }
    try Self.validate(logits: logits, queryLength: tokens.count)
    return logits
  }

  private func runPrediction(
    provider: MLDictionaryFeatureProvider,
    state: MLState,
    options: MLPredictionOptions
  ) throws -> MLFeatureProvider {
    try Task.checkCancellation()
    let output = try predictionQueue.sync {
      try model.prediction(
        from: provider,
        using: state,
        options: options
      )
    }
    // A synchronous Core ML prediction cannot be interrupted while executing.
    // Drop its partially advanced state immediately when control returns.
    try Task.checkCancellation()
    return output
  }

  private func makeInputIDs(_ tokens: [Int]) throws -> MLMultiArray {
    let result = try MLMultiArray(
      shape: [1, NSNumber(value: tokens.count)],
      dataType: .int32
    )
    let strides = result.strides.map(\.intValue)
    let pointer = result.dataPointer.bindMemory(
      to: Int32.self,
      capacity: result.count
    )
    for (index, token) in tokens.enumerated() {
      guard let value = Int32(exactly: token) else {
        throw InferenceError.invalidModel("Identifiant de jeton hors Int32.")
      }
      pointer[index * strides[1]] = value
    }
    return result
  }

  private func makeCausalMask(
    queryLength: Int,
    endStep: Int
  ) throws -> MLMultiArray {
    guard queryLength > 0, endStep >= queryLength else {
      throw InferenceError.invalidModel("Dimensions du masque causal invalides.")
    }
    let result = try MLMultiArray(
      shape: [1, 1, NSNumber(value: queryLength), NSNumber(value: endStep)],
      dataType: .float16
    )
    let strides = result.strides.map(\.intValue)
    let pointer = result.dataPointer.bindMemory(
      to: UInt16.self,
      capacity: result.count
    )
    for row in 0..<queryLength {
      for column in 0..<endStep {
        let offset = row * strides[2] + column * strides[3]
        // IEEE-754 binary16: +0 for visible positions, -65504 otherwise.
        pointer[offset] = DolphinRuntimeContract.causalMaskAllows(
          row: row,
          column: column,
          queryLength: queryLength,
          endStep: endStep
        ) ? 0x0000 : 0xFBFF
      }
    }
    return result
  }

  private static func validate(model: MLModel) throws {
    let description = model.modelDescription
    let inputIDs = description.inputDescriptionsByName["inputIds"]
    let causalMask = description.inputDescriptionsByName["causalMask"]
    guard
      description.inputDescriptionsByName.count == 2,
      inputIDs?.multiArrayConstraint?.dataType == .int32,
      inputIDs?.multiArrayConstraint?.shape.count == 2,
      inputIDs?.multiArrayConstraint?.shape.first?.intValue == 1,
      causalMask?.multiArrayConstraint?.dataType == .float16,
      causalMask?.multiArrayConstraint?.shape.count == 4
    else {
      throw InferenceError.invalidModel("Entrees Dolphin Core ML invalides.")
    }

    for stateName in ["keyCache", "valueCache"] {
      guard
        let state = description.stateDescriptionsByName[stateName],
        let constraint = state.stateConstraint,
        constraint.bufferShape.map(\.intValue) == MonGARSModelManifest.kvCacheShape,
        constraint.dataType == .float16
      else {
        throw InferenceError.invalidModel("Etat \(stateName) absent ou invalide.")
      }
    }

    guard
      description.stateDescriptionsByName.count == 2,
      let output = description.outputDescriptionsByName["logits"],
      let outputConstraint = output.multiArrayConstraint,
      outputConstraint.dataType == .float16
    else {
      throw InferenceError.invalidModel("Sortie logits Dolphin invalide.")
    }

    let metadata = description.metadata[
      MLModelMetadataKey.creatorDefinedKey
    ] as? [String: String]
    guard
      metadata?["com.ales27pm.dolphin.source_revision"]
        == MonGARSModelManifest.sourceRevision,
      metadata?["com.ales27pm.dolphin.max_context_length"]
        == String(MonGARSModelManifest.contextLength),
      metadata?["com.ales27pm.dolphin.max_query_length"]
        == String(MonGARSModelManifest.maximumQueryLength)
    else {
      throw InferenceError.invalidModel("Metadonnees Dolphin incompatibles.")
    }
  }

  private static func validate(
    logits: MLMultiArray,
    queryLength: Int
  ) throws {
    let shape = logits.shape.map(\.intValue)
    guard
      logits.dataType == .float16,
      shape == [1, queryLength, MonGARSModelManifest.vocabularySize]
    else {
      throw InferenceError.invalidModel(
        "Sortie logits ne respecte pas [1, \(queryLength), "
          + "\(MonGARSModelManifest.vocabularySize)]."
      )
    }
  }

}
#endif
