import Foundation

public enum MonGARSModelManifest {
  public static let modelID = "mariocde/Qwen3-1.7B-CoreML-LUT6"
  public static let revision = "51c5bc038afa962216e3880bf870e92b219328e6"
  public static let sourceModelID = "Qwen/Qwen3-1.7B"
  public static let sourceRevision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

  public static let displayName = "Qwen3 1.7B · Core ML LUT6"
  public static let compiledDirectory = "qwen_monolithic_full_lut6.mlmodelc"
  public static let contextLength = 512
  public static let vocabularySize = 151_936
  public static let logitsChunkCount = 16
  public static let logitsChunkSize = 9_496
  public static let kvCacheShape = [56, 8, 512, 128]
  public static let defaultMaxNewTokens = 96
  public static let maximumNewTokens = 192
  public static let installedBytes: Int64 = 1_566_306_707
  public static let requiredFreeDiskBytes: Int64 = 2_500_000_000

  public static let files = [
    "config.json",
    "meta.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
    "\(compiledDirectory)/analytics/coremldata.bin",
    "\(compiledDirectory)/coremldata.bin",
    "\(compiledDirectory)/metadata.json",
    "\(compiledDirectory)/model.mil",
    "\(compiledDirectory)/weights/weight.bin",
  ]

  public struct ExpectedFile: Sendable, Equatable {
    public let path: String
    public let bytes: Int64
    public let sha256: String

    public init(path: String, bytes: Int64, sha256: String) {
      self.path = path
      self.bytes = bytes
      self.sha256 = sha256
    }
  }

  public static let expectedFiles = [
    ExpectedFile(
      path: "config.json",
      bytes: 67,
      sha256: "4d94b801ff5bfcf173898ec2a47988f9fec3a93a621200875d9cc732f5bf9a20"
    ),
    ExpectedFile(
      path: "meta.yaml",
      bytes: 1_856,
      sha256: "e1d32a5aa4ca503d51724c194c524c6939ca7c2705a83f274982d389178dee39"
    ),
    ExpectedFile(
      path: "tokenizer.json",
      bytes: 11_422_654,
      sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    ),
    ExpectedFile(
      path: "tokenizer_config.json",
      bytes: 9_732,
      sha256: "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    ),
    ExpectedFile(
      path: "\(compiledDirectory)/analytics/coremldata.bin",
      bytes: 243,
      sha256: "d979784eba8948f332cd4c355fb00d047b4d98c744d3eebb852efaa2e29a812a"
    ),
    ExpectedFile(
      path: "\(compiledDirectory)/coremldata.bin",
      bytes: 1_797,
      sha256: "e0309bc6a03ee8e4437b5ba0a5162f500d8ad06bb40fa37006872c5804f92bcd"
    ),
    ExpectedFile(
      path: "\(compiledDirectory)/metadata.json",
      bytes: 25_227,
      sha256: "5346a6d41d46cf920bd9a2d050746ccdd52a3d7116a10cb5eb594b008e944361"
    ),
    ExpectedFile(
      path: "\(compiledDirectory)/model.mil",
      bytes: 2_252_363,
      sha256: "bc1229df0107d8be2410cf4d5803a3817357bbbc1a8e92203b837d43a4b2836e"
    ),
    ExpectedFile(
      path: "\(compiledDirectory)/weights/weight.bin",
      bytes: 1_552_592_768,
      sha256: "59f0f1b1732c48d0c67e4f38734e36310e9c9b2d903e3de287751d68d46d5588"
    ),
  ]

  public static let eosTokenIDs: Set<Int> = [151_643, 151_645]

  public static let systemPrompt = """
    Tu es monGARS, un assistant personnel local, fiable et direct. Reponds dans la langue de \
    l'utilisateur. Si tu ne sais pas, dis-le clairement. Tes reponses restent sur cet iPhone.
    """
}
