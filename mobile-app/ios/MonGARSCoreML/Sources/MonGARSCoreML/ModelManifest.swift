import Foundation

public enum MonGARSModelManifest {
  public static let modelID = "ales27pm/Dolphin3.0-CoreML"
  public static let revision = "95671cf9a2f56d2a381816ae264cd9aae335d96f"
  public static let sourceModelID = "dphn/Dolphin3.0-Llama3.2-3B"
  public static let sourceRevision = "392a6f57223e7ccfe6ef4ebdb2ff101a42d57364"

  public static let displayName = "Dolphin 3.0 · Llama 3.2 3B · Core ML INT4"
  public static let packageDirectory =
    "Dolphin3.0-Llama3.2-3B-stateful-int4.mlpackage"
  public static let compiledDirectory =
    "Dolphin3.0-Llama3.2-3B-stateful-int4.mlmodelc"
  public static let contextLength = 2_048
  public static let maximumQueryLength = 512
  public static let vocabularySize = 128_258
  public static let kvCacheShape = [28, 1, 8, 2_048, 128]
  public static let defaultMaxNewTokens = 96
  public static let maximumNewTokens = 192
  public static let downloadBytes: Int64 = 1_825_812_981
  // Runtime compilation keeps both the verified source package and a derived
  // mlmodelc. Leave enough room for both plus Core ML compiler scratch space.
  public static let requiredFreeDiskBytes: Int64 = 5_000_000_000
  public static let requiredCompilationFreeDiskBytes: Int64 = 2_500_000_000

  public static let files = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "\(packageDirectory)/Data/com.apple.CoreML/model.mlmodel",
    "\(packageDirectory)/Data/com.apple.CoreML/weights/weight.bin",
    "\(packageDirectory)/Manifest.json",
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
      bytes: 935,
      sha256: "e21ff53ea39726f972362beba869807216775d5e308bc2f531784846c06a0249"
    ),
    ExpectedFile(
      path: "generation_config.json",
      bytes: 206,
      sha256: "e627b5a8b2dc371f90388947ada64fa6e71de0f991c04c835f0c0bc97e305a4f"
    ),
    ExpectedFile(
      path: "special_tokens_map.json",
      bytes: 444,
      sha256: "2df2c4620bb1a9eb877bc7c90c7fa04608bda9fa7c0cf2cdcc0a17b849649683"
    ),
    ExpectedFile(
      path: "tokenizer.json",
      bytes: 17_210_298,
      sha256: "e40b93124a3e29f62d5f4ff41be56cb2af34ecacf9239acd9da53a98860380b5"
    ),
    ExpectedFile(
      path: "tokenizer_config.json",
      bytes: 53_573,
      sha256: "51ad9580aba8d00016efda43357185a0d8ff9884584dcc82ab58ca552afd14e1"
    ),
    ExpectedFile(
      path: "\(packageDirectory)/Data/com.apple.CoreML/model.mlmodel",
      bytes: 809_496,
      sha256: "a34a00a253c98153cf3b231105493edddde532086e224443e40b255b0f10a924"
    ),
    ExpectedFile(
      path: "\(packageDirectory)/Data/com.apple.CoreML/weights/weight.bin",
      bytes: 1_807_737_412,
      sha256: "6240edc377b1a0158812454c4bb6e3053d8e8a75a7eedb751b9740fffdfd3e15"
    ),
    ExpectedFile(
      path: "\(packageDirectory)/Manifest.json",
      bytes: 617,
      sha256: "5b8ac347a822f02ba3a6d9ccff60dd723f2649424c8e88570961f12b1c59afb6"
    ),
  ]

  public static let eosTokenIDs: Set<Int> = [128_256, 128_001, 128_008, 128_009]

  public static let systemPrompt = """
    Tu es monGARS, un assistant personnel local, fiable et direct. Reponds dans la langue de \
    l'utilisateur. Si tu ne sais pas, dis-le clairement. Tes reponses restent sur cet iPhone.
    """
}
