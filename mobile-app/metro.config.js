const { getDefaultConfig, mergeConfig } = require("@react-native/metro-config");

const config = {
  transformer: {
    babelTransformerPath: require.resolve("react-native-svg-transformer"),
  },
  resolver: {
    assetExts: ["db", "png", "jpg", "ttf", "otf", "mp3", "wav"],
    sourceExts: ["ts", "tsx", "js", "jsx", "json"],
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
