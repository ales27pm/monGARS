module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: false,
    },
    android: {},
  },
  dependencies: {
    "react-native-vector-icons": {
      platforms: {
        ios: {
          project: "ios/MonGARSMobile.xcodeproj",
        },
      },
    },
  },
  reactNativePath: require.resolve("react-native"),
};
