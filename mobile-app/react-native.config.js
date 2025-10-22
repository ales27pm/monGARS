const path = require('path');

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
  reactNativePath: path.dirname(require.resolve('react-native/package.json')),
};
