const path = require('path');

module.exports = {
  project: {
    ios: {
      automaticPodsInstallation: false,
    },
    android: {},
  },
  reactNativePath: path.dirname(require.resolve('react-native/package.json')),
};
