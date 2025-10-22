const React = require('react');

const SafeAreaInsetsContext = React.createContext({
  top: 0,
  bottom: 0,
  left: 0,
  right: 0,
});

function SafeAreaProvider({ children }) {
  return React.createElement(
    SafeAreaInsetsContext.Provider,
    {
      value: { top: 0, bottom: 0, left: 0, right: 0 },
    },
    children,
  );
}

function SafeAreaView({ children }) {
  return React.createElement('SafeAreaView', null, children);
}

module.exports = {
  SafeAreaProvider,
  SafeAreaView,
  SafeAreaInsetsContext,
  useSafeAreaInsets: () => ({ top: 0, bottom: 0, left: 0, right: 0 }),
  useSafeAreaFrame: () => ({ x: 0, y: 0, width: 320, height: 640 }),
};
