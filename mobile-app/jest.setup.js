jest.setTimeout(20000);

require('react-native-gesture-handler/jestSetup');

jest.mock('react-native-reanimated', () =>
  require('react-native-reanimated/mock'),
);
const ReactNative = require('react-native');
const { NativeModules } = ReactNative;

if (typeof global.localStorage === 'undefined') {
  let store = {};
  global.localStorage = {
    getItem: (key) => (key in store ? store[key] : null),
    setItem: (key, value) => {
      store[key] = value;
    },
    removeItem: (key) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
}

class MockNativeEventEmitter {
  static handlers = {};

  addListener(event, handler) {
    MockNativeEventEmitter.handlers[event] = handler;
    return { remove: jest.fn() };
  }

  removeAllListeners() {
    MockNativeEventEmitter.handlers = {};
  }
}

ReactNative.NativeEventEmitter = MockNativeEventEmitter;

jest.mock(
  'react-native/Libraries/EventEmitter/NativeEventEmitter',
  () => MockNativeEventEmitter,
);

jest.mock('@react-native-async-storage/async-storage', () => {
  let storage = {};
  const mock = {
    setItem: jest.fn(async (key, value) => {
      storage[key] = String(value);
    }),
    getItem: jest.fn(async (key) =>
      key in storage ? String(storage[key]) : null,
    ),
    removeItem: jest.fn(async (key) => {
      delete storage[key];
    }),
    clear: jest.fn(async () => {
      storage = {};
    }),
  };
  return { __esModule: true, default: mock, ...mock };
});

NativeModules.RNGestureHandlerModule = NativeModules.RNGestureHandlerModule || {
  State: {},
  Directions: {},
};

NativeModules.PlatformConstants = NativeModules.PlatformConstants || {
  forceTouchAvailable: false,
};

jest.mock('./src/native/diagnostics', () => ({
  prepare: jest.fn().mockResolvedValue(undefined),
  refreshNetworkSnapshot: jest.fn().mockResolvedValue({
    timestamp: new Date().toISOString(),
    ssid: null,
    bssid: null,
    ip: null,
    interfaces: [],
    vpnActive: false,
    cellularActive: false,
  }),
  listInterfaces: jest.fn().mockResolvedValue([]),
  startCapture: jest
    .fn()
    .mockResolvedValue({ captureId: 'test', path: '/tmp/test.pcap' }),
  stopCapture: jest
    .fn()
    .mockResolvedValue({ captureId: 'test', path: '/tmp/test.pcap' }),
  enablePacketTunnel: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('./src/native/voice', () => ({
  configureAudioSession: jest.fn().mockResolvedValue(undefined),
  startListening: jest.fn().mockResolvedValue(undefined),
  stopListening: jest.fn().mockResolvedValue(undefined),
  setOnResultListener: jest.fn(),
  removeOnResultListener: jest.fn(),
}));

jest.mock('react-native-config', () => ({
  MONGARS_API_URL: 'https://localhost:8443/api/v1',
  MONGARS_WS_URL: 'wss://localhost:8443/ws/chat',
  MONGARS_VOICE_LOCALE: 'fr-FR',
}));
