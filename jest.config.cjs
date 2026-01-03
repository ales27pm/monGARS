module.exports = {
  testEnvironment: "jsdom",
  roots: ["<rootDir>/webapp/static/js/src"],
  testMatch: ["**/__tests__/**/*.test.js"],
  transform: {},
  collectCoverageFrom: ["webapp/static/js/src/services/**/*.js"],
  coverageReporters: ["text", "lcov", "json-summary"],
  coverageThreshold: {
    global: {
      branches: 15,
      functions: 14,
      lines: 12,
      statements: 12,
    },
  },
};
