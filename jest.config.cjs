module.exports = {
  testEnvironment: "jsdom",
  roots: ["<rootDir>/webapp/static/js/src"],
  testMatch: ["**/__tests__/**/*.test.js"],
  transform: {},
  collectCoverageFrom: ["webapp/static/js/src/services/**/*.js"],
};
