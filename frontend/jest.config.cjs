// frontend/jest.config.cjs

module.exports = {
  testEnvironment: "jsdom",
  moduleFileExtensions: ["js", "json", "vue"],
  transform: {
    // use the Vue 3 transformer
    "^.+\\.vue$": [
      "@vue/vue3-jest",
      {
        babelConfig: true,
      },
    ],
    "^.+\\.js$": "babel-jest",
  },
  moduleNameMapper: {
    "\\.(svg|png|jpg|jpeg|gif)$": "<rootDir>/tests/unit/fileMock.js",
    "\\.(css|less|scss)$": "<rootDir>/tests/unit/styleMock.js",
    // map the test-utils package to the CJS build:
    "^@vue/test-utils$":
      "<rootDir>/../node_modules/@vue/test-utils/dist/vue-test-utils.cjs.js",
    // This has to be last so it doesn't clobber the other mappings
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  testMatch: ["**/tests/unit/**/*.spec.[jt]s?(x)"],
  setupFiles: ["<rootDir>/tests/setup.js"],
};
