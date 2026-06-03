// Root ESLint flat config for the repo
// - Lints frontend Vue + JS (using plugin:vue and Prettier integration)
// - Lints plain JS in analysis/static
// - Provides Jest globals for frontend tests

import js from "@eslint/js";
import prettierConfig from "eslint-config-prettier";
import jestPlugin from "eslint-plugin-jest";
import sonarjsPlugin from "eslint-plugin-sonarjs";
import vuePlugin from "eslint-plugin-vue";
import globals from "globals";

export default [
  // Base recommended rules for all JS
  js.configs.recommended,

   // SonarJS plugin for code quality checks
  {
    plugins: {
      sonarjs: sonarjsPlugin,
    },
    rules: {
      "sonarjs/cognitive-complexity": ["warn", 15],
      "sonarjs/no-identical-functions": "warn",
      "sonarjs/no-duplicate-string": ["warn", { threshold: 3 }],
    },
  },

  // Apply Vue and Prettier integration using flat configs
  ...vuePlugin.configs["flat/recommended"],
  prettierConfig,

  // Global ignores
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "analysis/figures/**",
      "logs/**",
      "results/**",
      "env-continuouspersuasion/**",
    ],
  },

  // Frontend application sources
  {
    files: ["frontend/src/**/*.{js,vue}"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
        process: "readonly",
      },
    },
  },

  // Frontend tests (Jest)
  {
    files: [
      "frontend/tests/**/*.js",
      "frontend/**/*.spec.js",
      "frontend/**/*.test.js",
    ],
    // inject the plugin itself so you can use its rules
    plugins: {
      jest: jestPlugin,
    },
    settings: {
      jest: {
        // Lock the Jest version to avoid auto-detect issues when running from repo root
        version: 29,
      },
    },
    // enable all of the Jest globals (describe, it, expect, jest, etc.)
    languageOptions: {
      globals: {
        ...globals.jest,
        ...globals.node,
      },
    },
  },

  // Analysis JS (plain browser JS)
  {
    files: ["analysis/static/**/*.js"],
    languageOptions: {
      globals: {
        ...globals.browser,
      },
    },
  },
];
