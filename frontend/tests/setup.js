// frontend/tests/setup.js

// make Vue available globally so the test‐utils “browser” build stops complaining
import * as Vue from "vue";
import { renderToString } from "vue/server-renderer";

global.Vue = Vue;
global.VueCompilerDOM = require("@vue/compiler-dom");

// Provide a minimal server‐renderer for VTU
global.VueServerRenderer = { renderToString };
