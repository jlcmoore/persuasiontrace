// frontend/babel.config.cjs

module.exports = {
  presets: [["@babel/preset-env", { targets: { node: "current" } }]],
  plugins: [
    [
      "babel-plugin-transform-define",
      {
        "import.meta.env.VITE_TURNSTILE_SITE_KEY": "FIXED_KEY",
      },
    ],
  ],
};
