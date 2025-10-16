import prettierCompat from "./eslint/prettierCompat.js";

export default [
  prettierCompat,
  {
    files: ["webapp/static/js/src/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
    },
    rules: {
      "no-console": "off",
    },
  },
];
