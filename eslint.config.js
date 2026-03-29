import prettierCompat from "./eslint/prettierCompat.js";

export default [
  prettierCompat,
  {
    files: ["webapp/static/js/src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      ecmaFeatures: {
        jsx: true,
      },
    },
    rules: {
      "no-console": "off",
    },
  },
];
