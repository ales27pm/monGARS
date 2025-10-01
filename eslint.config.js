import prettier from "eslint-config-prettier";

export default [
  prettier,
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
