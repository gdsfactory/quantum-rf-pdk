export default [
  {
    languageOptions: {
      globals: {
        console: "readonly",
        document: "readonly",
        DOMParser: "readonly",
        fetch: "readonly",
      },
    },
    rules: {
      "no-constant-condition": "error",
      "no-debugger": "error",
      "no-undef": "error",
      "no-unreachable": "error",
      "no-unused-vars": "error",
    },
  },
];
