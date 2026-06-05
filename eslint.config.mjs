// @ts-check

import js from '@eslint/js';
import tseslint from 'typescript-eslint';
// 1. Import the Prettier config helper
import eslintConfigPrettier from 'eslint-config-prettier';
// 1. Import the JSDoc plugin
import jsdoc from 'eslint-plugin-jsdoc';
import { readFileSync } from 'node:fs';

// 2. Read, split, and clean the .prettierignore entries
const prettierIgnores = readFileSync('.prettierignore', 'utf8')
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith('#'));

export default tseslint.config(
  {
    ignores: prettierIgnores,
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  // 2. Add the JSDoc TypeScript-recommended configuration
  jsdoc.configs['flat/recommended-typescript'],
  {
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
      // Ensures @param tags exist for every function parameter
      'jsdoc/require-param': 'error',
      // Ensures documented @param names match the code variables
      'jsdoc/check-param-names': 'error',
      // Ensures @returns documentation matches actual return values
      'jsdoc/require-returns': 'error',
    },
  },
  // 2. Always append this last to disable conflicting rules
  eslintConfigPrettier,
);
