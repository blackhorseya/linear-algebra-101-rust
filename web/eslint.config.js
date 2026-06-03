import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // 自動產生物不檢查:routeTree.gen.ts(TanStack Router)、src/lib/wasm(wasm-pack)
  globalIgnores(['dist', 'src/routeTree.gen.ts', 'src/lib/wasm']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
  },
  {
    // file-based route 檔依慣例要 export `Route` 物件,HMR 由 router-plugin
    // 自行接管,這條保護 Fast Refresh 的規則在這裡不適用,關掉即可。
    files: ['src/routes/**/*.tsx'],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
])
