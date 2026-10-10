import path from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^zustand$/, replacement: path.resolve(__dirname, './src/renderer/test-support/zustandSsrBinding.mjs') },
      { find: '@', replacement: path.resolve(__dirname, './src/renderer') },
    ],
  },
  esbuild: {
    target: 'es2020',
  },
  test: {
    environment: 'node',
    include: ['*.test.mts', 'src/**/*.test.mts', 'scripts/**/*.test.mts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary', 'html'],
      include: ['src/**/*.{ts,tsx,mts}', 'scripts/**/*.{mjs,ts,mts}'],
      exclude: ['**/*.test.mts', 'src/renderer/test-support/**'],
    },
  },
})
