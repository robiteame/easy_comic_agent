import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src/renderer'),
    },
  },
  server: {
    port: 5173,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (
            id.includes('antd') ||
            id.includes('@ant-design/icons') ||
            id.includes('@rc-component') ||
            id.includes('/rc-')
          )
            return 'antd'
          if (id.includes('/react/') || id.includes('/react-dom/')) return 'react'
          if (id.includes('/axios/')) return 'axios'
        },
      },
    },
  },
})
