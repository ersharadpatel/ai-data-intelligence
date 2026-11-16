import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// For project pages: https://USER.github.io/REPO/ set VITE_BASE=/REPO/
const base = process.env.VITE_BASE || '/'

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
