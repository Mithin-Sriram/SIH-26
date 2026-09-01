import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    // Build straight into the backend so FastAPI serves the whole app
    // from a single origin/port (see backend/app/main.py).
    outDir: '../static',
    emptyOutDir: true,
  },
})
