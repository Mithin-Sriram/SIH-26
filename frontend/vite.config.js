import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev proxy: /api hits the local FastAPI backend, so no CORS needed.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
