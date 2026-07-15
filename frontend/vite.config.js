import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: Vite serves the SPA on :5173 and proxies API + static assets (PDFs,
// logo) to the Flask backend on :8080. Build: emits to dist/ for Flask to serve.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8080', changeOrigin: true },
      '/static': { target: 'http://127.0.0.1:8080', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // One stylesheet instead of a CSS file per route. Per-route CSS chunks get
    // "preloaded" by Vite's helper, and a single 429 on that preload (cold Cloud
    // Run instance, min-instances 0) throws "Unable to preload CSS" and blanks the
    // whole app. Bundling all CSS into one file removes that failure mode and the
    // parallel-request burst that triggers the 429 in the first place.
    cssCodeSplit: false,
  },
})
