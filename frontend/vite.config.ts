import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const apiTarget = process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
          router: ['react-router-dom'],
        },
      },
    },
  },
  server: {
    proxy: Object.fromEntries(
      ['/health', '/models', '/predict', '/api', '/result-figures'].map((path) => [
        path,
        { target: apiTarget, changeOrigin: true },
      ]),
    ),
  },
});