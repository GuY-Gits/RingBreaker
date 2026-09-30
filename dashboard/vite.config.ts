/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The built dashboard is served by the API at /app; in dev, Vite proxies /api.
export default defineConfig(() => {
  const target = `http://127.0.0.1:${process.env.BACKEND_PORT || '8001'}`;
  return {
    base: '/app/',
    plugins: [react()],
    server: { port: 5173, proxy: { '/api': target } },
    test: { environment: 'jsdom', globals: false },
  };
});
