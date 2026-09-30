import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
  const backendPort = process.env.BACKEND_PORT || '8001';
  const target = `http://127.0.0.1:${backendPort}`;
  return {
    plugins: [react()],
    server: {
      port: 3000,
      proxy: {
        '/score': target,
        '/alerts': target,
        '/graph': target,
        '/admin': target,
        '/health': target,
      },
    },
  };
});
