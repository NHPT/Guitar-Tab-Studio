import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const desktop = mode === 'desktop'
  return {
    base: './',
    plugins: [react()],
    build: {
      outDir: desktop ? 'dist-desktop' : 'dist',
      rollupOptions: desktop
        ? {
            input: fileURLToPath(new URL('./desktop.html', import.meta.url)),
          }
        : undefined,
    },
  }
})
