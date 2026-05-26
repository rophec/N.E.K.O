import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))

// 输出 IIFE 到 ../../static/react/card-forge/，由 NEKO 主应用 templates/card_forge.html 引用。
// dev 模式下走 5174 端口；/forge/* 被 proxy 到 forge_server (3002)。
// /battle-arena/avatar 仍走主应用，复用头像接口。
export default defineConfig(({ command }) => ({
  plugins: [react()],
  server: {
    proxy: {
      '/forge': {
        target: 'http://localhost:3002',
        changeOrigin: true,
      },
      '/battle-arena/avatar': {
        target: 'http://localhost:48911',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolve(__dirname, '../../static/react/card-forge'),
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {
      entry: resolve(__dirname, 'src/main.jsx'),
      name: 'NekoCardForge',
      formats: ['iife'],
      fileName: () => 'card-forge.iife.js',
    },
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        assetFileNames: (asset) => {
          if (asset.name && asset.name.endsWith('.css')) return 'card-forge.css'
          return 'assets/[name]-[hash][extname]'
        },
      },
    },
  },
}))
