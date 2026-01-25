import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { copyFileSync, readFileSync, writeFileSync } from 'fs'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'copy-manifest',
      writeBundle() {
        copyFileSync(
          resolve(__dirname, 'manifest.json'),
          resolve(__dirname, 'dist/manifest.json')
        )
        
        // Fix paths in popup.html to be relative for Chrome extension
        const htmlPath = resolve(__dirname, 'dist/popup.html')
        let html = readFileSync(htmlPath, 'utf-8')
        // Replace absolute paths with relative paths
        html = html.replace(/href="\/assets\//g, 'href="./assets/')
        html = html.replace(/src="\/assets\//g, 'src="./assets/')
        // Remove crossorigin attribute which can cause issues in Chrome extensions
        html = html.replace(/\s+crossorigin="[^"]*"/g, '')
        html = html.replace(/\s+crossorigin/g, '')
        writeFileSync(htmlPath, html)
      }
    }
  ],
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        popup: resolve(__dirname, 'popup.html')
      }
    }
  }
})
