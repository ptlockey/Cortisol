import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "build",
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: "src/index.jsx"
    }
  },
  server: {
    port: 3001,
    strictPort: true
  }
});
