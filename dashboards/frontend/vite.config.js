import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  // Single-file build: JS/CSS inline into index.html so a rolling deploy can
  // never serve an index.html whose hashed assets live only on the other
  // container version (the request races 404 and the page renders black).
  plugins: [tailwindcss(), svelte(), viteSingleFile()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
