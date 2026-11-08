import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiPort = Number(process.env.DERMAGUARD_API_PORT || 8000);
if (!Number.isInteger(apiPort) || apiPort < 1 || apiPort > 65535) {
  throw new Error("DERMAGUARD_API_PORT must be a valid port number");
}
const proxy = {
  "/v1": `http://127.0.0.1:${apiPort}`,
  "/health": `http://127.0.0.1:${apiPort}`,
};

export default defineConfig({
  plugins: [react()],
  build: { target: "es2018", cssTarget: "safari14", emptyOutDir: false },
  server: {
    host: "0.0.0.0",
    port: 8501,
    strictPort: true,
    proxy,
  },
  preview: { proxy },
});
