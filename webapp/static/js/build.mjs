import { build } from "esbuild";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const entry = path.resolve(__dirname, "src/index.js");
const outfile = path.resolve(__dirname, "chat.js");

const isProd = process.env.NODE_ENV === "production";

await build({
  entryPoints: [entry],
  bundle: true,
  outfile,
  format: "iife",
  platform: "browser",
  target: ["es2018"],
  sourcemap: isProd ? false : "inline",
  minify: isProd,
  logLevel: "info",
});
