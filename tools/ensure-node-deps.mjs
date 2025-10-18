import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const requiredPackages = ["jest", "eslint"];

const missingPackages = requiredPackages.filter((pkg) => {
  const packageJsonPath = path.join(
    projectRoot,
    "node_modules",
    pkg,
    "package.json",
  );
  return !existsSync(packageJsonPath);
});

if (missingPackages.length === 0) {
  process.exit(0);
}

const installCommand = process.platform === "win32" ? "npm.cmd" : "npm";

console.log(
  `Installing npm dependencies because ${missingPackages.join(", ")} package(s) were not found in node_modules.`,
);

const result = spawnSync(installCommand, ["install"], {
  cwd: projectRoot,
  stdio: "inherit",
  env: process.env,
});

if (result.status !== 0) {
  console.error("npm install failed; aborting.");
  process.exit(result.status ?? 1);
}
