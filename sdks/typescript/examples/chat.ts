import { MonGARSClient } from "../src/index.js";

async function main() {
  const client = new MonGARSClient({
    baseUrl: process.env.MONGARS_BASE_URL ?? "http://localhost:8000",
  });
  await client.login({
    username: process.env.MONGARS_USERNAME ?? "u1",
    password: process.env.MONGARS_PASSWORD ?? "x",
  });

  const response = await client.chat({ message: "Hello from TypeScript!" });
  console.log("Assistant:", response.response);
}

main().catch((error) => {
  console.error("Failed to run chat example", error);
  process.exit(1);
});
