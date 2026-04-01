import { loadConfig, assertConfigured } from "../config.ts";

export async function cmdStart(_args: string[]): Promise<void> {
  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }
  console.log("ivllm start — not yet implemented");
}
