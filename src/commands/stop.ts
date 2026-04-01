import { loadConfig, assertConfigured } from "../config.ts";

export async function cmdStop(_args: string[]): Promise<void> {
  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }
  console.log("ivllm stop — not yet implemented");
}
