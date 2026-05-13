import { loadConfig } from "../config.ts";
import {
  getAvailableAssistants,
  getAvailableWrappers,
  getAssistantLabel,
  buildLaunchCommand,
  ensureSbxSandbox,
  generatePiModelsConfig,
  getScoderAvailable,
  getSbxAvailable
} from "../assistant.ts";
import { spawnSync } from "child_process";
import { join } from "path";
import { existsSync, mkdirSync, writeFileSync, copyFileSync, readFileSync } from "fs";

interface V1ModelsResponse {
  object: "list";
  data: Array<{
    id: string;
    [key: string]: any;
  }>;
}

/**
 * Handle the ivllm agent command for launching AI assistants connected to a local vLLM server
 */
export async function cmdAgent(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: ivllm agent [options] <assistant>

Options:
  --port <port>       Port of the local vLLM server (default: 11434 or from ivllm config)
  --help, -h          Show this help message

Assistants:
  opencode            Launch OpenCode assistant
  pi                  Launch Pi assistant (updates ~/.pi/agent/models.json)
  copilot             Launch GitHub Copilot assistant
  claude              Launch Claude Code assistant

Examples:
  ivllm agent --port 11434 opencode
  ivllm agent pi
  ivllm agent --port 8000 claude
`);
    return;
  }

  // Parse port argument
  let port: number | undefined;
  let assistantIndex = 0;
  
  // Look for --port=<value> or --port <value>
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--port=")) {
      port = parseInt(args[i].split("=")[1], 10);
      args.splice(i, 1);
      break;
    } else if (args[i] === "--port" && i + 1 < args.length) {
      port = parseInt(args[i + 1], 10);
      args.splice(i, 2);
      break;
    }
  }
  
  // If port not specified, get from config or use default
  if (port === undefined) {
    try {
      const config = loadConfig();
      port = config.defaultLocalPort ?? 11434;
    } catch (error) {
      // If config loading fails, use default
      port = 11434;
    }
  }
  
  // Get assistant name (first non-flag argument)
  const assistantArg = args[0];
  if (!assistantArg) {
    console.error("Error: Assistant name required");
    console.error("Usage: ivllm agent [--port=<port>] <opencode|pi|copilot|claude>");
    process.exit(1);
  }
  
  const assistantName = assistantArg as "opencode" | "pi" | "copilot" | "claude";
  
  // Query model info from vLLM server
  console.log(`🔍 Querying localhost:${port}/v1/models for available models...`);
  let modelId: string;
  try {
    const response = await fetch(`http://localhost:${port}/v1/models`, {
      // Note: In Bun, fetch is available globally
      // Adding timeout would require additional wrapper, but we'll keep it simple for now
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const json: V1ModelsResponse = await response.json();
    
    if (json.object !== "list" || !Array.isArray(json.data) || json.data.length === 0) {
      throw new Error("Invalid response format: expected non-empty data array");
    }
    
    if (!json.data[0].id) {
      throw new Error("Invalid response format: missing model id in first item");
    }
    
    modelId = json.data[0].id;
    console.log(`📦 Found model: ${modelId}`);
  } catch (error) {
    console.error(`❌ Failed to query model info: ${error.message}`);
    console.error("   Make sure a vLLM server is running and accessible at localhost:${port}/v1");
    process.exit(1);
  }
  
  // Validate assistant is available
  const availableAssistants = getAvailableAssistants();
  if (!availableAssistants.includes(assistantName)) {
    console.error(`❌ ${getAssistantLabel(assistantName)} not available. Install it first.`);
    process.exit(1);
  }
  
  // Prepare launch options (reasoning and toolCall default to true as requested)
  const launchOpts = {
    model: modelId,
    localPort: port,
    maxModelLen: undefined, // We could extract this from model permissions if needed
    toolCall: true,
    reasoning: true
  };
  
  // Launch the assistant
  await launchAssistant({
    ...launchOpts,
    assistant: assistantName,
    shutdown: () => process.exit(0)
  });
}

/**
 * Launch the specified assistant with the given options
 */
async function launchAssistant(opts: {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall: boolean;
  reasoning: boolean;
  assistant: "opencode" | "pi" | "copilot" | "claude";
  shutdown: () => void;
}): Promise<void> {
  let cwd = process.cwd();
  const availableAssistants = getAvailableAssistants();
  const hasScoder = getScoderAvailable();
  const hasSbx = getSbxAvailable();
  
  // Show assistant menu (similar to start.ts but simplified for agent command)
  while (true) {
    const targetChoice = await promptAssistantMenu(
      `\n🤖 AI assistant launcher\n📍 Working directory: ${cwd}\n\nChoose an assistant target:\n`,
      [
        { key: "opencode", label: "OpenCode", input: "1" },
        { key: "pi", label: "Pi", input: "2" },
        { key: "copilot", label: "GitHub Copilot", input: "3" },
        { key: "claude", label: "Claude Code", input: "4" },
        { key: "change-dir", label: "Change directory", input: "d" },
        { key: "shutdown", label: "Shutdown agent launcher", input: "0" },
      ] as const
    );
    
    if (targetChoice === "change-dir") {
      const newDir = await promptInput("\nEnter directory path (or press Enter to keep current): ");
      if (!newDir) continue;
      
      const nextCwd = require("path").resolve(newDir);
      if (!existsSync(nextCwd)) {
        console.log(`⚠️  Directory not found: ${newDir}. Keeping current directory.\n`);
        continue;
      }
      cwd = nextCwd;
      console.log(`✅ Changed directory to: ${cwd}\n`);
      continue;
    }
    
    if (targetChoice === "shutdown") {
      opts.shutdown();
      return;
    }
    
    const assistant = targetChoice;
    const wrappers = getAvailableWrappers(assistant, availableAssistants, hasScoder, hasSbx);
    
    if (wrappers.length === 0) {
      console.log(`\n⚠️  No launch wrappers are available for ${getAssistantLabel(assistant)}.`);
      console.log("Install the local assistant binary for direct/scoder launch, or install sbx for sandbox launch.\n");
      continue;
    }
    
    const wrapperChoice = await promptWrapperMenu(
      `\n🎯 Target: ${getAssistantLabel(assistant)}\n📍 Working directory: ${cwd}\n\nChoose a wrapper:\n`,
      [
        ...wrappers.map((wrapper, index) => ({
          key: wrapper,
          label: wrapper === "none" ? "Direct launch" : wrapper.toUpperCase(),
          input: String(index + 1),
        })),
        { key: "back", label: "Back", input: "0" },
      ] as const
    );
    
    if (wrapperChoice === "back") continue;
    
    const wrapper = wrapperChoice;
    const action = await promptActionMenu(
      `\n🚀 ${getAssistantLabel(assistant)} via ${wrapper === "none" ? "direct launch" : wrapper.toUpperCase()}\n\nChoose an action:\n`,
      [
        { key: "launch", label: "Launch now", input: "1" },
        { key: "show", label: "Show copy-paste command", input: "2" },
        { key: "back", label: "Back", input: "0" },
      ] as const
    );
    
    if (action === "back") continue;
    
    if (action === "show") {
      const launchCommand = buildLaunchCommand({
        assistant,
        wrapper,
        cwd,
        model: opts.model,
        localPort: opts.localPort,
        maxModelLen: opts.maxModelLen,
        toolCall: opts.toolCall,
        reasoning: opts.reasoning,
        sandboxName: wrapper === "sbx" ? 
          (opts.assistant === "pi" ? "pi-isambard-vllm" : `${opts.assistant}-${require("path").basename(cwd)}`) : 
          undefined
      });
      
      console.log("\n📋 Command:");
      console.log(launchCommand);
      console.log("");
      continue;
    }
    
    // Launch the assistant
    try {
      // Special handling for Pi: update models.json before launching
      if (assistant === "pi") {
        await updatePiModelsConfigForLaunch({
          model: opts.model,
          localPort: opts.localPort,
          maxModelLen: opts.maxModelLen,
          toolCall: opts.toolCall,
          reasoning: opts.reasoning
        }, cwd);
      }
      
      // Handle sandbox creation if needed
      let sandboxName: string | undefined;
      if (wrapper === "sbx") {
        const ensured = ensureSbxSandbox(assistant, cwd);
        sandboxName = ensured.sandboxName;
        if (ensured.created) {
          console.log(`✅ Created sandbox: ${sandboxName}`);
        }
      }
      
      // Build and execute launch command
      const launchCommand = buildLaunchCommand({
        assistant,
        wrapper,
        cwd,
        model: opts.model,
        localPort: opts.localPort,
        maxModelLen: opts.maxModelLen,
        toolCall: opts.toolCall,
        reasoning: opts.reasoning,
        sandboxName
      });
      
      console.log(`\n🚀 Launching ${getAssistantLabel(assistant)}...`);
      
      const tmuxResult = spawnSync("tmux", ["new-window", "-n", assistant, "bash", "-lc", launchCommand], {
        stdio: "inherit",
      });
      
      if (tmuxResult.status !== 0) {
        console.log(`⚠️  Failed to launch ${getAssistantLabel(assistant)} in tmux. Run the command above manually.`);
        console.log(`Command: ${launchCommand}`);
        continue;
      }
      
      console.log(`\n✅ ${getAssistantLabel(assistant)} launched. Return to menu when done.`);
      await promptInput("Press Enter to return to menu...");
    } catch (error) {
      console.log(`⚠️  Failed to launch ${getAssistantLabel(assistant)}. Error: ${(error as Error).message}`);
      continue;
    }
  }
}

/**
 * Update ~/.pi/agent/models.json with isambard-vllm configuration before launching Pi
 */
async function updatePiModelsConfigForLaunch(
  opts: { 
    model: string; 
    localPort: number; 
    maxModelLen?: number; 
    toolCall: boolean; 
    reasoning: boolean 
  },
  cwd: string
): Promise<void> {
  const homeDir = process.env.HOME || "";
  if (!homeDir) {
    console.log("⚠️  Warning: HOME environment variable not set, skipping Pi models.json update");
    return;
  }
  
  const piConfigPath = join(homeDir, '.pi', 'agent', 'models.json');
  const piConfigDir = require("path").dirname(piConfigPath);
  
  // Ensure the directory exists
  if (!existsSync(piConfigDir)) {
    mkdirSync(piConfigDir, { recursive: true });
  }
  
  // Read existing config or create empty structure
  let existingConfig: unknown = {};
  if (existsSync(piConfigPath)) {
    try {
      const fileContent = readFileSync(piConfigPath, 'utf-8');
      existingConfig = JSON.parse(fileContent);
    } catch (error) {
      console.log(`⚠️  Warning: Could not parse existing ${piConfigPath}. Creating new config.`);
      existingConfig = {};
    }
  }
  
  // Backup existing config if it exists and is not empty
  if (existsSync(piConfigPath)) {
    const backupPath = `${piConfigPath}.backup-${Date.now()}`;
    try {
      copyFileSync(piConfigPath, backupPath);
      console.log(`📋 Backed up existing config to ${backupPath}`);
    } catch (error) {
      console.log(`⚠️  Warning: Could not backup existing config: ${error.message}`);
    }
  }
  
  // Generate the new Pi config for isambard-vllm
  const newPiProviderConfig = generatePiModelsConfig(opts);
  
  // Merge with existing config
  const mergedConfig: Record<string, unknown> = {
    ...(existingConfig as Record<string, unknown>),
    providers: {
      ...((existingConfig as Record<string, unknown>).providers || {}),
      ...newPiProviderConfig.providers
    }
  };
  
  // Write the updated config
  try {
    writeFileSync(piConfigPath, JSON.stringify(mergedConfig, null, 2) + '\n');
    console.log(`✅ Updated ${piConfigPath} with isambard-vllm configuration`);
  } catch (error) {
    console.error(`❌ Failed to write to ${piConfigPath}: ${error.message}`);
    throw error;
  }
}

/**
 * Prompt for assistant selection
 */
async function promptAssistantMenu(title: string, options: readonly MenuOption<string>[]): Promise<string> {
  while (true) {
    console.log(title);
    for (const option of options) {
      console.log(`  [${option.input}] ${option.label}`);
    }
    console.log("");
    
    const answer = await promptInput("Selection: ");
    const selected = options.find((option) => option.input === answer);
    if (selected) return selected.key;
    
    console.log(`Invalid selection: ${answer || "(empty)"}. Please try again.\n`);
  }
}

/**
 * Prompt for wrapper selection
 */
async function promptWrapperMenu(title: string, options: readonly MenuOption<string>[]): Promise<string> {
  while (true) {
    console.log(title);
    for (const option of options) {
      console.log(`  [${option.input}] ${option.label}`);
    }
    console.log("");
    
    const answer = await promptInput("Selection: ");
    const selected = options.find((option) => option.input === answer);
    if (selected) return selected.key;
    
    console.log(`Invalid selection: ${answer || "(empty)"}. Please try again.\n`);
  }
}

/**
 * Prompt for action selection
 */
async function promptActionMenu(title: string, options: readonly MenuOption<string>[]): Promise<string> {
  while (true) {
    console.log(title);
    for (const option of options) {
      console.log(`  [${option.input}] ${option.label}`);
    }
    console.log("");
    
    const answer = await promptInput("Selection: ");
    const selected = options.find((option) => option.input === answer);
    if (selected) return selected.key;
    
    console.log(`Invalid selection: ${answer || "(empty)"}. Please try again.\n`);
  }
}

/**
 * Generic input prompt
 */
async function promptInput(question: string): Promise<string> {
  const { createInterface } = await import("readline");
  const rl = createInterface({ 
    input: process.stdin, 
    output: process.stdout, 
    terminal: true 
  });
  
  return new Promise<string>((resolve) => {
    rl.question(question, (value) => {
      rl.close();
      resolve(value.trim());
    });
  });
}

/**
 * Menu option interface
 */
interface MenuOption<T extends string> {
  key: T;
  label: string;
  input: string;
}