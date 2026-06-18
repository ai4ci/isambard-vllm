/**
 * AI coding assistant utilities.
 * Detects available assistants, generates config/env vars for each.
 */

import { spawnSync } from 'child_process';
import { basename } from 'path';

export interface AssistantConfig {
  name: string;
  env: Record<string, string>;
  args: string[];
}

export interface AssistantOption {
  id: number;
  label: string;
  assistant: string;
  useScoder: boolean;
}

export interface OpencodeConfigOptions {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall?: boolean;
  reasoning?: boolean;
  endpointHost?: string;
}

export type AssistantName = 'opencode' | 'claude' | 'copilot' | 'pi';
export type LaunchWrapper = 'none' | 'scoder' | 'sbx';

interface AssistantDefinition {
  name: AssistantName;
  label: string;
}

export interface AssistantEnvOptions extends OpencodeConfigOptions {}

export interface LaunchCommandOptions extends AssistantEnvOptions {
  assistant: AssistantName;
  wrapper: LaunchWrapper;
  cwd: string;
  sandboxName?: string;
}

export interface SbxSandbox {
  name: string;
  agent: string;
  workspace: string;
  status?: string;
}

const ASSISTANTS: AssistantDefinition[] = [
  { name: 'opencode', label: 'OpenCode' },
  { name: 'copilot', label: 'GitHub Copilot' },
  { name: 'claude', label: 'Claude Code' },
  { name: 'pi', label: 'Pi' },
];

/**
 * Check if a binary exists on PATH.
 * @param name
 */
export function binaryExists(name: string): boolean {
  const result = spawnSync('which', [name], {
    shell: true,
    encoding: 'utf-8',
  });
  return (
    result.status === 0 &&
    result.output[1] != null &&
    result.output[1].trim().length > 0
  );
}

/**
 * Get the list of assistant binaries available on PATH.
 */
export function getAvailableAssistants(): string[] {
  const candidates = ['opencode', 'claude', 'copilot'];
  return candidates.filter(binaryExists);
}

/**
 * Check if scoder is available on PATH.
 */
export function getScoderAvailable(): boolean {
  return binaryExists('scoder');
}

/**
 * Check if sbx is available on PATH.
 */
export function getSbxAvailable(): boolean {
  return binaryExists('sbx');
}

/**
 *
 * @param assistant
 */
export function getAssistantLabel(assistant: AssistantName): string {
  return ASSISTANTS.find((item) => item.name === assistant)?.label ?? assistant;
}

/**
 *
 * @param assistant
 * @param availableAssistants
 * @param hasScoder
 * @param hasSbx
 */
export function getAvailableWrappers(
  assistant: AssistantName,
  availableAssistants: string[],
  hasScoder: boolean,
  hasSbx: boolean,
): LaunchWrapper[] {
  const wrappers: LaunchWrapper[] = [];
  const hasLocalAssistant = availableAssistants.includes(assistant);

  // For Pi, we allow wrappers even if the binary isn't installed
  // since it's primarily configuration-based
  if (assistant === 'pi' || hasLocalAssistant) {
    wrappers.push('none');
    if (hasScoder) {
      wrappers.push('scoder');
    }
  }

  if (hasSbx) {
    wrappers.push('sbx');
  }

  return wrappers;
}

/**
 * Generate OpenCode config content for ivllm launches.
 * @param opts
 */
export function generateOpencodeConfig(
  opts: OpencodeConfigOptions,
): Record<string, unknown> {
  const context = opts.maxModelLen ?? 4096;
  const endpointHost = opts.endpointHost ?? 'localhost';

  const modelEntry: Record<string, unknown> = {
    name: `${opts.model} (Isambard)`,
    limit: { context, output: context },
  };
  if (opts.toolCall) modelEntry['tool_call'] = true;
  if (opts.reasoning) modelEntry['reasoning'] = true;

  return {
    $schema: 'https://opencode.ai/config.json',
    model: `isambard-vllm/${opts.model}`,
    provider: {
      'isambard-vllm': {
        npm: '@ai-sdk/openai-compatible',
        name: 'Isambard vLLM Server',
        options: {
          baseURL: `http://${endpointHost}:${opts.localPort}/v1`,
          apiKey: 'EMPTY',
        },
        models: {
          [opts.model]: modelEntry,
        },
      },
    },
  };
}

/**
 * Generate runtime environment overrides for OpenCode.
 * @param opts
 */
export function generateOpencodeEnv(
  opts: OpencodeConfigOptions,
): Record<string, string> {
  return {
    OPENCODE_CONFIG_CONTENT: JSON.stringify(generateOpencodeConfig(opts)),
  };
}

/**
 * Generate Pi models.json configuration for vLLM integration.
 * @param opts
 */
export function generatePiModelsConfig(opts: OpencodeConfigOptions): Record<string, any> {
  const context = opts.maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    id: opts.model,
    name: `${opts.model} (Isambard)`,
    contextWindow: context,
    maxTokens: context,
    input: ['text'],
    reasoning: opts.reasoning ?? false,
    // vLLM doesn't understand the "developer" role used for reasoning-capable models
    // so we send the system prompt as a "system" message instead.
    compat: {
      supportsDeveloperRole: false,
    },
  };

  // Add thinking level map if reasoning is enabled
  if (opts.reasoning) {
    modelEntry['thinkingLevelMap'] = {
      off: null, // Disable thinking when not needed
      minimal: null,
      low: null,
      medium: null,
      high: 'high',
      xhigh: 'max',
    };
  }

  return {
    providers: {
      'isambard-vllm': {
        baseUrl: `http://localhost:${opts.localPort}/v1`,
        apiKey: 'EMPTY',
        api: 'openai-completions',
        models: [modelEntry],
      },
    },
  };
}

/**
 * Generate environment variables for GitHub Copilot.
 * @param localPort
 * @param model
 */
export function generateCopilotEnv(
  localPort: number,
  model: string,
): Record<string, string> {
  return {
    COPILOT_PROVIDER_BASE_URL: `http://localhost:${localPort}/v1`,
    COPILOT_MODEL: model,
  };
}

/**
 * Generate environment variables for Claude Code.
 * @param localPort
 * @param model
 */
export function generateClaudeEnv(
  localPort: number,
  model: string,
): Record<string, string> {
  return {
    ANTHROPIC_BASE_URL: `http://localhost:${localPort}`,
    ANTHROPIC_API_KEY: 'ollama',
    ANTHROPIC_DEFAULT_SONNET_MODEL: model,
    ANTHROPIC_DEFAULT_OPUS_MODEL: model,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: model,
    CLAUDE_CODE_SUBAGENT_MODEL: model,
  };
}

/**
 *
 * @param assistant
 * @param opts
 */
export function generateAssistantEnv(
  assistant: AssistantName,
  opts: AssistantEnvOptions,
): Record<string, string> {
  const endpointHost = opts.endpointHost ?? 'localhost';

  switch (assistant) {
    case 'opencode':
      return generateOpencodeEnv({ ...opts, endpointHost });
    case 'copilot':
      return {
        ...generateCopilotEnv(opts.localPort, opts.model),
        COPILOT_PROVIDER_BASE_URL: `http://${endpointHost}:${opts.localPort}/v1`,
      };
    case 'claude':
      return {
        ...generateClaudeEnv(opts.localPort, opts.model),
        ANTHROPIC_BASE_URL: `http://${endpointHost}:${opts.localPort}`,
      };
    case 'pi':
      // Pi assistant works via configuration files, not environment variables
      return {};
  }
}

/**
 *
 * @param assistant
 * @param cwd
 */
export function buildSandboxName(
  assistant: AssistantName | undefined,
  cwd: string,
): string {
  const assistantName = assistant ?? 'opencode';
  const workspace = basename(cwd)
    .replace(/[^A-Za-z0-9.+-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${assistantName}-${workspace || 'workspace'}`;
}

/**
 *
 * @param assistant
 * @param cwd
 * @param sandboxName
 */
export function buildSandboxCreateCommand(
  assistant: AssistantName,
  cwd: string,
  sandboxName?: string,
): string {
  const name = sandboxName ?? buildSandboxName(assistant, cwd);
  return `sbx create --name ${shellQuote(name)} ${assistant} ${shellQuote(cwd)}`;
}

/**
 *
 * @param value
 */
function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

/**
 *
 * @param env
 */
function formatInlineEnv(env: Record<string, string>): string {
  return Object.entries(env ?? {})
    .map(([key, value]) => `${key}=${shellQuote(value)}`)
    .join(' ');
}

/**
 *
 * @param assistant
 */
function getAssistantArgs(assistant: AssistantName): string[] {
  return assistant === 'copilot' ? ['--continue'] : [];
}

/**
 *
 * @param opts
 */
export function buildLaunchCommand(opts: LaunchCommandOptions): string {
  const endpointHost =
    opts.wrapper === 'sbx' ? 'host.docker.internal' : 'localhost';
  const env = generateAssistantEnv(opts.assistant, { ...opts, endpointHost });
  const agentCommand = [
    opts.assistant,
    ...getAssistantArgs(opts.assistant),
  ].join(' ');

  if (opts.wrapper === 'sbx') {
    const sandboxName =
      opts.sandboxName ?? buildSandboxName(opts.assistant, opts.cwd);
    const envFlags = Object.entries(env)
      .map(([key, value]) => `-e ${shellQuote(`${key}=${value}`)}`)
      .join(' ');
    return `sbx exec -it -w ${shellQuote(opts.cwd)} ${envFlags} ${sandboxName} ${agentCommand}`.trim();
  }

  const envPrefix = formatInlineEnv(env);
  const base =
    opts.wrapper === 'scoder'
      ? `scoder --llm-port ${opts.localPort} ${agentCommand}`
      : agentCommand;
  return `cd ${shellQuote(opts.cwd)} && ${envPrefix} ${base}`.trim();
}

/**
 *
 * @param raw
 */
export function parseSbxSandboxes(raw: string): SbxSandbox[] {
  if (!raw.trim()) return [];

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }

  const rows: unknown[] = Array.isArray(parsed)
    ? parsed
    : typeof parsed === 'object' &&
        parsed !== null &&
        Array.isArray((parsed as Record<string, unknown>)['sandboxes'])
      ? ((parsed as Record<string, unknown>)['sandboxes'] as unknown[])
      : [];

  return rows
    .map((row) => {
      const record = row as Record<string, unknown>;
      const name = String(
        record['name'] ??
          record['Name'] ??
          record['sandbox'] ??
          record['SANDBOX'] ??
          '',
      );
      const agent = String(
        record['agent'] ?? record['Agent'] ?? record['AGENT'] ?? '',
      );
      const workspace = String(
        record['workspace'] ?? record['Workspace'] ?? record['WORKSPACE'] ?? '',
      );
      const status = String(
        record['status'] ?? record['Status'] ?? record['STATUS'] ?? '',
      );

      if (!name || !agent || !workspace) return null;
      return {
        name,
        agent,
        workspace,
        status: status || undefined,
      } as SbxSandbox;
    })
    .filter((row): row is SbxSandbox => row !== null);
}

/**
 *
 */
function listSbxSandboxes(): SbxSandbox[] {
  const result = spawnSync('sbx', ['ls', '--json'], {
    encoding: 'utf-8',
  });
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || 'sbx ls --json failed');
  }
  return parseSbxSandboxes(result.stdout);
}

/**
 *
 * @param sandboxes
 * @param assistant
 * @param cwd
 */
export function findMatchingSandbox(
  sandboxes: SbxSandbox[],
  assistant: AssistantName,
  cwd: string,
): SbxSandbox | null {
  // Normalize the input cwd for comparison
  const normalizedCwd = cwd.trim();

  return (
    sandboxes.find((sandbox) => {
      // Normalize the sandbox fields for comparison
      const normalizedAgent = sandbox.agent?.toLowerCase() ?? '';
      const normalizedWorkspace = sandbox.workspace?.trim() ?? '';

      // Check for exact match first
      if (
        normalizedAgent === assistant.toLowerCase() &&
        normalizedWorkspace === normalizedCwd
      ) {
        return true;
      }

      // Check if workspace matches when ignoring trailing slashes
      const workspaceWithoutTrailingSlash = normalizedWorkspace.replace(
        /\/+$/,
        '',
      );
      const cwdWithoutTrailingSlash = normalizedCwd.replace(/\/+$/, '');
      if (
        normalizedAgent === assistant.toLowerCase() &&
        workspaceWithoutTrailingSlash === cwdWithoutTrailingSlash
      ) {
        return true;
      }

      return false;
    }) ?? null
  );
}

/**
 *
 * @param assistant
 * @param cwd
 */
export function ensureSbxSandbox(
  assistant: AssistantName,
  cwd: string,
): { sandboxName: string; created: boolean } {
  const sandboxes = listSbxSandboxes();
  const existing = findMatchingSandbox(sandboxes, assistant, cwd);
  if (existing) {
    return { sandboxName: existing.name, created: false };
  }

  const sandboxName = buildSandboxName(assistant, cwd);
  const result = spawnSync(
    'sbx',
    ['create', '--name', sandboxName, assistant, cwd],
    {
      encoding: 'utf-8',
      stdio: 'inherit',
    },
  );
  if (result.status !== 0) {
    throw new Error(`sbx create failed for ${sandboxName}`);
  }
  return { sandboxName, created: true };
}

/**
 * Build the list of menu options based on available assistants and scoder.
 * @param assistants
 * @param hasScoder
 */
export function buildAssistantMenuOptions(
  assistants: string[],
  hasScoder: boolean,
): AssistantOption[] {
  const options: AssistantOption[] = [];
  let id = 1;

  for (const assistant of assistants) {
    // Direct launch option
    options.push({ id, label: `${assistant}`, assistant, useScoder: false });
    id++;

    // Scoder option (if available)
    if (hasScoder) {
      options.push({
        id,
        label: `scoder ${assistant}`,
        assistant,
        useScoder: true,
      });
      id++;
    }
  }

  return options;
}

/**
 * Get the binary name and args for launching an assistant.
 * @param assistant
 * @param useScoder
 */
export function getLaunchCommand(
  assistant: AssistantName,
  useScoder: boolean,
): { binary: string; args: string[] } {
  if (useScoder) {
    return {
      binary: 'scoder',
      args: [assistant, ...getAssistantArgs(assistant)],
    };
  }
  return { binary: assistant, args: getAssistantArgs(assistant) };
}

export interface OpencodeSnippetOptions {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall?: boolean;
  reasoning?: boolean;
}

/**
 *
 * @param opts
 */
export function formatOpencodeSnippet(opts: OpencodeSnippetOptions): string {
  const { model, localPort, maxModelLen, toolCall, reasoning } = opts;
  const context = maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    name: `${model} (Isambard)`,
    limit: { context, output: context },
  };
  if (toolCall) modelEntry['tool_call'] = true;
  if (reasoning) modelEntry['reasoning'] = true;

  const snippet = {
    $schema: 'https://opencode.ai/config.json',
    model: `isambard-vllm/${model}`,
    provider: {
      'isambard-vllm': {
        npm: '@ai-sdk/openai-compatible',
        name: 'Isambard vLLM Server',
        options: {
          baseURL: `http://localhost:${localPort}/v1`,
          apiKey: 'EMPTY',
        },
        models: {
          [model]: modelEntry,
        },
      },
    },
  };

  return JSON.stringify(snippet, null, 2);
}

