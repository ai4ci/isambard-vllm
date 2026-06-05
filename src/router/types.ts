/**
 * Router configuration and types
 */

export interface RouterConfig {
  models: Record<string, ModelConfig>;
}

export interface ModelConfig {
  /** Absolute path to vLLM config YAML */
  configPath: string;
  /** Auto-shutdown after N minutes (-1 = never) */
  idleTimeoutMinutes?: number;
  /** Auto-start model on first request */
  autoStart?: boolean;
}

/**
 * Runtime state for a model (in-memory only)
 */
export interface ModelState {
  name: string;
  status: 'stopped' | 'starting' | 'running' | 'stopping' | 'failed';
  /** Assigned port from pool (11435-11534) */
  port?: number;
  /** SLURM job ID for cleanup */
  slurmJobId?: number;
  /** COMPUTE node hostname */
  nodeHostname?: string;
  /** When model became 'running' */
  startedAt?: Date;
  /** Last proxied request timestamp */
  lastActivityAt?: Date;
  /** Idle timeout handle */
  timeoutId?: NodeJS.Timeout;
  /** Health check interval handle */
  healthCheckInterval?: NodeJS.Timeout;
  /** Error message if failed */
  error?: string;
}

/**
 * Port pool manager interface
 */
export interface PortPool {
  acquire(): number;
  release(port: number): void;
  isAvailable(port: number): boolean;
}
