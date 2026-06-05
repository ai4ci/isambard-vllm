import { RouterConfig, ModelConfig, ModelState, PortPool } from './types.js';
import { PortPoolManager } from './port-pool.js';
import { loadModelRegistry, saveModelRegistry, validateModelConfig } from './registry.js';

/**
 * ModelRegistry manages model configurations and runtime state
 */
export class ModelRegistry {
  private config: RouterConfig;
  private runtimeState: Map<string, ModelState> = new Map();
  private portPool: PortPool;

  constructor(portPool?: PortPool) {
    this.config = loadModelRegistry();
    this.portPool = portPool ?? new PortPoolManager();
  }

  /**
   * Get all models (config + runtime state)
   */
  getAllModels(): Array<ModelConfig & { name: string; runtime?: ModelState }> {
    return Object.entries(this.config.models).map(([name, config]) => ({
      name,
      ...config,
      runtime: this.runtimeState.get(name),
    }));
  }

  /**
   * Get a specific model by name
   */
  getModel(name: string): (ModelConfig & { name: string; runtime?: ModelState }) | null {
    const config = this.config.models[name];
    if (!config) return null;

    return {
      name,
      ...config,
      runtime: this.runtimeState.get(name),
    };
  }

  /**
   * Add a new model configuration
   */
  addModel(name: string, config: ModelConfig): string[] {
    const errors = validateModelConfig(name, config);
    if (errors.length > 0) return errors;

    if (this.config.models[name]) {
      return [`Model '${name}' already exists`];
    }

    this.config.models[name] = config;
    saveModelRegistry(this.config);

    // Initialize runtime state
    this.runtimeState.set(name, {
      name,
      status: 'stopped',
    });

    return [];
  }

  /**
   * Remove a model configuration (must be stopped first)
   */
  removeModel(name: string): string[] {
    const model = this.getModel(name);
    if (!model) return [`Model '${name}' not found`];

    if (model.runtime?.status === 'running' || model.runtime?.status === 'starting') {
      return [`Cannot remove model '${name}' while it's ${model.runtime.status}`];
    }

    delete this.config.models[name];
    this.runtimeState.delete(name);
    saveModelRegistry(this.config);

    return [];
  }

  /**
   * Update model runtime state
   */
  updateState(name: string, updates: Partial<ModelState>): void {
    const state = this.runtimeState.get(name);
    if (!state) {
      this.runtimeState.set(name, { name, status: 'stopped', ...updates });
    } else {
      Object.assign(state, updates);
    }
  }

  /**
   * Get runtime state for a model
   */
  getState(name: string): ModelState | undefined {
    return this.runtimeState.get(name);
  }

  /**
   * Acquire a port for a model
   */
  acquirePort(): number {
    return this.portPool.acquire();
  }

  /**
   * Release a port back to the pool
   */
  releasePort(port: number): void {
    this.portPool.release(port);
  }

  /**
   * Get all running models
   */
  getRunningModels(): Array<ModelConfig & { name: string; runtime: ModelState }> {
    return this.getAllModels().filter(
      (m): m is ModelConfig & { name: string; runtime: ModelState } =>
        m.runtime?.status === 'running'
    );
  }

  /**
   * Get models by status
   */
  getModelsByStatus(status: ModelState['status']): Array<ModelConfig & { name: string; runtime: ModelState }> {
    return this.getAllModels().filter(
      (m): m is ModelConfig & { name: string; runtime: ModelState } =>
        m.runtime?.status === status
    );
  }
}
