import { RouterConfig, ModelConfig } from '../types.js';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Load model registry from ~/.config/ivllm/models.json
 */
export function loadModelRegistry(): RouterConfig {
  const configPath = path.join(process.env.HOME || '', '.config', 'ivllm', 'models.json');
  
  try {
    const content = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(content) as RouterConfig;
  } catch (error) {
    // File doesn't exist yet - return empty registry
    return { models: {} };
  }
}

/**
 * Save model registry to ~/.config/ivllm/models.json
 */
export function saveModelRegistry(config: RouterConfig): void {
  const configDir = path.join(process.env.HOME || '', '.config', 'ivllm');
  const configPath = path.join(configDir, 'models.json');
  
  // Ensure directory exists
  fs.mkdirSync(configDir, { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
}

/**
 * Validate model configuration
 */
export function validateModelConfig(name: string, config: ModelConfig): string[] {
  const errors: string[] = [];
  
  // Name validation: alphanumeric + hyphens
  if (!/^[a-z0-9-]+$/.test(name)) {
    errors.push(`Model name must be alphanumeric with hyphens only (got: ${name})`);
  }
  
  // Config path must exist
  if (!fs.existsSync(config.configPath)) {
    errors.push(`Config file not found: ${config.configPath}`);
  }
  
  // Idle timeout must be -1 or positive integer
  if (config.idleTimeoutMinutes !== undefined && 
      config.idleTimeoutMinutes < -1) {
    errors.push(`idleTimeoutMinutes must be -1 (never) or positive integer`);
  }
  
  return errors;
}
