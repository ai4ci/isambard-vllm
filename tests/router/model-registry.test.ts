import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { ModelRegistry } from '../../src/router/model-registry.js';
import { PortPoolManager } from '../../src/router/port-pool.js';
import * as fs from 'fs';
import * as path from 'path';

describe('Model Registry', () => {
  const testConfigDir = process.env.HOME + '/.config/ivllm-test';
  const testConfigPath = path.join(testConfigDir, 'models.json');

  beforeEach(() => {
    fs.mkdirSync(testConfigDir, { recursive: true });
    if (fs.existsSync(testConfigPath)) {
      fs.unlinkSync(testConfigPath);
    }
    // Temporarily override HOME for testing
    (process.env as any).HOME = process.env.HOME;
  });

  afterEach(() => {
    if (fs.existsSync(testConfigPath)) {
      fs.unlinkSync(testConfigPath);
    }
    fs.rmdirSync(testConfigDir);
  });

  describe('constructor', () => {
    it('loads empty registry when no file exists', () => {
      const registry = new ModelRegistry();
      const models = registry.getAllModels();
      expect(models).toEqual([]);
    });

    it('uses custom port pool if provided', () => {
      const customPool = new PortPoolManager(20000, 20010);
      const registry = new ModelRegistry(customPool);
      const port = registry.acquirePort();
      expect(port).toBe(20000);
    });
  });

  describe('addModel', () => {
    it('adds a new model successfully', () => {
      const registry = new ModelRegistry();
      const errors = registry.addModel('test-model', {
        configPath: __filename,
        idleTimeoutMinutes: 30,
        autoStart: true,
      });

      expect(errors).toEqual([]);
      const model = registry.getModel('test-model');
      expect(model?.name).toBe('test-model');
      expect(model?.idleTimeoutMinutes).toBe(30);
      expect(model?.autoStart).toBe(true);
    });

    it('rejects duplicate model names', () => {
      const registry = new ModelRegistry();
      registry.addModel('test-model', { configPath: __filename });
      
      const errors = registry.addModel('test-model', { configPath: __filename });
      expect(errors).toContain("Model 'test-model' already exists");
    });

    it('validates model config before adding', () => {
      const registry = new ModelRegistry();
      const errors = registry.addModel('Invalid_Name!', {
        configPath: '/nonexistent/path.yaml',
      });

      expect(errors.length).toBeGreaterThan(0);
    });
  });

  describe('removeModel', () => {
    it('removes a stopped model', () => {
      const registry = new ModelRegistry();
      registry.addModel('test-model', { configPath: __filename });
      
      const errors = registry.removeModel('test-model');
      expect(errors).toEqual([]);
      expect(registry.getModel('test-model')).toBeNull();
    });

    it('rejects removing a running model', () => {
      const registry = new ModelRegistry();
      registry.addModel('test-model', { configPath: __filename });
      registry.updateState('test-model', { status: 'running' });
      
      const errors = registry.removeModel('test-model');
      expect(errors).toContain("Cannot remove model 'test-model' while it's running");
    });

    it('rejects removing non-existent model', () => {
      const registry = new ModelRegistry();
      const errors = registry.removeModel('nonexistent');
      expect(errors).toContain("Model 'nonexistent' not found");
    });
  });

  describe('updateState', () => {
    it('creates state for model if not exists', () => {
      const registry = new ModelRegistry();
      registry.addModel('test-model', { configPath: __filename });
      
      registry.updateState('test-model', { status: 'running', port: 11435 });
      
      const state = registry.getState('test-model');
      expect(state?.status).toBe('running');
      expect(state?.port).toBe(11435);
    });

    it('updates existing state', () => {
      const registry = new ModelRegistry();
      registry.addModel('test-model', { configPath: __filename });
      registry.updateState('test-model', { status: 'stopped' });
      
      registry.updateState('test-model', { status: 'running', port: 11435 });
      
      const state = registry.getState('test-model');
      expect(state?.status).toBe('running');
      expect(state?.port).toBe(11435);
    });
  });

  describe('getRunningModels', () => {
    it('returns only running models', () => {
      const registry = new ModelRegistry();
      registry.addModel('model1', { configPath: __filename });
      registry.addModel('model2', { configPath: __filename });
      registry.addModel('model3', { configPath: __filename });

      registry.updateState('model1', { status: 'running' });
      registry.updateState('model2', { status: 'stopped' });
      registry.updateState('model3', { status: 'running' });

      const running = registry.getRunningModels();
      expect(running.length).toBe(2);
      expect(running.map(m => m.name)).toEqual(expect.arrayContaining(['model1', 'model3']));
    });

    it('returns empty array when no models running', () => {
      const registry = new ModelRegistry();
      const running = registry.getRunningModels();
      expect(running).toEqual([]);
    });
  });

  describe('getModelsByStatus', () => {
    it('returns models filtered by status', () => {
      const registry = new ModelRegistry();
      registry.addModel('model1', { configPath: __filename });
      registry.addModel('model2', { configPath: __filename });

      registry.updateState('model1', { status: 'starting' });
      registry.updateState('model2', { status: 'running' });

      const starting = registry.getModelsByStatus('starting');
      expect(starting.length).toBe(1);
      expect(starting[0].name).toBe('model1');
    });
  });

  describe('port management', () => {
    it('acquires and releases ports correctly', () => {
      const registry = new ModelRegistry();
      
      const port1 = registry.acquirePort();
      const port2 = registry.acquirePort();
      
      expect(port1).toBe(11435);
      expect(port2).toBe(11436);
      
      registry.releasePort(port1);
      const port3 = registry.acquirePort();
      expect(port3).toBe(11435); // Reuses released port
    });
  });
});
