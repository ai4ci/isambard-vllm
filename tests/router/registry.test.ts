import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { loadModelRegistry, saveModelRegistry, validateModelConfig } from '../../src/router/registry.js';
import * as fs from 'fs';
import * as path from 'path';

describe('Model Registry', () => {
  const testConfigDir = process.env.HOME + '/.config/ivllm';
  const testConfigPath = path.join(testConfigDir, 'models.json');

  beforeEach(() => {
    // Ensure test directory exists
    fs.mkdirSync(testConfigDir, { recursive: true });
    // Clean up any existing test config
    if (fs.existsSync(testConfigPath)) {
      fs.unlinkSync(testConfigPath);
    }
  });

  afterEach(() => {
    // Clean up test config
    if (fs.existsSync(testConfigPath)) {
      fs.unlinkSync(testConfigPath);
    }
  });

  describe('loadModelRegistry', () => {
    it('returns empty registry when file doesn\'t exist', () => {
      const config = loadModelRegistry();
      expect(config).toEqual({ models: {} });
    });

    it('loads existing registry', () => {
      const testConfig = {
        models: {
          'test-model': {
            configPath: '/path/to/config.yaml',
            idleTimeoutMinutes: 30,
            autoStart: true,
          },
        },
      };
      fs.writeFileSync(testConfigPath, JSON.stringify(testConfig));
      
      const loaded = loadModelRegistry();
      expect(loaded).toEqual(testConfig);
    });
  });

  describe('saveModelRegistry', () => {
    it('creates directory and saves config', () => {
      const testConfig = {
        models: {
          'test-model': {
            configPath: '/path/to/config.yaml',
            idleTimeoutMinutes: 30,
          },
        },
      };
      
      saveModelRegistry(testConfig);
      
      expect(fs.existsSync(testConfigPath)).toBe(true);
      const loaded = JSON.parse(fs.readFileSync(testConfigPath, 'utf-8'));
      expect(loaded).toEqual(testConfig);
    });
  });

  describe('validateModelConfig', () => {
    it('accepts valid config', () => {
      const errors = validateModelConfig('qwen3-5-397b', {
        configPath: __filename, // Use this file as it exists
        idleTimeoutMinutes: 30,
        autoStart: true,
      });
      expect(errors).toEqual([]);
    });

    it('rejects invalid model name', () => {
      const errors = validateModelConfig('Invalid_Model_Name!', {
        configPath: __filename,
      });
      expect(errors).toContain('Model name must be alphanumeric with hyphens only (got: Invalid_Model_Name!)');
    });

    it('rejects non-existent config path', () => {
      const errors = validateModelConfig('test-model', {
        configPath: '/nonexistent/path/config.yaml',
      });
      expect(errors).toContain('Config file not found: /nonexistent/path/config.yaml');
    });

    it('rejects invalid idle timeout', () => {
      const errors = validateModelConfig('test-model', {
        configPath: __filename,
        idleTimeoutMinutes: -2,
      });
      expect(errors).toContain('idleTimeoutMinutes must be -1 (never) or positive integer');
    });

    it('accepts idleTimeoutMinutes = -1 (never timeout)', () => {
      const errors = validateModelConfig('test-model', {
        configPath: __filename,
        idleTimeoutMinutes: -1,
      });
      expect(errors).toEqual([]);
    });
  });
});
