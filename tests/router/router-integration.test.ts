import { describe, it, expect } from 'bun:test';
import { RouterService } from '../../src/router/router-service.js';
import { LocalExecutor } from '../../src/router/executor.js';

describe('Router Service Integration', () => {
  it('can instantiate and start router with local executor', async () => {
    const executor = new LocalExecutor();
    const router = new RouterService({
      port: 11450,
      host: '127.0.0.1',
      executor,
      loginHost: 'localhost', // Not used with LocalExecutor
    });

    try {
      await router.start();
      
      // Verify server is running
      const response = await fetch('http://127.0.0.1:11450/health');
      expect(response.status).toBe(200);
      
      const data = await response.json();
      expect(data.status).toBe('ok');
      
      await router.stop();
    } catch (error) {
      await router.stop();
      throw error;
    }
  });

  it('exposes admin endpoints', async () => {
    const executor = new LocalExecutor();
    const router = new RouterService({
      port: 11451,
      host: '127.0.0.1',
      executor,
      loginHost: 'localhost',
    });

    try {
      await router.start();

      // Test /admin/models endpoint
      const modelsResponse = await fetch('http://127.0.0.1:11451/admin/models');
      expect(modelsResponse.status).toBe(200);
      const models = await modelsResponse.json();
      expect(models.models).toEqual([]);

      // Test /admin/provider endpoint
      const providerResponse = await fetch('http://127.0.0.1:11451/admin/provider');
      expect(providerResponse.status).toBe(200);
      const provider = await providerResponse.json();
      expect(provider.provider).toBe('isambard-ivllm');
      expect(provider.baseUrl).toBe('http://127.0.0.1:11451');

      await router.stop();
    } catch (error) {
      await router.stop();
      throw error;
    }
  });

  it('returns empty model list initially', async () => {
    const executor = new LocalExecutor();
    const router = new RouterService({
      port: 11452,
      host: '127.0.0.1',
      executor,
      loginHost: 'localhost',
    });

    try {
      await router.start();

      const response = await fetch('http://127.0.0.1:11452/v1/models');
      expect(response.status).toBe(200);
      const data = await response.json();
      expect(data.object).toBe('list');
      expect(data.data).toEqual([]);

      await router.stop();
    } catch (error) {
      await router.stop();
      throw error;
    }
  });
});
