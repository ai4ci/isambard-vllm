import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { createRouterServer, shutdownServer } from '../../src/router/server.js';
import * as fs from 'fs';
import * as path from 'path';

describe('Router Server', () => {
  const testConfigDir = '/tmp/ivllm-test_config';
  const testModelsFile = path.join(testConfigDir, 'models.json');

  beforeEach(() => {
    fs.mkdirSync(testConfigDir, { recursive: true });
    fs.writeFileSync(testModelsFile, JSON.stringify({ models: [] }));
  });

  afterEach(() => {
    fs.rmSync(testConfigDir, { recursive: true, force: true });
  });

  it('starts and responds to health check', async () => {
    const server = await createRouterServer(11440); // Use non-standard port for testing
    
    try {
      const response = await server.inject({
        method: 'GET',
        url: '/health',
      });
      
      expect(response.statusCode).toBe(200);
      const body = response.json();
      expect(body.status).toBe('ok');
      expect(body.timestamp).toBeDefined();
    } finally {
      await shutdownServer(server);
    }
  });

  it('returns empty model list initially', async () => {
    const server = await createRouterServer(11441);
    
    try {
      const response = await server.inject({
        method: 'GET',
        url: '/v1/models',
      });
      
      expect(response.statusCode).toBe(200);
      const body = response.json();
      expect(body.object).toBe('list');
      expect(body.data).toEqual([]);
    } finally {
      await shutdownServer(server);
    }
  });

  it('returns empty admin models list', async () => {
    const server = await createRouterServer(11442);
    
    try {
      const response = await server.inject({
        method: 'GET',
        url: '/admin/models',
      });
      
      expect(response.statusCode).toBe(200);
      const body = response.json();
      expect(body.models).toEqual([]);
    } finally {
      await shutdownServer(server);
    }
  });

  it('shuts down gracefully', async () => {
    const server = await createRouterServer(11443);
    
    await shutdownServer(server);
    
    // Server should be closed - this would throw if we tried to use it
    expect(server.server.listening).toBe(false);
  });

  it('returns 404 for unimplemented proxy endpoint in stub server', async () => {
    const server = await createRouterServer(11444);
    
    try {
      const response = await server.inject({
        method: 'POST',
        url: '/v1/chat/completions',
        body: {
          model: 'test-model',
          messages: [{ role: 'user', content: 'Hello' }],
        },
      });
      
      // Stub server doesn't have full router service, returns 404
      expect(response.statusCode).toBe(404);
    } finally {
      await shutdownServer(server);
    }
  });
});
