import { describe, it, expect } from 'bun:test';
import { createRouterServer, shutdownServer } from '../../src/router/server.js';

describe('Router Server', () => {
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
});
