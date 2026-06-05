import Fastify, { FastifyInstance } from 'fastify';

/**
 * Create and configure the Fastify HTTP server
 */
export async function createRouterServer(port: number = 11434): Promise<FastifyInstance> {
  const server = Fastify({
    logger: {
      level: 'info',
    },
  });

  // Health check endpoint
  server.get('/health', async (request, reply) => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  // OpenAI-compatible model listing (stub - will be implemented in F3.4)
  server.get('/v1/models', async (request, reply) => {
    return {
      object: 'list',
      data: [],
    };
  });

  // Admin endpoint stub
  server.get('/admin/models', async (request, reply) => {
    return { models: [] };
  });

  // Start server
  await server.listen({ port, host: '127.0.0.1' });
  
  return server;
}

/**
 * Gracefully shutdown the server
 */
export async function shutdownServer(server: FastifyInstance): Promise<void> {
  await server.close();
}
