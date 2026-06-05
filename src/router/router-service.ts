import Fastify, { FastifyInstance } from 'fastify';
import { ModelRegistry } from './model-registry.js';
import { JobManager } from './job-manager.js';
import { RemoteExecutor } from '../types.js';
import * as http from 'http';

interface RouterOptions {
  port?: number;
  host?: string;
  executor: RemoteExecutor;
  loginHost: string;
}

/**
 * Main router service - HTTP API + model lifecycle management
 */
export class RouterService {
  private server: FastifyInstance | null = null;
  private registry: ModelRegistry;
  private jobManager: JobManager;
  private idleCheckInterval?: NodeJS.Timeout;
  private readonly options: Required<RouterOptions>;

  constructor(options: RouterOptions) {
    this.options = {
      port: options.port ?? 11434,
      host: options.host ?? '127.0.0.1',
      executor: options.executor,
      loginHost: options.loginHost,
    };

    this.registry = new ModelRegistry();
    this.jobManager = new JobManager(options.executor, this.registry, options.loginHost);
  }

  /**
   * Start the router HTTP server
   */
  async start(): Promise<void> {
    this.server = await this.createServer();
    await this.server.listen({ port: this.options.port, host: this.options.host });

    // Start idle checker (runs every 60 seconds)
    this.idleCheckInterval = setInterval(() => this.checkIdleTimeouts(), 60000);

    this.server.log.info(`Router started on http://${this.options.host}:${this.options.port}`);
  }

  /**
   * Stop the router and cleanup all models
   */
  async stop(): Promise<void> {
    this.server?.log.info('Shutting down router...');

    // Stop idle checker
    if (this.idleCheckInterval) {
      clearInterval(this.idleCheckInterval);
    }

    // Stop all running models
    const runningModels = this.registry.getRunningModels();
    await Promise.all(
      runningModels.map(async (model) => {
        try {
          await this.jobManager.cancelJob(model.name);
        } catch (error) {
          this.server?.log.error(`Failed to stop model ${model.name}: ${error}`);
        }
      })
    );

    // Close HTTP server
    if (this.server) {
      await this.server.close();
      this.server = null;
    }
  }

  /**
   * Create and configure Fastify server
   */
  private async createServer(): Promise<FastifyInstance> {
    const server = Fastify({
      logger: {
        level: 'info',
      },
    });

    // Health check
    server.get('/health', async () => ({
      status: 'ok',
      timestamp: new Date().toISOString(),
    }));

    // OpenAI API: List models (only running)
    server.get('/v1/models', async () => {
      const runningModels = this.registry.getRunningModels();
      return {
        object: 'list',
        data: runningModels.map((m) => ({
          id: m.name,
          object: 'model',
          created: Math.floor((m.runtime?.startedAt?.getTime() ?? Date.now()) / 1000),
          owned_by: 'isambard-ivllm',
        })),
      };
    });

    // OpenAI API: Chat completions (with lazy startup)
    server.post('/v1/chat/completions', async (request, reply) => {
      const body = request.body as any;
      const modelName = body.model;

      if (!modelName) {
        return reply.code(400).send({ error: 'model field is required' });
      }

      const model = this.registry.getModel(modelName);
      if (!model) {
        return reply.code(400).send({ error: `Model '${modelName}' not configured` });
      }

      // Handle lazy startup
      if (model.runtime?.status === 'stopped') {
        if (model.autoStart) {
          // Start the model
          await this.startModel(modelName);
          return reply.code(503).send({
            error: {
              message: `Starting up ${modelName}, retry in 30s`,
              type: 'startup_in_progress',
            },
          });
        } else {
          return reply.code(400).send({
            error: `Model '${modelName}' is not running. Start it via POST /admin/models/${modelName}/start`,
          });
        }
      }

      if (model.runtime?.status === 'starting') {
        return reply.code(503).send({
          error: {
            message: `Starting up ${modelName}, retry in 30s`,
            type: 'startup_in_progress',
          },
        });
      }

      if (model.runtime?.status !== 'running') {
        return reply.code(500).send({ error: `Model ${modelName} is in invalid state` });
      }

      // Proxy to vLLM backend
      try {
        const result = await this.proxyRequest(modelName, body);
        
        // Update last activity timestamp
        this.registry.updateState(modelName, { lastActivityAt: new Date() });
        
        return reply.code(result.statusCode).send(result.body);
      } catch (error) {
        this.server?.log.error(`Proxy error for ${modelName}: ${error}`);
        return reply.code(502).send({ 
          error: `Failed to proxy request to ${modelName}: ${error}` 
        });
      }
    });

    // Admin API: List all models
    server.get('/admin/models', async () => {
      const models = this.registry.getAllModels();
      return {
        models: models.map((m) => ({
          name: m.name,
          status: m.runtime?.status ?? 'stopped',
          port: m.runtime?.port,
          slurmJobId: m.runtime?.slurmJobId,
          nodeHostname: m.runtime?.nodeHostname,
          startedAt: m.runtime?.startedAt?.toISOString(),
          lastActivityAt: m.runtime?.lastActivityAt?.toISOString(),
          idleTimeoutMinutes: m.idleTimeoutMinutes ?? 15,
          autoStart: m.autoStart ?? false,
        })),
      };
    });

    // Admin API: Add model
    server.post('/admin/models', async (request, reply) => {
      const body = request.body as any;
      const { name, configPath, idleTimeoutMinutes, autoStart } = body;

      if (!name || !configPath) {
        return reply.code(400).send({ error: 'name and configPath are required' });
      }

      const errors = this.registry.addModel(name, {
        configPath,
        idleTimeoutMinutes,
        autoStart,
      });

      if (errors.length > 0) {
        return reply.code(400).send({ error: errors.join(', ') });
      }

      return { success: true, model: name };
    });

    // Admin API: Remove model
    server.delete('/admin/models/:name', async (request, reply) => {
      const { name } = request.params as { name: string };
      const errors = this.registry.removeModel(name);

      if (errors.length > 0) {
        return reply.code(400).send({ error: errors.join(', ') });
      }

      return { success: true };
    });

    // Admin API: Start model
    server.post('/admin/models/:name/start', async (request, reply) => {
      const { name } = request.params as { name: string };
      const model = this.registry.getModel(name);

      if (!model) {
        return reply.code(404).send({ error: `Model '${name}' not found` });
      }

      if (model.runtime?.status === 'running' || model.runtime?.status === 'starting') {
        return reply.code(400).send({ error: `Model '${name}' is already ${model.runtime.status}` });
      }

      try {
        await this.startModel(name);
        return { success: true, model: name, status: 'starting' };
      } catch (error) {
        return reply.code(500).send({ error: `Failed to start model: ${error}` });
      }
    });

    // Admin API: Stop model
    server.post('/admin/models/:name/stop', async (request, reply) => {
      const { name } = request.params as { name: string };
      const model = this.registry.getModel(name);

      if (!model) {
        return reply.code(404).send({ error: `Model '${name}' not found` });
      }

      if (model.runtime?.status !== 'running' && model.runtime?.status !== 'starting') {
        return reply.code(400).send({ error: `Model '${name}' is not running` });
      }

      try {
        await this.jobManager.cancelJob(name);
        return { success: true, model: name, status: 'stopped' };
      } catch (error) {
        return reply.code(500).send({ error: `Failed to stop model: ${error}` });
      }
    });

    // Admin API: Get model logs
    server.get('/admin/models/:name/logs', async (request, reply) => {
      const { name } = request.params as { name: string };

      try {
        const logs = await this.jobManager.getLogs(name);
        return { logs };
      } catch (error) {
        return reply.code(500).send({ error: `Failed to get logs: ${error}` });
      }
    });

    // Admin API: Get provider config (for opencode)
    server.get('/admin/provider', async () => {
      const runningModels = this.registry.getRunningModels();
      return {
        provider: 'isambard-ivllm',
        baseUrl: `http://${this.options.host}:${this.options.port}`,
        models: runningModels.map((m) => m.name),
        env: {
          OPENAI_BASE_URL: `http://${this.options.host}:${this.options.port}`,
        },
      };
    });

    // Admin API: Get model health
    server.get('/admin/models/:name/health', async (request, reply) => {
      const { name } = request.params as { name: string };
      const model = this.registry.getModel(name);

      if (!model) {
        return reply.code(404).send({ error: `Model '${name}' not found` });
      }

      if (model.runtime?.status !== 'running' || !model.runtime?.port) {
        return reply.code(400).send({ error: `Model '${name}' is not running` });
      }

      try {
        // Check health endpoint on vLLM backend
        const nodeHostname = model.runtime.nodeHostname || 'localhost';
        const port = model.runtime.port;
        const healthUrl = `http://${nodeHostname}:${port}/health`;

        const healthResponse = await new Promise<any>((resolve, reject) => {
          http.get(healthUrl, { timeout: 5000 }, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
              try {
                resolve({ statusCode: res.statusCode, body: JSON.parse(data) });
              } catch {
                resolve({ statusCode: res.statusCode, body: { raw: data } });
              }
            });
          }).on('error', reject);
        });

        // Update last checked timestamp
        this.registry.updateState(name, { lastActivityAt: new Date() });

        return {
          status: 'healthy',
          model: name,
          backend: healthUrl,
          response: healthResponse,
        };
      } catch (error) {
        return reply.code(500).send({
          status: 'unhealthy',
          model: name,
          error: `Health check failed: ${error}`,
        });
      }
    });

    return server;
  }

  /**
   * Start a model (internal helper)
   */
  private async startModel(name: string): Promise<void> {
    const model = this.registry.getModel(name);
    if (!model) throw new Error(`Model ${name} not found`);

    this.server?.log.info(`Starting model ${name}...`);

    // Submit SLURM job
    await this.jobManager.submitJob(name, model.configPath);

    // Poll until running or failed
    let attempts = 0;
    const maxAttempts = 120; // 10 minutes max (120 * 5s)

    while (attempts < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 5000)); // Poll every 5s

      try {
        const status = await this.jobManager.pollJobStatus(name);
        
        if (status.status === 'running') {
          this.server?.log.info(`Model ${name} is now running on ${status.node_hostname || 'unknown'}:${status.server_port}`);
          // Initialize lastActivityAt for idle timeout tracking
          this.registry.updateState(name, { lastActivityAt: new Date() });
          break;
        }
        
        if (status.status === 'failed' || status.status === 'timeout') {
          const errorMsg = status.error || 'unknown error';
          this.server?.log.error(`Model ${name} startup failed: ${errorMsg}`);
          throw new Error(`Model startup failed: ${errorMsg}`);
        }

        if (attempts % 12 === 0) { // Log progress every minute
          this.server?.log.info(`Model ${name} still starting... (${attempts * 5}s)`);
        }
      } catch (error) {
        this.server?.log.error(`Error polling ${name}: ${error}`);
        throw error;
      }

      attempts++;
    }

    if (attempts >= maxAttempts) {
      this.server?.log.error(`Model ${name} startup timeout after 10 minutes`);
      throw new Error('Model startup timeout');
    }
  }

  /**
   * Check and enforce idle timeouts
   */
  private checkIdleTimeouts(): void {
    const runningModels = this.registry.getRunningModels();
    const now = Date.now();

    runningModels.forEach(async (model) => {
      const timeout = model.idleTimeoutMinutes ?? 15;
      if (timeout < 0) return; // Never timeout

      const lastActivity = model.runtime?.lastActivityAt?.getTime() ?? 0;
      const idleTime = now - lastActivity;

      if (idleTime > timeout * 60 * 1000) {
        this.server?.log.info(`Model ${model.name} idle for ${Math.floor(idleTime / 60000)}m, shutting down`);
        try {
          await this.jobManager.cancelJob(model.name);
        } catch (error) {
          this.server?.log.error(`Failed to shutdown idle model ${model.name}: ${error}`);
        }
      }
    });
  }

  /**
   * Proxy a request to a vLLM backend
   */
  private async proxyRequest(modelName: string, body: any): Promise<{ statusCode: number; body: any }> {
    const model = this.registry.getModel(modelName);
    if (!model?.runtime?.port) {
      throw new Error('Model port not available');
    }

    // For now, assume direct network access to COMPUTE node
    // In future, may need SSH tunnel for laptop deployment
    const nodeHostname = model.runtime.nodeHostname || 'localhost';
    const port = model.runtime.port;
    const url = `http://${nodeHostname}:${port}/v1/chat/completions`;

    return new Promise((resolve, reject) => {
      const postData = JSON.stringify(body);
      
      const req = http.request(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData),
        },
        timeout: 120000, // 2 minute timeout
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            const responseBody = JSON.parse(data);
            resolve({ statusCode: res.statusCode || 200, body: responseBody });
          } catch {
            resolve({ statusCode: res.statusCode || 200, body: { raw: data } });
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });

      req.write(postData);
      req.end();
    });
  }
}
