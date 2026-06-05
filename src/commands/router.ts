import { RouterService } from '../router/router-service.js';
import { SSHExecutor } from '../router/executor.js';

/**
 * CLI command: ivllm router
 * Starts the model router HTTP server
 */
export async function cmdRouter(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: ivllm router [options]

Start the model router HTTP server for multi-model orchestration.

Options:
  --port <port>           HTTP port for router (default: 11434)
  --host <host>           Bind address (default: 127.0.0.1)
  --login-host <host>     SSH login node (required, or set ISAMBARDC_LOGIN_HOST env)
  --help, -h              Show this help message

Examples:
  ivllm router
  ivllm router --port 8080
  ivllm router --login-host login.isambard.ac.uk

The router manages multiple vLLM instances on Isambard AI COMPUTE nodes.
Agents connect to http://localhost:<port> for OpenAI API access.
`);
    return;
  }

  const options: Record<string, string> = {};
  for (let i = 0; i < args.length - 1; i++) {
    if (args[i]?.startsWith("--")) options[args[i]!.slice(2)] = args[i + 1] ?? "";
  }

  const port = options["port"] ? parseInt(options["port"]!, 10) : undefined;
  const host = options["host"];
  const loginHost = options["login-host"] || process.env.ISAMBARDC_LOGIN_HOST;
  
  if (!loginHost) {
    console.error('Error: --login-host is required or set ISAMBARDC_LOGIN_HOST environment variable');
    process.exit(1);
  }

  const executor = new SSHExecutor(loginHost);
  const router = new RouterService({
    port,
    host,
    executor,
    loginHost,
  });

  // Handle shutdown signals
  const shutdown = async () => {
    console.log('\nShutting down router...');
    await router.stop();
    process.exit(0);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  try {
    await router.start();
    console.log(`Router running at http://${host || '127.0.0.1'}:${port || 11434}`);
    console.log('Press Ctrl+C to stop');
    console.log('');
    console.log('Available endpoints:');
    console.log('  GET  /health              - Health check');
    console.log('  GET  /v1/models           - List running models');
    console.log('  POST /v1/chat/completions - Chat with models (lazy startup)');
    console.log('  GET  /admin/models        - List all configured models');
    console.log('  GET  /admin/provider      - Get provider config for agents');
  } catch (error) {
    console.error('Failed to start router:', error);
    process.exit(1);
  }
}
