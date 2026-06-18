import type {
  CloseableEventEmitter,
  Credentials,
  LocalOps,
  InferenceJobOptions,
  V1ModelsResponse,
} from './types.ts';

/**
 * Returns real RemoteOps that execute SSH/SCP, or dry-run ops that print
 * what would happen and copy files to a local preview directory.
 * @param config
 * @param dryRun
 * @param dryRunDir
 */
export function makeLocalOps(localPort: number, dryRun: boolean): LocalOps {
  if (!dryRun) {
    return {
      checkLocalHealth: (localPort) => {
        return isHealthy(localPort, 200);
      },
      queryModels: (localPort) => {
        return queryModels(localPort, 2000);
      },
      isLocalPortInUse: (localPort) => {
        return isLocalPortInUse(localPort);
      },
    };
  }

  // Mock localhost network actions for E2E testing:
  return {
    async checkLocalHealth(localPort) {
      console.log(`  [dry-run] heartbeat found on ${localPort}`);
      return true;
    },
    async queryModels(localPort) {
      return {
        object: 'list',
        data: [{ id: `test model on ${localPort}` }],
      };
    },
    async isLocalPortInUse(localPort) {
      console.log(`  [dry-run] local port check: ${localPort}`);
      return null;
    },
  };
}

/**
 *
 * @param port
 */
async function isLocalPortInUse(
  port: number,
): Promise<{ pid: string; process: string } | null> {
  const { execFile } = await import('child_process');
  return new Promise((resolve) => {
    execFile('lsof', ['-ti', `:${port}`, '-sTCP:LISTEN'], (err, stdout) => {
      if (err || !stdout.trim()) {
        resolve(null);
        return;
      }
      const pid = stdout.trim().split('\n')[0] as string;
      execFile('ps', ['-p', pid, '-o', 'comm='], (_err2, psOut) => {
        const process = psOut?.trim() || 'unknown';
        resolve({ pid, process });
      });
    });
  });
}

// Check whether localhost is up.
async function isHealthy(
  localPort: number,
  timeoutMs: number,
): Promise<boolean> {
  // Create a promise that resolves to false after the timeout
  const timeout = new Promise<boolean>((resolve) =>
    setTimeout(() => resolve(false), timeoutMs),
  );

  // Create a promise that tries to fetch the health status
  const networkRequest = fetch(`http://localhost:${localPort}/health`)
    .then((res) => res.ok) // Returns true if status is 200-299
    .catch(() => false); // Returns false if server is down / network error

  // Returns whichever one finishes first
  return Promise.race([networkRequest, timeout]);
}

// Grab the model json from localhost
async function queryModels(
  localPort: number,
  timeoutMs: number,
): Promise<V1ModelsResponse> {
  const timeout = new Promise<Response>((resolve) =>
    setTimeout(() => {
      throw new Error('Timed out');
    }, timeoutMs),
  );

  const request = fetch(`http://localhost:${localPort}/v1/models`, {});

  const response = await Promise.race([request, timeout]);

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const json = (await response.json()) as V1ModelsResponse;

  return json;
}
