import type { LocalOps, V1ModelsResponse } from './types.ts';

/**
 * Factory that returns a {@link LocalOps} implementation.
 *
 * Produces real HTTP / `lsof`-based ops when `dryRun` is `false`, or mock
 * implementations that return synthetic responses when `dryRun` is `true`
 * (used for E2E testing without a live vLLM server).
 *
 * **Real-mode behaviour**
 *
 * | Method | Implementation |
 * |--------|----------------|
 * | `checkLocalHealth` | HTTP `GET /health` with 200 ms timeout |
 * | `queryModels`      | HTTP `GET /v1/models` with 2 s timeout |
 * | `isLocalPortInUse` | `lsof -ti` + `ps -p` to report PID and process name |
 *
 * **Dry-run behaviour**
 *
 * All methods return success/fake data and log `[dry-run]` prefixes.
 * @param localPort - Port the vLLM server listens on locally (default `11434`)
 * @param dryRun - When `true` return mock implementations for testing
 * @returns An object conforming to the {@link LocalOps} interface
 * @example
 * ```ts
 * // Real-mode: check if a vLLM server is healthy
 * const ops = makeLocalOps(11434, false);
 * const healthy = await ops.checkLocalHealth(11434);
 *
 * // Dry-run mode: E2E tests without a live server
 * const mock = makeLocalOps(8000, true);
 * const models = await mock.queryModels(8000);
 * ```
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
 * Detect whether a TCP listener is bound on `localhost` at the given
 * `port` by querying the OS through `lsof` and `ps`.
 *
 * Returns an object containing the PID and process name of the listener,
 * or `null` if nothing is listening.
 *
 * **Platform requirement:** Linux or macOS (relies on `lsof`). Not
 * available on Windows.
 * @param port — TCP port number to check.
 * @returns `{ pid, process }` when a listener exists, or `null`.
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

/**
 * Probe the `/health` endpoint on `localhost:<localPort>` and return
 * `true` if the HTTP server responds with a 2xx status before the
 * timeout elapses.
 *
 * Uses a `Promise.race` between an HTTP `fetch` and a `setTimeout`
 * so the call never blocks longer than `timeoutMs`.
 * @param localPort — Port forwarded by the SSH tunnel.
 * @param timeoutMs — Maximum milliseconds to wait before returning `false`.
 * @returns `true` when the health endpoint is reachable within the timeout;
 *   `false` on timeout or network error.
 */
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

/**
 * Fetch the model catalog from the OpenAI-compatible `/v1/models`
 * endpoint on `localhost:<localPort>`.
 *
 * Throws an `Error` containing the HTTP status and reason phrase if the
 * response is non-2xx or if the request times out.
 * @param localPort — Port forwarded by the SSH tunnel.
 * @param timeoutMs — Maximum milliseconds before throwing.
 * @returns The parsed {@link V1ModelsResponse} body.
 * @throws Error on non-2xx HTTP status or timeout.
 */
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
