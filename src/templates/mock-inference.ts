import type { SessionState } from '../types';

/**
 * Generates a mock SLURM script for testing vLLM inference without a GPU.
 *
 * Produces a bash script that starts a lightweight Python HTTP server
 * mimicking the vLLM `/health` and `/v1/models` endpoints. Used for
 * dry-run testing, CI validation, and local debugging.
 *
 * Script behavior:
 * 1. Writes `job_details.json` with `initialising` status
 * 2. Starts a Python mock HTTP server on the configured port
 * 3. Simulates `${startupDelaySecs}s` startup delay
 * 4. Verifies the mock server is still running
 * 5. Updates `job_details.json` to `running` status
 * 6. Waits for the server process to exit
 *
 * | SBATCH Directive | Value |
 * |------------------|-------|
 * | `--job-name` | User-provided job name |
 * | `--nodes` | `1` (single node only) |
 * | `--ntasks` | `1` |
 * | `--time` | From `{@link InferenceJobOptions.timeLimit}` |
 * @param ss - Session state containing job config, paths, and credentials
 * @returns A bash script string that can be written to a `.slurm.sh` file
 * @throws {Error} If `SessionState.startArgs` is undefined
 * @see SessionState
 * @see InferenceJobOptions
 * @see renderInferenceScript
 */
export function renderMockInferenceScript(ss: SessionState): string {
  if (ss.startArgs === undefined)
    throw new Error('Incorrectly setup SessionState.');
  const opts = ss.startArgs;
  const startupDelaySecs = 5;

  return `#!/bin/bash
#SBATCH --job-name=${opts.jobName}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=${opts.timeLimit}

set -euo pipefail

WORK_DIR="${ss.paths.remoteJobDir}"
exec > "$WORK_DIR/${opts.jobName}.slurm.log" 2>&1
JOB_DETAILS="$WORK_DIR/job_details.json"
SERVER_PORT=${opts.serverPort}
MODEL="${opts.configYaml.model}"

# Write initialising status
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${opts.jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$(hostname)" \\
  --arg model "$MODEL" \\
  --argjson server_port $SERVER_PORT \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

echo "Mock vLLM: starting HTTP server on port $SERVER_PORT for model $MODEL"

# Write Python mock HTTP server script
cat > "$WORK_DIR/mock_server.py" << 'PYEOF'
import http.server, json, os, sys

MODEL = os.environ["MOCK_MODEL"]
PORT = int(os.environ["MOCK_PORT"])

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, fmt, *args):
        pass

print(f"Mock vLLM listening on port {PORT}, model={MODEL}", flush=True)
http.server.HTTPServer(("0.0.0.0", PORT), MockHandler).serve_forever()
PYEOF

MOCK_MODEL="$MODEL" MOCK_PORT="$SERVER_PORT" python3 "$WORK_DIR/mock_server.py" &
HTTP_PID=$!

echo "Mock vLLM: simulating ${startupDelaySecs}s startup delay..."
sleep ${startupDelaySecs}

# Verify HTTP server is still running
if ! kill -0 $HTTP_PID 2>/dev/null; then
  echo "Mock HTTP server failed to start"
  jq '.status = "failed" | .error = "mock HTTP server exited during startup"' \\
    "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
  exit 1
fi

echo "Mock vLLM: ready"
jq '.status = "running"' "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"

wait $HTTP_PID
`.trimStart();
}
