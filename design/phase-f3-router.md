# Phase F3 — Model Routing Server

## Overview

A local HTTP server that acts as an OpenAI API-compatible proxy, managing multiple vLLM instances running on Isambard AI COMPUTE nodes. Designed for agent orchestration scenarios where multiple AI coding assistants (opencode, Copilot, Claude) need access to different models without manual start/stop overhead.

## Architecture

### Deployment Model (MVP — Option D)

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│   AI Agents     │      │   ivllm router   │      │  Isambard AI   │
│   (opencode,    │─────▶│   (user laptop)  │─────▶│  LOGIN node    │
│   Copilot, etc) │ HTTP │   Port 11434     │ SSH  │  (SSH, SLURM)  │
└─────────────────┘      └──────────────────┘      └────────────────┘
         ▲                                                │
         │                                                │ SLURM
         │                                                ▼
         │                                         ┌────────────────┐
         │                                         │  COMPUTE nodes │
         │                                         │  (vLLM serve)  │
         │                                         │  Ports 11435+  │
         │                                         └────────────────┘
         │                                                │
         └────────────────────────────────────────────────┘
                    SSH Tunnel (agent → laptop → COMPUTE)
```

**Key characteristics:**
- Router runs on user's laptop or local server
- Agents connect to `http://localhost:11434`
- Router manages SLURM jobs via SSH to Isambard LOGIN
- Agents SSH-tunnel through router to COMPUTE nodes (or router proxies)

### Future Deployment (Login Node Mode)

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│   AI Agents     │      │  Isambard AI     │      │  COMPUTE nodes │
│   (opencode,    │─────▶│  LOGIN node      │─────▶│  (vLLM serve)  │
│   Copilot, etc) │ SSH  │  ivllm router    │ LAN  │  Ports 11435+  │
└─────────────────┘ tunnel│  Port 11434      │      └────────────────┘
                          └──────────────────┘
```

**Migration path:**
- Abstract SSH layer (interface: `RemoteExecutor`)
- Two implementations: `SSHExecutor` (laptop) and `LocalExecutor` (login node)
- Config flag: `mode: "remote" | "local"`
- Router business logic remains identical

---

## Model Registry

### Config File: `~/.config/ivllm/models.json`

```json
{
  "models": {
    "qwen3.5-0.5b": {
      "configPath": "/home/vp22681/Git/isambard-vllm/examples/qwen3.5-0.5b.yaml",
      "idleTimeoutMinutes": -1,
      "autoStart": true
    },
    "qwen3.5-397b": {
      "configPath": "/home/vp22681/Git/isambard-vllm/examples/qwen3.5-397b.yaml",
      "idleTimeoutMinutes": 60,
      "autoStart": false
    }
  }
}
```

**Fields:**
- `configPath`: Absolute path to vLLM config YAML (same format as `ivllm start`)
- `idleTimeoutMinutes`: Auto-shutdown after N minutes of inactivity (`-1` = never)
- `autoStart`: If `true`, router automatically starts model on first request (default: `false`). Useful for frequently-used models that should start on-demand, while expensive models require explicit `/admin/models/:name/start` calls.

### Runtime State (In-Memory)

```typescript
interface ModelState {
  name: string;
  status: 'stopped' | 'starting' | 'running' | 'stopping' | 'failed';
  port?: number;           // Assigned from pool (11435–11534)
  slurmJobId?: number;     // For scancel
  nodeHostname?: string;   // From job_details.json
  startedAt?: Date;        // When status became 'running'
  lastActivityAt?: Date;   // Last proxied request
  timeoutId?: NodeJS.Timeout;  // For idle shutdown
  healthCheckInterval?: NodeJS.Timeout;
  error?: string;
}
```

**Port pool:**
- Default range: 11435–11534 (100 ports)
- Configurable in `~/.config/ivllm/config.json`
- Assigned dynamically from available pool
- Released on model shutdown

---

## HTTP API

### OpenAI-Compatible Endpoints (Proxied)

These endpoints are **pass-through** to the appropriate vLLM backend:

#### `GET /v1/models`
Returns list of **running** models in OpenAI format:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-397b",
      "object": "model",
      "created": 1717588800,
      "owned_by": "isambard-ivllm"
    },
    {
      "id": "qwen3.5-0.5b",
      "object": "model",
      "created": 1717588800,
      "owned_by": "isambard-ivllm"
    }
  ]
}
```

**Note:** Only models with `status: 'running'` are included.

#### `POST /v1/chat/completions`
Proxied to the vLLM backend for the specified model:
```json
{
  "model": "qwen3.5-397b",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Lazy startup behavior:**
- If model is `stopped`: submit SLURM job, return `503 Service Unavailable`
- Response: `{"error": {"message": "Starting up qwen3.5-397b, retry in 30s", "type": "startup_in_progress"}}`
- Agent should retry after 30–60 seconds
- Once running, proxy requests normally

#### `POST /v1/completions`
Legacy completions API (same lazy startup behavior).

#### `GET /v1/models/:model`
Get details for a specific model (proxied to vLLM or router state).

---

### Admin Endpoints (Router Management)

These endpoints manage the router itself:

#### `GET /admin/models`
List all configured models with status:
```json
{
  "models": [
    {
      "name": "qwen3.5-397b",
      "status": "running",
      "port": 11435,
      "slurmJobId": 12345,
      "nodeHostname": "gh200-001.isambard.ac.uk",
      "startedAt": "2026-06-05T12:00:00Z",
      "lastActivityAt": "2026-06-05T12:30:00Z",
      "idleTimeoutMinutes": 60
    },
    {
      "name": "qwen3.5-0.5b",
      "status": "stopped",
      "idleTimeoutMinutes": -1
    }
  ]
}
```

#### `POST /admin/models`
Add a new model configuration:
```bash
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "deepseek-v4-pro",
    "configPath": "/path/to/deepseek-v4-pro.yaml",
    "idleTimeoutMinutes": 30
  }'
```

**Validation:**
- `name`: unique, alphanumeric + hyphens
- `configPath`: must exist and be valid YAML
- `idleTimeoutMinutes`: integer, -1 or positive

#### `DELETE /admin/models/:name`
Remove a model configuration (must be stopped first):
```bash
curl -X DELETE http://localhost:11434/admin/models/qwen3.5-397b
```

**Errors:**
- `400 Bad Request` if model is running
- `404 Not Found` if model doesn't exist

#### `POST /admin/models/:name/start`
Explicitly start a model:
```bash
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/start
```

**Behavior:**
- Submits SLURM job via SSH
- Assigns port from pool
- Polls `job_details.json` until running
- Returns `200 OK` with model state

**Errors:**
- `400 Bad Request` if already running
- `500 Internal Server Error` if SLURM submission fails

#### `POST /admin/models/:name/stop`
Stop a running model:
```bash
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/stop
```

**Behavior:**
- Sends `scancel` for SLURM job
- Removes `job_details.json`
- Releases port back to pool
- Returns `200 OK`

**Errors:**
- `400 Bad Request` if already stopped
- `500 Internal Server Error` if cleanup fails

#### `GET /admin/models/:name/logs`
Stream vLLM logs (tail -f style):
```bash
curl http://localhost:11434/admin/models/qwen3.5-397b/logs
```

**Implementation:**
- SSH to COMPUTE node, `tail -f <log_path>`
- Stream via Server-Sent Events (SSE) or chunked transfer

#### `GET /admin/provider`
Return opencode-compatible provider configuration:
```json
{
  "provider": "isambard-ivllm",
  "baseUrl": "http://localhost:11434",
  "models": ["qwen3.5-397b", "qwen3.5-0.5b"],
  "env": {
    "OPENAI_BASE_URL": "http://localhost:11434"
  }
}
```

**Usage:**
- Agents can auto-configure by fetching this endpoint
- Returns all currently running models

---

## Lifecycle Management

### Lazy Startup Flow

1. Agent calls `POST /v1/chat/completions` with `model: "qwen3.5-397b"`
2. Router checks model state:
   - `running`: proxy to vLLM backend immediately
   - `stopped` with `autoStart: true`: initiate startup, return `503 Service Unavailable`
   - `stopped` with `autoStart: false`: return `400 Bad Request` with `{"error": "Model 'qwen3.5-397b' is not running. Start it via POST /admin/models/qwen3.5-397b/start"}`
   - `starting`: return `503 Service Unavailable` with `{"error": "Starting up qwen3.5-397b, retry in 30s"}`
3. On startup:
   - Submit SLURM job via SSH
   - Poll `job_details.json` every 5s
   - On `status: "running"`: start health check, reset idle timer
   - Proxy queued request (if any)

### Idle Timeout Flow

1. Each proxied request updates `lastActivityAt`
2. Background checker runs every 60s:
   ```typescript
   if (now - model.lastActivityAt > model.idleTimeoutMinutes * 60 * 1000) {
     await stopModel(model.name);
   }
   ```
3. Models with `idleTimeoutMinutes: -1` are never auto-stopped
4. Before stopping: optional warning log entry

### Shutdown Flow (Router Exit)

On `SIGINT` / `SIGTERM` / Ctrl+C:

1. **Stop accepting new requests** (close HTTP server)
2. **Stop all running models**:
   ```bash
   for model in runningModels:
     scancel <slurmJobId>
     rm job_details.json
   ```
3. **Clear runtime state** (in-memory only, no persistence needed)
4. **Exit cleanly**

**No persistence required** — hard cleanup is the design goal.

---

## Implementation Plan

### F3.1 — Project Scaffold
- [ ] Create `src/router/` directory
- [ ] HTTP server setup (Express or Fastify?)
- [ ] Basic routing structure
- [ ] Config loader (`~/.config/ivllm/models.json`)
- [ ] Unit tests for config parsing

### F3.2 — Model Registry
- [ ] `ModelRegistry` class (CRUD operations)
- [ ] Port pool manager (11435–11534)
- [ ] ModelState tracking (in-memory)
- [ ] Unit tests for registry operations

### F3.3 — SLURM Integration
- [ ] Reuse `renderInferenceScript` from existing `ivllm start`
- [ ] SSH wrapper (abstract for future login-node mode)
- [ ] Job submission, polling, cleanup
- [ ] Integration tests (mock SLURM)

### F3.4 — OpenAI API Proxy
- [ ] `GET /v1/models` endpoint
- [ ] `POST /v1/chat/completions` proxy
- [ ] Request/response transformation (if needed)
- [ ] Error handling (timeouts, connection refused)

### F3.5 — Admin API
- [ ] `GET /admin/models` — list all
- [ ] `POST /admin/models` — add model
- [ ] `DELETE /admin/models/:name` — remove
- [ ] `POST /admin/models/:name/start` — explicit start
- [ ] `POST /admin/models/:name/stop` — explicit stop
- [ ] `GET /admin/models/:name/logs` — log streaming
- [ ] `GET /admin/provider` — opencode config

### F3.6 — Lifecycle Management
- [ ] Lazy startup implementation
- [ ] Idle timeout checker (background interval)
- [ ] Auto-start flag support
- [ ] Health check loop (per-model)

### F3.7 — CLI Wrapper
- [ ] `ivllm router` command (starts HTTP server)
- [ ] Optional convenience commands:
  - `ivllm router add <config.yaml>`
  - `ivllm router start <model>`
  - `ivllm router stop <model>`
  - `ivllm router status`

### F3.8 — Documentation
- [ ] API endpoint reference
- [ ] Model registry configuration guide
- [ ] Agent integration examples (opencode, Copilot, Claude)
- [ ] Troubleshooting guide (SSH timeout, port conflicts)

### F3.9 — End-to-End Testing
- [ ] Single model lazy startup
- [ ] Multiple concurrent models
- [ ] Idle timeout behavior
- [ ] Router shutdown cleanup
- [ ] Agent integration test (opencode)

---

## Technical Decisions

### HTTP Framework: Express vs. Fastify

**Recommendation: Fastify**
- Lower overhead, better for proxy workloads
- Built-in schema validation
- TypeScript support excellent
- Same ecosystem as Express

**Alternative: Express**
- More familiar, larger community
- Slightly higher overhead (negligible for this use case)

### SSH Abstraction

**Interface-based design:**
```typescript
interface RemoteExecutor {
  runCommand(command: string): Promise<string>;
  copyFile(localPath: string, remotePath: string): Promise<void>;
  readFile(remotePath: string): Promise<string>;
  fileExists(remotePath: string): Promise<boolean>;
}

class SSHExecutor implements RemoteExecutor { /* existing SSH logic */ }
class LocalExecutor implements RemoteExecutor { /* bash.exec, fs.copyFile */ }
```

**Router uses dependency injection:**
```typescript
const executor = config.mode === 'remote'
  ? new SSHExecutor(config.loginHost)
  : new LocalExecutor();

const router = new Router(executor);
```

### Port Conflict Resolution

**Strategy:**
1. Maintain in-memory set of assigned ports
2. Before assigning: quick `netstat` check (optional)
3. On collision: try next port in pool
4. Max retries: 10, then fail with `500 No available ports`

---

## Known Limitations (MVP)

1. **12-hour SSH timeout** — Router on laptop requires SSH reconnection after 12h
2. **No persistence** — Router restart loses model state (manual recovery required)
3. **Single-user** — No authentication, assumes trusted local network
4. **No rate limiting** — Agents can overwhelm router (future enhancement)
5. **No metrics** — No Prometheus/statsd integration (future enhancement)

---

## Success Criteria

- [ ] Agents can connect to `http://localhost:11434` and discover models via `GET /v1/models`
- [ ] Lazy startup works end-to-end (agent request → SLURM job → proxied response)
- [ ] Idle timeout shuts down unused models automatically
- [ ] Router shutdown cleans up all SLURM jobs
- [ ] Multiple concurrent models supported (port isolation)
- [ ] All 50+ unit tests passing

---

## Future Enhancements (Post-F3)

- **Login node mode** — Run router on Isambard LOGIN (no SSH timeout)
- **Job manager service** — Decouple SSH operations from router
- **Authentication** — API keys for multi-user scenarios
- **Metrics dashboard** — Real-time model usage, GPU utilization
- **Scheduled startup** — Cron-like model warm-up
- **Model snapshotting** — Save/restore KV cache (if vLLM adds support)
- **Load balancing** — Multiple instances of same model
- **Agent launcher integration** — Spin off agent launcher to separate project, rearchitect as router client

---

## Agent Launcher Separation

The agent launcher feature (Phase F2.6/F2.6b) will be **spun off into a separate project** and rearchitected to use the model router as its backend.

**Rationale:**
- Router is a persistent service; agent launcher is a CLI convenience tool
- Separation of concerns: router manages models, launcher manages agents
- Cleaner architecture: launcher becomes a thin client that queries router's `/admin/provider` endpoint

**Future workflow:**
```bash
# Router runs as service
ivllm router

# Agent launcher (separate project) queries router
agent-launcher opencode
# → Fetches available models from http://localhost:11434/v1/models
# → Launches agent with correct OPENAI_BASE_URL
```

**Implementation:**
- Create separate repository: `isambard-agent-launcher` (or similar)
- Deprecate agent launcher code in this fork
- Router remains the canonical model management service
