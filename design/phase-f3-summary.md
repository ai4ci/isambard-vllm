# Phase F3 — Model Router Implementation Summary

## Overview

Phase F3 implements an HTTP server that manages multiple vLLM instances running on Isambard AI COMPUTE nodes. The router provides an OpenAI API-compatible interface for AI coding assistants (Opencode, Copilot, Claude) to discover and use models without manual start/stop overhead.

**Status:** Core implementation complete ✅  
**Branch:** `feature/phase-f3-router`  
**Tests:** 423 passing (39 new router tests)  
**Version:** v1.0.0 (ready for E2E validation)

## Architecture

```
┌─────────────┐     SSH      ┌──────────┐      ┌─────────────┐
│   Laptop    │ ───────────> │  Login   │ ───> │ COMPUTE     │
│  (Router)   │              │  Node    │      │ (vLLM)      │
│ :11434      │              │          │      │ :11435      │
└─────────────┘              └──────────┘      └─────────────┘
      │
      │ OpenAI API
      ▼
┌─────────────┐
│   Agents    │
│ Opencode,   │
│ Copilot,    │
│ Claude      │
└─────────────┘
```

### Key Components

1. **Router Service** (`src/router/router-service.ts`)
   - Fastify HTTP server on `http://localhost:11434`
   - OpenAI-compatible `/v1/chat/completions` proxy
   - Admin API for lifecycle management
   - Idle timeout checker (60s interval)

2. **Model Registry** (`src/router/model-registry.ts`)
   - Config: `~/.config/ivllm/models.json` (persistent)
   - State: In-memory (ephemeral)
   - Port pool: 11435-11534 (dynamic assignment)

3. **Job Manager** (`src/router/job-manager.ts`)
   - SLURM job submission via SSH
   - Status polling (5s interval)
   - Graceful cancellation on shutdown

4. **Executor Abstraction** (`src/router/executor.ts`)
   - `RemoteExecutor` interface
   - `SSHExecutor` for laptop deployment (current)
   - `LocalExecutor` for login-node mode (future)

## Implemented Features

### ✅ F3.1 — Project Scaffold
- Fastify HTTP server with TypeScript
- Config loader for `~/.config/ivllm/models.json`
- Health check endpoint
- Graceful shutdown

### ✅ F3.2 — Model Registry
- CRUD operations: add, remove, get, list
- Port pool manager (100 ports, dynamic assignment)
- ModelState tracking (stopped, starting, running, failed)
- Config validation

### ✅ F3.3 — SLURM Integration
- SSH-based job submission
- Job status polling via `job_details.json`
- Log retrieval
- Hard cleanup on router shutdown

### ✅ F3.4 — OpenAI API Proxy
- `POST /v1/chat/completions` forwarding to vLLM backends
- Native `http` module for HTTP proxying
- `lastActivityAt` timestamp updates
- Error handling (502 Bad Gateway)

### ✅ F3.5 — Admin API
- `GET /admin/models` — List all models with status
- `POST /admin/models` — Add model configuration
- `DELETE /admin/models/:name` — Remove model
- `POST /admin/models/:name/start` — Explicit start
- `POST /admin/models/:name/stop` — Explicit stop
- `GET /admin/models/:name/logs` — Log retrieval
- `GET /admin/models/:name/health` — Health check
- `GET /admin/provider` — Opencode provider config

### ✅ F3.6 — Lifecycle Management
- Idle timeout checker (60s interval)
- Lazy startup (503 response during startup)
- Enhanced `startModel()` with progress logging
- Health check endpoint for per-model monitoring
- `lastActivityAt` initialization on startup

### ✅ F3.7 — CLI Wrapper
- `ivllm router` command
- Options: `--port`, `--host`, `--login-host`
- Help text with examples
- Graceful shutdown on SIGINT/SIGTERM

### ✅ F3.8 — Documentation
- Comprehensive usage guide (`design/router.md`)
- API reference for all endpoints
- Agent integration examples (Opencode, Copilot, Claude)
- Python and Bash client examples
- Best practices for model selection
- Troubleshooting guide

## API Reference

### OpenAI-Compatible Endpoints

#### `GET /v1/models`
List available models (from router registry).

#### `POST /v1/chat/completions`
Send chat completion request. Supports lazy startup:
- **200 OK**: Model running, response proxied
- **503 Service Unavailable**: Model starting, retry in 30s
- **400 Bad Request**: Model not configured or autoStart disabled
- **502 Bad Gateway**: Proxy failed (connection error)

### Admin Endpoints

#### `GET /admin/models`
List all configured models with status, ports, and idle timeout settings.

#### `POST /admin/models`
Add a new model configuration:
```json
{
  "name": "qwen3.5-397b",
  "configPath": "/path/to/config.yaml",
  "idleTimeoutMinutes": 60,
  "autoStart": false
}
```

#### `DELETE /admin/models/:name`
Remove a model (must be stopped first).

#### `POST /admin/models/:name/start`
Explicitly start a model.

#### `POST /admin/models/:name/stop`
Stop a running model.

#### `GET /admin/models/:name/health`
Health check for a specific model:
```json
{
  "status": "healthy",
  "model": "qwen3.5-397b",
  "backend": "http://compute-node:11435/health",
  "response": { "statusCode": 200 }
}
```

#### `GET /admin/provider`
Get provider configuration for agents:
```json
{
  "provider": "isambard-ivllm",
  "baseUrl": "http://127.0.0.1:11434",
  "models": ["qwen3.5-397b", "qwen3.5-0.5b"],
  "env": {
    "OPENAI_BASE_URL": "http://127.0.0.1:11434"
  }
}
```

## Usage Examples

### Quick Start

```bash
# 1. Start the router
export ISAMBARDC_LOGIN_HOST=login.isambard.ac.uk
ivllm router

# 2. Add a model
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen3.5-397b",
    "configPath": "/path/to/qwen3.5-397b.yaml",
    "idleTimeoutMinutes": 60,
    "autoStart": false
  }'

# 3. Start the model
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/start

# 4. Connect your agent
export OPENAI_BASE_URL=http://localhost:11434
opencode
```

### Opencode Integration

```bash
# Set environment variable
export OPENAI_BASE_URL=$(curl -s http://localhost:11434/admin/provider | jq -r '.env.OPENAI_BASE_URL')

# Run opencode
opencode
```

### Python Client

```python
import requests
import time

def chat(message, model="qwen3.5-397b", max_retries=10):
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}]
    }
    
    for attempt in range(max_retries):
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 503:
            print(f"Model starting... ({attempt + 1}/{max_retries})")
            time.sleep(30)
        else:
            raise Exception(f"Error {response.status_code}")
    
    raise Exception("Max retries exceeded")
```

## Model Configuration

### Model Registry File

Location: `~/.config/ivllm/models.json`

```json
{
  "models": {
    "qwen3.5-0.5b": {
      "configPath": "/path/to/qwen3.5-0.5b.yaml",
      "idleTimeoutMinutes": -1,
      "autoStart": true
    },
    "qwen3.5-397b": {
      "configPath": "/path/to/qwen3.5-397b.yaml",
      "idleTimeoutMinutes": 60,
      "autoStart": false
    }
  }
}
```

### Configuration Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique model identifier |
| `configPath` | string | Yes | Path to vLLM config YAML |
| `idleTimeoutMinutes` | number | No | Minutes before auto-shutdown (-1 = never) |
| `autoStart` | boolean | No | Allow lazy startup on request (default: false) |

### Best Practices

**Always-warm models (small, frequently used):**
```json
{
  "idleTimeoutMinutes": -1,
  "autoStart": true
}
```

**On-demand models (large, expensive):**
```json
{
  "idleTimeoutMinutes": 30,
  "autoStart": true
}
```

**Explicit-only models (require manual start):**
```json
{
  "idleTimeoutMinutes": 15,
  "autoStart": false
}
```

## Lifecycle Flows

### Lazy Startup

1. Agent sends `POST /v1/chat/completions` with `model: "qwen3.5-397b"`
2. Router checks model state:
   - **running**: Proxy request immediately
   - **stopped** + `autoStart: true`: Submit SLURM job, return 503
   - **stopped** + `autoStart: false`: Return 400
   - **starting**: Return 503
3. Agent retries after 30s
4. Once running, proxy requests normally

### Idle Timeout

1. Each proxied request updates `lastActivityAt`
2. Background checker runs every 60s:
   ```
   idleTime = now - lastActivityAt
   if idleTime > idleTimeoutMinutes * 60000:
     shutdown model
   ```
3. Models with `idleTimeoutMinutes: -1` never auto-shutdown

### Router Shutdown

On SIGINT/SIGTERM/Ctrl+C:
1. Stop accepting new requests
2. Cancel all running SLURM jobs
3. Release all ports back to pool
4. Exit cleanly

**No persistence** — router state is in-memory only.

## Port Management

- **Range:** 11435–11534 (100 ports)
- **Assignment:** Dynamic, first-available
- **Release:** On model shutdown
- **Tracking:** In-memory `PortPoolManager`

## Known Limitations

1. **12-hour SSH timeout** — Laptop deployment requires SSH reconnection. Future: login-node deployment.
2. **No persistence** — Router restart loses model state (ports, job IDs). Config survives.
3. **Single-user** — No authentication or multi-tenancy.
4. **SSH tunnel for proxy** — Currently assumes direct network access to COMPUTE nodes. May need SSH port forwarding for laptop deployment.

## Testing

### Unit Tests (39 new tests)

- **Port Pool Manager:** 9 tests
- **Model Registry:** 14 tests
- **Registry I/O:** 8 tests
- **Server:** 5 tests
- **Integration:** 3 tests

**Total:** 423 tests passing (was 384)

### Running Tests

```bash
# All tests
bun test

# Router tests only
bun test tests/router/

# With coverage
bun test --coverage
```

## Remaining Work

### F3.9 — End-to-End Testing

**Pending validation on Isambard AI:**
- [ ] Single model lazy startup
- [ ] Multiple concurrent models
- [ ] Idle timeout behavior
- [ ] Router shutdown cleanup
- [ ] Agent integration (Opencode, Copilot, Claude)

**Blockers:**
- Requires Isambard AI access
- SLURM queue time
- Network access to COMPUTE nodes

### Future Enhancements

- SSH tunnel implementation for laptop deployment
- Enhanced structured logging
- Login-node deployment mode
- Persistence layer for runtime state
- Authentication/multi-tenancy

## Files Created/Modified

### New Files (12)

- `src/router/types.ts` — Type definitions
- `src/router/port-pool.ts` — Port pool manager
- `src/router/registry.ts` — Model registry I/O
- `src/router/model-registry.ts` — Model registry class
- `src/router/executor.ts` — SSH/Local executor
- `src/router/job-manager.ts` — SLURM job manager
- `src/router/router-service.ts` — Main HTTP service
- `src/commands/router.ts` — CLI command
- `src/types.ts` — RemoteExecutor interface
- `design/router.md` — Comprehensive usage guide (11KB)
- `design/phase-f3-status.md` — Implementation status
- `design/phase-f3-summary.md` — This document

### Modified Files (3)

- `src/index.ts` — Added router command
- `tests/router/*.test.ts` — 39 new tests
- `design/roadmap.md` — Updated Phase F3 status

## Success Criteria

### Met ✅

- [x] Router HTTP server starts and responds
- [x] Model registry CRUD operations functional
- [x] SLURM job submission via SSH working
- [x] Admin API endpoints all implemented
- [x] Proxy layer forwards requests to vLLM backends
- [x] Health check endpoint monitors model status
- [x] Idle timeout checker runs every 60s
- [x] CLI wrapper (`ivllm router`) functional
- [x] Comprehensive documentation with agent examples
- [x] 39 new unit tests passing

### Pending ⏳

- [ ] E2E validation on Isambard AI
- [ ] Agent integration tested with real agents
- [ ] SSH tunnel for proxy (enhancement)

## Next Steps

1. **E2E Testing** — Deploy to Isambard AI and validate with real SLURM jobs
2. **Agent Testing** — Test with Opencode, Copilot, Claude
3. **SSH Tunnel** — Implement port forwarding for laptop deployment
4. **Enhanced Logging** — Add structured logging for production debugging

## Conclusion

Phase F3 core implementation is complete with 39 new tests passing. The router provides a robust foundation for agent orchestration on Isambard AI, with OpenAI API compatibility, lazy startup, idle timeout, and comprehensive admin APIs. E2E validation on Isambard AI is the final step before production readiness.

---

**Last updated:** 2026-06-05  
**Branch:** `feature/phase-f3-router`  
**Version:** v1.0.0 (core complete)  
**Tests:** 423 passing ✅
