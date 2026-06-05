# Phase F3 — Model Router

## Overview

The model router is an HTTP server that manages multiple vLLM instances running on Isambard AI COMPUTE nodes. It provides an OpenAI API-compatible interface for AI coding assistants to discover and use models without manual start/stop overhead.

## Quick Start

### 1. Start the Router

```bash
# Using CLI
ivllm router --login-host login.isambard.ac.uk

# Or set environment variable once
export ISAMBARDC_LOGIN_HOST=login.isambard.ac.uk
ivllm router
```

The router runs on `http://127.0.0.1:11434` by default.

### 2. Add a Model Configuration

```bash
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen3.5-397b",
    "configPath": "/home/vp22681/Git/isambard-vllm/examples/qwen3.5-397b.yaml",
    "idleTimeoutMinutes": 60,
    "autoStart": false
  }'
```

### 3. Start the Model

```bash
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/start
```

### 4. Connect Your Agent

```bash
# Get provider configuration
curl http://localhost:11434/admin/provider

# Response:
# {
#   "provider": "isambard-ivllm",
#   "baseUrl": "http://127.0.0.1:11434",
#   "models": ["qwen3.5-397b"],
#   "env": {
#     "OPENAI_BASE_URL": "http://127.0.0.1:11434"
#   }
# }
```

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ AI Agents    │─────▶│  ivllm       │─────▶│  Isambard    │
│ (opencode,   │ HTTP │  router      │ SSH  │  LOGIN       │
│ Copilot)     │      │  :11434      │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                                                 │
                                                 │ SLURM
                                                 ▼
                                          ┌──────────────┐
                                          │  COMPUTE     │
                                          │  vLLM :11435 │
                                          └──────────────┘
```

## API Reference

### OpenAI-Compatible Endpoints

These endpoints are compatible with any OpenAI API client.

#### `GET /v1/models`

List all **running** models.

```bash
curl http://localhost:11434/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3.5-397b",
      "object": "model",
      "created": 1717588800,
      "owned_by": "isambard-ivllm"
    }
  ]
}
```

#### `POST /v1/chat/completions`

Send a chat completion request. Supports lazy startup if `autoStart: true`.

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-397b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Lazy Startup Response (503):**
```json
{
  "error": {
    "message": "Starting up qwen3.5-397b, retry in 30s",
    "type": "startup_in_progress"
  }
}
```

### Admin Endpoints

#### `GET /admin/models`

List all configured models with status.

```bash
curl http://localhost:11434/admin/models
```

**Response:**
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
      "idleTimeoutMinutes": 60,
      "autoStart": false
    }
  ]
}
```

#### `POST /admin/models`

Add a new model configuration.

```bash
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "deepseek-v4-pro",
    "configPath": "/path/to/deepseek-v4-pro.yaml",
    "idleTimeoutMinutes": 30,
    "autoStart": true
  }'
```

**Fields:**
- `name` (required): Unique model identifier (alphanumeric + hyphens)
- `configPath` (required): Absolute path to vLLM config YAML
- `idleTimeoutMinutes` (optional): Auto-shutdown after N minutes (-1 = never)
- `autoStart` (optional): Auto-start on first request (default: false)

#### `DELETE /admin/models/:name`

Remove a model configuration (must be stopped first).

```bash
curl -X DELETE http://localhost:11434/admin/models/qwen3.5-397b
```

#### `POST /admin/models/:name/start`

Explicitly start a model.

```bash
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/start
```

#### `POST /admin/models/:name/stop`

Stop a running model.

```bash
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/stop
```

#### `GET /admin/models/:name/logs`

Get vLLM logs for a model.

```bash
curl http://localhost:11434/admin/models/qwen3.5-397b/logs
```

#### `GET /admin/provider`

Get provider configuration for AI assistants.

```bash
curl http://localhost:11434/admin/provider
```

**Response:**
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

## Model Configuration

### Model Registry File

Models are stored in `~/.config/ivllm/models.json`:

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

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `configPath` | string | required | Path to vLLM config YAML |
| `idleTimeoutMinutes` | number | 15 | Auto-shutdown after N minutes (-1 = never) |
| `autoStart` | boolean | false | Auto-start on first request |

## Lifecycle Management

### Lazy Startup

When `autoStart: true`, the router automatically starts models on first request:

1. Agent calls `POST /v1/chat/completions` with `model: "qwen3.5-397b"`
2. Router checks model state:
   - If `stopped`: submits SLURM job, returns `503 Starting up...`
   - If `starting`: returns `503 Starting up...`
   - If `running`: proxies request immediately
3. Agent retries after 30s once model is running

### Idle Timeout

The router checks every 60 seconds for idle models:

```
idleTime = now - lastActivityAt
if idleTime > idleTimeoutMinutes * 60000:
  shutdown model
```

Models with `idleTimeoutMinutes: -1` never auto-shutdown (useful for frequently-used models).

### Router Shutdown

On Ctrl+C or SIGTERM:

1. Stop accepting new requests
2. Cancel all running SLURM jobs
3. Release all ports back to pool
4. Exit cleanly

**No persistence** — router state is in-memory only.

## Port Management

The router manages a port pool for vLLM backends:

- **Default range:** 11435–11534 (100 ports)
- **Assignment:** Dynamic, first-available
- **Release:** On model shutdown

Ports are tracked in-memory and released when models stop.

## CLI Options

```bash
ivllm router [options]

Options:
  --port <port>           HTTP port for router (default: 11434)
  --host <host>           Bind address (default: 127.0.0.1)
  --login-host <host>     SSH login node (or set ISAMBARDC_LOGIN_HOST)
  --help, -h              Show help
```

## Examples

### Example 1: Quick Test

```bash
# Start router
ivllm router

# In another terminal, add a model
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-model",
    "configPath": "/path/to/config.yaml",
    "autoStart": false
  }'

# Start the model
curl -X POST http://localhost:11434/admin/models/test-model/start

# Check status
curl http://localhost:11434/admin/models

# Stop the model
curl -X POST http://localhost:11434/admin/models/test-model/stop
```

### Example 2: Opencode Integration

```bash
# Get provider config
PROVIDER=$(curl http://localhost:11434/admin/provider)

# Set environment variable
export OPENAI_BASE_URL=$(echo $PROVIDER | jq -r '.env.OPENAI_BASE_URL')

# Run opencode
opencode
```

### Example 3: Always-Warm Model

```bash
# Add a small model that never shuts down
curl -X POST http://localhost:11434/admin/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen3.5-0.5b",
    "configPath": "/path/to/qwen3.5-0.5b.yaml",
    "idleTimeoutMinutes": -1,
    "autoStart": true
  }'

# Start it once
curl -X POST http://localhost:11434/admin/models/qwen3.5-0.5b/start

# Model stays running for quick requests
```

## Troubleshooting

### Router won't start

**Error:** `--login-host is required`

**Solution:** Provide `--login-host` flag or set `ISAMBARDC_LOGIN_HOST` env var.

### Model fails to start

**Check logs:**
```bash
curl http://localhost:11434/admin/models/<model-name>/logs
```

**Common issues:**
- Config file path incorrect
- vLLM not installed on HPC
- SLURM job failed (check Isambard queue)

### 503 Startup in Progress

**Normal behavior** — model is starting up. Retry after 30s.

## Agent Integration Examples

### Opencode

**Setup:**
```bash
# Start the router
ivllm router

# Get provider configuration
export OPENAI_BASE_URL=$(curl -s http://localhost:11434/admin/provider | jq -r '.env.OPENAI_BASE_URL')

# Verify models are available
curl -s http://localhost:11434/admin/models | jq
```

**Usage:**
```bash
# Run opencode with router as backend
opencode

# Or set permanently in ~/.bashrc
echo 'export OPENAI_BASE_URL=http://127.0.0.1:11434' >> ~/.bashrc
```

**Lazy startup:** When you send a request to a stopped model with `autoStart: true`, opencode will receive a 503 response. Configure opencode to retry after 30 seconds.

### GitHub Copilot (Custom Backend)

**Configuration:**

If using Copilot with a custom OpenAI-compatible backend, set the following in your IDE or environment:

```bash
# Environment variables
export OPENAI_BASE_URL=http://127.0.0.1:11434
export OPENAI_API_KEY=not-needed  # Router doesn't require auth by default
```

**VS Code Settings:**
```json
{
  "github.copilot.advanced": {
    "customModels": [
      {
        "name": "qwen3.5-397b",
        "baseUrl": "http://127.0.0.1:11434",
        "apiVersion": "v1"
      }
    ]
  }
}
```

### Claude Code

**Setup:**
```bash
# Start router and ensure model is running
ivllm router &
sleep 5
curl -X POST http://localhost:11434/admin/models/qwen3.5-397b/start

# Configure Claude Code to use router
export OPENAI_BASE_URL=http://127.0.0.1:11434

# Run claude
claude
```

**Note:** Claude Code expects OpenAI-compatible APIs. The router provides `/v1/chat/completions` which is compatible.

### Custom Agent Script

Example Python script for programmatic access:

```python
import requests
import time

ROUTER_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5-397b"

def chat_with_model(message: str, max_retries: int = 10):
    """Send a message to the model with lazy startup support."""
    
    url = f"{ROUTER_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            
            elif response.status_code == 503:
                # Model is starting up, wait and retry
                print(f"Model starting up... (attempt {attempt + 1}/{max_retries})")
                time.sleep(30)
                continue
            
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
        
        except requests.exceptions.Timeout:
            print(f"Request timeout, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(5)
    
    raise Exception("Max retries exceeded")

# Usage
if __name__ == "__main__":
    response = chat_with_model("Explain the router architecture")
    print(response)
```

### Automated Model Management Script

Bash script for managing models:

```bash
#!/bin/bash
# manage-models.sh

ROUTER_URL="http://localhost:11434"

case "$1" in
  start)
    echo "Starting model: $2"
    curl -X POST "$ROUTER_URL/admin/models/$2/start"
    ;;
  
  stop)
    echo "Stopping model: $2"
    curl -X POST "$ROUTER_URL/admin/models/$2/stop"
    ;;
  
  status)
    echo "Model status:"
    curl -s "$ROUTER_URL/admin/models" | jq
    ;;
  
  health)
    echo "Health check for $2:"
    curl -s "$ROUTER_URL/admin/models/$2/health" | jq
    ;;
  
  logs)
    echo "Logs for $2:"
    curl -s "$ROUTER_URL/admin/models/$2/logs" | jq
    ;;
  
  list)
    echo "All models:"
    curl -s "$ROUTER_URL/admin/models" | jq '.models[] | {name, status, port}'
    ;;
  
  *)
    echo "Usage: $0 {start|stop|status|health|logs|list} [model-name]"
    exit 1
    ;;
esac
```

**Usage:**
```bash
./manage-models.sh status
./manage-models.sh start qwen3.5-397b
./manage-models.sh health qwen3.5-397b
./manage-models.sh list
```

### CI/CD Integration

Example GitHub Actions workflow for testing with router:

```yaml
name: Test with Isambard Router

on: [push]

jobs:
  test:
    runs-on: self-hosted
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Start router
        run: |
          ivllm router &
          sleep 5
      
      - name: Start test model
        run: |
          curl -X POST http://localhost:11434/admin/models \
            -H "Content-Type: application/json" \
            -d '{
              "name": "test-model",
              "configPath": "/path/to/test-config.yaml",
              "autoStart": true
            }'
          sleep 60  # Wait for model to start
      
      - name: Run tests
        env:
          OPENAI_BASE_URL: http://localhost:11434
        run: |
          npm test
```

## Best Practices

### Model Selection Strategy

**For frequently-used models:**
- Set `idleTimeoutMinutes: -1` (never shutdown)
- Use `autoStart: true` for instant availability
- Example: Small models for quick tasks (Qwen3.5-0.5B)

**For large, expensive models:**
- Set `idleTimeoutMinutes: 30-60` (aggressive shutdown)
- Use `autoStart: false` (require explicit start)
- Example: Large models for complex tasks (Qwen3.5-397B)

### Resource Optimization

```json
{
  "models": {
    "always-warm": {
      "configPath": "/path/to/small-model.yaml",
      "idleTimeoutMinutes": -1,
      "autoStart": true
    },
    "on-demand": {
      "configPath": "/path/to/large-model.yaml",
      "idleTimeoutMinutes": 30,
      "autoStart": true
    },
    "explicit-only": {
      "configPath": "/path/to/expensive-model.yaml",
      "idleTimeoutMinutes": 15,
      "autoStart": false
    }
  }
}
```

### Monitoring and Alerting

**Health check script:**
```bash
#!/bin/bash
# health-monitor.sh

ROUTER_URL="http://localhost:11434"
MODELS=("qwen3.5-397b" "qwen3.5-0.5b")

for model in "${MODELS[@]}"; do
  health=$(curl -s "$ROUTER_URL/admin/models/$model/health")
  status=$(echo "$health" | jq -r '.status')
  
  if [ "$status" != "healthy" ]; then
    echo "ALERT: $model is unhealthy"
    # Send notification (email, Slack, etc.)
  fi
done
```

**Cron job for periodic checks:**
```bash
# Add to crontab (every 5 minutes)
*/5 * * * * /path/to/health-monitor.sh
```

## Performance Considerations

### Network Latency

**Laptop deployment (SSH tunnel):**
- Requests go: Laptop → SSH → Login node → COMPUTE node
- Added latency: ~5-20ms per request
- Suitable for interactive use

**Login node deployment (future):**
- Requests go: Login node → COMPUTE node
- Lower latency: ~1-5ms per request
- Better for high-throughput scenarios

### Concurrent Models

The router supports multiple concurrent models:
- Each model runs on a separate port
- Models can share COMPUTE nodes if resources allow
- Port pool: 100 ports (11435-11534)

**Example scenario:**
```
Model A: qwen3.5-0.5b  → Port 11435 → Node 1
Model B: qwen3.5-32b   → Port 11436 → Node 2
Model C: qwen3.5-397b  → Port 11437 → Nodes 3-4
```

### Idle Timeout Tuning

**Short timeout (5-15 minutes):**
- Pros: Frees up HPC resources quickly
- Cons: Users wait for model startup
- Use case: Expensive models, low usage

**Long timeout (60+ minutes):**
- Pros: Instant response for users
- Cons: Holds HPC resources
- Use case: Frequently-used models

**No timeout (-1):**
- Pros: Always available
- Cons: Permanent resource usage
- Use case: Small models, critical workflows

```bash
# Poll until running
watch -n 5 'curl http://localhost:11434/v1/models'
```

### Port Conflicts

**Error:** `No available ports in range`

**Solution:** Stop unused models or increase port range in code.

## Known Limitations (MVP)

- **12-hour SSH timeout** — Router on laptop requires SSH reconnection after 12h
- **No persistence** — Router restart loses model state
- **Single-user** — No authentication, assumes trusted local network
- **No metrics** — No Prometheus/statsd integration
- **Proxy not implemented** — `/v1/chat/completions` returns 501 placeholder

## Future Enhancements

- [ ] Login node mode (run router on Isambard LOGIN)
- [ ] Job manager service (decouple SSH operations)
- [ ] Authentication (API keys)
- [ ] Metrics dashboard
- [ ] Scheduled startup
- [ ] Proxy implementation for `/v1/*` endpoints
- [ ] Agent launcher integration (separate project)

## Testing

```bash
# Run all router tests
bun test tests/router/

# Run specific test file
bun test tests/router/model-registry.test.ts
```

**Test coverage:**
- Model registry CRUD operations
- Port pool management
- HTTP server endpoints
- Configuration validation

## Architecture Notes

### RemoteExecutor Interface

The router uses an abstraction for SSH vs local execution:

```typescript
interface RemoteExecutor {
  runCommand(command: string): Promise<string>;
  copyFile(localPath: string, remotePath: string): Promise<void>;
  readFile(remotePath: string): Promise<string>;
  fileExists(remotePath: string): Promise<boolean>;
}
```

**Implementations:**
- `SSHExecutor` — For laptop deployment (default)
- `LocalExecutor` — For login node deployment (future)

### ModelRegistry

Manages both persistent config (`~/.config/ivllm/models.json`) and runtime state (in-memory):

- Config survives router restarts
- Runtime state (ports, job IDs, timestamps) is ephemeral
- Port pool dynamically assigns from 11435–11534 range

### RouterService

Main service class that:
- Creates Fastify HTTP server
- Manages model lifecycle (start/stop)
- Handles lazy startup logic
- Enforces idle timeouts
- Provides OpenAI API + admin endpoints
