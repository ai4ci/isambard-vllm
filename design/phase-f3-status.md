# Phase F3 Implementation Status

## Summary

**Status:** Core infrastructure complete, proxy implementation pending

**Tests:** 35 passing ✅

**Files Created:**
- `src/router/types.ts` - Type definitions
- `src/router/port-pool.ts` - Port pool manager (11435-11534)
- `src/router/registry.ts` - Model registry file I/O
- `src/router/model-registry.ts` - Model registry class with state tracking
- `src/router/executor.ts` - SSH/Local executor abstraction
- `src/router/job-manager.ts` - SLURM job lifecycle management
- `src/router/router-service.ts` - Main router HTTP service
- `src/commands/router.ts` - CLI command
- `src/types.ts` - RemoteExecutor interface
- `design/router.md` - Comprehensive usage documentation
- `tests/router/*.test.ts` - 35 unit tests

## Completed Components

### ✅ F3.1 — Project Scaffold
- Fastify HTTP server setup
- Basic routing structure
- Config loader for `~/.config/ivllm/models.json`
- Health check endpoint (`GET /health`)
- **Tests:** 4 passing

### ✅ F3.2 — Model Registry
- `ModelRegistry` class with full CRUD operations
- Port pool manager (11435-11534 range, dynamic assignment)
- ModelState tracking (in-memory)
- Validation for model configs
- **Tests:** 17 passing

### ✅ F3.3 — SLURM Integration
- `RemoteExecutor` interface (SSH vs local abstraction)
- `SSHExecutor` implementation for laptop deployment
- `LocalExecutor` implementation for future login-node mode
- `JobManager` class for job submission/polling/cleanup
- Reuses existing `renderInferenceScript` from `ivllm start`
- **Tests:** Integrated in model-registry tests

### ✅ F3.5 — Admin API
All admin endpoints implemented:
- `GET /admin/models` — List all configured models with status
- `POST /admin/models` — Add model configuration
- `DELETE /admin/models/:name` — Remove model
- `POST /admin/models/:name/start` — Explicit start
- `POST /admin/models/:name/stop` — Explicit stop
- `GET /admin/models/:name/logs` — Log retrieval
- `GET /admin/provider` — Opencode provider config
- **Tests:** 4 passing (server.test.ts)

### ✅ F3.7 — CLI Wrapper
- `ivllm router` command implemented
- Options: `--port`, `--host`, `--login-host`
- Help text with examples
- Graceful shutdown on SIGINT/SIGTERM
- **Tests:** Manual testing pending

### ✅ Documentation
- `design/router.md` — Complete usage guide including:
  - Quick start examples
  - API reference (all endpoints)
  - Model configuration guide
  - Lifecycle management explanation
  - Troubleshooting section
  - Architecture notes
  - Known limitations

## Remaining Work

### ⏳ F3.4 — OpenAI API Proxy

**Current state:** Placeholder returns 501 Not Implemented

**What's needed:**
```typescript
// In router-service.ts line 153
// Need to proxy POST /v1/chat/completions to vLLM backend

async function proxyToVllm(port: number, path: string, body: any) {
  // 1. Get model's assigned port from registry
  // 2. Forward HTTP request to http://<compute-node>:<port>/v1/chat/completions
  // 3. Stream response back to client
  // 4. Update lastActivityAt timestamp
}
```

**Implementation approach:**
- Use `http` or `undici` for HTTP proxying
- Need to handle SSH tunnel or direct network access to COMPUTE nodes
- Update `lastActivityAt` on each successful proxy request
- Handle connection errors gracefully

**Priority:** HIGH — This is the core functionality for agent orchestration

### ⏳ F3.6 — Lifecycle Management

**Partially implemented:**
- ✅ Idle timeout checker (runs every 60s)
- ✅ Model state tracking
- ❌ Lazy startup polling loop (needs completion)
- ❌ Health check loop (per-model)

**What's needed in `startModel()` method:**
```typescript
// Already has polling loop, but needs:
// 1. Better error handling
// 2. Progress logging
// 3. Update lastActivityAt on successful startup
```

**Priority:** HIGH — Required for autonomous operation

### ⏳ F3.8 — Agent Integration Examples

**What's needed:**
- Opencode integration example
- Copilot Coding Agent example
- Claude Code example
- Generic OpenAI client example

**Priority:** MEDIUM — Documentation/enabling feature

### ⏳ F3.9 — End-to-End Testing

**Test scenarios:**
1. Single model lazy startup
2. Multiple concurrent models
3. Idle timeout behavior
4. Router shutdown cleanup
5. Agent integration test

**Priority:** HIGH — Required for production readiness

## Known Issues

1. **Proxy not implemented** — `/v1/chat/completions` returns 501
2. **No SSH tunnel for proxy** — Job manager assumes direct network access to COMPUTE
3. **Error handling** — Needs improvement in job submission/polling
4. **Logging** — Minimal, needs structured logging for debugging

## Architecture Decisions

### 1. Config vs Runtime State Separation

**Decision:** Split persistence
- `~/.config/ivllm/models.json` — Persistent config (survives restarts)
- In-memory `Map<string, ModelState>` — Runtime state (ephemeral)

**Rationale:**
- Config changes infrequently, should persist
- Runtime state (ports, job IDs) is transient
- Simplifies recovery: re-read config, rebuild state from SLURM

### 2. Port Pool Management

**Decision:** Dynamic assignment from fixed range (11435-11534)

**Rationale:**
- Avoids port conflicts
- Allows 100 concurrent models max
- First-available strategy is simple and effective

### 3. RemoteExecutor Abstraction

**Decision:** Interface-based design with SSH/Local implementations

**Rationale:**
- Enables future login-node deployment without code changes
- Clean separation of concerns
- Easy to test with mocks

### 4. Hard Cleanup on Shutdown

**Decision:** Router shutdown cancels ALL managed SLURM jobs

**Rationale:**
- Router owns lifecycle
- No orphaned jobs on COMPUTE nodes
- Simpler than detach/reattach logic

## Next Steps (Priority Order)

1. **Implement proxy** (F3.4) — Core functionality
2. **Complete lifecycle management** (F3.6) — Lazy startup, health checks
3. **E2E testing** (F3.9) — Validate on Isambard AI
4. **Agent examples** (F3.8) — Documentation

## Estimated Effort

- **F3.4 (Proxy):** 4-6 hours
- **F3.6 (Lifecycle):** 2-3 hours
- **F3.9 (E2E Testing):** 4-6 hours (includes Isambard queue time)
- **F3.8 (Docs):** 1-2 hours

**Total remaining:** ~12-18 hours

## Success Criteria (Unmet)

- [ ] Agents can connect to `http://localhost:11434` and discover models
- [ ] Lazy startup works end-to-end (agent request → SLURM job → proxied response)
- [ ] Idle timeout shuts down unused models automatically
- [ ] Router shutdown cleans up all SLURM jobs
- [ ] Multiple concurrent models supported
- [ ] **50+ unit tests passing** (currently 35)
- [ ] **Proxy functional** (currently 501)

---

**Last updated:** 2026-06-05
**Branch:** `feature/phase-f3-router`
**Version:** v1.0.0 (in development)
