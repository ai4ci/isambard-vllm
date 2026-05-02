---
name: isambard-compute-tunnel
description: 'Use when designing or building a system that runs a service on an Isambard AI or Isambard 3 compute node and needs to connect to it from a local machine via SSH tunnel. Triggers on phrases like "run a service on Isambard and connect locally", "SSH tunnel to compute node", "expose port from SLURM job", "vLLM on HPC with local access", "connect to job running on compute node", "port forward from Isambard". Do NOT use for generic Slurm job submission, monitoring, or HPC tasks that do not require a local network connection to the compute node.'
license: MIT
compatibility: 'Isambard AI (GH200, aarch64), Isambard 3 (Grace CPU, aarch64). Requires SSH access via Clifton. Language-agnostic: examples use bash; patterns apply to TypeScript, Python, Java, or any language.'
metadata:
  author: Rob Challen
  version: 0.1
---

# Isambard Compute Tunnel Pattern

A design and implementation guide for running a long-lived service inside a Slurm job on Isambard AI/3 and connecting to it from your local machine via an SSH port-forward tunnel through the login node.

The pattern has two components: a **SLURM job script** that runs the service and writes its hostname to a rendezvous file, and a **local daemon** that polls for that file, discovers the compute hostname, establishes the SSH tunnel, and manages the session lifecycle.

## When to Use This Skill

- You are building a system that submits a SLURM job on Isambard that runs a network service (HTTP API, JMX, Jupyter, database proxy, etc.) and needs to make it accessible locally.
- The compute node is not directly reachable from the user's machine — all access goes through the login node.
- You need the local session to manage the full lifecycle: submit → wait → tunnel → heartbeat → clean shutdown.
- You want the local client to cancel the job automatically when the user exits.

Do not use this skill for jobs that only produce files as output, or where the user can retrieve results by copying files after the job completes.

## The Core Problem

Isambard compute nodes are not directly reachable from outside the cluster. The login node is reachable via SSH, but compute node hostnames (e.g. `gpu-node-042`) are only resolvable within the cluster. Compute nodes are allocated dynamically at job start — you cannot know the hostname in advance.

The solution is a **rendezvous file**: the job writes its compute hostname (and any other connection details) to a file on shared storage (`$HOME` or `$PROJECTDIR`) once it starts. The local client polls for this file, reads it, then opens an SSH port-forward tunnel through the login node.

```
LOCAL MACHINE                ISAMBARD LOGIN              ISAMBARD COMPUTE
    |                              |                             |
    |---ssh -L port:node:port----->|---internal TCP forward----->| service
    |                              |                             |
    | poll rendezvous file ------->| (reads from $HOME NFS) ---->|
```

## Architecture

### 1. SLURM Job Script (runs on compute node)

The job script is responsible for:

1. **Writing the rendezvous file** immediately after the job starts, including:
   - `status`: lifecycle state (`initialising` → `running` → `failed`)
   - `compute_hostname`: result of `$(hostname)` — the private compute hostname
   - `slurm_job_id`: `$SLURM_JOB_ID`
   - Any service-specific connection details (port, model name, etc.)
2. **Starting the service** and updating the status to `running` once healthy
3. **Updating status to `failed`** if the service dies unexpectedly
4. **Cleaning up** on exit (persist diagnostics, update status)

```bash
JOB_DETAILS="$HOME/${JOB_NAME}/job_details.json"
COMPUTE_HOSTNAME=$(hostname)

# Write rendezvous file as soon as job starts
jq -n \
  --arg status "initialising" \
  --arg job_name "$JOB_NAME" \
  --arg slurm_job_id "$SLURM_JOB_ID" \
  --arg compute_hostname "$COMPUTE_HOSTNAME" \
  --argjson server_port "$SERVICE_PORT" \
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, server_port: $server_port}' \
  > "$JOB_DETAILS"

# ... start your service ...

# Once service is healthy, update status
jq '.status = "running"' "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
```

### 2. Local Client Daemon (runs on local machine)

The local daemon is responsible for:

1. **Pre-flight checks**: SSH connectivity, any required remote resources (venv, binaries, etc.)
2. **Creating a lockfile**: write a `pending` record to the rendezvous path before submitting (so the job script can overwrite it, and the client knows if the file pre-exists from a stale session)
3. **Submitting the job**: `sbatch <script>`
4. **Polling the rendezvous file**: `cat $HOME/job_name/job_details.json` via SSH every 5 seconds (this is a plain SSH+file-read, not a Slurm scheduler call — polling interval can be short)
5. **Slurm scheduler queries**: `squeue` and `sacct` calls must be throttled to **at most once per 60 seconds** (Isambard AUP policy — see [Isambard Policy Compliance](#isambard-policy-compliance))
6. **Establishing the tunnel** once status = `running`: `ssh -N -L localPort:computeHost:servicePort loginNode`
7. **Heartbeat monitoring**: poll the service endpoint locally at a reasonable interval (15–30s)
8. **Clean shutdown**: cancel the job (`scancel jobId`), kill the tunnel, remove the rendezvous file

## SLURM Script Patterns

### Rendezvous file lifecycle

The rendezvous file should go through these states:

| Status | Written by | Meaning |
|--------|-----------|---------|
| `pending` | LOCAL client (before sbatch) | Job submitted, not yet started |
| `initialising` | Job script (on start) | Job running, service starting up |
| `running` | Job script (after health check) | Service ready, tunnel can connect |
| `failed` | Job script (on error) | Service died, includes error message |
| `timeout` | Job script (on time limit) | Job hit wall time |

Writing `pending` before `sbatch` and using noclobber (`set -C; echo '...' > file`) to detect concurrent sessions is a useful pattern. If the write fails, a session is already running.

### Exit trap for diagnostics

Always capture diagnostics in an EXIT trap so they survive even if the job is cancelled:

```bash
on_exit() {
  EXIT_CODE=$?
  trap - EXIT
  # Update status if still showing initialising/running
  CURRENT_STATUS=$(jq -r '.status' "$JOB_DETAILS" 2>/dev/null || echo "unknown")
  if [ "$CURRENT_STATUS" = "initialising" ] || [ "$CURRENT_STATUS" = "running" ]; then
    jq '.status = "failed" | .error = "job exited unexpectedly"' \
      "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
  fi
  # Persist sacct diagnostics
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,ExitCode,MaxRSS,AllocTRES > \
    "$WORK_DIR/slurm-accounting.txt" 2>/dev/null || true
  exit $EXIT_CODE
}
trap on_exit EXIT
```

### Optional: graceful shutdown signal

If your service supports graceful shutdown (e.g. flushing writes, checkpointing), add:

```bash
#SBATCH --signal=B:TERM@60
```

This sends `SIGTERM` to the batch script 60 seconds before the time limit or `scancel`, giving the service time to clean up. Wire it with a signal handler:

```bash
handle_term() {
  echo "SIGTERM received — shutting down gracefully"
  kill -TERM $SERVICE_PID 2>/dev/null || true
}
trap handle_term TERM
```

### Service health check loop

Rather than sleeping a fixed time and hoping the service is ready, poll a health endpoint:

```bash
while true; do
  if curl -sf http://localhost:$SERVICE_PORT/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 $SERVICE_PID 2>/dev/null; then
    echo "Service process died during startup"
    jq '.status = "failed" | .error = "process died during startup"' \
      "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
    exit 1
  fi
  sleep 2
done
```

For services without an HTTP health endpoint, adapt to check the process is alive and the port is listening:

```bash
# TCP-level check without curl
while ! bash -c "echo > /dev/tcp/localhost/$SERVICE_PORT" 2>/dev/null; do
  sleep 2
done
```

## Login Node Address

The login node address (the SSH jump host for the tunnel) depends on which system you are targeting and can be derived from the project ID.

### From inside the SLURM job (recommended)

The most reliable approach is to have the compute node calculate its own login host and write it into the rendezvous file. This avoids the local client needing to know the system type in advance.

```bash
# Inside the SLURM script:
PROJECT_ID=$(basename "$PROJECTDIR")

# Isambard 3 (Grace CPU):
LOGIN_HOST="$PROJECT_ID.3.isambard"

# Isambard AI (GH200 GPU):
LOGIN_HOST="$PROJECT_ID.aip2.isambard"
```

The full login SSH target (user@host) is then:

```bash
LOGIN_TARGET="$SLURM_JOB_USER@$LOGIN_HOST"
# e.g. alice@myproject.aip2.isambard
```

Include `login_host` and `login_user` in the rendezvous file so the local daemon can use them directly without any configuration:

```bash
jq -n \
  --arg status "initialising" \
  --arg compute_hostname "$COMPUTE_HOSTNAME" \
  --arg login_host "$LOGIN_HOST" \
  --arg login_user "$SLURM_JOB_USER" \
  --argjson server_port "$SERVICE_PORT" \
  '{status: $status, compute_hostname: $compute_hostname,
    login_host: $login_host, login_user: $login_user, server_port: $server_port}' \
  > "$JOB_DETAILS"
```

The local daemon then reads `login_user@login_host` from the JSON rather than needing it hardcoded.

### Before the job is submitted (local client)

If the local client needs to SSH to the login node before the job starts (e.g. to copy files or run pre-flight checks), it must know the login address in advance because `$SLURM_JOB_USER` and `$PROJECTDIR` are not yet available.

The address format is:

| System | Login host | SSH user |
|--------|-----------|----------|
| Isambard 3 | `<project-id>.3.isambard` | `<brics-id>.<project-id>` |
| Isambard AI | `<project-id>.aip2.isambard` | `<brics-id>.<project-id>` |

Both systems use the same compound username format: `<brics-id>.<project-id>@<project-id>.<system>.isambard`.

These should be supplied by the user as configuration (e.g. `--user`, `--project`, `--system` CLI flags, or a config file) since they cannot be inferred programmatically from the local machine.

Example config (YAML):
```yaml
system: ai          # or: i3
brics_id: alice
project_id: myproject
# Derived:
# AI  login: alice@myproject.aip2.isambard
# I3  login: alice.myproject@myproject.3.isambard
```

## SSH Tunnel

The tunnel command to run locally:

```bash
ssh -N \
  -o BatchMode=yes \
  -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L ${LOCAL_PORT}:${COMPUTE_HOSTNAME}:${SERVICE_PORT} \
  ${USERNAME}@${LOGIN_HOST}
```

- `BatchMode=yes` — prevents interactive password prompts; fails fast if auth is broken
- `ServerAliveInterval=10` / `ServerAliveCountMax=3` — detects broken connections within 30s
- `ExitOnForwardFailure=yes` — exits immediately if the port-forward cannot be established (e.g. the compute node port isn't listening yet)
- Run as a background process; monitor its exit code and treat unexpected exit as a session failure

For multiple ports (e.g. JMX which uses two ports):

```bash
ssh -N ... \
  -L ${LOCAL_PORT_1}:${COMPUTE_HOSTNAME}:${REMOTE_PORT_1} \
  -L ${LOCAL_PORT_2}:${COMPUTE_HOSTNAME}:${REMOTE_PORT_2} \
  ${USERNAME}@${LOGIN_HOST}
```

## Local Client Design

A minimal local daemon in pseudocode:

```
1. Pre-flight: SSH echo ok  →  fail fast if unreachable
2. Lockfile: SSH noclobber write "pending" to rendezvous path
            →  fail if file already exists (stale session)
3. Copy SLURM script to login node
4. sbatch <script>  →  capture job ID
5. Poll loop (every 5s — plain SSH file read, not scheduler):
   a. cat rendezvous file
   b. If missing: check job state via sacct (≤once/60s); if FAILED → crash diagnostics → exit
   c. If status = "pending": check squeue state (≤once/60s) to report queue position
   d. If status = "initialising": stream SLURM log incrementally
   e. If status = "running": break → proceed to tunnel
   f. If status = "failed": print error, print SLURM log tail → exit
6. Spawn SSH tunnel subprocess
7. Heartbeat loop (every 15–30s): HTTP GET /health  →  if fails → shutdown
8. On SIGINT/SIGTERM/user exit:
   scancel $JOB_ID
   kill SSH tunnel
   rm rendezvous file (remote)
   exit
```

See `references/ivllm-implementation.md` for a detailed example from the `isambard-vllm` project.

## Isambard Policy Compliance

This pattern makes several types of remote calls; their polling constraints differ:

| Call type | Tool | Max frequency | Reason |
|-----------|------|--------------|--------|
| `cat job_details.json` | SSH file read | Every 5s | Just an NFS read, no Slurm scheduler involved |
| SLURM log streaming (`tail`) | SSH file read | Every 5s | Same — file read only |
| `squeue` | Slurm scheduler | ≤ once/60s | AUP: excessive polling disrupts all users |
| `sacct` | Slurm scheduler | ≤ once/60s | Same |
| Heartbeat (`curl /health`) | Local TCP | Every 15–30s | No Slurm involvement; reasonable for liveness |

Violating the 60s rule for `squeue`/`sacct` is a breach of the Isambard Acceptable Use Policy and can result in account suspension.

**Also comply with:**
- Never use `/tmp` for large or shared files in job scripts — use `$LOCALDIR` for ephemeral in-job scratch (wiped at job end), `$SCRATCHDIR` for inter-job data, `$PROJECTDIR` for persistent shared data.
- `$HOME` is NFS-mounted and slow for large I/O — use `$PROJECTDIR` (Lustre) for anything big.
- Do not run long computations on the login node. All job submission and file copying is fine; do not run the service itself on the login node.

See also: [adopt-isambard-policies](../adopt-isambard-policies/SKILL.md) skill for the full policy checklist.

## Storage Layout Convention

For a job named `my-job`, use a consistent layout:

```
$HOME/my-job/
├── job_details.json       # rendezvous file (small, NFS is fine)
├── my-job.slurm.sh        # SLURM script (copied by local client)
├── my-job.slurm.log       # job stdout/stderr (redirect via exec in script)
└── slurm-accounting.txt   # sacct output written on exit
```

Keep large data (model weights, inputs, outputs) in `$PROJECTDIR` not `$HOME`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Rendezvous file never appears | Job stuck in queue, or crashed before writing | Check `squeue --me`; check SLURM log |
| Status stays `initialising` forever | Service health check failing; service crashing silently | Tail the SLURM log; check service error output |
| Tunnel exits immediately | Compute node port not yet listening; `ExitOnForwardFailure=yes` | Add a brief wait after status → `running` before spawning tunnel; or retry tunnel |
| Tunnel connects but requests hang | Wrong hostname resolution inside tunnel | Verify `compute_hostname` in rendezvous file matches what `hostname` returns in the job |
| `sacct` shows OOM | Service requested more memory than allocated | Increase `--mem` or `--gpus` (each GH200 GPU brings 115 GiB RAM); check `gpu-memory-utilization` |
| Job cancelled but tunnel persists | Tunnel process not in shutdown handler | Ensure tunnel subprocess PID is captured and killed in shutdown |
| `squeue` returns nothing | Job already completed/failed | Switch to `sacct` for post-completion state |

## References

- [Isambard AI user documentation](https://docs.isambard.ac.uk/)
- [Isambard Slurm job management](https://docs.isambard.ac.uk/user-documentation/guides/slurm/)
- [Isambard storage spaces](https://docs.isambard.ac.uk/user-documentation/information/system-storage/)
- [adopt-isambard-policies skill](../adopt-isambard-policies/SKILL.md)
- [ivllm implementation reference](references/ivllm-implementation.md)
