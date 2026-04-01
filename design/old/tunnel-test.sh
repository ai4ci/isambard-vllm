#!/bin/bash
# DEPRECATED: This script tests a reverse SSH tunnel (COMPUTE -> LOGIN) which does not work
# on Isambard AI — COMPUTE nodes cannot initiate outbound SSH connections.
# The mock HTTP server idea here is still useful as a testing pattern, but the tunnel
# must be established from LOCAL using a forward tunnel:
#   ssh -L <local_port>:<compute_node>:<server_port> <user>@<login_node>
# See design/requirements.md for the current design.
#SBATCH --job-name=tunnel-test
#SBATCH --nodes=1
#SBATCH --gpus=1              # Keep GPU allocation same as vLLM job
#SBATCH --time=00:30:00       # Short test
#SBATCH --output=out/tunnel-%j.out

# === CONFIG ===
SERVER_PORT=8000              # Same port vLLM will use
TUNNEL_PORT=11434             # Local port you'll connect to
LOGIN_NODE="${SLURM_SUBMIT_HOST}"

echo "=== Tunnel Test Starting ==="
echo "Server port: $SERVER_PORT"
echo "Tunnel port: $TUNNEL_PORT"
echo "Login node: $LOGIN_NODE"
echo "Compute node: $(hostname)"

# === START SIMPLE HTTP SERVER ===
# This simulates vLLM's /health and /v1 endpoints
python3 << EOF &
import http.server
import socketserver
import json
import time

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"object":"list","data":[{"id":"test-model"}]}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]}", flush=True)

with socketserver.TCPServer(("0.0.0.0", $SERVER_PORT), Handler) as httpd:
    print(f"✅ Test server running on port $SERVER_PORT", flush=True)
    httpd.serve_forever()
EOF

HTTP_PID=$!
echo "HTTP server PID: $HTTP_PID"

# === WAIT FOR SERVER TO BE READY ===
sleep 3
if curl -s http://localhost:$SERVER_PORT/health > /dev/null; then
    echo "✅ Local server is responding"
else
    echo "❌ Local server failed to start"
    kill $HTTP_PID 2>/dev/null
    exit 1
fi

# === ESTABLISH REVERSE SSH TUNNEL ===
echo "Establishing reverse tunnel..."
ssh -N -f -R $TUNNEL_PORT:localhost:$SERVER_PORT \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=30 \
    -v \
    $USER@$LOGIN_NODE

SSH_EXIT=$?
if [ $SSH_EXIT -eq 0 ]; then
    echo "✅ Tunnel established successfully"
    echo ""
    echo "=== TEST FROM YOUR LOCAL MACHINE ==="
    echo "Run these commands locally:"
    echo "  curl http://localhost:$TUNNEL_PORT/health"
    echo "  curl http://localhost:$TUNNEL_PORT/v1/models"
    echo ""
    echo "Tunnel will stay open for $SBATCH_TIME seconds"
else
    echo "❌ Tunnel failed with exit code $SSH_EXIT"
    echo "Check SSH keys and login node access"
    kill $HTTP_PID 2>/dev/null
    exit 1
fi

# === KEEP JOB ALIVE ===
echo "Job running... Ctrl+C in job output to cancel early"
wait $HTTP_PID

# === CLEANUP ===
pkill -f "ssh.*-R $TUNNEL_PORT:localhost:$SERVER_PORT" 2>/dev/null
echo "=== Tunnel Test Complete ==="
