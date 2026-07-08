# Scaling vLLM with Nginx Load Balancing

This guide provides instructions for scaling `vllm-gr` throughput using Data Parallelism (DP) by running multiple independent `vllm-gr serve` instances behind an Nginx load balancer.

## Overview

In standard `vllm-gr serve` configurations, a single Python process manages the API server, tokenization, and request scheduling. For high-throughput workloads (like **Beam Search**), the CPU-bound ApiServer front-end vllm process often becomes a bottleneck, preventing linear scaling across multiple GPUs.

By deploying $N$ independent vLLM processes, we distribute the CPU load across multiple Python interpreters (bypassing the GIL).

---

## 1. Launch Independent vLLM Instances (example for DP=2)

Assign each vLLM instance to a specific GPU using `CUDA_VISIBLE_DEVICES`. Ensure each instance listens on a unique port.

### Instance 1 (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm-gr serve \
    --port 9001 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --attention-backend CUSTOM \
    OpenOneRec/OneRec-1.7B
```

### Instance 2 (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 \
vllm-gr serve \
    --port 9002 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max_num_seqs 1024 \
    --max_num_batched_tokens 16384 \
    --scheduling-policy priority \
    --attention-backend CUSTOM \
    OpenOneRec/OneRec-1.7B
```

---

## 2. Nginx

### 2.1 Nginx installation

```bash
sudo apt install nginx
```

### 2.2 Nginx Configuration

Create a configuration file at `nginx_vllm.conf`. The `least_conn` algorithm is recommended as it accounts for the varying execution times of LLM generation requests.

```nginx
events {
    worker_connections 1024;
}

http {
    upstream vllm_cluster {
        # Directs traffic to the instance with the fewest active connections
        least_conn;

        server 127.0.0.1:9001;
        server 127.0.0.1:9002;
    }

    server {
        listen 8000; # Entry point for your application

        location / {
            proxy_pass http://vllm_cluster;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            # Optimized for Server-Sent Events (Streaming)
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 600s;
        }
    }
}
```

---

## 3. Running the Proxy

Start the Nginx service pointing to your configuration file. Use the absolute path to ensure the config is located correctly.

```bash
sudo nginx -c /path/to/file/nginx_vllm.conf -g "daemon off;"
```

### Process Management

| Task | Command |
| :--- | :--- |
| **Start Proxy** | `sudo nginx -c /path/to/file/nginx_vllm.conf -g "daemon off;"` |
| **Graceful Shutdown** | `sudo nginx -s quit` |
| **Reload Config** | `sudo nginx -s reload` |
| **Verify Port** | `netstat -tulpn \| grep 8000` |

---

## 4. Client Integration

Update your client or benchmark to point to the load balancer port (`8000`) instead of the 'vllm-gr serve' individual engine ports. For example:

```bash
python -m benchmarks.open_one_rec.one_rec_main bench serve --endpoint /v1/chat/completions --backend openai-chat --model OpenOneRec/OneRec-1.7B --dataset-name onerec --dataset-path ./data --task-type video --num-prompts 200 --use-beam-search --profile --n 128 --port 8000
```

## 5. Bash script to run it all

The script does the following:

1. Gets an optional argument (default: 2) of number of GPUs to use (N)
2. Launches N `vllm-gr serve` processes (in parallel) and wait for all to be ready.
    Each vllm process uses one GPU.
    Each vllm process's output is directed to log file under temp directory nginx_dp_run
3. Launches a single Nginx load balnacer on port 8000.

Example:
a. Run server:
    ./server_dp_nginx.sh 2
        To quit use Ctrl+C
b. Run HTTP client to port 8000

```bash
#!/bin/bash

# --- Configuration & Defaults ---
DEFAULT_N=2
TIMEOUT_LIMIT=120
MODEL="OpenOneRec/OneRec-1.7B"
NGINX_DIR="$(pwd)/nginx_dp_run"
NGINX_PORT=8000

# 1. Argument Parsing & Validation
N=${1:-$DEFAULT_N}
MAX_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ] || [ "$N" -gt "$MAX_GPUS" ]; then
    echo "Error: Invalid GPU request ($N). Server Max: $MAX_GPUS"
    exit 1
fi

# 2. Setup Working Directory
# Clean and recreate the nginx_dp_run directory for a fresh start
rm -rf "$NGINX_DIR"
mkdir -p "$NGINX_DIR/tmp"/{client_body,proxy,fastcgi,uwsgi,scgi}

NGINX_CONF="${NGINX_DIR}/nginx.conf"
VLLM_PIDS=()
LOG_FILES=()

# 3. Cleanup Function
cleanup() {
    echo -e "\n[Terminating] Stopping Nginx and vLLM instances..."

    if [ -f "$NGINX_CONF" ]; then
        # Use -g to ensure we can close the error log without sudo
        nginx -g "error_log ${NGINX_DIR}/error.log;" -c "$NGINX_CONF" -s quit 2>/dev/null
    fi

    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done

    echo "[Done] All processes stopped."
    exit
}

trap cleanup SIGINT

# 4. Prepare Nginx Configuration
cat <<EOF > "$NGINX_CONF"
daemon off;
pid ${NGINX_DIR}/nginx.pid;
error_log ${NGINX_DIR}/error.log;

events { worker_connections 1024; }

http {
    access_log            ${NGINX_DIR}/access.log;
    client_body_temp_path ${NGINX_DIR}/tmp/client_body;
    proxy_temp_path       ${NGINX_DIR}/tmp/proxy;
    fastcgi_temp_path     ${NGINX_DIR}/tmp/fastcgi;
    uwsgi_temp_path       ${NGINX_DIR}/tmp/uwsgi;
    scgi_temp_path        ${NGINX_DIR}/tmp/scgi;

    upstream vllm_cluster {
        least_conn;
EOF

# 5. Launch vLLM Instances
echo "Launching $N vLLM instances on GPUs 0 to $((N-1))..."
for ((id=0; id<N; id++)); do
    current_gpu=$id
    run_port=$((9000 + id))
    log_file="${NGINX_DIR}/vllm_gpu_${id}.log"
    LOG_FILES+=("$log_file")

    # Add instance to Nginx upstream
    echo "        server 127.0.0.1:$run_port;" >> "$NGINX_CONF"

    # Launch using standard environment
    CUDA_VISIBLE_DEVICES=$current_gpu \
    vllm-gr serve \
        --port $run_port \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --max-logprobs 1024 \
        --max_num_seqs 1024 \
        --max_num_batched_tokens 16384 \
        --scheduling-policy priority \
        --attention-backend CUSTOM \
        $MODEL > "$log_file" 2>&1 &

    VLLM_PIDS+=($!)
done

# Finalize Nginx Config
cat <<EOF >> "$NGINX_CONF"
    }
    server {
        listen $NGINX_PORT;
        location / {
            proxy_pass http://vllm_cluster;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_read_timeout 600s;
        }
    }
}
EOF

# 6. Wait for Startup
echo -n "Waiting for $N instances to initialize..."
start_time=$(date +%s)
while true; do
    ready_count=0
    for log in "${LOG_FILES[@]}"; do
        if grep -q "Application startup complete" "$log"; then
            ((ready_count++))
        fi
    done

    if [ "$ready_count" -eq "$N" ]; then
        echo -e "\n[Success] All instances ready."
        break
    fi

    if [ $(( $(date +%s) - start_time )) -ge "$TIMEOUT_LIMIT" ]; then
        echo -e "\n[Error] Timeout reached. Checking logs..."
        cleanup
    fi
    echo -n "."
    sleep 2
done

# 7. Start Nginx
echo "Starting Nginx on port $NGINX_PORT..."
nginx -g "error_log ${NGINX_DIR}/error.log;" -c "$NGINX_CONF" &
NGINX_PROC_PID=$!

echo "Cluster online at http://localhost:$NGINX_PORT"
wait $NGINX_PROC_PID
```
