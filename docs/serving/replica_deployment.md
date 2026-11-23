# Scaling Beam serving with Nginx

Use Nginx to distribute HTTP requests across two OneRec servers on separate
CUDA GPUs. Each replica owns its EngineCore and KV cache.

## Prerequisites

- Two CUDA GPUs on the same host.
- The [CUDA environment and a local OneRec snapshot](../../examples/online_serving/beam_search/README.md#prepare-the-environment-and-inputs).
- Follow [Prepare a persistent service](../../examples/online_serving/beam_search/README.md#prepare-a-persistent-service)
  with `ONEREC_BACKEND=cuda` to generate `$ONEREC_WORKDIR/vllm_gr_config.yaml`
  once for both replicas.

## Start two replicas

In each replica terminal, activate the CUDA environment and change to the
repository root. Export the same model path and work directory used during
preparation. The settings below use the default work directory. Replace the
model path and use your prepared work directory if it differs.

```bash
export ONEREC_MODEL=/absolute/path/to/OneRec-1.7B
export ONEREC_WORKDIR="$PWD/results/onerec-serving"
export ONEREC_BACKEND=cuda
```

Select the GPU for each terminal, then run the
[HTTP server command](../../examples/online_serving/beam_search/README.md#start-the-server).
Replace `--port 8000` with the replica's port. Keep `--host 127.0.0.1` and the
same served model name (`onerec` by default) for both replicas.

| Replica | GPU selection | Server port |
| --- | --- | --- |
| First | `export CUDA_VISIBLE_DEVICES=0` | `--port 9001` |
| Second | `export CUDA_VISIBLE_DEVICES=1` | `--port 9002` |

## Configure Nginx

### Install Nginx

```bash
sudo apt install nginx
```

### Configure the upstream

Save the following as `nginx_vllm.conf`. The `least_conn` policy sends each
request to the replica with fewer active connections.

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
        listen 127.0.0.1:8000; # Entry point for your application

        location / {
            proxy_pass http://vllm_cluster;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 600s;
        }
    }
}
```

## Test the proxy

Run Nginx in the foreground with your configuration. The packaged Nginx
installation may require elevated privileges for its default PID and log paths:

```bash
sudo nginx -c /absolute/path/to/nginx_vllm.conf -g 'daemon off;'
```

In a separate client terminal, set `ONEREC_WORKDIR` and `ONEREC_VERIFY_CATALOG`
to the values from preparation. Use the
[HTTP client request](../../examples/online_serving/beam_search/README.md#send-a-request-and-verify-the-result)
against `http://127.0.0.1:8000/v1/completions`. For load testing, use the
[HTTP throughput benchmark](../benchmarking/throughput.md) with the proxy's host
and port.
