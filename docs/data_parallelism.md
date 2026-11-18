# Scaling Beam serving with Nginx

Nginx can route HTTP requests to independent Beam Search V1 replicas.
Each replica has its own EngineCore, GPU and KV cache.

## Start two replicas

Prepare the [canonical V1 configuration](../examples/online_serving/beam_search/README.md#prepare-the-canonical-configuration).
Use the same model, served name and config for both replicas. In two terminals
run the [V1 server command](../examples/online_serving/beam_search/README.md#start-the-server)
with these additions:

| Replica | Environment | Additional arguments |
| --- | --- | --- |
| First | `CUDA_VISIBLE_DEVICES=0` | `--host 127.0.0.1 --port 9001` |
| Second | `CUDA_VISIBLE_DEVICES=1` | `--host 127.0.0.1 --port 9002` |

## Configure Nginx

### Install Nginx

```bash
sudo apt install nginx
```

### Configure the upstream

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
        listen 127.0.0.1:8000; # Entry point for your application

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

## Test the proxy

Run Nginx in the foreground with your configuration. The packaged Nginx
installation may require elevated privileges for its default PID and log paths:

```bash
sudo nginx -c /absolute/path/to/nginx_vllm.conf -g 'daemon off;'
```

Use the [V1 client request](../examples/online_serving/beam_search/README.md#send-a-request)
against `http://127.0.0.1:8000/v1/completions`. For load testing, use the
[HTTP throughput benchmark](benchmarks/throughput.md) with the proxy's host
and port.
