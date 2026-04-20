# LMCache support for vllm-gr

## Overview

### LMCache overview

**LMCache** is an open-source high-performance Key-Value (KV) cache management system designed to sit alongside Large Language Model (LLM) engines like vLLM. Its primary goal is to eliminate redundant "prefill" computations by enabling KV cache sharing across different requests, users, and even separate GPU instances.

#### Core Philosophy: "Prefill Once, Reuse Everywhere"

In standard LLM serving, if two users send prompts with a shared 2,000-token document, the engine computes the KV cache for that document twice. LMCache breaks this "cache silo" by decoupling the KV cache from the local GPU memory and making it globally accessible.

#### Key Capabilities

* **Cross-Instance Sharing:** Allows multiple vLLM processes to share the same KV cache. If Instance A computes a prefix, Instance B can retrieve it from the shared pool instead of re-computing it.
* **Hierarchical Storage:** Automatically manages KV cache across a hierarchy of storage layers:
    * **Local CPU RAM:** For ultra-fast local access.
    * **Peer-to-Peer (P2P):** Instances can fetch cache blocks directly from each other’s CPU RAM.
    * **Remote Storage:** Persistent storage in systems like Redis, S3, or shared filesystems for "warm starts" even after a server reboot.
* **Semantic Chunking:** Unlike simple prefix caching that requires an exact match from the start of the string, LMCache can handle shared content appearing at different positions or within different contexts.
* **Extreme Performance:** It utilizes low-level optimizations like **NIXL** (NVIDIA Inference Xfer Library) and zero-copy memory transfers to ensure that fetching a cached block is significantly faster than recomputing it on a GPU.

#### Architecture Overview

LMCache typically operates through a **Controller-Worker** model. The workers (integrated into vLLM) handle the actual storage and retrieval of KV blocks, while a centralized (or distributed) controller tracks which blocks are stored where in the cluster.

#### Why It Matters for Production

1. **Reduces TTFT (Time to First Token):** By skipping the prefill phase for known text, response times for long-context queries (like RAG or multi-turn chat) drop from seconds to milliseconds.
2. **Saves GPU Cycles:** Frees up GPU compute power that would otherwise be wasted on redundant math, effectively increasing the total throughput of your cluster.
3. **Cost Efficiency:** By maximizing cache hits, you can serve more users with the same number of GPUs, often reducing total inference costs by 3–10x in context-heavy workloads.

### LMCache and vllm-gr overview

 vllm-gr supports LMCache in 2 modes of operation:

 1. Standalone: each vllm-gr process uses LMCache to store KVC in RAM. KV chunks are not shared with other vllm-gr processes.
 2. Controller+P2P: In this mode all vllm-gr processes are connected to a central controller via network (UDS on local server or TCP/IP between servers). Controller doesn't store KVC chunks, it only stores the knowledge of which vllm-gr process holds each KVC chunk.
 KVC chunk data itself is transmitted from vllm-X (RAM/HBM) to vllm-Y (RAM/HBM) via UCX (Unified Communication X) and NIXL (on Nvidia GPUs).

 Other modes are not supported since:
 a. LMCache MP (Multi-Process) mode: in this mode LMCache runs as a standalone server process that multiple vLLM instances connect to over ZMQ. This allows all vLLM processes on the same node to share a single L1 cache, maximizing KV cache reuse, providing process isolation, and enabling independent scaling of cache and inference resources (Multiprocess Mode docs, Architecture Guide).
 In this mode lmcache.server process can store KVC chunks in process's memory, in shared memory (/dev/shm/) or on RAMdisk (/mnt/lmcache/).
 Currently MP mode can't be used with vllm-gr since vllm version 0.14.0 rc1 and lmcache (all versions from v0.3.13 to v0.4.3) disagree on network protocol to send KVC data with.
 b. RAMdisk: In this mode there is no LMCache process involved (neither controller nor server).
 In this mode all vllm-gr processes are configured to store KVC chunks in a shared RAMdisk (/mnt/lmcache/).
 This mode works well but performance is very low (e.g.: TTFT is high).

## How to run vllm-gr with LMCache

### Install lmcache

 ```bash
pip install lmcache==0.4.1
 ```

### Run standalone vllm-gr with lmcache

1. Create lmcache config file lmcache_standalone.yaml:

```bash
echo "chunk_size: 256
backend: "local"
local_cpu: true
max_local_cpu_size: 120" > lmcache_standalone.yaml
```

(*) max_local_cpu_size is in GB. Set it as big as possible depending on available DRAM and the number of vLLM processes you launch.

2. Run vllm-gr pointing to lmcache_standalone.yaml

```bash
LMCACHE_CONFIG_FILE=lmcache_standalone.yaml \
vllm serve --gr \
    OpenOneRec/OneRec-1.7B \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 16384 \
    --scheduling-policy priority \
    --attention-backend CUSTOM \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### Run vllm-gr with lmcache controller in P2P mode

In this example we use 2 vllm-gr processes and a single LMCache controller

1. prepare lmcache config yaml file for each vllm

lmcache_controller_vllm_1.yaml:

```bash
chunk_size: 256
local_cpu: True
max_local_cpu_size: 100
enable_async_loading: True

# P2P configurations
enable_p2p: True
p2p_host: "localhost"
p2p_init_ports: 8200
p2p_lookup_ports: 8201
transfer_channel: "nixl"

# Controller configurations
enable_controller: True
lmcache_instance_id: "lmcache_instance_1"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: 8500

extra_config:
  lookup_backoff_time: 0.001
```

lmcache_controller_vllm_2.yaml:

```bash
chunk_size: 256
local_cpu: True
max_local_cpu_size: 100
enable_async_loading: True

# P2P configurations
enable_p2p: True
p2p_host: "localhost"
p2p_init_ports: 8202
p2p_lookup_ports: 8203
transfer_channel: "nixl"

# Controller configurations
enable_controller: True
lmcache_instance_id: "lmcache_instance_2"
controller_pull_url: "localhost:8300"
controller_reply_url: "localhost:8400"
lmcache_worker_ports: 8501

extra_config:
  lookup_backoff_time: 0.001
```

2. run LMCache controller

```bash
TMPDIR=/tmp \
PYTHONHASHSEED=0 \
lmcache_controller --host localhost --port 9000 \
    --monitor-ports '{"pull": 8300, "reply": 8400, "heartbeat": 8082}'
```

3. run the 2 vllm-gr processes

```bash
PYTHONHASHSEED=0 \
CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/tmp \
LMCACHE_LOG_LEVEL=WARNING \
LMCACHE_CONFIG_FILE=lmcache_controller_vllm_1.yaml \
vllm serve --gr OpenOneRec/OneRec-1.7B \
    --port 10501 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 16384 \
    --scheduling-policy priority \
    --attention-backend CUSTOM \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

```bash
PYTHONHASHSEED=0 \
CUDA_VISIBLE_DEVICES=1 \
TMPDIR=/tmp \
LMCACHE_LOG_LEVEL=WARNING \
LMCACHE_CONFIG_FILE=lmcache_controller_vllm_2.yaml \
vllm serve --gr OpenOneRec/OneRec-1.7B \
    --port 10502 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 16384 \
    --scheduling-policy priority \
    --attention-backend CUSTOM \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

4. Test cache sharing:

a. Send a request to vllm-gr process 1 (port 10501) to populate the cache.

b. Send the same request to vllm-gr process 2 (port 10502); it should retrieve the cache from engine 1.

## Using LMCache in offline vllm mode

To use LMCache in offline mode, modify kv_transfer_config arg when calling LLM(), for example:

    from vllm.config import KVTransferConfig

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model=model,
        max_logprobs=beam_width * 2,
        max_num_seqs=512 * 5,
        max_num_batched_tokens=4096 * 5,
        kv_transfer_config=ktc)
