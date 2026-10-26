# AB_Tests — A/B performance testing for vllm-gr PRs

Runs an **A-B-B-A** offline beam-search benchmark (baseline → head → head → baseline) against a vllm-gr PR and produces a base-vs-head latency comparison report. **Mode B: each developer runs their own benchmark server** — the scripts are shared, the config is private.

## Directory structure

| File | Where it runs | Purpose |
| --- | --- | --- |
| `ab_benchmark.sh` | **Local (developer machine)** | Entry point: resolves the PR's base/head SHAs → SSHes to the server to launch A-B-B-A → polls for the report → fetches and prints it |
| `ab_runner.sh` | Server | Runs the A-B-B-A four-pass offline benchmark (checkout → docker `run_offline_benchmark.py` → `generate_summary.py`) |
| `compare_ab.py` | Server | Compares the two sides' summaries and emits a markdown + json report |
| `benchmark.env.example` | Server | Config template (container name / GPU / paths); copy to `benchmark.env` (git-ignored) |
| `run_offline_benchmark.py` | Server (in container) | Offline beam-search benchmark engine, invoked directly by `ab_runner.sh` |
| `generate_summary.py` | Server (in container) | Aggregates `raw-result.json` → `summary.json` |
| `download_onerec_dataset.py` | Server (in container) | Verifies / downloads the OneRec dataset |
| `instrumentation/` | Server (in container) | Lightweight CPU timing probes (`sitecustomize.py` + `vllm_gr_lightweight_timing.py`), injected via `PYTHONPATH` |
| `constraints/` | Server (in container) | Constraint catalog triples doc (`video_constraint_triples.json`) for constrained beam decoding, committed with the tooling |

**This directory is self-contained**: the benchmark engine dependencies above are committed with this directory; nothing needs to be copied from `tools/daily_benchmark/`.

## Prerequisites

**Local**: `bash`, `ssh`; [`gh`](https://cli.github.com/) (with `gh auth login`) only if you resolve a PR number.

**Server** (all four required):

1. **SSH access** — an alias in `~/.ssh/config` using your own key (never share private keys).
2. **A benchmark container with the vllm-gr image** — inside the container `/opt/vllm-gr` mounts the vllm-gr repo, and `nvidia-smi` / `torch.cuda` work.
3. **The OneRec dataset** — at `/opt/vllm-gr/data` in the container (`download_onerec_dataset.py --tasks video` downloads/verifies it).
4. **HF model access** — the container can pull `OpenOneRec/OneRec-1.7B` (`HF_ENDPOINT` can point at a mirror).

## One-time setup

### 1. Server side: `benchmark.env`

```bash
# in the vllm-gr worktree on the server
cd tools/AB_Tests
cp benchmark.env.example benchmark.env
vi benchmark.env    # set PROJECT_DIR / CONTAINER_NAME / GPU_INDEX / CONTAINER_DATA_DIR / ...
```

### 2. Local: SSH alias + env vars

```bash
# ~/.ssh/config
Host my-gpu
    HostName <your server IP>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

```bash
# optional: set defaults in your local shell so you don't pass --server every time
export AB_SERVER=my-gpu
export AB_REMOTE_DIR=/path/on/server/to/vllm-gr
```

## Usage

```bash
# A/B PR #358 with the default scenario (beam 128, input 1024)
./ab_benchmark.sh 358

# pick server / scenario
./ab_benchmark.sh 358 --server my-gpu --scenario 256:1024

# compare any two commits (no gh needed)
./ab_benchmark.sh --base <base_sha> --head <head_sha>

# also download the report locally
./ab_benchmark.sh 358 --save-dir ./reports
```

The **scenario** defaults to `128:1024` (`beam_width:input_tokens`; beam width is restricted to 64/128/256). Tuning such as the number of prompts and warmup requests lives in the server's `benchmark.env`.

A PR's four A-B-B-A passes (each reloads the model + 100 requests) take roughly **10–15 minutes**; the script polls until the report is generated or it times out (default 1 hour).

## Reading the results

The report is a `base→head (delta%)` table; **positive delta% = slower (regression), negative = faster (improvement)**. Noise thresholds: mean/p50 +/-2%, p90 +/-3%, p99 +/-5%; values beyond threshold are flagged ⚠️.

Pitfalls to know:

- **The baseline is the merge base (fork point) of the PR head and `origin/decode_graph`, not the current `origin/decode_graph` tip** — the tip may already include other merged PRs and would mix in unrelated changes. `ab_runner.sh` computes `git merge-base origin/decode_graph <head>` server-side; `ab_benchmark.sh` passes only the head, so a stale "Base: xxx" from the PR body is never used.
- **A p50 ⚠️ is often just the hit/miss bimodal boundary**: prefill / total_beam's p50 lands in the fast (hit ~13ms / ~53ms) sample band, while mean/p90/p99 land in the slow (miss ~43ms / ~82ms) band. Judge a real regression by whether **mean/p90/p99 move in the same direction**; a p50-only move is usually not real.
- The primary signal is `e2el` / `e2el_hit` / `decode` / `decode_hit` — those are the end-to-end latencies.

## Known limitations

- `run_offline_benchmark.py` measures `GRLLM.beam_search_v1()` through public text materialization. It reports cold/hit E2E; legacy frontend stage diagnostics are retired. Keep model, catalog, width, input length, warmup, and output consumption identical on both sides.
- The report is only saved/printed; it is **not** auto-commented to the PR — whether to post it is up to you.
- Server-side `ab_runner.sh` requires the worktree to be on a clean `decode_graph` branch, and it runs `git fetch origin decode_graph` and `git fetch origin pull/<N>/head` (the server must be able to reach GitHub).

## How it survives `git checkout`

`ab_runner.sh` switches the worktree between base and head with `git checkout --detach <sha>`. Because the benchmark engine now lives in the *tracked* `tools/AB_Tests/` directory, checking out a base/head commit that predates this directory would delete those files from the working tree mid-run. To avoid that, `ab_runner.sh` first copies the engine (`run_offline_benchmark.py`, `generate_summary.py`, `download_onerec_dataset.py`, `compare_ab.py`, `instrumentation/`, and `constraints/`) into the git-ignored `results/ab/.tooling/` directory and runs everything from there. That staging dir sits under `results/`, which is git-ignored and holds only untracked files, so `git checkout` never touches it — the A/B can run against any base/head commit regardless of when this tooling was added.
