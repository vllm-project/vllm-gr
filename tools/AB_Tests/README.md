# A/B benchmark utilities

Use the [`ab-offline` CLI](../../docs/cli/bench.md#offline-ab) with your local
CUDA runtime and model. Supply both `--data-dir` and `--catalog`.

The video benchmark uses the gated OpenOneRec dataset. Accept its terms, run
`hf auth login`, and download the pinned RecIF revision and its manifest.
Use a local model snapshot or pass a full Hub commit with `--model-revision`.

```bash
DATA_DIR=/path/to/onerec
python tools/AB_Tests/download_onerec_dataset.py --tasks video --data-dir "$DATA_DIR"
python tools/AB_Tests/download_onerec_dataset.py --tasks video --data-dir "$DATA_DIR" --verify-only
```

The manifest records the downloaded revision, file sizes, and SHA-256 hashes.
Verification checks the files without writing. To replace a manifest from an
older downloader, run the download command before verifying.

To rebuild the video triples, use `benchmark_data/video/video_test.parquet`
from `OpenOneRec/OpenOneRec-RecIF@8f7cf2ee0b949e955a87a708d02024687be232c8`.
The downloaded parquet has SHA-256
`66079915e3c3daab7df3487f9babb658df99ec1672c3ef42d7893d87cef2e6d8`.
From the source checkout, run:

```bash
python benchmarks/open_one_rec/tools/create_task_catalog.py \
  -t video -d "$DATA_DIR" -o "$DATA_DIR/video-sid-sequences.json"
PYTHONPATH=. python tools/constraints/build_constraint_triples.py \
  --catalog "$DATA_DIR/video-sid-sequences.json" \
  --begin '<|sid_begin|>' --end '<|sid_end|>' \
  --output "$DATA_DIR/video-constraint-triples.json"
```

The rebuilt catalog contains 234,749 video triples. Use it for both checkouts.

## Compare revisions

Use a separate virtual environment for each source checkout. The benchmark CLI
prefers the checkout from which `vllm_gr` is loaded, so changing only the
working directory can measure the same revision twice.

In Bash, set the two checkout paths and choose two fresh virtual environment
paths. Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and apply the [GR CLI patch](../../docs/cli/README.md#enable-gr-dispatch) in each
environment. The following setup uses the vLLM wheel installed by `pip`.

```bash
BASE_CHECKOUT=/path/to/baseline-checkout
HEAD_CHECKOUT=/path/to/head-checkout
BASE_VENV=/path/to/venvs/onerec-baseline
HEAD_VENV=/path/to/venvs/onerec-head
BASE_SHA=$(git -C "$BASE_CHECKOUT" rev-parse HEAD)
HEAD_SHA=$(git -C "$HEAD_CHECKOUT" rev-parse HEAD)

prepare_checkout() (
  set -eu
  cd -- "$1"
  if [[ -e "$2" || -L "$2" ]]; then
    printf 'Choose a fresh virtual environment path for %s\n' "$PWD" >&2
    exit 1
  fi
  unset PYTHONPATH PYTHONHOME
  uv venv --python 3.12 --seed "$2"
  source "$2/bin/activate"
  VLLM_GR_TARGET_DEVICE=cuda python -m pip install -e .
  site_packages=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  cli_patch=docs/cli/patches/vllm-0.30.0-gr-cli.patch
  patch --dry-run --forward -d "$site_packages" -p1 < "$cli_patch"
  patch --forward -d "$site_packages" -p1 < "$cli_patch"
)

(
  set -e
  prepare_checkout "$BASE_CHECKOUT" "$BASE_VENV"
  prepare_checkout "$HEAD_CHECKOUT" "$HEAD_VENV"
)
```

Confirm that `BASE_SHA` and `HEAD_SHA` are the intended commit IDs. Before
each measurement, verify the loaded package and benchmark checkout. Use the
same model, data, and catalog paths for both runs and save separate outputs.

```bash
MODEL=/path/to/OneRec
DATA_DIR=/path/to/onerec
CATALOG="$DATA_DIR/video-constraint-triples.json"

measure_checkout() (
  set -eu
  cd -- "$1"
  git diff --quiet HEAD --
  unset PYTHONPATH PYTHONHOME
  export PYTHONNOUSERSITE=1
  source "$2/bin/activate"
  "$VIRTUAL_ENV/bin/python" -I - "$3" <<'PY'
import subprocess
import sys
from pathlib import Path

import vllm_gr
from vllm_gr.benchmarks import scenarios

checkout = Path.cwd().resolve()
package = Path(vllm_gr.__file__).resolve()
root = scenarios._REPO_ROOT
if package != checkout / "vllm_gr" / "__init__.py" or root != checkout:
    raise SystemExit(f"Expected {checkout}, loaded package {package} and benchmark checkout {root}")
entry = scenarios.SCENARIOS["ab-offline"]
relative = Path(*entry.module.split(".")).with_suffix(".py") if entry.module else entry.path
if relative is None:
    raise SystemExit("No driver registered for ab-offline")
driver = (root / relative).resolve(strict=True)
if not driver.is_file() or not driver.is_relative_to(checkout):
    raise SystemExit(f"Invalid benchmark driver for checkout {checkout}: {driver}")
sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if sha != sys.argv[1]:
    raise SystemExit(f"Expected Git SHA {sys.argv[1]}, loaded {sha}")
print(f"GR package {package}\nBenchmark driver {driver}\nGit SHA {sha}")
PY
  "$VIRTUAL_ENV/bin/vllm" bench ab-offline --gr \
    --model "$MODEL" --data-dir "$DATA_DIR" --catalog "$CATALOG" \
    --beam-width 128 --input-length 1024 --output "$4"
)

(
  set -e
  measure_checkout "$BASE_CHECKOUT" "$BASE_VENV" "$BASE_SHA" /path/to/baseline.json
  measure_checkout "$HEAD_CHECKOUT" "$HEAD_VENV" "$HEAD_SHA" /path/to/head.json
)
```

Keep the dataset manifest and verified Git SHAs with the results. In each
checkout's activated environment, convert its raw result with the same
measurement options. For example:

```bash
python tools/AB_Tests/generate_summary.py \
  --execution-mode offline --raw-result /path/to/baseline.json \
  --dataset-manifest "$DATA_DIR/.onerec-manifest.json" \
  --git-sha "$(git rev-parse HEAD)" --git-subject "$(git log -1 --format=%s)" \
  --run-id baseline --warmup-requests 4 --beam-width 128 --input-length 1024 \
  --output /path/to/baseline-summary.json
```

Direct `generate_summary.py` invocations accept `--repo-dir` for compatibility.
Online summaries require `--catalog-path` matching the server configuration and
`--service-metrics`. To record GPU metadata, pass the full `--gpu-uuid` of the
measured device on a host that can query it. Without it the device is unknown.

Compare from the updated checkout:

```bash
python tools/AB_Tests/compare_ab.py \
  --base /path/to/baseline-summary.json --head /path/to/head-summary.json \
  --label "Local OneRec A/B" --output /path/to/ab.md --json-output /path/to/ab.json
```

Repeat `--base` or `--head` to include additional measurements of the same
revision. The report shows the number of measurements and total samples for
each side. It compares the median of each per-measurement latency metric.

For a cache-hit performance verdict, each run needs at least 100 measured
requests, comparable input fingerprints and measurement environments, and
`VLLM_GR_LIGHTWEIGHT_TIMING=0`. Every cold call must reuse zero prefix-cache tokens
and every repeated call must reuse at least one. The driver reads the counters
outside the timed Beam call. A fixed-work verdict requires the same ordered
warm cached-token counts across all measurements. Differences in reuse yield
descriptive latency. A dataset SHA recorded only in the manifest,
an older summary with incomplete evidence, a smoke run, or unconfirmed cache
observations yield descriptive latency. For v3 single-request samples, cache
counts above the recorded input length plus the SID start token are invalid.
Invalid latency and conflicting measured inputs also stop comparison.
