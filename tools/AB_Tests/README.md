# A/B benchmark utilities

Use the [`ab-offline` CLI](../../docs/cli/README.md#offline-ab) with your local
CUDA runtime, model, and OneRec dataset. The
[OneRec catalog guide](../../examples/online_serving/catalog/README.md#build-a-catalog-from-the-onerec-dataset)
shows how to generate the constraint triples to pass with `--catalog`.

The video benchmark uses the gated OpenOneRec dataset. Accept its terms, run
`hf auth login`, and use the downloader to write the data and its manifest.

```bash
python tools/AB_Tests/download_onerec_dataset.py --data-dir /path/to/onerec

vllm bench ab-offline --gr \
  --model /path/to/OneRec --data-dir /path/to/onerec --catalog /path/to/triples.json \
  --beam-width 128 --input-length 1024 --output /path/to/ab-result.json
```

## Compare revisions

Use a separate virtual environment for each source checkout. The benchmark CLI
prefers the checkout from which `vllm_gr` is loaded, so changing only the
working directory can measure the same revision twice.

In Bash, set the two checkout paths and choose two fresh virtual environment
paths. Follow the [CUDA installation guide](../../docs/getting_started/installation/cuda.md)
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
CATALOG=/path/to/triples.json

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
script = (root / scenarios.SCENARIOS["ab-offline"].path).resolve(strict=True)
if not script.is_relative_to(checkout):
    raise SystemExit(f"Benchmark script escapes checkout {checkout}: {script}")
sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if sha != sys.argv[1]:
    raise SystemExit(f"Expected Git SHA {sys.argv[1]}, loaded {sha}")
print(f"GR package {package}\nBenchmark script {script}\nGit SHA {sha}")
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

Keep the dataset manifest and verified Git SHAs with the results. Convert the
raw results with `generate_summary.py` and compare the resulting summaries with
`compare_ab.py`.

Direct `generate_summary.py` invocations accept `--repo-dir` for compatibility.
Online summaries require `--catalog-path` matching the server configuration.
Supply `--gpu-uuid` when you have the full UUID of the device that ran the benchmark.
Without a UUID, the summary records the GPU identity as unknown.
