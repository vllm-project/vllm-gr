# AMD ROCm CI

The Ready pipeline can run vLLM-GR on an AMD Radeon PRO W7900 (`gfx1100`)
through a self-hosted Buildkite agent. This page explains what the job checks,
how to reproduce it locally and how to deploy an agent.

## What the job checks

The **ROCm Test (W7900)** step runs [run_rocm_tests.sh](../../../tools/ci/run_rocm_tests.sh)
in the [ROCm CI image](../../../docker/Dockerfile.ci.rocm) on one GPU:

| Check | What passes |
| --- | --- |
| `vllm check-env --gr` | The installed vLLM is supported and the platform is `rocm` |
| L1 unit tests | The same `l1` CPU selection as the CUDA Unit tests step |
| ROCm device tests | Paged prefix, suffix and LSE merge against FP32 attention |
| OneRec offline, ModelRunner V2 and V1 | Both return 128 beams, and the beams are identical |
| Accuracy | FP32 Hugging Face teacher forcing and an exact FP32 Beam search agree with `cumulative_logprob` and the returned beams ([thresholds](../../../tools/ci/check_onerec_accuracy.py)) |
| OneRec HTTP serving | The HTTP quickstart returns SIDs from the sample catalog |

A full run takes about ten minutes on a W7900. Performance is not gated.

The step runs for non-draft PRs labeled `ready` when the pipeline environment
sets `VLLM_GR_ROCM_CI=1`. It is a soft failure until the agent is stable, so it
reports results without blocking a merge.

## Reproduce a run locally

On a host with an AMD GPU, Docker and the ROCm kernel driver, build the image
from the repository root and run the checks:

```bash
docker build -f docker/Dockerfile.ci.rocm --build-arg PYTORCH_ROCM_ARCH=gfx1100 -t vllm-gr-ci-rocm .
docker run --rm --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home \
  --shm-size 16g -v "$PWD:/workspace" -w /workspace \
  vllm-gr-ci-rocm bash -c 'mkdir -p "$HOME" && tools/ci/run_rocm_tests.sh'
```

- Pass the render node of the GPU to test. Find it with
  `ls /sys/bus/pci/devices/<PCI address>/drm/`. The group IDs come from the
  host, because the image may number `video` and `render` differently.
- The script downloads `OpenOneRec/OneRec-1.7B` inside the container, as the
  CI job does. Set `HF_ENDPOINT` to use a mirror. Locally, you can also mount
  a snapshot yourself and pass `--model`; the CI job never mounts a model. Use
  `--skip-model` for the unit and device tests only.
- [run-rocm-ci.sh](../../../.buildkite/common/scripts/run-rocm-ci.sh) runs the
  same steps that the Buildkite job runs. Set `BUILDKITE_COMMIT` to a label of
  your choice to run it outside Buildkite.

The image builds vLLM from source for the selected architecture, which takes
about 20 minutes the first time. Later builds reuse the cached layers and only
reinstall vLLM-GR.

## Deploy an AMD agent

### Requirements

- An AMD GPU supported by the image. The pipeline step targets `gfx1100`;
  another architecture needs `VLLM_GR_ROCM_ARCH` and its own queue.
- The amdgpu kernel driver, Docker, Git and at least 100 GB of free disk
  space for images and caches. ROCm user space and PyTorch come from the image.
- Network access to GitHub, Buildkite and Hugging Face, or the alternatives in
  [hosts without direct GitHub or Docker Hub access](#hosts-without-direct-github-or-docker-hub-access).
- A host where running PR code in containers is acceptable. The job only runs
  for PRs that maintainers label `ready`. Containers run as the agent user,
  receive only the GPUs assigned to the agent and mount no host path except
  the checkout. The model is downloaded inside the container for each job.

### Install and register the agent

1. Ask the vLLM CI maintainers for an agent token for the vLLM-GR pipeline.
2. [Install the Buildkite agent](https://buildkite.com/docs/agent/v3/installation)
   and add its user to the `docker`, `video` and `render` groups.
3. Set the queue tag and keep one job per agent, because a job uses one GPU:

   ```text
   token="<agent token>"
   tags="queue=amd-w7900"
   spawn=1
   ```

### Assign GPUs

Choose one setup per host:

| Setup | Configuration |
| --- | --- |
| A reserved GPU per agent (recommended) | Run one agent per reserved GPU and set `VLLM_GR_ROCM_RENDER_DEVICES` to its render node, with `VLLM_GR_ROCM_DEVICE_ID=0` |
| GPUs shared with other users | Leave `VLLM_GR_ROCM_DEVICE_ID` unset. The job waits up to ten minutes for an idle GPU among `VLLM_GR_ROCM_RENDER_DEVICES` and locks it; see [ROCm device selection](local_runner.md#select-devices-and-test-levels) |

Set these variables in the agent's environment hook, for example
`/etc/buildkite-agent/hooks/environment`:

```bash
export VLLM_GR_ROCM_RENDER_DEVICES=/dev/dri/renderD128
export VLLM_GR_ROCM_DEVICE_ID=0
```

Exit status 75 means that no idle GPU became available or the model could not
be fetched. The pipeline retries the job twice in that case.

### Model

Like the CUDA jobs, each job downloads `OpenOneRec/OneRec-1.7B` (about 4.3 GB)
from Hugging Face inside the container before it locks a GPU, and the copy is
discarded with the container. No model directory or cache from the host is
mounted, so a job cannot read or change other files on the host. Set
`HF_ENDPOINT` to use a mirror, and `HF_TOKEN` if the model requires one.

### Cloud instances

For agents on cloud instances that are created per job or replaced often:

- Share the Docker build cache through a registry: set
  `VLLM_GR_ROCM_CI_CACHE_REF` to a repository that the agents can push to.
  Without it, each new instance rebuilds vLLM.
- Render node numbers depend on the instance type. Check them after the
  instance starts and set `VLLM_GR_ROCM_RENDER_DEVICES`.

### Hosts without direct GitHub or Docker Hub access

An agent can run behind a firewall that only allows the GitHub REST API (for
example through an API proxy), Hugging Face mirrors and the Python package
index:

- **Checkout:** install [checkout](../../../.buildkite/common/agent-hooks/checkout)
  and [fetch_github_source.sh](../../../tools/ci/fetch_github_source.sh) in the
  agent hooks directory. The hook downloads the build commit as a tarball
  through the API, including PR commits from forks, and creates a one-commit
  Git repository for the tests. Set the API and a read-only token in the
  environment hook. The token raises the API rate limit; it is read from a
  file and never passed on a command line.
- **Image sources:** the image fetches vLLM, ROCm Triton kernels and
  FlashAttention with its AITER submodule through the same API. Pass the API
  and the token file to the build, which hands the token to Docker as a build
  secret, so it is not stored in the image.
- **Models:** set `HF_ENDPOINT` to a Hugging Face mirror.
- **Ubuntu and Python packages:** if `archive.ubuntu.com` or
  `files.pythonhosted.org` is slow or blocked, set `VLLM_GR_ROCM_APT_MIRROR`
  and `VLLM_GR_ROCM_PIP_INDEX_URL` to mirrors for the image build.
- **Base image:** if the host cannot pull from Docker Hub, load the base image
  from another host, or set `VLLM_GR_ROCM_BASE_IMAGE` to a copy in a reachable
  registry. To stream an image between hosts without temporary files:

  ```bash
  docker save rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.12.0 \
    | zstd -T0 -3 | ssh <agent host> 'zstd -d | docker load'
  ```

Add the settings to the environment hook:

```bash
export GITHUB_API=https://<api proxy>          # used by the checkout hook
export GITHUB_TOKEN_FILE=/data/vllm-gr/secrets/github-token
export VLLM_GR_GITHUB_API="$GITHUB_API"        # used by the image build
export VLLM_GR_GITHUB_TOKEN_FILE="$GITHUB_TOKEN_FILE"
export HF_ENDPOINT=https://<hugging face mirror>
export VLLM_GR_ROCM_APT_MIRROR=https://<ubuntu mirror>/ubuntu
export VLLM_GR_ROCM_PIP_INDEX_URL=https://<pypi mirror>/simple
```

Keep the token file readable only by the agent user, and use a token that can
only read public repositories.

### Test the agent before enabling the step

On the agent host, as the agent user, check the GPU and run the job once from
a checkout:

```bash
ls -l /dev/kfd /dev/dri/renderD*
git clone https://github.com/vllm-project/vllm-gr.git && cd vllm-gr
source /etc/buildkite-agent/hooks/environment
BUILDKITE_COMMIT=$(git rev-parse HEAD) bash .buildkite/common/scripts/run-rocm-ci.sh
```

Without direct Git access, create the checkout with the hook instead of
`git clone`:

```bash
source /etc/buildkite-agent/hooks/environment
export BUILDKITE_REPO=https://github.com/vllm-project/vllm-gr.git
export BUILDKITE_COMMIT=<commit> BUILDKITE_BUILD_CHECKOUT_PATH=/data/vllm-gr/builds/test
bash /etc/buildkite-agent/hooks/checkout && cd "$BUILDKITE_BUILD_CHECKOUT_PATH"
bash .buildkite/common/scripts/run-rocm-ci.sh
```

The last lines print a summary in which every check passes. The logs are in
`ci-output/rocm`. After the agent connects, confirm that it appears in the
`amd-w7900` queue in Buildkite.

### Enable the step

Ask a maintainer to set `VLLM_GR_ROCM_CI=1` in the vLLM-GR pipeline
environment. Until then, the step is skipped, so PR builds do not wait for an
agent. Make the step blocking by removing `soft_fail` once it is stable.

## Troubleshoot

| Symptom | Cause and fix |
| --- | --- |
| `No AMD ROCm GPU is visible` | `/dev/kfd` or the render node is not passed to the container, or the agent user lacks the `video` and `render` groups |
| Exit 75 repeatedly | Every assigned GPU is busy, or the model download fails; check `model_download.log` |
| `No module named pybind11` or an AITER JIT build error | The image must set `AITER_TRITON_ONLY=1`; rebuild it from `docker/Dockerfile.ci.rocm` |
| Triton differs from the base image | AITER replaced it; the image build checks this and fails |
| L1 tests fail with Git errors | The checkout is not a Git repository; run from a clone |
