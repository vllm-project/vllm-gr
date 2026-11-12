# Dependency and version policy

## Runtime compatibility

vLLM-GR 0.30.0 supports the vLLM 0.30.0 public API and internal patch targets.
`requirements/cuda.txt` pins the CUDA runtime; `vllm_gr/_env_check.py` rejects
other public vLLM versions at initialization. A package version override or
device suffix does not change this contract. Update dependencies, guards, and
runtime regression coverage together when porting to another vLLM release.

GR may publish a patch release such as `0.30.1` while retaining the declared
`0.30.0` runtime requirement. CI checks the GR release version independently
and requires its installed distribution and imported module to have the same
complete version, including local suffixes.

Ascend uses the prepared vendor runtime described in the
[Ascend guide](../getting_started/installation/ascend.md). Its NumPy/OpenCV constraints differ from
CUDA, so those packages remain in platform-specific dependency lists.

## Build versions

`setup.py` owns version selection and device suffixes. The empty
`[tool.setuptools_scm]` section enables SCM integration without declaring a
second fallback version.

| Source | Behavior |
| --- | --- |
| `VLLM_GR_VERSION_OVERRIDE` | Complete PEP 440 version, normalized before writing; no automatic device suffix |
| Standard SCM override | `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_GR`, then `SETUPTOOLS_SCM_PRETEND_VERSION`, with the same complete-version behavior |
| Clean checkout exactly on a version tag | SCM supplies the release version, then `setup.py` adds the device suffix |
| Untagged, post-tag, or dirty checkout | Uses `VLLM_GR_BASE_VERSION` in `setup.py`, currently `0.30.0`, without guessing a next release |
| Unpacked sdist | Preserves the complete version from `PKG-INFO`, including any original device suffix |
| Source copy without Git or sdist metadata | Uses the same declared base version |

CUDA has no device suffix; NPU and CPU use `+npu` and `+cpu`. If the base
already has a local version, append `.npu` or `.cpu`. The final normalized
version is written to `vllm_gr/_version.py` and passed to setuptools. The
temporary distribution-specific SCM override is restored after generation.
Keep a complete override set for every build step, including sdist rebuilds,
when automatic device suffixing must remain disabled.

An sdist's filename and prepared metadata must declare the same version for
pip resolution. Rebuilding it for another target preserves that version while
selecting dependencies for the requested device; its original suffix does not
identify the rebuild target. Build from a checkout to create a new artifact
with a different device suffix. Set complete version overrides when creating
the sdist, and keep them unchanged during subsequent resolution and builds.

CI resolves the GR package version from overrides, sdist metadata, or a clean
Git tag independently of the pinned runtime version. Untagged, post-tag, and
dirty checkouts retain the declared base. Historical checkouts that predate
tag-based selection keep their original fixed-base policy. Resolution uses
only the standard library and Git; installed-version verification uses
`packaging` for PEP 440 normalization.

Builds require setuptools-scm 8.0.4 or newer. The 8.0.0 version-file template
generates invalid Python; the packaging tests execute the generated file to
catch this class of failure.

Set `VLLM_GR_TARGET_DEVICE` explicitly for source installs and builds. Build
isolation can hide Torch from backend detection. A wheel's dependencies are
fixed when built; installing that wheel does not rerun platform detection.

```bash
VLLM_GR_TARGET_DEVICE=cpu python -m build --wheel
VLLM_GR_TARGET_DEVICE=npu python -m build --sdist
VLLM_GR_TARGET_DEVICE=cuda VLLM_GR_VERSION_OVERRIDE=0.30.0rc1 python -m build --wheel
```

Install the `build` frontend first if needed. Do not edit the generated
`_version.py`. The lightweight source-tree fallback in `vllm_gr/version.py`
is for imports before building, not a source for wheel metadata.

## Dependency constraints and developer tools

Use `requirements/common.txt` for shared libraries and platform files for
runtime and ABI constraints. Exact pins, lower bounds, and bounded ranges serve
different purposes; keep the reason for non-obvious constraints next to them.
Changing a constraint requires evidence from the affected installation and
runtime paths, not just a successful resolver run.

`pyarrow==23.0.0` retains the existing data-loading baseline. Its introduction
does not document an exclusive API requirement for that release. Keep the pin
until separate Parquet/data-loading checks establish a wider supported range
on the CUDA and Ascend NumPy stacks. General dependency lower bounds do not
claim that every newer release has been validated.

Ruff owns Python formatting and linting through pre-commit. Black is not a
development dependency; isort's `--profile black` names a formatting preset
and does not require installing Black. Installation with pip and uv follows
the same target selection and version constraints; see the
[CUDA quickstart](../getting_started/installation/cuda.md).

## Packaging validation

Run the isolated packaging tests with pip, pytest, pytest-timeout, setuptools,
setuptools-scm, packaging, and Git installed:

```bash
python -m pytest --confcutdir=tests/packaging -q tests/packaging/test_install_metadata.py
```

The tests cover actual SCM tags, fallback behavior, complete overrides,
repeated generation, wheel/sdist metadata, and pip resolution of an sdist
across build targets. The
[wheel smoke test](../contributing/tools/README.md#packaging) additionally checks an installed
wheel outside the checkout. These checks do not validate accelerator inference.
