# Dependency and version policy

Use the dependency files for the target runtime when updating supported versions.

## Changing a dependency

| Dependency | Where to declare it |
| --- | --- |
| Required on every target | [Shared runtime requirements](../../requirements/common.txt) |
| Specific to a target or its vendor runtime | The corresponding file in [requirements/](../../requirements/) |
| Needed to build the package | `[build-system].requires` in [pyproject.toml](../../pyproject.toml) |
| Needed for development or an optional feature | The corresponding extra in `[project.optional-dependencies]` in [pyproject.toml](../../pyproject.toml) |

Explain non-obvious pins or bounds next to the requirement. When changing a
constraint, verify installation and exercise the affected code on each platform
that uses it. The [installation guides](../getting_started/installation/README.md)
describe which dependencies the platform environment supplies.

## Runtime compatibility and package versions

A vLLM-GR release can keep its existing vLLM requirement. When changing the
supported vLLM version, update the relevant requirements, the
[runtime compatibility checks](../../vllm_gr/_env_check.py), and the tests for
affected runtime paths together.

The installed package metadata and `vllm_gr.__version__` must agree. A wheel
built from a source distribution must retain that distribution's version.
See [setup.py](../../setup.py) and the [Git version helper](../../tools/packaging/git_version.py)
for release tags and version overrides.

## Packaging validation

With the development dependencies installed, run the packaging tests after
changing requirements or version metadata.

```bash
python -m pytest --confcutdir=tests/packaging -q tests/packaging/test_install_metadata.py
```

### Wheel smoke test

Check the built wheel's dependency metadata and import it outside the
checkout using the target you changed (`cpu`, `cuda` or `npu`).

!!! warning "Build artifacts"

    Run the wheel smoke test in a clean checkout. It replaces existing build artifacts.

```bash
VLLM_GR_TARGET_DEVICE=cpu ./tools/packaging/run_wheel_smoke_test.sh --python python3
```
