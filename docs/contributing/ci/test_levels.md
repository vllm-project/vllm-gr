# Test levels and platform markers

Classify each pytest case by the **scope of the behavior it checks** (one
level) and **where it can run** (one or more platform markers).

<a id="test-levels-and-ci-selection"></a>

## Validation levels

| Level | Use it for | GR example |
| --- | --- | --- |
| L1 | Fast checks of an isolated rule or public contract | Reject an invalid beam width. |
| L2 | A small, representative execution through a component or device | Compare a CUDA or Ascend kernel with a reference on one small input. |
| L3 | Integration across components or a broader functional matrix | Exercise a model request through the frontend, scheduler, and worker. |
| L4 | Expensive accuracy, performance, or large parameter sweeps | Measure Recall on a fixed dataset or check many attention shapes. |
| L5 | Long-running stability and stress | Check memory growth across repeated model reloads over an extended run. |

Choose the level using both regression risk and execution cost. If a large
matrix tests an important behavior, keep a representative case at a lower
level and classify the remaining cases separately.

## Marking a test

Use the [registered pytest markers](../../../pyproject.toml). A marker can
apply to a module, class, function, or individual parameter.

<a id="test-tiers"></a>

### Platform markers

| Platform marker | When to use it |
| --- | --- |
| `cpu_test` | The case can run without an accelerator. Its Python dependencies and shared fixtures still apply. |
| `gpu_test` | The case exercises CUDA. |
| `npu_test` | The case needs the Ascend runtime or device. |

Use both accelerator markers for a case that can run on either platform.
The markers describe eligibility. Keep an availability check with a test
that must skip when its device is unavailable. In `tests/kernels/` and
`tests/e2e/`, cases with no platform marker default to `gpu_test`. Mark CPU
and Ascend cases there explicitly. Add `model_test` when weights are
required. The `slow` and `slow_test` markers require `--run-slow`.

### Host test example

This L1 test checks invalid configuration.

```python
import pytest

from vllm_gr.config import GRConfig


@pytest.mark.l1
@pytest.mark.cpu_test
def test_zero_beam_width_is_rejected():
    with pytest.raises(ValueError, match="beam_max_width"):
        GRConfig(beam_max_width=0)
```

### Parameterized cases

Use an explicit level when all cases of a test have the same scope. If the
level depends on the parameters or a supported pytest option, use the
[level inventory](../../../tests/level_inventory.json). Key the entry by its
pytest node ID without the parameter suffix. Choose one default and make
the exceptions non-overlapping. For example, the
[CUDA attention classification test](../../../tests/ci/test_level_policy.py)
checks that the small cases are L2 and the rest of the matrix is L4. Keep
the entry in sync when moving or renaming the test.

### Hardware selection

Use standard pytest markers for the platform, eligible accelerator models and
minimum card count. For example, add `@pytest.mark.gpu_test`, `@pytest.mark.H100`
and `@pytest.mark.cards_2` to a two-card H100 test. A job can select exactly
those declarations with `-m 'gpu_test and H100 and cards_2'`. Use separate
`pytest.param(..., marks=[...])` entries when platforms need different counts.

CUDA targets are L20, A100, H800, H100, L4 and B200. Ascend targets are A2, A3
and A5. Lowercase `l4` selects a test level. Uppercase `L4` selects a GPU model.
On the allocated runner, this command includes host, generic CUDA and H100
tests at levels L1 through L3:

```bash
python -m pytest tests --test-hardware=cuda:H100 -m 'l1 or l2 or l3'
```

Use `--test-hardware=npu:A3` for an A3 runner or `--test-hardware=host` for
host-only tests. Set device visibility to the allocated cards: during setup
every visible device must match the explicit target. A wrong target stops
the run and an insufficient card count skips the case. With plain `-m`,
unavailable model- or card-marked cases skip. Collection and cases skipped by
pytest do not probe accelerators.

## Running by level

In a prepared development environment, run these commands from the repository
root. The first checks that each collected case in the selected module has exactly
one level. The second runs that module's L1 host tests.

```bash
python -m pytest --collect-only --strict-test-levels tests/ci/test_level_policy.py
python -m pytest --strict-test-levels -m 'l1 and cpu_test' tests/ci/test_level_policy.py
```

To run a level across the suite on CUDA or Ascend, use the
[local runner guide](local_runner.md) and the [level runner](../../../tools/ci/run_test_level.sh).
