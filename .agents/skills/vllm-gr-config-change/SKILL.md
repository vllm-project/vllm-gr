---
name: vllm-gr-config-change
description: Add, change, deprecate, or retire vLLM-GR settings, defaults, validation, or configuration transport across YAML, Python, CLI, engine, and worker boundaries. Use for configuration implementation changes, not simply choosing deployment values.
---

# Change GR configuration

Paths below are relative to the repository root. Read
`docs/contributing/development.md` and `docs/configuration/README.md` for the
current ownership and public contract, and the relevant design contract when
the change affects runtime behavior.

## Establish the setting's contract

- Identify whether the setting is a native vLLM argument, a GR engine setting,
  a frontend/listener option, a provider-specific option, or derived state. Use
  the existing owner and transport boundary.
- Determine its public name, type, default, precedence, valid values, scope,
  and behavior when omitted. Distinguish omitted values from explicit `null`,
  `false`, or empty values where the contract does. For renamed or retired
  settings, identify existing callers and establish compatibility behavior
  before editing.
- Identify the supported devices, runner generations, and affected entrypoints.
  Trace actual producers and consumers. A frontend-only option does not need
  to cross into the worker snapshot.

## Trace and implement the change

Use this map to inspect the applicable boundaries. Follow actual references,
callers, and consumers rather than changing every layer indiscriminately.

| Boundary | Source to inspect |
| --- | --- |
| Declaration, defaults, validation | `vllm_gr/config/models.py`, `vllm_gr/config/schema.py`, `vllm_gr/config/validation.py`, `vllm_gr/config/default_config.yaml` |
| YAML layering and canonical mapping | `vllm_gr/config/parser.py`, `vllm_gr/engine/startup.py` |
| Public Python and CLI startup | `vllm_gr/entrypoints/llm.py`, `vllm_gr/entrypoints/cli/serve.py`, `vllm_gr/entrypoints/cli/serve_config.py`, `vllm_gr/engine/arg_utils.py` (`vllm_gr/engine/arg_utils_gr.py` is its compatibility import) |
| Frontend/listener settings | `vllm_gr/entrypoints/grpc/serve.py`, `vllm_gr/entrypoints/serve/management.py`, applicable startup parser |
| Device admission and backend plan | `vllm_gr/config/execution.py`, `vllm_gr/config/backend_plan.py`, `vllm_gr/platforms/config.py`, `vllm_gr/platforms/planning.py`, applicable CUDA/Ascend services |
| Worker binding and consumers | `vllm_gr/v1/worker/runner_binding.py`, `vllm_gr/v1/worker/gr_context.py`, the relevant runner, attention, or constraint implementation |

Preserve these boundaries where applicable:

- Keep the owning scope's field declarations, schema metadata, packaged
  defaults, validation, and public documentation consistent. Preserve unaffected
  platform-specific defaults and admission behavior. Preserve precedence:
  packaged defaults, imported YAML, startup YAML, then explicit startup
  arguments. Mappings merge by key and lists replace inherited lists, subject
  to validation. Maintain the distinction between derived defaults and explicit
  user values.
- Keep native vLLM options in native arguments and let native validation own
  them. GR settings use the canonical `additional_config["vllm_gr"]` snapshot
  when transported into the engine. Preserve unrelated native options,
  `additional_config` entries, and native-only execution paths.
- Keep frontend/listener settings out of the engine-to-worker snapshot. For
  transported settings, preserve the canonical primitive mapping, driver-side
  path resolution, admitted backend plan identity, and worker validation. Do
  not silently reinterpret a previously published field or change the wire
  schema unless the task calls for it.
- Inspect applicable CUDA/Ascend and V1/V2 paths, fallbacks, tests, examples,
  and CLI arguments. Update consumers that need the change and preserve
  unaffected behavior. After editing, search for superseded names, defaults,
  mapping keys, and aliases; determine whether each remaining reference is an
  intentional compatibility path or an omission.

## Validate the resulting contract

Read `docs/contributing/ci/test_levels.md` and the affected fixtures and device
requirements before selecting tests. Start with the current regression tests
in `tests/config/` and `tests/engine/test_gr_config.py`, and select applicable
`tests/platforms/`, entrypoint, worker, or contract tests. Use the existing
configuration metadata check in `tools/pre_commit/validate_config.py` when
changing declarations or schema metadata.

Cover the contract that changed: default, explicit, and invalid inputs;
precedence and omitted-versus-explicit values; canonical mapping round trips;
and compatibility or unchanged native behavior where relevant. For a public
startup setting, verify both `GRLLM` and `vllm serve --gr` where exposed. For a
frontend-only setting, verify the owning frontend and that the option stays out
of the worker snapshot. For a worker-consumed setting, verify the admitted plan
and affected consumer or device branch; successful host parsing alone does not
establish runtime correctness.

Update the affected public configuration documentation and examples. If a
setting affects GPU performance, check `docs/configuration/gpu_tuning.md` and
update any affected parameter guidance, eligibility, or resource calculation.
For measured performance comparisons, follow the benchmark skill. If a runtime
patch target, owner, activation, or removal changes, follow the patch inventory
and contract requirements in `AGENTS.md`.

Report the changed contract, affected entrypoints and platforms, commands run,
and outcomes, including device or integration coverage that could not run.
