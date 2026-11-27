# Optional Buildkite gate suites

The `gate-tests` PR label opts a non-draft PR into the optional gate jobs.
The generator reuses `.buildkite/common/ci_mirror_hardwares.yml` and its
existing preset expansion. Each step invokes `run_gate`, with an explicit
hardware target, card allocation and suite. Buildkite displays each result
and propagates failed or cancelled steps through the pipeline.

`.buildkite/gate-runners.json` records provisioned allocations. The existing
`l4_1` preset runs the unit suite. L20, A100, H800, H100, B200 and A2/A3/A5
have no allocations configured. Add real presets before enabling those targets.
Multiple allocation entries give independent one-, two- or four-card jobs.
Each preset must expose exactly that many matching devices to its process.

Preview all configured unit jobs, or an explicitly provisioned selection:

```bash
python -m tools.ci.gate_matrix
python -m tools.ci.gate_matrix --targets cuda:L4 --suites unit
```

The generator validates model/suite routing, requires isolation for resource
probes and resolves the committed benchmark policy for each allocation.
The original PR-event checks and base/candidate provenance remain mandatory
for A/B, which runs through its existing dedicated workflow.

Supply prepared runtimes and Hub access through the runner environment.
The model fixtures prepare missing inputs on each executing machine.
The entrypoint prints pytest output and produces no generic gate report.
Required measurement outputs remain owned by their existing drivers.
