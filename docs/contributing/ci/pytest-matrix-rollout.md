# Pytest suite matrix rollout

The opt-in `pytest-gates.yml` workflow plans one job per hardware target, card count and
suite. Every device job uses `python -m tools.ci.run_gate`; the final summary
requires all planned jobs and matching current-commit reports. Failure,
cancellation, skipped jobs/tests and missing artifacts cannot satisfy the gate.
The existing accelerator concurrency group is retained while runner allocation
is being provisioned. The workflow contains no separate model-download jobs.

No new runners are registered yet. All six GPU SKUs (L20, A100, H800, H100, L4,
B200) and three Ascend SKUs (A2, A3, A5) are explicitly disabled in
`.github/gate-runners.json`. Requesting one fails in planning before scheduling
any device job. Empty labels never fall back to a generic self-hosted runner.

Each target has an `allocations` object keyed by visible card count (1 through 8).
Each lane requires a dedicated `routing_label` included in its real runner labels,
an absolute prepared Python path,
an existing benchmark policy board selector (or explicit `default`) and isolation status. Provision the
matching vendor runtime and development dependencies; the workflow installs
only the checkout itself with `--no-deps`, then verifies required imports.
Hardware policy checks the actual selected devices at execution time. CUDA-only
functional/recall/performance suites cannot be scheduled on Ascend. Run resource
probes only on genuinely isolated hosts, including permission to inspect devices.

This is a staged migration, not a replacement for required checks yet:

1. Merge hardware selection, input preparation and the pytest adapters.
2. Register and validate the actual runners and policy selectors.
3. Run old/new gates on identical model/data identities and compare reports,
   accuracy, graph behavior and repeated measurements on each supported target.
4. Only then move existing PR job triggers to the unified entrypoint and remove
   redundant download/orchestration steps. Retain the existing required summary
   check name and branch-protection semantics during that reviewed cutover.

A/B remains in the existing PR workflow because it requires the synthetic merge
commit, event base/head and verified base/candidate environments. Manual dispatch
does not synthesize that provenance. The `performance-ab` pytest adapter is ready
for that workflow's later cutover, with its original environment unchanged.

Reruns overwrite artifacts by job ID within the same workflow run. Retried jobs
replace their previous artifacts; successful jobs omitted from a partial rerun
retain theirs. The summary checks run ID, commit, card count and board.

Host matrix tests exercise planning and summary outcomes. They do not establish
runner availability, hardware compatibility or benchmark equivalence.
