# Running gate suites

Run the entrypoint from the checkout in the prepared device environment:

```bash
python -m tools.ci.run_gate --suite unit --hardware cuda:H100
python -m tools.ci.run_gate --suite resource --hardware npu:A3 --cards 2
python -m tools.ci.run_gate --suite functional --hardware cuda:H100 --cards 1
```

The unit suite runs L1 through L3. Override its expression with `--levels`.
The target option uses #867's native pytest selection. Card allocations are
verified against runtime-visible devices before selecting `cards_N` markers.
Host tests and tests without an explicit card count join the single-card lane.

Repeat `--suite` to run several suites. The entrypoint prints to the console
and returns the first failing exit code. It creates no report directory.
The dedicated resource and functional suites use `--strict-gate`: skips,
expected failures, or a run with no passes fail the gate. Unit tests retain
their ordinary hardware availability skips.

Resource probes require an isolated allocation and
`VLLM_GR_ISOLATED_RUNNER=true`. Keep accelerator tests serial on each allocated
runner. The process helper bounds execution and cleans up owned descendants.
