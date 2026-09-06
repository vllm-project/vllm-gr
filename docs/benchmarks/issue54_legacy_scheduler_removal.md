# Issue 54 legacy GR scheduler removal

The per-step GR Beam scheduler path was removed after the persistent ASYNC GPU
path completed Step 9. Constraint Table requests now have one production
lifecycle:

```text
native ADD
  -> one stable Request
  -> GRAsyncScheduler look-ahead
  -> device-resident Beam state
  -> FINAL_ONLY result
```

Removed components:

- the `ADD_BATCH`, `BEAM_REQUEST_STEP_UPDATE`, `BEAM_SESSION_TEMPLATE`, and
  `SUBMIT_SESSION` EngineCore wire extensions;
- per-step Request reconstruction, EngineCore self-advance, output buffering,
  and session-template handling;
- the `Scheduler.schedule` prefix-cache and Beam-metadata monkey patch;
- the synchronous in-frame `sample_tokens` shortcut used by the old
  engine-driven mode; and
- frontend branches that selected between legacy and persistent execution.

Constraint Table configuration is normalized at engine creation to
`persistent_request=True`, `persistent_runner_state=True`, Worker decisions,
and `GRSchedulingMode.ASYNC`. The engine selects `GRAsyncScheduler` and enables
vLLM async scheduling automatically. A historical `beam_engine_driven=True`
setting fails fast; `False` and `null` remain parseable only so older config
files can migrate without restoring the deleted path.

Trie-constrained and unconstrained Beam generation continue through their
ordinary frontend loops and vLLM's native request protocol. Ordinary vLLM
requests remain supported by the upstream scheduler implementation;
`Scheduler.schedule` is no longer replaced process-wide.

NPU code keeps the same persistent design boundary. Hardware validation remains
deferred until an NPU environment is available.

## Validation

- CPU regression: `1670 passed, 3121 skipped` (the skips are environment and
  accelerator gated); log: `/tmp/vllm-gr-legacy-removal-full-cpu-final.log`.
- CUDA kernel regression through `gpu run --gpu-ids 3`: `44 passed`; log:
  `/tmp/vllm-gr-legacy-removal-cuda-tests.log`.
- OneRec GPU profile through `gpu run --gpu-ids 3`, with Prefix Cache disabled
  and FULL Beam CUDA Graph enabled: status `complete`, 128/128 final sequences
  passed the Constraint Table, no intermediate Beam-state D2H occurred, and
  the three measured requests had 25.67 ms mean / 26.89 ms P99. Artifact:
  `/tmp/vllm-gr-legacy-removal-graph-disabled.json`. Its aggregate evidence
  gate remains `FAIL` because this Worker-side profiler does not capture the
  EngineCore Scheduler/Request CPU spans required by the existing Step 0 trace
  gate; the artifact records that limitation explicitly.
- RecIF-Bench `ad` smoke test through `gpu run --gpu-ids 3`: two B=1, W=128
  requests completed with one native request and one Scheduler ADD per session,
  zero decode request objects, one final D2H, and balanced GR-state creation and
  release. Prefix Cache was disabled and Beam CUDA Graph was enabled. Artifact:
  `/tmp/vllm-gr-legacy-removal-recif/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b_ad_bw128_2.json`.
