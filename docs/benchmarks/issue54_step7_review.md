# Issue #54 Step 7 review: final-only frontend lifecycle

Step 7 closes the persistent request lifecycle in both synchronous and online
frontends. Real GPU acceptance was run from commit
`dd48845061208a06883b9593e8e1015388bca2fe`; NPU hardware validation remains
deferred with the rest of the NPU acceptance matrix.

## Implemented boundary

- Online prepared requests are marked `gr_persistent_request` before EngineCore
  admission, so terminal shutdown can discover and release them.
- Failure or cancellation during batch submission and final-only collection
  aborts every prepared persistent native request through the normal internal
  abort API.
- Synchronous collection aborts every still-live native request if an engine
  step fails. Admission failure also releases frontend ownership.
- Cleanup failure stays authoritative and records the original failure as its
  chained cause and exception note. Resources are therefore not silently
  reused after an unsuccessful release.
- The online path returns native IDs for persistent GR even when the legacy
  beam-request capability is absent. This lets final-only lifecycle tracking
  use the actual native request identity.

The persistent constraint-table path produces no intermediate frontend Beam
result. GPU counters in both acceptance modes recorded one native request, one
scheduler admission, no decode request objects, no legacy step updates, no
intermediate Beam-state D2H, and exactly one final-result D2H per completed
session.

The Trie and ordinary non-persistent compatibility path remains present. Its
removal depends on the Step 8 parity gate; the known prefix hit/miss and
legacy-hit output differences mean deleting the reference path at Step 7 would
remove evidence before the discrepancy is resolved.

## Validation

CPU-only full repository regression:

```text
1933 passed, 3121 skipped
```

The focused frontend, terminal shutdown, and EngineCore abort suite passed 36
tests. It includes online collection failure, failure while attaching metadata,
offline engine-step failure, immediate abort, shutdown, release retry, and
cleanup-error attribution.

All GPU commands were submitted through `gpu run`. The OneRec-1.7B BF16
B=1/W=128/S=3/L=1024 persistent lifecycle probe used prefix caching disabled
and the RecIF-derived constraint triples. It injected real aborts at Prefill and
Decode boundaries, then tested both drain and immediate same-process reuse.

| GPU mode | Artifact | Result | Graph evidence |
| --- | --- | --- | --- |
| eager | `/tmp/vllm-gr-step7-final-only-lifecycle-eager.json` | four lifecycle cases PASS | Beam Graph disabled |
| CUDA Graph | `/tmp/vllm-gr-step7-final-only-lifecycle-graph.json` | four lifecycle cases PASS | width 128 captured; full dispatch counter advanced on every completed probe |

The required native completion extension was loaded from the isolated Step 6
runtime build. Its source is byte-identical to the current
`gr_completion_ticket.cc`; the binary SHA256 is
`e7ee3380631ad44babcbdd8a4864ad7e3cf944ccdcd855f63e73e6bffd860f02`.
The shared venv was not modified.

## Remaining acceptance work

Step 8 must execute the GPU correctness matrix across eager/Graph, prefix-cache
disabled/miss/hit, supported widths and dtypes, and retain exact output
fingerprints. NPU eager/ACL/cascade and cross-device parity remain deferred
until hardware is available. Step 9 owns repeated E2E latency/goodput, trace
attribution, component memory/soak, and final CI.
