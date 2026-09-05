# Issue #54 Step 8 review: GPU correctness matrix

Step 8 validates the persistent final-only GPU path across execution mode,
Prefix Cache state, dtype, and supported Beam width. The clean same-commit
matrix was run from `98e17deaba20124382da962156ddf34cfa09a978` with
OneRec-1.7B, B=1, L=1024, S=3, and RecIF-derived constraint triples. All GPU
commands were submitted through `gpu run`.

## Matrix result

The `compiled` column enables model compilation while disabling the vLLM-GR
FULL Beam Graph. The `graph` column enables the same compiled model path plus
the FULL Beam Graph. This isolates Beam Graph behavior from the separate
uncompiled `enforce_eager` numerical mode.

| Prefix state | dtype | width | compiled / Graph full ABI | Direct Prefix evidence | Constraint result |
| --- | --- | ---: | --- | --- | --- |
| disabled | BF16 | 128 | `a51a5ed2b2fb…` / identical | 2048 queried, 0 reusable; cache disabled | 128/128 valid |
| cold miss | BF16 | 128 | `a51a5ed2b2fb…` / identical | 2048 queried, 0 hit | 128/128 valid |
| warm hit | BF16 | 128 | `2b712b9d1905…` / identical | 2048 queried, 2016 hit | 128/128 valid |
| disabled | FP16 | 128 | `57ec06cff3c6…` / identical | 2048 queried, 0 reusable; cache disabled | 128/128 valid |
| disabled | BF16 | 32 | `f91cb682170b…` / identical | 2048 queried, 0 reusable; cache disabled | 32/32 valid |

Each row represents two measured sessions in each execution mode. Graph
diagnostics advanced by four FULL dispatches for the two sessions, exactly two
Decode stages per session. Compiled mode recorded zero Beam Graph dispatches.
Every result repeated its full public ABI fingerprint within the process. The
validator covers tokens, scores, logprobs, finish fields, and the final SID
triple against all 16,981 catalog triples.

The artifacts are `/tmp/vllm-gr-step8-{compiled,graph}-*.json`. Their report
level exit gate is `FAIL` because Step 8 runs intentionally omitted a device
trace and predated the Step 9 component-memory diagnostics. That status is not
used as a correctness PASS; Step 9 supplies both missing evidence classes.

## Numerical observations

Prefix Cache disabled and a real cold miss are byte-for-byte identical in both
compiled and Graph modes. This is direct evidence that disabling Prefix Cache
does not reproduce the old accuracy anomaly on the persistent path.

A warm Prefix hit differs from full Prefill, but compiled and Graph remain
identical to each other for that hit. Earlier diagnostics localized the first
difference to Prefill model computation: shape M=1024 versus the cached-tail
M=16 path changes BF16 QKV projection results before Decode. The cache's retained
Prefix K/V was elementwise equal in the bounded 28-layer diagnostic. This is a
known model/numerical consistency issue and is not silently accepted as strict
hit/miss parity.

The true uncompiled `enforce_eager` results also differ from compiled results.
They were captured separately at `2ed19dabb00deb33d597065f2edfd88154750508`.
Those differences do not implicate the Beam Graph because the clean compiled
versus Graph pairs are exact.

## Acceptance boundary

The GPU persistent correctness matrix passes for the covered OneRec shape. NPU
eager, ACL Graph, and cascade hardware validation remains deferred until an NPU
environment is available. Cross-device numerical parity therefore remains an
explicit hardware follow-up and does not block completion of the software
integration requested for Steps 6-9.
