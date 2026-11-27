# Measurement suites

The unified entrypoint also accepts `recall`,
`benchmark`, `performance`, and `performance-ab`. Each selected pytest case owns
input preparation, an isolated driver process and required
output checks. The original shell drivers remain the source of measurement
and policy decisions; their command-line entrypoints remain supported.

```bash
python -m tools.ci.run_gate --suite recall --hardware cuda:H100
```

Resolve the existing benchmark policy before invoking a policy-controlled
suite. The adapter does not invent thresholds or relax failure modes. Recall
retains baseline provenance, fixed samples, screening/confirmation and repeat
logic. A/B retains event/base/head checks, isolated base/candidate environments,
ABBA measurements, thresholds and confirmation policy. It requires the original
PR merge checkout and A/B environment variables; a feature-branch checkout is
not a valid substitute. Performance-only remains a measurement, not an A/B pass.

Recall uses the pinned dataset and
mapping-derived triples, and prepares the reviewed model pin inside its selected
case while preserving the model revision argument used by baseline checks.
Benchmark uses a distinct cached video task catalog and triples; it never
substitutes full-mapping triples. Performance/A-B retains its pinned model
preparation inside the driver and the committed video triples default.

Explicit dataset/triples overrides pass through to the existing driver.
For the video benchmark,
provide both `CATALOG_PATH` and `TRIPLES_PATH` when overriding derived inputs.
Driver outputs and diagnostics use pytest's temporary directory.
Execution, missing reports and parse failures always
fail, even when a metric policy allows warnings.

Host adapter tests establish routing and failure handling only. New/old output
equivalence, accuracy and performance require execution on the same prepared
device with identical pins, sample identities and resolved policy before these
suites replace the existing required jobs.
