# Benchmark policy changes in a pull request

A pull request may change both its tests and `.github/benchmark-gate.yaml`.
For a non-draft PR targeting `main` or `decode_graph` with the `ready` label,
CI automatically uses the proposed policy for that PR's benchmark run; a
GitHub approval is not required to start or pass the benchmark gate.

The policy job publishes the base and proposed policy and their SHA-256 hashes
in its run summary. It reads the candidate policy as data and validates it with
the resolver from the pinned target-base commit. The PR head, base, and merge
revision must match the run, and the merged policy must match the head policy.
The final gate checks the revisions and policy digest again before reporting
success. If these change during the run, rerun CI on the current revisions.
When the policy is unchanged, CI uses the base policy. Existing benchmark
execution and other checks remain blocking according to their normal rules.
The `abtest` label keeps its existing A/B-only behavior.

The candidate policy must be valid under the **base** resolver. If a change
needs a new policy format or resolver feature, deploy that trusted resolver
first; a pull request cannot supply its own executable resolver. If the base
does not have the policy runtime, the workflow uses its inlined bootstrap. If
the base lacks the trusted policy resolver or normalizer, the policy job fails
closed rather than executing candidate-owned code to supply fallback values.

## Jenkins and repository configuration

This implementation changes GitHub Actions; it does not automatically change
an independent Jenkins pipeline. Jenkins needs its own equivalent policy
selection and validation. It should read the resolver and policy runtime from
the same base SHA and record the effective policy digest and PR revisions.

The pull-request workflow is itself part of the candidate merge commit and can
be modified by a PR. Protect the implementation using a repository ruleset
that requires a trusted workflow or GitHub App check, or appropriately
configured CODEOWNERS and branch protection for workflow, resolver and policy
runtime changes. A privileged trusted check must treat PR files as data and
must not execute PR code.
