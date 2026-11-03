# Benchmark policy changes in a pull request

Benchmark policy normally comes from the pull request's target-base commit.
A pull request may change both its tests and `.github/benchmark-gate.yaml` and
use the changed policy in its own benchmark run once that exact revision has an
eligible approval. The candidate policy does not take effect automatically.

## How to use it

1. Open a non-draft pull request targeting `main` or `decode_graph`, make the
   test and policy changes together, and add the existing `ready` label.
2. The policy job publishes the base and proposed policy and their hashes in
   its run summary, including when approval is still pending. Review these
   policy changes together with the tests in the pull request.
3. A human other than the pull
   request author, who currently has repository write or admin access, must
   submit an ordinary GitHub **Approve** review for the current head commit.
   A label, comment, bot review, or approval of an older head is not enough.
4. The review event starts a fresh validation. The benchmark-policy job reads
   the candidate policy as data and validates it with the resolver from the
   pinned target-base commit. The log records the policy source, head and base
   revisions, SHA-256 digest, and approving reviewer; the policy summary shows
   the effective device thresholds.
5. The benchmark gates use the approved policy. The final blocking gate checks
   the PR refs, approval, and policy digest again before reporting success.

Pushing a new head requires a new approval, even if the policy bytes are
unchanged. Submitted, edited and dismissed review events re-evaluate the PR.
If the approval is dismissed or the reviewer loses permission, the changed
policy no longer qualifies. The PR head, base, and merge revision must still
match the run, and the merged policy must match the reviewed head policy; if
they do not, update the branch and request a new review. Without a qualifying
review, a changed policy blocks the gate; it does not silently fall back to
the base policy and report success.

When the policy is unchanged, CI uses the base policy without an additional
policy approval. The existing checks, including benchmark execution failures,
remain blocking according to their normal rules. The `abtest` label keeps its
existing A/B-only behavior.

The candidate policy must be valid under the **base** resolver. If a change
needs a new policy format or resolver feature, deploy that trusted resolver
first; a pull request cannot supply its own executable resolver to authorize
its policy. When the base does not yet contain the approval runtime, the
workflow performs the same review checks using its inlined bootstrap logic.
This allows the PR introducing the workflow to change policy after approval;
the workflow itself must be protected by trusted repository checks.
If the target base does not have the trusted policy resolver or normalizer,
the policy job fails closed. Unlike the former strict-bootstrap path, it does
not execute a candidate-owned resolver to supply fallback policy values.

## Jenkins and repository configuration

This implementation changes GitHub Actions; it does not automatically change
an independent Jenkins pipeline. Jenkins must implement equivalent checks or
consume a trusted, revision-bound decision before using a candidate policy.
It should read the policy, resolver, and approval runtime from the same base
SHA and record the effective policy digest and PR revisions.

The pull-request workflow is itself part of the candidate merge commit and
can be modified by a PR. Therefore this workflow alone cannot enforce an
unbypassable approval requirement against a PR that edits CI. Review checks do
not by themselves constitute a trusted required check. Protect the
implementation using a repository ruleset that requires a trusted workflow or
GitHub App check, or appropriately configured CODEOWNERS and strict branch
protection for workflow, resolver, policy runtime, and approval changes. A
privileged trusted check must treat PR files as data and must not execute PR
code. The two approval checks reduce, but cannot eliminate, a race in which a
review is dismissed or a reviewer's permission changes after the final check;
repository merge protection and fresh required checks remain necessary.
