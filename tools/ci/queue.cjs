// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Loaded only from the default branch by a privileged workflow. Never execute PR code here.
const STATE_BRANCH = 'ci/queue-state';
const STATE_PATH = 'queue-state.json';
const DEVICE_WORKFLOW = 'device-validation.yml';
const TERMINAL_STEP = 'Complete device validation';
const ACTIVE = new Set(['queued', 'in_progress', 'waiting', 'pending', 'requested']);
const CONTEXT = 'Benchmark blocking gate';
const VALIDATION_PROTOCOL = 2;

function sameRepository(pr, repositoryId) {
  return Number.isSafeInteger(repositoryId) && pr.head.repo?.id === repositoryId &&
    pr.base.repo?.id === repositoryId;
}

function priority(labels) {
  const names = labels.map((label) => typeof label === 'string' ? label : label.name);
  return names.includes('ci-priority:high') ? 0 : names.includes('ci-priority:normal') ? 1 : 2;
}
function order(candidates) {
  return [...candidates].sort((a, b) => a.priority - b.priority ||
    a.queuedAt.localeCompare(b.queuedAt) || a.pr - b.pr);
}
function snapshot(pr) {
  if (pr.state !== 'open' || pr.draft || !pr.mergeable ||
      !pr.labels.some((label) => label.name === 'ready')) return null;
  const value = {pr: pr.number, head: pr.head.sha, base: pr.base.sha,
    merge: pr.merge_commit_sha, baseRef: pr.base.ref, priority: priority(pr.labels)};
  if (![value.head, value.base, value.merge].every((sha) => /^[a-f0-9]{40}$/.test(sha))) return null;
  return value;
}
function keyOf(value) {
  return [value.pr, value.baseRef, value.head, value.base, value.merge].join(':');
}
function runName(value) {
  return `ci-device:v1;pr=${value.pr};head=${value.head};base=${value.base};merge=${value.merge};target=${value.baseRef}`;
}
function docsOnly(files) {
  const documentation = (filename) =>
    !filename.split('/').includes('..') &&
    ((filename.startsWith('docs/') && /\.(md|png|jpg|svg)$/i.test(filename)) ||
      ['README.md', 'README_CN.md'].includes(filename));
  return files.length > 0 && files.every(({filename, previous_filename}) =>
    documentation(filename) && (!previous_filename || documentation(previous_filename)));
}
function matchingCPU(run, pr, defaultBranch) {
  const expected = `ci-meta:v1;run=${run.id};pr=${pr.number};base=${pr.base.ref};` +
    `base_sha=${pr.base.sha};merge_sha=${pr.merge_commit_sha}`;
  if (run.display_title !== expected) return false;
  if (run.event === 'pull_request') return run.head_sha === pr.head.sha &&
    run.head_repository?.id === pr.head.repo?.id;
  return run.event === 'workflow_dispatch' && run.head_branch === defaultBranch &&
    sameRepository(pr, run.head_repository?.id);
}

async function loadState(github, repo, defaultSha, now) {
  let response;
  try {
    response = await github.rest.repos.getContent({...repo, path: STATE_PATH, ref: STATE_BRANCH});
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.git.createRef({...repo, ref: `refs/heads/${STATE_BRANCH}`, sha: defaultSha});
    } catch (createError) {
      if (createError.status !== 422) throw createError;
      // Existing branch is fine; a missing/malformed state must not overwrite
      // unrelated content. getContent below remains the authoritative read.
    }
    try {
      response = await github.rest.repos.getContent({...repo, path: STATE_PATH, ref: STATE_BRANCH});
    } catch (readError) {
      if (readError.status !== 404) throw readError;
      response = {data: null};
    }
  }
  const state = response.data ? JSON.parse(Buffer.from(response.data.content, 'base64').toString()) :
    {version: 2, records: {}, published: {}, queuedAt: {}, retry: {}};
  const originalState = JSON.stringify(state);
  if (![1, 2].includes(state.version) || !state.records || !state.published || !state.queuedAt || !state.retry) {
    throw new Error('Invalid CI queue state; refusing to dispatch without durable reservations');
  }
  if (state.version === 1) {
    // Retire legacy head-wide successes, but keep all device reservations until
    // their actual runs finish. Old validation evidence cannot satisfy v2 gates.
    state.legacyHeads = [...new Set(Object.values(state.published)
      .filter((value) => typeof value === 'string')
      .map((value) => value.split('|')[0].split(':')[2])
      .filter((sha) => /^[a-f0-9]{40}$/.test(sha || '')))];
    state.published = {};
    state.version = 2;
  }
  state.snapshots ||= {};
  state.commentCursor ||= {};
  state.retestSince ||= now();
  state.cpuRequests ||= {};
  let sha = response.data?.sha;
  let saved = response.data ? originalState : null;
  return {state, save: async () => {
    if (saved === JSON.stringify(state)) return;
    const result = await github.rest.repos.createOrUpdateFileContents({...repo,
      path: STATE_PATH, branch: STATE_BRANCH, ...(sha ? {sha} : {}),
      message: 'ci: update device queue reservations',
      content: Buffer.from(JSON.stringify(state, null, 2) + '\n').toString('base64')});
    sha = result.data.content.sha;
    saved = JSON.stringify(state);
  }};
}

async function collectRetries({github, repo, context, pr, state, save}) {
  // GitHub comments are the durable inbox. Polling recovers a comment event
  // whose workflow was canceled while pending, and IDs make consumption idempotent.
  const comments = await github.paginate(github.rest.issues.listComments,
    {...repo, issue_number: pr.number, per_page: 100});
  for (const comment of comments.sort((a, b) => a.id - b.id)) {
    if (comment.id <= (state.commentCursor[pr.number] || 0)) continue;
    const direct = context.eventName === 'issue_comment' &&
      context.payload.comment?.id === comment.id;
    const recent = Date.parse(comment.created_at) >= Date.parse(state.retestSince);
    if (comment.body === '/retest' && (direct || recent)) {
      const {data: access} = await github.rest.repos.getCollaboratorPermissionLevel({...repo,
        username: comment.user.login});
      if (['write', 'maintain', 'admin'].includes(access.permission)) {
        const current = snapshot(pr);
        if (current) state.retry[pr.number] = current.head;
      } else if (direct) throw new Error('Retest requires write access');
    }
    state.commentCursor[pr.number] = comment.id;
  }
  await save();
}

async function runQueue({github, context, core, now = () => new Date().toISOString(),
  slots = 1, regressionBranches = ['decode_graph'], autoMerge = false}) {
  slots = Number(slots);
  if (!Number.isInteger(slots) || slots < 1 || slots > 8) throw new Error('CI_DEVICE_SLOTS must be 1..8');
  if (!Array.isArray(regressionBranches) || regressionBranches.some((branch) => typeof branch !== 'string')) {
    throw new Error('CI_REGRESSION_BRANCHES must be a JSON array of branch names');
  }
  const repo = context.repo;
  const {data: repository} = await github.rest.repos.get(repo);
  const defaultBranch = repository.default_branch;
  const {data: branch} = await github.rest.repos.getBranch({...repo, branch: defaultBranch});
  const store = await loadState(github, repo, branch.commit.sha, now);
  const {state, save} = store;
  const list = (method, args) => github.paginate(method, {...repo, per_page: 100, ...args});
  const runs = await list(github.rest.actions.listWorkflowRuns, {workflow_id: DEVICE_WORKFLOW});
  const cpuRuns = await list(github.rest.actions.listWorkflowRuns, {workflow_id: 'pre-commit.yml'});
  let previews = [];
  try {
    previews = await list(github.rest.actions.listWorkflowRuns,
      {workflow_id: 'device-validation-preview.yml'});
  } catch (error) {
    if (error.status !== 404) throw error; // Preview workflow may not be registered yet.
  }
  let abRuns = [];
  try {
    abRuns = await list(github.rest.actions.listWorkflowRuns,
      {workflow_id: 'beam-performance-ab.yml'});
  } catch (error) {
    if (error.status !== 404) throw error;
  }
  // Reconcile reserved dispatches from GitHub itself. Missing dispatch responses
  // remain reserved, even after a timeout, so another tick cannot overbook slots.
  for (const record of Object.values(state.records)) {
    if (record.status === 'complete') {
      const rerun = runs.find((run) => run.id === record.runId &&
        (run.run_attempt || 1) > (record.runAttempt || 1));
      if (!rerun) continue;
    }
    const matches = runs.filter((run) => run.event === 'workflow_dispatch' &&
      run.head_branch === defaultBranch && run.display_title === runName(record) &&
      (record.runId ? run.id === record.runId : run.id > (record.afterRunId || 0)) &&
      Date.parse(run.created_at) >= Date.parse(record.reservedAt) - 5000);
    const run = matches.sort((a, b) => b.id - a.id)[0];
    if (!run) continue;
    record.runId = run.id;
    record.runAttempt = run.run_attempt || 1;
    record.url = run.html_url;
    if (run.status === 'completed') {
      record.status = 'complete';
      record.conclusion = run.conclusion;
      record.completedAt = run.updated_at || now();
      if (run.conclusion === 'success') {
        const jobs = await list(github.rest.actions.listJobsForWorkflowRun, {run_id: run.id});
        const validated = jobs.some((job) => job.name === 'Continuous device validation' &&
          job.conclusion === 'success' && job.steps.some((step) =>
            step.name === TERMINAL_STEP && step.conclusion === 'success'));
        if (!validated) record.conclusion = 'failure';
      }
    } else record.status = 'running';
  }
  await save();

  const reservations = Object.values(state.records).filter((r) => r.status !== 'complete');
  const knownRuns = new Set(reservations.map((r) => r.runId).filter(Boolean));
  const unmanaged = [...runs, ...previews, ...abRuns].filter((run) => ACTIVE.has(run.status) && !knownRuns.has(run.id));
  // Manual and preview runs also own their PR, even when another slot is free
  // or the PR has a newer head. The shared workflow lock is the final backstop.
  const externallyActivePRs = new Map();
  for (const run of unmanaged) {
    const match = /^(?:ci-device:v1;pr=(\d+);|(?:Device PR preview|Beam V1 A\/B) #(\d+) @)/.exec(run.display_title || '');
    if (match) externallyActivePRs.set(Number(match[1] || match[2]), run);
    for (const pr of run.pull_requests || []) externallyActivePRs.set(pr.number, run);
  }

  const report = [];
  const desired = new Map();
  const statusRank = {success: 0, pending: 1, failure: 2};
  const propose = (sha, status, description, target_url) => {
    if (!/^[a-f0-9]{40}$/.test(sha || '')) return;
    const previous = desired.get(sha);
    // Even identical merge SHAs shared by multiple PRs must fail closed.
    if (!previous || statusRank[status] > statusRank[previous.status]) {
      desired.set(sha, {status, description, target_url});
    }
  };
  for (const sha of state.legacyHeads || []) {
    propose(sha, 'pending', 'Legacy head status retired; current merge validation is required');
  }
  const mergeCandidates = [];
  const publish = async (current, status, description, target_url) => {
    if (status === 'success') {
      const {data: fresh} = await github.rest.pulls.get({...repo, pull_number: current.pr});
      const actual = snapshot(fresh);
      if (!actual || keyOf(actual) !== keyOf(current)) {
        status = 'pending';
        description = 'PR changed during validation; waiting for a current merge snapshot';
      }
    }
    propose(current.merge, status, description, target_url);
    state.snapshots[current.pr] = current;
    report.push({pr: current.pr, priority: current.priority, status, description});
  };
  const candidates = [];
  let waitingCPU = false;
  const pulls = await list(github.rest.pulls.list, {state: 'open'});
  for (const summary of pulls) {
    if (!['main', 'decode_graph'].includes(summary.base.ref)) continue;
    const {data: pr} = await github.rest.pulls.get({...repo, pull_number: summary.number});
    await collectRetries({github, repo, context, pr, state, save});
    const current = snapshot(pr);
    const previous = state.snapshots[pr.number];
    if (previous && (!current || previous.merge !== current.merge)) {
      propose(previous.merge, 'pending', 'PR changed; a current ready merge snapshot is required');
    }
    if (!current) {
      // A previously green head must not remain green after losing its current
      // merge snapshot (e.g. a base update creates a conflict).
      propose(pr.merge_commit_sha, 'pending', 'Waiting for a current ready, mergeable PR snapshot');
      continue;
    }
    state.queuedAt[pr.number] ||= now();
    current.queuedAt = state.queuedAt[pr.number];
    const key = keyOf(current);
    const record = state.records[key];
    const retry = state.retry[pr.number] === current.head;
    const currentEvidence = record?.protocol === VALIDATION_PROTOCOL && sameRepository(pr, repository.id);
    if (currentEvidence && record.status === 'complete' && record.conclusion === 'success') {
      delete state.retry[pr.number];
      await publish(current, 'success', 'All required stages passed for the current merge snapshot', record.url);
      if (autoMerge) mergeCandidates.push(pr);
      continue;
    }
    if (currentEvidence && record.status === 'complete' && !retry) {
      await publish(current, 'failure', 'Device validation failed; /retest queues a stage-aware retry', record.url);
      continue;
    }
    const managedActive = reservations.some((r) => r.pr === pr.number);
    if (managedActive || externallyActivePRs.has(pr.number)) {
      // A preview cannot complete the production gate, so retain a requested
      // retry until that external run releases the PR.
      await publish(current, 'pending', 'Device validation reserved/running; it will not be preempted',
        record?.url || externallyActivePRs.get(pr.number)?.html_url);
      continue;
    }
    const cpu = cpuRuns.filter((run) => matchingCPU(run, pr, defaultBranch))
      .sort((a, b) => b.id - a.id)[0];
    let cpuPassed = false;
    if (cpu?.status === 'completed' && cpu.conclusion === 'success') {
      const jobs = await list(github.rest.actions.listJobsForWorkflowRun, {run_id: cpu.id});
      cpuPassed = jobs.some((job) => job.name === 'CPU checks' && job.conclusion === 'success' &&
        job.steps.some((step) => step.name === 'CPU contracts and scheduling tests' && step.conclusion === 'success'));
    }
    if (!cpuPassed) {
      if (!cpu && sameRepository(pr, repository.id) && !state.cpuRequests[key]) {
        // Refresh a stale merge on hosted runners. An ambiguous dispatch retains
        // its reservation, just like device dispatches, rather than duplicating it.
        state.cpuRequests[key] = {reservedAt: now(), pr: pr.number};
        await save();
        try {
          await github.rest.actions.createWorkflowDispatch({...repo, workflow_id: 'pre-commit.yml',
            ref: defaultBranch, inputs: {pr_number: String(pr.number), head_sha: current.head,
              base_sha: current.base, merge_sha: current.merge, base_ref: current.baseRef}});
        } catch (error) {
          if (error.status >= 400 && error.status < 500 && ![408, 429].includes(error.status)) {
            delete state.cpuRequests[key];
            await save();
          }
          throw error;
        }
      }
      if (retry && cpu?.status === 'completed' && ['failure', 'timed_out'].includes(cpu.conclusion)) {
        await github.rest.actions.reRunWorkflowFailedJobs({...repo, run_id: cpu.id});
        delete state.retry[pr.number];
        await save();
      }
      waitingCPU ||= !cpu || cpu.status !== 'completed' || retry;
      await publish(current, cpu?.status === 'completed' && !retry ? 'failure' : 'pending',
        'Waiting for successful CPU checks on this exact PR merge snapshot', cpu?.html_url);
      continue;
    }
    const files = await list(github.rest.pulls.listFiles, {pull_number: pr.number});
    if (files.length === pr.changed_files && docsOnly(files)) {
      delete state.retry[pr.number];
      await publish(current, 'success', 'Documentation-only change; CPU checks passed', cpu.html_url);
      if (autoMerge) mergeCandidates.push(pr);
      continue;
    }
    if (!sameRepository(pr, repository.id)) {
      await publish(current, 'failure', 'Fork device execution is disabled; use a reviewed branch in this repository');
      continue;
    }
    current.reuseSuccess = Boolean(retry && currentEvidence && record.status === 'complete');
    candidates.push(current);
    await publish(current, 'pending', `Queued: ${['high', 'normal', 'low'][current.priority]} priority`);
  }
  // Unready PRs must not retain an old success on the same commit.
  for (const pr of pulls.filter((p) => p.draft || !p.labels.some((l) => l.name === 'ready'))) {
    delete state.queuedAt[pr.number];
    delete state.retry[pr.number];
  }
  // Bound the state file without ever expiring a lease or an open PR's evidence.
  const openNumbers = new Set(pulls.map((pr) => pr.number));
  for (const [number, previous] of Object.entries(state.snapshots)) {
    if (!openNumbers.has(Number(number))) {
      propose(previous.merge, 'pending', 'PR closed; this validation is no longer active');
      delete state.snapshots[number];
    }
  }
  for (const field of ['queuedAt', 'retry', 'commentCursor']) {
    for (const number of Object.keys(state[field])) {
      if (!openNumbers.has(Number(number))) delete state[field][number];
    }
  }
  for (const [key, request] of Object.entries(state.cpuRequests)) {
    if (!openNumbers.has(request.pr)) delete state.cpuRequests[key];
  }
  for (const [sha, value] of desired) {
    const {status, description, target_url} = value;
    const signature = JSON.stringify(value);
    if (state.published[sha] !== signature) {
      await github.rest.repos.createCommitStatus({...repo, sha, context: CONTEXT,
        state: status, description: description.slice(0, 140), ...(target_url ? {target_url} : {})});
      state.published[sha] = signature;
    }
  }
  // Retired snapshots were marked pending above. Only active SHA signatures
  // need deduplication; retaining every past base update would grow state forever.
  for (const sha of Object.keys(state.published)) {
    if (!desired.has(sha)) delete state.published[sha];
  }
  delete state.legacyHeads;
  await save();
  for (const pr of mergeCandidates) {
    if (desired.get(pr.merge_commit_sha)?.status === 'success') {
      await enableAutoMerge({github, repo, pr, repository, core});
    }
  }
  const latestRegression = new Map();
  for (const record of Object.values(state.records).filter((r) => r.pr === 0)) {
    const previous = latestRegression.get(record.baseRef);
    if (!previous || previous.reservedAt < record.reservedAt) latestRegression.set(record.baseRef, record);
  }
  for (const [key, record] of Object.entries(state.records)) {
    if (record.status !== 'complete' || openNumbers.has(record.pr) ||
        latestRegression.get(record.baseRef) === record) continue;
    if (Date.parse(now()) - Date.parse(record.completedAt) > 7 * 24 * 60 * 60 * 1000) {
      delete state.records[key];
    }
  }
  await save();
  let available = Math.max(0, slots - reservations.length - unmanaged.length);
  const planned = order(candidates);
  // A regression is only admitted when there are no PRs waiting for CPU/device
  // admission and no active device work. It never interrupts an existing PR.
  if (!planned.length && !waitingCPU && !reservations.length && !unmanaged.length) {
    for (const baseRef of regressionBranches) {
      const {data: base} = await github.rest.repos.getBranch({...repo, branch: baseRef});
      const value = {pr: 0, head: base.commit.sha, base: base.commit.sha, merge: base.commit.sha,
        baseRef, priority: 3, queuedAt: now()};
      const previous = state.records[keyOf(value)];
      if (!previous || (previous.status === 'complete' && previous.protocol !== VALIDATION_PROTOCOL)) {
        planned.push(value);
      }
    }
  }
  for (const candidate of planned.slice(0, available)) {
    const key = keyOf(candidate);
    state.records[key] = {...candidate, protocol: VALIDATION_PROTOCOL, status: 'reserved', reservedAt: now(),
      afterRunId: Math.max(0, ...runs.map((run) => run.id))};
    delete state.retry[candidate.pr];
    // Durable reservation BEFORE network mutation; save conflicts fail closed.
    await save();
    try {
      await github.rest.actions.createWorkflowDispatch({...repo, workflow_id: DEVICE_WORKFLOW,
        ref: defaultBranch, inputs: {pr_number: String(candidate.pr), head_sha: candidate.head,
          base_sha: candidate.base, merge_sha: candidate.merge, base_ref: candidate.baseRef,
          priority: ['high', 'normal', 'low', 'regression'][candidate.priority],
          reuse_success: String(Boolean(candidate.reuseSuccess))}});
      core.info(`Reserved device slot for ${candidate.pr ? `PR #${candidate.pr}` : candidate.baseRef}`);
    } catch (error) {
      if (error.status >= 400 && error.status < 500 && ![408, 429].includes(error.status)) {
        state.records[key].status = 'complete';
        state.records[key].conclusion = 'dispatch_failed';
        state.records[key].completedAt = now();
        await save();
      }
      // Timeout/5xx/unknown outcome keeps the reservation. Do not silently
      // resubmit a possibly accepted run or start another PR in its slot.
      throw error;
    }
  }
  const held = Object.values(state.records).filter((r) => r.status === 'reserved');
  for (const record of held) {
    if (Date.parse(now()) - Date.parse(record.reservedAt) > 15 * 60 * 1000) {
      core.warning(`Unconfirmed dispatch for ${keyOf(record)}; reservation retained for operator review`);
    }
  }
  const activeCount = Object.values(state.records).filter((r) => r.status !== 'complete').length;
  await core.summary.addHeading('CI device queue').addTable([
    ['PR', 'Priority', 'State', 'Reason'],
    ...report.map((r) => [String(r.pr), ['high', 'normal', 'low'][r.priority], r.status, r.description]),
  ]).addRaw(`\nSlots: ${slots}; active reservations: ${activeCount}; no running task was cancelled.\n`).write();
  return {state, report};
}

async function enableAutoMerge({github, repo, pr, repository, core}) {
  // Native auto-merge remains opt-in and relies on strict required checks and
  // stale-review dismissal. Never implement an unprotected direct merge.
  if (!repository.allow_auto_merge || !pr.labels.some((l) => l.name === 'automerge')) return;
  let protection;
  try {
    ({data: protection} = await github.rest.repos.getBranchProtection({...repo, branch: pr.base.ref}));
  } catch (error) {
    if (![403, 404].includes(error.status)) throw error;
    core.warning(`PR #${pr.number}: cannot verify branch protection; auto-merge remains disabled`);
    return;
  }
  const checks = protection.required_status_checks;
  const reviews = protection.required_pull_request_reviews;
  const names = [...(checks?.contexts || []), ...(checks?.checks || []).map((check) => check.context)];
  if (!checks?.strict || !names.includes(CONTEXT) || !reviews?.dismiss_stale_reviews ||
      reviews.required_approving_review_count < 1) {
    core.warning(`PR #${pr.number}: auto-merge requires strict checks and current reviews`);
    return;
  }
  const {data: latest} = await github.rest.pulls.get({...repo, pull_number: pr.number});
  if (latest.head.sha !== pr.head.sha || latest.base.sha !== pr.base.sha || latest.draft ||
      latest.state !== 'open' || !latest.labels.some((l) => l.name === 'ready') ||
      !latest.labels.some((l) => l.name === 'automerge') || latest.auto_merge) return;
  await github.graphql(`mutation($id: ID!) {
    enablePullRequestAutoMerge(input: {pullRequestId: $id, mergeMethod: SQUASH}) {
      pullRequest { number }
    }
  }`, {id: latest.node_id});
}
module.exports = {runQueue, priority, order, snapshot, keyOf, runName, docsOnly, matchingCPU, sameRepository, enableAutoMerge};
