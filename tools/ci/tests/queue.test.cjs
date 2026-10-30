// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
const test = require('node:test');
const assert = require('node:assert/strict');
const {runQueue, priority, snapshot, keyOf, runName, docsOnly} = require('../queue.cjs');
const sha = (n) => n.toString(16).padStart(40, '0');
const time = '2026-10-22T08:00:00.000Z';
function pr(number, labels = ['ready']) {
  return {number, node_id: `PR_${number}`, state: 'open', draft: false, mergeable: true, changed_files: 1,
    head: {sha: sha(number), repo: {id: 1}}, base: {sha: sha(100), ref: 'decode_graph', repo: {id: 1}},
    merge_commit_sha: sha(1000 + number), labels: labels.map((name) => ({name}))};
}
function cpuRun(p, id = p.number) {
  return {id, event: 'pull_request', head_sha: p.head.sha,
    head_repository: {id: p.head.repo.id},
    display_title: `ci-meta:v1;run=${id};pr=${p.number};base=${p.base.ref};base_sha=${p.base.sha};merge_sha=${p.merge_commit_sha}`,
    status: 'completed', conclusion: 'success', html_url: 'https://github.com/cpu'};
}
function fixture(pulls = [pr(1)]) {
  const f = {pulls, runs: [], cpu: [], files: {}, calls: [], persisted: null, clock: time,
    permission: 'write', branches: {main: sha(99), decode_graph: sha(100)},
    comments: {}, nextComment: 1, repository: {id: 1, default_branch: 'main', allow_auto_merge: false}, previews: [], abRuns: []};
  f.cpu = pulls.map(p => cpuRun(p));
  f.refreshCPU = (number = 1) => {
    const p = f.pulls.find(value => value.number === number);
    f.cpu[f.cpu.findIndex(run => run.id === number)] = cpuRun(p);
  };
  const method = (name, fn) => Object.assign(async (args) => {
    f.calls.push({name, args});
    return fn(args);
  }, {methodName: name});
  const error = (status) => Object.assign(new Error(`HTTP ${status}`), {status});
  f.github = {rest: {
    repos: {
      get: method('repo', () => ({data: f.repository})),
      getBranch: method('branch', ({branch}) => ({data: {commit: {sha: f.branches[branch]}}})),
      getContent: method('read', () => {
        if (!f.persisted) throw error(404);
        return {data: {sha: 'state-blob', content: Buffer.from(JSON.stringify(f.persisted)).toString('base64')}};
      }),
      createOrUpdateFileContents: method('save', ({content}) => {
        if (f.failSave) throw error(409);
        f.persisted = JSON.parse(Buffer.from(content, 'base64').toString());
        return {data: {content: {sha: 'state-blob'}}};
      }),
      createCommitStatus: method('status', () => ({})),
      getCollaboratorPermissionLevel: method('permission', () => ({data: {permission: f.permission}})),
      getBranchProtection: method('protection', () => {
        if (f.protectionError) throw error(f.protectionError);
        return {data: f.protection};
      }),
    },
    issues: {listComments: method('comments', ({issue_number}) => f.comments[issue_number] || [])},
    git: {createRef: method('createRef', () => ({}))},
    pulls: {
      list: method('pulls', () => f.pulls),
      get: method('pull', ({pull_number}) => ({data: structuredClone(
        f.getPull ? f.getPull(pull_number) : f.pulls.find((p) => p.number === pull_number))})),
      listFiles: method('files', ({pull_number}) => f.files[pull_number] || [{filename: 'vllm_gr/engine.py'}]),
    },
    actions: {
      listWorkflowRuns: method('runs', ({workflow_id, head_sha}) => workflow_id === 'pre-commit.yml' ?
        f.cpu.filter((run) => !head_sha || run.head_sha === head_sha) :
        workflow_id === 'device-validation-preview.yml' ? f.previews :
        workflow_id === 'beam-performance-ab.yml' ? f.abRuns : f.runs),
      listJobsForWorkflowRun: method('jobs', ({run_id}) => run_id < 100 ? (f.cpuJobs || [{
        name: 'CPU checks', conclusion: 'success', steps: [{name: 'CPU contracts and scheduling tests',
          conclusion: 'success'}]}]) : f.jobs || [{name: 'Continuous device validation',
        conclusion: 'success', steps: [{name: 'Complete device validation', conclusion: 'success'}]}]),
      createWorkflowDispatch: method('dispatch', ({workflow_id}) => {
        if (workflow_id === 'pre-commit.yml') f.calls.at(-1).name = 'dispatchCPU';
        if (f.dispatchError) throw error(f.dispatchError);
        return {};
      }),
      reRunWorkflowFailedJobs: method('rerunCPU', () => {
        if (f.rerunError) throw error(f.rerunError);
        return {};
      }),
    },
  }, graphql: method('automerge', () => ({})), paginate: (fn, args) => fn(args)};
  f.context = {repo: {owner: 'test', repo: 'repo'}, eventName: 'schedule', payload: {}};
  f.core = {info: () => {}, warning: (message) => f.calls.push({name: 'warning', message}),
    summary: {addHeading() { return this; }, addTable() { return this; },
      addRaw() { return this; }, async write() {}}};
  f.tick = (args = {}) => runQueue({github: f.github, context: f.context, core: f.core,
    now: () => f.clock, regressionBranches: [], ...args});
  f.of = (name) => f.calls.filter((c) => c.name === name);
  f.clear = () => { f.calls = []; };
  f.retest = (number = 1) => {
    const comment = {id: f.nextComment++, created_at: f.clock, body: '/retest', user: {login: 'maintainer'}};
    (f.comments[number] ||= []).push(comment);
    f.context = {...f.context, eventName: 'issue_comment', payload: {
      issue: {number, pull_request: {}}, comment}};
  };
  f.run = (p = pulls[0], values = {}) => ({id: 100, event: 'workflow_dispatch',
    head_branch: 'main', display_title: runName(snapshot(p)), created_at: f.clock,
    status: 'completed', conclusion: 'success', html_url: 'https://github.com/device', ...values});
  return f;
}

test('forks never enter trusted CPU refresh or persistent device dispatch', async () => {
  for (const hasCPU of [true, false]) {
    const p = pr(1); p.head.repo.id = 2;
    const f = fixture([p]);
    if (!hasCPU) f.cpu = [];
    await f.tick();
    f.retest(); await f.tick();
    assert.equal(f.of('dispatch').length, 0);
    assert.equal(f.of('dispatchCPU').length, 0);
    assert.ok(f.of('status').every(c => c.args.state !== 'success'));
    if (hasCPU) assert.match(f.of('status').at(-1).args.description, /Fork device execution is disabled/);
  }
  const p = pr(1); p.head.repo.id = 2;
  const docs = fixture([p]); docs.files[1] = [{filename: 'docs/help.md'}];
  await docs.tick();
  assert.equal(docs.of('status').at(-1).args.state, 'success');
  assert.equal(docs.of('dispatch').length, 0);
});
test('same head on distinct base branches has independent merge gates', async () => {
  const a = pr(1); const b = pr(2); b.head.sha = a.head.sha;
  b.base = {...b.base, ref: 'main', sha: sha(99)};
  const f = fixture([a, b]); f.files[1] = [{filename: 'README.md'}];
  f.cpu[1].conclusion = 'failure';
  await f.tick();
  assert.deepEqual(f.of('status').map(c => [c.args.sha, c.args.state]),
    [[a.merge_commit_sha, 'success'], [b.merge_commit_sha, 'failure']]);
  assert.ok(f.of('status').every(c => c.args.sha !== a.head.sha));
  f.clear(); await f.tick();
  assert.equal(f.of('status').length, 0);
});
test('identical merge SHAs are aggregated fail-closed independent of PR order', async () => {
  for (const reverse of [false, true]) {
    const a = pr(1); const b = pr(2);
    b.head.sha = a.head.sha; b.merge_commit_sha = a.merge_commit_sha;
    const f = fixture(reverse ? [b, a] : [a, b]);
    f.files[1] = [{filename: 'README.md'}]; f.files[2] = [{filename: 'README.md'}];
    f.cpu.find(c => c.id === 2).conclusion = 'failure';
    await f.tick();
    assert.deepEqual(f.of('status').map(c => c.args.state), ['failure']);
    f.clear(); await f.tick();
    assert.equal(f.of('status').length, 0);
    f.pulls.find(p => p.number === 2).labels = [];
    await f.tick();
    assert.deepEqual(f.of('status').map(c => c.args.state), ['pending']);
  }
});
test('base advancement retires docs success and refreshes exactly one hosted snapshot', async () => {
  const f = fixture(); f.files[1] = [{filename: 'README.md'}];
  await f.tick(); f.clear();
  const p = f.pulls[0]; p.base.sha = sha(101); p.merge_commit_sha = sha(2001);
  await f.tick();
  assert.deepEqual(f.of('status').map(c => [c.args.sha, c.args.state]),
    [[sha(1001), 'pending'], [sha(2001), 'pending']]);
  assert.equal(f.of('dispatch').length, 0);
  assert.deepEqual(f.of('dispatchCPU')[0].args.inputs,
    {pr_number: '1', head_sha: p.head.sha, base_sha: p.base.sha, merge_sha: p.merge_commit_sha, base_ref: p.base.ref});
  f.clear(); await f.tick(); assert.equal(f.of('dispatchCPU').length, 0);
  assert.equal(f.persisted.published[sha(1001)], undefined);
  f.cpu.push({...cpuRun(p, 50), event: 'workflow_dispatch', head_branch: 'main', head_sha: sha(99)});
  await f.tick();
  assert.equal(f.of('status').at(-1).args.sha, sha(2001));
  assert.equal(f.of('status').at(-1).args.state, 'success');
});
test('untrusted dispatch and incomplete CPU metadata cannot approve docs', async () => {
  for (const update of [
    {display_title: 'ci-meta:v1;run=1;pr=1;base=decode_graph'},
    {event: 'workflow_dispatch', head_branch: 'feature'},
    {event: 'workflow_dispatch', head_branch: 'main', head_repository: {id: 2}},
  ]) {
    const f = fixture(); f.files[1] = [{filename: 'README.md'}];
    Object.assign(f.cpu[0], update); await f.tick();
    assert.equal(f.of('status').at(-1).args.state, 'pending');
    assert.equal(f.of('dispatchCPU').length, 1);
  }
});
test('ambiguous CPU refresh is durably reserved across ticks', async () => {
  const f = fixture(); f.cpu = []; f.dispatchError = 504;
  await assert.rejects(f.tick(), /504/);
  assert.equal(Object.keys(f.persisted.cpuRequests).length, 1);
  f.dispatchError = null; f.clear(); await f.tick();
  assert.equal(f.of('dispatchCPU').length, 0);
});
test('scheduled polling recovers missed retest events and consumes each comment once', async () => {
  const f = fixture([pr(1), pr(2)]);
  await f.tick({slots: 2});
  f.runs = [f.run(f.pulls[0], {id: 100, conclusion: 'failure'}),
    f.run(f.pulls[1], {id: 101, conclusion: 'failure'})];
  await f.tick({slots: 2}); f.clear();
  f.retest(1); f.retest(2); f.context.eventName = 'schedule';
  await f.tick({slots: 2});
  assert.deepEqual(f.of('dispatch').map(c => c.args.inputs.pr_number), ['1', '2']);
  assert.ok(f.of('dispatch').every(c => c.args.inputs.reuse_success === 'true'));
  assert.deepEqual(f.persisted.commentCursor, {1: 1, 2: 2});
  f.runs.push(f.run(f.pulls[0], {id: 102, conclusion: 'failure'}),
    f.run(f.pulls[1], {id: 103, conclusion: 'failure'}));
  f.clear(); await f.tick({slots: 2});
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(f.of('permission').length, 0);
});
test('retest arriving during managed validation survives until a failed run ends', async () => {
  const f = fixture(); await f.tick();
  f.runs = [f.run(undefined, {status: 'in_progress', conclusion: null})];
  f.retest(); f.context.eventName = 'schedule'; f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(f.persisted.retry[1], sha(1));
  f.runs[0].status = 'completed'; f.runs[0].conclusion = 'failure';
  f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 1);
  assert.equal(f.of('dispatch')[0].args.inputs.reuse_success, 'true');
});
test('polling ignores unauthorized and pre-deployment comments', async () => {
  for (const historical of [false, true]) {
    const f = fixture(); f.cpu[0].conclusion = 'failure';
    f.retest(); f.context.eventName = 'schedule';
    if (historical) f.comments[1][0].created_at = '2026-01-01T00:00:00Z';
    else f.permission = 'read';
    await f.tick();
    assert.equal(f.of('rerunCPU').length, 0);
    assert.equal(f.persisted.commentCursor[1], 1);
    assert.equal(f.persisted.retry[1], undefined);
  }
});
test('a failed comment checkpoint does not consume the durable inbox', async () => {
  const f = fixture(); f.cpu[0].conclusion = 'failure'; await f.tick();
  f.retest(); f.context.eventName = 'schedule'; f.failSave = true;
  await assert.rejects(f.tick(), /409/);
  assert.equal(f.persisted.commentCursor[1], undefined);
  assert.equal(f.of('rerunCPU').length, 0);
  f.failSave = false; await f.tick();
  assert.equal(f.of('rerunCPU').length, 1);
});
test('CPU retry rejection leaves the accepted request pending', async () => {
  const f = fixture(); f.cpu[0].conclusion = 'failure'; await f.tick();
  f.retest(); f.rerunError = 503;
  await assert.rejects(f.tick(), /503/);
  assert.equal(f.persisted.retry[1], sha(1));
  f.rerunError = null; f.clear(); f.context.eventName = 'schedule'; await f.tick();
  assert.equal(f.of('rerunCPU').length, 1);
  assert.equal(f.persisted.retry[1], undefined);
});
test('v1 migration retires head successes and does not trust old completed evidence', async () => {
  const f = fixture(); const s = snapshot(f.pulls[0]);
  f.persisted = {version: 1, published: {1: `${keyOf(s)}|success|old`}, queuedAt: {}, retry: {},
    records: {[keyOf(s)]: {...s, status: 'complete', conclusion: 'success', reservedAt: time}}};
  await f.tick();
  assert.equal(f.persisted.version, 2);
  assert.ok(f.of('status').some(c => c.args.sha === sha(1) && c.args.state === 'pending'));
  assert.ok(f.of('status').every(c => c.args.state !== 'success'));
  assert.equal(f.of('dispatch').length, 1);
  assert.equal(f.of('dispatch')[0].args.inputs.reuse_success, 'false');
});
test('v1 migration preserves in-flight device reservations', async () => {
  const f = fixture(); const s = snapshot(f.pulls[0]);
  f.persisted = {version: 1, published: {}, queuedAt: {}, retry: {},
    records: {[keyOf(s)]: {...s, status: 'reserved', reservedAt: time}}};
  await f.tick();
  assert.equal(f.persisted.version, 2);
  assert.equal(Object.values(f.persisted.records)[0].status, 'reserved');
  assert.equal(f.of('dispatch').length, 0);
});

test('high priority wins; FIFO within priority; slots are bounded', async () => {
  const f = fixture([pr(1), pr(2, ['ready', 'ci-priority:high']), pr(3, ['ready', 'ci-priority:high'])]);
  await f.tick({slots: 2});
  assert.deepEqual(f.of('dispatch').map((c) => c.args.inputs.pr_number), ['2', '3']);
  assert.equal(Object.values(f.persisted.records).length, 2);
  assert.ok(f.of('dispatch').every((c) => c.args.ref === 'main'));
});
test('A/B label run holds its PR and a slot without satisfying the full gate', async () => {
  const f = fixture([pr(1), pr(2)]);
  f.abRuns = [{id: 150, status: 'in_progress', display_title: `Beam V1 A/B #1 @${sha(1001)}`,
    html_url: 'https://github.com/ab'}];
  await f.tick({slots: 2});
  assert.deepEqual(f.of('dispatch').map((c) => c.args.inputs.pr_number), ['2']);
  assert.ok(f.of('status').some((c) => c.args.sha === sha(1001) && c.args.state === 'pending'));
  f.abRuns[0].status = 'completed';
  f.abRuns[0].conclusion = 'success';
  f.clear();
  await f.tick({slots: 2});
  assert.deepEqual(f.of('dispatch').map((c) => c.args.inputs.pr_number), ['1']);
  assert.ok(!f.of('status').some((c) => c.args.sha === sha(1001) && c.args.state === 'success'));
});
test('older eligible default-low PR precedes a newer one', async () => {
  const f = fixture([pr(2), pr(1)]);
  f.persisted = {version: 1, records: {}, published: {}, retry: {},
    queuedAt: {2: '2026-10-22T07:00:00.000Z', 1: time}};
  await f.tick();
  assert.equal(f.of('dispatch')[0].args.inputs.pr_number, '2');
});
test('ordinary PRs default to low; explicit normal and high promote waiting PRs', async () => {
  assert.equal(priority([]), 2);
  assert.equal(priority(['ready']), 2);
  assert.equal(priority(['ready', 'ci-priority:low']), 2);
  assert.equal(priority(['ci-priority:normal', 'ci-priority:low']), 1);
  assert.equal(priority(['ci-priority:high', 'ci-priority:normal']), 0);
  const f = fixture([pr(1), pr(2, ['ready', 'ci-priority:normal'])]);
  await f.tick();
  assert.equal(f.of('dispatch')[0].args.inputs.pr_number, '2');
  assert.equal(f.of('dispatch')[0].args.inputs.priority, 'normal');
  assert.ok(f.of('status').some((c) => c.args.sha === sha(1001) && c.args.description === 'Queued: low priority'));
});
test('high priority never preempts active work, even after head changes', async () => {
  const f = fixture([pr(1), pr(2)]);
  await f.tick();
  f.runs = [f.run(pr(1), {status: 'in_progress', conclusion: null})];
  f.pulls[0].head.sha = sha(5);
  f.pulls[1].labels.push({name: 'ci-priority:high'});
  f.clear();
  await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(Object.values(f.persisted.records)[0].status, 'running');
});
test('complete validated run releases slot to highest priority waiter', async () => {
  const f = fixture([pr(1), pr(2)]);
  await f.tick();
  f.runs = [f.run()];
  f.clear();
  await f.tick();
  assert.equal(f.of('dispatch')[0].args.inputs.pr_number, '2');
  assert.ok(f.of('status').some((c) => c.args.sha === sha(1001) && c.args.state === 'success'));
});
test('success without terminal device proof is failure', async () => {
  const f = fixture();
  await f.tick();
  f.runs = [f.run()]; f.jobs = [];
  f.clear(); await f.tick();
  assert.equal(f.of('status').at(-1).args.state, 'failure');
  assert.equal(f.of('dispatch').length, 0);
});
test('failed stage requires explicit retry and enables same-snapshot reuse', async () => {
  const f = fixture();
  await f.tick(); f.runs = [f.run(undefined, {conclusion: 'failure'})];
  f.clear(); await f.tick(); assert.equal(f.of('dispatch').length, 0);
  f.retest(); await f.tick();
  assert.equal(f.of('dispatch')[0].args.inputs.reuse_success, 'true');
  f.context.eventName = 'schedule'; f.clear();
  // The retry is not visible in the API yet. Do not reconcile it with the previous run.
  await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(Object.values(f.persisted.records)[0].status, 'reserved');
  f.runs.push(f.run(undefined, {id: 101}));
  await f.tick();
  assert.equal(Object.values(f.persisted.records)[0].runId, 101);
});
test('changed head or base never reuses old successful stages', async () => {
  for (const field of ['head', 'base']) {
    const f = fixture(); await f.tick(); f.runs = [f.run()];
    f.pulls[0][field].sha = sha(77); f.pulls[0].merge_commit_sha = sha(88);
    f.refreshCPU();
    f.clear(); await f.tick();
    assert.equal(f.of('dispatch')[0].args.inputs.reuse_success, 'false');
    assert.ok(f.of('status').every((c) => c.args.state !== 'success'));
  }
});
test('removing ready invalidates previous success without cancelling running work', async () => {
  const f = fixture(); await f.tick(); f.pulls[0].labels = [];
  f.clear(); await f.tick();
  assert.equal(f.of('status')[0].args.state, 'pending');
  assert.equal(Object.values(f.persisted.records)[0].status, 'reserved');
});
test('CPU failure blocks device admission; retry re-runs only failed CPU jobs', async () => {
  const f = fixture(); f.cpu[0].conclusion = 'failure';
  await f.tick(); assert.equal(f.of('dispatch').length, 0);
  f.retest(); await f.tick();
  assert.equal(f.of('rerunCPU').length, 1);
});
test('CPU result from another repository or PR cannot satisfy the gate', async () => {
  for (const update of [{head_repository: {id: 2}}, {display_title: 'ci-meta:v1;pr=9;'}]) {
    const f = fixture(); Object.assign(f.cpu[0], update);
    await f.tick(); assert.equal(f.of('dispatch').length, 0);
  }
});
test('documentation bypass is conservative, and rename sources require device coverage', async () => {
  assert.equal(docsOnly([]), false);
  assert.equal(docsOnly([{filename: '.github/workflows/ci.yml'}]), false);
  assert.equal(docsOnly([{filename: 'docs/../setup.py'}]), false);
  assert.equal(docsOnly([{filename: 'docs/engine.md', previous_filename: 'vllm_gr/engine.py'}]), false);
  const f = fixture(); f.files[1] = [{filename: 'docs/help.md'}];
  await f.tick(); assert.equal(f.of('dispatch').length, 0);
  assert.equal(f.of('status').at(-1).args.state, 'success');
});
test('truncated file inventory and skipped CPU jobs cannot bypass device checks', async () => {
  const f = fixture(); f.files[1] = [{filename: 'README.md'}]; f.pulls[0].changed_files = 3001;
  await f.tick(); assert.equal(f.of('dispatch').length, 1);
  const skipped = fixture(); skipped.cpuJobs = [];
  await skipped.tick(); assert.equal(skipped.of('dispatch').length, 0);
});
test('timeout after dispatch retains durable slot; later tick does not duplicate', async () => {
  const f = fixture(); f.dispatchError = 504;
  await assert.rejects(f.tick(), /504/);
  assert.equal(Object.values(f.persisted.records)[0].status, 'reserved');
  f.dispatchError = null; f.clear(); f.clock = '2026-10-22T08:20:00.000Z';
  await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(f.of('warning').length, 1);
});
test('definitive dispatch rejection releases capacity but requires retry', async () => {
  const f = fixture(); f.dispatchError = 422;
  await assert.rejects(f.tick(), /422/);
  assert.equal(Object.values(f.persisted.records)[0].status, 'complete');
  f.dispatchError = null; f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 0);
});
test('state write conflict prevents dispatch', async () => {
  const f = fixture(); f.failSave = true;
  await assert.rejects(f.tick(), /409/);
  assert.equal(f.of('dispatch').length, 0);
});
test('unmanaged active device runs consume capacity', async () => {
  const f = fixture(); f.runs = [f.run(undefined, {display_title: 'manual run', status: 'queued'})];
  await f.tick(); assert.equal(f.of('dispatch').length, 0);
});
test('regressions coalesce to latest branch head and remain lower priority than PRs', async () => {
  const f = fixture([]); await f.tick({regressionBranches: ['decode_graph']});
  assert.equal(f.of('dispatch')[0].args.inputs.priority, 'regression');
  const old = Object.values(f.persisted.records)[0];
  f.runs = [{...f.run(pr(1)), display_title: runName(old)}];
  f.branches.decode_graph = sha(200); f.clear();
  await f.tick({regressionBranches: ['decode_graph']});
  assert.equal(f.of('dispatch')[0].args.inputs.head_sha, sha(200));
  const busy = fixture(); busy.cpu[0].status = 'in_progress';
  await busy.tick({regressionBranches: ['decode_graph']});
  assert.equal(busy.of('dispatch').length, 0);
});
test('unauthorized retest is refused', async () => {
  const f = fixture(); f.retest(); f.permission = 'read';
  await assert.rejects(f.tick(), /write access/);
  assert.equal(f.of('dispatch').length, 0);
});
test('same head moving base during publication cannot turn green', async () => {
  const f = fixture(); await f.tick(); f.runs = [f.run()]; f.clear();
  let reads = 0;
  f.getPull = () => { const value = structuredClone(f.pulls[0]);
    if (++reads > 1) value.base.sha = sha(333);
    return value; };
  await f.tick();
  assert.ok(f.of('status').every((c) => c.args.state !== 'success'));
});
test('native auto-merge requires opt-in, current review protection and current head', async () => {
  const f = fixture([pr(1, ['ready', 'automerge'])]);
  f.repository.allow_auto_merge = true;
  f.files[1] = [{filename: 'README.md'}];
  f.protection = {required_status_checks: {strict: true, contexts: ['Benchmark blocking gate']},
    required_pull_request_reviews: {dismiss_stale_reviews: true, required_approving_review_count: 1}};
  await f.tick(); assert.equal(f.of('automerge').length, 0);
  await f.tick({autoMerge: true}); assert.equal(f.of('automerge').length, 1);
  f.clear(); f.protection.required_status_checks.strict = false;
  await f.tick({autoMerge: true}); assert.equal(f.of('automerge').length, 0);
  f.protectionError = 403;
  await f.tick({autoMerge: true}); assert.equal(f.of('automerge').length, 0);
});
test('invalid slot configuration fails before mutation', async () => {
  const f = fixture();
  for (const slots of [0, 9, 'invalid']) await assert.rejects(f.tick({slots}), /1..8/);
  assert.equal(f.calls.length, 0);
});
test('retention removes closed completed work but never expires an active lease', async () => {
  const f = fixture([]);
  const completed = {...snapshot(pr(2)), status: 'complete', conclusion: 'success',
    reservedAt: '2026-10-01T08:00:00Z', completedAt: '2026-10-01T09:00:00Z'};
  const active = {...snapshot(pr(3)), status: 'reserved', reservedAt: '2026-10-01T08:00:00Z'};
  f.persisted = {version: 1, records: {[keyOf(completed)]: completed, [keyOf(active)]: active},
    published: {2: 'old'}, queuedAt: {2: time}, retry: {2: sha(2)}};
  await f.tick();
  assert.deepEqual(Object.keys(f.persisted.records), [keyOf(active)]);
  assert.deepEqual(f.persisted.published, {});
  assert.equal(f.of('dispatch').length, 0);
});
test('operator rerun of a known run invalidates the prior completion', async () => {
  const f = fixture(); await f.tick(); f.runs = [f.run()]; await f.tick();
  f.clear(); f.runs[0].run_attempt = 2; f.runs[0].status = 'in_progress';
  f.runs[0].conclusion = null;
  await f.tick();
  assert.equal(f.of('status').at(-1).args.state, 'pending');
  assert.equal(f.of('dispatch').length, 0);
  f.runs[0].status = 'completed'; f.runs[0].conclusion = 'failure';
  await f.tick();
  assert.equal(f.of('status').at(-1).args.state, 'failure');
});
test('bootstrap preview occupies a slot without producing a production gate result', async () => {
  const f = fixture();
  f.previews = [{id: 501, status: 'in_progress', event: 'pull_request'}];
  await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.ok(f.of('status').every((c) => c.args.state !== 'success'));
});

test('an external run blocks the same PR across head changes even with free slots', async () => {
  for (const kind of ['preview', 'manual']) {
    for (const status of ['queued', 'pending', 'in_progress']) {
      const f = fixture([pr(1, ['ready', 'ci-priority:high']), pr(2)]);
      const original = structuredClone(f.pulls[0]);
      const run = f.run(original, {id: 501, status, conclusion: null});
      if (kind === 'preview') {
        Object.assign(run, {event: 'pull_request',
          display_title: `Device PR preview #1 @${original.head.sha}`});
        f.previews = [run];
      } else f.runs = [run];
      f.pulls[0].head.sha = sha(51);
      f.pulls[0].merge_commit_sha = sha(1051);
      f.refreshCPU();
      await f.tick({slots: 2});
      assert.deepEqual(f.of('dispatch').map((c) => c.args.inputs.pr_number), ['2'], `${kind}: ${status}`);
      assert.ok(f.of('status').some((c) => c.args.sha === sha(1051) &&
        c.args.description.includes('will not be preempted')));
      f.runs = [f.run(pr(2), {id: 502, status: 'in_progress', conclusion: null})];
      f.previews = [];
      f.clear(); await f.tick({slots: 2});
      assert.deepEqual(f.of('dispatch').map((c) => c.args.inputs.pr_number), ['1']);
    }
  }
});

test('changing waiting priorities preserves the running PR and admits the current highest label', async () => {
  const f = fixture([pr(1), pr(2), pr(3)]);
  await f.tick();
  const admitted = structuredClone(Object.values(f.persisted.records)[0]);
  f.runs = [f.run(pr(1), {status: 'in_progress', conclusion: null})];
  f.pulls[0].labels.push({name: 'ci-priority:high'});
  f.pulls[1].labels.push({name: 'ci-priority:high'});
  f.pulls[2].labels.push({name: 'ci-priority:normal'});
  f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(Object.values(f.persisted.records)[0].priority, admitted.priority);
  f.pulls[1].labels = [{name: 'ready'}];
  f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  f.runs[0].status = 'completed'; f.runs[0].conclusion = 'success';
  f.clear(); await f.tick();
  assert.deepEqual(f.of('dispatch').map((c) => [c.args.inputs.pr_number, c.args.inputs.priority]),
    [['3', 'normal']]);
  f.runs.push(f.run(pr(3), {id: 101}));
  f.clear(); await f.tick();
  assert.deepEqual(f.of('dispatch').map((c) => [c.args.inputs.pr_number, c.args.inputs.priority]),
    [['2', 'low']]);
});

test('a newly promoted high-priority PR goes next without interrupting the active low PR', async () => {
  const f = fixture([pr(1), pr(2), pr(3)]);
  await f.tick();
  f.runs = [f.run(pr(1), {status: 'in_progress', conclusion: null})];
  f.pulls[2].labels.push({name: 'ci-priority:high'});
  f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(Object.values(f.persisted.records)[0].status, 'running');
  f.runs[0].status = 'completed'; f.runs[0].conclusion = 'success';
  f.clear(); await f.tick();
  assert.deepEqual(f.of('dispatch').map((c) => [c.args.inputs.pr_number, c.args.inputs.priority]),
    [['3', 'high']]);
});

test('an explicit failed-run retry waits for a preview without losing the retry request', async () => {
  const f = fixture();
  await f.tick(); f.runs = [f.run(undefined, {conclusion: 'failure'})];
  await f.tick();
  f.previews = [{id: 501, status: 'in_progress', event: 'pull_request',
    display_title: `Device PR preview #1 @${f.pulls[0].head.sha}`}];
  f.retest(); f.clear(); await f.tick({slots: 2});
  assert.equal(f.of('dispatch').length, 0);
  assert.equal(f.persisted.retry[1], f.pulls[0].head.sha);
  f.context.eventName = 'schedule'; f.previews = [];
  f.clear(); await f.tick();
  assert.equal(f.of('dispatch').length, 1);
  assert.equal(f.of('dispatch')[0].args.inputs.reuse_success, 'true');
});
