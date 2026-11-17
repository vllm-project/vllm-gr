// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {classify} = require('../tools/ci/docs_only');

const head = 'a'.repeat(40);
const base = 'b'.repeat(40);
const merge = 'c'.repeat(40);

async function classifyFiles(files, {labels = ['ready'], action = 'synchronize',
  eventName = 'pull_request', eventLabel = null, draft = false, snapshots} = {}) {
  const pr = {number: 577, head: {sha: head}, base: {sha: base}};
  const live = {state: 'open', number: pr.number, draft,
    labels: labels.map(name => ({name})), head: pr.head, base: pr.base,
    merge_commit_sha: merge, changed_files: files.length};
  let calls = 0;
  const github = {rest: {pulls: {
    get: async () => ({data: snapshots ? snapshots[calls++] : live}),
    listFiles: async ({per_page, page}) => ({data: files.slice((page - 1) * per_page, page * per_page)}),
  }}};
  const context = {eventName, sha: merge, repo: {owner: 'org', repo: 'repo'},
    payload: {pull_request: pr, action, label: eventLabel ? {name: eventLabel} : null}};
  return classify({github, context, core: {warning: () => {}}});
}

test('classifies documentation formats across the repository', async () => {
  const paths = [
    'examples/online_serving/catalog/README.md',
    'benchmarks/tools/notes.md', 'other.md', 'README.MD',
    'docs/guide.rst', 'guide.RST', 'docs/tutorial.mdx', 'guide.MDX',
    'guide.markdown', 'guide.MarkDown',
    '.github/PULL_REQUEST_TEMPLATE.md', '.github/ISSUE_TEMPLATE/bug.md',
    '.github/workflows/README.md', 'docs/.github/README.md',
  ];
  const files = paths.map(filename => ({status: 'modified', filename}));
  assert.deepEqual(await classifyFiles(files), {docsOnly: true, files: paths});
});

test('allows images only in documentation asset directories', async () => {
  const paths = ['docs/assets/architecture.PNG', 'docs/images/nested/flow.svg'];
  for (const extension of ['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'avif']) {
    paths.push(`docs/assets/example.${extension}`);
  }
  const files = paths.map(filename => ({status: 'added', filename}));
  assert.deepEqual(await classifyFiles(files), {docsOnly: true, files: paths});
  assert.equal((await classifyFiles([
    {status: 'modified', filename: 'README.md'}, ...files,
  ])).docsOnly, true);
});

test('rejects mixed code, configuration, unsafe paths and non-documentation assets', async () => {
  for (const filename of [
    'examples/client.py', 'docs/conf.py', 'docs/patch_inventory.yaml',
    '.github/workflows/ci.yml', '.github/actions/build/action.yaml',
    '.github/scripts/check.sh', '.github/CODEOWNERS',
    'tests/fixtures/input.png', 'examples/image.jpg', 'assets/logo.svg',
    'docs/fixtures/input.png', 'docs/assets/run.py', 'docs/assets/data.json',
    'docs/assets/model.bin', 'docs/assets/file.svg.js',
    'docs/assets-old/image.png', 'docs/assetsimage.png',
    'md', 'docs/rst', 'docs/assets/png', 'README.md.bak',
    '../README.md', '/README.md', 'docs//README.md', 'docs/./README.md',
    'docs\\README.md', 'docs/line\nbreak.md', 'docs/../assets/image.png',
  ]) {
    const files = [
      {status: 'modified', filename: 'examples/README.md'},
      {status: 'modified', filename},
    ];
    assert.deepEqual(await classifyFiles(files), {docsOnly: false, files: []}, filename);
  }
});

test('requires both paths of a rename to qualify as documentation', async () => {
  const renamed = (previous_filename, filename) => [{status: 'renamed', previous_filename, filename}];
  for (const [previous, next] of [
    ['examples/old.md', 'benchmarks/new.md'],
    ['.github/README.md', 'examples/README.MD'],
    ['docs/old.rst', 'docs/new.mdx'],
    ['docs/assets/old.png', 'docs/images/new.PNG'],
  ]) {
    assert.deepEqual(await classifyFiles(renamed(previous, next)),
      {docsOnly: true, files: [next]});
  }
  for (const [previous, next] of [
    ['examples/client.py', 'examples/client.md'],
    ['examples/README.md', 'examples/client.py'],
    ['.github/workflows/ci.yml', '.github/workflows/ci.md'],
    ['tests/fixtures/input.png', 'docs/assets/input.png'],
    ['docs/assets/input.png', 'tests/fixtures/input.png'],
  ]) {
    assert.deepEqual(await classifyFiles(renamed(previous, next)), {docsOnly: false, files: []});
  }
});

test('classifies deletions but only returns surviving documents and images', async () => {
  const files = [
    {status: 'removed', filename: '.github/old.MD'},
    {status: 'removed', filename: 'docs/assets/old.png'},
    {status: 'added', filename: 'docs/new.rst'},
    {status: 'added', filename: 'docs/assets/new.png'},
  ];
  assert.deepEqual(await classifyFiles(files),
    {docsOnly: true, files: ['docs/new.rst', 'docs/assets/new.png']});
  assert.equal((await classifyFiles([
    ...files, {status: 'removed', filename: 'src/model.py'},
  ])).docsOnly, false);
});

test('examines files beyond the first API page before skipping full validation', async () => {
  const files = Array.from({length: 100}, (_, i) => ({status: 'modified', filename: `docs/${i}.md`}));
  assert.equal((await classifyFiles([
    ...files, {status: 'modified', filename: '.github/README.MD'},
  ])).docsOnly, true);
  assert.deepEqual(await classifyFiles([
    ...files, {status: 'modified', filename: '.github/workflows/ci.yml'},
  ]), {docsOnly: false, files: []});
});

// Construct legal-length components whose JSON output is exactly the limit.
function boundaryFiles() {
  return Array.from({length: 150}, (_, i) => {
    const prefix = `docs/${String(i).padStart(3, '0')}-`;
    const targetLength = i < 67 ? 216 : 215;
    return {status: 'modified', filename: prefix + 'x'.repeat(targetLength - prefix.length - 3) + '.md'};
  });
}

test('accepts the exact JSON byte limit and falls back above it', async () => {
  const files = boundaryFiles();
  assert.equal(Buffer.byteLength(JSON.stringify(files.map(file => file.filename)), 'utf8'), 32768);
  assert.equal((await classifyFiles(files)).docsOnly, true);
  const over = files.map(file => ({...file}));
  over[0].filename = over[0].filename.replace('.md', 'x.md');
  assert.deepEqual(await classifyFiles(over), {docsOnly: false, files: []});
});

test('counts JSON output bytes including non-ASCII paths, rather than characters', async () => {
  const files = boundaryFiles();
  files[0].filename = files[0].filename.replace('x', '\u6c49');
  assert.equal(Buffer.byteLength(JSON.stringify(files.map(file => file.filename)), 'utf8'), 32770);
  assert.deepEqual(await classifyFiles(files), {docsOnly: false, files: []});
});

test('does not count deleted paths in the lint file list', async () => {
  const files = boundaryFiles().map(file => ({...file, status: 'removed'}));
  files[0].filename = files[0].filename.replace('.md', 'x.md');
  assert.deepEqual(await classifyFiles(files), {docsOnly: true, files: []});
});

test('only an explicit A/B label event can classify an unready PR', async () => {
  const files = [{status: 'modified', filename: 'docs/readme.md'}];
  const abEvent = {labels: ['abtest'], action: 'labeled', eventLabel: 'abtest'};
  assert.deepEqual(await classifyFiles(files, abEvent), {docsOnly: true, files: ['docs/readme.md']});
  for (const changes of [
    {action: 'synchronize'}, {eventLabel: 'ready'}, {labels: []},
    {eventName: 'pull_request_review'}, {draft: true},
  ]) {
    assert.deepEqual(await classifyFiles(files, {...abEvent, ...changes}), {docsOnly: false, files: []});
  }
});

test('requires the A/B label and the same snapshot at both reads', async () => {
  const files = [{status: 'modified', filename: 'README.md'}];
  const live = {state: 'open', number: 577, draft: false, labels: [{name: 'abtest'}],
    head: {sha: head}, base: {sha: base}, merge_commit_sha: merge, changed_files: 1};
  for (const changed of [{labels: []}, {head: {sha: 'd'.repeat(40)}}, {draft: true}]) {
    assert.deepEqual(await classifyFiles(files, {labels: ['abtest'], action: 'labeled',
      eventLabel: 'abtest', snapshots: [live, {...live, ...changed}]}),
    {docsOnly: false, files: []});
  }
});

test('a recomputed test merge commit still classifies the same pull request', async () => {
  const files = [{status: 'modified', filename: 'docs/readme.md'}];
  const live = {state: 'open', number: 577, draft: false, labels: [{name: 'ready'}],
    head: {sha: head}, base: {sha: base}, merge_commit_sha: merge, changed_files: 1};
  // GitHub rebuilds this throwaway commit when the base branch advances. A run
  // that waited in the queue must still classify an unchanged pull request.
  const later = {...live, merge_commit_sha: 'd'.repeat(40)};
  assert.deepEqual(await classifyFiles(files, {snapshots: [live, later]}),
    {docsOnly: true, files: ['docs/readme.md']});
});
