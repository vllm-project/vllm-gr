// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
