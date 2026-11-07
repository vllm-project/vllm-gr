// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {preparePolicy} = require('../tools/ci/benchmark_policy_runtime');

const base = 'a'.repeat(40);
const head = 'b'.repeat(40);
const merge = 'c'.repeat(40);
const policy = '.github/benchmark-gate.yaml';

function fixture({mergePolicy = 'candidate policy', headPolicy = 'candidate policy',
  changedHead = false} = {}) {
  const pr = {number: 1, state: 'open', user: {login: 'author'},
    head: {sha: head, repo: {owner: {login: 'fork'}, name: 'repo'}},
    base: {sha: base}, merge_commit_sha: merge};
  let calls = 0;
  const github = {rest: {
    pulls: {get: async () => ({data: ++calls === 2 && changedHead ?
      {...pr, head: {...pr.head, sha: 'd'.repeat(40)}} : pr})},
    repos: {getContent: async ({path: name, ref}) => {
      const value = name === policy ? (ref === base ? 'base policy' :
        ref === head ? headPolicy : mergePolicy) : '# base-owned code';
      const content = Buffer.from(value);
      return {data: {type: 'file', encoding: 'base64', size: content.length,
        content: content.toString('base64')}};
    }},
  }};
  let summary = '';
  const core = {info() {}, summary: {addRaw(value) {summary += value; return this;},
    async write() {}}};
  const context = {eventName: 'pull_request', sha: merge, repo: {owner: 'org', repo: 'repo'},
    payload: {pull_request: pr}};
  return {github, core, context, summary: () => summary};
}

test('changed policy runs without any review APIs', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'policy-test-'));
  t.after(() => fs.rmSync(tempDir, {recursive: true, force: true}));
  const f = fixture();
  const result = await preparePolicy({...f, tempDir});
  assert.equal(result.policyMode, 'candidate');
  assert.equal(result.sourceRef, head);
  assert.equal(fs.readFileSync(result.policyPath, 'utf8'), 'candidate policy');
  assert.match(f.summary(), /Candidate policy/);
  assert.doesNotMatch(f.summary(), /approval.*required/i);
});

test('changed policy still rejects inconsistent merge and changed PR head', async t => {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'policy-test-'));
  t.after(() => fs.rmSync(tempDir, {recursive: true, force: true}));
  await assert.rejects(preparePolicy({...fixture({mergePolicy: 'other'}), tempDir}), /Merged policy differs/);
  await assert.rejects(preparePolicy({...fixture({changedHead: true}), tempDir}), /no longer matches/);
});
