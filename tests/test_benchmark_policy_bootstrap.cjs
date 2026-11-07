// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Execute both real inline workflow fallbacks with mock GitHub responses.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const workflow = fs.readFileSync(path.join(__dirname, '../.github/workflows/pre-commit.yml'), 'utf8');
function scriptFor(name) {
  const start = workflow.indexOf(`      - name: ${name}\n`);
  assert.notEqual(start, -1);
  const scriptStart = workflow.indexOf('          script: |\n', start) + '          script: |\n'.length;
  const end = workflow.indexOf('\n      - name:', scriptStart);
  assert.ok(scriptStart > start && end > scriptStart);
  return workflow.slice(scriptStart, end).split('\n').map(line => line.startsWith('            ') ? line.slice(12) : line).join('\n');
}
const scripts = [
  'Fetch target branch benchmark policy runtime',
  'Revalidate candidate benchmark policy',
].map(scriptFor);
const base = 'a'.repeat(40), head = 'b'.repeat(40), merge = 'c'.repeat(40);
const candidate = 'candidate policy\n';
const candidateDigest = crypto.createHash('sha256').update(candidate).digest('hex');
const pr = {number: 577, head: {sha: head, repo: {owner: {login: 'owner'}, name: 'repo'}}, base: {sha: base}, user: {login: 'author'}, state: 'open', merge_commit_sha: merge};

async function run(script, options = {}) {
  let snapshots = 0;
  const writes = new Map(), outputs = {};
  const fakeFs = {mkdtempSync: () => '/temp/bootstrap', mkdirSync: () => {}, writeFileSync: (file, value) => writes.set(file, value)};
  const github = {rest: {
    repos: {getContent: async ({path: file, ref}) => {
      if (file.endsWith('benchmark_policy_runtime.js')) throw Object.assign(new Error('missing'), {status: 404});
      let value = file === '.github/benchmark-gate.yaml' ? (ref === base ? 'base policy\n' : candidate) : 'base owned code';
      if (options.mismatch && file === '.github/benchmark-gate.yaml' && ref === merge) value = 'mismatch';
      const bytes = Buffer.from(value);
      return {data: {type: 'file', encoding: 'base64', size: bytes.length, content: bytes.toString('base64')}};
    }},
    pulls: {get: async () => {
      snapshots++;
      return {data: options.changed && snapshots > 1 ? {...pr, merge_commit_sha: 'd'.repeat(40)} : pr};
    }},
  }};
  const core = {summary: {addRaw() { return this; }, async write() {}}, info() {}, setOutput(name, value) {outputs[name] = value;}};
  const env = {POLICY_REF: base, RUNNER_TEMP: '/temp', ...(options.final ? {POLICY_VERIFY_FINAL: 'true', EXPECTED_POLICY_DIGEST: options.digest || candidateDigest} : {})};
  const fakeRequire = name => name === 'fs' ? fakeFs : require(name);
  const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
  await new AsyncFunction('require', 'github', 'context', 'core', 'process', 'Buffer', script)(fakeRequire, github, {repo: {owner: 'owner', repo: 'repo'}, payload: {pull_request: pr}, sha: merge}, core, {env}, Buffer);
  return {outputs, writes};
}

for (const [i, script] of scripts.entries()) {
  test(`workflow fallback ${i + 1} uses candidate without approval`, async () => {
    const {outputs, writes} = await run(script, {final: i === 1});
    assert.equal(outputs.mode, 'candidate');
    assert.equal(outputs.digest, candidateDigest);
    assert.equal(outputs.ref, head);
    assert.equal(writes.get(outputs['policy-path']), candidate);
  });
  test(`workflow fallback ${i + 1} rejects mismatched merge policy`, async () => {
    await assert.rejects(run(script, {mismatch: true}), /Merged policy differs/);
  });
  test(`workflow fallback ${i + 1} rejects a changed PR snapshot`, async () => {
    await assert.rejects(run(script, {changed: true}), /PR head, base or merge changed/);
  });
}
test('final workflow fallback rejects changed candidate digest', async () => {
  await assert.rejects(run(scripts[1], {final: true, digest: '0'.repeat(64)}), /Candidate benchmark policy changed/);
});
