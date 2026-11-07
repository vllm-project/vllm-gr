// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'use strict';

// This module must itself be loaded from the protected base revision. It reads
// candidate files as data only, and never loads a candidate policy resolver.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const POLICY = '.github/benchmark-gate.yaml';
const RESOLVER = 'tools/ci/resolve_benchmark_policy.py';
const NORMALIZER = 'tools/ci/normalize_benchmark_policy_env.py';
const MAX_FILE_BYTES = 1024 * 1024;
const shaPattern = /^[a-f0-9]{40}$/i;
const hash = value => crypto.createHash('sha256').update(value).digest('hex');

// PR policy is untrusted data. Escape before inserting it into the Markdown
// summary's HTML block, and cap the preview well below GitHub's summary limit.
function policyPreview(value) {
  const limit = 65536;
  const preview = value.slice(0, limit).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g,
    char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
  const escaped = preview.replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
  return escaped + (value.length > limit ? '\n[Preview truncated; inspect the full PR diff.]' : '');
}

async function reportPolicyProposal(core, basePolicy, candidatePolicy, baseSha, headSha) {
  if (!core.summary || typeof core.summary.addRaw !== 'function') {
    throw new Error('Cannot publish the required policy proposal summary.');
  }
  const section = `### Proposed benchmark policy change\n\n` +
    `Base commit: \`${baseSha}\`  \nBase policy SHA-256: \`${hash(basePolicy)}\`  \n` +
    `Candidate commit: \`${headSha}\`  \nCandidate policy SHA-256: \`${hash(candidatePolicy)}\`\n\n` +
    `#### Base policy\n\n<pre><code>${policyPreview(basePolicy)}</code></pre>\n\n` +
    `#### Candidate policy\n\n<pre><code>${policyPreview(candidatePolicy)}</code></pre>\n\n` +
    `Review the policy and test changes at this exact candidate commit.\n`;
  await core.summary.addRaw(section).write();
}

function requireSha(value, label) {
  if (typeof value !== 'string' || !shaPattern.test(value)) {
    throw new Error(`${label} must be a full commit SHA.`);
  }
  return value.toLowerCase();
}

async function readFile(github, owner, repo, filePath, ref, optional = false) {
  let response;
  try {
    response = await github.rest.repos.getContent({owner, repo, path: filePath, ref});
  } catch (error) {
    if (optional && error.status === 404) return null;
    throw error;
  }
  const item = response && response.data;
  if (!item || Array.isArray(item) || item.type !== 'file' ||
      item.encoding !== 'base64' || typeof item.content !== 'string' ||
      !Number.isSafeInteger(item.size) || item.size < 0 || item.size > MAX_FILE_BYTES) {
    throw new Error(`${filePath}@${ref} is not a readable bounded regular file.`);
  }
  const compact = item.content.replace(/\s/g, '');
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(compact)) {
    throw new Error(`${filePath}@${ref} has invalid base64 contents.`);
  }
  const bytes = Buffer.from(compact, 'base64');
  if (bytes.length !== item.size) throw new Error(`${filePath}@${ref} has an invalid size.`);
  // Reject lossy UTF-8 decoding, including a corrupted policy or executable.
  const text = bytes.toString('utf8');
  if (!Buffer.from(text, 'utf8').equals(bytes)) throw new Error(`${filePath}@${ref} is not UTF-8.`);
  return text;
}

function validatePr(pr, expected, label) {
  if (!pr || pr.state !== 'open' || pr.number !== expected.number || !pr.head || !pr.base ||
      !pr.user || typeof pr.user.login !== 'string' || !pr.user.login ||
      requireSha(pr.head.sha, `${label} head`) !== expected.headSha ||
      requireSha(pr.base.sha, `${label} base`) !== expected.baseSha ||
      requireSha(pr.merge_commit_sha, `${label} merge`) !== expected.mergeSha) {
    throw new Error(`${label} no longer matches the event's pinned head, base, and merge commits.`);
  }
  if (!pr.head.repo || !pr.head.repo.owner || !pr.head.repo.owner.login || !pr.head.repo.name) {
    throw new Error('The head repository is unavailable; candidate policy cannot be verified.');
  }
}

async function preparePolicy({github, context, core, tempDir}) {
  if (!github || !github.rest || !context || !context.repo || !core || !tempDir) {
    throw new Error('Missing trusted policy runtime inputs.');
  }
  const {owner, repo} = context.repo;
  const event = context.eventName;
  const payloadPr = context.payload && context.payload.pull_request;
  const isPr = event === 'pull_request' || event === 'pull_request_review';
  if (!isPr && event !== 'workflow_dispatch') throw new Error(`Unsupported event: ${event}`);
  const mergeSha = requireSha(context.sha, 'workflow commit');
  let baseSha = mergeSha;
  let headSha = null;
  let number = null;
  let initialPr = null;
  if (isPr) {
    if (!payloadPr || !Number.isSafeInteger(payloadPr.number) || payloadPr.number <= 0 ||
        !payloadPr.head || !payloadPr.base) throw new Error('Missing PR event data.');
    number = payloadPr.number;
    headSha = requireSha(payloadPr.head.sha, 'event head');
    baseSha = requireSha(payloadPr.base.sha, 'event base');
    initialPr = (await github.rest.pulls.get({owner, repo, pull_number: number})).data;
    validatePr(initialPr, {number, headSha, baseSha, mergeSha}, 'PR');
  }

  // Use only base-owned runtime and policy; each file has the same pinned SHA.
  const [basePolicy, resolver, normalizer] = await Promise.all([
    readFile(github, owner, repo, POLICY, baseSha),
    readFile(github, owner, repo, RESOLVER, baseSha),
    readFile(github, owner, repo, NORMALIZER, baseSha),
  ]);
  let effectivePolicy = basePolicy;
  let policyMode = 'base';
  if (isPr) {
    const mergePolicy = await readFile(github, owner, repo, POLICY, mergeSha);
    if (mergePolicy !== basePolicy) {
      // Pull-request commits are available via the base repository's PR refs.
      // Resolve the immutable head SHA through the base repository so the
      // base-scoped token does not also need permission on a private fork.
      const candidate = await readFile(github, owner, repo, POLICY, headSha);
      if (candidate !== mergePolicy) {
        throw new Error('Merged policy differs from the head policy; update the PR branch.');
      }
      await reportPolicyProposal(core, basePolicy, candidate, baseSha, headSha);
      effectivePolicy = candidate;
      policyMode = 'candidate';
    }
    // The API may have changed while we resolved the immutable snapshots.
    const currentPr = (await github.rest.pulls.get({owner, repo, pull_number: number})).data;
    validatePr(currentPr, {number, headSha, baseSha, mergeSha}, 'Current PR');
    if (currentPr.user.login.toLowerCase() !== initialPr.user.login.toLowerCase()) {
      throw new Error('The PR author changed while verifying policy.');
    }
  }

  const dir = fs.mkdtempSync(path.join(tempDir, 'benchmark-policy-result-'));
  const policyPath = path.join(dir, 'benchmark-gate.yaml');
  // The normalizer imports tools.ci.resolve_benchmark_policy. Build a regular
  // package rooted in this private directory so that import cannot fall back
  // to the candidate checkout or an unrelated installed namespace package.
  const toolsDir = path.join(dir, 'tools');
  const ciDir = path.join(toolsDir, 'ci');
  fs.mkdirSync(ciDir, {recursive: true, mode: 0o700});
  fs.writeFileSync(path.join(toolsDir, '__init__.py'), '', {mode: 0o600});
  fs.writeFileSync(path.join(ciDir, '__init__.py'), '', {mode: 0o600});
  const resolverPath = path.join(ciDir, 'resolve_benchmark_policy.py');
  const normalizerPath = path.join(ciDir, 'normalize_benchmark_policy_env.py');
  fs.writeFileSync(policyPath, effectivePolicy, {encoding: 'utf8', mode: 0o600});
  fs.writeFileSync(resolverPath, resolver, {encoding: 'utf8', mode: 0o600});
  fs.writeFileSync(normalizerPath, normalizer, {encoding: 'utf8', mode: 0o600});
  const sourceRef = policyMode === 'candidate' ? headSha : baseSha;
  const policySha256 = hash(effectivePolicy);
  core.info(`Benchmark policy mode=${policyMode} source=${sourceRef} base=${baseSha} sha256=${policySha256}`);
  return {policyPath, resolverPath, normalizerPath, sourceRef, policyMode,
    baseSha, headSha, mergeSha, policySha256};
}

module.exports = {preparePolicy};
