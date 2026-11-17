// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
'use strict';

// This module must be loaded from the pinned base revision, not the PR.
// GitHub's PR files endpoint is capped at 3,000 files. Reject that boundary
// rather than risk treating a truncated list as a complete documentation diff.
const MAX_FILES = 3000;
const PAGE_SIZE = 100;
// The JSON list is passed as a single environment variable in two jobs.
// Keep it well below Linux's per-string execve limit and GitHub output limits.
const MAX_FILES_JSON_BYTES = 32 * 1024;
const shaPattern = /^[0-9a-f]{40}$/i;
const statuses = new Set(['added', 'modified', 'removed', 'renamed']);
const textExtensions = new Set(['md', 'markdown', 'mdx', 'rst']);
const imageExtensions = new Set(['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp', 'ico', 'avif']);

function isDocumentation(path) {
  if (typeof path !== 'string' ||
      /[\x00-\x1f\x7f\\\\]/.test(path)) return false;
  const parts = path.split('/');
  if (parts.some(part => !part || part === '.' || part === '..')) {
    return false;
  }
  const extension = parts[parts.length - 1].match(/\.([^.]+)$/)?.[1].toLowerCase();
  if (textExtensions.has(extension)) return true;
  // Images elsewhere may be runtime inputs or test fixtures. Only these
  // documentation asset directories can take the lightweight route.
  return parts.length >= 3 && parts[0] === 'docs' &&
    (parts[1] === 'assets' || parts[1] === 'images') &&
    imageExtensions.has(extension);
}

// The ephemeral test merge commit is deliberately not compared here. GitHub
// recomputes it whenever the base branch advances, so a run that queues for a
// while would otherwise reject an unchanged pull request. The immutable head
// and base commits still pin every fact this classifier depends on.
function currentSnapshot(live, pr, allowUnreadyAbtest) {
  if (!live || live.state !== 'open' || live.number !== pr.number ||
      live.draft !== false || !Array.isArray(live.labels) ||
      !(live.labels.some(label => label && label.name === 'ready') ||
        (allowUnreadyAbtest && live.labels.some(label => label && label.name === 'abtest'))) ||
      !live.head || !live.base || live.head.sha !== pr.head.sha ||
      live.base.sha !== pr.base.sha ||
      !Number.isSafeInteger(live.changed_files) || live.changed_files <= 0 ||
      live.changed_files >= MAX_FILES) {
    throw new Error('PR revisions or changed file count are invalid or changed.');
  }
  return live.changed_files;
}

// Errors and uncertain states deliberately return the full-validation path.
// Callers must independently preserve ready/draft and final-gate conditions.
async function classify({github, context, core}) {
  try {
    const pr = context && context.payload && context.payload.pull_request;
    const repo = context && context.repo;
    if (!context || !['pull_request', 'pull_request_review'].includes(context.eventName) ||
        !pr || !Number.isSafeInteger(pr.number) || pr.number <= 0 ||
        !pr.head || !pr.base || !shaPattern.test(pr.head.sha || '') ||
        !shaPattern.test(pr.base.sha || '') ||
        !repo || typeof repo.owner !== 'string' || !repo.owner ||
        typeof repo.repo !== 'string' || !repo.repo) {
      throw new Error('Missing pinned pull request context.');
    }
    const args = {owner: repo.owner, repo: repo.repo, pull_number: pr.number};
    // Explicit A/B-only requests may arrive before the ready label. Allow
    // classification only for that label event, not arbitrary unready PRs.
    const allowUnreadyAbtest = context.eventName === 'pull_request' &&
      context.payload.action === 'labeled' && context.payload.label &&
      context.payload.label.name === 'abtest';
    const getCurrent = async () => {
      const {data} = await github.rest.pulls.get(args);
      return currentSnapshot(data, pr, allowUnreadyAbtest);
    };
    const expected = await getCurrent();
    const seen = new Set();
    const files = [];
    let count = 0;
    for (let page = 1; page <= MAX_FILES / PAGE_SIZE; page++) {
      const {data} = await github.rest.pulls.listFiles({
        ...args, per_page: PAGE_SIZE, page,
      });
      if (!Array.isArray(data) || data.length > PAGE_SIZE) {
        throw new Error('Invalid PR files response.');
      }
      for (const file of data) {
        if (!file || !statuses.has(file.status) || !isDocumentation(file.filename) ||
            seen.has(file.filename)) {
          throw new Error('PR contains an unsupported or non-documentation file.');
        }
        if (file.status === 'renamed') {
          if (!isDocumentation(file.previous_filename) ||
              file.previous_filename === file.filename) {
            throw new Error('The original renamed path is not documentation.');
          }
        } else if (file.previous_filename != null) {
          throw new Error('Unexpected previous path.');
        }
        seen.add(file.filename);
        count++;
        if (count > expected || count >= MAX_FILES) {
          throw new Error('PR files response exceeds the expected count.');
        }
        if (file.status !== 'removed') files.push(file.filename);
      }
      if (data.length < PAGE_SIZE) break;
      if (page === MAX_FILES / PAGE_SIZE) {
        throw new Error('PR files response may be truncated.');
      }
    }
    if (count !== expected || await getCurrent() !== expected) {
      throw new Error('PR files response is incomplete or the PR changed.');
    }
    if (Buffer.byteLength(JSON.stringify(files), 'utf8') > MAX_FILES_JSON_BYTES) {
      throw new Error('Documentation file list is too large for workflow environment variables.');
    }
    return {docsOnly: true, files};
  } catch (error) {
    if (core && typeof core.warning === 'function') {
      core.warning('Documentation-only classification could not be established; full validation is required.');
    }
    return {docsOnly: false, files: []};
  }
}

module.exports = {classify};
