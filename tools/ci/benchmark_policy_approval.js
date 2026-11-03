// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'use strict';

// Pure decision function; callers must obtain complete reviews and current
// permissions from GitHub and verify the event's PR refs against the API.
function decidePolicy({headSha, baseSha, prAuthor, basePolicy, candidatePolicy, reviews, permissions}) {
  const blocked = reason => ({mode: 'blocked', reason});
  const sha = value => typeof value === 'string' && /^[0-9a-f]{40}$/i.test(value);
  if (!sha(headSha) || !sha(baseSha) || typeof prAuthor !== 'string' || !prAuthor ||
      typeof basePolicy !== 'string' || typeof candidatePolicy !== 'string') {
    return blocked('Missing or invalid pinned PR, policy, review, or permission data.');
  }
  if (headSha.toLowerCase() === baseSha.toLowerCase() && basePolicy !== candidatePolicy) {
    return blocked('The same commit cannot have different policy contents.');
  }
  if (basePolicy === candidatePolicy) {
    return {mode: 'base', reason: 'The candidate policy is identical to the base policy.'};
  }
  if (!Array.isArray(reviews) || !permissions || typeof permissions !== 'object') {
    return blocked('Missing or invalid review or permission data.');
  }
  const allowed = new Set(['write', 'maintain', 'admin']);
  const normalizedPermissions = Object.fromEntries(
    Object.entries(permissions).map(([name, permission]) => [name.toLowerCase(), permission]),
  );
  const latest = new Map();
  for (const review of reviews) {
    if (!review || !review.user || review.user.type !== 'User' ||
        typeof review.user.login !== 'string' || !review.user.login ||
        !['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(review.state) ||
        typeof review.submitted_at !== 'string' || !Number.isFinite(Date.parse(review.submitted_at)) ||
        !Number.isSafeInteger(review.id)) continue;
    const name = review.user.login.toLowerCase();
    const previous = latest.get(name);
    const timestamp = Date.parse(review.submitted_at);
    if (!previous || timestamp > previous.timestamp ||
        (timestamp === previous.timestamp && review.id > previous.review.id)) {
      latest.set(name, {review, timestamp});
    }
  }
  const eligible = [...latest.entries()].filter(([name]) =>
    name !== prAuthor.toLowerCase() && allowed.has(normalizedPermissions[name]),
  );
  if (eligible.some(([, {review}]) => review.state === 'CHANGES_REQUESTED')) {
    return blocked('An authorized reviewer has an outstanding request for changes.');
  }
  const approval = eligible.find(([, {review}]) => review.state === 'APPROVED' &&
    typeof review.commit_id === 'string' && review.commit_id.toLowerCase() === headSha.toLowerCase());
  if (!approval) return blocked('The changed policy requires a current-head approval from a non-author with write access.');
  return {mode: 'candidate', reason: `Policy approved by ${approval[0]} for the current head.`, reviewer: approval[0]};
}

module.exports = {decidePolicy};
