// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'use strict';
const fs = require('node:fs');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [];
const outputs = {};
let getCount = 0;
const github = {rest: {pulls: {
  get: async args => {
    calls.push(args);
    if (payload.api_error) throw new Error('GitHub unavailable');
    getCount++;
    return {data: getCount > 1 && payload.live2 !== undefined ? payload.live2 : payload.live};
  },
  listFiles: async args => {
    if (payload.file_api_error) throw new Error('files unavailable');
    const files = payload.files || [];
    return {data: files.slice((args.page - 1) * args.per_page, args.page * args.per_page)};
  },
}}};
const core = {info() {}, warning() {}, setOutput(name, value) {outputs[name] = value;}};
(async () => {
  try {
    await new AsyncFunction('context', 'github', 'core', payload.script)(payload.context, github, core);
    process.stdout.write(JSON.stringify({ok: true, calls, outputs}));
  } catch (error) {
    process.stdout.write(JSON.stringify({ok: false, calls, outputs, error: String(error)}));
  }
})();
