# Copyright and source attribution

This page records how vLLM-GR marks authorship and third-party code. It follows
the convention of vLLM-Omni. It is an engineering convention, not a legal
determination of ownership.

## File headers

Original project code uses exactly these two lines:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
```

No year is added, because no verified year range exists.

A file adapted from another project keeps the project line and records its
origin in plain comments below it. Upstream copyright is written as a plain
comment, and only when the upstream file carries one:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Copyright 2025 The xLLM Authors. All Rights Reserved.
# Adapted from https://github.com/xLLM-AI/xllm-ops
```

- Use the repository URL after `Adapted from`. Commit-pinned revisions and the
  upstream path of each file are listed in the root `NOTICE`.
- If the adapted file is under a different license, say so in
  `SPDX-License-Identifier` (for example `Apache-2.0 AND BSD-3-Clause`).
- A file copied without changes keeps the upstream `SPDX-FileCopyrightText` and
  does not get the project line.
- Do not add import dates, per-file change lists, or repeated "Adapted from"
  blocks; the change history lives in git.

SPDX completeness does not prove who owns the code. The root `LICENSE` and
`NOTICE` do not replace file-level notices.

## Third-party license texts

The root `NOTICE` lists what was taken from each upstream project, including the
revision and the upstream path of every adapted file. It reproduces the MIT and
BSD-3-Clause texts, so they are present in every distribution. Apache-2.0
upstream code needs no separate copy of the license text: that text is the root
`LICENSE`, and `NOTICE` records each upstream's own copyright line.

## Checks

```text
pre-commit run check-spdx-header --all-files
python -m unittest tests.legal.test_spdx_header
python tools/generate_grpc_proto.py --check
```

The SPDX hook always runs over every tracked source file (Python, shell,
JavaScript, proto). It reads only the leading comment block: a tag after code,
inside a string or in a heredoc is not a header. It requires a non-empty
`SPDX-License-Identifier` holding a valid SPDX expression and a non-empty
`SPDX-FileCopyrightText`, and accepts several copyright lines.

It completes a header only for a file that has no header line at all, and it
reports every other case for a person to resolve: a copyright line without a
license line, a license line without a copyright line, and a plain
`Copyright ...` comment with no SPDX line. The last one matters most, because
an upstream copy often carries only that comment, and filling in the project
holder would claim the file and pass on the next run. A rewrite keeps the BOM,
the encoding cookie, the shebang and the line endings as they were, and a file
that is not UTF-8 is reported instead of decoded with replacement characters.

The hook enumerates the repository itself, so it checks the same files from any
subdirectory. It fails the run when it completes a header, because pre-commit
passes only when the hook changed nothing: stage the file again.

`tools/pre_commit/spdx_policy.json` records the holder every file must carry,
plus the six files that are copies of third-party code and keep the upstream
holder instead. Any other holder written as `SPDX-FileCopyrightText` fails the
check, as does registering a copy that is no longer tracked. A file derived
from another project still carries the project holder; its upstream copyright
is a plain comment, which the policy does not police.

The policy covers the holder, not the license, so a project file may declare
`Apache-2.0`, `Apache-2.0 AND BSD-3-Clause` and so on. It cannot show that a
file is free of undisclosed copying, so provenance is still reviewed when a
file is added.
