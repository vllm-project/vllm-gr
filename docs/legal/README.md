# Copyright and source attribution

## Project license

vLLM-GR is distributed under the [Apache License, Version 2.0](../../LICENSE).
The root [NOTICE](../../NOTICE) identifies third-party sources, their revisions
and applicable license texts. Check the entry for the material you are using
alongside its file header.

## File headers

For original project code, use the following header with the comment syntax of
the file's language.

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
```

### Adapted and upstream files

For an adapted file, keep the GR project holder, preserve upstream copyright
notices in comments, identify the source and mark the modification. Set the
SPDX license expression to reflect the licenses that apply. Record the
upstream path and revision in [NOTICE](../../NOTICE). For example, the
[configuration validator](../../tools/pre_commit/validate_config.py) adapts
vLLM code and includes an MIT-licensed helper by David Lord. Its header uses
`Apache-2.0 AND MIT`.

For a file that retains upstream authorship, preserve the applicable holder and
license and register them in [spdx_policy.json](../../tools/pre_commit/spdx_policy.json).

### Generated bindings

For generated gRPC bindings, update the source schema and regenerate the
bindings with the [generator](../../tools/codegen/generate_grpc_proto.py).

## Checks

Run the same read-only check as pre-commit from the repository root.

```bash
pre-commit run check-spdx-header --all-files
```

The hook passes `--all-tracked` to the [checker](../../tools/pre_commit/check_spdx_header.py).
It reads the leading comment block of tracked Python, shell, JavaScript and
Protobuf sources, skipping zero-byte `__init__.py` files. Each header must have
a valid SPDX license expression and the expected holder. For registered paths,
the license expression must also match the policy entry.

To add the standard header to a new original source file, pass its path
explicitly. Review the resulting diff and rerun the read-only check.

```bash
python tools/pre_commit/check_spdx_header.py path/to/new_file.py
```

This command returns a nonzero status when it adds a header. It leaves partial
headers and existing plain copyright notices for the contributor to resolve.
