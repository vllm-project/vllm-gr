# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm

import sys
from enum import Enum
from pathlib import Path

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression


class SPDXStatus(Enum):
    """SPDX header status enumeration"""

    EMPTY = "empty"  # empty __init__.py
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"  # Only has copyright line
    MISSING_COPYRIGHT = "missing_copyright"  # Only has license line
    MISSING_BOTH = "missing_both"  # Completely missing
    INVALID = "invalid"  # Empty value or invalid license expression


LICENSE_PREFIXES = ("# SPDX-License-Identifier:", "// SPDX-License-Identifier:")
COPYRIGHT_PREFIXES = ("# SPDX-FileCopyrightText:", "// SPDX-FileCopyrightText:")
HASH_COMMENT_SUFFIXES = {".py", ".pyi", ".sh"}
SLASH_COMMENT_SUFFIXES = {".proto", ".js", ".cjs", ".mjs"}

def comment_prefix(file_path):
    """Comment token this file type uses, or None when the hook must not rewrite it.

    Writing "#" into a .proto would make it unparsable, so the prefix follows
    the file type instead of being fixed.
    """
    suffix = Path(file_path).suffix
    if suffix in SLASH_COMMENT_SUFFIXES:
        return "//"
    if suffix in HASH_COMMENT_SUFFIXES:
        return "#"
    return None


def spdx_header(prefix):
    """The two-line header written into a file that has neither line."""
    return (
        f"{prefix} SPDX-License-Identifier: Apache-2.0\n"
        f"{prefix} SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project"
    )


def valid_license(value):
    """True for a non-empty, valid SPDX license expression such as "A AND B"."""
    value = value.strip()
    if not value:
        return False
    try:
        canonicalize_license_expression(value)
    except InvalidLicenseExpression:
        return False
    return True


def check_spdx_header_status(file_path):
    """Check SPDX header status of the file"""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()
        if not lines:
            # Empty file
            return SPDXStatus.EMPTY

        # Skip shebang line
        start_idx = 0
        if lines and lines[0].startswith("#!"):
            start_idx = 1

        has_license = False
        has_copyright = False
        invalid = False

        # Check all lines for SPDX headers (not just the first two)
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            for prefix in LICENSE_PREFIXES:
                if line.startswith(prefix):
                    has_license = True
                    invalid |= not valid_license(line[len(prefix) :])
            for prefix in COPYRIGHT_PREFIXES:
                if line.startswith(prefix):
                    has_copyright = True
                    invalid |= not line[len(prefix) :].strip()

        # Determine status based on what we found
        if invalid:
            return SPDXStatus.INVALID
        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        elif has_license and not has_copyright:
            # Only has license line
            return SPDXStatus.MISSING_COPYRIGHT
            # Only has copyright line
        elif not has_license and has_copyright:
            return SPDXStatus.MISSING_LICENSE
        else:
            # Completely missing both lines
            return SPDXStatus.MISSING_BOTH


def add_header(file_path, status):
    """Add or supplement SPDX header based on status"""
    prefix = comment_prefix(file_path)
    if prefix is None:
        return
    with open(file_path, "r+", encoding="UTF-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            # Completely missing, add complete header
            if lines and lines[0].startswith("#!"):
                # Preserve shebang line
                file.write(lines[0])
                file.write(spdx_header(prefix) + "\n")
                file.writelines(lines[1:])
            else:
                # Add header directly
                file.write(spdx_header(prefix) + "\n")
                file.writelines(lines)

        elif status == SPDXStatus.MISSING_COPYRIGHT:
            # Only has license line, need to add copyright line
            # Find the license line and add copyright line after it
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{prefix} SPDX-License-Identifier:"):
                    # Insert copyright line after license line
                    lines.insert(
                        i + 1,
                        f"{prefix} SPDX-FileCopyrightText: "
                        "Copyright contributors to the vLLM-GR project\n",
                    )
                    break

            file.writelines(lines)

        else:
            # MISSING_LICENSE / INVALID: leave the file untouched.
            file.writelines(lines)


def main():
    """Main function"""
    files_missing_both = []
    files_missing_copyright = []
    files_missing_license = []
    files_invalid = []
    files_unsupported = []

    for file_path in sys.argv[1:]:
        status = check_spdx_header_status(file_path)

        if comment_prefix(file_path) is None:
            # Unknown comment syntax: a guess would corrupt the file.
            if status not in (SPDXStatus.COMPLETE, SPDXStatus.EMPTY):
                files_unsupported.append(file_path)
            continue

        if status == SPDXStatus.MISSING_BOTH:
            files_missing_both.append(file_path)
        elif status == SPDXStatus.MISSING_COPYRIGHT:
            files_missing_copyright.append(file_path)
        elif status == SPDXStatus.MISSING_LICENSE:
            files_missing_license.append(file_path)
        elif status == SPDXStatus.INVALID:
            files_invalid.append(file_path)
        else:
            continue

    # Files that need a human decision are never rewritten
    manual = files_missing_license + files_invalid + files_unsupported
    if files_unsupported:
        print("Unknown comment syntax for this file type; add the SPDX header by hand:")
        for file_path in files_unsupported:
            print(f"  {file_path}")
    if files_missing_license:
        print("Copyright line without a license line; confirm the license and add it:")
        for file_path in files_missing_license:
            print(f"  {file_path}")
    if files_invalid:
        print("Empty value or invalid SPDX license expression:")
        for file_path in files_invalid:
            print(f"  {file_path}")

    # Collect all files that can be fixed automatically
    all_files_to_fix = files_missing_both + files_missing_copyright
    if all_files_to_fix:
        print("The following files are missing the SPDX header:")
        if files_missing_both:
            for file_path in files_missing_both:
                print(f"  {file_path}")
                add_header(file_path, SPDXStatus.MISSING_BOTH)

        if files_missing_copyright:
            for file_path in files_missing_copyright:
                print(f"  {file_path}")
                add_header(file_path, SPDXStatus.MISSING_COPYRIGHT)

    sys.exit(1 if all_files_to_fix or manual else 0)


if __name__ == "__main__":
    main()
