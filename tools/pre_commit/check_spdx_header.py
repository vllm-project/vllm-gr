# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# Adapted from https://github.com/vllm-project/vllm

"""Check and complete file-level SPDX headers.

Two modes:

- ``--all-tracked`` enumerates every tracked source file, checks it against
  ``spdx_policy.json`` and never writes. This is the pre-commit hook.
- given file names (a manual developer convenience, not the hook), it
  completes a file that has no header at all and reports everything else. A
  file that already states a license, states a copyright in any form, or
  cannot be decoded is never rewritten: a license expression does not determine
  who holds the copyright, and a plain "Copyright" comment is often the only
  notice an upstream copy carries.
"""

import json
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression

POLICY_PATH = Path(__file__).with_name("spdx_policy.json")
REPO_ROOT = POLICY_PATH.parents[2]
SOURCE_SUFFIXES = {".py", ".pyi", ".sh", ".js", ".cjs", ".mjs", ".proto"}
LICENSE_PREFIXES = ("# SPDX-License-Identifier:", "// SPDX-License-Identifier:")
COPYRIGHT_PREFIXES = ("# SPDX-FileCopyrightText:", "// SPDX-FileCopyrightText:")
COMMENT_PREFIXES = ("#", "//")
HASH_COMMENT_SUFFIXES = {".py", ".pyi", ".sh"}
SLASH_COMMENT_SUFFIXES = {".proto", ".js", ".cjs", ".mjs"}
# Only the leading comment block is a header. A tag further down is prose, a
# fixture inside a test, or a heredoc, and must not satisfy the check.
BOM = "\ufeff"  # UTF-8 byte order mark, kept as an escape so it stays visible
ENCODING_COOKIE = re.compile(r"^[#/]{1,2}.*coding[:=]\s*[-\w.]+")
PLAIN_COPYRIGHT = re.compile(r"^[#/]{1,2}\s*Copyright\b", re.IGNORECASE)


class SPDXStatus(Enum):
    """SPDX header status enumeration"""

    EMPTY = "empty"  # empty __init__.py
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"  # Only has copyright tag lines
    MISSING_COPYRIGHT = "missing_copyright"  # Only has license tag lines
    MISSING_BOTH = "missing_both"  # No tag and no plain copyright comment
    PLAIN_COPYRIGHT = "plain_copyright"  # No tag, but a plain copyright comment
    INVALID = "invalid"  # Empty value or invalid license expression
    UNDECODABLE = "undecodable"  # Not UTF-8; the header cannot be read or written


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


def is_empty_init(relative_path):
    """True for the zero-byte ``__init__.py`` files an empty package needs.

    Only those are exempt. Every other empty tracked source is a mistake that
    the policy must see: a truncated verbatim copy would otherwise keep a clean
    report while silently dropping the notice it is registered for.
    """
    return Path(relative_path).name == "__init__.py"


def spdx_header(prefix, newline):
    """The two-line header written into a file that has neither line."""
    return (
        f"{prefix} SPDX-License-Identifier: Apache-2.0{newline}"
        f"{prefix} SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project{newline}"
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


def read_lines(file_path):
    """Return (lines, bom) with line endings intact, or (None, False) if not UTF-8.

    A BOM is stripped and handed back so a rewrite can put it back where it was;
    decoding with the endings intact means a rewrite cannot convert CRLF to LF.
    """
    raw = Path(file_path).read_bytes()
    bom = raw.startswith(BOM.encode("utf-8"))
    if bom:
        raw = raw[len(BOM.encode("utf-8")) :]
    try:
        return raw.decode("utf-8").splitlines(keepends=True), bom
    except UnicodeDecodeError:
        return None, bom


def leading_block(lines):
    """The lines a header may live in: shebang, then blank and comment lines.

    Stops at the first line that is neither, so a module docstring, a string
    literal or a statement ends the block.
    """
    block = []
    for index, line in enumerate(lines):
        text = line.rstrip("\r\n")
        if index == 0 and text.startswith("#!"):
            block.append(text)
            continue
        if text.strip() == "" or text.lstrip().startswith(COMMENT_PREFIXES):
            block.append(text)
            continue
        break
    return block


def scan_header(file_path):
    """Return (status, licenses, copyrights) read from the leading comment block."""
    lines, _ = read_lines(file_path)
    if lines is None:
        return SPDXStatus.UNDECODABLE, [], []
    if not lines:
        return SPDXStatus.EMPTY, [], []

    licenses, copyrights, plain = [], [], False
    for line in leading_block(lines):
        tag = line.strip()
        for prefix in LICENSE_PREFIXES:
            if tag.startswith(prefix):
                licenses.append(tag[len(prefix) :].strip())
        for prefix in COPYRIGHT_PREFIXES:
            if tag.startswith(prefix):
                copyrights.append(tag[len(prefix) :].strip())
        if not plain and PLAIN_COPYRIGHT.match(tag):
            plain = True

    if any(not valid_license(value) for value in licenses) or any(
        not value for value in copyrights
    ):
        return SPDXStatus.INVALID, licenses, copyrights
    if licenses and copyrights:
        return SPDXStatus.COMPLETE, licenses, copyrights
    if licenses:
        return SPDXStatus.MISSING_COPYRIGHT, licenses, copyrights
    if copyrights:
        return SPDXStatus.MISSING_LICENSE, licenses, copyrights
    if plain:
        return SPDXStatus.PLAIN_COPYRIGHT, licenses, copyrights
    return SPDXStatus.MISSING_BOTH, licenses, copyrights


def check_spdx_header_status(file_path):
    """Check SPDX header status of the file"""
    return scan_header(file_path)[0]


def cookie_line(lines):
    """Index of the line a coding cookie occupies, or None.

    PEP 263 honours a cookie on line one or two only, after an optional shebang
    and after an ordinary comment. Inserting the header above such a cookie
    would push it out of that window and change how the file decodes.
    """
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    for candidate in (index, index + 1):
        if candidate < len(lines) and ENCODING_COOKIE.match(lines[candidate].rstrip("\r\n")):
            return candidate
    return None


def add_header(file_path):
    """Write the two-line header into a file that has neither line.

    Only MISSING_BOTH reaches here. The BOM, the encoding cookie, the shebang
    and the line endings are all left as they were: a coding cookie is only
    honoured on the first two lines, and a BOM that moved would break the file.
    """
    prefix = comment_prefix(file_path)
    if prefix is None:
        return
    lines, bom = read_lines(file_path)
    if lines is None:
        return

    newline = "\r\n" if any(line.endswith("\r\n") for line in lines) else "\n"
    cookie = cookie_line(lines)
    if cookie is not None:
        insert_at = cookie + 1
    else:
        # No cookie: the header still goes below a shebang, which must stay first.
        insert_at = 1 if lines and lines[0].startswith("#!") else 0

    text = "".join(lines[:insert_at]) + spdx_header(prefix, newline) + "".join(lines[insert_at:])
    data = text.encode("utf-8")
    Path(file_path).write_bytes((BOM.encode("utf-8") if bom else b"") + data)


def policy_errors(relative_path, licenses, copyrights, policy):
    """Return the reasons the file disagrees with the attribution policy."""
    copy = policy["verbatim_copies"].get(relative_path)
    if copy is None:
        errors = []
        if policy["copyright"] not in copyrights:
            errors.append(f"missing '{policy['copyright']}'")
        extra = [value for value in copyrights if value != policy["copyright"]]
        if extra:
            errors.append(
                f"copyright holder(s) not registered in verbatim_copies: {', '.join(extra)}"
            )
        return errors

    errors = []
    missing = [value for value in copy["copyrights"] if value not in copyrights]
    if missing:
        errors.append(f"missing registered holder(s): {', '.join(missing)}")
    extra = [value for value in copyrights if value not in copy["copyrights"]]
    if extra:
        errors.append(f"unregistered holder(s): {', '.join(extra)}")
    if any(
        canonicalize_license_expression(value) != canonicalize_license_expression(copy["license"])
        for value in licenses
        if valid_license(value)
    ):
        errors.append(f"license must be '{copy['license']}'")
    return errors


def tracked_sources():
    """Every tracked file whose suffix we check, as repository-relative paths.

    ``--full-name`` and an explicit cwd keep the paths repository-relative even
    when the hook is run from a subdirectory; otherwise the policy keys stop
    matching and every file outside that subdirectory goes unchecked.
    """
    out = subprocess.run(
        ["git", "ls-files", "--full-name", "-z"],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.decode()
    return [path for path in out.split("\0") if Path(path).suffix in SOURCE_SUFFIXES]


def check_all_tracked():
    """Read-only pass over every tracked source file; return the error list."""
    policy = json.loads(POLICY_PATH.read_text(encoding="UTF-8"))
    tracked = tracked_sources()
    errors = []

    registered = set(policy["verbatim_copies"])
    for path in sorted(registered - set(tracked)):
        errors.append(f"{path}: registered in spdx_policy.json but not tracked")

    for path in tracked:
        full = REPO_ROOT / path
        status, licenses, copyrights = scan_header(full)
        if status == SPDXStatus.EMPTY and is_empty_init(path):
            continue
        if status != SPDXStatus.COMPLETE:
            errors.append(f"{path}: {status.value}")
            continue
        for error in policy_errors(path, licenses, copyrights, policy):
            errors.append(f"{path}: {error}")
    return errors


def report(status, file_path):
    """Print the message for one file; return True when the run must not pass.

    A completed header fails the run as well: pre-commit passes the hook only
    when it changed nothing, so a rewritten file has to be staged again.
    """
    if status == SPDXStatus.MISSING_BOTH:
        print("The following files are missing the SPDX header:")
        print(f"  {file_path}")
        add_header(file_path)
        return True
    if status == SPDXStatus.MISSING_LICENSE:
        print("Copyright line without a license line; confirm the license and add it:")
    elif status == SPDXStatus.MISSING_COPYRIGHT:
        print(
            "License line without a copyright line; confirm who holds the copyright and add it "
            "(a license expression does not name the holder):"
        )
    elif status == SPDXStatus.PLAIN_COPYRIGHT:
        print(
            "Plain copyright comment with no SPDX header; confirm the holder and add the header "
            "(it may be an upstream copy):"
        )
    elif status == SPDXStatus.INVALID:
        print("Empty value or invalid SPDX license expression:")
    elif status == SPDXStatus.UNDECODABLE:
        print("Not UTF-8, so the header cannot be read or completed:")
    elif status == SPDXStatus.EMPTY:
        if is_empty_init(file_path):
            return False
        print("Empty source file that is not an __init__.py; restore it or add the header:")
    else:
        return False
    print(f"  {file_path}")
    return True


def main():
    """Main function"""
    if "--all-tracked" in sys.argv[1:]:
        errors = check_all_tracked()
        for error in errors:
            print(error)
        sys.exit(1 if errors else 0)

    files = sys.argv[1:]
    undecodable = [f for f in files if comment_prefix(f) is None]
    reportable = [f for f in files if comment_prefix(f) is not None]

    needs_a_person = False
    for file_path in reportable:
        status = scan_header(file_path)[0]
        needs_a_person |= report(status, file_path)

    if undecodable:
        print("Unknown comment syntax for this file type; add the SPDX header by hand:")
        for file_path in undecodable:
            print(f"  {file_path}")
        needs_a_person = True

    sys.exit(1 if needs_a_person else 0)


if __name__ == "__main__":
    main()
