#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

# Download a GitHub repository at a ref through the REST API tarball endpoint,
# for hosts that can reach the GitHub API (or an API proxy) but not git.
#
#   tools/ci/fetch_github_source.sh OWNER/REPO REF DEST [--submodule PATH]... [--git-init]
#
#   REF          commit SHA, tag, branch or refs/pull/<number>/head
#   --submodule  also fetch the submodule at PATH at the commit that REF pins
#   --git-init   turn DEST into a Git repository with one commit, for tools
#                that need `git ls-files` or `git rev-parse`
#
#   GITHUB_API         API base URL (default https://api.github.com)
#   GITHUB_TOKEN_FILE  file with a token; GITHUB_TOKEN is used if set instead.
#                      A token raises the API rate limit; it is never passed on
#                      the command line. A request that fails with the token is
#                      retried without it, because some organizations reject
#                      certain tokens even for public repositories.
#
# DEST must not exist or must be empty.

set -euo pipefail

usage() { sed -n '5,20p' "$0" | sed 's/^# \{0,1\}//'; }
[ $# -ge 3 ] || { usage >&2; exit 2; }
repo=$1 ref=$2 dest=$3
shift 3
submodules=()
git_init=false
while [ $# -gt 0 ]; do
    case "$1" in
        --submodule) submodules+=("$2"); shift 2 ;;
        --git-init) git_init=true; shift ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ $repo =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo "Invalid repository: $repo" >&2; exit 2; }
if [ -e "$dest" ] && [ -n "$(ls -A "$dest")" ]; then
    echo "Destination is not empty: $dest" >&2
    exit 2
fi

api="${GITHUB_API:-https://api.github.com}"
api="${api%/}"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# Keep the token out of the process list: curl reads the header from a file.
headers="$work/headers"
anonymous="$work/anonymous"
(umask 077 && : > "$headers")
printf 'Accept: application/vnd.github+json\n' > "$anonymous"
token="${GITHUB_TOKEN:-}"
if [ -z "$token" ] && [ -n "${GITHUB_TOKEN_FILE:-}" ] && [ -s "$GITHUB_TOKEN_FILE" ]; then
    token=$(<"$GITHUB_TOKEN_FILE")
fi
if [ -n "$token" ]; then
    printf 'Authorization: Bearer %s\n' "$token" >> "$headers"
fi
printf 'Accept: application/vnd.github+json\n' >> "$headers"

fetch() {
    if curl -sSfL --retry 3 --retry-delay 5 --connect-timeout 30 -H @"$headers" "$@"; then
        return 0
    fi
    [ -n "$token" ] || return 1
    echo "Request with the token failed; retrying without it" >&2
    curl -sSfL --retry 3 --retry-delay 5 --connect-timeout 30 -H @"$anonymous" "$@"
}

echo "Fetching $repo@$ref" >&2
fetch -o "$work/source.tar.gz" "$api/repos/$repo/tarball/$ref"
mkdir -p "$dest"
tar -xzf "$work/source.tar.gz" -C "$dest" --strip-components=1

for path in "${submodules[@]}"; do
    info=$(fetch "$api/repos/$repo/contents/$path?ref=$ref")
    read -r sub_repo sub_sha < <(python3 -c '
import json, re, sys
info = json.load(sys.stdin)
if info.get("type") != "submodule":
    sys.exit(f"{sys.argv[1]} is not a submodule")
url = info["submodule_git_url"]
match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", url)
if not match:
    sys.exit("Unsupported submodule URL: " + url)
print(match.group(1), info["sha"])
' "$path" <<< "$info")
    rmdir "$dest/$path" 2>/dev/null || true
    GITHUB_TOKEN="$token" GITHUB_API="$api" "$0" "$sub_repo" "$sub_sha" "$dest/$path"
done

if [ "$git_init" = true ]; then
    git -C "$dest" init -q
    git -C "$dest" add -A
    git -C "$dest" -c user.name=vllm-gr-ci -c user.email=ci@localhost \
        commit -q --no-verify -m "Source of $repo at $ref"
fi
echo "Fetched $repo@$ref into $dest" >&2
