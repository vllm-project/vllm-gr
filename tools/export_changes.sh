#!/bin/bash
set -e

ORIGINAL_REF=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD)
REMOTE_BRANCH=$1
COMMIT_MESSAGE=$2

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Working tree has uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Configuration
UPSTREAM_REMOTE="upstream"
UPSTREAM_BRANCH="main"
UPSTREAM_REF="${UPSTREAM_REMOTE}/${UPSTREAM_BRANCH}"
EXPORT_BRANCH="export-branch"
PATCH_FILE=$(mktemp /tmp/changes.patch.XXXXXX)
ADDED_FILES_LIST=$(mktemp /tmp/added_files.txt.XXXXXX)
EXCLUDE_DIR="trc"

# Ensure we have the latest upstream ref
echo "Fetching latest changes from $UPSTREAM_REF..."
if ! git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"; then
    echo "Error: Failed to fetch $UPSTREAM_REMOTE $UPSTREAM_BRANCH."
    exit 1
fi

if ! git rev-parse --verify "$UPSTREAM_REF" >/dev/null 2>&1; then
    echo "Error: $UPSTREAM_REF not found. Please fetch upstream."
    exit 1
fi

echo "Starting export process..."

# 1. Merge upstream/main branch into current branch
echo "1. Merging with $UPSTREAM_REF..."
if ! git merge --no-edit "$UPSTREAM_REF"; then
    echo "Error: Failed to merge $UPSTREAM_REF into the current branch. Please resolve conflicts manually."
    exit 1
fi

# 2. Create a diff between current version and upstream/main
# We use HEAD to represent the current version.
# We exclude the 'trc' folder using pathspec magic ':!trc'.
# We use --binary to ensure binary files are handled correctly.
echo '2. Creating diff (excluding' "$EXCLUDE_DIR" ')...'
git diff --binary "$UPSTREAM_REF" HEAD -- . ":!$EXCLUDE_DIR" > "$PATCH_FILE"

# 3. Create a list of added files
# We filter for Added (A) files.
echo "3. Listing added files..."
git diff --name-only --diff-filter=A "$UPSTREAM_REF" HEAD -- . ":!$EXCLUDE_DIR" > "$ADDED_FILES_LIST"

# 4. Create an export branch based on upstream/main
echo "4. Creating export branch $EXPORT_BRANCH..."
git checkout -B "$EXPORT_BRANCH" "$UPSTREAM_REF"

# 5. Apply the diff created in stage 2
echo "5. Applying diff..."
if [ -s "$PATCH_FILE" ]; then
    git apply "$PATCH_FILE"
else
    echo "No changes detected in patch file."
fi

# 6. Add the files created at stage 3
echo "6. Adding added files..."
if [ -s "$ADDED_FILES_LIST" ]; then
    while IFS= read -r file; do
        if [ -e "$file" ]; then
            git add "$file"
            echo "  Staged: $file"
        fi
    done < "$ADDED_FILES_LIST"
else
    echo "No added files to stage."
fi

# Cleanup
rm -f "$PATCH_FILE" "$ADDED_FILES_LIST"

if [ -n "$REMOTE_BRANCH" ] && [ -n "$COMMIT_MESSAGE" ]; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Committing changes with message: $COMMIT_MESSAGE"
        git commit -a -m "$COMMIT_MESSAGE"
        echo "Pushing to origin/$REMOTE_BRANCH..."
        git push origin -f "HEAD:$REMOTE_BRANCH"
    else
        echo "No changes detected to commit."
    fi
    echo "Returning to original branch/commit: $ORIGINAL_REF"
    git checkout "$ORIGINAL_REF"
    echo "Done."
else
    if [ -n "$REMOTE_BRANCH" ] || [ -n "$COMMIT_MESSAGE" ]; then
        echo "Warning: Both remote branch and commit message must be specified to auto-commit, push, and return."
    fi
    echo "Done. Switched to $EXPORT_BRANCH."
fi
