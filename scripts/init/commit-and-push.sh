#!/bin/bash
# Commit and push changes to git
# Usage: commit-and-push.sh [commit_message] [branch_name]
#
# Arguments:
#   commit_message  - The commit message (use empty string "" to skip commit)
#   branch_name     - Target branch name (default: current branch)

set -euo pipefail

COMMIT_MESSAGE="${1:-}"
BRANCH_NAME="${2:-$(git branch --show-current)}"

echo "Configuring git..."
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

# Commit changes if message is provided
if [[ -n "$COMMIT_MESSAGE" ]]; then
  echo "Checking for changes..."
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Found changes, committing..."
    git add .
    git commit -m "$COMMIT_MESSAGE"
  else
    echo "✓ No changes to commit"
  fi
fi

# Push to the specified branch
echo "Pushing to branch '$BRANCH_NAME'..."
# Check if branch exists upstream
if git rev-parse --verify "origin/$BRANCH_NAME" >/dev/null 2>&1; then
  # Branch exists upstream, push normally
  git push origin "$BRANCH_NAME"
else
  # Branch doesn't exist upstream, push with -u to set upstream
  git push -u origin "$BRANCH_NAME"
fi
echo "✓ Pushed to '$BRANCH_NAME'"
