#!/bin/bash
# Get repository information and output repo name, description, and slug
# Usage: get-repo-info.sh
# Can be sourced to set variables or called to output to GITHUB_OUTPUT

set -euo pipefail

REPO_INFO=$(gh api "/repos/$GITHUB_REPOSITORY" -H "Accept: application/vnd.github+json")
REPO_NAME=$(echo "$REPO_INFO" | jq -r '.name')
REPO_DESC=$(echo "$REPO_INFO" | jq -r '.description // ""')
REPO_SLUG=$(echo "$REPO_INFO" | jq -r '.full_name')

echo "Repository name: $REPO_NAME"
echo "Repository description: $REPO_DESC"
echo "Repository slug: $REPO_SLUG"

# Export variables for use in parent shell
export REPO_NAME REPO_DESC REPO_SLUG

# Output for GitHub Actions (only if GITHUB_OUTPUT is defined)
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "REPO_SLUG=$REPO_SLUG" >> "$GITHUB_OUTPUT"
fi
