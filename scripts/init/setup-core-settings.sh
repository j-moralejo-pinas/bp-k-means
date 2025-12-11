#!/bin/bash
# Setup core repository settings
# Usage: setup-core-settings.sh <repo_slug>

set -euo pipefail

REPO_SLUG="${1:?Repository slug is required}"

echo "Setting up core repository settings for $REPO_SLUG..."

BODY=$(jq -n \
  '{has_wiki:false,
    allow_auto_merge:true,
    allow_update_branch:true,
    delete_branch_on_merge:true}')

gh api --method PATCH -H "Accept: application/vnd.github+json" \
  "/repos/$REPO_SLUG" \
  --input <(printf '%s' "$BODY")

echo "✓ Core settings updated"
