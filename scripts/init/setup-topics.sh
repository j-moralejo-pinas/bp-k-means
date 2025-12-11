#!/bin/bash
# Setup repository topics
# Usage: setup-topics.sh <repo_slug> <topics>
# Example: setup-topics.sh owner/repo "python automation"

set -euo pipefail

REPO_SLUG="${1:?Repository slug is required}"
REPO_TOPICS="${2:-}"

echo "Setting up topics for $REPO_SLUG..."

# Convert space-separated topics to JSON array
if [ -n "${REPO_TOPICS:-}" ]; then
  REPO_TOPICS_JSON=$(echo "$REPO_TOPICS" | tr ' ' '\n' | jq -R . | jq -s .)
  echo "Topics to set: $REPO_TOPICS_JSON"
else
  REPO_TOPICS_JSON='[]'
  echo "No topics specified, setting empty array"
fi

gh api --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "/repos/$REPO_SLUG/topics" \
  --input <(jq -n --argjson names "$REPO_TOPICS_JSON" '{names:$names}')

echo "✓ Topics updated"
