#!/bin/bash
# Local script to configure repository (mirrors GitHub Actions workflow)
# Usage: local-configure-repo.sh <python_version> [python_version_max] [repo_topics] [workflow]
#
# Prerequisites:
#   - gh CLI is installed and authenticated
#   - You are in the git project root directory
#   - All setup scripts are available in ./scripts/init/

set -euo pipefail

# Validate inputs
PYTHON_VERSION="${1:?Python version is required}"
PYTHON_VERSION_MAX="${2:-}"
REPO_TOPICS="${3:-}"
WORKFLOW="${4:?Workflow is required}"

# Get GITHUB_REPOSITORY from git remote URL
# Convert from git@github.com:owner/repo.git or https://github.com/owner/repo.git to owner/repo
REPO_URL=$(git config --get remote.origin.url)
if [[ $REPO_URL =~ git@github.com:([^/]+)/(.+)\.git$ ]]; then
  GITHUB_REPOSITORY="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
elif [[ $REPO_URL =~ https://github.com/([^/]+)/(.+)\.git$ ]]; then
  GITHUB_REPOSITORY="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
else
  echo "Error: Could not determine repository from git remote"
  exit 1
fi

echo "=========================================="
echo "Local Repository Configuration Script"
echo "=========================================="
echo "Repository: $GITHUB_REPOSITORY"
echo "Python version: $PYTHON_VERSION"
echo "Python max version: ${PYTHON_VERSION_MAX:-'Not specified'}"
echo "Topics: ${REPO_TOPICS:-'Not specified'}"
echo "Workflow: $WORKFLOW"
echo "=========================================="
echo ""

# Check if gh is available
if ! command -v gh >/dev/null; then
  echo "Error: gh CLI is not installed or not in PATH"
  exit 1
fi

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh CLI is not authenticated. Please run 'gh auth login' first"
  exit 1
fi

echo "✓ gh CLI is authenticated"
echo ""

# Ensure dependencies
echo "Ensuring dependencies..."
chmod +x ./scripts/init/ensure-dependencies.sh
./scripts/init/ensure-dependencies.sh
echo ""

# Step 1: Get current repo info and initialize project
echo "Step 1: Get repo info and initialize project..."
export GITHUB_REPOSITORY
chmod +x ./scripts/init/configure-project.sh
./scripts/init/configure-project.sh \
  "$PYTHON_VERSION" \
  "$PYTHON_VERSION_MAX" \
  "$REPO_TOPICS" \
  "$WORKFLOW"

echo "✓ Step 1 complete"
echo ""

# Step 2: Core settings
echo "Step 2: Core repo settings..."
chmod +x ./scripts/init/setup-core-settings.sh
./scripts/init/setup-core-settings.sh "$GITHUB_REPOSITORY"
echo "✓ Step 2 complete"
echo ""

# Step 3: Topics
echo "Step 3: Set repository topics..."
chmod +x ./scripts/init/setup-topics.sh
./scripts/init/setup-topics.sh "$GITHUB_REPOSITORY" "$REPO_TOPICS"
echo "✓ Step 3 complete"
echo ""

# Step 4: Workflow token permissions
echo "Step 4: Set workflow token permissions..."
chmod +x ./scripts/init/setup-workflow-permissions.sh
./scripts/init/setup-workflow-permissions.sh "$GITHUB_REPOSITORY"
echo "✓ Step 4 complete"
echo ""

# Step 5: Create or update rulesets
echo "Step 5: Create or update rulesets..."
chmod +x ./scripts/init/setup-rulesets.sh
./scripts/init/setup-rulesets.sh "$GITHUB_REPOSITORY" "$WORKFLOW"
echo "✓ Step 5 complete"
echo ""

# Step 6: Create venv with uv and install optional dependencies
echo "Step 6: Create virtual environment and install optional dependencies..."
chmod +x ./scripts/init/setup-venv.sh
./scripts/init/setup-venv.sh
echo "✓ Step 6 complete"
echo ""

# Step 7: Remove init scripts (no longer needed after initialisation)
echo "Step 7: Removing init scripts..."
rm -rf ./scripts/init
if [[ -d "./scripts" ]] && [[ -z "$(ls -A ./scripts)" ]]; then
    rm -rf ./scripts
fi
rm -f ./.github/workflows/configure_repo.yml
git add -A scripts/ .github/workflows/configure_repo.yml
git commit -m "chore: remove init scripts after project initialisation"
git push
echo "✓ Step 7 complete"
echo ""

echo "=========================================="
echo "✓ Repository configuration completed!"
echo "=========================================="
