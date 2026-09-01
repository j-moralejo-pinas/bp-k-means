#!/bin/bash
# Configure and initialize project
# Usage: configure-project.sh <python_version> [python_version_max] [repo_topics] [workflow]

set -euo pipefail

PYTHON_VERSION="${1:?Python version is required}"
PYTHON_VERSION_MAX="${2:-}"
REPO_TOPICS="${3:-}"
WORKFLOW="${4:?Workflow is required}"

echo "Configuring project..."

# Get repository info using the dedicated script
chmod +x ./scripts/init/get-repo-info.sh
source ./scripts/init/get-repo-info.sh

echo "Python version: $PYTHON_VERSION"
echo "Python max version: ${PYTHON_VERSION_MAX:-'Not specified'}"
echo "Topics: ${REPO_TOPICS:-'Not specified'}"
echo "Workflow: ${WORKFLOW:-'Not specified'}"

# Initialize project metadata
chmod +x ./scripts/init/setup-project-metadata.sh
./scripts/init/setup-project-metadata.sh "$REPO_NAME" "$REPO_DESC" "$PYTHON_VERSION" "$PYTHON_VERSION_MAX" "$REPO_TOPICS" "$WORKFLOW"

# Pin ci-cd-python workflows to the latest released major version
chmod +x ./scripts/init/setup-cicd-version.sh
./scripts/init/setup-cicd-version.sh

# Commit and push changes
chmod +x ./scripts/init/commit-and-push.sh
./scripts/init/commit-and-push.sh "Initialize project with name '$REPO_NAME' and Python $PYTHON_VERSION"

# Create and push dev branch for gitflow workflow
if [[ "$WORKFLOW" == "gitflow" ]]; then
  echo "Creating dev branch for gitflow workflow..."
  git checkout -b dev
  ./scripts/init/commit-and-push.sh "" "dev"
  git checkout main
  echo "✓ Dev branch created and pushed"
fi
