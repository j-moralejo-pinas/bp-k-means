#!/bin/bash
# Pin all ci-cd-python workflow references to the latest major release tag.
# Replaces @main (or any existing @<ref>) with @<tag> so the project stays
# locked to the ci-cd-python version it was initialised with.
#
# Usage: setup-cicd-version.sh [ci_cd_repo]
#   ci_cd_repo  Owner/repo of the ci-cd source (default: j-moralejo-pinas/ci-cd-python)

set -euo pipefail

CI_CD_REPO="${1:-j-moralejo-pinas/ci-cd-python}"
WORKFLOW_DIR=".github/workflows"

if [[ ! -d "$WORKFLOW_DIR" ]]; then
    echo "  ⚠ Warning: $WORKFLOW_DIR not found, skipping ci-cd version pinning"
    exit 0
fi

echo "  Fetching latest tag for $CI_CD_REPO..."
CI_CD_TAG=$(gh api "repos/$CI_CD_REPO/tags" --jq '[.[].name | select(test("^v[0-9]+$"))] | first')

if [[ -z "$CI_CD_TAG" ]]; then
    echo "  ⚠ Warning: Could not retrieve a release tag for $CI_CD_REPO, skipping version pinning"
    exit 0
fi

echo "  Pinning ci-cd-python to tag $CI_CD_TAG"

find "$WORKFLOW_DIR" -name "*.yml" -type f | while IFS= read -r file; do
    if grep -q "$CI_CD_REPO" "$file"; then
        perl -pi -e "s#(\Q$CI_CD_REPO\E/(?:actions|\.github/workflows)/[^\s'\"]+@)[^\s'\"]+#\${1}$CI_CD_TAG#g" "$file"
        echo "  ✓ Pinned $file"
    fi
done

echo "  ✓ ci-cd-python version pinned to $CI_CD_TAG"
