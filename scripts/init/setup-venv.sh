#!/bin/bash
# Setup virtual environment with uv and install optional dependencies
# Usage: setup-venv.sh
#
# Prerequisites:
#   - uv is installed and in PATH
#   - You are in the git project root directory

set -euo pipefail

# Check if uv is available
if ! command -v uv >/dev/null; then
  echo "Error: uv is not installed or not in PATH. This should not happen."
  exit 1
fi

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install all optional dependencies
echo "Installing optional dependencies..."
uv pip install -e ".[dev,docs]"

# Set up pre-commit hooks
if command -v pre-commit >/dev/null; then
    echo "Setting up pre-commit hooks..."
    # pre-commit autoupdate # We shouldn't use unlocked versions in the boilerplate
    pre-commit install
else
    echo "Error: pre-commit not found. This should not happen."
    exit 1
fi
