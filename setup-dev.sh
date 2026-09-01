#!/bin/bash
# Setup development environment
# This script automates the setup steps described in CONTRIBUTING.rst

set -e

echo "Setting up development environment..."

# 1. Allow direnv to manage your environment automatically
if command -v direnv >/dev/null; then
    echo "Configuring direnv..."
    direnv allow .
else
    echo "direnv not found. Skipping direnv setup."
    if command -v nix >/dev/null; then
        nix develop
    else
        echo "Install nix for environment management."
        exit 1
    fi
fi

# 2. Create and activate a virtual environment using uv
if ! command -v uv >/dev/null; then
    echo "Error: uv is not installed. This should not happen."
    exit 1
fi

echo "Creating virtual environment with uv..."
uv venv

# Activate the virtual environment for the script's session
# Note: This only affects the script execution. The user must activate it in their shell.
source .venv/bin/activate

# 3. Install the package in development mode
echo "Installing package in development mode with [dev,docs]..."
uv pip install -e ".[dev,docs]"

# 4. Set up pre-commit hooks
if command -v pre-commit >/dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit autoupdate
    pre-commit install
else
    echo "Error: pre-commit not found. This should not happen."
    exit 1
fi

echo ""
echo "Setup complete!"
