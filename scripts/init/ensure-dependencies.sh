#!/bin/bash
# Ensure required dependencies are available
# Usage: ensure-dependencies.sh

set -euo pipefail

# Helper function to check and install a dependency
ensure_dependency() {
  local cmd="$1"
  local package="${2:-$cmd}"  # Use provided package name or default to command name
  local description="${3:-$cmd}"  # Use provided description or default to command name

  if ! command -v "$cmd" >/dev/null; then
    echo "Installing $description..."
    sudo apt-get install -y "$package"
  else
    echo "✓ $description is available"
  fi
}

echo "Checking required dependencies..."

# Update package manager
sudo apt-get update -y

# Ensure all required dependencies
ensure_dependency "gh" "gh" "gh (GitHub CLI)"
ensure_dependency "jq" "jq" "jq"
ensure_dependency "git" "git" "git"
ensure_dependency "curl" "curl" "curl"

echo "✓ All dependencies are available"
