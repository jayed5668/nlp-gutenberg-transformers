#!/usr/bin/env bash
# Create GitHub repo and push all step commits.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO_NAME="${1:-nlp-gutenberg-transformers}"
GITHUB_USER="${2:-jayed5668}"

GH="$(command -v gh || echo /opt/homebrew/bin/gh)"

if ! "$GH" auth status &>/dev/null; then
  echo "GitHub CLI is not logged in. Run:"
  echo "  $GH auth login"
  exit 1
fi

if ! git rev-parse --git-dir &>/dev/null; then
  echo "Run from initialized project (git init already done)."
  exit 1
fi

# Create remote repo if missing
if ! git remote get-url origin &>/dev/null; then
  "$GH" repo create "$REPO_NAME" --public --source=. --remote=origin --description "Assignment 3: NLP with Transformers on Project Gutenberg catalog"
  echo "Created https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
  echo "Remote origin already set."
fi

git push -u origin main
echo "Pushed all commits to origin/main."
