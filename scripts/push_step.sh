#!/usr/bin/env bash
# Push current branch to GitHub (after setup_github.sh once).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git push origin main
echo "Pushed latest commit: $(git log -1 --oneline)"
