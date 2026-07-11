#!/usr/bin/env bash
# Portable launcher: resolves everything relative to this script, so it works
# from any checkout location without hardcoded paths.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/.venv/bin/python" "$here/server.py" "$@"
