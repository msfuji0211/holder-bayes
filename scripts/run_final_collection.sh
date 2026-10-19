#!/usr/bin/env bash
# Invoke with bash from bash or zsh. Paths remain relative to the caller's cwd.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$script_dir/run_final_collection.py" "$@"
