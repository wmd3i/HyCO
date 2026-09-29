#!/usr/bin/env bash
# Fetch the official COExpander code, apply the HyCO modifications
# (prefix conditioning, OP support, probe interfaces) and place the
# patched `co_expander` package next to the MIS/OP scripts.
#
# Usage (from mis_op/):  bash setup_coexpander.sh
set -euo pipefail

COMMIT=f77926950b8a7f20239eed748a4d8de9b14ea5ea
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"

git clone https://github.com/Thinklab-SJTU/COExpander.git "$TMP/COExpander"
cd "$TMP/COExpander"
git checkout "$COMMIT"
git apply "$HERE/coexpander_hyco.patch"

rm -rf "$HERE/co_expander"
cp -r co_expander "$HERE/co_expander"
rm -rf "$TMP"
echo "Patched co_expander installed to $HERE/co_expander"
