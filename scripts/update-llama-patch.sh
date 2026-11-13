#!/usr/bin/env bash
#
# Regenerate patches/llama.cpp-vulkan-watchdog.patch from the current working
# tree of externals/llama.cpp.
#
# Usage:
#   ./scripts/update-llama-patch.sh
#
# Run this after editing files under externals/llama.cpp, then commit the
# updated patch file so the modifications survive a fresh clone of this
# repository (CMake applies the patch automatically and idempotently).
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/externals/llama.cpp"
out="$root/patches/llama.cpp-vulkan-watchdog.patch"

if [[ ! -d "$src/.git" ]]; then
  echo "error: $src is not a git checkout" >&2
  exit 1
fi

if [[ -z "$(git -C "$src" status --porcelain)" ]]; then
  echo "error: $src has no local modifications - nothing to write" >&2
  exit 1
fi

git -C "$src" diff > "$out"

if [[ ! -s "$out" ]]; then
  echo "error: generated patch is empty" >&2
  exit 1
fi

echo "wrote $out ($(wc -l < "$out") lines)"
echo
echo "Files covered by the patch:"
git -C "$src" diff --stat
echo
echo "Remember to commit the updated patch file."
