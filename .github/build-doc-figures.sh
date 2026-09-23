#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/notebooks/figures"

for source in "$repo_root"/docs/figures/*.typ; do
    name="$(basename "$source" .typ)"
    [ "$name" = style ] && continue
    typst compile --root "$repo_root" --format svg "$source" "$repo_root/notebooks/figures/$name.svg"
    printf '\n' >> "$repo_root/notebooks/figures/$name.svg"
done
