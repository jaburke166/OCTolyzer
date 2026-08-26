#!/usr/bin/env bash
# Wrap the Nuitka-built OCTolyzer.app in a plain, unsigned .dmg with a
# drag-to-Applications layout, using only hdiutil (no paid tooling).
#
# Usage: build_dmg.sh <path/to/OCTolyzerGUI.app> <path/to/output.dmg> [volume-name]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <app-bundle> <output-dmg> [volume-name]" >&2
    exit 1
fi

app_bundle="$1"
output_dmg="$2"
volume_name="${3:-OCTolyzer}"

if [[ ! -d "$app_bundle" ]]; then
    echo "App bundle not found: $app_bundle" >&2
    exit 1
fi

staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT

cp -R "$app_bundle" "$staging_dir/"
ln -s /Applications "$staging_dir/Applications"

rm -f "$output_dmg"
mkdir -p "$(dirname "$output_dmg")"
hdiutil create -volname "$volume_name" -srcfolder "$staging_dir" -ov -format UDZO "$output_dmg"

echo "Built $output_dmg"
