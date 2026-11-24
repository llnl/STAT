#!/usr/bin/env bash
# Install the standalone STAT GUI and graphics dependencies on macOS or Linux.
#
# Use a local STAT checkout to create a protected Pixi workspace
# under --location (default: STAT_HOME or ~/stat-gui). Prepare GTK schemas and
# Graphviz plugins, install the Python client, and print commands to open it.
#
# Installation checks need no display; launching the GUI requires a desktop.
# This helper installs desktop components only, without building native STAT.
# Run with --help for arguments; see README.md for prerequisites.

set -euo pipefail

location="${STAT_HOME:-$HOME/stat-gui}"
source_location=""

usage() {
    cat <<'EOF'
Usage: install.sh --source DIRECTORY [--location DIRECTORY]

Options:
  --location DIRECTORY  Installation root (default: $STAT_HOME or ~/stat-gui)
  --source DIRECTORY    Local STAT repository directory (required)
  -h, --help            Show this help message
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --location)
            [ "$#" -ge 2 ] || die '--location requires a directory'
            location="$2"
            shift 2
            ;;
        --source)
            [ "$#" -ge 2 ] || die '--source requires a local repository directory'
            source_location="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

# Expand quoted home-relative paths before validating the checkout.
case "$location" in
    '~/'*) location="$HOME/${location#\~/}" ;;
esac

[ -n "$source_location" ] || die '--source is required'
case "$source_location" in
    '~/'*) source_location="$HOME/${source_location#\~/}" ;;
esac

# Resolve the local checkout before creating the installation directory.
[ -d "$source_location" ] || die "--source must be a local STAT checkout: $source_location"
stat_dir="$(cd "$source_location" && pwd)"

mkdir -p "$location"
location="$(cd "$location" && pwd)"
pixi_dir="$location/pixi"

# Protect existing environments from accidental replacement.
[ ! -e "$pixi_dir/pixi.toml" ] && [ ! -e "$pixi_dir/pyproject.toml" ] && [ ! -e "$pixi_dir/.pixi" ] ||
    die "Pixi workspace already exists: $pixi_dir"

# Keep installer-managed tools and caches under the selected location.
export PIXI_HOME="$pixi_dir"
export PIXI_CACHE_DIR="$location/cache/pixi"
export XDG_CACHE_HOME="$location/cache"
export PATH="$PIXI_HOME/bin:$PATH"
# Reuse an available Pixi executable, or install one locally.
if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | PIXI_BIN_DIR="$PIXI_HOME/bin" PIXI_NO_PATH_UPDATE=1 sh
fi

command -v pixi >/dev/null 2>&1 || die 'Pixi installation did not place pixi on PATH'

mkdir -p "$pixi_dir"
cd "$pixi_dir"

# Create the desktop environment from the checkout's dependency manifest.
cp "$stat_dir/scripts/gui/pixi.toml" "$pixi_dir/pixi.toml"
pixi install

# Prepare graphics resources explicitly because package post-link hooks may be skipped.
pixi run bash -c 'glib-compile-schemas "$CONDA_PREFIX/share/glib-2.0/schemas"'
pixi run dot -c
pixi run python -c 'import gi; gi.require_version("Gtk", "3.0"); from gi.repository import Gtk; print("GTK", Gtk.get_major_version(), Gtk.get_minor_version())'

# Install the GUI using Pixi's dependencies and check startup without a display.
pixi run python -m pip install --no-deps "$stat_dir/scripts/gui"
pixi run stat-gui --help

# Print launch commands for use in a graphical desktop session.
printf 'STAT GUI installed in %s\n' "$location"
printf '\nIf Pixi is not on PATH in a new terminal:\n'
printf '  export PATH="%s:$PATH"\n' "$(dirname "$(command -v pixi)")"
printf '\nTo open the remote GUI:\n'
printf '  pixi run --manifest-path "%s/pixi.toml" stat-gui\n' "$pixi_dir"
printf '\nTo open a saved graph:\n'
printf '  pixi run --manifest-path "%s/pixi.toml" stat-view /path/to/graph.dot\n' "$pixi_dir"
