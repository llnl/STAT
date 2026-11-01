#!/usr/bin/env bash
#
# Check pip installation and basic GUI startup using Ubuntu's GTK packages.
set -euo pipefail

source_dir="${1:?usage: smoke-gui.sh SOURCE_DIR}"
work_dir="$(mktemp -d /tmp/stat-ci-gui.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

# Keep Pixi and native STAT out of this standalone installation.
export PATH=/usr/bin:/bin
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH CONDA_PREFIX GSETTINGS_SCHEMA_DIR GDK_PIXBUF_MODULE_FILE

# Pip needs a writable source copy because CI mounts the checkout read-only.
cp -a "$source_dir/scripts" "$work_dir/"
cp "$source_dir/LICENSE" "$source_dir/STATlogo.gif" "$work_dir/"

/usr/bin/python3 -m venv --system-site-packages "$work_dir/venv"

export PATH="$work_dir/venv/bin:$PATH"
python -m pip install "$work_dir/scripts/gui"

# Each viewer should stay open until the timeout; an early exit is a failure.
for command in stat-gui stat-view; do
    status=0
    xvfb-run -a timeout --kill-after=5s 5 "$command" || status=$?
    if [ "$status" -ne 124 ]; then
        echo "$command exited early (status $status)" >&2
        exit 1
    fi
    echo "PASS: $command started and stayed running for five seconds"
done
