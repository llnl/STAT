#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# run COMMAND with a temporary loopback SSH service in the prepared CI image.
# the CI workflow uses this helper as a Docker entrypoint.
#
# workflow:
#  1. Create private keys and start sshd as the current ordinary user.
#  2. Wait for SSH readiness and export the integration runner's SSH settings.
#  3. Run COMMAND, preserving its exit status; stop sshd and remove keys on exit.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: with-loopback-ssh.sh COMMAND [ARG...]" >&2
  exit 2
fi

test_user=$(id -un)
if [ "$(id -u)" -eq 0 ]; then
  echo "Run with-loopback-ssh.sh as the ordinary test user." >&2
  exit 2
fi
work=$(mktemp -d /tmp/stat-ssh.XXXXXX)
sshd_pid=

# run on success or failure. Only this script's sshd and temporary directory are
# removed; graphs and workload logs are retained for diagnosing a failed check.
cleanup() {
  if [ -n "$sshd_pid" ]; then
    kill "$sshd_pid" 2>/dev/null || true
    wait "$sshd_pid" 2>/dev/null || true
  fi
  if [ -f "$work/sshd.log" ]; then
    cat "$work/sshd.log" >&2
  fi
  rm -rf "$work"
}
trap cleanup EXIT

# generate per-run keys without using the account's normal SSH files.
ssh-keygen -q -t ed25519 -N '' -f "$work/host"
ssh-keygen -q -t ed25519 -N '' -f "$work/client"
cp "$work/client.pub" "$work/authorized_keys"

# use the image's loopback/key-only policy.
/usr/sbin/sshd -D -e -f /opt/stat-ci/sshd_config \
  -h "$work/host" -o "AuthorizedKeysFile=$work/authorized_keys" \
  -o "PidFile=$work/sshd.pid" -o "AllowUsers=$test_user" > "$work/sshd.log" 2>&1 &
sshd_pid=$!

# poll for readiness rather than relying on a fixed startup delay.
for attempt in {1..25}; do
  if ssh-keyscan -t ed25519 -T 1 -p 2222 127.0.0.1 > "$work/known_hosts" 2>/dev/null; then
    break
  fi
  kill -0 "$sshd_pid"
  sleep 0.2
done
[ -s "$work/known_hosts" ]

export STAT_TEST_REMOTE_LOGIN="$test_user@127.0.0.1"

# use port 2222 and temporary keys instead of SSH defaults, and disable interactive prompts.
export STAT_TEST_REMOTE_SSH_OPTIONS="-p 2222 -i $work/client -o BatchMode=yes -o UserKnownHostsFile=$work/known_hosts"

# keep the wrapper alive so its EXIT trap runs after the command completes.
"$@"
