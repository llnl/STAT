#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# exercise STAT remote session control using an already available SSH service.
# usage: smoke-remote-connect.sh SOURCE PREFIX RESULTS
# run under /opt/stat-ci/with-loopback-ssh.sh in CI or set STAT_TEST_REMOTE_* yourself.
#
# workflow:
#   1. Compile the fixed two-rank workload using system Open MPI.
#   2. Check attach/sample/source reading and detach, shutdown and EOF cleanup.
#   3. Run the shared remote matrix with ordinary and Enroot-contained ranks.
#
# source, executable and results must be visible at the same paths through SSH.
# each lifecycle run retains workload output, graphs and session acknowledgements.

set -euo pipefail

# select matching system MPI, compiler and Python before running any tests.
export PATH="/usr/bin:/bin:$PATH"
export OMPI_CC=gcc

if [ "$#" -ne 3 ]; then
  echo "Usage: smoke-remote-connect.sh SOURCE PREFIX RESULTS" >&2
  exit 2
fi

source_dir=$1
prefix=$2
results=$3

if [ -z "${STAT_TEST_REMOTE_LOGIN:-}" ]; then
  echo "Run through with-loopback-ssh.sh or set STAT_TEST_REMOTE_LOGIN." >&2
  exit 2
fi

export STAT_TEST_REMOTE_COMMAND="${STAT_TEST_REMOTE_COMMAND:-$prefix/bin/stat-gui-server}"
mkdir -p "$results"

# Step 1: use the same system MPI for compilation and launch. Mixing it with an MPI runtime
# from Conda/Pixi can prevent LaunchMON from controlling the expected launcher.
mpicc -g -O0 \
  "$source_dir/tests/integration/01-mpi-ring/mpi_ringtopo.c" -o "$results/mpi_ringtopo"

# Step 2: Each run ends while targets are paused. The Python smoke verifies that
# explicit detach, server.shutdown, or closing protocol stdin releases them.
# separate result directories prevent one run's graphs/logs satisfying another.
for mode in detach shutdown eof; do
  python3 "$source_dir/tests/integration/12-remote-connect-lifecycle/remote_connect_lifecycle.py" \
    --workload "$results/mpi_ringtopo" --cleanup "$mode" --output-dir "$results/$mode"
done

# step 3 runs the same SSH tests in two rank modes.
# plain runs ranks directly in Docker. enroot runs them inside Enroot within Docker.
for rank_mode in plain enroot; do
  rank_options=()
  if [ "$rank_mode" = enroot ]; then
    rank_options+=(--enroot)
  fi
  bash "$source_dir/tests/integration/run-tests.sh" --remote "${rank_options[@]}" 01 01:gdb 02 03 04:serial 05:attach 06 07
done

echo 'PASS: SSH remote control and shared integration matrix'
