#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect STAT samples over SSH for the shared integration runner.

The runner starts the workload and passes a local PID for MPI or Slurm, or an
application job ID for Flux. For a nested Flux allocation, the SSH server can
enter that allocation before attaching. This helper saves the DOT graphs and
attach/detach responses so the runner can validate the results.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from remote.stat_client_connection import StatRemoteClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int)
    target.add_argument("--flux-job", help="application job ID in the selected Flux instance")

    parser.add_argument("--prefix", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--backend", choices=("dyninst", "pystack"), required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--with-children", action="store_true")
    parser.add_argument("--serial", action="store_true", help="attach directly to the given process PID")
    parser.add_argument("--function-only", action="store_true")
    parser.add_argument("--threads", action="store_true")
    parser.add_argument("--results", type=Path, required=True)

    args = parser.parse_args()
    if (args.pid is not None and args.pid < 1) or args.samples < 1 or args.interval < 0:
        parser.error("PID/sample count must be positive and interval nonnegative")

    # require a fresh directory before opening an SSH connection.
    args.results.mkdir(parents=True, exist_ok=False)

    command = os.environ.get("STAT_TEST_REMOTE_COMMAND") or StatRemoteClient.command_for_prefix(args.prefix)
    ssh_options = shlex.split(os.environ.get("STAT_TEST_REMOTE_SSH_OPTIONS", ""))
    allocation = os.environ.get("STAT_TEST_FLUX_ALLOCATION") if args.flux_job else None
    client = StatRemoteClient.ssh(
        os.environ["STAT_TEST_REMOTE_LOGIN"], command, ssh_options, flux_allocation=allocation or None)
    client.request_timeout = 90
    session_id = None

    try:
        health = client.request("health")
        if args.flux_job:
            target_params = {"mode": "jobid", "jobid": args.flux_job, "scheduler": "flux"}
        else:
            # PID attach requires the SSH server to share the runner's host and PID namespace.
            if health["hostname"] != socket.gethostname():
                raise RuntimeError("Remote server must run on the workload launcher's host/PID namespace")

            if args.serial:
                target_params = {"mode": "serial", "processes": [str(args.pid)]}
            else:
                target_params = {"mode": "attach", "pid": args.pid}

        attached = client.request("sessions.attach", dict(target_params, backend=args.backend))
        session_id = attached["session_id"]

        params = {
            "session_id": session_id,
            "native": args.native,
            "with_children": args.with_children,
            "sample_type": "function only" if args.function_only else "function and line",
            "with_threads": args.threads,
            "clear_on_sample": True,
        }

        # repeated sampling also returns a merged 3D graph.
        if args.samples > 1:
            params.update(num_traces=args.samples, trace_frequency_ms=int(args.interval * 1000))
            merged = client.request("sessions.sample_multiple", params)
            snapshots = merged["samples"]
            (args.results / "merged_remote.3D.dot").write_text(merged["dot_text"])
        else:
            snapshots = [client.request("sessions.sample", params)]

        for index, result in enumerate(snapshots):
            snapshot = args.results / ("%02d_remote.2D.dot" % index)
            snapshot.write_text(result["dot_text"])

        detached = client.request("sessions.detach", params)
        if detached["state"] != "detached":
            raise RuntimeError("Remote detach was not acknowledged")
        session_id = None

        (args.results / "session.json").write_text(
            json.dumps({"attach": attached, "detach": detached}, indent=2) + "\n"
        )
    finally:
        # retry detach if the session is still open, then close SSH even if detach fails.
        try:
            if session_id is not None:
                client.request("sessions.detach", {"session_id": session_id})
        finally:
            client.close()


if __name__ == "__main__":
    main()
