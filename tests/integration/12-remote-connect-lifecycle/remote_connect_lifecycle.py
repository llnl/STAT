#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the lifecycle of an MPI job controlled by a remote STAT session.

Over one SSH connection, launch and attach to the two-rank MPI ring from test
01. Verify that STAT rejects a second attachment and that the active session
can still pause and resume the job, collect two stack snapshots, and retrieve
the source file referenced by a sample.

Leave the job paused after the second snapshot to test whether ending the
session releases the ranks. CI repeats the test for explicit detach, server
shutdown, and client EOF. The ring deliberately stalls rank 1, so its progress
message confirms that the job resumed after each kind of session teardown.
"""

import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

INTEGRATION_DIR = Path(__file__).resolve().parents[1]
MPI_RING_DIR = INTEGRATION_DIR / "01-mpi-ring"

sys.path.insert(0, str(INTEGRATION_DIR.parents[1] / "scripts"))
from remote.stat_client_connection import StatRemoteClient, StatRemoteClientError


def wait_for_marker(log_path, marker, interval):
    """Wait up to 60 seconds for the workload to report a change of state."""
    deadline = time.monotonic() + 60
    while marker not in log_path.read_text():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for {marker!r} in {log_path}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path, help="compiled MPI ring executable")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for logs and graphs")
    parser.add_argument("--cleanup", choices=["detach", "shutdown", "eof"], default="detach",
                        help="how to release the paused ranks (default: detach)")
    args = parser.parse_args()

    login = os.environ.get("STAT_TEST_REMOTE_LOGIN")
    command = os.environ.get("STAT_TEST_REMOTE_COMMAND")

    if not login or not command:
        parser.error("set STAT_TEST_REMOTE_LOGIN and STAT_TEST_REMOTE_COMMAND")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    graphs = args.output_dir / "stat_results"
    graphs.mkdir()
    workload_log = args.output_dir / "workload.log"
    workload_log.touch()

    client = StatRemoteClient.ssh(
        login, command, shlex.split(os.environ.get("STAT_TEST_REMOTE_SSH_OPTIONS", "")),
    )
    client.request_timeout = 90
    session_id = None

    summary = {"cleanup": args.cleanup}
    log_path = args.output_dir / "application.log"

    with log_path.open("w", buffering=1) as log, contextlib.redirect_stderr(log):
        try:
            # check 1: confirm the server responds and advertises Dyninst.
            summary["health"] = client.request("health")
            assert "dyninst" in summary["health"]["supported_backends"], "Server lacks Dyninst"

            # check 2: reject an invalid backend while the server is idle.
            try:
                client.request("sessions.attach", {"backend": "invalid-backend"})
            except StatRemoteClientError as exc:
                assert exc.error["code"] == "invalid_backend", exc.error
            else:
                raise AssertionError("Server accepted an invalid backend")

            # check 3: launch and attach to the MPI job.
            # exec preserves the rank PID and redirects workload output to the shared log.
            # use orterun until CI includes the mpirun symlink fix: https://github.com/llnl/LaunchMON/pull/76
            argv = [
                "/usr/bin/orterun", "--oversubscribe", "--bind-to", "none", "-n", "2",
                "bash", "-c", 'exec "$@" >> %s 2>&1' % shlex.quote(str(workload_log)),
                "stat-ssh-workload", str(args.workload.resolve()), "30",
            ]
            launch = {"mode": "launch", "argv": argv, "backend": "dyninst"}
            attached = client.request("sessions.attach", launch)
            session_id = attached["session_id"]

            # check 4: verify health and reject a second attachment without replacing the active session.
            # native frontends share graph state, so the original session must remain usable.
            assert "dyninst" in client.request("health")["supported_backends"]
            for rejected in (launch, {"backend": "invalid-backend"}):
                try:
                    client.request("sessions.attach", rejected)
                except StatRemoteClientError as exc:
                    assert exc.error["code"] == "session_busy", exc.error
                    assert exc.error["context"]["session_id"] == session_id, exc.error
                else:
                    raise AssertionError("Server accepted a second live session")

            params = {"session_id": session_id, "backend": "dyninst",
                      "sample_type": "function and line", "clear_on_sample": True}
            assert client.request("sessions.status", params)["state"] == "attached"

            # check 5: resume the job and wait for rank 1 to reach its sampling region.
            client.request("sessions.resume", params)
            wait_for_marker(workload_log, "MPI task 1 of 2 stalling", interval=0.5)

            # check 6: pause and sample twice, resuming briefly between snapshots.
            # leave the job paused after the second snapshot so cleanup must release it.
            for index in range(2):
                client.request("sessions.pause", params)
                sample = client.request("sessions.sample", params)
                (graphs / ("%02d_ssh.2D.dot" % index)).write_text(sample["dot_text"])
                if index == 0:
                    client.request("sessions.resume", params)
                    time.sleep(0.1)

            # check 7: retrieve the sampled source and compare its bytes with the original file.
            source_path = MPI_RING_DIR / "mpi_ringtopo.c"
            source = client.request("sources.get", dict(params, source=str(source_path)))
            assert base64.b64decode(source["content_b64"]) == source_path.read_bytes()

            # check 8: end the session through detach, server shutdown, or client EOF.
            if args.cleanup == "detach":
                summary["detach"] = client.request("sessions.detach", params)
                assert summary["detach"]["state"] == "detached"
                session_id = None
                try:
                    client.request("sessions.status", params)
                except StatRemoteClientError as exc:
                    assert exc.error["code"] == "unknown_session", exc.error
                else:
                    raise AssertionError("Detached session remains registered")
            else:
                proc = client.proc
                if args.cleanup == "shutdown":
                    summary["shutdown"] = client.request("server.shutdown")
                    assert summary["shutdown"]["shutdown"]
                else:
                    # an EOF reaches the server's stdio loop while targets are paused.
                    proc.stdin.close()
                assert proc.wait(timeout=30) == 0, "Server cleanup failed"
                session_id = None

            # check 9: confirm that rank 1 made progress after the session ended.
            wait_for_marker(workload_log, "MPI task 1 of 2 proceeding", interval=0.1)
        finally:
            # on assertion/request failure, still try normal detach before
            # closing the transport. Keep native ownership until detach succeeds.
            try:
                if session_id is not None:
                    client.request("sessions.detach", {"session_id": session_id})
            finally:
                client.close()

    (args.output_dir / "session.json").write_text(json.dumps(summary, indent=2) + "\n")

    # check 10: validate workload output and both graphs against the MPI ring expectations.
    validator = [sys.executable, str(INTEGRATION_DIR / "lib/validate_stat.py"),
                 "--checks", str(MPI_RING_DIR / "README.md"), "--check-set", "default",
                 "--backend", "dyninst", "--world-size", "2"]
    subprocess.run(validator + ["--output", str(workload_log)], check=True, timeout=90)
    subprocess.run(validator + ["--results", str(graphs), "--sample-count", "2"],
                   check=True, timeout=90)

    print("PASS: scenario 12 remote connect lifecycle over SSH, including pause/resume, source read and %s" % args.cleanup)


if __name__ == "__main__":
    main()
