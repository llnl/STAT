# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Let users attach STAT to a running Flux application by job ID.

STAT needs the PID of a process that exposes the application's ranks. Start
`flux job attach --read-only --debug` for the given job ID, then give LaunchMON
that process's PID once its rank table is ready. Keep it alive until STAT detaches.
"""

import json
import re
import subprocess
import threading


# match the readiness event, for example: 1.234s: exec.shell.start {"taskmap": {...}}
# TODO: verify whether Flux exposes a more robust, machine-readable readiness signal.
FLUX_SHELL_START = re.compile(rb"^[ \t]*-?\d+(?:[.,]\d+)?s:[ \t]*exec\.shell\.start(?:\s|$)")
STDERR_READ_SIZE = 4096


def list_flux_jobs(user, recursive=True):
    """List running applications and the allocation to proxy into for each one."""
    command = ["flux", "jobs", "--json", "--filter=run", "--user", user]
    if recursive:
        command.extend(["--recursive", "--level=1"])

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "Flux job listing failed") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Could not list Flux jobs: %s" % exc) from exc

    def display_id(job):
        # flux-core 0.83 added the printable job ID used by the CLI and GUI.
        if "jobid" not in job:
            raise RuntimeError("Automatic Flux job discovery requires flux-core 0.83.0 or newer. "
                               "To attach directly, supply the allocation ID and application job ID.")
        return job["jobid"]

    def application(job, allocation=None):
        return {"jobid": display_id(job), "id": str(job["id"]), "name": job.get("name", ""),
                "ntasks": job.get("ntasks"), "nodelist": job.get("nodelist", ""), "allocation": allocation}

    rows = []
    for job in json.loads(result.stdout)["jobs"]:
        if not job.get("uri"):
            rows.append(application(job))
            continue

        # a job with a broker URI is an allocation. Attach to the applications inside it.
        allocation = display_id(job)
        for child in job.get("jobs", []):
            if not child.get("uri"):
                rows.append(application(child, allocation))

    return rows


class FluxDebugHelper:
    """Keep the selected job's debugger process alive until STAT has detached."""

    def __init__(self, jobid):
        self.jobid = jobid
        self.process = None

    @property
    def pid(self):
        return self.process.pid

    def start(self):
        ready = threading.Event()
        initialized = False
        last_line = ""

        # flux reports shell.start when STAT can read the rank table.
        # label application output so it is distinct from Flux events.
        self.process = subprocess.Popen(
            ["flux", "job", "attach", "--read-only", "--debug", "--show-exec", "--label-io", "--", self.jobid],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            start_new_session=True)

        def watch_and_drain_stderr():
            nonlocal initialized, last_line
            with self.process.stderr as stream:
                at_line_start = True
                for chunk in iter(lambda: stream.readline(STDERR_READ_SIZE), b""):
                    if initialized:
                        # discard later output so a full pipe cannot block Flux.
                        continue

                    last_line = chunk.decode(errors="replace").strip()[-512:]
                    if at_line_start and FLUX_SHELL_START.match(chunk):
                        initialized = True
                        ready.set()

                    # look for the ready message only at the start of a line.
                    at_line_start = chunk.endswith(b"\n")

            ready.set()  # wake start() even if Flux exits before it is ready.

        threading.Thread(target=watch_and_drain_stderr, daemon=True).start()
        if not ready.wait(30):
            raise RuntimeError("Timed out waiting for the Flux debugger for job %s" % self.jobid)

        if not initialized or self.process.poll() is not None:
            raise RuntimeError("Flux debugger exited for job %s: %s" % (self.jobid, last_line))

    def close(self):
        """Reap only our read-only helper, including on failed attach."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
