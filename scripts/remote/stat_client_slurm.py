# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Choose where to run stat-gui-server when attaching to a Slurm job.

Slurm can be queried from a login host whose CPU architecture differs from
the compute nodes. For example, an x86-64 login host can report a job
running on AArch64 nodes, where an AArch64 STAT installation can run.
The GUI uses SSH through the login host to start STAT on a suitable node
in the allocation and sends attach and sampling requests there.
"""

import shlex

from remote.stat_client_connection import StatRemoteClient
from remote.stat_server_slurm import SlurmJobDiscovery


def check_launcher(job, jobid, host, pid):
    """Check that Slurm still lists the launcher host and PID selected in the GUI."""
    if job['remote_host'] != host or pid not in job['pids']:
        raise RuntimeError(
            'Launcher %s:%s is no longer listed for job %s. Refresh Process List.'
            % (host, pid, jobid)
        )


def resolve_on_login(client, jobid, selected_launcher=None):
    """Resolve the Slurm job and return a connected STAT client for its launcher host.

    If a launcher was selected in the GUI, verify its host and PID against the job.
    If the connection fails and the launcher is outside the allocation, try the first
    allocated node. Report both errors if that attempt also fails.
    """

    def connect(host):
        """Start STAT over SSH through the login host and check its health reply."""
        candidate = StatRemoteClient.ssh(
            client.remote_login,
            client.remote_command,
            client.ssh_options,
            client.remote_prefix,
            host,
        )

        # The SSH client starts the server on its first health request.
        try:
            candidate.ensure_remote_server()
            return candidate
        except Exception:
            candidate.close()
            raise

    job = SlurmJobDiscovery(client._run_ssh).resolve(jobid)
    launcher = job['remote_host']
    if selected_launcher is not None:
        check_launcher(job, jobid, *selected_launcher)

    try:
        return connect(launcher)
    except Exception as launcher_error:
        nodelist = job['nodelist']
        if not nodelist or nodelist == launcher:
            raise

        # Look for a fallback only when the launcher is outside the allocation.
        result = client._run_ssh('scontrol show hostnames %s' % shlex.quote(nodelist))
        if result.returncode:
            detail = result.stderr.strip() or 'Could not expand Slurm node list'
            raise RuntimeError(
                f'Cannot start STAT for job {jobid} on {launcher}: {launcher_error}. {detail}'
            ) from launcher_error

        nodes = result.stdout.split()
        if not nodes or launcher in nodes:
            raise

        try:
            return connect(nodes[0])
        except Exception as node_error:
            raise RuntimeError(
                f'Cannot start STAT for job {jobid}. '
                f'{launcher}: {launcher_error}. {nodes[0]}: {node_error}'
            ) from node_error
