# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""List running Slurm jobs and find their srun launchers for STAT attachment.

For batch jobs, match srun processes on the batch host by their inherited job ID.
For interactive jobs, verify the launcher host and PID recorded for each running
workload step. STAT uses the launcher to discover the application ranks during attach.

Interactive launcher discovery is new and has been tested with srun inside an allocation.
TODO: test more complex workflows, such as allocations with concurrent workload steps,
and refine launcher selection where needed.
"""

import getpass
import os
import re
import shlex
import subprocess


# use explicit separators for job names with spaces. These field names have been stable since Slurm 14.11.
# TODO: revisit this format if a supported Slurm release changes the fields or output syntax.
SLURM_JOB_LIST_FORMAT = (
    "JobID:0|,Name:0|,UserName:0|,State:0|,TimeUsed:0|,"
    "NumNodes:0|,NodeList:0|,BatchFlag:0"
)

# the same user can have several jobs on one batch host, so report each srun's inherited Slurm job ID.
# capture ps before the loop so a failed process query reaches the caller.
BATCH_LAUNCHER_PROBE = r'''
process_list=$(ps ww -u "$(id -u)" -o pid=,ppid=,args=) || exit $?
printf '%s\n' "$process_list" |
while read -r pid ppid executable arguments; do
    case "$executable" in
        srun|*/srun) ;;
        *) continue ;;
    esac
    # split the NUL-separated environment into lines so sed can match either Slurm job ID variable.
    job=$(
        tr '\000' '\n' 2>/dev/null < /proc/$pid/environ |
        sed -n 's/^SLURM_JOB_ID=//p; s/^SLURM_JOBID=//p' |
        head -n 1
    )
    printf '%s\t%s %s %s %s\n' "$job" "$pid" "$ppid" "$executable" "$arguments"
done
'''.strip()


def list_slurm_jobs(user=None):
    """Return running Slurm jobs and distinguish batch from interactive jobs.

    Default to the current user when available; an empty or unavailable user
    leaves scheduler visibility unfiltered. Raise RuntimeError if squeue fails.
    """
    if user == None:
        try:
            user = getpass.getuser()
        except:
            user = None

    cmd = ["squeue", "-r", "-h", "-t", "R", "--Format", SLURM_JOB_LIST_FORMAT]
    if user not in [None, '']:
        cmd[1:1] = ["-u", user]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, check=False, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("squeue timed out after %s seconds" % exc.timeout) from exc
    except OSError as exc:
        raise RuntimeError("Could not run squeue: %s" % exc) from exc

    if proc.returncode != 0:
        message = "squeue exited with status %d" % proc.returncode
        detail = proc.stderr.strip()[:512]
        raise RuntimeError("%s: %s" % (message, detail) if detail else message)

    jobs = []
    for line in proc.stdout.splitlines():
        jobid, separator, rest = line.partition('|')
        fields = rest.rsplit('|', 6)
        if not separator or len(fields) != 7:
            continue
        name, job_user, state, runtime, nodes, nodelist, batch_flag = fields
        try:
            is_batch = int(batch_flag.strip()) > 0
        except ValueError:
            continue
        jobs.append({'rm': 'Slurm',
                     'jobid': jobid.strip(),
                     'name': name.strip(),
                     'user': job_user.strip(),
                     'state': state.strip(),
                     'time': runtime.strip(),
                     'nodes': nodes.strip(),
                     'nodelist': nodelist.strip(),
                     'is_batch': is_batch})
    return jobs


def parse_processes(output):
    """Return top-level srun launchers, excluding STAT daemon launches.

    Example input from ps (PID, parent PID, command):

        4100 3000 /usr/bin/srun -n 4 ./app
        4101 4100 /usr/bin/srun -n 4 ./app

    Only PID 4100 is returned because PID 4101 is its child.
    """
    processes = []
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 3:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        # check the executable, not arguments mentioning another srun.
        tokens = fields[2].split()
        if os.path.basename(tokens[0]) != 'srun':
            continue
        if any(os.path.basename(token) in ('STATD', 'stat-gui-server')
               for token in tokens):
            continue
        processes.append({'pid': pid, 'ppid': ppid, 'command_line': fields[2]})

    launcher_pids = {process['pid'] for process in processes}
    return [process for process in processes if process['ppid'] not in launcher_pids]


class SlurmJobDiscovery(object):
    """Find the application's launcher using Slurm job and step records."""

    def __init__(self, run_command):
        self.run = run_command

    def resolve(self, jobid):
        """Resolve one running job, including an array or heterogeneous component.

        Validate the requested ID and canonical scheduler record before selecting
        batch or interactive discovery. Return launcher host/PIDs, process details,
        job type and node list; unavailable or ambiguous job records raise.
        """
        jobid = str(jobid).strip()
        if not re.fullmatch(r'[0-9]+(?:_[0-9]+)?(?:\+[0-9]+)?', jobid):
            raise RuntimeError('A valid Slurm jobid is required')

        # one-line scontrol records expose the canonical numeric ID for array
        # elements and heterogeneous components, without relying on squeue's
        # presentation IDs or named output fields.
        result = self.run('scontrol -o show job %s' % shlex.quote(jobid))

        # keep the first occurrence: SubmitLine may contain key=value arguments.
        records = [dict(token.split('=', 1) for token in reversed(line.split()) if '=' in token)
                   for line in result.stdout.splitlines() if line.strip().startswith('JobId=')]
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or 'Could not inspect job %s' % jobid)

        # a numeric ID can also be an array/heterogeneous leader. Slurm then
        # returns all members; match the exact element shown by squeue's JobID.
        if jobid.isdigit():
            records = [record for record in records if record.get('JobId') == jobid]
        if len(records) != 1 or records[0].get('JobState') != 'RUNNING':
            raise RuntimeError('Running job %s is no longer available or is ambiguous' % jobid)

        fields = records[0]
        canonical_id = fields.get('JobId', '')
        if not re.fullmatch(r'[0-9]+', canonical_id):
            raise RuntimeError('Could not read canonical job ID for job %s' % jobid)

        batch_host = fields.get('BatchHost', '')
        nodelist = fields.get('NodeList', '')
        try:
            is_batch = int(fields['BatchFlag']) > 0
        except (KeyError, ValueError):
            raise RuntimeError('Could not read job type for job %s' % jobid)

        if is_batch:
            if not batch_host or batch_host == '(null)':
                raise RuntimeError('Could not resolve the batch host for job %s' % jobid)
            host, processes = self._batch_launchers(canonical_id, batch_host, jobid)
        else:
            host, processes = self._interactive_launchers(canonical_id, jobid)

        return {'remote_host': host, 'launcher_exe': 'srun',
                'pids': [process['pid'] for process in processes],
                'processes': processes, 'is_batch': is_batch, 'nodelist': nodelist}

    def _batch_launchers(self, jobid, host, job_alias=None):
        """Match top-level srun processes on the batch host by their inherited job ID."""
        output = self._probe_processes(BATCH_LAUNCHER_PROBE, host)
        accepted_jobs = {jobid, job_alias or jobid}
        process_lines = []
        process_jobs = {}
        for line in output.splitlines():
            environment_job, separator, process_line = line.partition('\t')
            fields = process_line.split(None, 1)
            if not separator or not fields:
                continue
            try:
                pid = int(fields[0])
            except ValueError:
                continue
            process_lines.append(process_line)
            process_jobs[pid] = environment_job.strip()

        launchers = parse_processes('\n'.join(process_lines))
        return host, [process for process in launchers
                      if process_jobs.get(process['pid']) in accepted_jobs]

    def _interactive_launchers(self, jobid, job_alias=None):
        """Find a live workload launcher from Slurm step records.

        Report host failures if no attachable workload launcher is found.
        """
        result = self.run('scontrol -o show step %s' % shlex.quote(jobid))
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or
                               'Could not inspect steps for job %s' % jobid)
        hosts = {}
        for line in result.stdout.splitlines():
            fields = dict(token.split('=', 1) for token in line.split() if '=' in token)
            stepid = fields.get('StepId', '')
            if (stepid.partition('.')[0] not in {jobid, job_alias or jobid} or
                    fields.get('State') != 'RUNNING'):
                continue
            if stepid.rsplit('.', 1)[-1] in ('extern', 'interactive', 'batch'):
                continue
            # the Slurm scheduler directly identifies the launcher for this workload step.
            match = re.fullmatch(r'([A-Za-z0-9_.-]+):([1-9][0-9]*)',
                                 fields.get('SrunHost:Pid', ''))
            if match:
                host, pid = match.groups()
                hosts.setdefault(host, set()).add(int(pid))

        failures = []
        for host, pids in hosts.items():
            try:
                pid_list = ','.join(str(pid) for pid in sorted(pids))
                output = self._probe_processes('ps ww -p %s -o pid=,ppid=,args=' % pid_list, host)
                # Slurm records the owning step even when srun --jobid overrides the inherited job ID.
                processes = [process for process in parse_processes(output) if process['pid'] in pids]
                if processes:
                    return host, processes
            except RuntimeError as exc:
                failures.append(str(exc))

        if failures:
            raise RuntimeError('\n'.join(failures))
        return None, []

    def _probe_processes(self, command, host):
        """Return process output, treating a launcher that disappeared as an empty result."""
        message = 'Could not inspect launcher processes on %s' % host
        try:
            result = self.run(command, host)
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = ('SSH timed out after %s seconds' % exc.timeout
                      if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
            raise RuntimeError('%s: %s' % (message, detail)) from exc
        if result.returncode == 1:
            return ''
        if result.returncode != 0:
            detail = result.stderr.strip() or 'command exited with status %s' % result.returncode
            raise RuntimeError('%s: %s' % (message, detail))
        return result.stdout
