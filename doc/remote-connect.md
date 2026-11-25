# Using STAT remotely over SSH

This remote-connect interface will also be used by the standalone remote GUI client, which will be introduced separately in a
follow-up change. This guide covers basic command-line usage for connecting and sampling applications.

The remote client lets you collect STAT stack traces without logging into an interactive shell on the cluster. It starts a
server process over SSH on the cluster, attaches to your application, collects samples and detaches. No additional network port
or persistent service is needed as long as passwordless access to cluster login is setup correctly.

## Before you start

On the cluster, [install STAT](../INSTALL) with the default GUI-enabled build. The server runs without a display and
uses the Python bindings included in that build. Note that `--enable-gui=no` disables the remote-connect API, so do
not add that option when installing STAT on the cluster.

On your laptop or desktop, use the `stat-gui-remote-cli` command from a STAT installation. Check that you can run
`ssh user@cluster true` successfully from that machine.

Dyninst is the default sampling backend. Use `--backend gdb` for CPU GDB sampling on a GDB-enabled server.
For faster CUDA device traces, configure the server to use `cuda-gdb` (`STAT_GDB`) and add `--with-threads --cuda-quick`.
To collect Python stacks, use `--with-pystack=/path/to/bin/pystack` to
point client to STAT's installation directory on the cluster. Then add `--backend pystack` to your `sample-process`
or `sample-job` command. Add `--native` for C/C++ frames, `--with-children` for descendant processes, or `--with-threads`
for thread stacks. These options also apply to repeated samples.

## Check your connection

Assuming you are able to ssh to `user@cluster` without any password or 2FA prompt, you can check the status of the STAT
server using the below command. Replace `user@cluster` with your SSH login.


```bash
stat-gui-remote-cli --remote-prefix /path/to/stat/install/dir --remote-login user@cluster status
```

The output should look like:

```json
{
  "git_commit": "1a9df8d2ae86dcb119c3175b9a37cfb0bfcf9fab",
  "git_dirty": true,
  "hostname": "cluster-0629",
  "stat_version": "4.3.0",
  "supported_backends": [
    "dyninst",
    "pystack"
  ],
  "user": "user"
}
```

The command reports the server's hostname, user, supported backends and build version. `git_commit` records the source revision
and `git_dirty` indicates tracked changes at configure time. Git fields are `"unknown"` when source metadata is unavailable.

Commands print JSON results on standard output and diagnostics on standard error. An error response contains `ok: false` and
an `error` object; the command exits with a nonzero status.

**NOTE:** Pass the STAT installation path on the server explicitly using `--remote-prefix /path/to/stat-install`.
If an example below omits this option, add it before the subcommand. SSH settings in `~/.ssh/config` apply as usual.


## Sampling options and output

After a successful sample, stdout reports the local output directory, server hostname and remote output directory as JSON.
Open the local path in `response_file` for the full response, including DOT text. The client saves this file and the DOT graphs
under `./stat_results/run-*` by default. Use `--output-dir ./my-samples` to choose a directory. Files with the same names
in an explicitly chosen directory are overwritten.

For example, sampling an MPI ring job with the default options prints:

```json
{
  "output_dir": "/home/user/stat_results/run-20261124-141708-byuqqlhv",
  "remote_host": "compute-01",
  "remote_output_dir": "/home/user/stat_results/mpi_ringtopo.0035",
  "response_file": "/home/user/stat_results/run-20261124-141708-byuqqlhv/response.json"
}
```

Add `-v` or `--verbose` to `sample-job` or `sample-process` to print session details and the remote (`dot_filename`)
and local (`local_dot`) graph paths. Use `-vv` for the full JSON previously printed by default, including the DOT text.
This output matches the saved `response.json`.

Sampling collects one snapshot by default and detaches before returning. With multiple snapshots, the `merged` entry
identifies the accumulated 3D graph. Open a downloaded graph with `stat-view /path/to/graph.dot`, or start `stat-gui`
and choose **File > Open**. The examples below show excerpts from the saved `response.json`.

Use `--with-threads` to include threads, or `--sample-type 'function only'` for function-only traces. By default, the
application stays paused between snapshots. To let it run between snapshots, add `--trace-frequency MS` to a command
with `--samples 2` or more. For example, `--samples 2 --trace-frequency 2500` collects two snapshots with 2.5 seconds of
application run time between them. Use `--trace-frequency 0` to keep it paused. You can also use
`--resume-between-samples` as a shortcut for 1000 ms when you omit `--trace-frequency`.

Sampling defaults match the classic GUI: up to five retries of a failed or incomplete stack walk, with a 10-microsecond
interval. You normally do not need to set these options. Use `--num-retries` and `--retry-freq` to adjust them when
troubleshooting incomplete native traces; these do not retry SSH connections. Dyninst may briefly resume a stopped process
between stack-walk retries. `--num-retries 0` disables retries.

Graph edges retain the full list of contributing ranks, matching the classic GUI default.

## Sample a serial process

On the cluster, compile and start the example from the STAT source directory:

```bash
cc -g -O0 examples/src/serial.c -o serial
./serial > serial.log 2>&1 &
echo "$!"
```

The example sleeps for about 30 seconds, so run the client command promptly while it is alive. Replace `12345` with its
printed PID:

```bash
stat-gui-remote-cli --remote-login user@server \
  --remote-prefix /path/to/stat-install \
  sample-process 12345 --serial --samples 2 --output-dir ./serial-samples
```

`--serial` attaches directly to that process on the server. A separate single-sample run captured `myfun` and `mysleep` in one 2D snapshot.

```json
{
  "attach": {"mode": "serial", "state": "attached"},
  "detach": {"state": "detached"}
}
```

Open the DOT files in `./serial-samples` on your client. The command above writes two snapshots and a merged graph.

## Sample an MPI application

On the cluster, start your MPI application and record its launcher PID:

```bash
mpirun -n 4 ./my-mpi-application > application.log 2>&1 &
echo "$!"
```

While it is running, use the printed PID from your client machine:

```bash
stat-gui-remote-cli --remote-login user@launcher-host \
  --remote-prefix /path/to/stat-install \
  sample-process 12345 --samples 2 --output-dir ./mpi-samples
```

Here `12345` is the `mpirun` PID, not a rank PID or shell wrapper's PID. Omit `--serial` so STAT samples the ranks registered
with the launcher. The server runs on the launcher's host even when ranks run on other nodes. For an existing application,
identify its launcher rather than starting another `mpirun`.

Both snapshots from this run contained all four MPI ranks.

```json
{
  "attach": {"pid": 3794321, "state": "attached"},
  "detach": {"state": "detached"}
}
```

Look in `./mpi-samples` on your client. This run saved `00_attach-demo.00XX.2D.dot`, `01_attach-demo.00XX.2D.dot`,
and the merged graph `02_attach-demo.00XX.3D.dot`.

## Sample a Slurm job

For a running Slurm job, provide its job ID from a server with access to Slurm:

```bash
stat-gui-remote-cli --remote-login user@login-host \
  --remote-prefix /path/to/stat-install \
  sample-job JOBID --scheduler slurm --samples 2 --output-dir ./job-samples
```

Replace `JOBID` with your job ID. STAT resolves the launcher through its resource-manager helper; you do not need to supply
rank PIDs or call discovery commands first.

`sample-job JOBID --scheduler slurm` selects Slurm explicitly. The default `--scheduler auto` detects available launcher
commands on the server and requires explicit selection when detection is missing or ambiguous. Job IDs remain strings.
Schedulers like `lsf` and `alps` reuse STAT's existing helpers and have not been cluster-tested here.
Job listing currently supports Slurm.

This run sampled four ranks across two Slurm nodes.

```json
{
  "attach": {
    "rm": "slurm",
    "job": {"jobid": "81", "launcher_exe": "srun"},
    "state": "attached"
  },
  "detach": {"state": "detached"}
}
```

Look in `./job-samples` on your client. This run saved `00_rank_test.0011.2D.dot`, `01_rank_test.0011.2D.dot`,
and the merged graph `02_rank_test.0011.3D.dot`.

## Sample a Flux job

On the login host, run `flux jobs -R` to see the outer allocation and the application running inside it.

```text
$ flux jobs -R
       JOBID USER      NAME       ST NTASKS NNODES     TIME INFO
  ƒESxaZPeGP user      stat-demo+  R      2      2   6.231s compute-[01,02]

ƒESxaZPeGP:
    ƒ25GUqV1 user      stat-demo+  R      4      2   4.093s compute-[01,02]
```

Take the allocation ID from the outer row (`ƒESxaZPeGP`) and the application job ID from the indented row
(`ƒ25GUqV1`). To confirm the application ID, list the running jobs inside the allocation.

```text
$ flux proxy ƒESxaZPeGP flux jobs -f running
       JOBID USER      NAME       ST NTASKS NNODES     TIME INFO
    ƒ25GUqV1 user      stat-demo+  R      4      2   6.059s compute-[01,02]
```

From your client, pass the outer ID to `--flux-allocation` and the inner ID to `sample-job`. Replace both IDs with those from your running job.

```bash
stat-gui-remote-cli --remote-login user@login-host --flux-allocation ƒESxaZPeGP \
  sample-job ƒ25GUqV1 --scheduler flux --samples 2
```

The server loads its installed environment before running `flux proxy ALLOCATION_ID`. It can use the Flux path found
when STAT was configured, or the one already on the SSH session's `PATH`. The SSH session must also be able to reach
the cluster's Flux broker. Check this with `ssh user@login-host 'flux jobs'` if Flux is already on your SSH `PATH`.
Select `--scheduler flux` explicitly on clusters that also provide Slurm.

The server starts `flux job attach --read-only --debug` for the selected application, attaches STAT to the helper PID,
and releases the helper after detach. This requires a running job and a Flux-capable LaunchMON installation with permission
to launch STAT daemons in that instance. If you already started a debug helper, use `sample-process PID` on its host in
the same Flux instance.

A separate STAT sampling run captured four ranks across two Flux nodes.

```json
{
  "attach": {
    "rm": "flux",
    "job": {"jobid": "ƒ21EEnxX", "launcher_exe": "flux-job"},
    "state": "attached"
  },
  "detach": {"state": "detached"}
}
```

## Connect through a login node to a compute node

Use `--remote-server-host` when you need to reach a compute node through a login node. This is useful when the login node
serves as a jump host or when the login and compute nodes have different architectures. For example, if the login node is
x86_64 and the compute nodes are AArch64, build STAT for the compute nodes. The resulting STAT binaries and libraries
cannot run on the x86_64 login node.

Choose an AArch64 node in your allocation where STAT can reach the job launcher. Use the first compute node if your
launcher runs there. From the login node, verify that you can SSH to the selected compute node without a prompt. Then run
the command below from your laptop or desktop, replacing the hosts, installation path, and job ID with your own values.

```bash
stat-gui-remote-cli --remote-login user@login-host \
  --remote-server-host compute-arm-01 \
  --remote-prefix /path/to/arm-stat-install \
  sample-job JOBID --scheduler slurm --samples 2
```

The client connects to `user@login-host` first. From there it uses SSH to start `stat-gui-server` on `compute-arm-01`.
Specify the installation path as it appears on that compute node.

Automatic server host selection is planned for a future release. For now, specify `--remote-server-host` when
the STAT installation targets a different architecture from the login node. If you omit it, the client checks the
installation on the login node. If the selected login host has a different architecture from the STAT installation,
the client reports an error like this one for an Aarch64 installation on an x86_64 login node.

```text
STAT installation is aarch64, but login host login-01 is X86-64
```

## Sessions and connection loss

Normal sampling commands detach before returning. The server attempts to detach when the client closes its connection. An SSH
disconnect or command timeout does not guarantee cleanup: an operation may still be running, and the application may remain paused.
Check that the application is running before starting another attachment. Future releases will add extra API for retrying cleanup.
