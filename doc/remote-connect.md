# Using STAT remotely over SSH

The desktop GUI and command-line client start a STAT server on the cluster over SSH. Install STAT on cluster
and then locally on laptop/desktop to attach to your application, collects stack samples, and detaches.

## Install on the cluster and on your desktop

- **Cluster** — [Build and install STAT](../INSTALL) on the host that will run `stat-gui-server`. Keep the default
  GUI-enabled build, which installs the server and its Python bindings. The server runs without a display.
  Make the installation available on the compute nodes where STAT launches its daemons (shared filesustem).
  Dyninst is the default sampling backend. Use `--backend gdb` for CPU GDB sampling on a GDB-enabled server.
  For faster CUDA device traces, configure the server to use `cuda-gdb` (`STAT_GDB`) and add `--with-threads --cuda-quick`.
  In the remote GUI, select GDB and use **Attach → Advanced → GDB Path** to override the server default for this attach.

- **Desktop GUI** — Install the standalone `stat-gui` package and its graphics dependencies using the
  [desktop GUI installation guide](../scripts/gui/README.md). The desktop GUI can run without native STAT libraries.

- **Desktop CLI** — The same standalone package installs `stat-gui-remote-cli`. The command runs with Python and OpenSSH
  and can connect without native STAT libraries or a graphical display on the desktop.

Start with your cluster's recommended SSH settings in the **desktop's** `~/.ssh/config`.
Use a host entry to save the hostname, username, identity file, or jump host as needed.
For example, if your login host uses SSH port 2222:

```text
Host my-cluster
    HostName login.example.org
    User myuser
    Port 2222
```

Verify noninteractive access from the desktop with `ssh -o BatchMode=yes my-cluster true`, then use `--remote-login my-cluster`.
SSH uses port 22 by default. For a one-off port override, `--ssh-option=-p --ssh-option=2222` passes `ssh -p 2222` to the desktop SSH client.
In the examples, `--remote-prefix` points to the STAT installation on the host that runs the server.

### SSH authentication from the login node to compute nodes

Remote Connect uses SSH from your desktop to the login node.
For a Slurm batch job, the `srun` launcher often runs on the batch compute node.
STAT may then open a second SSH connection from the login node to find the launcher and start the server on that node.

Your existing SSH configuration may already support both connections, either through agent forwarding or through authentication configured by the cluster.
To check, connect from your desktop using the same SSH host alias you use with `--remote-login`.
Replace `LOGIN_HOST` with that alias.

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new LOGIN_HOST
```

From the login shell, list your running jobs and choose a compute node allocated to one of them.
Replace `COMPUTE_NODE` with its hostname and run the second command while the job is still running.

```bash
squeue --me --states=RUNNING --format="%.18i %N"
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new COMPUTE_NODE hostname
```

If the second command prints the compute node's hostname, both SSH connections work without a prompt and you can proceed with Remote Connect.

Some clusters require a short-lived SSH certificate.
If the login succeeds but the compute node reports `Permission denied (publickey)`, check which credential the cluster accepts and whether the login node can use it.
The Daint example below shows how to make a desktop-held credential available for the second connection.

#### CSCS Alps example (Daint)

CSCS requires multi-factor authentication (MFA) for account access and uses a signed SSH key for cluster login.
On your desktop, renew the certificate and load the key into your SSH agent.
With the filenames shown here, `ssh-add` also loads the matching `cscs-key-cert.pub` certificate.
See the [CSCS SSH guide](https://docs.cscs.ch/access/ssh/).

```bash
cscs-key sign
ssh-add -t 1d ~/.ssh/cscs-key
```

Add `ForwardAgent yes` to the existing Daint entry in your desktop's `~/.ssh/config`, alongside its user, identity, and jump-host settings.

```text
Host daint
    ForwardAgent yes
```

Reconnect to Daint with `ssh daint`. While the connection is open, SSH on the login node can ask your desktop agent to sign the compute node's authentication request.
The private key stays on your desktop.
Run `ssh-add -L` on the login node to check that the forwarded agent offers the CSCS identity, then repeat the compute-node hostname check above.

### SSH host-key confirmation

Remote Connect defaults to `BatchMode=yes` and `StrictHostKeyChecking=accept-new` for desktop SSH,
compute-node server startup, Slurm launcher discovery, and remote process and source queries.
OpenSSH records a new host key in `known_hosts` without prompting and rejects changed keys.
Explicit desktop `--ssh-option` values can override the desktop defaults. Authentication credentials are still required.

Each SSH hop uses the host-key files on the machine that opens it.
Desktop `--ssh-option` values apply to the desktop connection.
The login node's separate SSH commands use their own settings.

`Host key verification failed` points to host trust. `Permission denied (publickey)` points to user authentication.
If STAT reports `Could not find a running launcher for job ...`, check the second hop while the batch job is still running.
Restart the remote GUI connection after changing SSH settings or renewing the certificate, then refresh the job list.

## Standalone desktop GUI

Once STAT GUI is installed, you can start STAT GUI and connect to cluster as:

```bash
stat-gui --remote-login user@login-host --remote-prefix /path/to/stat/install/prefix/dir
```

This command connects to the cluster head node (`login-host`) over SSH and starts the STAT server from the
STAT installation directory specified by `--remote-prefix` on the cluster.

If you open the standalone GUI without an SSH login, **Attach** prompts you to restart it with the command above.

With `--remote-login`, the **Attach** dialog lists the user's running jobs. The server checks for `squeue` and `flux` commands
in its environment to detect shceduler. One can explicitly pass `--scheduler flux or slurm` or switch **Specify Resource Manager**
in the dialog.

For Slurm, choose a job or enter its ID and click **Search for Job**. The **Current Process List** shows launcher PIDs
and commands on the resolved host. Select the launcher, choose the backend on **Sample Options**, and click **Attach**.

The main window and Attach dialog show the SSH login and the current operation while a request is running.
The status returns to **Ready** when the request finishes, or **Disconnected** if the SSH connection closes.
If the server reports an error, expand **Recent SSH output** in the dialog to see diagnostics from that connection.
Collapse and reopen the section to include messages that arrived later.

![STAT remote GUI Attach dialog showing Slurm job and launcher selection](src/figs/stat_remote_gui_connect.png)

Use **Refresh Jobs** to find newly started jobs and **Refresh Process List** to look up the selected job again.

Browsing requires a working STAT installation on the login host or the host supplied with `--remote-server-host`.

You can also start with a known target:

```bash
stat-gui --remote-login user@launcher-host --remote-prefix /path/to/stat-install -i -a JOB_LAUNCHER_PID
stat-gui --remote-login user@server --remote-prefix /path/to/stat-install --backend pystack --native --with-children -i -I SERIAL_PROC_PID
```

With flux-core `0.83.0` or newer, the dropdown lists your running Flux applications, including those inside allocations.
Nested entries show `ALLOCATION_ID/APPLICATION_JOB_ID`. Choose a row or enter an application ID and click **Search for Job**.
If an ID occurs in multiple allocations, choose the qualified row. **Attach** starts the server in the owning allocation
and samples the application. The login host's noninteractive SSH session must be able to run `flux jobs` successfully.

To attach directly, supply both the allocation ID and the application job ID.

```bash
stat-gui --remote-login user@login-host --remote-prefix /path/to/stat-install \
  --flux-allocation ALLOCATION_ID --remote-job APPLICATION_JOB_ID --scheduler flux
```

The **Attach** dialog opens with the Flux job ID filled in. **Selected Flux Job** shows the application name and ID, its allocation when known,
the nodes, and the task count reported by Flux. Click **Attach** to start the helper and collect the first sample.

![STAT Attach dialog with the Flux job ID filled in](src/figs/stat_remote_flux_attach.png)

## Using the Remote Connect CLI

STAT remote client installation also provides CLI interface that can be used if needed for specific use cases.

### List running jobs

From the desktop, list your applications on the chosen scheduler. Each command prints a JSON object with a `jobs` array.

```bash
stat-gui-remote-cli --remote-login user@login-host --remote-prefix /opt/stat --list-jobs
stat-gui-remote-cli --remote-login user@login-host --remote-prefix /opt/stat --list-jobs --scheduler flux
```

By default, STAT checks for `squeue` on the server and uses Slurm if available. Otherwise, it checks for `flux`.
You can specify `--scheduler slurm` or `--scheduler flux` to query only that scheduler.

You can also print the same list from `stat-gui` without opening a window.

```bash
stat-gui --remote-login user@login-host --remote-prefix /opt/stat --list-jobs --scheduler flux
```

Automatic Flux job discovery requires flux-core 0.83.0 or newer on the cluster. For a nested application, the row includes
its owning `allocation`. Use the application's `jobid` directly:

```bash
stat-gui-remote-cli --remote-login user@login-host --remote-prefix /opt/stat sample-job APPLICATION_JOB_ID --scheduler flux
```

If several instances contain the same ID, use the returned allocation/application pair or specify
`--flux-allocation ALLOCATION_ID` before the subcommand. With an explicit allocation, `sample-job` attaches directly and `list-jobs` lists only that instance.

### Check your connection

Run the status command from the client machine. Replace the login and installation prefix with your cluster settings.

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

### Sampling options and output

For Python stacks, add `--backend pystack` to `sample-process` or `sample-job`. Add `--native` for C/C++ frames
or `--with-children` for descendant processes.

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

### Sample a serial process

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

### Sample an MPI application

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

### Sample a Slurm job

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
The GUI can list Slurm and Flux jobs.

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

### Sample a Flux job

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
stat-gui-remote-cli --remote-login user@login-host --remote-prefix /path/to/stat-install --flux-allocation ƒESxaZPeGP \
  sample-job ƒ25GUqV1 --scheduler flux --samples 2
```

The server loads its installed environment before running `flux proxy ALLOCATION_ID`. It can use the Flux path found
when STAT was configured, or the one already on the SSH session's `PATH`. The noninteractive SSH session must also
be able to reach the Flux broker through a local socket or a site-configured `FLUX_URI`. A profile loaded only in a login
shell may leave `FLUX_URI` unset for Remote Connect. Check with `ssh -T user@login-host 'flux jobs'` if Flux is on your SSH `PATH`.
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

Choose an AArch64 compute node in your job allocation where STAT can reach the job launcher. Use the first compute node if your
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

The CLI uses the login host unless you specify `--remote-server-host`. The desktop GUI can choose a host for a
Slurm attachment. It tries the launcher host first, then tries the first allocated node if the launcher is
outside the allocation and server startup there fails. To browse jobs when STAT cannot run on the login host,
supply `--remote-server-host` with a compatible host that can query Slurm.

For help with startup, SSH, and sampling errors, see
[Slurm Job Attachment](src/stat_troubleshoot.md#slurm-job-attachment) in the Troubleshooting Guide.

## Capture logs for a remote sample

For `sample-job` or `sample-process`, use `--remote-log-dir` to collect diagnostic logs on the cluster
and `--output-dir` to save the CLI response and graphs on your workstation.

- **On the cluster**, `--remote-log-dir` enables frontend, daemon, and stack-walker error logs.
  Choose a fresh absolute path on storage shared by the STAT server and the application nodes.
  The server creates the directory and any missing parents when logging starts.
- **On your workstation**, `--output-dir` saves `response.json` and any DOT graphs returned by the server.

For example, from your workstation:

```bash
stat-gui-remote-cli \
  --remote-login user@login-host \
  --remote-server-host compute-host \
  --remote-prefix /path/to/stat-install \
  sample-job JOBID --scheduler slurm \
  --backend pystack --native \
  --remote-log-dir /shared/scratch/user/stat-JOBID-attempt-1 \
  --output-dir ./stat-JOBID-attempt-1
```

## Sessions and connection loss

Normal sampling commands detach before returning. The server attempts to detach when the client closes its connection. An SSH
disconnect or command timeout does not guarantee cleanup: an operation may still be running, and the application may remain paused.
Check that the application is running before starting another attachment. Future releases will add extra API for retrying cleanup.
