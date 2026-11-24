#!/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Control a STAT server over SSH without opening the GUI.

Connection options:
    --remote-login USER@HOST       Required hostname to SSH to the cluster head node.
    --remote-prefix PREFIX         STAT installation prefix path on the cluster head node.
    --flux-allocation ID           In case of Flux jobs, Flux instance to use for flux proxy.
    --ssh-option ARG               Extra SSH argument (only needed for non-default SSH config).
    --remote-server-host HOST      Optional second SSH hop from the remote login (only needed to connect with compute nodes directly).

Commands:
    status                        Report STAT server side version and supported backends.
    sample-process PID            Sample given job launcher by PID, or one process with --serial.
    sample-job JOBID              Discover the launcher by job ID (e.g. slurm) and sample its ranks.

Dyninst is the default backend. Use --backend pystack on a PyStack-enabled
server. PyStack supports --native for C/C++ frames and --with-children for
descendant processes.

Sampling options include --samples, --with-threads, --sample-type and
--trace-frequency for the run time between snapshots in milliseconds. Advanced controls are --num-retries, --retry-freq
(delay in microseconds) and --max-threads-per-daemon.

sample-job defaults to --scheduler auto. The server requires exactly one of
the Slurm, LSF or ALPS launchers on its PATH for auto selection. Explicitly
select the scheduler if detection is ambiguous. Flux jobs require --scheduler flux.

Examples using the installed CLI:
    stat-gui-remote-cli --remote-login user@cluster status
    stat-gui-remote-cli --remote-login user@server --remote-prefix /opt/stat \\
        sample-process 12345 --serial --backend dyninst
    stat-gui-remote-cli --remote-login user@login-host \\
        sample-job JOBID --scheduler slurm --backend pystack --native --with-children \\
        --samples 2 --output-dir ./job-samples
    stat-gui-remote-cli --remote-login user@login-host --flux-allocation ALLOCATION_ID \\
        sample-job APPLICATION_JOB_ID --scheduler flux --samples 2

Sampling prints the local and remote result locations as JSON. Add -v for session details and graph paths,
or -vv for the full response including DOT text. The client saves response.json and DOT files under
./stat_results/run-* by default. Use --output-dir DIR to select an exact directory.
"""

from __future__ import print_function

import argparse
import json
import os
import tempfile
from datetime import datetime

from remote.stat_client_connection import StatRemoteClient, StatRemoteClientError

# the remote server checks which of these backends its build provides.
SUPPORTED_BACKENDS = ("dyninst", "pystack", "gdb")
DEFAULT_BACKEND = "dyninst"


def build_client(args):
    """Create the SSH client using the selected server command or installation prefix."""
    command = StatRemoteClient.command_for_prefix(args.remote_prefix) if args.remote_prefix else "stat-gui-server"
    return StatRemoteClient.ssh(
        args.remote_login, command, ssh_options=args.ssh_option, remote_prefix=args.remote_prefix,
        remote_server_host=args.remote_server_host, flux_allocation=args.flux_allocation)


def print_json(data, output_dir=None, stdout_data=None):
    """Save the full response when requested, then print the selected JSON view."""
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if output_dir:
        with open(os.path.join(output_dir, "response.json"), "w") as stream:
            stream.write(text)
    if stdout_data is not None and stdout_data is not data:
        text = json.dumps(stdout_data, indent=2, sort_keys=True) + "\n"
    print(text, end="")


def sampling_response_for_stdout(response, verbosity, server_host):
    """Return result locations by default, session details at -v, or the full response at -vv."""
    if verbosity == 0:
        sample = response["samples"][0]
        return {
            "output_dir": response["output_dir"],
            "remote_host": server_host,
            "remote_output_dir": sample.get("out_dir") or os.path.dirname(sample["dot_filename"]),
            "response_file": os.path.join(response["output_dir"], "response.json"),
        }
    if verbosity >= 2:
        return response

    def without_dot_text(graph):
        return {key: value for key, value in graph.items() if key != "dot_text"}

    details = dict(response)
    details["samples"] = [without_dot_text(sample) for sample in response["samples"]]
    if "merged" in response:
        details["merged"] = without_dot_text(response["merged"])
    return details


def create_output_dir(requested_dir):
    """Create a client output directory, choosing a unique run directory by default."""
    if requested_dir:
        path = os.path.abspath(requested_dir)
        os.makedirs(path, exist_ok=True)
        return path

    root = os.path.abspath("stat_results")
    os.makedirs(root, exist_ok=True)

    # the random suffix prevents collisions between runs started in the same second.
    prefix = datetime.now().strftime("run-%Y%m%d-%H%M%S-")
    return tempfile.mkdtemp(prefix=prefix, dir=root)


def write_dot(output_dir, result):
    """Save returned DOT text in the client run directory."""
    dot_text = result.get("dot_text")
    if not dot_text:
        return None

    # preserve STAT filenames, but never use a server filesystem path on the client.
    path = os.path.join(output_dir, os.path.basename(result["dot_filename"]))
    with open(path, "w") as stream:
        stream.write(dot_text)
    return path


def common_sample_params(args):
    """Translate shared CLI sampling options into server request parameters."""
    # keep protocol field names separate from the user-facing CLI option names.
    params = {
        "backend": args.backend,
        "cuda_quick": args.cuda_quick,
        "with_threads": args.with_threads,
        "native": args.native,
        "with_children": args.with_children,
        "sample_type": args.sample_type,
        "num_retries": args.num_retries,
        # retry frequency is measured in microseconds.
        "retry_freq": args.retry_freq,
        "max_threads_per_daemon": args.max_threads_per_daemon,
    }
    if args.remote_log_dir:
        params.update(
            log_frontend=True,
            log_backend=True,
            log_swerr=True,
            log_dir=args.remote_log_dir,
        )
    return params


def main(argv=None):
    """Run one SSH command and release the session and transport before returning."""
    parser = argparse.ArgumentParser(description="Drive STAT remote connect server without opening stat-gui.")
    parser.add_argument("--remote-login", required=True, help="SSH login to cluster head node, e.g. user@login-node")
    parser.add_argument("--remote-server-host",
                        help="run the server through a second SSH hop from --remote-login")

    parser.add_argument("--flux-allocation", help="Flux allocation ID to proxy into before starting the server")
    parser.add_argument("--remote-prefix", help="Path on cluster head node to STAT installation directory")
    parser.add_argument("--ssh-option", action="append", default=[], help="extra ssh option/argument; repeat for multiple args")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="show server version and supported backends")

    # scheduler selection applies only to job-ID discovery, not PID-based sampling.
    sample_job = subparsers.add_parser("sample-job", help="sample ranks belonging to a scheduler job")
    sample_job.add_argument("jobid")
    sample_job.add_argument(
        "--scheduler", default="auto", choices=("auto", "slurm", "lsf", "alps", "flux"),
        help="job-ID resolver on the server (default: auto-detect Slurm, LSF or ALPS on PATH). "
             "Select flux explicitly for Flux jobs.")

    sample_process = subparsers.add_parser("sample-process", help="sample an MPI launcher or a serial process")
    sample_process.add_argument("pid", type=int, help="launcher PID on the server host")
    # without --serial, the PID identifies an MPI launcher whose registered ranks are sampled.
    sample_process.add_argument("--serial", action="store_true", help="attach directly to one process")

    # both attach modes use the same graph format and stack-walk controls.
    for sample_parser in (sample_job, sample_process):
        sample_parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND)
        sample_parser.add_argument("--native", action="store_true", help="include C/C++ frames with PyStack")
        sample_parser.add_argument("--with-children", action="store_true", help="sample descendant processes with PyStack")
        sample_parser.add_argument("--with-threads", action="store_true")
        sample_parser.add_argument("--cuda-quick", action="store_true",
                                   help="collect faster CUDA device traces with --backend gdb --with-threads")
        sample_parser.add_argument(
            "--sample-type", default="function and line",
            choices=["function only", "function and pc", "module offset", "function and line"],
            help="information retained in each stack frame")

        # match the GUI options: five stack-walk retries with a 10-microsecond delay.
        # the Dyninst backend can briefly resume a stopped target between failed or incomplete walks.
        sample_parser.add_argument("--num-retries", type=int, default=5,
                                   help="retries for failed or incomplete stack walks, default: 5")

        # despite the name, this is a delay in microseconds, not a frequency in hertz.
        sample_parser.add_argument("--retry-freq", type=int, default=10,
                                   help="delay between stack-walk retries in microseconds, default: 10")

        # capacity for thread membership in graphs, not a limit on application thread creation.
        sample_parser.add_argument("--max-threads-per-daemon", type=int, default=512,
                                   help="number of threads to track per STAT daemon, default: 512")

        sample_parser.add_argument("--samples", type=int, default=1,
                                   help="number of snapshots to collect, default: 1")
        sample_parser.add_argument("--trace-frequency", type=int, metavar="MS",
                                   help="run time between snapshots in milliseconds (default: 0, paused)")
        sample_parser.add_argument("--resume-between-samples", action="store_true",
                                   help="run for 1000 ms between snapshots unless --trace-frequency is set")

        sample_parser.add_argument("-v", "--verbose", action="count", default=0,
                                   help="print session details and graph paths (-vv includes DOT text)")

        sample_parser.add_argument(
            "--remote-log-dir",
            help="absolute shared path for STAT frontend, daemon and stack-walker error logs",
        )

        # the DOT text comes back in JSON; saving it here writes only to the client filesystem.
        sample_parser.add_argument("--output-dir",
                                   help="client directory for DOT graphs and response.json (default: ./stat_results/run-*)")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")

    if args.command != "status" and args.backend != "pystack" and (args.native or args.with_children):
        parser.error("--native and --with-children require --backend pystack")

    if args.command != "status" and args.cuda_quick and (args.backend != "gdb" or not args.with_threads):
        parser.error("--cuda-quick requires --backend gdb --with-threads")

    if args.command != "status" and args.trace_frequency is not None and args.trace_frequency < 0:
        parser.error("--trace-frequency must be nonnegative")

    if args.command != "status" and args.remote_log_dir is not None and not os.path.isabs(args.remote_log_dir):
        parser.error("--remote-log-dir must be an absolute path on the cluster")

    # one connection owns this command; close it even after a server or sampling error.
    client = build_client(args)
    output_dir = None
    try:
        client.ensure_remote_server()

        if args.command == "status":
            print_json(client.request("health"))
        elif args.command in ("sample-job", "sample-process"):
            if args.cuda_quick and not client.remote_health.get("cuda_quick"):
                parser.error("the remote server does not support --cuda-quick")
            params = common_sample_params(args)
            if args.samples < 1:
                parser.error("--samples must be positive")

            if args.command == "sample-job":
                params.update({"mode": "jobid", "jobid": args.jobid, "scheduler": args.scheduler})
            elif args.pid <= 0:
                parser.error("pid must be positive")
            elif args.serial:
                params.update({"mode": "serial", "processes": [str(args.pid)]})
            else:
                params.update({"mode": "attach", "pid": args.pid})

            # prepare local storage before attaching so a directory error cannot pause the job.
            output_dir = create_output_dir(args.output_dir)

            # keep the attachment alive across samples, then wait for detach acknowledgement.
            attach_result = client.request("sessions.attach", params)
            session_id = attach_result["session_id"]
            outputs = {"attach": attach_result, "samples": [], "output_dir": output_dir}

            try:
                # the server owns repeated sampling and the final merge, as in classic STAT.
                sample_params = dict(params, session_id=session_id)
                if args.samples > 1:
                    sample_params["num_traces"] = args.samples
                    trace_frequency = args.trace_frequency
                    if trace_frequency is None:
                        trace_frequency = 1000 if args.resume_between_samples else 0
                    sample_params["trace_frequency_ms"] = trace_frequency
                    merged = client.request("sessions.sample_multiple", sample_params)
                    outputs["samples"] = merged.pop("samples")
                    merged["local_dot"] = write_dot(output_dir, merged)
                    outputs["merged"] = merged
                else:
                    outputs["samples"] = [client.request("sessions.sample", sample_params)]

                for result in outputs["samples"]:
                    result["local_dot"] = write_dot(output_dir, result)

            except Exception as sample_error:
                try:
                    client.request("sessions.detach", {"session_id": session_id})
                except Exception as cleanup_error:
                    if isinstance(sample_error, StatRemoteClientError):
                        sample_error.error["cleanup_error"] = str(cleanup_error)
                    else:
                        raise RuntimeError("%s. Detach failed: %s" % (sample_error, cleanup_error)) from sample_error
                raise

            outputs["detach"] = client.request("sessions.detach", {"session_id": session_id})
            display = sampling_response_for_stdout(outputs, args.verbose, client.remote_health["hostname"])
            print_json(outputs, output_dir, stdout_data=display)
    except StatRemoteClientError as exc:
        print_json({"ok": False, "error": exc.error}, output_dir)
        return 1
    except RuntimeError as exc:
        print_json({"ok": False, "error": {"message": str(exc)}}, output_dir)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
