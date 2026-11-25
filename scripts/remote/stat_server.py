#!/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Protocol entrypoint for STAT control over an SSH stdio connection.

The installed stat-gui-server wrapper starts this module. Each input line is a
JSON request with an id, method and params; replies and progress events carry
the same id. The server dispatches each request to StatSessionController.
Protocol stdout carries only JSON. Native output and diagnostics go to
stderr, and native tools receive /dev/null as stdin.

Operations run synchronously. Application control belongs to
stat_server_session.py; this module only handles the protocol and connection
lifecycle. EOF or successful server.shutdown ends the loop and triggers session
cleanup. EOF cannot interrupt a native operation already executing.

Usage (normally started automatically by stat-gui-remote-cli or the GUI)::

    stat-gui-server
    ssh -T user@cluster stat-gui-server

For a manual protocol check, send this line on stdin::

    {"id": 1, "method": "health", "params": {}}

The server replies with the same id, ok=true and its version/capabilities.
The CLI exposes this check as status. Close stdin or send server.shutdown
to end a manual session. Use the remote client for sampling and detach.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import traceback

from remote.stat_server_session import StatSessionController, StatRemoteError


class JsonStdioServer(object):
    """Serve sequential requests through a session controller for the connection."""

    def __init__(self, session_controller, input_stream=None, protocol_output=None):
        self.session_controller = session_controller
        self.input_stream = input_stream or sys.stdin
        self.protocol_output = protocol_output or sys.stdout

    def _write_response(self, response):
        """Flush one complete JSON line so the client can consume it immediately."""
        self.protocol_output.write(json.dumps(response, sort_keys=True) + "\n")
        self.protocol_output.flush()

    def _write_progress(self, request_id, progress):
        """Associate a progress event with the request that produced it."""
        self._write_response({"id": request_id, "event": "progress", "progress": progress})

    def _handle_request(self, request):
        """Dispatch one operation, forwarding progress before returning its result."""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not method:
            raise StatRemoteError("invalid_request", "method is required")

        result = self.session_controller.dispatch(
            method,
            params,
            progress=lambda event: self._write_progress(request_id, event),
        )
        return {"id": request_id, "ok": True, "result": result}

    def serve_requests(self):
        """Read requests until EOF or an acknowledged shutdown, then release sessions.

        Return structured errors without closing the connection so a client can
        correct a request or retry detach. Unexpected failures also emit a
        traceback on stderr. Failed shutdown leaves the loop open for a retry;
        cleanup on EOF has no reply channel and must report failure on stderr.
        """
        try:
            for line in self.input_stream:
                line = line.strip()
                if not line:
                    continue

                request_id = None
                request = {}
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise StatRemoteError("invalid_request", "request must be a JSON object")
                    request = parsed
                    request_id = request.get("id")
                    response = self._handle_request(request)
                except StatRemoteError as exc:
                    response = {"id": request_id, "ok": False, "error": exc.to_dict()}
                except ValueError as exc:
                    response = {
                        "id": request_id,
                        "ok": False,
                        "error": {"code": "invalid_json", "message": str(exc), "phase": "parse", "context": {}},
                    }
                except Exception as exc:
                    traceback.print_exc(file=sys.stderr)
                    response = {
                        "id": request_id,
                        "ok": False,
                        "error": {"code": "internal_error", "message": str(exc), "phase": None, "context": {}},
                    }

                self._write_response(response)
                if (request.get('method') == 'server.shutdown' and
                        response.get('ok')):
                    break
        finally:
            try:
                self.session_controller.close_session()
            except Exception as exc:
                # an EOF cannot carry a protocol response. Report uncertainty on stderr
                # and exit unsuccessfully rather than claiming successful cleanup.
                sys.stderr.write("STAT: Remote cleanup is unconfirmed: %s\n" % exc)
                raise


def _prepare_stdio():
    """Reserve private protocol streams before starting native STAT tools.

    Duplicate stdin/stdout for the JSON loop. Redirect descriptor 0 to /dev/null
    and descriptor 1 to stderr so child processes cannot read requests or write
    into the response stream. Update Python streams to follow those descriptors.
    """
    protocol_input_fd = os.dup(0)
    protocol_fd = os.dup(1)
    protocol_input = os.fdopen(protocol_input_fd, "r", 1)
    protocol_output = os.fdopen(protocol_fd, "w", 1)

    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
    finally:
        os.close(devnull)

    os.dup2(2, 1)

    try:
        sys.stdin = os.fdopen(0, "r", 1)
    except Exception:
        pass

    try:
        sys.stdout = os.fdopen(1, "w", 1)
    except Exception:
        pass

    return protocol_input, protocol_output


def main(argv=None):
    """Parse startup options and run the connection-owned stdio server."""
    parser = argparse.ArgumentParser(description="Run the STAT remote stdio JSON server.")
    parser.parse_args(argv)

    protocol_input, protocol_output = _prepare_stdio()
    server = JsonStdioServer(
        StatSessionController(), input_stream=protocol_input,
        protocol_output=protocol_output)
    server.serve_requests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
