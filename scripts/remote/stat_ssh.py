# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Build noninteractive SSH commands for Remote Connect on either side."""

import os


SSH_DEFAULT_OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new")


def remote_shell_argv(host, command, options=None, executable="ssh"):
    """Trust new SSH host keys, reject changed keys, and preserve explicit options.

    OpenSSH uses the first value for an option, so caller options precede the
    defaults. Other remote shells retain their original command syntax.
    """
    if os.path.basename(executable) != "ssh":
        return [executable, host, command]
    return ([executable, "-T"] + list(options or ()) +
            list(SSH_DEFAULT_OPTIONS) + ["--", host, command])
