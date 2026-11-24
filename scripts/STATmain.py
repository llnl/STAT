#!/bin/env python

"""@package STATview
Visualizes dot graphs outputted by STAT."""

__copyright__ = """Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC."""
__license__ = """Produced at the Lawrence Livermore National Laboratory
Written by Gregory Lee <lee218@llnl.gov>, Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from stat_build_info import VERSION_SUFFIX

__author__ = ["Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"]
__version_major__ = 4
__version_minor__ = 3
__version_revision__ = 0
__version__ = "%d.%d.%d" % (__version_major__, __version_minor__, __version_revision__)

import sys
import ctypes
import os
import argparse

HAVE_DLOPEN = hasattr(sys, 'getdlopenflags')
if HAVE_DLOPEN is True:
    dlflags = sys.getdlopenflags()
    new_dlflags = ctypes.RTLD_GLOBAL | dlflags
    sys.setdlopenflags(new_dlflags)

HAVE_STATVIEW = True
import_exception = None
try:
    from STATview import STATview_main
except Exception as e:
    HAVE_STATVIEW = False
    import_exception = e
HAVE_STATGUI = True
try:
    from STATGUI import HAVE_NATIVE_STAT, STAT_PYSTACK_BACKEND_ENABLED, STATGUI_main
except Exception as e:
    HAVE_STATGUI = False
    import_exception = e
from STAThelper import exec_and_exit

def STATmain_main(in_arg_list, command=None):
    env_var = 'STAT_BIN'
    if command == 'bench':
        env_var = 'STATBENCH_BIN'
    try:
        stat_bin = os.environ[env_var]
    except:
        sys.stderr.write('%s environment variable not set. Please ensure that you are running through the STAT, stat-cl, stat-view, stat-gui, STATBench, or stat-bench script\n' % (env_var))
        sys.exit(1)
    arg_list = [stat_bin]
    arg_list += in_arg_list
    exec_and_exit(arg_list)


if __name__ == '__main__':
    args = None
    arg_parser = argparse.ArgumentParser(prog='STAT')
    subparsers = arg_parser.add_subparsers()


    if sys.argv[1] in ['gui', 'view'] and import_exception is not None:
        raise import_exception

    # argument parsing for the stat-view command or STAT view subcommand
    if HAVE_STATVIEW:
        view_parser = subparsers.add_parser('view')
        view_parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__ + VERSION_SUFFIX))
        view_parser.add_argument("files", nargs='*', help="optional list of .dot files")
        view_parser.set_defaults(func=STATview_main)

    # argument parsing for the stat-gui command or STAT gui subcommand
    if HAVE_STATGUI:
        gui_parser = subparsers.add_parser('gui')
        from remote.stat_client_gui import add_gui_arguments
        # the standalone client can request PyStack from the server without local bindings.
        add_gui_arguments(gui_parser, version=__version__ + VERSION_SUFFIX,
                          have_pystack=STAT_PYSTACK_BACKEND_ENABLED or not HAVE_NATIVE_STAT)
        gui_parser.set_defaults(func=STATGUI_main)

    # argument parsing for the stat-cl command or STAT cl subcommand
    cl_parser = subparsers.add_parser('cl')

    # argument parsing for the stat-bench command or STAT bench subcommand
    bench_parser = subparsers.add_parser('bench')

    # argument parsing for the stat-merge command or STAT merge subcommand
    merge_parser = subparsers.add_parser('merge')

    # TODO: ultimately the subcommands below should be parsed here an not in their own files
    if len(sys.argv) > 1 and sys.argv[1] == 'cl':
        # TODO: this will work best if STAT.C functionality is completely replaced and implemented here
        STATmain_main(['-r', '5', '-c', '-A'] + sys.argv[2:], 'cl')
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == 'bench':
        STATmain_main(sys.argv[2:], 'bench')
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == 'merge':
        # TODO: this will be the hardest to modify since the argument parsing is done with internal classes
        from core_file_merger import STATmerge_main
        STATmerge_main(sys.argv[1:])
        sys.exit(0)

    args = arg_parser.parse_args(sys.argv[1:])
    args.func(args)
