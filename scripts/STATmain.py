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
__author__ = ["Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 2
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

HAVE_GDB_SUPPORT = True
try:
    from STAT import STAT_SAMPLE_CUDA_QUICK
except:
    HAVE_GDB_SUPPORT = False
HAVE_STATVIEW = True
import_exception = None
try:
    from STATview import STATview_main
except Exception as e:
    HAVE_STATVIEW = False
    import_exception = e
HAVE_STATGUI = True
try:
    from STATGUI import STATGUI_main
except Exception as e:
    HAVE_STATGUI = False
    import_exception = e
from STAThelper import exec_and_exit
from core_file_merger import STATmerge_main

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
        view_parser.add_argument("files", nargs='*', help="optional list of .dot files")
        view_parser.set_defaults(func=STATview_main)

    # argument parsing for the stat-gui command or STAT gui subcommand
    if HAVE_STATGUI:
        gui_parser = subparsers.add_parser('gui')
        trace_group = gui_parser.add_mutually_exclusive_group()
        trace_group.add_argument("-P", "--withpc", help="sample program counter in addition to function name", action="store_true")
        trace_group.add_argument("-m", "--withmoduleoffset", help="sample module offset only", action="store_true")
        trace_group.add_argument("-i", "--withline", help="sample source line number in addition to function name", action="store_true")
        gui_parser.add_argument("-s", "--sleep", help="sleep for the specified number of seconds before attaching", metavar="SLEEPTIME", type=int)
        gui_parser.add_argument("-o", "--withopenmp", help="translate OpenMP stacks to logical application view", action="store_true")
        gui_parser.add_argument("-w", "--withthreads", help="sample helper threads in addition to the main thread", action="store_true")
        gui_parser.add_argument("-y", "--pythontrace", help="gather Python script level stack traces", action="store_true")
        gui_parser.add_argument("-U", "--countrep", help="only gather count and a single representative", action="store_true")
        gui_parser.add_argument("-d", "--debugdaemons", help="launch the daemons under the debugger", action="store_true")
        gui_parser.add_argument("-j", "--jobid", help="append specified job ID to output directory name")
        gui_parser.add_argument("-L", "--logdir", help="logging output directory")
        gui_parser.add_argument("-l", "--log", help="enable debug logging", choices=['FE', 'BE', 'CP'], action="append")
        if HAVE_GDB_SUPPORT:
            gui_parser.add_argument("-G", "--gdb", help="use (cuda) gdb to drive the daemons", action="store_true")
            gui_parser.add_argument("-Q", "--cudaquick", help="gather less comprehensive, but faster cuda traces", action="store_true")
        trace_group.add_argument("-M", "--mrnetprintf", help="use MRNet's print for logging", action="store_true")
        gui_parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
        attach_group = gui_parser.add_mutually_exclusive_group()
        attach_group.add_argument("-a", "--attach", help="attach to the specified [hostname:]PID", metavar='LAUNCHERPID')
        attach_group.add_argument("-C", "--create", help="launch the application under STAT's control. All args after -C are used to launch the app", nargs=argparse.REMAINDER)
        attach_group.add_argument("-I", "--serial", help="attach to the specified [<exe>@<host>:<pid>]+. All args after -I are interpreted as serial processes to attach to", nargs=argparse.REMAINDER)
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
        STATmerge_main(sys.argv[1:])
        sys.exit(0)

    args = arg_parser.parse_args(sys.argv[1:])
    args.func(args)
