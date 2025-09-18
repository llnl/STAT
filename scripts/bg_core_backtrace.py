"""@package bg_core_backtrace
An implementation for merging BlueGene lightweight core files into a .dot
format suitable for the Stack Trace Analysis Tool."""

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
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import sys
import os
import re
from stat_merge_base import StatTrace, StatMerger, StatMergerArgs
from subprocess import Popen, PIPE
from enum import Enum

class CoreType(Enum):
    UNKNOWN = 0
    BGQ = 1
    DYSECT = 2
    LCF = 3
    FUNCTION_SOURCE_LINE = 4
    HEXADDR = 5
    FUNCTION_SOURCE = 6

addr2line_exe = '/usr/bin/addr2line'
addr2line_map = {}
job_ids = []


# exapmle trace formats:
# From BG/Q:
# +++STACK
# Frame Address     Saved Link Reg
# 00000019c8afd7e0  0000000001014068
# 00000019c8afd9a0  0000000001024dec
# ---STACK

# From CUDA text lightweight:
# +++STACK
# gadd<<<(10,1,1),(32,1,1)>>>:cuda_example.cu:39
# ---STACK

# From DysectAPI
# +++STACK
# Module+Offset
# /g/g0/lee218/src/STAT/examples/sessions/mpi_ringtopo2+0x401729
# /lib64/libc.so.6+0x21b35
# /collab/usr/global/tools/stat/toss_3_x86_64_ib/stat-test/lib/libcallpathwrap.so+0x1dbb
# ---STACK

class BgCoreTrace(StatTrace):
    def get_traces(self):
        global addr2line_map, addr2line_exe
        if self.options['addr2line'] != 'NULL':
            addr2line_exe = self.options['addr2line']
        self.rank = int(self.file_path[self.file_path.rfind('.')+1:])

        self.tid = 0
        f = open(self.file_path, 'r')
        line_number_traces = []
        function_only_traces = []
        in_stack = False
        module_offset = False
        job_id = -1
        in_trace = False
        core_type = CoreType.UNKNOWN
        patterns = {}
        patterns[CoreType.HEXADDR] = r"0x([0-9a-fA-F]+)"
        patterns[CoreType.BGQ] = r"([0-9a-fA-F]+)[ \t]+([0-9a-fA-F]+)"
        patterns[CoreType.DYSECT] = r"([^\+]+)\+([^\+]+)"
        patterns[CoreType.LCF] = r"([^:]+):(.*)"
        patterns[CoreType.FUNCTION_SOURCE_LINE] = r"([^@]+)@([^:]+):([0-9\?]+)"
        patterns[CoreType.FUNCTION_SOURCE] = r"([^@]+)@([^@]+)"
        patterns[CoreType.UNKNOWN] = r"[.]*"
        any_lcf = False
        for line in f:
            if line == '\n' or line[0] == '#':
                print(line)
                continue
            line = line.strip('"')
            if line.find('Job ID') != -1:
                job_id = int(line.split(':')[1][1:])
                if job_id not in job_ids:
                    job_ids.append(job_id)
                    if len(job_ids) == 2 and self.options['jobid'] is None:
                        sys.stderr.write('\n\nWarning, multiple Job IDs detected in core files. Stack traces will still be merged into a single tree. To have traces differentiated by job ID, please use the -j or --jobid option\n\n')
                continue
            if line.find('+++STACK') != -1 or line.find('Function Call Chain') != -1:
                in_trace = True
                core_type = CoreType.UNKNOWN
                line_number_trace = []
                function_only_trace = []
                continue
            elif line.find('---STACK') != -1 or line.find('End of stack') != -1:
                #if (core_type == CoreType.DYSECT or core_type == CoreType.FUNCTION_SOURCE_LINE) and any_lcf == False:
                if core_type == CoreType.DYSECT:
                    # module offset frames are coming from callpath and need to be
                    # flipped such that the TOS is at the end of the trace
                    line_number_trace.reverse()
                    function_only_trace.reverse()
                if self.options['jobid'] is not None:
                    line_number_trace.insert(0, str(job_id))
                    function_only_trace.insert(0, str(job_id))
                line_number_traces.append(line_number_trace)
                function_only_traces.append(function_only_trace)
                in_trace = False
                core_type = CoreType.UNKNOWN
                continue
            if not in_trace:
                continue
            if line.find("Frame Address") != -1:
                core_type = CoreType.BGQ
                continue
            elif line.find("Module+Offset") != -1:
                core_type = CoreType.DYSECT
                continue
            line = line.strip('\n')
            module = None
            addr = None
            function = "??"
            source = "??"
            line_num = 0
            line_info = None
            if line in addr2line_map:
                function, line_info = addr2line_map[line]
            match = re.match(patterns[core_type], line)
            if line_info is None and core_type == CoreType.BGQ and match:
                module = self.options["exe"]
                addr = match.group(2)
                line_info = ''
            elif line_info is None and core_type == CoreType.DYSECT and match:
                module = match.group(1)
                offset = match.group(2)
                addr = format(int(offset, 0), '#x')
                line_info = ''
            match = re.match(patterns[CoreType.FUNCTION_SOURCE_LINE], line)
            if line_info is None and match:
                core_type = CoreType.FUNCTION_SOURCE_LINE
                function = match.group(1)
                source = match.group(2)
                try:
                    line_num = int(match.group(3))
                except:
                    pass
                line_info = '%s@%s:%d' %(function, source, line_num)
            match = re.match(patterns[CoreType.LCF], line.replace("::", "STATDOUBLECOLON"))
            if line_info is None and match:
                core_type = CoreType.LCF
                any_lcf = True
                function = match.group(1).replace("STATDOUBLECOLON", "::")
                source_line = match.group(2).replace("STATDOUBLECOLON", "::")
                match = re.match(r"([^:]+):(.*)", source_line)
                if match:
                    source = match.group(1)
                    line_num = match.group(2)
                    try:
                        line_num = int(match.group(2))
                    except:
                        pass
                line_info = '%s@%s:%d' %(function, source, line_num)
            match = re.match(patterns[CoreType.HEXADDR], line)
            if line_info is None and match:
                core_type = CoreType.HEXADDR
                module = self.options["exe"]
                addr = match.group(1)
                line_info = ''
            match = re.match(patterns[CoreType.FUNCTION_SOURCE], line)
            if line_info is None and match:
                core_type = CoreType.FUNCTION_SOURCE
                function = match.group(1)
                source = match.group(2)
                line_info = '%s@%s:-1' %(function, source)
            if line_info is None:
                if line != '***FAULT':
                    sys.stderr.write('\nWarning: format of stack frame "%s" not recognized for core type %s\n\n' %(line, core_type))
                function = line
                line_info = line
            elif core_type == CoreType.BGQ or core_type == CoreType.DYSECT or core_type == CoreType.HEXADDR:
                output = Popen([addr2line_exe, '-e', module, '--demangle', '-s', '-f', addr], stdout=PIPE, universal_newlines=True).communicate()[0]
                out_lines = output.split('\n')
                line_info = '%s@%s' % (out_lines[0], out_lines[1])
                if line_info.find('@') != -1:
                    function = line_info[:line_info.find('@')]
                elif line_info.find(':') != -1:
                    function = line_info[:line_info.find(':')]
                    line_info = line_info[line_info.find(':')+1:] + '@??'
                else:
                    function = line_info
                    line_info = '??@??'
            line_info = line_info.replace('<', '\<').replace('>', '\>')
            addr2line_map[addr] = (function, line_info)
            line_number_trace.insert(0, line_info)
            function_only_trace.insert(0, function)
        return [function_only_traces, line_number_traces]


class BgCoreMerger(StatMerger):
    def get_high_rank(self, trace_files):
        # determine the highest ranked task for graphlib initialization
        high_rank = 1
        for filename in trace_files:
            if filename.rfind('.') != -1:
                rank = filename[filename.rfind('.')+1:]
                try:
                    rank = int(rank)
                except Exception as e:
                    print("error determining rank of %s", filename)
                    print(e)
                    continue
                if rank > high_rank:
                    high_rank = rank
        return high_rank


class BgCoreMergerArgs(StatMergerArgs):
    def __init__(self):
        StatMergerArgs.__init__(self)

        # add the -j --jobid option to prefix traces with job ID
        self.arg_map["jobid"] = self.StatMergerArgElement("j", False, None, None, "delineate traces based on Job ID in the core file")

        # add an agrument type to take the application executable path
        self.arg_map["exe"] = StatMergerArgs.StatMergerArgElement("x", True, str, "NULL", "the executable path")

        # add an agrument type to take the application executable path
        self.arg_map["addr2line"] = StatMergerArgs.StatMergerArgElement("a", True, str, "NULL", "the path to addr2line")
        self.arg_map["loglevel"] = StatMergerArgs.StatMergerArgElement("L", True, str, "error", "the verbosity level (critical|error|warning|info|verbose|debug|insane)")
        self.arg_map["logfile"] = StatMergerArgs.StatMergerArgElement("F", True, str, "stdout", "the log file name (defaults to stdout)")
        self.arg_map["force"] = StatMergerArgs.StatMergerArgElement("r", False, int, 0, "whether to force parsing on warnings and errors")

        # override the usage messages:
        self.usage_msg_synopsis = '\nThis tool will merge the stack traces from the user-specified lightweight core files and output 2 .dot files, one with just function names, the other with function names + line number information\n'
        self.usage_msg_command = '\nUSAGE:\n\tpython %s [options] -x <exe_path> -c <corefile>*\n\tpython tracer_merge.py [options] -x <exe_path> -c <core_files_dir>\n' % (sys.argv[0])
        self.usage_msg_examples = '\nEXAMPLES:\n\tpython %s -x a.out -c core.0 core.1\n\tpython %s -x a.out -c core.*\n\tpython %s -x a.out -c ./\n\tpython %s -x a.out -c core_dir\n' % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])

    def is_valid_file(self, file_path):
        if file_path.find('core') == 0:
            return True
        return False

    def error_check(self, options):
        if options["exe"] == "NULL":
            sys.stderr.write('\nYou must specify an executable with the -x or --exe= options\n')
            self.print_usage()
        if not os.path.exists(options["exe"]):
            sys.stderr.write('\nFailed to find executable "%s".\n' % options["exe"])
            self.print_usage()
        StatMergerArgs.error_check(self, options)


if __name__ == '__main__':
    merger = BgCoreMerger(BgCoreTrace, BgCoreMergerArgs)
    ret = merger.run()
    if ret != 0:
        sys.stderr.write('Merger failed\n')
        sys.exit(ret)
