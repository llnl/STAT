"""@package stat_merge_base
A base class and implementation for merging trace files into a .dot format
suitable for the Stack Trace Analysis Tool."""

__copyright__ = """Copyright (c) 2007-2018, Lawrence Livermore National Security, LLC."""
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
import os
import subprocess
import STATmerge
import getopt
import time
from collections import namedtuple


class StatTrace(object):
    """
    the StatTrace is responsible for parsing individual trace files.  It can be
    extended to handle trace files of arbitrary format

    Derrived classes should implement the get_traces function
    """

    def __init__(self, file_path, options):
        self.file_path = file_path
        self.options = options
        self.rank = 0
        self.tid = 0
        self.traces = self.get_traces()

    def get_traces(self):
        """
        The base implementation simply uses each line of the trace file as the
        frame name in the call trace.

        The return type is a list of list of (sub) traces, where the former
        list is to support different levels of detail and the latter list is to
        support multiple traces within a trace file.  For example, consider a
        trace file that includes traces from multiple threads (thr) and assume
        we want two levels of detail, function name only (fn) and source line
        number (sl).  The return value would be of the form:

        [[fn.thr1, fn.thr2], [sl.thr1, sl.thr2]]

        Each trace is a list of frame names, for example:

        ["start", "main", "foo", "bar"]

        Note that '<' and '>' characters need to be prefixed with '\' for the
        .dot format.
        """

        self.rank = 0
        self.tid = 0
        traces = []
        sub_traces = []
        current = []
        with open(self.file_path, 'r') as f:
            for line in f:
                current.append(line.strip('\n').replace('<', '\<').replace('>', '\>'))
        sub_traces.append(current)
        traces.append(sub_traces)
        return traces


class StatMergerArgs(object):
    """
    The StatMergerArgs class is responsible for parsing command line options.
    It can be extended to add additional arguments.

    Derrived classes should implement the is_valid_file function.  If adding
    command line options, the derrived class should also include an __init__
    function and add to the arg_map.  The derrived __init__ method should first
    call StatMergerArgs.__init__(self).
    """
    StatMergerArgElement = namedtuple("StatMergerArgElement", ["short_name", "required_arg", "arg_type", "default_value", "help_string"])

    def __init__(self):
        # arg map format: arg_map[name] = (short name, required arg, arg type, value, help string)
        self.arg_map = {}
        self.arg_map["fileprefix"] = self.StatMergerArgElement("f", True, str, "NULL", "the output filename prefix")
        self.arg_map["type"] = self.StatMergerArgElement("t", True, str, "dot", "the output file type (raw or dot)")
        self.arg_map["high"] = self.StatMergerArgElement("H", True, int, -1, "the high rank of the input traces")
        self.arg_map["limit"] = self.StatMergerArgElement("l", True, int, 8192, "the max number of traces to parse per process")
        self._short_args_string = ""
        self._long_args_list = []

        # the default usage messages
        self.usage_msg_synopsis = '\nThis tool will merge the stack traces from the user specified files and output a .dot files suitible for viewing with STATview.\n'
        self.usage_msg_command = '\nUSAGE:\n\tpython %s [options] -c <file>*\n\tpython %s [options] -c <files_dir>\n' % (sys.argv[0], sys.argv[0])
        self.usage_msg_examples = '\nEXAMPLES:\n\tpython %s -c trace.0 trace.1\n\tpython %s -c trace.*\n\tpython %s -c ./\n\tpython %s -c trace_dir\n' % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])

    def is_valid_file(self, file_path):
        """
        The is_valid_file function should be overloaded to verify that a file
        is a valid trace file.  This is necssary in the case where a directory
        is specified and that directory contains files that are not trace
        files.
        """
        return True

    def error_check(self, options):
        """
        The error_check function is used to validate the specified arguments.
        This function can be overloaded to handle and additional arguments.
        The overloaded function should call the base implemntation:
        StatMergerArgs.error_check(self, options)
        """
        if not options["type"] in ['dot', 'raw']:
            sys.stderr.write('\nUnknown file type "%s". Type may be either "dot" or "raw"\n' % str(options["type"]))
            self.print_usage()

    def set_args(self):
        if self._short_args_string == "" or self._long_args_list == []:
            for name, arg_tuple in self.arg_map.items():
                short_name = arg_tuple.short_name
                if arg_tuple.required_arg is True:
                    short_name += ":"
                    name += "="
                self._short_args_string += short_name
                self._long_args_list.append(name)

    def get_short_args_string(self):
        self.set_args()
        return self._short_args_string

    def get_long_args_list(self):
        self.set_args()
        return self._long_args_list

    def print_options(self):
        sys.stderr.write('\nOPTIONS:\n')
        for name, arg_tuple in self.arg_map.items():
            if arg_tuple.required_arg is True:
                sys.stderr.write('\t-%s <value>, --%-20s %s\n' % (arg_tuple.short_name, name + '=<value>', arg_tuple.help_string))
            else:
                sys.stderr.write('\t-%s, --%-28s %s\n' % (arg_tuple.short_name, name, arg_tuple.help_string))

    def print_usage(self, return_value=-1):
        sys.stderr.write(self.usage_msg_synopsis)
        sys.stderr.write(self.usage_msg_command)
        sys.stderr.write('\nNOTE: linux has a limit for the number of command line arguments.  For a large number of files, specify -c <files_dir> instead of individual files.\n')
        self.print_options()
        sys.stderr.write(self.usage_msg_examples)
        sys.exit(return_value)

    def parse_args(self):
        # first delineate the file list from the options
        try:
            trace_file_delim_index = sys.argv.index('-c')
        except ValueError as e:
            sys.stderr.write('ERROR: You must specify trace files with the -c argument')
            self.print_usage()
        trace_file_args = sys.argv[trace_file_delim_index + 1:]
        sys.argv = sys.argv[0:trace_file_delim_index]

        # get list of trace files
        if len(trace_file_args) == 1:
            if os.path.isdir(trace_file_args[0]):
                trace_files = []
                for file_path in os.listdir(trace_file_args[0]):
                    if self.is_valid_file(file_path):
                        trace_files.append(trace_file_args[0] + '/' + file_path)
            else:
                trace_files = [trace_file_args[0]]
        else:
            trace_files = trace_file_args[0:]

        # parse the args
        try:
            opts, args = getopt.getopt(sys.argv[1:], self.get_short_args_string(), self.get_long_args_list())
        except getopt.GetoptError as err:
            sys.stderr.write('\n%s\n' % str(err))
            self.print_usage()
        except Exception as e:
            sys.stderr.write('Exception: %s\n%s, %s\n' % (str(e), self.get_short_args_string(), str(self.get_long_args_list())))
            self.print_usage()

        options = {}
        for name, arg_tuple in self.arg_map.items():
            options[name] = arg_tuple.default_value

        for option, arg in opts:
            if option in ("-h", "--help"):
                self.print_usage(0)
            handled = False
            for name, arg_tuple in self.arg_map.items():
                if option in ("--" + name, "-" + arg_tuple.short_name):
                    if (arg_tuple.required_arg is True):
                        try:
                            options[name] = arg_tuple.arg_type(arg)
                        except ValueError as e:
                            sys.stderr.write('\nInvalid specification "%s" for option "%s", expecting type %s.\n' % (arg, name, repr(arg_tuple.arg_type)))
                            self.print_usage()
                        handled = True
                        break
                    else:
                        options[name] = 1
                        handled = True
                        break
            if handled is False:
                sys.stderr.write('Unknown option %s\n' % (option))
                self.print_usage()

        # error checking
        self.error_check(options)

        return options, trace_files


class StatMerger(object):
    """
    The StatMerger class is the object that contains all the traces and is
    responsible for merging and outputting.

    Derrived classes should implement the get_high_rank function
    """
    def __init__(self, trace_type=StatTrace, arg_type=StatMergerArgs):
        self.trace_type = trace_type
        self.args = arg_type()
        self.options, self.trace_files = self.args.parse_args()
        if self.options["high"] == -1:
            self.options["high"] = self.get_high_rank(self.trace_files)
        self.verbose = True

    def get_high_rank(self, trace_files):
        """
    Given a list of trace file names, return the highest rank value.  This is
    needed for graphlib initialization.  This function should be overloaded.
        """
        return 0

    def generate_unique_filenames(self, count):
        if self.options["fileprefix"] == 'NULL':
            # make sure we generate unique file names
            for i in range(8192):
                filenames = []
                for j in range(count):
                    num = ''
                    if j != 0:
                        num = '_%d' % j
                    if i == 0:
                        filename = 'STAT_merge' + num + '.' + self.options["type"]
                    else:
                        filename = 'STAT_merge' + num + '.' + str(i) + '.' + self.options["type"]
                    if os.path.exists(filename):
                        break
                    filenames.append(filename)
                if len(filenames) == count:
                    break
        else:
            filenames = []
            for j in range(count):
                num = ''
                if j != 0:
                    num = '_%d' % j
                filenames.append(self.options["fileprefix"] + num + '.' + self.options["type"])
        return filenames

    def run(self):
        # we don't want sub processes to print this out...
        if self.options["type"] != 'raw':
            sys.stdout.write('merging %d trace files\n' % len(self.trace_files))
            sys.stdout.flush()

        filenames = self.generate_unique_filenames(2)

        if len(self.trace_files) > self.options["limit"]:
            # spawn sub processes to parallelize the merging
            sub_processes = self.spawn_sub_processes()

            # initialize graphlib
            ret = STATmerge.Init_Graphlib(self.options["high"])
            if ret != 0:
                sys.stderr.write('Failed to initialize graphlib\n')
                return ret

            # synchronize the sub processes
            handles = self.sync_sub_processes(sub_processes)

        else:
            handles = []
            self.verbose = False
            if self.options["type"] == 'dot':
                self.verbose = True

            # parse and merge the traces
            length = len(self.trace_files)
            for i, filename in enumerate(self.trace_files):
                if self.verbose:
                    sys.stdout.write('\b\b\b\b%03u%%' % (i / (length / 100.0)))
                    sys.stdout.flush()

                trace_object = self.trace_type(filename, self.options)
                for j, trace in enumerate(trace_object.traces):
                    if i == 0:
                        # initialize graphlib and generate the graph objects
                        ret = STATmerge.Init_Graphlib(self.options["high"])
                        if ret != 0:
                            sys.stderr.write('Failed to initialize graphlib\n')
                            sys.exit(1)
                        handle = STATmerge.New_Graph()
                        if handle == -1:
                            sys.stderr.write('Failed to create new graph\n')
                            sys.exit(1)
                        handles.append(handle)
                    for k, sub_trace in enumerate(trace):
                        # add the current trace
                        try:
                            ret = STATmerge.Add_Trace(handles[j], trace_object.rank, trace_object.tid, sub_trace)
                            if ret != 0:
                                sys.stderr.write('Failed to add trace\n')
                                sys.exit(1)
                        except:
                            pass

            if self.verbose:
                sys.stdout.write('... done!\n')

        # now get a merge with only the function name
        self.output_files(handles, filenames)

        if self.verbose:
            sys.stdout.write('View the outputted .dot files with `STATview`\n')
        return 0

    def output_files(self, handles, filenames):
        for handle, filename in zip(handles, filenames):
            if self.verbose:
                sys.stdout.write('outputing to file "%s" ...' % (filename))
            if self.options["type"] == 'dot':
                STATmerge.Output_Graph(handle, filename)
            elif self.options["type"] == 'raw':
                STATmerge.Serialize_Graph(handle, filename)
            if self.verbose:
                sys.stdout.write('done!\n')

    def spawn_sub_processes(self):
        sub_processes = {}
        chunks = len(self.trace_files) / self.options["limit"]
        if len(self.trace_files) % self.options["limit"] != 0:
            chunks += 1
        sys.stdout.write('spawning sub processes: 000%')
        sys.stdout.flush()
        for i in range(chunks):
            sys.stdout.write('\b\b\b\b%03u%%' % ((i + 1) / (chunks / 100.0)))
            sys.stdout.flush()
            tmp_file_prefix = 'tmp.%d' % i
            if i == chunks - 1:
                trace_files_subset = self.trace_files[i * self.options["limit"]:]
            else:
                trace_files_subset = self.trace_files[i * self.options["limit"]:(i + 1) * self.options["limit"]]
            command = [sys.executable, sys.argv[0]]
            for name, value in self.options.items():
                if name == "type":
                    value = "raw"
                if name == "fileprefix":
                    value = tmp_file_prefix
                if value is not None:
                    if self.args.arg_map[name].required_arg is True:
                        command.append('--%s=%s' % (name, str(value)))
                    else:
                        command.append('--%s' % (name))
            command.append('-c')
            command += trace_files_subset
            sub_processes[i] = [tmp_file_prefix, subprocess.Popen(command, universal_newlines=True)]
        sys.stdout.write('\n')
        sys.stdout.flush()
        return sub_processes

    def sync_sub_processes(self, sub_processes):
        total = len(sub_processes)
        remain = total
        handles = []
        file_path = ''
        sys.stdout.write('synching sub processes: 000%')
        sys.stdout.flush()
        while remain > 0:
            time.sleep(.1)
            for i in range(remain):
                key, (tmp_file_prefix, sub_process) = sub_processes.items()[i]
                ret = sub_process.poll()
                if ret is not None:
                    try:
                        if ret != 0:
                            raise Exception(ret, 'sub process %d of %d returned with error %d.' % (key, total, ret))
                        sys.stdout.write('\b\b\b\b%03u%%' % ((1 + total - remain) / (total / 100.0)))
                        sys.stdout.flush()

                        for j in range(8192):
                            num = ''
                            if j != 0:
                                num = '_%d' % j
                            file_path = tmp_file_prefix + num + '.raw'
                            if not os.path.exists(file_path):
                                break
                            current_handle = STATmerge.Deserialize_Graph(file_path)
                            if current_handle == -1:
                                raise Exception(-1, 'failed to deserialize file %s from sub process %d of %d.' % (file_path, key, total))
                            if (remain == total):
                                handles.append(current_handle)
                            else:
                                ret = STATmerge.Merge_Traces(handles[j], current_handle)
                                if ret == -1:
                                    raise Exception(-1, 'failed to merge handle %d into handle %d from sub process %d of %d.' % (handles[j], current_handle, key, total))
                            os.remove(file_path)
                    except Exception as e:
                        sys.stderr.write(repr(e) + '... Continuing without this subset\n')
                    finally:
                        del sub_processes[key]
                        remain -= 1
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    break
        sys.stdout.write('\n')
        sys.stdout.flush()
        return handles

if __name__ == '__main__':
    merger = StatMerger(StatTrace, StatMergerArgs)
    ret = merger.run()
    if ret != 0:
        sys.stderr.write('Merger failed\n')
        sys.exit(ret)
