#!/bin/env python

"""@package STATview
Visualizes dot graphs outputted by STAT."""

__copyright__ = """Modifications Copyright (C) 2022-2025 Intel Corporation
SPDX-License-Identifier: BSD-3-Clause"""
__license__ = """Produced by Intel Corporation for Lawrence Livermore National Security, LLC.
Written by Matti Puputti matti.puputti@intel.com, M. Oguzhan Karakaya oguzhan.karakaya@intel.com
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Abdul Basit Ijaz <abdul.b.ijaz@intel.com>", "Matti Puputti <matti.puputti@intel.com>", "M. Oguzhan Karakaya <oguzhan.karakaya@intel.com>"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 3
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import sys
import os
import logging
import re
from gdb import GdbDriver, check_lines

def clean_cpp_template_brackets_and_call_signature(string):
    """
    Prettify function name output by Intel(R) Distribution for GDB*.
    Long template parameters and function signatures are collapsed.
    """
    sub_strings = []
    (found, begin, end, counter) = (False, 0, 0, 0)
    for idx, char in enumerate(string):
        if char == "<" and not found:
            found = True
            begin = idx
            counter = 1
        elif char == "<" and found:
            counter += 1
        elif char == ">" and found and counter > 1:
            counter -= 1
        elif char == ">" and found and counter == 1:
            end = idx
            counter -= 1
            found = False
            sub_strings.append(string[begin:end+1:])
    # Removing template variables from the function name
    for sub_string in sub_strings:
        string = string.replace(sub_string, "<...>")
    # Removing call signature for the function name
    ptrn = re.compile(r"\([^()]+\)(?=[^()]*$)")
    match = ptrn.findall(string)
    if match:
        string = string.replace(match[-1], "(...)")

    return string

def parse_frameinfo_from_backtrace(string):
    """
    Takes the gdb backtrace output and call
    clean_cpp_template_brackets_and_call_signature function over
    extracted function names. See oneapi_gdb_test.py for
    sample inputs and outputs.
    """
    function = 'Unknown'
    source = '??'
    linenum = 0
    error = True

    frame_fromat = re.compile(
        r"^#\d+\s+(?:0x\S+)*(?: in )*(.*)\s\(.*\)\s(?:at|from)\s([^:]+)(?::)?(\d+)?$")
    match = frame_fromat.match(string)
    if match:
        error = False
        function, source, linenum = match.groups()
        if linenum is None:
            linenum = 0
        else:
            linenum = int(linenum)
    function = clean_cpp_template_brackets_and_call_signature(function)
    return {'function':function,
            'source':source,
            'linenum':linenum,
            'error':error}

class OneAPIGdbDriver(GdbDriver):
    """ A class to drive Intel(R) Distribution for GDB* """

    gdb_command = 'gdb-oneapi'

    def get_thread_list(self):
        """
        Gets the list of threads in the target process. For
        Intel(R) Distribution for GDB* this function extracts active
        threads ID list.

        NOTE: STAT_INTELGT_EXPR_FILTER is validated for few security checks
        before it is evaluated by GDB.
        """
        filter_expr_str = os.environ.get("STAT_INTELGT_EXPR_FILTER")

        cmd = (
            'thread apply all -q -c printf "%d.%d\\n", '
            '$_inferior, $_thread'
        )

        # Thread apply commmand may end up sending commands like "shell rm *",
        # so this list ensure to not allow these system commands.
        danger_list = ['shell', 'call', 'python', 'source', 'dump', 'restore']

        if filter_expr_str and filter_expr_str.strip():
            filter_expr_str = filter_expr_str.strip()

            # Check for unsecure characters and also if it is a valid input
            # for GDB.
            # Each line may be treated as separate command in GDB and
            # fail the expression evaluation.
            if '\n' in filter_expr_str or '\r' in filter_expr_str:
                logging.error('STAT_INTELGT_EXPR_FILTER contains newline')
                logging.error(f'Invalid expression: "{repr(filter_expr_str)}"')
            # Check for dangerous GDB commands that could execute arbitrary code.
            elif any(keyword in filter_expr_str.lower()
                     for keyword in danger_list):
                logging.error(f'STAT_INTELGT_EXPR_FILTER should not use: '
                              f'{danger_list}')
                logging.error(f'Invalid expression: "{filter_expr_str}"')
            else:
                # Check if the expression is valid by evaluating it with GDB.
                validate_cmd = f'print {filter_expr_str}'
                logging.debug(f'Validating filter expression: {validate_cmd}')
                validation_result = self.communicate(validate_cmd)

                logging.debug(f'GDB validation result: {validation_result}')
                is_valid = False

                if validation_result:
                    for line in validation_result:
                        line_str = line.strip()

                        # GDB successfully evaluated the expression.
                        if line_str.startswith('$') and '=' in line_str:
                            is_valid = True
                            logging.info(f'Using expression filter: '
                                         f'"{filter_expr_str}"')
                            break

                        if re.search(r"(No symbol.*in current context) \
                                     |(cannot subscript something of type)",
                                     line_str, re.IGNORECASE):
                            is_valid = True
                            logging.info(f'Using expression filter '
                                         f'"{filter_expr_str}"')
                            break

                if not is_valid:
                    logging.warning(f'STAT_INTELGT_EXPR_FILTER expression has'
                                    f'syntax error: "{filter_expr_str}"')
                    logging.warning(f'GDB returned: {validation_result}')
                else:
                    cmd = (
                        f'thread apply all -q -s '
                        f'eval "printf %s, $_inferior, $_thread", '
                        f'({filter_expr_str} ? "\\\"%d.%d\\n\\\"" : "\\\"\\\"")'
                    )

        # Require only in case of cray-stat.
        try:
            self.flushInput()
        except AttributeError:
            # Function is only available in cray-stat.
            pass

        logging.info(f'gdb-oneapi: {cmd}')

        tids = self.communicate(cmd)
        if not tids:
            if filter_expr_str:
                logging.warning(f'Expression filter "{filter_expr_str}" '
                                'resulted in zero threads')
            return []

        max_threads = os.environ.get("STAT_INTELGT_MAX_THREADS")

        if max_threads and max_threads.strip():
            try:
                int_max_threads = int(max_threads)
                if int_max_threads <= 0:
                    logging.warning(f'STAT_INTELGT_MAX_THREADS value {max_threads}'
                                    'must be positive, ignoring')
                else:
                    tids = tids[:int_max_threads]
                    logging.info(f'Limiting to first {int_max_threads} '
                                 f'threads for callstack capture')
            except ValueError:
                logging.warning(f'STAT_INTELGT_MAX_THREADS value "{max_threads}" '
                                'is invalid, must be a positive integer')

        logging.debug(f'Filtered thread IDs list: {tids}')

        return tids

    def bt(self, thread_id):
        """
        Gets a backtrace from the requested thread id.
        returns list of frames, where each frame is a map of attributes.
        """
        logging.info(f'GDB thread bt ID {thread_id}')
        ret = []

        wpfa = "with print frame-arguments none --"
        lines = self.communicate(f"{wpfa} thread apply {thread_id} bt")

        lines = lines[2:]
        for line_num, line in enumerate(lines):
            if line[0] != '#':
                continue
            logging.debug(f'Parsing line #{line_num}: {line}')
            ret.append(parse_frameinfo_from_backtrace(line))
        return ret

    def attach(self):
        """
        Attaches to the target process and checkes if there is
        an error related to gdbserver-gt at the gdb console output.
        """
        logging.info('GDB attach to PID %d' %(self.pid))
        lines = self.communicate("attach %d" %self.pid)
        # We need to check if there is an error with starting
        # gdbserver-gt and inform the user for possible missing
        # device backtrace features.
        ptrn = re.compile(
            r"^intelgt\:\s*(\S*)\s*failed to start\.")
        for line in lines:
            match = ptrn.match(line)
            if match:
                gdbserver_name = match.groups()[0]
                logging.error("""
OneAPI %s initialization failed for process %d. Device backtrace will not be available.""",
                    gdbserver_name, self.pid)
        logging.debug('%s', repr(lines))
        return check_lines(lines)

    def target_core(self, path):
        """
        Intel(R) Distribution for GDB* targets a core dump
        file given by path.
        """
        logging.info('GDB open core %s' %(path))
        self.communicate("target core %s" %path)

if __name__ == "__main__":
    gdb = OneAPIGdbDriver(0, 'debug', 'log.txt')
    if gdb.launch() is False:
        sys.stderr.write('gdb launch failed\n')
        sys.exit(1)
    gdb.target_core(sys.argv[1])
    thread_ids = gdb.get_thread_list()
    thread_ids.sort()
    for tid in thread_ids:
        bt = gdb.bt(tid)
        print("Thread", tid)
        for i, frame in enumerate(bt):
            print("%d) %s:%s" %(i, frame['source'], frame['linenum']))
        print()
    gdb.close()
