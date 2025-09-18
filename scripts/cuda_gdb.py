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
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import sys
import os
import logging
import time
from gdb import *

class CudaGdbDriver(GdbDriver):
    """A class to drive CUDA GDB"""

    input_prompt = '(cuda-gdb)'
    try:
        gdb_command = os.environ['STAT_GDB']
    except:
        gdb_command = 'cuda-gdb'
    gdb_args = ['--cuda-use-lockfile=0']

    def __init__(self, pid, log_level='error', log_file='stderr'):
        """Constructor"""
        GdbDriver.__init__(self, pid, log_level, log_file)
        logging.info('CUDA GDB init')

    def get_cuda_threads(self, retries=5, retry_frequency=10000):
        """Gets a list of cuda threads.
           returns list of threads, where each thread is a map of attributes"""

        logging.info('get cuda threads')
        ret = []
        lines = self.communicate("info cuda threads")
        # Expected format: BlockIdx ThreadIdx To BlockIdx ThreadIdx Count Virtual PC Filename Line
        logging.debug('%s' %(repr(lines)))
        if check_lines(lines) == False:
            return ret
        while "Focus not set on any active CUDA kernel." in lines and retries > 0:
            logging.debug('retry %d %d' %(retries, retry_frequency))
            self.resume()
            time.sleep(retry_frequency / 1000000)
            self.pause()
            retries -= 1
            lines = self.communicate("info cuda threads")
            logging.debug('%s' %(repr(lines)))
        start_block_index = 0; start_thread_index = 1; end_block_index = 2
        end_thread_index = 3; count_index = 4; pc_index = 5; filename_index = 6
        line_index = 7
        lines = lines[2:]
        for line in lines:
            if line[0] == '*':
                line = line[1:]
            error = False; start_block = ''; start_thread = ''; end_block = ''
            end_thread = ''; count = 0; filename = '??'; linenum = 0
            try:
                split_line = line.split()
                start_block = split_line[start_block_index]
                start_thread = split_line[start_thread_index]
                end_block = split_line[end_block_index]
                end_thread = split_line[end_thread_index]
                count = int(split_line[count_index])
                filename = split_line[filename_index]
                linenum = int(split_line[line_index])
            except Exception as e:
                logging.warning('Failed to get source line from "%s": %s' %(line, repr(e)))
                error = True
            ret.append({'start_block':start_block, 'start_thread':start_thread, 'end_block':end_block, 'end_thread':end_thread, 'count':count, 'filename':filename, 'linenum':linenum, 'error':error})
        return ret

    def cuda_thread_focus(self, thread_spec):
        """Sets the thread focus to the requested CUDA thread"""
        logging.info('cuda thread focus')
        lines = self.communicate("cuda thread %s" %thread_spec)
        logging.debug('%s' %(repr(lines)))
        return check_lines(lines)


    def cuda_block_thread_focus(self, block_spec, thread_spec):
        """Sets the thread focus to the requested CUDA block and thread"""
        logging.info('cuda thread focus')
        lines = self.communicate("cuda block %s thread %s" %(block_spec, thread_spec))
        logging.debug('%s' %(repr(lines)))
        return check_lines(lines)

    def get_cuda_devices(self):
        """Gets a list of cuda device indexes"""
        logging.info('get cuda devices')
        devices = []
        lines = self.communicate("info cuda devices")
        logging.debug('%s' %(repr(lines)))
        if check_lines(lines) == False:
            return devices
        for line in lines:
            line = line.split()
            try:
                if line[0].isdigit():
                    devices.append(int(line[0]))
                elif line[0] == '*' and line[1].isdigit():
                    devices.append(int(line[1]))
            except Exception as e:
                pass
        return devices

    def cuda_device_focus(self, device_index):
        """Sets the device focus to the requested CUDA device index"""
        logging.info('cuda device focus')
        lines = self.communicate("cuda device %d" %device_index)
        logging.debug('%s' %(repr(lines)))
        for line in lines:
            if 'Request cannot be satisfied' in line:
                return False
        return check_lines(lines)

    def get_cuda_kernels(self):
        """Gets a list of cuda kernel indexes"""
        logging.info('get cuda kernels')
        kernels = []
        lines = self.communicate("info cuda kernels")
        logging.debug('%s' %(repr(lines)))
        if check_lines(lines) == False:
            return kernels
        for line in lines:
            line = line.split()
            try:
                if line[0].isdigit():
                    kernels.append(int(line[0]))
                elif line[0] == '*' and line[1].isdigit():
                    kernels.append(int(line[1]))
            except Exception as e:
                pass
        return kernels

    def cuda_kernel_focus(self, kernel_index):
        """Sets the kernel focus to the requested CUDA kernel index"""
        logging.info('cuda kernel focus')
        lines = self.communicate("cuda kernel %d" %kernel_index)
        logging.debug('%s' %repr((lines)))
        return check_lines(lines)

    def cuda_bt(self):
        """Gets a backtrace from the current cuda thread.
           returns list of frames, where each frame is a map of attributes"""
        logging.info('cuda bt')
        ret = []
        lines = self.communicate("bt")
        logging.debug('%s' %(repr(lines)))
        if check_lines(lines) == False:
            return ret

        patched_lines = []
        # some frames may span multiple lines
        # each frame begins with #<frame_number>
        for i, line in enumerate(lines):
            if line[0] != '#':
                continue
            new_line = line
            j = i + 1
            while j < len(lines) and lines[j][0] != '#':
                new_line += lines[j]
                j += 1
            patched_lines.append(new_line)

        for line in patched_lines:
            split_line = line.split()
            function = ''
            source = ''
            linenum = 0
            error = False
            try:
                function_index = 1
                source_line_index = 0
                if ('in' in split_line):
                    function_index = split_line.index('in') + 1
                if ('at' in split_line):
                    source_line_index = split_line.index('at') + 1
                function = split_line[function_index] # do we want to strip off args?
                #function = split_line[function_index].split('(')[0]
                source_line = split_line[source_line_index].split(':')
                source = source_line[0]
                linenum = int(source_line[1])
            except Exception as e:
                logging.debug('Failed to get frame info from "%s": %s' %(line, repr(e)))
                error = True
            ret.append({'error':error, 'function':function, 'source':source, 'linenum':linenum, 'frame':line})
        return ret


if __name__ == "__main__":
    gdb = CudaGdbDriver(int(sys.argv[1]), 'debug', 'log.txt')
    if gdb.launch() is False:
        sys.stderr.write('gdb launch failed\n')
        sys.exit(1)
    gdb.attach()
    tids = gdb.get_thread_list()
    tids.sort()
    for thread_id in tids:
        bt = gdb.bt(thread_id)
        for frame in bt:
            print(frame)
        print()
    gdb.resume()
    gdb.pause()
#    bt = gdb.bt(1)
#    print(bt)
    devices = gdb.get_cuda_devices()
    print('\ndevices')
    for device in devices:
        if gdb.cuda_device_focus(device):
            print('device focused', device)
    kernels = gdb.get_cuda_kernels()
    print('\nkernels')
    for kernel in kernels:
        if gdb.cuda_kernel_focus(kernel):
            print('kernel focused', kernel)
    bt = gdb.cuda_bt()
#    for frame in bt:
#        print(frame)
#    print()
    threads = gdb.get_cuda_threads()
    for thread in threads:
        gdb.cuda_block_thread_focus(thread['start_block'], thread['start_thread'])
        bt = gdb.cuda_bt()
        print(thread['start_block'], thread['start_thread'], thread['count'])
        for frame in bt:
            print(frame)
        print()
        break
#    gdb.kill()
    print('\ndetaching')
    gdb.detach()
