#!/bin/env python

"""@package STATview
Visualizes dot graphs outputted by STAT."""

__copyright__ = """Copyright (c) 2007-2022, Lawrence Livermore National Security, LLC."""
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
import pwd

try:
    from gdb import GdbDriver
    from cuda_gdb import CudaGdbDriver
    from roc_gdb import RocGdbDriver
    from oneapi_gdb import OneAPIGdbDriver
except Exception as e:
    print(e)
gdb_instances = {}

def new_gdb_instance(pid, gdb_type='gdb'):

    # if TMPDIR is not set, you may get this error and fail to see GPU devices:
    #    The CUDA driver could not allocate operating system resources for attaching to the application.
    #
    #    An error occurred while in a function called from GDB.
    #    Evaluation of the expression containing the function
    #    (cudbgApiAttach) will be abandoned.
    #    When the function is done executing, GDB will silently stop.
    if "TMPDIR" not in os.environ:
        os.environ["TMPDIR"]="/var/tmp"

    try:
        if 'cuda-gdb' in os.environ['STAT_GDB']:
            gdb = CudaGdbDriver(pid, 'error', 'stderr')
        elif 'rocgdb' in os.environ['STAT_GDB']:
            gdb = RocGdbDriver(pid, 'error', 'stderr')
        elif 'gdb-oneapi' in os.environ['STAT_GDB']:
            gdb = OneAPIGdbDriver(pid, 'error', 'stderr')
        else:
            gdb = GdbDriver(pid, 'error', 'stderr')
    except:
        gdb = GdbDriver(pid, 'error', 'stderr')
    if gdb.launch() is False:
        return -1
    gdb_instances[pid] = gdb
    return pid

def attach(pid):
    gdb_instances[pid].attach()
    return 0

def detach(pid):
    gdb_instances[pid].detach()
    return 0

def pause(pid):
    gdb_instances[pid].pause()
    return 0

def resume(pid):
    gdb_instances[pid].resume()
    return 0

def get_all_host_traces(pid, retries=5, retry_frequency=1000, cuda_quick=0):
    ret = ''
    tids = gdb_instances[pid].get_thread_list()
    tids.sort()
    for thread_id in tids:
        bt = gdb_instances[pid].bt(thread_id)
        bt.reverse()
        for frame in bt:
            ret += '%s@%s:%d\n' %(frame['function'], frame['source'], frame['linenum'])
        ret += '#endtrace\n'
    return ret

def get_all_device_traces(pid, retries=5, retry_frequency=1000, cuda_quick=0):
    if type(gdb_instances[pid]) != CudaGdbDriver:
        return ''
    ret = ''
    threads = gdb_instances[pid].get_cuda_threads(retries, retry_frequency)
    for thread in threads:
        ret += '#count#%d\n' %(thread['count'])
        if cuda_quick == 1:
            ret += '?@%s:%d\n' %(thread['filename'], thread['linenum'])
        else:
            gdb_instances[pid].cuda_block_thread_focus(thread['start_block'], thread['start_thread'])
            bt = gdb_instances[pid].cuda_bt()
            bt.reverse()
            for frame in bt:
                ret += '%s@%s:%d\n' %(frame['function'], frame['source'], frame['linenum'])
        ret += '#endtrace\n'
    return ret

if __name__ == "__main__":
    pids = sys.argv[1:]
    for pid in pids:
        pid = int(pid)
        handle = new_gdb_instance(pid)
        if handle == -1:
            sys.stderr.write('gdb launch failed\n')
            sys.exit(1)
        if attach(handle) != 0:
            sys.stderr.write('gdb attach failed\n')
            sys.exit(1)
    for pid in pids:
        pid = int(pid)
        host_traces = get_all_host_traces(pid)
        resume(pid)
        pause(pid)
        device_traces = get_all_device_traces(pid)
    #    devices = gdb_instances[pid].get_cuda_devices()
    #    print('\ndevices')
    #    for device in devices:
    #        if gdb_instances[pid].cuda_device_focus(device):
    #            print('device focused', device)
    #    kernels = gdb_instances[pid].get_cuda_kernels()
    #    print('\nkernels')
    #    for kernel in kernels:
    #        if gdb_instances[pid].cuda_kernel_focus(kernel):
    #            print('kernel focused', kernel)
    #    bt = gdb_instances[pid].cuda_bt()
    #    for frame in bt:
    #        print(frame)
    #    print()
    #    gdb_instances[pid].kill()
    for pid in pids:
        pid = int(pid)
        print('\ndetaching %d' %(pid))
        detach(pid)
