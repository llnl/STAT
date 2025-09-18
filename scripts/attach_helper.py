#!/bin/env python
"""@package attach_helper
A helper script to translate resource manager job IDs into hostname:PID pairs
suitable for the Stack Trace Analysis Tool."""

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

import subprocess
import sys
import argparse

rms = {"alps":"aprun", "slurm":"srun", "lsf":"jsrun"}

def jobid_to_hostname_pid(rm, jobid, remoteshell):
    pids = []
    remotehosts = []
    remotehost = None

    if rm == None or rm.lower() == 'auto':
        rm = auto_detect_rm()
    if rm == None:
        return None, None, pids

    rm = rm.lower()
    if rm == 'slurm':
        proc = subprocess.Popen(["squeue", "-j", jobid, "-tr", "-o", '%B"'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = proc.stdout.read()
        lines = ret.splitlines()
        if not lines or lines[0].find("EXEC_HOST") != 0 or len(lines) < 2:
            return None, None, []
        if lines[1][-1] == '"':
            lines[1] = lines[1][:len(lines[1]) - 1]
        remotehosts = [lines[1]]
    if rm == 'lsf':
        proc = subprocess.Popen(["bjobs", "-noheader", "-X", "-o", "exec_host", jobid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = proc.stdout.read()
        lines = ret.splitlines()
        if not lines or lines[0].find("not found") != -1 or lines[0].find("-") != -1:
            return None, None, []
        tokens = lines[0].split('*')
        remotehosts = [tokens[1][:tokens[1].find(':')], tokens[2][:tokens[1].find(':')]]
    if rm == 'alps':
        proc = subprocess.Popen(["qstat", "-f", jobid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ret = proc.stdout.read()
        lines = ret.splitlines()
        for line in lines:
            if line.find("login_node_id") != -1:
                remotehosts = [line.split(' = ')[1]]
                break
    for remotehost in remotehosts:
        if remotehost != None:
            if remotehost.find('.localdomain') != -1:
                import socket
                domain_name =  socket.getfqdn().strip(socket.gethostname())
                remotehost = remotehost[:remotehost.find('.localdomain')]
            proc = subprocess.Popen([remoteshell, remotehost, "ps", "x"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ret = proc.stdout.read()
            if lines == []:
                continue
            lines = ret.splitlines()
            try:
                fmt = lines[0].split()
                pid_index = fmt.index('PID')
                for line in lines[1:]:
                    if line.find(rms[rm]) != -1:
                        pids.append(int(line.split()[pid_index]))
            except:
                pass
    if pids != []:
        return remotehost, rms[rm], pids
    return None, None, []

def auto_detect_rm():
    for rm_key in rms:
        ret = subprocess.call(["which", rms[rm_key]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret == 0:
            rm = rm_key
            return rm
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate job ID to hostname:PID')
    parser.add_argument('-r', '--rm', dest='rm', default=None)
    parser.add_argument('-s', '--remoteshell', dest='remoteshell', default='rsh')
    parser.add_argument('jobid')
    args = parser.parse_args()
    rm = vars(args)['rm']
    remoteshell = vars(args)['remoteshell']
    jobid = vars(args)['jobid']

    if rm == None:
        rm = auto_detect_rm()

    hostname, rm_exe, pids = jobid_to_hostname_pid(rm, jobid, remoteshell)
    if hostname == None or pids == []:
        sys.stderr.write('failed to find hostname:pid for specified jobid\n')
        sys.exit(1)
    for pid in pids:
        print("%s:%d" %(hostname, pid))
