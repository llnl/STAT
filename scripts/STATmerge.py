"""@package STATmerge
Interface to the _STATmerge module, used to generate merged stack traces."""

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

import _STATmerge


## Need to call this function first
#  \param high_rank - the highest ranked task in the graphs to be created
#  \return 0 on success
def Init_Graphlib(high_rank):
    return _STATmerge.Init_Graphlib(high_rank)


## Create a new graph
#  \return an integer handle for the graph on success or -1 on error
def New_Graph():
    return _STATmerge.New_Graph()


## Add a trace to the graph
#  \param handle - the integer handle for the graph to add the trace to
#  \param rank - the rank of the task associated with the trace
#  \param tid - the thread ID of the task associated with the trace
#  \param trace - the stack trace (list of string frames) with the earliest frame (i.e., _start or main) first
#  \return 0 on success
def Add_Trace(handle, rank, tid, trace):
    trace_string = ''
    count = len(trace)
    # convert the list of frames to a string of frames separated by newlines
    for frame in trace:
        trace_string += frame + '\n'
    return _STATmerge.Add_Trace(handle, rank, tid, count, trace_string)


## Merge a graph to the graph
#  \param handle1 - the integer handle for the graph to add the trace to
#  \param handle2 - the integer handle for the graph to be added
#  \return 0 on success
def Merge_Traces(handle1, handle2):
    return _STATmerge.Merge_Traces(handle1, handle2)


## Serialize a graph to a file
#  \param handle - the integer handle for the graph
#  \param filename - the file to dump to
#  \return 0 on success
def Serialize_Graph(handle, filename):
    return _STATmerge.Serialize_Graph(handle, filename)


## Deserialize a graph from a file
#  \param filename - the file to read from
#  \return an integer handle for the graph on success or -1 on error
def Deserialize_Graph(filename):
    return _STATmerge.Deserialize_Graph(filename)


## Output a graph to a .dot file
#  \param handle - the integer handle for the graph to add the trace to
#  \param filename - the name of the .dot file to create
#  \return 0 on success
def Output_Graph(handle, filename):
    return _STATmerge.Output_Graph(handle, filename)
