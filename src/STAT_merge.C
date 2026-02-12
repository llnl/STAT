/*
Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Written by Gregory Lee [lee218@llnl.gov], Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <Python.h>
#include "graphlib.h"
#include <vector>
#include <string>
#include <errno.h>
#include "STAT_GraphRoutines.h"

using namespace std;

extern "C" {

//! a vector of generated graphs
vector<graphlib_graph_p> *gGraphs = NULL;

//! the highest rank task to represent
int gHighRank;

//! the bit merge routines
extern graphlib_functiontable_p gStatMergeFunctions;

//! the bit vector routines
extern graphlib_functiontable_p gStatBitVectorFunctions;

//! the count and representative routines
extern graphlib_functiontable_p gStatCountRepFunctions;

extern const char *gNodeAttrs[];
extern const char *gEdgeAttrs[];
extern int gNumNodeAttrs;
extern int gNumEdgeAttrs;

//! Initialize graphlib and set the graph routines
static PyObject *py_Init_Graphlib(PyObject *self, PyObject *args)
{
    graphlib_error_t graphlibError;

    if (!PyArg_ParseTuple(args, "i", &gHighRank))
    {
        fprintf(stderr, "Failed to parse args, expecting (int)\n");
        return Py_BuildValue("i", -1);
    }
    graphlibError = graphlib_Init();
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to init graphlib\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    statInitializeBitVectorFunctions();
    statInitializeMergeFunctions();
    statInitializeCountRepFunctions();

    return Py_BuildValue("i", 0);
}

//! Create a new graph and return the handle
static PyObject *py_newGraph(PyObject *self, PyObject *args)
{
    int handle;
    graphlib_graph_p newGraph;
    graphlib_error_t graphlibError;

#ifdef COUNTREP
    newGraph = createRootedGraph(0 | STAT_SAMPLE_COUNT_REP);
#else
    newGraph = createRootedGraph(0);
#endif

    if (gGraphs == NULL)
        gGraphs = new vector<graphlib_graph_p>;
    gGraphs->push_back(newGraph);
    handle = gGraphs->size() - 1;

    return Py_BuildValue("i", handle);
}

//! Translate a relative (local daemon) rank to an absolute (global MPI) rank
int statMergeRelativeRankToAbsoluteRank(int rank)
{
    return rank;
}

//! Add a stack trace to a previously generated graph
static PyObject *py_Add_Trace(PyObject *self, PyObject *args)
{
    int i, task, tid, count, size, nodeId, prevId, handle, bit, byte;
    int bvIndex, countIndex, repIndex, sumIndex, tcountIndex, tbvsumIndex, tbvIndex, functionIndex, sourceIndex, lineIndex;
    char *trace, *ptr, *next, *curFrame, *tmp;
    char path[BUFSIZE], prevPath[BUFSIZE];
    StatBitVectorEdge_t *newEdge = NULL, *newTEdge = NULL;
    StatCountRepEdge_t *countRepEdge = NULL;
    graphlib_nodeattr_t nodeAttr = {1,0,20,GRC_LIGHTGREY,0,0,(char *)"",-1,NULL};
    graphlib_edgeattr_t edgeAttr = {1,0,NULL,0,0,0,NULL};
    graphlib_graph_p currentGraph, graphPtr = NULL;
    graphlib_error_t graphlibError;

    if (!PyArg_ParseTuple(args, "iiiis", &handle, &task, &tid, &count, &trace))
    {
        fprintf(stderr, "Failed to parse args, expecting (int, int, string)\n");
        return Py_BuildValue("i", -1);
    }

    graphPtr = (*gGraphs)[handle];
#ifdef COUNTREP
    currentGraph = createRootedGraph(0 | STAT_SAMPLE_COUNT_REP);
#else
    currentGraph = createRootedGraph(0);
#endif
    if (currentGraph == NULL)
    {
        fprintf(stderr, "Error initializing graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    edgeAttr.attr_values = (void **)calloc(1, gNumEdgeAttrs * sizeof(void *));
    if (edgeAttr.attr_values == NULL)
    {
        fprintf(stderr, "%s: Error callocing %d edgeAttr.attr_values\n", strerror(errno), gNumEdgeAttrs);
        return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
    }
    newEdge = initializeBitVectorEdge(gHighRank);
    if (newEdge == NULL)
    {
        fprintf(stderr, "Failed to initialize newEdge\n");
        return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
    }
    newEdge->bitVector[task / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(task % STAT_BITVECTOR_BITS);
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "bv", &bvIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key bv\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "count", &countIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key count\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "rep", &repIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key rep\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "sum", &sumIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key sum\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "tbv", &tbvIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key tbv\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "tcount", &tcountIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key tcount\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getEdgeAttrIndex(currentGraph, "tbvsum", &tbvsumIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key tbvsum\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    edgeAttr.attr_values[bvIndex] = newEdge;
    countRepEdge = getBitVectorCountRep(newEdge, statMergeRelativeRankToAbsoluteRank);
    if (countRepEdge == NULL)
    {
        fprintf(stderr, "Failed to translate bit vector into count + representative\n");
        return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
    }
#ifdef COUNTREP
    edgeAttr.attr_values[countIndex] = statCopyEdgeAttr("count", (void *)&countRepEdge->count);
    edgeAttr.attr_values[repIndex] = statCopyEdgeAttr("rep" , (void *)&countRepEdge->representative);
    edgeAttr.attr_values[sumIndex] = statCopyEdgeAttr("sum" , (void *)&countRepEdge->checksum);
#endif
//    newTEdge = initializeBitVectorEdge(64);
//    if (newTEdge == NULL)
//    {
//        fprintf(stderr, "Failed to initialize newTEdge\n");
//        return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
//    }
//    tid = tid % 64;
//    newTEdge->bitVector[tid / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(tid % STAT_BITVECTOR_BITS);
//    edgeAttr.attr_values[tbvIndex] = newTEdge;
    edgeAttr.attr_values[tcountIndex] = statCopyEdgeAttr("tcount", (void *)&countRepEdge->count);
    int64_t tbvsum = countRepEdge->checksum * (task + 1);
    edgeAttr.attr_values[tbvsumIndex] = statCopyEdgeAttr("tbvsum", (void *)&tbvsum);
    statFreeCountRepEdge(countRepEdge);

    graphlibError = graphlib_getNodeAttrIndex(currentGraph, "function", &functionIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key function\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getNodeAttrIndex(currentGraph, "source", &sourceIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key source\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    graphlibError = graphlib_getNodeAttrIndex(currentGraph, "line", &lineIndex);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Failed to get node attr index for key line\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }
    snprintf(path, BUFSIZE, "/");
    prevId = 0;
    next = trace;
    ptr = trace;
    for (i = 0; i < count; i++)
    {
        size = 1;
        ptr = next;
        while (*ptr != '\n')
        {
            size++;
            ptr += 1;
        }
        curFrame = (char *)malloc((size + 1) * sizeof(char));
        if (curFrame == NULL)
        {
            fprintf(stderr, "%s: Error mallocing %d bytes for curFrame\n", strerror(errno), size + 1);
            return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
        }
        snprintf(curFrame, size, "%s", next);
        string curFrameString = curFrame;
        nodeAttr.attr_values = (void **)calloc(1, gNumNodeAttrs * sizeof(void *));
        if (nodeAttr.attr_values == NULL)
        {
            fprintf(stderr, "%s: Error callocing %d nodeAttr.attr_values\n", strerror(errno), gNumNodeAttrs);
            return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
        }
        if (curFrameString.find("@") != string::npos)
        {
            string function = curFrameString.substr(0, curFrameString.find("@"));
            string sourceLine = curFrameString.substr(curFrameString.find("@") + 1);
            string source = sourceLine.substr(0, sourceLine.find(":"));
            string line = sourceLine.substr(sourceLine.find(":") + 1);
            tmp = strdup(function.c_str());
            if (tmp == NULL)
            {
                fprintf(stderr, "Failed to strdup function %s\n", function.c_str());
                return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
            }
            nodeAttr.attr_values[functionIndex] = (void *)tmp;
            tmp = strdup(source.c_str());
            if (tmp == NULL)
            {
                fprintf(stderr, "Failed to strdup source %s\n", source.c_str());
                return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
            }
            nodeAttr.attr_values[sourceIndex] = (void *)tmp;
            tmp = strdup(line.c_str());
            if (tmp == NULL)
            {
                fprintf(stderr, "Failed to strdup line %s\n", line.c_str());
                return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
            }
            nodeAttr.attr_values[lineIndex] = (void *)tmp;
        }
        else
        {
            tmp = strdup(curFrame);
            nodeAttr.attr_values[functionIndex] = (void *)tmp;
        }
        next += size;
        snprintf(prevPath, BUFSIZE, "%s", path);
        snprintf(path, BUFSIZE, "%s%s", prevPath, curFrame);
        free(curFrame);
        nodeId = statStringHash(path);

        graphlibError = graphlib_addNode(currentGraph, nodeId, &nodeAttr);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            fprintf(stderr, "Failed to add node\n");
            return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
        }

        statFreeNodeAttrs(nodeAttr.attr_values, currentGraph);
        nodeAttr.attr_values = NULL;
        graphlibError = graphlib_addDirectedEdge(currentGraph, prevId, nodeId, &edgeAttr);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            fprintf(stderr, "Failed to add edge\n");
            return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
        }
        prevId = nodeId;
    }

    graphlibError = graphlib_mergeGraphs(graphPtr, currentGraph);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error merging graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    statFreeEdgeAttrs(edgeAttr.attr_values, currentGraph);
    edgeAttr.attr_values = NULL;
    graphlibError = graphlib_delGraph(currentGraph);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error deleting graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    return Py_BuildValue("i", 0);
}

//! Merge 2 graphs
static PyObject *py_Merge_Traces(PyObject *self, PyObject *args)
{
    int handle1, handle2;
    graphlib_graph_p currentGraph, graphPtr;
    graphlib_error_t graphlibError;

    if (!PyArg_ParseTuple(args, "ii", &handle1, &handle2))
    {
        fprintf(stderr, "Failed to parse args, expecting (int, int, string)\n");
        return Py_BuildValue("i", -1);
    }
    graphPtr = (*gGraphs)[handle1];
    currentGraph = (*gGraphs)[handle2];

    graphlibError = graphlib_mergeGraphs(graphPtr, currentGraph);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error merging graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    return Py_BuildValue("i", 0);
}

//! Serialze the graph to a file
static PyObject *py_Serialize_Graph(PyObject *self, PyObject *args)
{
    char *buf, *fileName;
    int handle, ret;
    uint64_t bufLen;
    FILE *f;
    graphlib_graph_p graphPtr = NULL;
    graphlib_error_t graphlibError;

    if (!PyArg_ParseTuple(args, "is", &handle, &fileName))
    {
        fprintf(stderr, "Failed to parse args, expecting (int, string)\n");
        return Py_BuildValue("i", -1);
    }

    graphPtr = (*gGraphs)[handle];

    graphlibError = graphlib_serializeBasicGraph(graphPtr, &buf, &bufLen);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "%d Error serializing graph %d\n", graphlibError, handle);
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    f = fopen(fileName, "w");
    if (f == NULL)
    {
        fprintf(stderr, "%s: Error opening file %s\n", strerror(errno), fileName);
        return Py_BuildValue("i", -1);
    }

    ret = fwrite(buf, sizeof(char), bufLen, f);
    if (ret != (int)bufLen)
    {
        fprintf(stderr, "%s: %d Error writing serialized graph %d to file %s\n", strerror(errno), ret, handle, fileName);
        return Py_BuildValue("i", -1);
    }

    ret = fclose(f);
    if (ret != 0)
    {
        fprintf(stderr, "%s: %d Error closing file %s\n", strerror(errno), ret, fileName);
        return Py_BuildValue("i", -1);
    }

    return Py_BuildValue("i", 0);
}

//! Deserialize the graph from a file and return its handle
static PyObject *py_Deserialize_Graph(PyObject *self, PyObject *args)
{
    char *buf, *fileName;
    long bufLen;
    int handle, ret;
    FILE *f;
    graphlib_graph_p graphPtr = NULL;
    graphlib_error_t graphlibError;

    if (!PyArg_ParseTuple(args, "s", &fileName))
    {
        fprintf(stderr, "Failed to parse args, expecting (int, string)\n");
        return Py_BuildValue("i", -1);
    }

    f = fopen(fileName, "r");
    if (f == NULL)
    {
        fprintf(stderr, "%s: Error opening file %s\n", strerror(errno), fileName);
        return Py_BuildValue("i", STAT_FILE_ERROR);
    }

    ret = fseek(f, 0, SEEK_END);
    if (ret != 0)
    {
        fprintf(stderr, "%s: %d Error seeking file %s\n", strerror(errno), ret, fileName);
        return Py_BuildValue("i", STAT_FILE_ERROR);
    }

    bufLen = ftell(f);
    if (bufLen < 0)
    {
        fprintf(stderr, "%s: %ld Error ftell file %s\n", strerror(errno), bufLen, fileName);
        return Py_BuildValue("i", STAT_FILE_ERROR);
    }

    ret = fseek(f, 0, SEEK_SET);
    if (ret != 0)
    {
        fprintf(stderr, "%s: %d Error seeking file %s\n", strerror(errno), ret, fileName);
        return Py_BuildValue("i", -1);
    }

    buf = (char *)malloc(bufLen * sizeof(char));
    if (buf == NULL)
    {
        fprintf(stderr, "%s: Error allocating %ld bytes for file %s\n", strerror(errno), bufLen, fileName);
        return Py_BuildValue("i", STAT_ALLOCATE_ERROR);
    }

    ret = fread(buf, bufLen, 1, f);
    if (ret != 1)
    {
        fprintf(stderr, "%s: Error reading serialized graph from file %s.  %d of %ld bytes read\n", strerror(errno), fileName, ret, bufLen);
        return Py_BuildValue("i", STAT_FILE_ERROR);
    }

    ret = fclose(f);
    if (ret != 0)
    {
        fprintf(stderr, "%s: %d Error closing file %s\n", strerror(errno), ret, fileName);
        return Py_BuildValue("i", STAT_FILE_ERROR);
    }

#ifdef COUNTREP
    graphlibError = graphlib_deserializeBasicGraph(&graphPtr, gStatCountRepFunctions, buf, (unsigned int)bufLen);
#else
    graphlibError = graphlib_deserializeBasicGraph(&graphPtr, gStatMergeFunctions, buf, (unsigned int)bufLen);
#endif
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error serializing graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    if (gGraphs == NULL)
        gGraphs = new vector<graphlib_graph_p>;
    gGraphs->push_back(graphPtr);
    handle = gGraphs->size() - 1;

    return Py_BuildValue("i", handle);
}

//! Output a graph to a .dot file
static PyObject *py_Output_Graph(PyObject *self, PyObject *args)
{
    char *fileName;
    int handle;
    graphlib_error_t graphlibError;
    graphlib_graph_p graphPtr = NULL;

    if (!PyArg_ParseTuple(args, "is", &handle, &fileName))
    {
        fprintf(stderr, "Failed to parse args, expecting (int, string)\n");
        return Py_BuildValue("i", -1);
    }

    graphPtr = (*gGraphs)[handle];

    graphlibError = graphlib_colorGraphByLeadingEdgeAttr(graphPtr, "tbvsum");
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "graphlib error coloring graph by leading edge attr\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    graphlibError =  graphlib_scaleNodeWidth(graphPtr, 80, 160);
    if ( GRL_IS_FATALERROR(graphlibError) )
    {
        fprintf(stderr, "graphlib error scaling node width\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    graphlibError = graphlib_exportGraph(fileName, GRF_DOT, graphPtr);
    if ( GRL_IS_FATALERROR(graphlibError) )
    {
        fprintf(stderr, "graphlib error exporting graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    graphlibError = graphlib_delGraph(graphPtr);
    if ( GRL_IS_FATALERROR(graphlibError) )
    {
        fprintf(stderr, "graphlib error deleting graph\n");
        return Py_BuildValue("i", STAT_GRAPHLIB_ERROR);
    }

    return Py_BuildValue("i", 0);
}

struct module_state {
    PyObject *error;
};

//! The Python method table for this module
static PyMethodDef _STATmergeMethods[] = {
    {"Init_Graphlib", py_Init_Graphlib, METH_VARARGS, "graphlib init."},
    {"New_Graph", py_newGraph, METH_VARARGS, "create new graph."},
    {"Add_Trace", py_Add_Trace, METH_VARARGS, "trace generator."},
    {"Merge_Traces", py_Merge_Traces, METH_VARARGS, "trace merger."},
    {"Serialize_Graph", py_Serialize_Graph, METH_VARARGS, "serialize graph."},
    {"Deserialize_Graph", py_Deserialize_Graph, METH_VARARGS, "deserialize graph."},
    {"Output_Graph", py_Output_Graph, METH_VARARGS, "graph exporter."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#if PY_MAJOR_VERSION >= 3
static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_STATmerge",
        NULL,
        sizeof(struct module_state),
        //myextension_methods,
        _STATmergeMethods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit__STATmerge(void)

#else
#define INITERROR return

//! The Python initialization routine for this module
void init_STATmerge()
#endif
{
#if PY_MAJOR_VERSION >=3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_STATmerge", _STATmergeMethods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("myextension.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

}
