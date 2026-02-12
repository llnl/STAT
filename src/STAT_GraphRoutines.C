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

#include "STAT_GraphRoutines.h"

using namespace std;

//! the bit vector routines
graphlib_functiontable_p gStatBitVectorFunctions = NULL;

//! the bit vector filter routines to append bit vectors
graphlib_functiontable_p gStatMergeFunctions = NULL;

//! the bit vector to reorder the bit vectors by MPI rank
graphlib_functiontable_p gStatReorderFunctions = NULL;

//! the count and representative routines
graphlib_functiontable_p gStatCountRepFunctions = NULL;

const char *gNodeAttrs[] = {"function", "source", "line", "module", "offset", "vars", "pc"};
const char *gEdgeAttrs[] = {"bv", "count", "rep", "sum", "tbv", "tcount", "tbvsum"};
int gNumNodeAttrs;
int gNumEdgeAttrs;

//! the final bit vector width
int gStatGraphRoutinesTotalWidth;

//! the input list of bit vector widths
int *gStatGraphRoutinesEdgeLabelWidths;

//! the current index into the bit vector
int gStatGraphRoutinesCurrentIndex;

//! the ranks list for the current bit vector
int *gStatGraphRoutinesRanksList;

//! the length of the ranks list
int gStatGraphRoutinesRanksListLength;

int statStringHash(const char *str)
{
    int hash = 0, c;

    if (str == NULL)
        return 1;

    /* perform a sdbm hash */
    while ((c = *str++))
        hash = c + (hash << 6) + (hash << 16) - hash;

    return hash;
}

graphlib_graph_p statNewGraph(graphlib_functiontable_p functions)
{
    int i, dummy;
    graphlib_error_t graphlibError;
    graphlib_graph_p retGraph = NULL;

    graphlibError = graphlib_newGraph(&retGraph, functions);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error creating new graph\n");
        return NULL;
    }
    static int init = -1;
    init++;
    for (i = 0; i < gNumNodeAttrs; i++)
    {
        graphlibError = graphlib_addNodeAttrKey(retGraph, gNodeAttrs[i], &dummy);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            fprintf(stderr, "Failed to add node attr %d %s\n", i, gNodeAttrs[i]);
            return NULL;
        }
    }
    for (i = 0; i < gNumEdgeAttrs; i++)
    {
        graphlibError = graphlib_addEdgeAttrKey(retGraph, gEdgeAttrs[i], &dummy);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            fprintf(stderr, "Failed to add edge attr %d %s\n", i, gEdgeAttrs[i]);
            return NULL;
        }
    }
    return retGraph;

}

graphlib_graph_p createRootedGraph(unsigned int sampleType)
{
    graphlib_error_t graphlibError;
    graphlib_graph_p retGraph = NULL;

    if (sampleType & STAT_SAMPLE_COUNT_REP)
        retGraph = statNewGraph(gStatCountRepFunctions);
    else
        retGraph = statNewGraph(gStatBitVectorFunctions);

    int index;
    graphlib_nodeattr_t nodeAttr = {1, 0, 20, GRC_LIGHTGREY, 0, 0, (char *)"/", -1, NULL};

    nodeAttr.attr_values = (void **)calloc(1, gNumNodeAttrs * sizeof(void *));
    if (nodeAttr.attr_values == NULL)
    {
        fprintf(stderr, "Error allocating %d attributes for sentinel node\n", gNumNodeAttrs);
        return NULL;
    }
    graphlibError = graphlib_getNodeAttrIndex(retGraph, "function", &index);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error getting node attribute index for 'function'\n");
        free(nodeAttr.attr_values);
        return NULL;
    }
    nodeAttr.attr_values[index] = strdup("/");
    if (nodeAttr.attr_values[index] == NULL)
    {
        fprintf(stderr, "Failed to strdup('/') for sentinal node %s\n", strerror(errno));
        free(nodeAttr.attr_values);
        return NULL;
    }
    graphlibError = graphlib_addNode(retGraph, 0, &nodeAttr);
    free(nodeAttr.attr_values[index]);
    free(nodeAttr.attr_values);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        fprintf(stderr, "Error adding sentinel node to graph\n");
        return NULL;
    }
    return retGraph;
}

StatBitVectorEdge_t *initializeBitVectorEdge(int numTasks)
{
    StatBitVectorEdge_t *edge;

    edge = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
    if (edge == NULL)
    {
        fprintf(stderr, "%s: Failed to malloc edge\n", strerror(errno));
        return NULL;
    }
    edge->length = statBitVectorLength(numTasks);
    edge->bitVector = (StatBitVector_t *)calloc(edge->length, STAT_BITVECTOR_BYTES);
    if (edge->bitVector == NULL)
    {
        fprintf(stderr, "%s: Failed to calloc %zu longs for edge->bitVector\n", strerror(errno), edge->length);
        free(edge);
        return NULL;
    }
    return edge;
}

void statInitializeBitVectorFunctions()
{
    if (gStatBitVectorFunctions != NULL)
        return;
    gStatBitVectorFunctions = (graphlib_functiontable_p)malloc(sizeof(graphlib_functiontable_t));
    if (gStatBitVectorFunctions == NULL)
    {
        fprintf(stderr, "Failed to malloc function table\n");
        return;
    }
    gStatBitVectorFunctions->serialize_node = statSerializeNode;
    gStatBitVectorFunctions->serialize_node_length = statSerializeNodeLength;
    gStatBitVectorFunctions->deserialize_node = statDeserializeNode;
    gStatBitVectorFunctions->node_to_text = statNodeToText;
    gStatBitVectorFunctions->merge_node = statMergeNode;
    gStatBitVectorFunctions->copy_node = statCopyNode;
    gStatBitVectorFunctions->free_node = statFreeNode;
    gStatBitVectorFunctions->serialize_edge = statSerializeEdge;
    gStatBitVectorFunctions->serialize_edge_length = statSerializeEdgeLength;
    gStatBitVectorFunctions->deserialize_edge = statDeserializeEdge;
    gStatBitVectorFunctions->edge_to_text = statEdgeToText;
    gStatBitVectorFunctions->merge_edge = statMergeEdge;
    gStatBitVectorFunctions->copy_edge = statCopyEdge;
    gStatBitVectorFunctions->free_edge = statFreeEdge;
    gStatBitVectorFunctions->edge_checksum = statEdgeCheckSum;
    gStatBitVectorFunctions->serialize_node_attr = statSerializeNodeAttr;
    gStatBitVectorFunctions->serialize_node_attr_length = statSerializeNodeAttrLength;
    gStatBitVectorFunctions->deserialize_node_attr = statDeserializeNodeAttr;
    gStatBitVectorFunctions->node_attr_to_text = statNodeAttrToText;
    gStatBitVectorFunctions->merge_node_attr = statMergeNodeAttr;
    gStatBitVectorFunctions->copy_node_attr = statCopyNodeAttr;
    gStatBitVectorFunctions->free_node_attr = statFreeNodeAttr;
    gStatBitVectorFunctions->serialize_edge_attr = statSerializeEdgeAttr;
    gStatBitVectorFunctions->serialize_edge_attr_length = statSerializeEdgeAttrLength;
    gStatBitVectorFunctions->deserialize_edge_attr = statDeserializeEdgeAttr;
    gStatBitVectorFunctions->edge_attr_to_text = statEdgeAttrToText;
    gStatBitVectorFunctions->merge_edge_attr = statMergeEdgeAttr;
    gStatBitVectorFunctions->copy_edge_attr = statCopyEdgeAttr;
    gStatBitVectorFunctions->free_edge_attr = statFreeEdgeAttr;
    gNumNodeAttrs = sizeof(gNodeAttrs) / sizeof(gNodeAttrs[0]);
    gNumEdgeAttrs = sizeof(gEdgeAttrs) / sizeof(gEdgeAttrs[0]);
}

void statFreeBitVectorFunctions()
{
    if (gStatBitVectorFunctions != NULL)
        free(gStatBitVectorFunctions);
    gStatBitVectorFunctions = NULL;
}

void statInitializeMergeFunctions()
{
    if (gStatMergeFunctions != NULL)
        return;
    gStatMergeFunctions = (graphlib_functiontable_p)malloc(sizeof(graphlib_functiontable_t));
    if (gStatMergeFunctions == NULL)
    {
        fprintf(stderr, "Failed to malloc function table\n");
        return;
    }
    gStatMergeFunctions->serialize_node = statSerializeNode;
    gStatMergeFunctions->serialize_node_length = statSerializeNodeLength;
    gStatMergeFunctions->deserialize_node = statDeserializeNode;
    gStatMergeFunctions->node_to_text = statNodeToText;
    gStatMergeFunctions->merge_node = statMergeNode;
    gStatMergeFunctions->copy_node = statCopyNode;
    gStatMergeFunctions->free_node = statFreeNode;
    gStatMergeFunctions->serialize_edge = statSerializeEdge;
    gStatMergeFunctions->serialize_edge_length = statSerializeEdgeLength;
    gStatMergeFunctions->deserialize_edge = statFilterDeserializeEdge;
    gStatMergeFunctions->edge_to_text = statEdgeToText;
    gStatMergeFunctions->merge_edge = statMergeEdge;
    gStatMergeFunctions->copy_edge = statCopyEdge;
    gStatMergeFunctions->free_edge = statFreeEdge;
    gStatMergeFunctions->edge_checksum = statEdgeCheckSum;
    gStatMergeFunctions->serialize_node_attr = statSerializeNodeAttr;
    gStatMergeFunctions->serialize_node_attr_length = statSerializeNodeAttrLength;
    gStatMergeFunctions->deserialize_node_attr = statDeserializeNodeAttr;
    gStatMergeFunctions->node_attr_to_text = statNodeAttrToText;
    gStatMergeFunctions->merge_node_attr = statMergeNodeAttr;
    gStatMergeFunctions->copy_node_attr = statCopyNodeAttr;
    gStatMergeFunctions->free_node_attr = statFreeNodeAttr;
    gStatMergeFunctions->serialize_edge_attr = statSerializeEdgeAttr;
    gStatMergeFunctions->serialize_edge_attr_length = statSerializeEdgeAttrLength;
    gStatMergeFunctions->deserialize_edge_attr = statFilterDeserializeEdgeAttr;
    gStatMergeFunctions->edge_attr_to_text = statEdgeAttrToText;
    gStatMergeFunctions->merge_edge_attr = statMergeEdgeAttr;
    gStatMergeFunctions->copy_edge_attr = statCopyEdgeAttr;
    gStatMergeFunctions->free_edge_attr = statFreeEdgeAttr;
    gNumNodeAttrs = sizeof(gNodeAttrs) / sizeof(gNodeAttrs[0]);
    gNumEdgeAttrs = sizeof(gEdgeAttrs) / sizeof(gEdgeAttrs[0]);
}

void statFreeMergeFunctions()
{
    if (gStatMergeFunctions != NULL)
        free(gStatMergeFunctions);
    gStatMergeFunctions = NULL;
}

void statInitializeReorderFunctions()
{
    if (gStatReorderFunctions != NULL)
        return;
    gStatReorderFunctions = (graphlib_functiontable_p)malloc(sizeof(graphlib_functiontable_t));
    if (gStatReorderFunctions == NULL)
    {
        fprintf(stderr, "Failed to malloc function table\n");
        return;
    }
    gStatReorderFunctions->serialize_node = statSerializeNode;
    gStatReorderFunctions->serialize_node_length = statSerializeNodeLength;
    gStatReorderFunctions->deserialize_node = statDeserializeNode;
    gStatReorderFunctions->node_to_text = statNodeToText;
    gStatReorderFunctions->merge_node = statMergeNode;
    gStatReorderFunctions->copy_node = statCopyNode;
    gStatReorderFunctions->free_node = statFreeNode;
    gStatReorderFunctions->serialize_edge = statSerializeEdge;
    gStatReorderFunctions->serialize_edge_length = statSerializeEdgeLength;
    gStatReorderFunctions->deserialize_edge = statDeserializeEdge;
    gStatReorderFunctions->edge_to_text = statEdgeToText;
    gStatReorderFunctions->merge_edge = statMergeEdgeOrdered;
    gStatReorderFunctions->copy_edge = statCopyEdgeInitializeEmpty;
    gStatReorderFunctions->free_edge = statFreeEdge;
    gStatReorderFunctions->edge_checksum = statEdgeCheckSum;
    gStatReorderFunctions->serialize_node_attr = statSerializeNodeAttr;
    gStatReorderFunctions->serialize_node_attr_length = statSerializeNodeAttrLength;
    gStatReorderFunctions->deserialize_node_attr = statDeserializeNodeAttr;
    gStatReorderFunctions->node_attr_to_text = statNodeAttrToText;
    gStatReorderFunctions->merge_node_attr = statMergeNodeAttr;
    gStatReorderFunctions->copy_node_attr = statCopyNodeAttr;
    gStatReorderFunctions->free_node_attr = statFreeNodeAttr;
    gStatReorderFunctions->serialize_edge_attr = statSerializeEdgeAttr;
    gStatReorderFunctions->serialize_edge_attr_length = statSerializeEdgeAttrLength;
    gStatReorderFunctions->deserialize_edge_attr = statDeserializeEdgeAttr;
    gStatReorderFunctions->edge_attr_to_text = statEdgeAttrToText;
    gStatReorderFunctions->merge_edge_attr = statMergeEdgeAttrOrdered;
    gStatReorderFunctions->copy_edge_attr = statCopyEdgeAttrInitializeEmpty;
    gStatReorderFunctions->free_edge_attr = statFreeEdgeAttr;
    gNumNodeAttrs = sizeof(gNodeAttrs) / sizeof(gNodeAttrs[0]);
    gNumEdgeAttrs = sizeof(gEdgeAttrs) / sizeof(gEdgeAttrs[0]);
}

void statFreeReorderFunctions()
{
    if (gStatReorderFunctions != NULL)
        free(gStatReorderFunctions);
    gStatReorderFunctions = NULL;
}

void statInitializeCountRepFunctions()
{
    if (gStatCountRepFunctions != NULL)
        return;
    gStatCountRepFunctions = (graphlib_functiontable_p)malloc(sizeof(graphlib_functiontable_t));
    if (gStatCountRepFunctions == NULL)
    {
        fprintf(stderr, "Failed to malloc function table\n");
        return;
    }
    gStatCountRepFunctions->serialize_node = statSerializeNode;
    gStatCountRepFunctions->serialize_node_length = statSerializeNodeLength;
    gStatCountRepFunctions->deserialize_node = statDeserializeNode;
    gStatCountRepFunctions->node_to_text = statNodeToText;
    gStatCountRepFunctions->merge_node = statMergeNode;
    gStatCountRepFunctions->copy_node = statCopyNode;
    gStatCountRepFunctions->free_node = statFreeNode;
    gStatCountRepFunctions->serialize_edge = statSerializeCountRepEdge;
    gStatCountRepFunctions->serialize_edge_length = statSerializeCountRepEdgeLength;
    gStatCountRepFunctions->deserialize_edge = statDeserializeCountRepEdge;
    gStatCountRepFunctions->edge_to_text = statCountRepEdgeToText;
    gStatCountRepFunctions->merge_edge = statMergeCountRepEdge;
    gStatCountRepFunctions->copy_edge = statCopyCountRepEdge;
    gStatCountRepFunctions->free_edge = statFreeCountRepEdge;
    gStatCountRepFunctions->edge_checksum = statCountRepEdgeCheckSum;
    gStatCountRepFunctions->serialize_node_attr = statSerializeNodeAttr;
    gStatCountRepFunctions->serialize_node_attr_length = statSerializeNodeAttrLength;
    gStatCountRepFunctions->deserialize_node_attr = statDeserializeNodeAttr;
    gStatCountRepFunctions->node_attr_to_text = statNodeAttrToText;
    gStatCountRepFunctions->merge_node_attr = statMergeNodeAttr;
    gStatCountRepFunctions->copy_node_attr = statCopyNodeAttr;
    gStatCountRepFunctions->free_node_attr = statFreeNodeAttr;
    gStatCountRepFunctions->serialize_edge_attr = statSerializeEdgeAttr;
    gStatCountRepFunctions->serialize_edge_attr_length = statSerializeEdgeAttrLength;
    gStatCountRepFunctions->deserialize_edge_attr = statDeserializeEdgeAttr;
    gStatCountRepFunctions->edge_attr_to_text = statEdgeAttrToText;
    gStatCountRepFunctions->merge_edge_attr = statMergeEdgeAttr;
    gStatCountRepFunctions->copy_edge_attr = statCopyEdgeAttr;
    gStatCountRepFunctions->free_edge_attr = statFreeEdgeAttr;
    gNumNodeAttrs = sizeof(gNodeAttrs) / sizeof(gNodeAttrs[0]);
    gNumEdgeAttrs = sizeof(gEdgeAttrs) / sizeof(gEdgeAttrs[0]);
}

void statFreeCountRepFunctions()
{
    if (gStatCountRepFunctions != NULL)
        free(gStatCountRepFunctions);
    gStatCountRepFunctions = NULL;
}

size_t statBitVectorLength(int numTasks)
{
    int intRet;

    intRet = numTasks / STAT_BITVECTOR_BITS;
    if (numTasks % STAT_BITVECTOR_BITS != 0)
        intRet += 1;
    return intRet;
}

void statSerializeNode(char *buf, const void *node)
{
  if (node !=  NULL)
    strcpy(buf, (char *)node);
}

unsigned int statSerializeNodeLength(const void *node)
{
  if (node != NULL)
    return strlen((char *)node) + 1;
  else
    return 0;
}

void statDeserializeNode(void **node, const char *buf, unsigned int bufLength)
{
  *node = malloc(bufLength);
  strcpy((char *)*node, buf);
}

char *statNodeToText(const void *node)
{
    return strdup((char *)node);
}

void *statMergeNode(void *node1, const void *node2)
{
    return node1;
}

void *statCopyNode(const void *node)
{
    return (void *)strdup((char *)node);
}

void statFreeNode(void *node)
{
    if (node != NULL)
        free(node);
}

void statSerializeEdge(char *buf, const void *edge)
{
    char *ptr;
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

    if (edge == NULL)
        return;
    ptr = buf;
    memcpy(ptr, (void *)&(e->length), sizeof(size_t));
    ptr += sizeof(size_t);
    memcpy(ptr, e->bitVector, STAT_BITVECTOR_BYTES * e->length);
}

unsigned int statSerializeEdgeLength(const void *edge)
{
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;
    if (edge == NULL)
        return 0;
    return sizeof(size_t) + STAT_BITVECTOR_BYTES * e->length;
}

void statDeserializeEdge(void **edge, const char *buf, unsigned int bufLength)
{
    char *ptr;
    StatBitVectorEdge_t *e;

    if (bufLength == 0)
    {
        *edge = NULL;
        return;
    }
    ptr = (char *)buf;
    e = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
    if (e == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for deserialized edge object\n", sizeof(StatBitVectorEdge_t));
        return;
    }

    memcpy((void *)&(e->length), ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    e->bitVector = (StatBitVector_t *)malloc(STAT_BITVECTOR_BYTES * e->length);
    if (e->bitVector == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for deserialized edge bit vector\n", STAT_BITVECTOR_BYTES * e->length);
        return;
    }
    memcpy(e->bitVector, ptr, STAT_BITVECTOR_BYTES * e->length);
    *edge = (void *)e;
}

char *statEdgeToText(const void *edge)
{
    int inRange = 0, firstIteration = 1, currentValue, lastValue = 0;
    unsigned int i, j, charRetSize = 0, count = 0;
    char val[128], *charRet;
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

    if (edge == NULL)
        return NULL;
    charRet = (char *)malloc(STAT_GRAPH_CHUNK * sizeof(char));
    charRetSize = STAT_GRAPH_CHUNK;
    if (charRet == NULL)
    {
        fprintf(stderr, "%s: Failed to allocte memory for edge label\n", strerror(errno));
        return NULL;
    }
    sprintf(charRet + count, "[");
    count += 1;
    for (i = 0; i < e->length; i++)
    {
        if (charRetSize - count < 1024)
        {
            /* Reallocate if we are within 1024 bytes of the end */
            /* This is a large threshold to keep it out of the inner loop */
            charRetSize += STAT_GRAPH_CHUNK;
            charRet = (char *)realloc(charRet, charRetSize * sizeof(char));
            if (charRet == NULL)
            {
                fprintf(stderr, "%s: Failed to reallocte %u bytes of memory for edge label\n", strerror(errno), charRetSize);
                return NULL;
            }
        }
        for (j = 0; j < 8 * sizeof(StatBitVector_t); j++)
        {
            if (e->bitVector[i] & STAT_GRAPH_BIT(j))
            {
                currentValue = i * 8 * sizeof(StatBitVector_t) + j;
                if (inRange == 0)
                {
                    snprintf(val, 128, "%d", currentValue);
                    if (firstIteration == 0)
                    {
                        if (currentValue == lastValue + 1)
                        {
                            inRange = 1;
                            sprintf(charRet + count, "-");
                            count += 1;
                        }
                        else
                        {
                            sprintf(charRet + count, ",");
                            count += 1;
                            sprintf(charRet + count, "%s", val);
                            count += strlen(val);
                        }
                    }
                    else
                    {
                        sprintf(charRet + count, "%s", val);
                        count += strlen(val);
                    }
                }
                else
                {
                    if (currentValue != lastValue + 1)
                    {
                        snprintf(val, 128, "%d,%d", lastValue, currentValue);
                        sprintf(charRet + count, "%s", val);
                        count += strlen(val);
                        inRange = 0;
                    }
                }
                firstIteration = 0;
                lastValue = currentValue;
            }
        }
    }
    if (inRange == 1)
    {
        snprintf(val, 128, "%d", lastValue);
        sprintf(charRet + count, "%s", val);
        count += strlen(val);
    }
    sprintf(charRet + count, "]");

    return charRet;
}

void *statMergeEdge(void *edge1, const void *edge2)
{
    unsigned int i;
    size_t length;
    StatBitVectorEdge_t *e1, *e2;

    if (edge1 == NULL)
        return statCopyEdge(edge2);
    if (edge2 == NULL)
        return edge1;
    e1 = (StatBitVectorEdge_t *)edge1;
    e2 = (StatBitVectorEdge_t *)edge2;

    length = e1->length;
    if (e2->length < e1->length)
        length = e2->length;
    for (i = 0; i < length; i++)
        e1->bitVector[i] |= e2->bitVector[i];
    return edge1;
}

void *statCopyEdge(const void *edge)
{
    StatBitVectorEdge_t *e, *bvRet;

    if (edge == NULL)
        return NULL;
    e = (StatBitVectorEdge_t *)edge;
    bvRet = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
    if (bvRet == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for edge copy\n", sizeof(StatBitVectorEdge_t));
        return NULL;
    }
    bvRet->length = e->length;
    bvRet->bitVector = (StatBitVector_t *)malloc(e->length * STAT_BITVECTOR_BYTES);
    if (bvRet->bitVector == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for bit vector\n", e->length * STAT_BITVECTOR_BYTES);
        return NULL;
    }
    memcpy(bvRet->bitVector, e->bitVector, STAT_BITVECTOR_BYTES * e->length);
    return (void *)bvRet;
}

void statFreeEdge(void *edge)
{
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

    if (edge == NULL)
        return;
    if (e->bitVector != NULL)
        free(e->bitVector);
    free(e);
}

long statEdgeCheckSum(const char *key, const void *edge)
{
    unsigned int i;
    long longRet = 0;
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

    if (edge == NULL)
        return 0;
    if (key == NULL)
    {
        for (i = 0; i < e->length; i++)
            longRet = longRet + e->bitVector[i] * (e->length - i + 1);
    }
    else if (strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
        longRet = *(int64_t *)edge;
    else if (strcmp(key, "bv") == 0)
    {
        for (i = 0; i < e->length; i++)
            longRet = longRet + e->bitVector[i] * (e->length - i + 1);
    }
    return longRet;
}

void statFilterDeserializeEdge(void **edge, const char *buf, unsigned int bufLength)
{
    int offset, i;
    char *ptr = (char *)buf;
    size_t currentEdgeLength;
    StatBitVectorEdge_t *e;

    if (bufLength == 0)
    {
        edge = NULL;
        return;
    }
    e = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
    if (e == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for deserialized edge object\n", sizeof(StatBitVectorEdge_t));
        return;
    }

    memcpy((void *)&(currentEdgeLength), ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    e->length = gStatGraphRoutinesTotalWidth;
    e->bitVector = (StatBitVector_t *)calloc(e->length, STAT_BITVECTOR_BYTES);
    if (e->bitVector == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for deserialized edge bit vector\n", STAT_BITVECTOR_BYTES * e->length);
        return;
    }

    offset = 0;
    for (i = 0; i < gStatGraphRoutinesCurrentIndex; i++)
        offset += gStatGraphRoutinesEdgeLabelWidths[i];

    memcpy((void *)&(e->bitVector[offset]), ptr, STAT_BITVECTOR_BYTES * currentEdgeLength);
    *edge = (void *)e;
}

void *statCopyEdgeInitializeEmpty(const void *edge)
{
    StatBitVectorEdge_t *bvRet;

    if (edge == NULL)
        return NULL;
    bvRet = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
    if (bvRet == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for edge copy\n", sizeof(StatBitVectorEdge_t));
        return NULL;
    }
    bvRet->length = gStatGraphRoutinesTotalWidth;
    bvRet->bitVector = (StatBitVector_t *)calloc(bvRet->length, STAT_BITVECTOR_BYTES);
    if (bvRet->bitVector == NULL)
    {
        fprintf(stderr, "Failed to allocate %zu bytes for bit vector\n", bvRet->length * STAT_BITVECTOR_BYTES);
        return NULL;
    }
    return (void *)bvRet;
}

int bitVectorContains(StatBitVector_t *vec, int val)
{
    return !!(vec[val / STAT_BITVECTOR_BITS] & STAT_GRAPH_BIT(val % STAT_BITVECTOR_BITS));
}

void *statMergeEdgeOrdered(void *edge1, const void *edge2)
{
    int i, bit, byte;
    StatBitVectorEdge_t *e1 = (StatBitVectorEdge_t *)edge1, *e2 = (StatBitVectorEdge_t *)edge2;

    if (edge1 == NULL || edge2 == NULL)
        return NULL;
    for (i = 0; i < gStatGraphRoutinesRanksListLength; i++)
    {
        if (bitVectorContains(e2->bitVector, gStatGraphRoutinesCurrentIndex * STAT_BITVECTOR_BITS + i) == 1)
        {
            byte = gStatGraphRoutinesRanksList[i] / STAT_BITVECTOR_BITS;
            bit = gStatGraphRoutinesRanksList[i] % STAT_BITVECTOR_BITS;
            e1->bitVector[byte] |= STAT_GRAPH_BIT(bit);
        }
    }
    return edge1;
}

void statSerializeCountRepEdge(char *buf, const void *edge)
{
    if (edge == NULL)
        return;
    memcpy(buf, edge, sizeof(StatCountRepEdge_t));
}

unsigned int statSerializeCountRepEdgeLength(const void *edge)
{
    if (edge == NULL)
        return 0;
    return sizeof(StatCountRepEdge_t);
}

void statDeserializeCountRepEdge(void **edge, const char *buf, unsigned int bufLength)
{
    StatCountRepEdge_t *e;

    if (bufLength == 0)
    {
        *edge = NULL;
        return;
    }
    e = (StatCountRepEdge_t *)malloc(sizeof(StatCountRepEdge_t));
    memcpy(e, buf, bufLength);
    *edge = (void *)e;
}

char *statCountRepEdgeToText(const void *edge)
{
    char *charRet;
    StatCountRepEdge_t *e;

    if (edge == NULL)
        return NULL;
    e = (StatCountRepEdge_t *)edge;
    charRet = (char *)malloc(STAT_GRAPH_CHUNK * sizeof(char));
    if (charRet != NULL)
        snprintf(charRet, STAT_GRAPH_CHUNK, "%ld:[%ld](%ld)", e->count, e->representative, e->checksum);
    else
        fprintf(stderr, "Failled to malloc %zu bytes for edge text\n", STAT_GRAPH_CHUNK * sizeof(char));
    return charRet;
}

void *statMergeCountRepEdge(void *edge1, const void *edge2)
{
    StatCountRepEdge_t *e1, *e2;

    if (edge1 == NULL || edge2 == NULL)
        return NULL;
    e1 = (StatCountRepEdge_t *)edge1;
    e2 = (StatCountRepEdge_t *)edge2;
    e1->count += e2->count;
    e1->checksum += e2->checksum;
    if (e2->representative < e1->representative)
        e1->representative = e2->representative;
    return edge1;
}

void *statCopyCountRepEdge(const void *edge)
{
    StatCountRepEdge_t *crRet;

    if (edge == NULL)
        return NULL;
    crRet = (StatCountRepEdge_t *)malloc(sizeof(StatCountRepEdge_t));
    memcpy((void *)crRet, edge, sizeof(StatCountRepEdge_t));
    return (void *)crRet;
}

void statFreeCountRepEdge(void *edge)
{
    if (edge != NULL)
        free((StatCountRepEdge_t *) edge);
}

long statCountRepEdgeCheckSum(const char *key, const void *edge)
{
    unsigned int i;
    long longRet = 0;
    StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

    if (edge == NULL)
        return 0;
    if (key == NULL)
        longRet = *(int64_t *)edge;
    else if (strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
        longRet = *(int64_t *)edge;
    else if (strcmp(key, "bv") == 0)
    {
        for (i = 0; i < e->length; i++)
            longRet = longRet + e->bitVector[i] * (e->length - i + 1);
    }
    return longRet;
    if (edge == NULL)
        return 0;
    if (key == NULL)
        return *(int64_t *)edge;
}

StatCountRepEdge_t *getBitVectorCountRep(StatBitVectorEdge_t *edge, int (relativeRankToAbsoluteRank)(int))
{
    unsigned int i, j, rank, absoluteRank;
    StatCountRepEdge_t *crRet;

    if (edge == NULL)
        return NULL;
    crRet = (StatCountRepEdge_t *)malloc(sizeof(StatCountRepEdge_t));
    if (crRet == NULL)
        return NULL;
    crRet->count = 0;
    crRet->representative = -1;
    crRet->checksum = 0;

    for (i = 0, rank = 0; i < edge->length; i++)
    {
        for (j = 0; j < STAT_BITVECTOR_BITS; j++, rank++)
        {
            if (edge->bitVector[i] & STAT_GRAPH_BIT(j))
            {
                absoluteRank = relativeRankToAbsoluteRank(rank);
                if (crRet->representative == -1)
                    crRet->representative = absoluteRank;
                crRet->count += 1;
                crRet->checksum += absoluteRank + 1;
            }
        }
    }

    return crRet;
}

void statSerializeNodeAttr(const char *key, char *buf, const void *node)
{
  if (node !=  NULL)
    strcpy(buf, (char *)node);
}

unsigned int statSerializeNodeAttrLength(const char *key, const void *node)
{
  if (node != NULL)
    return strlen((char *)node) + 1;
  else
    return 0;
}

void statDeserializeNodeAttr(const char *key, void **node, const char *buf, unsigned int bufLength)
{
  *node = malloc(bufLength);
  strcpy((char *)*node, buf);
}

char *statNodeAttrToText(const char *key, const void *node)
{
    if (node == NULL)
        return NULL;
    return strdup((char *)node);
}

void *statMergeNodeAttr(const char *key, void *node1, const void *node2)
{
    return node1;
}

void *statCopyNodeAttr(const char *key, const void *node)
{
    if (key == NULL || node == NULL)
        return NULL;
    return (void *)strdup((char *)node);
}

void statFreeNodeAttr(const char *key, void *node)
{
    if (node != NULL)
        free(node);
}

void statFreeNodeAttrs(void **nodeAttr, graphlib_graph_p currentGraph)
{
    int i;
    char *nodeAttrKey;
    graphlib_error_t graphlibError;

    if (nodeAttr == NULL)
        return;
    for (i = 0; i < gNumNodeAttrs; i++)
    {
        graphlibError = graphlib_getNodeAttrKey(currentGraph, i, &nodeAttrKey);
        if (GRL_IS_FATALERROR(graphlibError))
            fprintf(stderr, "Error getting key for node attribute %d\n", i);
        statFreeNodeAttr(nodeAttrKey, nodeAttr[i]);
    }
    free(nodeAttr);
}

void statSerializeEdgeAttr(const char *key, char *buf, const void *edge)
{
    if (edge == NULL)
    {
        buf = NULL;
        return;
    }
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        statSerializeEdge(buf, edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
        memcpy(buf, edge, sizeof(int64_t));
    else if (strcmp(key, "tid") == 0)
    {
        int64_t *count = (int64_t *)edge;
        memcpy(buf, edge, (*count + 1) * sizeof(int64_t));
    }
}

unsigned int statSerializeEdgeAttrLength(const char *key, const void *edge)
{
    if (edge == NULL)
        return 0;
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        return statSerializeEdgeLength(edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
        return sizeof(int64_t);
    else if (strcmp(key, "tid") == 0)
    {
        int64_t *count = (int64_t *)edge;
        return (*count + 1) * sizeof(int64_t);
    }
    return 0;
}

int popCount(uint64_t x) {
    int count;
    for (count = 0; x; count++)
        x &= x - 1;
    return count;
}

int statGetBitVectorCount(StatBitVectorEdge_t *edge)
{
    unsigned int i;
    int count = 0;
    for (i = 0; i < edge->length; i++)
        count += popCount(edge->bitVector[i]);
    return count;
}

void statDeserializeEdgeAttr(const char *key, void **edge, const char *buf, unsigned int bufLength)
{
    if (bufLength == 0)
    {
        *edge = NULL;
        return;
    }
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        statDeserializeEdge(edge, buf, bufLength);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0 || strcmp(key, "tid") == 0)
    {
        int64_t *e;

        e = (int64_t *)malloc(sizeof(int64_t));
        if (e == NULL)
        {
            fprintf(stderr, "Failed to allocate for deserialized edge attribute %s\n", key);
            return;
        }
        memcpy(e, buf, bufLength);
        *edge = (void *)e;
    }
}

char *statEdgeAttrToText(const char *key, const void *edge)
{
    if (edge == NULL)
        return NULL;
    if (strcmp(key, "bv") == 0)
        return statEdgeToText(edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
    {
        char *charRet;
        int64_t *e;

        e = (int64_t *)edge;
        charRet = (char *)malloc(STAT_GRAPH_CHUNK * sizeof(char));
        if (charRet != NULL)
            snprintf(charRet, STAT_GRAPH_CHUNK, "%ld", *e);
        else
            fprintf(stderr, "Failled to malloc %zu bytes for edge text\n", STAT_GRAPH_CHUNK * sizeof(char));
        return charRet;
    }
    else if (strcmp(key, "tid") == 0)
    {
        char *charRet, tidString[BUFSIZE];
        int64_t *e, i;

        e = (int64_t *)edge;
        charRet = (char *)calloc(1, STAT_GRAPH_CHUNK * sizeof(char));
        if (charRet == NULL)
        {
            fprintf(stderr, "Failled to malloc %zu bytes for edge text\n", STAT_GRAPH_CHUNK * sizeof(char));
            return NULL;
        }
        snprintf(charRet, STAT_GRAPH_CHUNK, "%ld:", e[0]);
        for (i = 1; i <= e[0]; i++)
        {
            snprintf(tidString, BUFSIZE, "%ld,", e[i]);
            strncat(charRet, tidString, STAT_GRAPH_CHUNK - 1);
        }
        return charRet;
    }
    else if (strcmp(key, "tbv") == 0)
    {
        int count;
        char *charRet;
        StatBitVectorEdge_t *e = (StatBitVectorEdge_t *)edge;

        count = statGetBitVectorCount(e);
        charRet = (char *)malloc(STAT_GRAPH_CHUNK * sizeof(char));
        if (charRet != NULL)
            snprintf(charRet, STAT_GRAPH_CHUNK, "%d", count);
        else
            fprintf(stderr, "%s: Failed to allocte memory for edge label\n", strerror(errno));
        return charRet;
    }
    return NULL;
}

void *statMergeEdgeAttr(const char *key, void *edge1, const void *edge2)
{
    if (edge1 == NULL)
        return statCopyEdgeAttr(key, edge2);
    if (edge2 == NULL)
        return edge1;
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        return statMergeEdge(edge1, edge2);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
    {
        *(int64_t *)edge1 = *(int64_t *)edge1 + *(int64_t *)edge2;
        return edge1;
    }
    else if (strcmp(key, "rep") == 0)
    {
        if (*(int64_t *)edge2 < *(int64_t *)edge1)
            *(int64_t *)edge1 = *(int64_t *)edge2;
        return edge1;
    }
    else if (strcmp(key, "tid") == 0)
    {
        int64_t originalCount, *e1, *e2, *e3, size;
        //TODO how do we associate an edge with a given rank?
        e1 = (int64_t *)edge1;
        originalCount = e1[0];
        e2 = (int64_t *)edge2;
        size = e1[0] + e2[0];
        e3 = (int64_t *)malloc((size + 1) * sizeof(int64_t));
        if (e3 == NULL)
        {
            fprintf(stderr, "Failed to allocate bytes for edge merge\n");
            return NULL;
        }
        e3[0] = size;
        memcpy(e3 + 1, &e1[1], e1[0] * sizeof(int64_t));
        memcpy(e3 + originalCount + 1, &e2[1], e2[0] * sizeof(int64_t));
        free(e1);
        return e3;
    }
    return NULL;
}

void *statCopyEdgeAttr(const char *key, const void *edge)
{
    if (edge == NULL)
        return NULL;
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        return statCopyEdge(edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
    {
        int64_t *intRet;

        intRet = (int64_t *)malloc(sizeof(int64_t));
        if (intRet == NULL)
        {
            fprintf(stderr, "Failed to allocate for edge attribute %s copy\n", key);
            return NULL;
        }
        memcpy((void *)intRet, (int64_t *)edge, sizeof(int64_t));
        return (int64_t *)intRet;
    }
    else if (strcmp(key, "tid") == 0)
    {
        int64_t *intRet;

        intRet = (int64_t *)malloc((*(int64_t *)edge + 1) * sizeof(int64_t));
        if (intRet == NULL)
        {
            fprintf(stderr, "Failed to allocate for edge attribute %s copy\n", key);
            return NULL;
        }
        memcpy((void *)intRet, (int64_t *)edge, (*(int64_t *)edge + 1) * sizeof(int64_t));
        return (int64_t *)intRet;
    }
    return NULL;
}

void statFreeEdgeAttr(const char *key, void *edge)
{
    if (edge == NULL)
        return;
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        statFreeEdge(edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
        free((int64_t *) edge);
}

void statFreeEdgeAttrs(void **edgeAttr, graphlib_graph_p currentGraph)
{
    int i;
    char *edgeAttrKey;
    graphlib_error_t graphlibError;

    if (edgeAttr == NULL)
        return;
    for (i = 0; i < gNumEdgeAttrs; i++)
    {
        graphlibError = graphlib_getEdgeAttrKey(currentGraph, i, &edgeAttrKey);
        if (GRL_IS_FATALERROR(graphlibError))
            fprintf(stderr, "Error getting key for edge attribute %d\n", i);
        statFreeEdgeAttr(edgeAttrKey, edgeAttr[i]);
    }
    free(edgeAttr);
}


void statFilterDeserializeEdgeAttr(const char *key, void **edge, const char *buf, unsigned int bufLength)
{
    if (bufLength == 0)
    {
        *edge = NULL;
        return;
    }
    if (strcmp(key, "bv") == 0 || strcmp(key, "tbv") == 0)
        statFilterDeserializeEdge(edge, buf, bufLength);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0 || strcmp(key, "tid") == 0)
    {
        int64_t *e;

        e = (int64_t *)malloc(sizeof(int64_t));
        if (e == NULL)
        {
            fprintf(stderr, "Failed to allocate for deserialized edge object %s\n", key);
            return;
        }
        memcpy(e, buf, bufLength);
        *edge = (void *)e;
    }
}

void *statCopyEdgeAttrInitializeEmpty(const char *key, const void *edge)
{
    if (edge == NULL)
        return NULL;
    if (strcmp(key, "bv") == 0)
        return statCopyEdgeInitializeEmpty(edge);
    else if (strcmp(key, "count") == 0 || strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0 || strcmp(key, "tid") == 0)
    {
        int64_t *intRet;

        intRet = (int64_t *)malloc(sizeof(int64_t));
        if (intRet == NULL)
        {
            fprintf(stderr, "Failed to allocate for edge attribute %s copy\n", key);
            return NULL;
        }
        memcpy((void *)intRet, (int64_t *)edge, sizeof(int64_t));
        return (int64_t *)intRet;
    }
    else if (strcmp(key, "tbv") == 0)
    {
        return NULL;
//        StatBitVectorEdge_t *bvRet;
//
//        bvRet = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
//        if (bvRet == NULL)
//        {
//            fprintf(stderr, "Failed to allocate %zu bytes for edge copy\n", sizeof(StatBitVectorEdge_t));
//            return NULL;
//        }
//        bvRet->length = gStatGraphRoutinesTotalWidth;
//        bvRet->bitVector = (StatBitVector_t *)calloc(bvRet->length, STAT_BITVECTOR_BYTES);
//        if (bvRet->bitVector == NULL)
//        {
//            fprintf(stderr, "Failed to allocate %zu bytes for bit vector\n", bvRet->length * STAT_BITVECTOR_BYTES);
//            return NULL;
//        }
    }
    return NULL;
}

void *statMergeEdgeAttrOrdered(const char *key, void *edge1, const void *edge2)
{
    if (edge1 == NULL)
        return statCopyEdgeAttr(key, edge2);
    if (edge2 == NULL)
        return edge1;
    if (strcmp(key, "bv") == 0)
        return statMergeEdgeOrdered(edge1, edge2);
    else if (strcmp(key, "count") == 0 || strcmp(key, "sum") == 0 || strcmp(key, "tbvsum") == 0)
    {
        *(int64_t *)edge1 = *(int64_t *)edge1 + *(int64_t *)edge2;
        return edge1;
    }
    else if (strcmp(key, "tcount") == 0 || strcmp(key, "rep") == 0)
        return edge1;
    else if (strcmp(key, "tid") == 0)
    {
        int64_t originalCount, *e1, *e2, *e3, size;
        //TODO how do we associate an edge with a given rank?
        e1 = (int64_t *)edge1;
        originalCount = e1[0];
        e2 = (int64_t *)edge2;
        size = e1[0] + e2[0];
        e3 = (int64_t *)malloc((size + 1) * sizeof(int64_t));
        if (e3 == NULL)
        {
            fprintf(stderr, "Failed to allocate bytes for edge merge\n");
            return NULL;
        }
        e3[0] = size;
        memcpy(e3 + 1, &e1[1], e1[0] * sizeof(int64_t));
        memcpy(e3 + originalCount + 1, &e2[1], e2[0] * sizeof(int64_t));
        free(e1);
        return e3;
    }
    else if (strcmp(key, "tbv") == 0) //TODO
    {
        unsigned int i;
        size_t length;
        StatBitVectorEdge_t *e1, *e2;

        e1 = (StatBitVectorEdge_t *)edge1;
        e2 = (StatBitVectorEdge_t *)edge2;

        length = e1->length;
        if (e2->length < e1->length)
            length = e2->length;
        for (i = 0; i < length; i++)
            e1->bitVector[i] |= e2->bitVector[i];
        return edge1;
    }
    return NULL;
}
