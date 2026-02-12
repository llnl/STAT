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

#ifndef __STAT_GRAPHROUTNINES_H
#define __STAT_GRAPHROUTNINES_H

#ifndef STAT_NO_STAT_H /* For merge scripts */
  #include "STAT.h"
#else
enum StatSampleOptions_t {
    STAT_SAMPLE_FUNCTION_ONLY = 0x00,
    STAT_SAMPLE_LINE = 0x01,
    STAT_SAMPLE_PC = 0x02,
    STAT_SAMPLE_COUNT_REP = 0x04,
    STAT_SAMPLE_THREADS = 0x08,
    STAT_SAMPLE_CLEAR_ON_SAMPLE = 0x10,
    STAT_SAMPLE_PYTHON = 0x20
} ;
#endif
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "graphlib.h"
#include <vector>
#include <string>
#include <map>

#define STAT_GRAPH_CHUNK 8192

//! The scalar data type that makes up the bit vector
typedef int64_t StatBitVector_t;
#define STAT_BITVECTOR_BITS 64
#define STAT_BITVECTOR_BYTES 8
#define STAT_GRAPH_BIT(i) ((StatBitVector_t)1<<(i))

//! A struct to specify STAT's bit vector edge label type
typedef struct
{
    size_t length;
    StatBitVector_t *bitVector;
} StatBitVectorEdge_t;

//! A struct to specify STAT's count + representative edge label type
typedef struct
{
    int64_t count;
    int64_t representative;
    int64_t checksum;
} StatCountRepEdge_t;

//! Generate a hash value for a given string
/*!
    \param str - the string to hash
    \return the hash value of the string
*/
int statStringHash(const char *str);

graphlib_graph_p statNewGraph(graphlib_functiontable_p functions);

//! Created a new graph of type sampleType with a root node
/*!
    \param sampleType - the level of detail in the sample
    \return a graphlib graph with a "/" root node
*/
graphlib_graph_p createRootedGraph(unsigned int sampleType);

//! Initialize a bit-vector edge
/*!
    \param numTasks - the number of tasks to represent with this bit vector
    \return a newly allocated, zeroed bit vector
*/
StatBitVectorEdge_t *initializeBitVectorEdge(int numTasks);

//! Initialize the standard bit vector functions
void statInitializeBitVectorFunctions();

//! Free the standard bit vector functions
void statFreeBitVectorFunctions();

//! Initialize the merge bit vector functions for the filter
void statInitializeMergeFunctions();

//! Free the merge bit vector functions for the filter
void statFreeMergeFunctions();

//! Initialize the bit vector reorder functions for the frontend
void statInitializeReorderFunctions();

//! Free the bit vector reorder functions for the frontend
void statFreeReorderFunctions();

//! Initialize the count and representative functions
void statInitializeCountRepFunctions();

//! Free the count and reprsentative functions
void statFreeCountRepFunctions();

//! Calculate the bit vector length
/*!
    \param numTasks - the number of tasks to represent in the bit vector
    \return the length of the bit vector (i.e., number of StatBitVector_t elements)
*/
size_t statBitVectorLength(int numTasks);


//! Serialize the STAT node object into a buffer
/*!
    \param buf - the buffer to write to
    \param node - a pointer to the node object
*/
void statSerializeNode(char *buf, const void *node);

//! Calculate the serialized length of the STAT node object
/*!
    \param node - a pointer to the node object
    \return the serialized length of the node
*/
unsigned int statSerializeNodeLength(const void *node);

//! Deserialize the STAT node object from a buffer
/*!
    \param[out] node - a return pointer to the new node object pointer
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeNode(void **node, const char *buf, unsigned int bufLength);

//! Translate the STAT node object into a string
/*!
    \param node - a pointer to the node object
    \return the string representation of the node
*/
char *statNodeToText(const void *node);

//! Merge two STAT node objects
/*!
    \param[in,out] node1 - the node to merge into
    \param node2 - the node to merge with
*/
void *statMergeNode(void *node1, const void *node2);

//! Copy a STAT node object
/*!
    \param node - a pointer to the node object to be copied
    \return a pointer to the new node copy
*/
void *statCopyNode(const void *node);

//! Free a STAT node object
/*!
    \param node - a pointer to the node object to be freed
*/
void statFreeNode(void *node);

int statGetBitVectorCount(StatBitVectorEdge_t *edge);

//! Serialize the STAT node object into a buffer
/*!
    \param key - the attribute key
    \param buf - the buffer to write to
    \param node - a pointer to the node object
*/
void statSerializeNodeAttr(const char *key, char *buf, const void *node);

//! Calculate the serialized length of the STAT node object
/*!
    \param key - the attribute key
    \param node - a pointer to the node object
    \return the serialized length of the node
*/
unsigned int statSerializeNodeAttrLength(const char *key, const void *node);

//! Deserialize the STAT node object from a buffer
/*!
    \param key - the attribute key
    \param[out] node - a return pointer to the new node object pointer
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeNodeAttr(const char *key, void **node, const char *buf, unsigned int bufLength);

//! Translate the STAT node object into a string
/*!
    \param key - the attribute key
    \param node - a pointer to the node object
    \return the string representation of the node
*/
char *statNodeAttrToText(const char *key, const void *node);

//! Merge two STAT node objects
/*!
    \param key - the attribute key
    \param[in,out] node1 - the node to merge into
    \param node2 - the node to merge with
*/
void *statMergeNodeAttr(const char *key, void *node1, const void *node2);

//! Copy a STAT node object
/*!
    \param key - the attribute key
    \param node - a pointer to the node object to be copied
    \return a pointer to the new node copy
*/
void *statCopyNodeAttr(const char *key, const void *node);

//! Free a STAT node object
/*!
    \param key - the attribute key
    \param node - a pointer to the node object to be freed
*/
void statFreeNodeAttr(const char *key, void *node);

//! Free a STAT node attribute object
/*!
    \param node - a pointer to the node object to be freed
    \param currentGraph - the current graph object
*/
void statFreeNodeAttrs(void **nodeAttr, graphlib_graph_p currentGraph);

//! Serialize a STAT edge object into a buffer
/*!
    \param buf - the buffer to write to
    \param edge - a pointer to the edge object
*/
void statSerializeEdge(char *buf, const void *edge);

//! Calculate the serialized length of a STAT edge object
/*!
    \param edge - a pointer to the edge object
    \return the serialized length of the edge
*/
unsigned int statSerializeEdgeLength(const void *edge);

//! Deserialize a STAT edge object from a buffer
/*!
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeEdge(void **edge, const char *buf, unsigned int bufLength);

//! Translate a STAT edge object into a string
/*!
    \param edge - a pointer to the edge object
    \return the string representation of the edge
*/
char *statEdgeToText(const void *edge);

//! Merge two STAT edge objects
/*!
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in
*/
void *statMergeEdge(void *edge1, const void *edge2);

//! Copy a STAT edge object
/*!
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyEdge(const void *edge);

//! Free a STAT edge object
/*!
    \param edge - a pointer to the edge object to be freed
*/
void statFreeEdge(void *edge);

//! Serialize a STAT edge object into a buffer
/*!
    \param key - the attribute key
    \param buf - the buffer to write to
    \param edge - a pointer to the edge object
*/
void statSerializeEdgeAttr(const char *key, char *buf, const void *edge);

//! Calculate the serialized length of a STAT edge object
/*!
    \param key - the attribute key
    \param edge - a pointer to the edge object
    \return the serialized length of the edge
*/
unsigned int statSerializeEdgeAttrLength(const char *key, const void *edge);

//! Deserialize a STAT edge object from a buffer
/*!
    \param key - the attribute key
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeEdgeAttr(const char *key, void **edge, const char *buf, unsigned int bufLength);

//! Translate a STAT edge object into a string
/*!
    \param key - the attribute key
    \param edge - a pointer to the edge object
    \return the string representation of the edge
*/
char *statEdgeAttrToText(const char *key, const void *edge);

//! Merge two STAT edge objects
/*!
    \param key - the attribute key
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in
*/
void *statMergeEdgeAttr(const char *key, void *edge1, const void *edge2);

//! Copy a STAT edge object
/*!
    \param key - the attribute key
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyEdgeAttr(const char *key, const void *edge);

//! Free a STAT edge object
/*!
    \param key - the attribute key
    \param edge - a pointer to the edge object to be freed
*/
void statFreeEdgeAttr(const char *key, void *edge);

//! Free a STAT edge attribute object
/*!
    \param edge - a pointer to the edge object to be freed
    \param currentGraph - the current graph object
*/
void statFreeEdgeAttrs(void **edgeAttr, graphlib_graph_p currentGraph);

//! Calculate a checksum of a STAT edge object
/*!
    \param key - the attribute key, or NULL if not applicable
    \param edge - a pointer to the edge object
    \return the checksum of the edge
*/
long statEdgeCheckSum(const char *key, const void *edge);

//! Deserialize a STAT edge object for the STAT filter.
/*!
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer

    Deserialize a STAT edge object from a buffer into the appropriate location
    in the bit vector for the STAT filter. This requires
    gStatGraphRoutinesTotalWidth to be set to the final bit vector width,
    gStatGraphRoutinesEdgeLabelWidths must be a list of bit vector widths, and
    gStatGraphRoutinesCurrentIndex must be set to the current index.
*/
void statFilterDeserializeEdge(void **edge, const char *buf, unsigned int bufLength);

//! Deserialize a STAT edge object for the STAT filter.
/*!
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer

    Deserialize a STAT edge object from a buffer into the appropriate location
    in the bit vector for the STAT filter. This requires
    gStatGraphRoutinesTotalWidth to be set to the final bit vector width,
    gStatGraphRoutinesEdgeLabelWidths must be a list of bit vector widths, and
    gStatGraphRoutinesCurrentIndex must be set to the current index.
*/
void statFilterDeserializeEdgeAttr(const char *key, void **edge, const char *buf, unsigned int bufLength);

//! Initialize an empty bit vector for initializing the FrontEnd's reorder graph
/*!
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyEdgeInitializeEmpty(const void *edge);

//! Initialize an empty bit vector for initializing the FrontEnd's reorder graph
/*!
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyEdgeAttrInitializeEmpty(const char *key, const void *edge);

//! Merge the unordered input vector bits into the reorder graph.
/*!
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in

    Merge the unordered input vector bits into the appropriate in-order bits of
    the frontend's reorder graph. This requires gStatGraphRoutinesRanksList to
    be set to the ranks list for the current bit vector,
    gStatGraphRoutinesRanksListLength must be set to the length of the ranks
    list for the current bitvector, and gStatGraphRoutinesCurrentIndex must be
    set to the start index for the bit vector in edge
*/
void *statMergeEdgeOrdered(void *edge1, const void *edge2);

//! Merge the unordered input vector bits into the reorder graph.
/*!
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in

    Merge the unordered input vector bits into the appropriate in-order bits of
    the frontend's reorder graph. This requires gStatGraphRoutinesRanksList to
    be set to the ranks list for the current bit vector,
    gStatGraphRoutinesRanksListLength must be set to the length of the ranks
    list for the current bitvector, and gStatGraphRoutinesCurrentIndex must be
    set to the start index for the bit vector in edge
*/
void *statMergeEdgeAttrOrdered(const char *key, void *edge1, const void *edge2);

//! Serialize a STAT count + rep edge object into a buffer
/*!
    \param buf - the buffer to write to
    \param edge - a pointer to the edge object
*/
void statSerializeCountRepEdge(char *buf, const void *edge);

//! Calculate the serialized length of a STAT count + rep edge object
/*!
    \param edge - a pointer to the edge object
    \return the serialized length of the edge
*/
unsigned int statSerializeCountRepEdgeLength(const void *edge);

//! Deserialize a STAT count + rep edge object from a buffer
/*!
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeCountRepEdge(void **edge, const char *buf, unsigned int bufLength);

//! Translate a STAT count + rep edge object into a string
/*!
    \param edge - a pointer to the edge object
    \return the string representation of the edge
*/
char *statCountRepEdgeToText(const void *edge);

//! Merge a pair of STAT count + rep edges
/*!
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in
*/
void *statMergeCountRepEdge(void *edge1, const void *edge2);

//! Copy a STAT count + rep edge
/*!
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyCountRepEdge(const void *edge);

//! Free a STAT count + rep edge object
/*!
    \param edge - a pointer to the edge object to be freed
*/
void statFreeCountRepEdge(void *edge);

//! Serialize a STAT count + rep edge object into a buffer
/*!
    \param buf - the buffer to write to
    \param edge - a pointer to the edge object
*/
void statSerializeCountRepEdgeAttr(const char * key, char *buf, const void *edge);

//! Calculate the serialized length of a STAT count + rep edge object
/*!
    \param edge - a pointer to the edge object
    \return the serialized length of the edge
*/
unsigned int statSerializeCountRepEdgeAttrLength(const char * key, const void *edge);

//! Deserialize a STAT count + rep edge object from a buffer
/*!
    \param[out] edge - a pointer to the edge object
    \param buf - the serialized buffer
    \param bufLength - the length of the serialized buffer
*/
void statDeserializeCountRepEdgeAttr(const char * key, void **edge, const char *buf, unsigned int bufLength);

//! Translate a STAT count + rep edge object into a string
/*!
    \param edge - a pointer to the edge object
    \return the string representation of the edge
*/
char *statCountRepEdgeAttrToText(const char * key, const void *edge);

//! Merge a pair of STAT count + rep edges
/*!
    \param[in,out] edge1 - the edge to merge into
    \param edge2 - the edge to merge in
*/
void *statMergeCountRepEdgeAttr(const char * key, void *edge1, const void *edge2);

//! Copy a STAT count + rep edge
/*!
    \param edge - a pointer to the edge object to be copied
    \return a pointer to the new edge copy
*/
void *statCopyCountRepEdgeAttr(const char * key, const void *edge);

//! Free a STAT count + rep edge object
/*!
    \param edge - a pointer to the edge object to be freed
*/
void statFreeCountRepEdgeAttr(const char * key, void *edge);

//! Calculate a checksum of a STAT count + rep edge object
/*!
    \param key - the attribute key, or NULL if not applicable
    \param edge - a pointer to the edge object
    \return the checksum of the edge
*/
long statCountRepEdgeCheckSum(const char *key, const void *edge);

//! Translate a full bit vector edge into a count + representative edge
/*!
    \param edge - a pointer to the edge object
    \param relativeRankToAbsoluteRank - a pointer to a function that translates relative (daemon) rank into absolute (global/MPI) rank
    \return the count + representative version of the edge
*/
StatCountRepEdge_t *getBitVectorCountRep(StatBitVectorEdge_t *edge, int (relativeRankToAbsoluteRank)(int));

////! Translate a full bit vector edge into a count + representative edge
///*!
//    \param edge - a pointer to the edge object
//    \param relativeRankToAbsoluteRank - a pointer to a function that translates relative (daemon) rank into absolute (global/MPI) rank
//    \return the count + representative version of the edge
//*/
//StatCountRepEdge_t *getBitVectorCountRep(StatBitVectorEdge_t *edge, int (relativeRankToAbsoluteRank)(int));

#endif
