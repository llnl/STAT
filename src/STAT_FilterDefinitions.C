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

#include <sys/resource.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#ifdef STAT_FGFS
  #include <stdint.h>
  #include <cstring>
#endif
#include "config.h"
#include "mrnet/Packet.h"
#include "graphlib.h"
#include "STAT_GraphRoutines.h"
#include "STAT.h"
#include "graphlib.h"

using namespace MRN;
using namespace std;

extern "C" {

/* Externals from STAT's graphlib routines */
//! the bit vector filter routines to append bit vectors
extern graphlib_functiontable_p gStatMergeFunctions;

//! the count and representative routines
extern graphlib_functiontable_p gStatCountRepFunctions;

//! the final bit vector width
extern int gStatGraphRoutinesTotalWidth;

//! the input list of bit vector widths
extern int *gStatGraphRoutinesEdgeLabelWidths;

//! the current index into the bit vector
extern int gStatGraphRoutinesCurrentIndex;

//! The MRNet format string for the STAT filter
const char *statMerge_format_string = "%Ac %d %d %ud";

//! The MRNet format string for the STAT version check
const char *STAT_checkVersion_format_string = "%d %d %d %d %d";

//! The MRNet format string for the STAT filter initialization
const char *filterInit_format_string = "%uc %s %d";

//! Global variable for file pointer
FILE *gStatOutFp = NULL;

//! Global variable for logging level
unsigned char gLogging = STAT_LOG_NONE;


#if (defined(HAVE_GETRLIMIT) && defined(HAVE_SETRLIMIT))
//! Increases nofile and nproc limits to maximum
/*!
    \return STAT_OK on success

    Increases limits to get a full core file on error for debugging.
*/
StatError_t increaseCoreLimit()
{
    struct rlimit rLim[1];
    StatError_t statError = STAT_OK;

    if (getrlimit(RLIMIT_CORE, rLim) < 0)
    {
        perror("getrlimit failed:\n");
        statError = STAT_SYSTEM_ERROR;
    }
    else if (rLim->rlim_cur < rLim->rlim_max)
    {
        rLim->rlim_cur = rLim->rlim_max;
        if (setrlimit(RLIMIT_CORE, rLim) < 0)
        {
            perror("Unable to increase max no. files:");
            statError = STAT_SYSTEM_ERROR;
        }
    }
    return statError;
}
#endif


//! The filter's logging/error messaging routine
/*!
    \param statError - the STAT error type
    \param sourceFile - the source file where the error was reported
    \param sourceLine - the source line where the error was reported
    \param fmt - the output format string
    \param ... - any variables to output
*/
void cpPrintMsg(StatError_t statError, const char *sourceFile, int sourceLine, const char *fmt, ...)
{
    va_list arg;
    char timeString[BUFSIZE], msg[BUFSIZE];
    const char *timeFormat = "%b %d %T";
    time_t currentTime;
    struct tm *localTime;

    /* If this is a log message and we're not logging, then return */
    if (statError == STAT_LOG_MESSAGE && gStatOutFp == NULL)
        return;

    /* Get the time */
    currentTime = time(NULL);
    localTime = localtime(&currentTime);
    if (localTime == NULL)
        snprintf(timeString, BUFSIZE, "NULL");
    else
        strftime(timeString, BUFSIZE, timeFormat, localTime);

    if (statError != STAT_LOG_MESSAGE && statError != STAT_STDOUT && statError != STAT_VERBOSITY)
    {
        /* Print the error to the screen (or designated error file) */
        fprintf(stderr, "<%s> <%s:% 4d> ", timeString, sourceFile, sourceLine);
        fprintf(stderr, "STAT returned error type ");
        statPrintErrorType(statError, stderr);
        fprintf(stderr, ": ");
        va_start(arg, fmt);
        vfprintf(stderr, fmt, arg);
        va_end(arg);
    }

    /* Print the message to the log */
    if (gStatOutFp != NULL)
    {
        if (gLogging & STAT_LOG_MRN)
        {
            va_start(arg, fmt);
            vsnprintf(msg, BUFSIZE, fmt, arg);
            va_end(arg);
            if (sourceLine != -1 && sourceFile != NULL)
                mrn_printf(sourceFile, sourceLine, "", gStatOutFp, "%s", msg);
            else
                fprintf(gStatOutFp, "%s", msg);
        }
        else
        {
            if (sourceLine != -1 && sourceFile != NULL)
                fprintf(gStatOutFp, "<%s> <%s:%d> ", timeString, sourceFile, sourceLine);
            if (statError != STAT_LOG_MESSAGE && statError != STAT_STDOUT && statError != STAT_VERBOSITY)
            {
                fprintf(gStatOutFp, "STAT returned error type ");
                statPrintErrorType(statError, gStatOutFp);
                fprintf(gStatOutFp, ": ");
            }
            va_start(arg, fmt);
            vfprintf(gStatOutFp, fmt, arg);
            va_end(arg);
            fflush(gStatOutFp);
        }
    }
}


//! initialize the filter
/*!
    \param inputPackets - the vector of input packets
    \param[out] outputPackets - the vector of output packets
    \param[out] outputPacketsReverse - the vector of output packets to send to the BackEnds (not used)
    \param filterState - pointer to the filter state (not used)
    \param params - the current configuration settings for the filtern instance (not used)
    \param topology - the MRNet topology

    Initialize graphlib and the STAT Graph Routines. Create the log file.
*/
void filterInit(vector<PacketPtr> &inputPackets,
                vector<PacketPtr> &outputPackets,
                vector<PacketPtr> &outputPacketsReverse,
                void **filterState,
                PacketPtr &params,
                const TopologyLocalInfo &topology)
{
    char *logDir;
    char fileName[BUFSIZE], hostName[BUFSIZE];
    int intRet, mrnetOutputLevel;
    unsigned int i;
    graphlib_error_t graphlibError;

    graphlibError = graphlib_Init();
    if (GRL_IS_FATALERROR(graphlibError))
    {
        cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to initialize graphlib\n");
        return;
    }
    statInitializeReorderFunctions();
    statInitializeBitVectorFunctions();
    statInitializeCountRepFunctions();
    statInitializeMergeFunctions();

    if (inputPackets[0]->get_Tag() == PROT_SEND_BROADCAST_STREAM)
    {
        for (i = 0; i < inputPackets.size(); i++)
        {
            if (inputPackets[i]->unpack("%uc %s %d", &gLogging, &logDir, &mrnetOutputLevel) == -1)
                cpPrintMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to unpack packet\n");
            if (topology.get_Network()->is_LocalNodeInternal())
            {
                if (gLogging & STAT_LOG_CP)
                {
                    /* Create the log directory */
                    intRet = mkdir(logDir, S_IRUSR | S_IWUSR | S_IXUSR);
                    if (intRet == -1 && errno != EEXIST)
                        cpPrintMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: mkdir failed to create log directory %s\n", strerror(errno), logDir);
                    intRet = gethostname(hostName, BUFSIZE);
                    if (intRet != 0)
                        cpPrintMsg(STAT_WARNING, __FILE__, __LINE__, "Warning, Failed to get hostName\n");
                    snprintf(fileName, BUFSIZE, "%s/%s.STATfilter.%d.log", logDir, hostName, topology.get_Rank());
                    gStatOutFp = fopen(fileName, "w");
                    if (gStatOutFp == NULL)
                        cpPrintMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to open FE log file %s\n", strerror(errno), fileName);
                    if (gLogging & STAT_LOG_MRN)
                        mrn_printf_init(gStatOutFp);
                    set_OutputLevel(mrnetOutputLevel);
                }
            }
            free(logDir);
        }
    }
    for (i = 0; i < inputPackets.size(); i++)
        outputPackets.push_back(inputPackets[i]);
}

//! A message to check the version of various components
/*!
    \param inputPackets - the vector of input packets
    \param[out] outputPackets - the vector of output packets
    \param[out] outputPacketsReverse - the vector of output packets to send to the BackEnds (not used)
    \param filterState - pointer to the filter state (not used)
    \param params - the current configuration settings for the filtern instance (not used)
    \param topology - the MRNet topology (not used)
*/
void STAT_checkVersion(vector<PacketPtr> &inputPackets,
                       vector<PacketPtr> &outputPackets,
                       vector<PacketPtr> &outputPacketsReverse,
                       void **filterState,
                       PacketPtr &params,
                       const TopologyLocalInfo &topology)
{
    PacketPtr currentPacket;
    int major, minor, revision, daemonCount = 0, filterCount = 0;
    unsigned int i;

    /* See if we're checking the filter version */
    if (inputPackets[0]->get_Tag() == PROT_CHECK_VERSION_RESP)
    {
        for (i = 0; i < inputPackets.size(); i++)
        {
            /* Update the filter and daemon counts */
            currentPacket = inputPackets[i];
            major = (*currentPacket)[0]->get_int32_t();
            minor = (*currentPacket)[1]->get_int32_t();
            revision = (*currentPacket)[2]->get_int32_t();
            daemonCount += (*currentPacket)[3]->get_int32_t();
            filterCount += (*currentPacket)[4]->get_int32_t();
        }

        /* Prepare the output values */
        if (major != STAT_MAJOR_VERSION || minor != STAT_MINOR_VERSION || revision != STAT_REVISION_VERSION)
        {
            cpPrintMsg(STAT_VERSION_ERROR, __FILE__, __LINE__, "Filter reports version mismatch: FE = %d.%d.%d, Filter = %d.%d.%d\n", major, minor, revision, STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
            filterCount += 1;
        }

        /* Send the packet */
        PacketPtr newPacket(new Packet(inputPackets[0]->get_StreamId(), inputPackets[0]->get_Tag(), "%d %d %d %d %d", major, minor, revision, daemonCount, filterCount));
        outputPackets.push_back(newPacket);
        return;
    }
}


//! Merges graphs from incoming packets into a single out packet
/*!
    \param inputPackets - the vector of input packets
    \param[out] outputPackets - the vector of output packets
    \param[out] outputPacketsReverse - the vector of output packets to send to the BackEnds (not used)
    \param filterState - pointer to the filter state (not used)
    \param params - the current configuration settings for the filtern instance (not used)
    \param topology - the MRNet topology (not used)
*/
void statMerge(vector<PacketPtr> &inputPackets,
               vector<PacketPtr> &outputPackets,
               vector<PacketPtr> &outputPacketsReverse,
               void **filterState,
               PacketPtr &params,
               const TopologyLocalInfo &topology)
{
    /* The byte arrays are declared static so we can manage the allocated memory
       This is because we need to be assured that the data exists when MRNet
       tries to send it.  They are safe to free up on the next invocation of
       this filter function */
    static char *sOutputByteArray = NULL;
    int totalWidth = 0, nChildren, *edgeLabelWidths = NULL, rank, inputRank, outputRank, child, tag;
    uint64_t outputByteArrayLen, byteArrayLen;
    unsigned int i, sampleType;
    char *byteArray;
    map<int, int> childrenOrder;
    map<int, int>::iterator childrenOrderIter;
    DataType type;
    graphlib_graph_p returnGraph = NULL, currentGraph = NULL;
    graphlib_error_t graphlibError;
    PacketPtr currentPacket;
    StatBitVectorEdge_t *edge, *retEdge;

#if (defined(HAVE_GETRLIMIT) && defined(HAVE_SETRLIMIT))
//    static int init = 0;
//    if (init == 0)
//        increaseCoreLimit();
//    init++;
#endif

    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "STAT filter invoked\n");

    /* Delete byte arrays from previous iterations */
    if (sOutputByteArray != NULL)
    {
        free(sOutputByteArray);
        sOutputByteArray = NULL;
    }

    /* We want to make sure we're processing packets in the same order
       every time the filter is invoked.  That way, we know where in the
       final bit vector to place each incoming bit vector */
    for (i = 0; i < inputPackets.size(); i++)
    {
        currentPacket = inputPackets[i];
        rank = (*currentPacket)[2]->get_int32_t();
        childrenOrder[rank] = i;
    }
    sampleType = (*inputPackets[0])[3]->get_uint32_t();
    tag = inputPackets[0]->get_Tag();

    nChildren = inputPackets.size();
    edgeLabelWidths = (int *)malloc(nChildren * sizeof(int));
    if (edgeLabelWidths == NULL)
    {
        cpPrintMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to allocate edgeLabelWidths\n", strerror(errno));
        return;
    }
    for (childrenOrderIter = childrenOrder.begin(), i = 0; childrenOrderIter != childrenOrder.end(); childrenOrderIter++, i++)
    {
        currentPacket = inputPackets[childrenOrderIter->second];
        edgeLabelWidths[i] = (*currentPacket)[1]->get_int32_t();
        totalWidth += edgeLabelWidths[i];
    }

    if (tag == PROT_SEND_NODE_IN_EDGE_RESP)
    {
        retEdge = (StatBitVectorEdge_t *)malloc(sizeof(StatBitVectorEdge_t));
        if (retEdge == NULL)
        {
            cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "%s: Failed to malloc edge\n", strerror(errno));
            return;
        }
        retEdge->length = totalWidth;
        retEdge->bitVector = (StatBitVector_t *)calloc(retEdge->length, STAT_BITVECTOR_BYTES);
        if (retEdge->bitVector == NULL)
        {
            cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "%s: Failed to calloc %d longs for edge->bitVector\n", strerror(errno), retEdge->length);
            return;
        }

        /* Loop around the packets in order of lowest task rank and merge into output */
        for (childrenOrderIter = childrenOrder.begin(), rank = 0; childrenOrderIter != childrenOrder.end(); childrenOrderIter++, rank++)
        {
            child = childrenOrderIter->second;
            currentPacket = inputPackets[child];

            /* Deserialize edge in packet element [0] */
            byteArray = (char *)((*currentPacket)[0]->get_array(&type, &byteArrayLen));
            gStatGraphRoutinesTotalWidth = totalWidth;
            gStatGraphRoutinesEdgeLabelWidths = edgeLabelWidths;
            gStatGraphRoutinesCurrentIndex = rank;
            statFilterDeserializeEdge((void **)&edge, byteArray, byteArrayLen);
            statMergeEdge(retEdge, edge);
            statFreeEdge(edge);

            /* Add the array of ranks associated with this packet */
            inputRank = (*currentPacket)[2]->get_int32_t();
            if (rank == 0)
                outputRank = inputRank;
            else if (inputRank < outputRank)
                outputRank = inputRank;
        }

        outputByteArrayLen = statSerializeEdgeLength(retEdge);
        sOutputByteArray = (char *)malloc(outputByteArrayLen);
        if (sOutputByteArray == NULL)
        {
            cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__,"%s: Failed to malloc sOutputByteArray\n", strerror(errno));
            return;
        }
        statSerializeEdge(sOutputByteArray, retEdge);
        statFreeEdge(retEdge);
    } /* if (tag == PROT_SEND_NODE_IN_EDGE_RESP) */
    else
    {
        /* Initialize the result graphs */
        if (sampleType & STAT_SAMPLE_COUNT_REP)
            returnGraph = statNewGraph(gStatCountRepFunctions);
        else
            returnGraph = statNewGraph(gStatMergeFunctions);

        /* Loop around the packets in order of lowest task rank and merge into output */
        for (childrenOrderIter = childrenOrder.begin(), rank = 0; childrenOrderIter != childrenOrder.end(); childrenOrderIter++, rank++)
        {
            child = childrenOrderIter->second;
            currentPacket = inputPackets[child];

            /* Deserialize graph in packet element [0] */
            byteArray = (char *)((*currentPacket)[0]->get_array(&type, &byteArrayLen));
            if (sampleType & STAT_SAMPLE_COUNT_REP)
                graphlibError = graphlib_deserializeBasicGraph(&currentGraph, gStatCountRepFunctions, byteArray, byteArrayLen);
            else
            {
                gStatGraphRoutinesTotalWidth = totalWidth;
                gStatGraphRoutinesEdgeLabelWidths = edgeLabelWidths;
                gStatGraphRoutinesCurrentIndex = rank;
                graphlibError = graphlib_deserializeBasicGraph(&currentGraph, gStatMergeFunctions, byteArray, byteArrayLen);
            }
            if (GRL_IS_FATALERROR(graphlibError))
            {
                cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to deserialize graph %d\n", rank);
                return;
            }

            graphlibError = graphlib_mergeGraphs(returnGraph, currentGraph);
            if (GRL_IS_FATALERROR(graphlibError))
            {
                cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to merge graph %d\n", rank);
                return;
            }

            graphlibError = graphlib_delGraph(currentGraph);
            if (GRL_IS_FATALERROR(graphlibError))
            {
                cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to delete graph %d\n", rank);
                return;
            }

            /* Add the array of ranks associated with this packet */
            inputRank = (*currentPacket)[2]->get_int32_t();
            if (rank == 0)
                outputRank = inputRank;
            else if (inputRank < outputRank)
                outputRank = inputRank;
        }

        graphlibError = graphlib_serializeBasicGraph(returnGraph, &sOutputByteArray, &outputByteArrayLen);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to serialize output graph\n");
            return;
        }

        graphlibError = graphlib_delGraph(returnGraph);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            cpPrintMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to delete output graph\n");
            return;
        }
    } /* else from if (tag == PROT_SEND_NODE_IN_EDGE_RESP) */

    PacketPtr newPacket(new Packet(inputPackets[0]->get_StreamId(), tag, "%Ac %d %d %ud", sOutputByteArray, outputByteArrayLen, totalWidth, outputRank, sampleType));
    outputPackets.push_back(newPacket);

    if (edgeLabelWidths != NULL)
        free(edgeLabelWidths);

    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "STAT filter completed\n");
}


#ifdef STAT_FGFS
/* New Filters for solving the Scalabilty problem of accesing files */

//! Cached file contents
map<string, pair<char*, unsigned long> > gFileNameToContentsMap;

//! The set of files that have been requested
set<string> gVisitedFileSet;

//! A mutex to protect the cached file map
pthread_mutex_t gFileNameToContentsMapMutex;

//! A mutex to protect the visited file set
pthread_mutex_t gVisitedFileSetMutex;

//! The MRNet format string for the file request upstream filter
const char *fileRequestUpStream_format_string = "%s";

//! Handles requests for file contents from children. Sends cached contents if available
/*!
    \param inputPackets - the vector of input packets
    \param[out] outputPackets - the vector of output packets
    \param[out] outputPacketsReverse - the vector of output packets to send to the BackEnds (not used)
    \param filterState - pointer to the filter state (not used)
    \param params - the current configuration settings for the filtern instance (not used)
    \param topology - the MRNet topology
*/
void fileRequestUpStream(vector<PacketPtr> &inputPackets,
                         vector<PacketPtr> &outputPackets,
                         vector<PacketPtr> &outputPacketsReverse,
                         void **filterState,
                         PacketPtr &params,
                         const TopologyLocalInfo &topology)
{
    char *fileName = NULL;
    unsigned int i;
    PacketPtr currentPacket, newPacket;
    map<string, pair<char *, unsigned long> >::iterator gFileNameToContentsMapIter;

    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "File request up stream filter begin.\n");

    if (inputPackets[0]->get_Tag() == PROT_FILE_REQ_RESP)
    {
        newPacket = inputPackets[0];
        outputPackets.push_back(newPacket);
    }
    else
    {
        for (i = 0; i < inputPackets.size(); i++)
        {
            currentPacket = inputPackets[i];
            currentPacket->unpack("%s", &fileName);
            pthread_mutex_lock(&gFileNameToContentsMapMutex);
            gFileNameToContentsMapIter = gFileNameToContentsMap.find(fileName);
            if (gFileNameToContentsMapIter != gFileNameToContentsMap.end())
            {
                pthread_mutex_unlock(&gFileNameToContentsMapMutex);
                cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Sending cached result for %s in reverse.\n", fileName);
                PacketPtr newPacketReverse(new Packet(inputPackets[0]->get_StreamId(),
                                           inputPackets[0]->get_Tag(), "%Ac %s", gFileNameToContentsMapIter->second.first,
                                           gFileNameToContentsMapIter->second.second, fileName));
                outputPacketsReverse.push_back(newPacketReverse);
            }
            else
            {
                pthread_mutex_unlock(&gFileNameToContentsMapMutex);
                pthread_mutex_lock(&gVisitedFileSetMutex);
                if (gVisitedFileSet.find(fileName) == gVisitedFileSet.end())
                {
                    gVisitedFileSet.insert(fileName);
                    pthread_mutex_unlock(&gVisitedFileSetMutex);
                    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Noting request for %s.\n", fileName);
                    outputPackets.push_back(inputPackets[i]);
                }
                else
                {
                    pthread_mutex_unlock(&gVisitedFileSetMutex);
                    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "%d: Duplicate request for %s.\n", topology.get_Network()->get_LocalRank(), fileName);
                }
            }
            if (fileName != NULL)
            {
                free(fileName);
                fileName = NULL;
            }
        } /* for (i = 0; i < inputPackets.size(); i++) */
    } /* else for if (inputPackets[0]->get_Tag() == PROT_FILE_REQ_RESP) */

    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "File request up stream filter end.\n");
}

//! The MRNet format string for the file request upstream filter
const char *fileRequestDownStream_format_string = "%Ac %s";

//! Handles file contents sent from parent, cachces contents if last level of CPs
/*!
    \param inputPackets - the vector of input packets
    \param[out] outputPackets - the vector of output packets
    \param[out] outputPacketsReverse - the vector of output packets to send to the BackEnds (not used)
    \param filterState - pointer to the filter state (not used)
    \param params - the current configuration settings for the filtern instance (not used)
    \param topology - the MRNet topology
*/
void fileRequestDownStream(vector<PacketPtr> &inputPackets,
                           vector<PacketPtr> &outputPackets,
                           vector<PacketPtr> &outputPacketsReverse,
                           void **filterState,
                           PacketPtr &params,
                           const TopologyLocalInfo &topology)
{
    char *fileContents, *fileName;
    unsigned long fileContentsLength;
    unsigned int i;
    PacketPtr currentPacket, newPacket;
    set<string>::iterator visitedFileSetIter;
    map<string, pair<char*, unsigned long> >::iterator fileNameToContentsMapIter;

    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "File request down stream filter begin.\n");

    if (inputPackets[0]->get_Tag() == PROT_FILE_REQ)
    {
        cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "PROT_FILE_REQ\n");
        pthread_mutex_init(&gFileNameToContentsMapMutex, NULL);
        pthread_mutex_init(&gVisitedFileSetMutex, NULL);
        newPacket = inputPackets[0];
        outputPackets.push_back(newPacket);
    }
    else
    {
        cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "!PROT_FILE_REQ\n");

        for (i = 0; i < inputPackets.size(); i++)
        {
            currentPacket = inputPackets[i];
            currentPacket->unpack("%Ac %s", &fileContents, &fileContentsLength, &fileName);
            if (topology.get_Network()->is_LocalNodeInternal())
            {
                cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "caching contents of %s at CP length %lu\n", fileName, fileContentsLength);
                pthread_mutex_lock(&gFileNameToContentsMapMutex);
                fileNameToContentsMapIter = gFileNameToContentsMap.find(fileName);
                if (fileNameToContentsMapIter == gFileNameToContentsMap.end())
                    gFileNameToContentsMap[fileName] = make_pair(fileContents, fileContentsLength);
                pthread_mutex_unlock(&gFileNameToContentsMapMutex);
            }
            else
                free(fileContents);

            pthread_mutex_lock(&gVisitedFileSetMutex);
            visitedFileSetIter = gVisitedFileSet.find(fileName);
            if (visitedFileSetIter != gVisitedFileSet.end())
            {
                gVisitedFileSet.erase(visitedFileSetIter);
                pthread_mutex_unlock(&gVisitedFileSetMutex);
                cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "passing along contents of %s, length %lu at CP\n", fileName, fileContentsLength);
                outputPackets.push_back(inputPackets[i]);
            }
            else
            {
                cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "ignoring file %s at CP\n", fileName);
                pthread_mutex_unlock(&gVisitedFileSetMutex);
            }
            if (fileName != NULL)
            {
                free(fileName);
                fileName = NULL;
            }
        }
    }
    cpPrintMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "File request down stream filter end.\n");
}

#endif /* STAT_FGFS */

} /* extern "C" */
