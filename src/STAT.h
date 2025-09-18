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

#ifndef __STAT_H
#define __STAT_H

#define BUFSIZE 8192
#define STAT_UNKNOWN -1
#define STAT_MAJOR_VERSION 4
#define STAT_MINOR_VERSION 2
#define STAT_REVISION_VERSION 2

#include "STAT_IncMRNet.h"

//! An enum of bit flags to determine which components to log
enum StatLogOptions_t {
    STAT_LOG_NONE = 0x00,
    STAT_LOG_FE = 0x01,
    STAT_LOG_BE = 0x02,
    STAT_LOG_CP = 0x04,
    STAT_LOG_MRN = 0x08,
    STAT_LOG_SW = 0x10,
    STAT_LOG_SWERR = 0x20
} ;

//! An enum of bit flags for sample options
enum StatSampleOptions_t {
    STAT_SAMPLE_FUNCTION_ONLY = 0x00,
    STAT_SAMPLE_LINE = 0x01,
    STAT_SAMPLE_PC = 0x02,
    STAT_SAMPLE_COUNT_REP = 0x04,
    STAT_SAMPLE_THREADS = 0x08,
    STAT_SAMPLE_CLEAR_ON_SAMPLE = 0x10,
    STAT_SAMPLE_PYTHON = 0x20,
    STAT_SAMPLE_MODULE_OFFSET = 0x40,
#ifdef OMP_STACKWALKER
    STAT_SAMPLE_OPENMP = 0x80,
#endif
#ifdef STAT_GDB_BE
    STAT_SAMPLE_CUDA_QUICK = 0x100,
#endif
    STAT_SAMPLE_UNKNOWN=0xff
} ;

//! An enum for MRNet message tags
typedef enum {
    PROT_ATTACH_APPLICATION = FirstApplicationTag,
    PROT_ATTACH_APPLICATION_RESP,
    PROT_SAMPLE_TRACES,
    PROT_SAMPLE_TRACES_RESP,
    PROT_SEND_TRACES,
    PROT_SEND_TRACES_RESP,
    PROT_DETACH_APPLICATION,
    PROT_DETACH_APPLICATION_RESP,
    PROT_TERMINATE_APPLICATION,
    PROT_TERMINATE_APPLICATION_RESP,
    PROT_STATBENCH_CREATE_TRACES,
    PROT_STATBENCH_CREATE_TRACES_RESP,
    PROT_CHECK_VERSION,
    PROT_CHECK_VERSION_RESP,
    PROT_EXIT,
    PROT_PAUSE_APPLICATION,
    PROT_PAUSE_APPLICATION_RESP,
    PROT_RESUME_APPLICATION,
    PROT_RESUME_APPLICATION_RESP,
    PROT_SEND_LAST_TRACE,
    PROT_SEND_LAST_TRACE_RESP,
    PROT_SEND_BROADCAST_STREAM,
    PROT_SEND_BROADCAST_STREAM_RESP,
    PROT_FILE_REQ,
    PROT_FILE_REQ_RESP,
    PROT_LIB_REQ,
    PROT_LIB_REQ_RESP,
    PROT_COLLECT_PERF,
    PROT_COLLECT_PERF_RESP,
    PROT_LIB_REQ_ERR,
    PROT_ATTACH_PERF,
    PROT_ATTACH_PERF_RESP,
    PROT_SEND_FGFS_STREAM,
    PROT_FGFS_REQUEST,
    PROT_SEND_NODE_IN_EDGE,
#ifdef DYSECTAPI
    PROT_SEND_NODE_IN_EDGE_RESP,
    PROT_LOAD_SESSION_LIB,
    PROT_LOAD_SESSION_LIB_RESP,
    PROT_NOTIFY_FINISH_BINDING,
#else
    PROT_SEND_NODE_IN_EDGE_RESP
#endif
} StatProt_t;

//! An enum for STAT error codes
typedef enum {
    STAT_OK = 0,
    STAT_SYSTEM_ERROR,
    STAT_MRNET_ERROR,
    STAT_FILTERLOAD_ERROR,
    STAT_GRAPHLIB_ERROR,
    STAT_ALLOCATE_ERROR,
    STAT_ATTACH_ERROR,
    STAT_DETACH_ERROR,
    STAT_SEND_ERROR,
    STAT_SAMPLE_ERROR,
    STAT_TERMINATE_ERROR,
    STAT_FILE_ERROR,
    STAT_LMON_ERROR,
    STAT_ARG_ERROR,
    STAT_VERSION_ERROR,
    STAT_NOT_LAUNCHED_ERROR,
    STAT_NOT_ATTACHED_ERROR,
    STAT_NOT_CONNECTED_ERROR,
    STAT_NO_SAMPLES_ERROR,
    STAT_WARNING,
    STAT_LOG_MESSAGE,
    STAT_STDOUT,
    STAT_VERBOSITY,
    STAT_STACKWALKER_ERROR,
    STAT_PAUSE_ERROR,
    STAT_RESUME_ERROR,
    STAT_DAEMON_ERROR,
    STAT_APPLICATION_EXITED,
    STAT_PENDING_ACK,
    STAT_DYSECT_ERROR
} StatError_t;

//! An enum for STAT verbosity type
typedef enum {
    STAT_VERBOSE_ERROR = 0,
    STAT_VERBOSE_STDOUT,
    STAT_VERBOSE_FULL
} StatVerbose_t;

//! Write the text for the specified error code to a file
/*!
    StatError_t statError - the STAT error enum value
    FILE *outFp - the file to write the text to
*/
#define statPrintErrorType(statError, outFp) \
    switch(statError) \
    { \
        case STAT_OK: \
            fprintf(outFp, "STAT_OK"); \
            break; \
        case STAT_SYSTEM_ERROR: \
            fprintf(outFp, "STAT_SYSTEM_ERROR"); \
            break; \
        case STAT_MRNET_ERROR: \
            fprintf(outFp, "STAT_MRNET_ERROR"); \
            break; \
        case STAT_FILTERLOAD_ERROR: \
            fprintf(outFp, "STAT_FILTERLOAD_ERROR"); \
            break; \
        case STAT_GRAPHLIB_ERROR: \
            fprintf(outFp, "STAT_GRAPHLIB_ERROR"); \
            break; \
        case STAT_ALLOCATE_ERROR: \
            fprintf(outFp, "STAT_ALLOCATE_ERROR"); \
            break; \
        case STAT_ATTACH_ERROR: \
            fprintf(outFp, "STAT_ATTACH_ERROR"); \
            break; \
        case STAT_DETACH_ERROR: \
            fprintf(outFp, "STAT_DETACH_ERROR"); \
            break; \
        case STAT_SEND_ERROR: \
            fprintf(outFp, "STAT_SEND_ERROR"); \
            break; \
        case STAT_SAMPLE_ERROR: \
            fprintf(outFp, "STAT_SAMPLE_ERROR"); \
            break; \
        case STAT_TERMINATE_ERROR: \
            fprintf(outFp, "STAT_TERMINATE_ERROR"); \
            break; \
        case STAT_FILE_ERROR: \
            fprintf(outFp, "STAT_FILE_ERROR"); \
            break; \
        case STAT_LMON_ERROR: \
            fprintf(outFp, "STAT_LMON_ERROR"); \
            break; \
        case STAT_ARG_ERROR: \
            fprintf(outFp, "STAT_ARG_ERROR"); \
            break; \
        case STAT_VERSION_ERROR: \
            fprintf(outFp, "STAT_VERSION_ERROR"); \
            break; \
        case STAT_NOT_ATTACHED_ERROR: \
            fprintf(outFp, "STAT_NOT_ATTACHED_ERROR"); \
            break; \
        case STAT_NOT_CONNECTED_ERROR: \
            fprintf(outFp, "STAT_NOT_CONNECTED_ERROR"); \
            break; \
        case STAT_NO_SAMPLES_ERROR: \
            fprintf(outFp, "STAT_NO_SAMPLES_ERROR"); \
            break; \
        case STAT_WARNING: \
            fprintf(outFp, "STAT_WARNING"); \
            break; \
        case STAT_LOG_MESSAGE: \
            fprintf(outFp, "STAT_LOG_MESSAGEING"); \
            break; \
        case STAT_STDOUT: \
            fprintf(outFp, "STAT_STDOUT"); \
            break; \
        case STAT_VERBOSITY: \
            fprintf(outFp, "STAT_VERBOSITY"); \
            break; \
        case STAT_STACKWALKER_ERROR: \
            fprintf(outFp, "STAT_STACKWALKER_ERROR"); \
            break; \
        case STAT_PAUSE_ERROR: \
            fprintf(outFp, "STAT_PAUSE_ERROR"); \
            break; \
        case STAT_RESUME_ERROR: \
            fprintf(outFp, "STAT_RESUME_ERROR"); \
            break; \
        case STAT_DAEMON_ERROR: \
            fprintf(outFp, "STAT_DAEMON_ERROR"); \
            break; \
        case STAT_APPLICATION_EXITED: \
            fprintf(outFp, "STAT_APPLICATION_EXITED"); \
            break; \
        case STAT_PENDING_ACK: \
            fprintf(outFp, "STAT_PENDING_ACK"); \
            break; \
        case STAT_DYSECT_ERROR: \
            fprintf(outFp, "STAT_DYSECT_ERROR"); \
            break; \
        default: \
            fprintf(outFp, "Unknown Error"); \
            break; \
    };

#endif /* __STAT_H */
