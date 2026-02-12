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

#ifndef __STAT_BACKEND_H
#define __STAT_BACKEND_H

#define STATBE_MAX_HN_LEN 256
#define STAT_SW_DEBUG_BUFFER_LENGTH 33554432

#ifdef STAT_GDB_BE
#include <Python.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <map>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdarg.h>
#include <signal.h>
#include <arpa/inet.h>
#include <set>
#include <sys/select.h>
#include <errno.h>
#include <execinfo.h>

#include "STAT.h"
#include "STAT_timer.h"
#include "STAT_CircularLogs.h"
#include "STAT_GraphRoutines.h"
#include "graphlib.h"

#include "dyninstversion.h"
#include "Symtab.h"
#include "walker.h"
#include "procstate.h"
#include "frame.h"
#include "swk_errors.h"
#include "Type.h"

#ifdef LOCAL_VAR_H
  #include "local_var.h"
#endif
#include "Variable.h"
#include "Function.h"
#ifdef SW_VERSION_8_0_0
  #include "PlatFeatures.h"
  #include "ProcessSet.h"
  #include "PCErrors.h"
  #define GROUP_OPS
#endif
#ifdef OMP_STACKWALKER
  #include "OpenMPStackWalker.h"
#endif

#ifdef STAT_FGFS
  #include "Comm/MRNetCommFabric.h"
  #include "AsyncFastGlobalFileStat.h"
  #include "MRNetSymbolReader.h"
#endif

#ifdef DYSECTAPI
  #include "BPatch.h"
  #include "DysectAPI/DysectAPIBE.h"
  #include "DysectAPI/ProcMap.h"
#endif

//! An enum type to determine who launched the daemon
typedef enum {
    STATD_LMON_LAUNCH = 0,
    STATD_SERIAL_LAUNCH,
    STATD_MRNET_LAUNCH,
} StatDaemonLaunch_t;

//! A struct that contains MRNet connection information to send to the daemons
typedef struct
{
    /* The char arrays are statically sized to make it easy to broadcast */
    char hostName[STATBE_MAX_HN_LEN];
    char parentHostName[STATBE_MAX_HN_LEN];
    int rank;
    int parentPort;
    int parentRank;
} StatLeafInfo_t;

//! A struct to contain backend processes
struct StatBackEndProcInfo_t
{
    char* executable_name;   // optional
    char* host_name;     // optional
    pid_t pid;
    int mpirank;
};
    
//! A struct to cache Python offsets and characteristics
typedef struct
{
    int coNameOffset;
    int coFileNameOffset;
    int obSizeOffset;
    int obSvalOffset;
    int coFirstLineNoOffset;
    int coLnotabOffset;
    int fLastIOffset;
    int pyVarObjectObSizeOffset;
    int pyBytesObjectObSvalOffset;
    bool isUnicode;
    bool isPython3;
} StatPythonOffsets_t;

//! A struct to hold a list of StatLeafInfo_t objects
typedef struct
{
    unsigned int size;
    StatLeafInfo_t *leaves;
} StatLeafInfoArray_t;

//! A struct to specify variables to gather
typedef struct
{
    char fileName[BUFSIZE];
    char variableName[BUFSIZE];
    int lineNum;
    int depth;
} StatVariable_t;


//! Unpacks the MRNet parent node info for all daemons
/*!
    \param buf - the input buffer
    \param bufLen - the length of the input buffer
    \param[out] data - the return data
    \return 0 on success
*/
int unpackStatBeInfo(void *buf, int bufLen, void *data);


//! Routine to compare the equivalence of two frames
/*!
    \param frame1 - the first frame
    \param frame2 - the second frame
    \return true if frame1 < frame2
*/
bool statFrameCmp(const Dyninst::Stackwalker::Frame &frame1, const Dyninst::Stackwalker::Frame &frame2);


//! Translate a relative (local daemon) rank to an absolute (global MPI) rank
/*!
    \param rank - the relative rank
    \return the absolute rank
*/
int statRelativeRankToAbsoluteRank(int rank);
int statDummyRelativeRankToAbsoluteRank(int rank);


//! The STAT daemon object used to gather and send stack traces
class STAT_BackEnd
{
#ifdef DYSECTAPI
    friend class DysectAPI::BE;
#endif
    public:
        //! Factory method
        static STAT_BackEnd* make(StatDaemonLaunch_t launchType);

        //! Destructor
        virtual ~STAT_BackEnd();

        /******************************/
        /* Core STAT BE functionality */
        /******************************/

        //! Initialize and set up this object
        /*
            \param[in,out] argc - pointer to the number of arguments
            \param[in,out] argv - pointer to the argument list
            \return STAT_OK on success
        */
        virtual StatError_t init(int *argc, char ***argv) = 0;

        virtual StatError_t finalize() = 0;

#ifdef STAT_GDB_BE
        //! Initialize and set up the GDB module
        /*
            \return STAT_OK on success
        */
        virtual StatError_t initGdb() = 0;
#endif

        //! Initialize and set up the job launcher
        /*
            \return STAT_OK on success
        */
        virtual StatError_t initLauncher() = 0;

        //! Add a serial process to the process table
        /*
            \param pidString - the input exe@hostname:pid string
            \return STAT_OK on success
        */
        StatError_t addSerialProcess(const char *pidString);

        //! Connect to the MRNet tree
        /*!
            \param argc - [optional] the arg count to pass to MRNet
            \param argv - [optional] the arg list to pass to MRNet
            \return STAT_OK on success

            Receive the connection information from the frontend and broadcast
            it to all the daemons.  Call the MRNet Network constructor with
            this daemon's MRNet personality.
        */
        virtual StatError_t connect(int argc = 0, char **argv = NULL) = 0;

        //! Receive messages from FE and execute the requests
        /*!
            \return STAT_OK on success

            Loops on MRNet receive and executes the requested commands
        */
        StatError_t mainLoop();

        //! Return the STAT name that should be given for a specific frame
        /*!
            \param nodeAttrs[out] - the node attributes for this frame
            \param frame - the Frame to gather the name from
            \param depth - [optional] the depth of this frame in the stack walk.
                   This is necessary for identifying the appropriate frame for
                   variable extraction.
            \return the name to use for this frame
        */
        std::string getFrameName(std::map<std::string, std::string> &nodeAttrs, const Dyninst::Stackwalker::Frame &frame, int depth = -1);

        //! Print the STAT error type and error message
        /*!
            \param statError - the STAT error type
            \param sourceFile - the source file where the error was reported
            \param sourceLine - the source line where the error was reported
            \param fmt - the output format string
            \param ... - any variables to output
        */
        void printMsg(StatError_t statError, const char *sourceFile, int sourceLine, const char *fmt, ...);

        //! Creates the debug log file
        /*!
            \param logType - the level of logging
            \param logOutDir - the output log directory
            \param mrnetOutputLevel - [optional] the MRNet logging output level
            \param withPid - [optional] whether to append the PID to the log file name
            \return STAT_OK on success
        */
        StatError_t startLog(unsigned int logType, char *logOutDir, int mrnetOutputLevel = 1, bool withPid = false);

        //! Dump the Stackwalker debug buffer to the log file
        void swDebugBufferToFile();

        //! Register signal handlers
        /*!
            \param enable - whether to enable signal handling
        */
	    void registerSignalHandlers(bool enable);

        //! Action to perform when signal caught
        /*!
            \param signal - the signal number
            \param sigInfo - the sigInfo struct
            \param context - the context
        */
    	void onCrash(int signal, siginfo_t *sigInfo, void *context);

        //! Get the process table
        /*!
            \return the process table
        */
        StatBackEndProcInfo_t * getProctab();

        //! Get the process table size
        /*!
            \return the proces table size
        */
        int getProctabSize();

        //! (STAT Bench) Write MRNet connection information to a named fifo
        /*!
            \return STAT_OK on success

            Called by the helper daemon to write MRNet connection information
            for the STATBench daemon emulators to a fifo.
        */
        virtual StatError_t statBenchConnectInfoDump() = 0;

        //! (STAT Bench) Connect to the MRNet tree
        /*!
            \return STAT_OK on success

            Reads in connection information from the named fifo and uses it to
            derrive this daemon emulator's MRNet personality, which is then
            passed to the MRNet Network constructor.
        */
        StatError_t statBenchConnect();

        //! Get the number of daemons per node
        /*!
            \return the number of daemons per node
        */
        int getNDaemonsPerNode();

        //! Set the number of daemons per node
        /*!
            \param nDaemonsPerNode - the number of daemons per node
        */
        void setNDaemonsPerNode(int nDaemonsPerNode);

        //! Get the local node rank
        /*!
            \return the number of daemons per node
        */
        int getMyNodeRank();

        //! Set the local node rank
        /*!
            \param localNodeRank - the local node rank
        */
        void setMyNodeRank(int localNodeRank);
#ifdef DYSECTAPI
        DysectAPI::BE *getDysectBe();
#endif

    protected:

        //! Constructor
        STAT_BackEnd(StatDaemonLaunch_t launchType);

        //! Attach to all application processes
        /*!
            \return STAT_OK on success
        */
        StatError_t attach();

        //! Pause all application processes
        /*!
            \return STAT_OK on success
        */
        StatError_t pause();

        //! Resume all application processes
        /*!
            \return STAT_OK on success
        */
        StatError_t resume();

        //! Pause a single process
        /*!
            \param walker - the process to pause
            \return STAT_OK on success
        */
        StatError_t pauseProcess(Dyninst::Stackwalker::Walker *walker);

        //! Resume a single process
        /*!
            \param walker - the process to resume
            \return STAT_OK on success
        */
        StatError_t resumeProcess(Dyninst::Stackwalker::Walker *walker);

        //! Gather the specified number of traces from all processes
        /*!
            \param nTraces - the number of traces to gather per process
            \param traceFrequency - the time (ms) to wait between samples
            \param nRetries - the number of attempts to try to get a complete
                   stack trace
            \param retryFrequency - the time (us) to wait between retries
                   before sampling
            \param variableSpecification - the specification of variables to
                   extract
            \return STAT_OK on success
        */
        StatError_t sampleStackTraces(unsigned int nTraces, unsigned int traceFrequency, unsigned int nRetries, unsigned int retryFrequency, char *variableSpecification);

#if defined GROUP_OPS
        //! Get a stack trace from every process
        /*!
            \param nRetries - the number of attempts to try to get a complete
                   stack trace
            \param retryFrequency - the time to wait between retries
            \return STAT_OK on success
        */
        StatError_t getStackTraceFromAll(unsigned int nRetries, unsigned int retryFrequency);

        //! Add a frame to a given graph
        /*!
            \param stackwalkerGraph - the current Stackwalker graph object
            \param graphlibNode - the node to append to
            \param stackwalkerNode - the current node in the Stackwalker graph
            \param nodeIdNames - the concatenated string of node IDs
            \param[out] errorThreads - a list of threads whose stack walks
                   ended in an error
            \param[out] outputRanks - the set of ranks in this stack walk
            \param[out] outputRankThreadsMap - the set of rank-threads in this stack walk
            \param[out] abort - the callee indicates whether the caller should
                   abort (delete) the caller node
            \param[in] branches - the max branch factor of the current node's
                   ancestors
            \return STAT_OK on success
        */
        StatError_t addFrameToGraph(Dyninst::Stackwalker::CallTree *stackwalkerGraph, graphlib_node_t graphlibNode, Dyninst::Stackwalker::FrameNode *stackwalkerNode, std::string nodeIdNames, std::set<std::pair<Dyninst::Stackwalker::Walker *, Dyninst::THR_ID> > *errorThreads, std::set<int> &outputRanks, std::map<int, std::set<Dyninst::THR_ID> > &outputRankThreadsMap, bool &abort, int branches);
#endif

        //! Get a single stack trace
        /*!
            \param proc - the current process
            \param rank - the current process rank
            \param nRetries - the number of attempts to try to get a complete
                   stack trace
            \param retryFrequency - the time to wait between retries
            \return STAT_OK on success
        */
        StatError_t getStackTrace(Dyninst::Stackwalker::Walker *proc, int rank, unsigned int nRetries, unsigned int retryFrequency);

        //! Translates an address into a source file and line number
        /*!
            \param proc - the current process
            \param addr - the address to translate
            \param[out] outBuf - the source file name
            \param[out] lineNum - the line number
            \return STAT_OK on success
        */
        StatError_t getSourceLine(Dyninst::Stackwalker::Walker *proc, Dyninst::Address addr, char *outBuf, int *lineNum);

        //! Extract a variable value from the application
        /*!
            \param frame - the frame from which to gather the variable
            \param variableName - the name of the variable to gather
            \param[out] outBuf - the value of the variable, caller must allocate
            \param outBufSize - the size of the output buffer
        */
        StatError_t getVariable(const Dyninst::Stackwalker::Frame &frame, char *variableName, char *outBuf, unsigned int outBufSize);

        //! Parse the variable specification
        /*!
            \param variableSpecification - the variable specification
            \return STAT_OK on success

            Parse the variable specification to get individual variables and
            frame references variableSpecification is of the form:

            "num_elements#filename:line.depth$var[,filename:line.depth$var]*"

            i.e., "1#foo.C:1.2$i" or "2#foo.C:1.2$i,bar.C:3.4$j"

            The results are stored in extractVariables_
        */
        StatError_t parseVariableSpecification(char *variableSpecification);

        //! Detach from all of the application processes
        /*!
            \param stopArray - array of process ranks to leave in stopped state
            \param stopArrayLen - the length of the stop array
            \return STAT_OK on success
        */
        StatError_t detach(unsigned int *stopArray, int stopArrayLen);

        //! Terminate all of the application processes
        /*!
            \return STAT_OK on success
        */
        StatError_t terminate();

        //! Send an acknowledgement to the front end
        /*!
            \param stream - the send stream
            \param tag - the packet tag
            \param val - the ack return value, 0 for success, 1 for failure
            \return STAT_OK on success
        */
        StatError_t sendAck(MRN::Stream *stream, int tag, int val);

        //! Get a struct's components for a given type from Symtab
        /*!
            \param type - the struct type
            \return the vector of the struct's components on success
        */
#if DYNINST_MAJOR_VERSION == 10 && DYNINST_MINOR_VERSION <= 1
        tbb::concurrent_vector<Dyninst::SymtabAPI::Field *> *getComponents(Dyninst::SymtabAPI::Type *type);
#else
        Dyninst::dyn_c_vector<Dyninst::SymtabAPI::Field *> *getComponents(Dyninst::SymtabAPI::Type *type);
#endif

        //! Get Python script level frame info
        /*!
            \param proc - the Walker object for this process
            \param frame - the index for the current frame
            \param[out] pyFun - the Python function name
            \param[out] pySource - the Python script source file
            \param[out] pyLineNo - the Python script line number
            \return STAT_OK on success
        */
        StatError_t getPythonFrameInfo(Dyninst::Stackwalker::Walker *proc, const Dyninst::Stackwalker::Frame &frame, char **pyFun, char **pySource, int *pyLineNo);

        //! Clear the nodes2d_ and edges2d_ maps
        /*!
            \return STAT_OK on success
        */
        void clear2dNodesAndEdges();

        //! Clear the nodes3d_ and edges3d_ maps
        /*!
            \return STAT_OK on success
        */
        void clear3dNodesAndEdges();

        //! Update the 3d node and edge maps with the current 2d nodes and edges
        /*!
            \return STAT_OK on success
        */
        StatError_t update3dNodesAndEdges();

        //! Update the src->dst edge with edge
        /*!
            \param src - the source node
            \param dst - the destination node
            \param edge - the edge bit vector
            \param tid - the thread ID
            \return STAT_OK on success

            If the edge does not exist, create it, otherwise just merge
        */
        StatError_t update2dEdge(int src, int dst, StatBitVectorEdge_t *edge, Dyninst::THR_ID tid);

        //! generate the 2d and 3d prefix trees based on nodes and edges maps
        /*!
            \param prefixTree2d - [in,out] the 2d prefix Tree
            \param prefixTree3d - [in,out] the 3d prefix Tree
            \return STAT_OK on success
        */
        StatError_t generateGraphs(graphlib_graph_p *prefixTree2d, graphlib_graph_p *prefixTree3d);

        //! Create stack traces (STAT Bench)
        /*!
            \param maxDepth - the maximum call path depth to generate
            \param nTasks - the number of tasks to emulate
            \param nTraces - the number of traces to generate per emulated task
            \param functionFanout - the max function fanout
            \param nEqClasses - the number of equivalence classes to generate
            \return STAT_OK on success
        */
        StatError_t statBenchCreateTraces(unsigned int maxDepth, int nTasks, unsigned int nTraces, unsigned int functionFanout, int nEqClasses);

        //! Create a single stack trace (STAT Bench)
        /*!
            \param maxDepth - the maximum call path depth to generate
            \param task - the emulated task rank
            \param nTasks - the total number of emulated tasks
            \param functionFanout - the max function fanout
            \param nEqClasses - the number of equivalence classes to generate
            \param iter - the iteration value to pass to the randomizer
            \return the generated graphlib graph
        */
        StatError_t statBenchCreateTrace(unsigned int maxDepth, unsigned int task, unsigned int nTasks, unsigned int functionFanout, int nEqClasses, unsigned int iter);

        /****************/
        /* Private data */
        /****************/

        int proctabSize_;               /*!< the size of the process table */
        int processMapNonNull_;         /*!< the number of active processes */
        unsigned int logType_;          /*!< the logging level */
        unsigned int nDaemonsPerNode_;  /*!< the number of daemons per node */
        unsigned int myNodeRank_;       /*!< the daemon rank on this node */
        char *parentHostName_;          /*!< the hostname of the MRNet parent */
        char logOutDir_[BUFSIZE];       /*!< the directory for log files */
        char outDir_[BUFSIZE];          /*!< the output directory */
        char filePrefix_[BUFSIZE];      /*!< the output file prefix for STAT */
        char localHostName_[BUFSIZE];   /*!< the local hostname */
        char localIp_[BUFSIZE];         /*!< the local IP address */
        CircularBuffer swLogBuffer_;    /*!< the memory buffer for stackwalker
                                             debug logging */
        FILE *swDebugFile_;             /*!< the stackwalker log file handle */
        FILE *errOutFp_;                /*!< the error output file handle */
        bool initialized_;              /*!< whether STAT has been
                                             initialized */
        bool connected_;                /*!< whether this daemon has been
                                             conected to MRNet */
        bool isRunning_;                /*!< whether the target processes are
                                             running */
        bool doGroupOps_;               /*!< do group operations through
                                             StackwalkerAPI */
        bool isPyTrace_;                /*!< whether the current trace includes
                                             Python script level functions */
        MRN::Network *network_;         /*!< the MRNet Network object */
        MRN::Rank myRank_;              /*!< this daemon's MRNet rank */
        MRN::Rank parentRank_;          /*!< the MRNet parent's rank */
        MRN::Port parentPort_;          /*!< the MRNet parent's port */
        MRN::Stream *broadcastStream_;  /*!< the main broadcast and
                                             acknowledgement stream */
        std::map<int, std::map<std::string, std::string> > nodeIdToAttrs_;
        std::map<int, std::map<std::string, void *> > edgeIdToAttrs_;
        std::map<int, std::map<std::string, void *> > edgeIdToAttrs3d_;
        unsigned int threadBvLength_;
        std::vector<Dyninst::THR_ID> threadIds_;
        std::map<int, std::string> nodes2d_; /*!< the 2D prefix tree nodes */
        std::map<int, std::string> nodes3d_; /*!< the 3D prefix tree nodes */
        std::map<int, std::pair<int, StatBitVectorEdge_t *> > edges2d_; /*!< the 2D prefix tree edges */
        std::map<int, std::pair<int, StatBitVectorEdge_t *> > edges3d_; /*!< the 3D prefix tree edges */
        StatBackEndProcInfo_t *proctab_;    /*!< the process table */
        StatDaemonLaunch_t launchType_; /*!< the launch type */
        unsigned int sampleType_;       /*!< type of sample we're currently
                                             collecting */
        int nVariables_;                /*!< the number of variables to
                                             extract */
        StatVariable_t *extractVariables_;  /*!< a list of variables to extract
                                                 for the current sample */

        std::map<unsigned int, Dyninst::Stackwalker::Walker *> processMap_;  /*!< the debug process objects */
        std::map<Dyninst::Stackwalker::Walker *, int> procsToRanks_;    /*!< translate a process into a rank */


#ifdef GROUP_OPS
        Dyninst::ProcControlAPI::ProcessSet::ptr procSet_;  /*< the set of process objects */
        Dyninst::Stackwalker::WalkerSet *walkerSet_;        /*< the set of walker objects */
        std::map<std::string, std::set<int> > exitedProcesses_;
  #ifdef DYSECTAPI
        std::map<int, Dyninst::ProcControlAPI::Process::ptr> mpiRankToProcessMap_;
        std::vector<BPatch_process *> tmpProcSet_;
        BPatch bpatch_;
  #endif
#endif

#ifdef STAT_FGFS
        FastGlobalFileStatus::CommLayer::CommFabric *fgfsCommFabric_; /*< the FGFS communication fabric handle */
#endif

#ifdef DYSECTAPI
        DysectAPI::BE* dysectBE_;
#endif

#ifdef STAT_GDB_BE
        PyObject *gdbModule_;   /*!< the Python stat_cuda_gdb module */
#endif
        bool usingGdb_;     /*!< whether we are using GDB instead of Dyninst */

};

#endif /* __STAT_BACKEND_H */
