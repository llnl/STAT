/*
Copyright (c) 2007-2018, Lawrence Livermore National Security, LLC.
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

%module STAT

/*%include "std_vector.i"
namespace std{
    %template(IntVector) vector<int>;
};*/

%{
#include "config.h"
%}

%{
#include "STAT_FrontEnd.h"
%}

%include "carrays.i"
%array_class(int, intArray)

// This tells SWIG to treat char ** as a special case
%typemap(in) char **
{
    /* Check if is a list */
    if (PyList_Check($input))
    {
        int size = PyList_Size($input);
        int i = 0;
        $1 = (char **) malloc((size+1)*sizeof(char *));
        for (i = 0; i < size; i++)
        {
            PyObject *o = PyList_GetItem($input,i);
            if (PyString_Check(o))
                $1[i] = PyString_AsString(PyList_GetItem($input,i));
            else
            {
                PyErr_SetString(PyExc_TypeError,"list must contain strings");
                free($1);
                return NULL;
            }
        }
        $1[i] = 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a list");
        return NULL;
    }
}

// This cleans up the char ** array we mallocd before the function call
%typemap(freearg) char **
{
    free((char *) $1);
}

#define STAT_UNKNOWN -1

enum StatLogOptions_t {
    STAT_LOG_NONE = 0x00,
    STAT_LOG_FE = 0x01,
    STAT_LOG_BE = 0x02,
    STAT_LOG_CP = 0x04,
    STAT_LOG_MRN = 0x08,
    STAT_LOG_SW = 0x10,
    STAT_LOG_SWERR = 0x20
} ;

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

typedef enum {
    STAT_LAUNCH = 0,
    STAT_ATTACH,
    STAT_SERIAL_ATTACH,
    STAT_GDB_ATTACH,
    STAT_SERIAL_GDB_ATTACH
} StatLaunch_t;

typedef enum {
    STAT_VERBOSE_ERROR = 0,
    STAT_VERBOSE_STDOUT,
    STAT_VERBOSE_FULL
} StatVerbose_t;

typedef enum {
    STAT_TOPOLOGY_DEPTH = 0,
    STAT_TOPOLOGY_FANOUT,
    STAT_TOPOLOGY_USER,
    STAT_TOPOLOGY_AUTO
} StatTopology_t;

typedef enum {
    STAT_CP_NONE = 0,
    STAT_CP_SHAREAPPNODES,
    STAT_CP_EXCLUSIVE
} StatCpPolicy_t;


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
               STAT_PENDING_ACK
} StatError_t;

#ifdef DYSECTAPI
%rename(DysectAPI_OK) DysectAPI::OK;
%rename(DysectAPI_Error) DysectAPI::Error;
%rename(DysectAPI_InvalidSystemState) DysectAPI::InvalidSystemState;
%rename(DysectAPI_LibraryNotLoaded) DysectAPI::LibraryNotLoaded;
%rename(DysectAPI_SymbolNotFound) DysectAPI::SymbolNotFound;
%rename(DysectAPI_SessionCont) DysectAPI::SessionCont;
%rename(DysectAPI_SessionQuit) DysectAPI::SessionQuit;
%rename(DysectAPI_DomainNotFound) DysectAPI::DomainNotFound;
%rename(DysectAPI_NetworkError) DysectAPI::NetworkError;
%rename(DysectAPI_DomainExpressionError) DysectAPI::DomainExpressionError;
%rename(DysectAPI_StreamError) DysectAPI::StreamError;
%rename(DysectAPI_OptimizedOut) DysectAPI::OptimizedOut;
namespace DysectAPI {
  typedef enum RunTimeErrorCode {
    OK,
    Error,
    InvalidSystemState,
    LibraryNotLoaded,
    SymbolNotFound,
    SessionCont,
    SessionQuit,
    DomainNotFound,
    NetworkError,
    DomainExpressionError,
    StreamError,
    OptimizedOut,
  } DysectErrorCode;
}
#endif


class STAT_FrontEnd
{
    public:
        ~STAT_FrontEnd();

        StatError_t attachAndSpawnDaemons(unsigned int pid, char *remoteNode = NULL);
        StatError_t launchAndSpawnDaemons(char *remoteNode = NULL, bool isStatBench = false);
        StatError_t setupForSerialAttach();
        StatError_t launchMrnetTree(StatTopology_t topologyType, char *topologySpecification, char *nodeList = NULL, bool blocking = true, StatCpPolicy_t cpPolicy = STAT_CP_SHAREAPPNODES);
        StatError_t connectMrnetTree(bool blocking = true);
        StatError_t setupConnectedMrnetTree();
        StatError_t attachApplication(bool blocking = true);
        StatError_t pause(bool blocking = true);
        StatError_t resume(bool blocking = true);
        bool isRunning();
        StatError_t sampleStackTraces(unsigned int sampleType, unsigned int nTraces, unsigned int traceFrequency, unsigned int nRetries, unsigned int retryFrequency, unsigned int maxDaemonNumThreads = 1024, bool blocking = true, const char *variableSpecification = "NULL");
        StatError_t gatherLastTrace(bool blocking = true, const char *altDotFilename = NULL);
        StatError_t gatherTraces(bool blocking = true, const char *altDotFilename = NULL);
        char *getLastDotFilename();
        void shutDown();
        StatError_t detachApplication(bool blocking = true);
        StatError_t detachApplication(int *stopList, int stopListSize, bool blocking = true);
        StatError_t terminateApplication(bool blocking = true);
        void printMsg(StatError_t statError, const char *sourceFile, int sourceLine, const char *fmt, ...);
        StatError_t startLog(unsigned char logType, char *logOutDir, bool withPid = false);
        StatError_t receiveAck(bool blocking = true);
        StatError_t statBenchCreateStackTraces(unsigned int maxDepth, unsigned int nTasks, unsigned int nTraces, unsigned int functionFanout, int nEqClasses, int iCountRep);
        char *getNodeInEdge(int nodeId);

        void printCommunicationNodeList();
        void printApplicationNodeList();
        unsigned int getLauncherPid();
        unsigned int getNumApplProcs();
        unsigned int getNumApplNodes();
        StatError_t setJobId(const char *jobId);
        char *getJobId();
        void setNDaemonsPerNode(unsigned int nDaemonsPerNode);
        unsigned int getNDaemonsPerNode();
        const char *getApplExe();
        StatError_t setToolDaemonExe(const char *toolDaemonExe);
        const char *getToolDaemonExe();
        StatError_t setOutDir(const char *outDir);
        const char *getOutDir();
        StatError_t setFilePrefix(const char *filePrefix);
        const char *getFilePrefix();
        void setProcsPerNode(unsigned int procsPerNode);
        unsigned int getProcsPerNode();
        StatError_t setFilterPath(const char *filterPath);
        const char *getFilterPath();
        const char *getRemoteNode();
        StatError_t addLauncherArgv(const char *launcherArg);
        StatError_t addSerialProcess(const char *pidString);
        const char **getLauncherArgv();
        unsigned int getLauncherArgc();
        void setVerbose(StatVerbose_t verbose);
        StatVerbose_t getVerbose();
        void setApplicationOption(StatLaunch_t applicationOption);
        StatLaunch_t getApplicationOption();
        char *getLastErrorMessage();
        StatError_t addPerfData(const char *buf, double time);
        const char *getInstallPrefix();
        void getVersion(int *version);
#ifdef DYSECTAPI
        StatError_t dysectSetup(const char *dysectApiSessionPath, int dysectTimeout, int argc, char **argv);
        StatError_t dysectListen(bool blocking = true);
        StatError_t dysectStop();
#endif
};

%extend STAT_FrontEnd {
    STAT_FrontEnd() { return STAT_FrontEnd::make(); }
};


%pythoncode %{

import time, sys, os, subprocess

class STATerror(Exception):
    def __init__(self, etype, emsg):
        self.etype = etype
        self.emsg = emsg

    def __str__(self):
        return '%s: %s' %(self.etype, self.emsg)

stat_fe = None

#  \return the STAT_FrontEnd object
#
#  \n
def get_stat_fe():
    global stat_fe
    return stat_fe

## \param application_option - STAT_ATTACH, STAT_LAUNCH, or STAT_SERIAL_ATTACH
## \param processes - the process(es) to attach to or the arguments for launch
## \param topology_type - the STAT topology type
## \param topology - the topology specification
## \param node_list - the nodes to use for communication processes
## \param procs_per_node - the max number of communication processes per node
## \param cp_policy - policy about where to locate communication processes
## \param remote_host - the remote host for the job launcher in the attach or launch case
## \param verbosity - the STAT verbosity level
## \param logging_tuple - the log options (log_type, log_dir)
#  \return true on successful launch/attach
#
#  \n
def attach_impl(application_option, processes, topology_type = STAT_TOPOLOGY_AUTO, topology = '1', node_list = '', procs_per_node = 8, cp_policy = STAT_CP_SHAREAPPNODES, remote_host = None, verbosity = STAT_VERBOSE_ERROR, logging_tuple = (STAT_LOG_NONE, '')):
    log_type, log_dir = logging_tuple
    global stat_fe

    try:
        if stat_fe == None:
            stat_fe = STAT_FrontEnd()
        if log_type != STAT_LOG_NONE:
            stat_fe.startLog(log_type, log_dir)
        stat_fe.setVerbose(verbosity)
        stat_fe.setProcsPerNode(procs_per_node)

        stat_fe.setApplicationOption(application_option)
        if application_option == STAT_ATTACH:
            stat_error = stat_fe.attachAndSpawnDaemons(int(processes), remote_host)
        elif application_option == STAT_SERIAL_ATTACH:
            for process in processes:
                stat_fe.addSerialProcess(process)
            stat_error = stat_fe.setupForSerialAttach()
        elif application_option == STAT_LAUNCH:
            for arg in processes:
                stat_fe.addLauncherArgv(arg)
            stat_error = stat_fe.launchAndSpawnDaemons(remote_host)
            time.sleep(3)
        if stat_error != STAT_OK:
            raise STATerror('tool launch failed', stat_fe.getLastErrorMessage())

        stat_error = stat_fe.launchMrnetTree(topology_type, topology, node_list, True, cp_policy)
        if stat_error != STAT_OK:
            raise STATerror('launch mrnet failed', stat_fe.getLastErrorMessage())

        if application_option != STAT_SERIAL_ATTACH:
            stat_error = stat_fe.connectMrnetTree(True)
            if stat_error != STAT_OK:
                raise STATerror('connect mrnet failed', stat_fe.getLastErrorMessage())

        stat_error = stat_fe.setupConnectedMrnetTree();
        if stat_error != STAT_OK:
            raise STATerror('setup mrnet failed', stat_fe.getLastErrorMessage())

        stat_error = stat_fe.attachApplication(True)
        if stat_error != STAT_OK:
            raise STATerror('attach failed', stat_fe.getLastErrorMessage())

    except Exception as e:
        if application_option == STAT_LAUNCH:
            pid = stat_fe.getLauncherPid()
            if pid != 0:
                subprocess.call(['kill', '-TERM', str(pid)])
        if type(e) == STATerror:
            if e.etype == 'attach failed':
                stat_fe.shutDown()
        stat_fe = None
        raise e
        return False
    return True


## \param processes - the processes to attach to
## \param topology_type - the STAT topology type
## \param topology - the topology specification
## \param node_list - the nodes to use for communication processes
## \param procs_per_node - the max number of communication processes per node
## \param cp_policy - policy about where to locate communication processes
## \param verbosity - the STAT verbosity level
## \param logging_tuple - the log options (log_type, log_dir)
#  \return true on successful attach
#
#  \n
def serial_attach(processes, topology_type = STAT_TOPOLOGY_AUTO, topology = '1', node_list = '', procs_per_node = 8, cp_policy = STAT_CP_SHAREAPPNODES, verbosity = STAT_VERBOSE_ERROR, logging_tuple = (STAT_LOG_NONE, '')):
    attach_impl(STAT_SERIAL_ATTACH, processes, topology_type, topology, node_list, procs_per_node, cp_policy, None, verbosity, logging_tuple)


## \param processes - the parallel job launcher PID to attach to
## \param topology_type - the STAT topology type
## \param topology - the topology specification
## \param node_list - the nodes to use for communication processes
## \param procs_per_node - the max number of communication processes per node
## \param cp_policy - policy about where to locate communication processes
## \param remote_host - the remote host where the job launcher process is running
## \param verbosity - the STAT verbosity level
## \param logging_tuple - the log options (log_type, log_dir)
#  \return true on successful attach
#
#  \n
def attach(processes, topology_type = STAT_TOPOLOGY_AUTO, topology = '1', node_list = '', procs_per_node = 8, cp_policy = STAT_CP_SHAREAPPNODES, remote_host = None, verbosity = STAT_VERBOSE_ERROR, logging_tuple = (STAT_LOG_NONE, '')):
    attach_impl(STAT_ATTACH, processes, topology_type, topology, node_list, procs_per_node, cp_policy, remote_host, verbosity, logging_tuple)


## \param processes - the arguments for launch
## \param topology_type - the STAT topology type
## \param topology - the topology specification
## \param node_list - the nodes to use for communication processes
## \param procs_per_node - the max number of communication processes per node
## \param cp_policy - policy about where to locate communication processes
## \param remote_host - the remote host on which to run the parallel job launcher process
## \param verbosity - the STAT verbosity level
## \param logging_tuple - the log options (log_type, log_dir)
#  \return true on successful launch and attach
#
#  \n
def launch(processes, topology_type = STAT_TOPOLOGY_AUTO, topology = '1', node_list = '', procs_per_node = 8, cp_policy = STAT_CP_SHAREAPPNODES, remote_host = None, verbosity = STAT_VERBOSE_ERROR, logging_tuple = (STAT_LOG_NONE, '')):
    attach_impl(STAT_LAUNCH, processes, topology_type, topology, node_list, procs_per_node, cp_policy, remote_host, verbosity, logging_tuple)


## \param sample_type - the level of detail of the stack traces
## \param num_traces - the number of traces to gather
## \param trace_frequency - the time (ms) between samples
## \param num_retries - the number of retry attemps
## \param retry_frequency - the time (ms) between retries
## \param max_num_threads - the max number of threads per daemon
## \param var_spec - the list of variables to gather
#  \return true on successful stack sampling
#
#  \n
def sample(sample_type = STAT_SAMPLE_FUNCTION_ONLY, num_traces = 1, trace_frequency = 100, num_retries = 5, max_num_threads = 512, retry_frequency = 100, var_spec = 'NULL', alt_dot_filename = ''):
    global stat_fe
    try:
        stat_error = stat_fe.sampleStackTraces(sample_type, num_traces, trace_frequency, num_retries, retry_frequency, max_num_threads, True, var_spec)
        if stat_error != STAT_OK:
            raise STATerror('sample failed', stat_fe.getLastErrorMessage())
        if (num_traces == 1):
            if alt_dot_filename != '':
                stat_error = stat_fe.gatherLastTrace(True, alt_dot_filename)
            else:
                stat_error = stat_fe.gatherLastTrace(True)
        else:
            if alt_dot_filename != '':
                stat_error = stat_fe.gatherTraces(True, alt_dot_filename)
            else:
                stat_error = stat_fe.gatherTraces(True)
        if stat_error != STAT_OK:
            raise STATerror('gather failed', stat_fe.getLastErrorMessage())
        filename = stat_fe.getLastDotFilename()
    except Exception as e:
        stat_fe.shutDown()
        stat_fe = None
        raise

    return True


#  \return true on successful pause
#
#  \n
def pause():
    global stat_fe
    stat_error = stat_fe.pause(True)
    if stat_error != STAT_OK:
        msg = stat_fe.getLastErrorMessage()
        stat_fe.shutDown()
        stat_fe = None
        raise STATerror('pause failed', msg)
    return True


#  \return true on successful resume
#
#  \n
def resume():
    global stat_fe
    stat_error = stat_fe.resume(True)
    if stat_error != STAT_OK:
        msg = stat_fe.getLastErrorMessage()
        stat_fe.shutDown()
        stat_fe = None
        raise STATerror('resume failed', msg)
    return True


#  \return true on successful detach
#
#  \n
def detach():
    global stat_fe
    stat_error = stat_fe.detachApplication(intArray(0), 0, True)
    if stat_error != STAT_OK:
        msg = stat_fe.getLastErrorMessage()
        stat_fe.shutDown()
        stat_fe = None
        raise STATerror('detach failed', msg)
    stat_fe.shutDown()
    stat_fe = None
    return True

%}
