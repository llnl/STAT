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


#include "config.h"
#include <getopt.h>
#include "STAT_FrontEnd.h"

#ifdef DYSECTAPI
  #include "DysectAPI/DysectAPIFE.h"
#endif

using namespace std;
using namespace MRN;

#ifdef DYSECTAPI
using namespace DysectAPI;
#endif


//! A struct to encapsulate user command line args
typedef struct
{
    int sleepTime;                          /*!< the sleep duration */
    unsigned int pid;                       /*!< the job launcher PID */
    unsigned int nTraces;                   /*!< the number of traces to gather*/
    unsigned int traceFrequency;            /*!< the time between samples */
    unsigned int nRetries;                  /*!< the number of retries per sample */
    unsigned int retryFrequency;            /*!< the time between retries */
    unsigned int nDaemonsPerNode;           /*!< the number of retries per sample */
    unsigned int maxDaemonNumThreads;       /*!< the max # threads per daemon */
    char topologySpecification[BUFSIZE];    /*!< the topology specification */
    char *remoteNode;                       /*!< the remote node running the job launcher */
    char *nodeList;                         /*!< the CP node list */
    bool individualSamples;                 /*!< whether to gather individual samples when sampling multiple */
    bool comprehensive;                     /*!< whether to gather a comprehensive set of samples */
    bool countRep;                          /*!< whether to gather just a count and representative */
    bool withPython;                        /*!< whether to gather Python script level traces */
    bool withThreads;                       /*!< whether to gather traces from threads */
    bool withOpenMP;                        /*!< whether to translate OpenMP stack walks */
    StatCpPolicy_t cpPolicy;                /*!< whether to use the application nodes to run communication processes */
    StatLaunch_t applicationOption;         /*!< attach or launch case */
    unsigned int sampleType;                /*!< the sample level of detail */
    StatTopology_t topologyType;            /*!< the topology specification type */
#ifdef DYSECTAPI
    int dysectArgc;
    char **dysectArgv;
#endif
#ifdef STAT_GDB_BE
    bool withCudaQuick;
#endif
} StatArgs_t;

//! Prints the usage directions
void printUsage();

//! Parses the command line arguments and return the options
/*!
    \param[out] statArgs - the options to return from parsing
    \param statFrontEnd - the STAT Frontend object
    \param argc - the number of arguments
    \param argv - the arguments
    \return STAT_OK on success
*/
StatError_t parseArgs(StatArgs_t *statArgs, STAT_FrontEnd *statFrontEnd, int argc, char **argv);

//! Sleeps for requested amount of time and prints a message to inform the user
void mySleep(int sleepTime);


#ifdef DYSECTAPI
bool dysectApiEnabled = false;
char *dysectApiSessionPath = NULL;
int dysectTimeout = 0;
#endif


//! main() function to run STAT
/*!
    \param argc - the number of arguments
    \param argv - the arguments
    \return 0 on success
*/
int main(int argc, char **argv)
{
    int i, j, samples = 1, traces;
    struct timeval timeStamp;
    struct tm *localtimeResult;
    time_t currentTime;
    char timeBuf[BUFSIZE];
    STAT_FrontEnd *statFrontEnd;
    StatError_t statError, retval;
    string invocationString;
    StatArgs_t *statArgs;

    statFrontEnd = STAT_FrontEnd::make();


    /* Parse arguments and fill in class variables */
    statArgs = (StatArgs_t *)calloc(1, sizeof(StatArgs_t));
    if (statArgs == NULL)
    {
        statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to calloc statArgs\n");
        delete statFrontEnd;
        return -1;
    }

    statError = parseArgs(statArgs, statFrontEnd, argc, argv);
    if (statError != STAT_OK)
    {
        if (statError != STAT_WARNING)
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to parse arguments\n");
        delete statFrontEnd;
        free(statArgs);
        if (statError != STAT_WARNING)
            return -1;
        else
            return 0;
    }

    gettimeofday(&timeStamp, NULL);
    currentTime = timeStamp.tv_sec;
    localtimeResult = localtime(&currentTime);
    if (localtimeResult == NULL)
        statFrontEnd->printMsg(STAT_WARNING, __FILE__, __LINE__, "localtime() returned NULL %s\n", strerror(errno));
    else if (strftime(timeBuf, BUFSIZE, "%Y-%m-%d-%T", localtimeResult) != 0)
        statFrontEnd->printMsg(STAT_STDOUT, __FILE__, __LINE__, "STAT started at %s\n", timeBuf);
    else
        statFrontEnd->printMsg(STAT_WARNING, __FILE__, __LINE__, "strftime() returned 0\n");

    /* Push the arguments into the output file */
    invocationString = "STAT invoked with: ";
    for (i = 0; i < argc; i++)
    {
        invocationString.append(argv[i]);
        invocationString.append(" ");
    }
    statFrontEnd->addPerfData(invocationString.c_str(), -1.0);

    /* If we're just attaching, sleep here */
    if (statArgs->applicationOption == STAT_ATTACH || statArgs->applicationOption == STAT_SERIAL_ATTACH || statArgs->applicationOption == STAT_SERIAL_GDB_ATTACH || statArgs->applicationOption == STAT_GDB_ATTACH)
        mySleep(statArgs->sleepTime);

    /* Launch the Daemons */
    statFrontEnd->setNDaemonsPerNode(statArgs->nDaemonsPerNode);
    statFrontEnd->setApplicationOption(statArgs->applicationOption);
    if (statArgs->applicationOption == STAT_ATTACH || statArgs->applicationOption == STAT_GDB_ATTACH)
        statError = statFrontEnd->attachAndSpawnDaemons(statArgs->pid, statArgs->remoteNode);
    else if (statArgs->applicationOption == STAT_LAUNCH)
        statError = statFrontEnd->launchAndSpawnDaemons(statArgs->remoteNode);
    else
        statError = statFrontEnd->setupForSerialAttach();
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to launch MRNet tree()\n");
        statFrontEnd->shutDown();
        delete statFrontEnd;
        free(statArgs);
        return -1;
    }

    /* Launch the MRNet Tree */
    statError = statFrontEnd->launchMrnetTree(statArgs->topologyType, statArgs->topologySpecification, statArgs->nodeList, true, statArgs->cpPolicy);
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to launch MRNet tree()\n");
        statFrontEnd->shutDown();
        delete statFrontEnd;
        free(statArgs);
        return -1;
    }

    statError = statFrontEnd->setupConnectedMrnetTree();
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to setup connected MRNet tree\n");
        delete statFrontEnd;
        free(statArgs);
        return -1;
    }

    /* Attach to the application */
    statError = statFrontEnd->attachApplication();
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to attach to application\n");
        statFrontEnd->shutDown();
        delete statFrontEnd;
        free(statArgs);
        return -1;
    }

    /* If we're launching, sleep here to let the job start */
#ifdef DYSECTAPI
    if (statArgs->applicationOption == STAT_LAUNCH && dysectApiEnabled == false)
#else
    if (statArgs->applicationOption == STAT_LAUNCH)
#endif
    {
        statError = statFrontEnd->resume();
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to resume application\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statArgs);
            return -1;
        }
        mySleep(statArgs->sleepTime);
    }

#ifdef SBRS
    sleep(1);
#endif

#ifdef DYSECTAPI
    if (dysectApiEnabled)
    {
        statFrontEnd->printMsg(STAT_STDOUT, __FILE__, __LINE__, "\n## Prototype DysectAPI enabled ##\n");
        statFrontEnd->printMsg(STAT_STDOUT, __FILE__, __LINE__, "Notice: Traditional sampling is disabled troughout session!\n");

        statError = statFrontEnd->dysectSetup(dysectApiSessionPath, dysectTimeout, statArgs->dysectArgc, statArgs->dysectArgv);
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to setup Dysect session\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statArgs);
            return -1;
        }

        statError = statFrontEnd->resume();
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to resume application\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statArgs);
            return -1;
        }
        mySleep(statArgs->sleepTime);

        statError = statFrontEnd->dysectListen(true);
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Dysect session failed\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statArgs);
            return -1;
        }
    } /* if dysectApiEnabled */
    else
    {
#endif

    retval = STAT_OK;
    if (statArgs->nTraces <= 0)
        traces = 1;
    else
        traces = statArgs->nTraces;

    if (statArgs->comprehensive)
        samples = 5;
    for (i = 0; i < samples; i++)
    {
        if (statArgs->comprehensive)
        {
            statArgs->sampleType = 0;
            statArgs->sampleType |= STAT_SAMPLE_CLEAR_ON_SAMPLE;
            if (statArgs->withThreads == true)
                statArgs->sampleType |= STAT_SAMPLE_THREADS;
#ifdef OMP_STACKWALKER
            if (statArgs->withOpenMP == true)
                statArgs->sampleType |= STAT_SAMPLE_OPENMP;
#endif
            if (statArgs->withPython == true)
                statArgs->sampleType |= STAT_SAMPLE_PYTHON;
            if (statArgs->countRep == true)
                statArgs->sampleType |= STAT_SAMPLE_COUNT_REP;

            traces = 1;
            if (i == 1)
                statArgs->sampleType |= STAT_SAMPLE_MODULE_OFFSET;
            else if (i == 2)
                statArgs->sampleType |= STAT_SAMPLE_PC;
            else if (i == 3)
                statArgs->sampleType |= STAT_SAMPLE_LINE;
            else if (i == 4)
                traces = statArgs->nTraces;
        }
        for (j = 0; j < traces; j++)
        {
            if (j == 1)
                statArgs->sampleType &= ~STAT_SAMPLE_CLEAR_ON_SAMPLE;
            /* Pause the application */
            statError = statFrontEnd->pause();
            if (statError == STAT_APPLICATION_EXITED)
                retval = STAT_APPLICATION_EXITED;
            else if (statError != STAT_OK)
            {
                statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to resume application\n");
                statFrontEnd->shutDown();
                delete statFrontEnd;
                free(statArgs);
                return -1;
            }

            /* Sample the traces */
            statError = statFrontEnd->sampleStackTraces(statArgs->sampleType, 1, statArgs->traceFrequency, statArgs->nRetries, statArgs->retryFrequency, statArgs->maxDaemonNumThreads);
            if (statError != STAT_OK)
            {
                if (statError == STAT_APPLICATION_EXITED)
                    break;
                statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to sample stack traces\n");
                statFrontEnd->shutDown();
                delete statFrontEnd;
                free(statArgs);
                return -1;
            }

            if (statArgs->individualSamples == true)
            {
                /* Gather the trace */
                statError = statFrontEnd->gatherLastTrace();
                if (statError != STAT_OK)
                {
                    statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to gather stack traces\n");
                    statFrontEnd->shutDown();
                    delete statFrontEnd;
                    free(statArgs);
                    return -1;
                }
            }

            /* Resume the application */
            if (statArgs->comprehensive == false or i == 3)
            {
                statError = statFrontEnd->resume();
                if (statError == STAT_APPLICATION_EXITED)
                    retval = STAT_APPLICATION_EXITED;
                else if (statError != STAT_OK)
                {
                    statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to resume application\n");
                    statFrontEnd->shutDown();
                    delete statFrontEnd;
                    free(statArgs);
                    return -1;
                }
                if (retval != STAT_OK)
                    break;
                if (j != traces - 1)
                    usleep(1000 * statArgs->traceFrequency);
            }
        } /* for j */

        /* Gather the traces */
        if (traces > 1)
            statError = statFrontEnd->gatherTraces();
        else
            statError = statFrontEnd->gatherLastTrace();
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to gather stack traces\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statArgs);
            return -1;
        }
    } /* for i */
#ifdef DYSECTAPI
    } /* else dysectApiEnabled */
#endif

    /* Detach from the application */
    statError = statFrontEnd->detachApplication();
    if (statError != STAT_OK)
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to detach from application\n");

    statFrontEnd->shutDown();
    statFrontEnd->printMsg(STAT_STDOUT, __FILE__, __LINE__, "\nResults written to %s\n\n", statFrontEnd->getOutDir());

    delete statFrontEnd;
    free(statArgs);
    return 0;
}

//! Prints the usage message
void printUsage()
{
    fprintf(stderr, "\nSTAT, the Stack Trace Analysis Tool, is a highly scalable, lightweight tool that gathers and merges stack traces from a parallel application's processes.  STAT can gather and merge multiple stack traces per process, showing the application's time varying behavior.  The output shows the global application behavior and can be used to identify process equivalence classes, subsets of tasks that exhibit similar behavior.  A representative of each equivalence class can be fed into a heavyweight debugger for root cause analysis.  Running STAT will create a stat_results directory that contains \".dot\" files that are best viewed with the STATview GUI.  They can also be viewed with the \"dotty\" utility from graphviz.\n");
    fprintf(stderr, "\nUsage:\n");
    fprintf(stderr, "\tstat-cl [OPTIONS] <launcherPid_>\n");
    fprintf(stderr, "\tstat-cl [OPTIONS] -C <application_launch_command>\n");
    fprintf(stderr, "\tstat-cl [OPTIONS] -I [<exe>@<host:pid>]+\n");
    fprintf(stderr, "\nStack trace sampling options:\n");
    fprintf(stderr, "  -t, --traces <count>\t\tnumber of traces per process\n");
    fprintf(stderr, "  -T, --tracefreq <frequency>\ttime between samples in milli-seconds\n");
    fprintf(stderr, "  -S, --sampleindividual\tsave all individual samples\n");
    fprintf(stderr, "  -r, --retries <count>\t\tretry attempts per sample to get a complete\n\t\t\t\ttrace\n");
    fprintf(stderr, "  -R, --retryfreq <frequency>\ttime between sample retries in micro-seconds\n");
    fprintf(stderr, "  -P, --withpc\t\t\tsample program counter in addition to\n\t\t\t\tfunction name\n");
    fprintf(stderr, "  -m, --withmoduleoffset\tsample module offset only\n");
    fprintf(stderr, "  -i, --withline\t\tsample source line number in addition\n\t\t\t\tto function name\n");
    fprintf(stderr, "  -w, --withthreads\t\tsample helper threads in addition to the\n\t\t\t\tmain thread\n");
    fprintf(stderr, "  -H, --maxdaemonthreads <num>\tmax number of threads per daemon\n");
#ifdef OMP_STACKWALKER
    fprintf(stderr, "  -o, --withopenmp\t\ttranslate OpenMP stacks to logical application\n\t\t\t\tview\n");
#endif
#ifdef STAT_GDB_BE
    fprintf(stderr, "  -G, --gdb\t\t\tuse (cuda) gdb to drive the daemons\n");
    fprintf(stderr, "  -Q, --cudaquick\t\tgather less comprehensive, but faster cuda traces\n");
#endif
    fprintf(stderr, "  -c, --comprehensive\t\tgather 5 traces: function only; module offset;\n\t\t\t\tfunction + pc; function + line; and 3D function\n\t\t\t\tonly\n");
    fprintf(stderr, "  -U, --countrep\t\tonly gather count and a single representative\n");
    fprintf(stderr, "  -y, --pythontrace\t\tgather Python script level stack traces\n");
    fprintf(stderr, "  -s, --sleep <secs>\t\tsleep time before attaching and gathering traces\n");
    fprintf(stderr, "\nTopology options:\n");
    fprintf(stderr, "  -a, --autotopo\t\tlet STAT automatically create topology\n");
    fprintf(stderr, "  -d, --depth <depth>\t\ttree topology depth\n");
    fprintf(stderr, "  -f, --fanout <width>\t\tmaximum tree topology fanout\n");
    fprintf(stderr, "  -z, --daemonspernode <num>\tnumber of daemons per node\n");
    fprintf(stderr, "  -u, --usertopology <topology>\tspecify the number of communication nodes per\n\t\t\t\tlayer in the tree topology, separated by dashes\n");
    fprintf(stderr, "  -n, --nodes <nodelist>\tlist of nodes for communication processes\n");
    fprintf(stderr, "\t\t\t\tExample node lists:\thost1\n\t\t\t\t\t\t\thost1,host2\n\t\t\t\t\t\t\thost[1,5-7,9]\n");
    fprintf(stderr, "  -N, --nodesfile <filename>\tfile containing list of nodes for communication\n\t\t\t\tprocesses\n");
    fprintf(stderr, "  -A, --appnodes\t\tuse the application nodes for communication\n\t\t\t\tprocesses\n");
    fprintf(stderr, "  -x, --exclusive\t\tdo not use the FE or BE nodes for communication\n\t\t\t\tprocesses\n");
    fprintf(stderr, "  -p, --procs <processes>\tthe maximum number of communication processes\n\t\t\t\tper node\n");
    fprintf(stderr, "\nMiscellaneous options:\n");
    fprintf(stderr, "  -C, --create [<args>]+\tlaunch the application under STAT's control.\n\t\t\t\tAll args after -C are used to launch the app.\n");
    fprintf(stderr, "  -I, --serial [<args>]+\tattach to a list of serial processes.\n\t\t\t\tAll args after -I are interpreted as processes.\n\t\t\t\tIn the form: [exe@][hostname:]pid\n");
    fprintf(stderr, "  -D, --daemon <path>\t\tthe full path to the STAT daemon\n");
    fprintf(stderr, "  -F, --filter <path>\t\tthe full path to the STAT filter shared object\n");
    fprintf(stderr, "  -l, --log [FE|BE|CP]\t\tenable debug logging\n");
    fprintf(stderr, "  -L, --logdir <log_directory>\tlogging output directory\n");
    fprintf(stderr, "  -M, --mrnetprintf\t\tuse MRNet's print for logging\n");
    fprintf(stderr, "  -j, --jobid <id>\t\tappend <id> to the STAT output directory\n");

#ifdef DYSECTAPI
    fprintf(stderr, "  -X, --dysectapi <session>\trun specified DySectAPI session\n");
    fprintf(stderr, "  -b, --dysectapi_batch <secs>\trun DySectAPI session in batch mode. Session\n\t\t\t\tstops after specified timeout or detach action.\n");
#endif
    fprintf(stderr, "\n%% man STAT\n  for more information\n\n");
}


//! Parses the arguments
/*
    param statArgs[out] - the return set of options
    statFrontEnd - the STAT_FrontEnd object
    argc - the number of args
    argv - the arg list
*/
StatError_t parseArgs(StatArgs_t *statArgs, STAT_FrontEnd *statFrontEnd, int argc, char **argv)
{
    int i, opt, optionIndex = 0;
    unsigned int logType = 0;
    bool createJob = false, serialJob = false;
    char logOutDir[BUFSIZE];
    string remotePid, remoteHost, nProcsOpt;
    string::size_type colonPos;
    StatError_t statError;
    struct option longOptions[] =
    {
        {"help",                no_argument,        0, 'h'},
        {"version",             no_argument,        0, 'V'},
        {"verbose",             no_argument,        0, 'v'},
        {"quiet",               no_argument,        0, 'q'},
        {"withpc",              no_argument,        0, 'P'},
        {"withmoduleoffset",    no_argument,        0, 'm'},
        {"withline",            no_argument,        0, 'i'},
        {"withopenmp",          no_argument,        0, 'o'},
        {"comprehensive",       no_argument,        0, 'c'},
        {"withthreads",         no_argument,        0, 'w'},
        {"pythontrace",         no_argument,        0, 'y'},
        {"autotopo",            no_argument,        0, 'a'},
        {"create",              no_argument,        0, 'C'},
        {"serial",              no_argument,        0, 'I'},
        {"appnodes",            no_argument,        0, 'A'},
        {"exclusive",           no_argument,        0, 'x'},
        {"sampleindividual",    no_argument,        0, 'S'},
        {"mrnetprintf",         no_argument,        0, 'M'},
        {"countrep",            no_argument,        0, 'U'},
        {"gdb",                 no_argument,        0, 'G'},
        {"cudaquick",           no_argument,        0, 'Q'},
        {"fanout",              required_argument,  0, 'f'},
        {"nodes",               required_argument,  0, 'n'},
        {"nodesfile",           required_argument,  0, 'N'},
        {"procs",               required_argument,  0, 'p'},
        {"jobid",               required_argument,  0, 'j'},
        {"retries",             required_argument,  0, 'r'},
        {"retryfreq",           required_argument,  0, 'R'},
        {"traces",              required_argument,  0, 't'},
        {"tracefreq",           required_argument,  0, 'T'},
        {"daemon",              required_argument,  0, 'D'},
        {"filter",              required_argument,  0, 'F'},
        {"sleep",               required_argument,  0, 's'},
        {"log",                 required_argument,  0, 'l'},
        {"logdir",              required_argument,  0, 'L'},
        {"usertopology",        required_argument,  0, 'u'},
        {"depth",               required_argument,  0, 'd'},
        {"daemonspernode",      required_argument,  0, 'z'},
        {"maxdaemonthreads",    required_argument,  0, 'H'},
#ifdef DYSECTAPI
        {"dysectapi",           required_argument, 0, 'X'},
        {"dysectapi_batch",     required_argument, 0, 'b'},
        {"dysectapi_arg",       required_argument, 0, 'Y'},
#endif
        {0,                     0,                  0, 0}
    };

    snprintf(logOutDir, BUFSIZE, "NULL");
    statArgs->sleepTime = -1;
    statArgs->nTraces = 10;
    statArgs->traceFrequency = 100;
    statArgs->nRetries = 0;
    statArgs->nDaemonsPerNode = 1;
    statArgs->retryFrequency = 100;
    statArgs->maxDaemonNumThreads = 512;
    statArgs->sampleType = 0;
    statArgs->topologyType = STAT_TOPOLOGY_AUTO;
    snprintf(statArgs->topologySpecification, BUFSIZE, "0");

    while (1)
    {
#ifdef DYSECTAPI
        opt = getopt_long(argc, argv,"hVvqPmiocwyaCIAxSMUGQf:n:p:j:r:R:t:T:d:F:s:l:L:u:D:z:H:X:b:Y:N:", longOptions, &optionIndex);
#else
        opt = getopt_long(argc, argv,"hVvqPmiocwyaCIAxSMUGQf:n:p:j:r:R:t:T:d:F:s:l:L:u:D:z:H:N:", longOptions, &optionIndex);
#endif
        if (opt == -1)
            break;
        if (opt == 'C')
        {
            createJob = true;
            break; /* All remaining args are for job launch */
        }
        else if (opt == 'I')
        {
            serialJob = true;
            break; /* All remaining args are for serial attach */
        }
        switch(opt)
        {
        case 'h':
            printUsage();
            return STAT_WARNING;
            break;
        case 'V':
            printf("STAT-%d.%d.%d\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
            return STAT_WARNING;
            break;
        case 'v':
            statFrontEnd->setVerbose(STAT_VERBOSE_FULL);
            break;
        case 'q':
            statFrontEnd->setVerbose(STAT_VERBOSE_ERROR);
            break;
        case 'w':
            statArgs->withThreads = true;
            statArgs->sampleType |= STAT_SAMPLE_THREADS;
            break;
        case 'y':
            statArgs->withPython = true;
            statArgs->sampleType |= STAT_SAMPLE_PYTHON;
            break;
        case 'P':
            statArgs->sampleType |= STAT_SAMPLE_PC;
            statArgs->comprehensive = false;
            break;
        case 'm':
            statArgs->sampleType |= STAT_SAMPLE_MODULE_OFFSET;
            statArgs->comprehensive = false;
            break;
        case 'i':
            statArgs->sampleType |= STAT_SAMPLE_LINE;
            statArgs->comprehensive = false;
            break;
        case 'o':
            statArgs->withOpenMP = true;
#ifdef OMP_STACKWALKER
            statArgs->sampleType |= STAT_SAMPLE_OPENMP;
#endif
            break;
        case 'c':
            statArgs->comprehensive = true;
            break;
        case 'a':
            statArgs->topologyType = STAT_TOPOLOGY_AUTO;
            break;
        case 'd':
            statArgs->topologyType = STAT_TOPOLOGY_DEPTH;
            snprintf(statArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'f':
            statArgs->topologyType = STAT_TOPOLOGY_FANOUT;
            snprintf(statArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'u':
            statArgs->topologyType = STAT_TOPOLOGY_USER;
            snprintf(statArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'S':
            statArgs->individualSamples = true;
            break;
        case 'A':
            statArgs->cpPolicy = STAT_CP_SHAREAPPNODES;
            break;
        case 'x':
            statArgs->cpPolicy = STAT_CP_EXCLUSIVE;
            break;
        case 'n':
            statArgs->nodeList = strdup(optarg);
            if (statArgs->nodeList == NULL)
            {
                statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) to statArgs->nodeList\n", strerror(errno), optarg);
                return STAT_ALLOCATE_ERROR;
            }
            break;
        case 'N':
            statFrontEnd->setNodeListFile(strdup(optarg));
            break;
        case 'p':
            statFrontEnd->setProcsPerNode(atoi(optarg));
            break;
        case 'j':
            statError = statFrontEnd->setJobId(optarg);
            if (statError != STAT_OK)
            {
                statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to set job ID\n");
                return statError;
            }
            break;
        case 'r':
            statArgs->nRetries = atoi(optarg);
            break;
        case 'R':
            statArgs->retryFrequency = atoi(optarg);
            break;
        case 't':
            statArgs->nTraces = atoi(optarg);
            statArgs->comprehensive = false;
            break;
        case 'T':
            statArgs->traceFrequency = atoi(optarg);
            break;
        case 'D':
            statError = statFrontEnd->setToolDaemonExe(optarg);
            if (statError != STAT_OK)
            {
                statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to set tool daemon exe path\n");
                return statError;
            }
            break;
        case 'F':
            statError = statFrontEnd->setFilterPath(optarg);
            if (statError != STAT_OK)
            {
                statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to set filter path\n");
                return statError;
            }
            break;
        case 's':
            statArgs->sleepTime = atoi(optarg);
            if (statArgs->sleepTime <= 0)
            {
                statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "'%s' is not a valid timeout, must be a positive integer\n", optarg);
                return STAT_ARG_ERROR;
            }
            break;
        case 'l':
            if (strcmp(optarg, "FE") == 0)
                logType |= STAT_LOG_FE;
            else if (strcmp(optarg, "BE") == 0)
                logType |= STAT_LOG_BE;
            else if (strcmp(optarg, "CP") == 0)
                logType |= STAT_LOG_CP;
            else if (strcmp(optarg, "SW") == 0)
                logType |= STAT_LOG_SW;
            else if (strcmp(optarg, "SWERR") == 0)
                logType |= STAT_LOG_SWERR;
            else
            {
                statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Log option must equal FE, BE, CP, SW, SWERR, you entered %s\n", optarg);
                return STAT_ARG_ERROR;
            }
            break;
        case 'L':
            snprintf(logOutDir, BUFSIZE, "%s", optarg);
            break;
        case 'M':
            logType |= STAT_LOG_MRN;
            break;
        case 'U':
            statArgs->countRep = true;
            statArgs->sampleType |= STAT_SAMPLE_COUNT_REP;
#ifdef STAT_GDB_BE
        case 'G':
            statArgs->applicationOption = STAT_GDB_ATTACH;
            break;
        case 'Q':
            statArgs->withCudaQuick = true;
            statArgs->sampleType |= STAT_SAMPLE_CUDA_QUICK;
            statArgs->comprehensive = false;
            break;
#endif
        case 'z':
            statArgs->nDaemonsPerNode = atoi(optarg);
            break;
        case 'H':
            statArgs->maxDaemonNumThreads = atoi(optarg);
            break;
#ifdef DYSECTAPI
        case 'X':
            setenv("STAT_GROUP_OPS", "1", 1);
            dysectApiEnabled = true;
            dysectApiSessionPath = strdup(optarg);
            break;
        case 'b':
            dysectTimeout = atoi(optarg);
            if (dysectTimeout <= 0)
            {
                statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "'%s' is not a valid timeout, must be a positive integer\n", optarg);
                return STAT_ARG_ERROR;
            }
            break;
        case 'Y':
            statArgs->dysectArgc++;
            statArgs->dysectArgv = (char **)realloc(statArgs->dysectArgv, statArgs->dysectArgc);
            if (statArgs->dysectArgv == NULL)
            {
                statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to realloc dysectArgv %d for %s\n", statArgs->dysectArgc, optarg);
                return STAT_ALLOCATE_ERROR;
            }
            statArgs->dysectArgv[statArgs->dysectArgc - 1] = strdup(optarg);
            break;
#endif
        case '?':
            statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Unknown option %c\n", opt);
            printUsage();
            return STAT_ARG_ERROR;
        default:
            statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Unknown option %c\n", opt);
            printUsage();
            return STAT_ARG_ERROR;
        }; /* switch(opt) */
    } /* while (1) */

    if (strcmp(logOutDir, "NULL") != 0 && logType != 0)
    {
        statError = statFrontEnd->startLog(logType, logOutDir);
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed start logging\n");
            return statError;
        }
    }

    if (optind == argc - 1 && (createJob == false && serialJob == false))
    {
#ifdef STAT_GDB_BE
        if (statArgs->applicationOption != STAT_GDB_ATTACH && statArgs->applicationOption != STAT_SERIAL_GDB_ATTACH)
            statArgs->applicationOption = STAT_ATTACH;
#else
        statArgs->applicationOption = STAT_ATTACH;
#endif
        remotePid = argv[optind++];
        colonPos = remotePid.find_first_of(":");
        if (colonPos != string::npos)
        {
            remoteHost = remotePid.substr(0, colonPos);
            statArgs->remoteNode = strdup(remoteHost.c_str());
            if (statArgs->remoteNode == NULL)
            {
                statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to set remote node from %s\n", remotePid.c_str());
                return STAT_ALLOCATE_ERROR;
            }
            statArgs->pid = atoi((remotePid.substr(colonPos + 1, remotePid.length())).c_str());
        }
        else
        {
            statArgs->pid = atoi(remotePid.c_str());
        }
    }
    else if (optind < argc - 1 && createJob == true)
    {
        statArgs->applicationOption = STAT_LAUNCH;
        for (i = optind; i < argc; i++)
            statFrontEnd->addLauncherArgv(argv[i]);
    }
    else if (optind < argc && serialJob == true)
    {
        if (statArgs->applicationOption == STAT_GDB_ATTACH)
            statArgs->applicationOption = STAT_SERIAL_GDB_ATTACH;
        else
            statArgs->applicationOption = STAT_SERIAL_ATTACH;
        for (i = optind; i < argc; i++)
            statFrontEnd->addSerialProcess(argv[i]);
    }
    else
    {
        printUsage();
        return STAT_ARG_ERROR;
    }

    return STAT_OK;
}


//! Sleep for a specified number of seconds
/*
    param sleepTime - the number of seconds to sleep
*/
void mySleep(int sleepTime)
{
    time_t currentTime;
    struct tm *localTime;

    if (sleepTime > 0)
    {
        currentTime = time(NULL);
        localTime = localtime(&currentTime);
        if (localTime == NULL)
            printf("STAT Sleeping for %d seconds... ", sleepTime);
        else
            printf("%s STAT Sleeping for %d seconds... ", asctime(localTime), sleepTime);
        fflush(stdout);
        sleep(sleepTime);
        printf("STAT has awoken\n");
        fflush(stdout);
    }
}

