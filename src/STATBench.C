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


/*********************************************************
 A benchmark that emulates STAT: the Stack Trace
 Analysis tool.

 Written by: Greg Lee
*********************************************************/


#include <getopt.h>
#include "config.h"
#include "STAT_FrontEnd.h"

using namespace MRN;
using namespace std;

//! A struct to encapsulate user command line args
typedef struct
{
    int maxDepth;                           /*!< the max trace depth */
    int nTasks;                             /*!< the number of tasks per daemon */
    int nTraces;                            /*!< the number of traces per task */
    int functionFanout;                     /*!< the max function fanout */
    int iters;                              /*!< the number of merge iterations */
    int nEqClasses;                         /*!< the number of equivalence classes */
    char topologySpecification[BUFSIZE];    /*!< the topology specification */
    char *nodeList;                         /*!< the list of nodes for CPs */
    StatCpPolicy_t cpPolicy;                /*!< whether to use the application nodes to run communication processes */
    unsigned int sampleType;                /*!< the sample level of detail */
    StatTopology_t topologyType;            /*!< the topology specification type */
} StatBenchArgs_t;

//! Prints the usage directions
void printUsage();

//! Parses the command line arguments and sets class variables
StatError_t parseArgs(StatBenchArgs_t *statBenchArgs, STAT_FrontEnd *statFrontEnd, int argc, char **argv);


//! The STATBench main
int main(int argc, char **argv)
{
    int i;
    char statBenchDaemon[BUFSIZE], perfData[BUFSIZE];
    STAT_FrontEnd *statFrontEnd;
    StatError_t statError;
    StatBenchArgs_t *statBenchArgs;

    statFrontEnd = new STAT_FrontEnd();

    /* Set default values */
    statBenchArgs = (StatBenchArgs_t *)calloc(1, sizeof(StatBenchArgs_t));
    if (statBenchArgs == NULL)
    {
        statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to calloc statBenchArgs\n");
        free(statBenchArgs);
        return -1;
    }
    statBenchArgs->sampleType = 0;
    statBenchArgs->maxDepth = 7;
    statBenchArgs->nTasks = 128;
    statBenchArgs->nTraces = 3;
    statBenchArgs->functionFanout = 2;
    statBenchArgs->iters = 5;
    statBenchArgs->nEqClasses = -1;
    snprintf(statBenchDaemon, BUFSIZE, "%s/bin/STATBenchD", statFrontEnd->getInstallPrefix());
    statFrontEnd->setToolDaemonExe(statBenchDaemon);

    /* Parse the arguments */
    statError = parseArgs(statBenchArgs, statFrontEnd, argc, argv);
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to parse arguments\n");
        free(statBenchArgs);
        return -1;
    }

    /* Push the arguments into the output file */
    snprintf(perfData, BUFSIZE, "STATBench invoked with: ");
    for (i = 0; i < argc; i++)
    {
        if (strlen(perfData) + strlen(argv[i]) + 1 >= BUFSIZE)
            break;
        strcat(perfData, argv[i]);
        strcat(perfData, " ");
    }
    if (strlen(perfData) < BUFSIZE - 1)
        strcat(perfData, "\n");
    statFrontEnd->addPerfData(perfData, -1.0);

    /* Print the STATBench header to the output file and to the screen */
    snprintf(perfData, BUFSIZE,
         "#######################################\n"
         "# STATBench: STAT emulation Benchmark #\n"
         "#######################################\n"
         "This benchmark emulates STAT, the Stack Trace\n"
         "Analysis Tool and can determine the expected\n"
         "performance for a specified machine architecture\n"
         "and application profile.\n\n"
         "Running %d iterations with %d tasks, %d traces per task,\n"
         "%d max call depth after main, %d function fanout, and %d\n"
         "equivalence classes.\n\n",
         statBenchArgs->iters, statBenchArgs->nTasks, statBenchArgs->nTraces, statBenchArgs->maxDepth, statBenchArgs->functionFanout, statBenchArgs->nEqClasses);

    statFrontEnd->addPerfData(perfData, -1.0);
    printf("%s", perfData);

    /* Launch the MRNet Tree */
    statFrontEnd->setApplicationOption(STAT_LAUNCH);
    statError = statFrontEnd->launchAndSpawnDaemons(NULL, true);
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to launch MRNet tree()\n");
        statFrontEnd->shutDown();
        delete statFrontEnd;
        free(statBenchArgs);
        return -1;
    }

    /* Launch the MRNet Tree */
    statError = statFrontEnd->launchMrnetTree(statBenchArgs->topologyType, statBenchArgs->topologySpecification, statBenchArgs->nodeList, true, statBenchArgs->cpPolicy);
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to launch MRNet tree()\n");
        statFrontEnd->shutDown();
        delete statFrontEnd;
        free(statBenchArgs);
        return -1;
    }

    statError = statFrontEnd->setupConnectedMrnetTree();
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to setup connected MRNet tree\n");
        delete statFrontEnd;
        free(statBenchArgs);
        return -1;
    }

    /* Generate the traces */
    statError = statFrontEnd->statBenchCreateStackTraces(statBenchArgs->maxDepth, statBenchArgs->nTasks, statBenchArgs->nTraces, statBenchArgs->functionFanout, statBenchArgs->nEqClasses, statBenchArgs->sampleType);
    if (statError != STAT_OK)
    {
        statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to generate stack traces\n");
        statFrontEnd->shutDown();
        free(statBenchArgs);
        return -1;
    }

    /* Gather the traces */
    for (i = 0; i < statBenchArgs->iters; i++)
    {
        statError = statFrontEnd->gatherLastTrace();
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to gather stack traces\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statBenchArgs);
            return -1;
        }
    }

    /* Gather the traces */
    for (i = 0; i < statBenchArgs->iters; i++)
    {
        statError = statFrontEnd->gatherTraces();
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed to gather stack traces\n");
            statFrontEnd->shutDown();
            delete statFrontEnd;
            free(statBenchArgs);
            return -1;
        }
    }

    statFrontEnd->shutDown();
    printf("\nResults written to %s\n\n", statFrontEnd->getOutDir());

    delete statFrontEnd;
    if (statBenchArgs != NULL)
        free(statBenchArgs);

    return 0;
}


//! Prints the usage message
void printUsage()
{
    fprintf(stderr, "\nUsage:\n\tSTATBench [ OPTIONS ]\n\n");
    fprintf(stderr, "OPTIONS:\n");
    fprintf(stderr, "\nStack trace generation options:\n");
    fprintf(stderr, "  -t, --traces <count>\t\tgenerate <count> traces.\n");
    fprintf(stderr, "  -i, --iters <count>\t\tperform <count> gathers.\n");
    fprintf(stderr, "  -N, --numtasks <count>\temulate <count> tasks per daemon.\n");
    fprintf(stderr, "  -m, --maxdepth <depth>\tgenerate traces with a maximum depth of <depth>.\n");
    fprintf(stderr, "  -b, --branch <width>\t\tgenerate traces with a max branching factor of\n\t\t\t\t<width>.\n");
    fprintf(stderr, "  -e, --eqclasses <count>\tgenerate traces within <count> equivalent\n\t\t\t\tclasses.\n\n");
    fprintf(stderr, "  -U, --countrep\t\tonly gather count and a single representative\n");
    fprintf(stderr, "Topology options:\n");
    fprintf(stderr, "  -a, --autotopo\t\tlet STAT automatically create topology\n");
    fprintf(stderr, "  -d, --depth <depth>\t\ttree topology depth\n");
    fprintf(stderr, "  -f, --fanout <width>\t\tmaximum tree topology fanout\n");
    fprintf(stderr, "  -u, --usertopology <topology>\tspecify the number of communication nodes per\n\t\t\t\tlayer in the tree topology, separated by dashes\n");
    fprintf(stderr, "  -n, --nodes <nodelist>\tlist of nodes for communication processes\n");
    fprintf(stderr, "\t\t\t\tExample node lists:\thost1\n\t\t\t\t\t\t\thost1,host2\n\t\t\t\t\t\t\thost[1,5-7,9]\n");
    fprintf(stderr, "  -A, --appnodes\t\tuse the application nodes for communication\n\t\t\t\tprocesses\n");
    fprintf(stderr, "  -x, --exclusive\t\tdo not use the FE or BE nodes for communication\n\t\t\t\tprocesses\n");
    fprintf(stderr, "  -p, --procs <processes>\tthe maximum number of communication processes\n\t\t\t\tper node\n");
    fprintf(stderr, "\nMiscellaneous options:\n");
    fprintf(stderr, "  -D, --daemon <path>\t\tthe full path to the STAT daemon\n");
    fprintf(stderr, "  -F, --filter <path>\t\tthe full path to the STAT filter shared object\n");
    fprintf(stderr, "  -l, --log [FE|BE|ALL]\t\tenable debug logging\n");
    fprintf(stderr, "  -L, --logdir <log_directory>\tlogging output directory\n");
    fprintf(stderr, "  -M, --mrnetprintf\t\tuse MRNet's print for logging\n");
    fprintf(stderr, "\n%% man STATBench\n  for more information\n\n");
}


//! Parses the arguments
/*
    param statArgs[out] - the return set of options
    statFrontEnd - the STAT_FrontEnd object
    argc - the number of args
    argv - the arg list
*/
StatError_t parseArgs(StatBenchArgs_t *statBenchArgs, STAT_FrontEnd *statFrontEnd, int argc, char **argv)
{
    int i, opt, optionIndex = 0;
    unsigned int logType = 0;
    char logOutDir[BUFSIZE];
    StatError_t statError;
    struct option longOptions[] =
    {
        {"help",            no_argument,        0, 'h'},
        {"version",         no_argument,        0, 'V'},
        {"verbose",         no_argument,        0, 'v'},
        {"autotopo",        no_argument,        0, 'a'},
        {"appnodes",        no_argument,        0, 'A'},
        {"exclusive",       no_argument,        0, 'x'},
        {"mrnetprintf",     no_argument,        0, 'M'},
        {"countrep",        no_argument,        0, 'U'},
        {"fanout",          required_argument,  0, 'f'},
        {"nodes",           required_argument,  0, 'n'},
        {"procs",           required_argument,  0, 'p'},
        {"traces",          required_argument,  0, 't'},
        {"maxdepth",        required_argument,  0, 'm'},
        {"branch",          required_argument,  0, 'b'},
        {"eqclasses",       required_argument,  0, 'e'},
        {"daemon",          required_argument,  0, 'D'},
        {"filter",          required_argument,  0, 'F'},
        {"log",             required_argument,  0, 'l'},
        {"logdir",          required_argument,  0, 'L'},
        {"usertopology",    required_argument,  0, 'u'},
        {"depth",           required_argument,  0, 'd'},
        {"numtasks",        required_argument,  0, 'N'},
        {"iters",           required_argument,  0, 'i'},
        {0,                 0,                  0, 0}
    };

    snprintf(logOutDir, BUFSIZE, "NULL");
    statBenchArgs->topologyType = STAT_TOPOLOGY_DEPTH;
    snprintf(statBenchArgs->topologySpecification, BUFSIZE, "0");

    while (1)
    {
        opt = getopt_long(argc, argv,"hVvaAxMUf:n:p:t:m:b:e:D:F:l:L:u:d:N:i:", longOptions, &optionIndex);
        if (opt == -1)
            break;
        switch(opt)
        {
        case 'h':
            printUsage();
            exit(0);
            break;
        case 'V':
            printf("STATBench-%d.%d.%d\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
            exit(0);
            break;
        case 'v':
            statFrontEnd->setVerbose(STAT_VERBOSE_FULL);
            break;
        case 'a':
            statBenchArgs->topologyType = STAT_TOPOLOGY_AUTO;
            break;
        case 'f':
            statBenchArgs->topologyType = STAT_TOPOLOGY_FANOUT;
            snprintf(statBenchArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'n':
            statBenchArgs->nodeList = strdup(optarg);
            if (statBenchArgs->nodeList == NULL)
            {
                statFrontEnd->printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) to statBenchArgs->nodeList\n", strerror(errno), optarg);
                return STAT_ALLOCATE_ERROR;
            }
            break;
        case 'A':
            statBenchArgs->cpPolicy = STAT_CP_SHAREAPPNODES;
            break;
        case 'x':
            statBenchArgs->cpPolicy = STAT_CP_EXCLUSIVE;
            break;
        case 'p':
            statFrontEnd->setProcsPerNode(atoi(optarg));
            break;
        case 't':
            statBenchArgs->nTraces = atoi(optarg);
            break;
        case 'm':
            statBenchArgs->maxDepth = atoi(optarg);
            break;
        case 'b':
            statBenchArgs->functionFanout = atoi(optarg);
            break;
        case 'e':
            statBenchArgs->nEqClasses = atoi(optarg);
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
        case 'l':
            if (strcmp(optarg, "FE") == 0)
                logType |= STAT_LOG_FE;
            else if (strcmp(optarg, "BE") == 0)
                logType |= STAT_LOG_BE;
            else if (strcmp(optarg, "CP") == 0)
                logType |= STAT_LOG_CP;
            else
            {
                statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Log option must equal FE, BE, or CP, you entered %s\n", optarg);
                return STAT_ARG_ERROR;
            }
            break;
        case 'L':
            snprintf(logOutDir, BUFSIZE, "%s", optarg);
            break;
        case 'M':
            logType |= STAT_LOG_MRN;
            break;
        case 'u':
            statBenchArgs->topologyType = STAT_TOPOLOGY_USER;
            snprintf(statBenchArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'd':
            statBenchArgs->topologyType = STAT_TOPOLOGY_DEPTH;
            snprintf(statBenchArgs->topologySpecification, BUFSIZE, "%s", optarg);
            break;
        case 'N':
            statBenchArgs->nTasks = atoi(optarg);
            break;
        case 'i':
            statBenchArgs->iters = atoi(optarg);
            break;
        case 'U':
            statBenchArgs->sampleType |= STAT_SAMPLE_COUNT_REP;
            break;
        case '?':
            statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Unknown option %c\n", opt);
            printUsage();
            return STAT_ARG_ERROR;
        default:
            statFrontEnd->printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Unknown option %c\n", opt);
            printUsage();
            return STAT_ARG_ERROR;
        }; /* switch */
    } /* while(1) */

    if (strcmp(logOutDir, "NULL") != 0 && logType != 0)
    {
        statError = statFrontEnd->startLog(logType, logOutDir);
        if (statError != STAT_OK)
        {
            statFrontEnd->printMsg(statError, __FILE__, __LINE__, "Failed start logging\n");
            return statError;
        }
    }

    if (statFrontEnd->getLauncherArgc() == 1)
        statFrontEnd->addLauncherArgv("srun");
    statFrontEnd->addLauncherArgv(statFrontEnd->getToolDaemonExe());
    statFrontEnd->addLauncherArgv("--STATBench");
    if (logType & STAT_LOG_BE)
    {
        if (strcmp(logOutDir, "NULL") != 0)
        {
            statFrontEnd->addLauncherArgv("-L");
            statFrontEnd->addLauncherArgv(logOutDir);
        }
        statFrontEnd->addLauncherArgv("-l");
        statFrontEnd->addLauncherArgv("BE");
    }

    return STAT_OK;
}

