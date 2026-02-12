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
#include "STAT_FrontEnd.h"

using namespace std;
using namespace MRN;

#ifdef STAT_FGFS
    using namespace FastGlobalFileStatus;
    using namespace FastGlobalFileStatus::MountPointAttribute;
    using namespace FastGlobalFileStatus::CommLayer;
    using namespace FastGlobalFileStatus::MountPointAttribute;
#endif

//! the bit vector routines
extern graphlib_functiontable_p gStatBitVectorFunctions;

//! the count and representative routines
extern graphlib_functiontable_p gStatCountRepFunctions;

//! the bit vector to reorder the bit vectors by MPI rank
extern graphlib_functiontable_p gStatReorderFunctions;

//! the current index into the bit vector
extern int gStatGraphRoutinesCurrentIndex;

//! the ranks list for the current bit vector
extern int *gStatGraphRoutinesRanksList;

//! the length of the ranks list
extern int gStatGraphRoutinesRanksListLength;

//! frame label to indicate an error
const char *gErrorLabel = "daemon error";

//! start timer for composite operations
STAT_timer gTotalgStartTime;

//! end timer for composite operations
STAT_timer gTotalgEndTime;

//! start timer for individual operations
STAT_timer gStartTime;

//! end timer for individual operations
STAT_timer gEndTime;

//! the STAT log handle, global for filter access
FILE *gStatOutFp = NULL;

//! The number of BEs that have connected
unsigned int gNumCallbacks;

//! A mutex to protect data/events during a callback
pthread_mutex_t gMrnetCallbackMutex = PTHREAD_MUTEX_INITIALIZER;

extern const char *gNodeAttrs[];
extern const char *gEdgeAttrs[];
extern int gNumNodeAttrs;
extern int gNumEdgeAttrs;


STAT_FrontEnd::STAT_FrontEnd()
{
    int intRet;
    char tmp[BUFSIZE], *envValue;
    struct timeval timeStamp;
    time_t currentTime;
    char timeBuf[BUFSIZE];
    string invocationString;
    graphlib_error_t graphlibError;

    /* Enable MRNet logging if requested */
    mrnetOutputLevel_ = 0;
    envValue = getenv("STAT_MRNET_OUTPUT_LEVEL");
    if (envValue != NULL)
    {
        mrnetOutputLevel_ = atoi(envValue);
        set_OutputLevel(mrnetOutputLevel_);
    }
    envValue = getenv("STAT_MRNET_DEBUG_LOG_DIRECTORY");
    if (envValue != NULL)
        setenv("MRNET_DEBUG_LOG_DIRECTORY", envValue, 1);

    /* Set MRNet and LMON rsh command */
    envValue = getenv("STAT_XPLAT_RSH");
    if (envValue != NULL)
        setenv("XPLAT_RSH", envValue, 1);

    /* Set the launchmon and mrnet_commnode paths based on STAT env vars */
    envValue = getenv("STAT_MRNET_COMM_PATH");
    if (envValue != NULL)
    {
        setenv("MRN_COMM_PATH", envValue, 1); /* for MRNet > 3.0.1 */
        setenv("MRNET_COMM_PATH", envValue, 1);
    }

    /* Set MRNet configuration parameters */
    envValue = getenv("STAT_MRNET_STARTUP_TIMEOUT");
    if (envValue != NULL)
        setenv("MRNET_STARTUP_TIMEOUT", envValue, 1);
    envValue = getenv("STAT_MRNET_PORT_BASE");
    if (envValue != NULL)
        setenv("MRNET_PORT_BASE", envValue, 1);

    /* Set the daemon path and filter paths to the environment variable
       specification if applicable.  Otherwise, set to default installation
       paths */
    envValue = getenv("STAT_DAEMON_PATH");
    if (envValue != NULL)
    {
        toolDaemonExe_ = strdup(envValue);
        if (toolDaemonExe_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set tool daemon exe with strdup()\n", strerror(errno));
            exit(STAT_ALLOCATE_ERROR);
        }
    }
    else
    {
        if (strlen(getInstallPrefix()) > 1)
            snprintf(tmp, BUFSIZE, "%s/bin/STATD", getInstallPrefix());
        else
            snprintf(tmp, BUFSIZE, "STATD");
        toolDaemonExe_ = strdup(tmp);
        if (toolDaemonExe_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set tool daemon exe with strdup()\n", strerror(errno));
            exit(STAT_ALLOCATE_ERROR);
        }
    }

    envValue = getenv("STAT_FILTER_PATH");
    if (envValue != NULL)
    {
        filterPath_ = strdup(envValue);
        if (filterPath_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set tool filter path with strdup()\n", strerror(errno));
            exit(STAT_ALLOCATE_ERROR);
        }
    }
    else
    {
        if (strlen(getInstallPrefix()) > 1)
            snprintf(tmp, BUFSIZE, "%s/lib/STAT_FilterDefinitions.so", getInstallPrefix());
        else
            snprintf(tmp, BUFSIZE, "STAT_FilterDefinitions.so");
        filterPath_ = strdup(tmp);
        if (filterPath_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set filter path with strdup()\n", strerror(errno));
            exit(STAT_ALLOCATE_ERROR);
        }
    }

#ifdef STAT_FGFS
    fgfsCommFabric_ = NULL;
    fileRequestStream_ = NULL;
    envValue = getenv("STAT_FGFS_FILTER_PATH");
    if (envValue != NULL)
    {
        fgfsFilterPath_ = strdup(envValue);
        if (fgfsFilterPath_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set FGFS filter path with strdup()\n", strerror(errno));
            exit(STAT_ALLOCATE_ERROR);
        }
    }
    else
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "STAT_FGFS_FILTER_PATH environment variable not set\n");
        exit(STAT_ARG_ERROR);
    }
#endif

#ifdef DYSECTAPI
    daemonsKilled_ = false;
#endif

    graphlibError = graphlib_Init();
    if (GRL_IS_FATALERROR(graphlibError))
    {
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to initialize graphlib\n");
        exit(STAT_GRAPHLIB_ERROR);
    }
    statInitializeReorderFunctions();
    statInitializeBitVectorFunctions();
    statInitializeCountRepFunctions();
    statInitializeMergeFunctions();

    /* Get the FE hostname */
    string temp;
    intRet = XPlat::NetUtils::GetLocalHostName(temp);
    if (intRet == 0)
        snprintf(hostname_, BUFSIZE, "%s", temp.c_str());
    else
    {
        intRet = gethostname(hostname_, BUFSIZE);
        if (intRet != 0)
            printMsg(STAT_WARNING, __FILE__, __LINE__, "gethostname failed with error code %d\n", intRet);
    }

    /* Initialize variables */
    launcherPid_ = 0;
    nApplNodes_ = 0;
    nApplProcs_ = 0;
    nDaemonsPerNode_ = 1;
#ifdef STAT_PROCS_PER_NODE
    procsPerNode_ = STAT_PROCS_PER_NODE;
#else
    procsPerNode_ = 64;
#endif
    envValue = getenv("STAT_PROCS_PER_NODE");
    if (envValue != NULL)
        procsPerNode_ = atoi(envValue);
    launcherArgc_ = 1;
    topologySize_ = 0;
    logging_ = 0;
    jobId_ = NULL;
    launcherArgv_ = NULL;
    applExe_ = NULL;
    remoteNode_ = NULL;
    nodeListFile_ = NULL;
    snprintf(outDir_, BUFSIZE, "NULL");
    snprintf(logOutDir_, BUFSIZE, "NULL");
    snprintf(filePrefix_, BUFSIZE, "NULL");
    snprintf(lastDotFileName_, BUFSIZE, "NULL");
    isStatBench_ = false;
    isLaunched_ = false;
    isConnected_ = false;
    isAttached_ = false;
    isRunning_ = false;
    isPendingAck_ = false;
    hasFatalError_ = false;
    checkNodeAccess_ = false;
    verbose_ = STAT_VERBOSE_STDOUT;
    applicationOption_ = STAT_ATTACH;
    network_ = NULL;
    broadcastCommunicator_ = NULL;
    broadcastStream_ = NULL;
    mergeStream_ = NULL;
    gStatOutFp = NULL;

    /* Add invocation time to perf file */
    gettimeofday(&timeStamp, NULL);
    currentTime = timeStamp.tv_sec;
    strftime(timeBuf, BUFSIZE, "%Y-%m-%d-%T", localtime(&currentTime));
    invocationString = "STAT started at ";
    invocationString.append(timeBuf);
    addPerfData(invocationString.c_str(), -1.0);
}


STAT_FrontEnd::~STAT_FrontEnd()
{
    unsigned int i;
    int intRet;
    map<int, IntList_t *>::iterator mrnetRankToMpiRanksMapIter;
    StatError_t statError;
    graphlib_error_t graphlibError;

    if (strcmp(outDir_, "NULL") != 0)
    {
        statError = dumpPerf();
        if (statError != STAT_OK)
            printMsg(statError, __FILE__, __LINE__, "Failed to dump performance results\n");
    }

    if (launcherArgv_ != NULL)
    {
        for (i = 0; i < launcherArgc_; i++)
        {
            if (launcherArgv_[i] != NULL)
                free(launcherArgv_[i]);
        }
        free(launcherArgv_);
        launcherArgv_ = NULL;
    }
    if (toolDaemonExe_ != NULL)
    {
        free(toolDaemonExe_);
        toolDaemonExe_ = NULL;
    }
    if (applExe_ != NULL)
    {
        free(applExe_);
        applExe_ = NULL;
    }
    if (filterPath_ != NULL)
    {
        free(filterPath_);
        filterPath_ = NULL;
    }
    if (remoteNode_ != NULL)
    {
        free(remoteNode_);
        remoteNode_ = NULL;
    }
    if (nodeListFile_ != NULL)
    {
        free(nodeListFile_);
        nodeListFile_ = NULL;
    }
    if (broadcastStream_ != NULL)
    {
        delete broadcastStream_;
        broadcastStream_ = NULL;
    }
    if (mergeStream_ != NULL)
    {
        delete mergeStream_;
        mergeStream_ = NULL;
    }
#ifdef STAT_FGFS
    if (fgfsFilterPath_ != NULL)
    {
        free(fgfsFilterPath_);
        fgfsFilterPath_ = NULL;
    }
    if (fileRequestStream_ != NULL)
    {
        delete fileRequestStream_;
        fileRequestStream_ = NULL;
    }
    if (fgfsCommFabric_ != NULL)
    {
        Stream *fgfsStream = (Stream *)fgfsCommFabric_->getChannel();
        if (fgfsStream != NULL)
        {
            delete fgfsStream;
            fgfsStream = NULL;
        }
        delete fgfsCommFabric_;
        fgfsCommFabric_ = NULL;
    }
#endif
    if (network_ != NULL)
    {
        delete network_;
        network_ = NULL;
    }
    for (mrnetRankToMpiRanksMapIter = mrnetRankToMpiRanksMap_.begin(); mrnetRankToMpiRanksMapIter != mrnetRankToMpiRanksMap_.end(); mrnetRankToMpiRanksMapIter++)
    {
        if (mrnetRankToMpiRanksMapIter->second != NULL)
        {
            if (mrnetRankToMpiRanksMapIter->second->list != NULL)
                free(mrnetRankToMpiRanksMapIter->second->list);
            free(mrnetRankToMpiRanksMapIter->second);
        }
    }
    communicationNodeSet_.clear();

    if (gStatOutFp != NULL)
    {
        intRet = fclose(gStatOutFp);
        if (intRet != 0)
            printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "%s: Failed to close gStatOutFp\n", strerror(errno));
        gStatOutFp = NULL;
    }

    statFreeReorderFunctions();
    statFreeBitVectorFunctions();
    statFreeCountRepFunctions();
    statFreeMergeFunctions();
    graphlibError = graphlib_Finish();
    if (GRL_IS_FATALERROR(graphlibError))
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to finish graphlib\n");
}


StatError_t STAT_FrontEnd::attachAndSpawnDaemons(unsigned int pid, char *remoteNode)
{
    StatError_t statError;

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Attaching to job launcher %s:%d and launching tool daemons...\n", remoteNode, pid);

    isStatBench_ = false;
    launcherPid_ = pid;
    if (remoteNode_ != NULL)
        free(remoteNode_);
    if (remoteNode != NULL)
    {
        remoteNode_ = strdup(remoteNode);
        if (remoteNode_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) for remoteNode_\n", strerror(errno), remoteNode);
            return STAT_ALLOCATE_ERROR;
        }
    }
    else
        remoteNode_ = NULL;

    statError = launchDaemons();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to attach and spawn daemons\n");
        return statError;
    }

    return STAT_OK;
}

StatError_t STAT_FrontEnd::launchAndSpawnDaemons(char *remoteNode, bool isStatBench)
{
    StatError_t statError;

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Launching application and tool daemons...\n");

    isStatBench_ = isStatBench;
    if (remoteNode_ != NULL)
        free(remoteNode_);
    if (remoteNode != NULL)
    {
        remoteNode_ = strdup(remoteNode);
        if (remoteNode_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) for remoteNode_\n", strerror(errno), remoteNode);
            return STAT_ALLOCATE_ERROR;
        }
    }
    else
        remoteNode_ = NULL;

    statError = launchDaemons();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to launch and spawn daemons\n");
        return statError;
    }

    return STAT_OK;
}

void beConnectCb(Event *event, void *dummy)
{
    if ((event->get_Class() == Event::TOPOLOGY_EVENT) && (event->get_Type() == TopologyEvent::TOPOL_ADD_BE))
    {
        pthread_mutex_lock(&gMrnetCallbackMutex);
        gNumCallbacks++;
        pthread_mutex_unlock(&gMrnetCallbackMutex);
    }
}


void nodeRemovedCb(Event *event, void *statObjectPtr)
{
    STAT_FrontEnd* statObject = (STAT_FrontEnd *)statObjectPtr;
    StatError_t statError;


    if (!statObject->isAttached())
    {
        // No need to generate warnings here if the node removal is just being
        // reported after the notification for application detaching.
        return;
    }

    if ((event->get_Class() == Event::TOPOLOGY_EVENT) && (event->get_Type() == TopologyEvent::TOPOL_REMOVE_NODE))
    {
        pthread_mutex_lock(&gMrnetCallbackMutex);
#ifndef DYSECTAPI
        statObject->printMsg(STAT_WARNING, __FILE__, __LINE__, "MRNet detected a tool process exit.  Recovering with available resources.\n");
#endif
        statError = ((STAT_FrontEnd *)statObject)->setRanksList();

#ifdef DYSECTAPI
        if (statError != STAT_APPLICATION_EXITED && statError != STAT_OK)
#else
        if (statError != STAT_OK)
#endif
        {
            statObject->printMsg(statError, __FILE__, __LINE__, "An error occurred when trying to recover from node removal\n");
            statObject->setHasFatalError(true);
        }
        pthread_mutex_unlock(&gMrnetCallbackMutex);
    }
}


void topologyChangeCb(Event *event, void *statObject)
{
    StatError_t statError;

    if ((event->get_Class() == Event::TOPOLOGY_EVENT) && (event->get_Type() == TopologyEvent::TOPOL_CHANGE_PARENT))
    {
        pthread_mutex_lock(&gMrnetCallbackMutex);
        ((STAT_FrontEnd *)statObject)->printMsg(STAT_WARNING, __FILE__, __LINE__, "MRNet detected a topology change.  Updating bit vector map.\n");
        statError = ((STAT_FrontEnd *)statObject)->setRanksList();
        if (statError != STAT_OK)
            ((STAT_FrontEnd *)statObject)->printMsg(statError, __FILE__, __LINE__, "An error occurred when trying to adjust to topology change\n");
        pthread_mutex_unlock(&gMrnetCallbackMutex);
    }
}


StatError_t STAT_FrontEnd::launchMrnetTree(StatTopology_t topologyType, char *topologySpecification, char *nodeList, bool blocking, StatCpPolicy_t cpPolicy)
{
    int daemonArgc = 1, statArgc, i;
    unsigned int j, currentArg;
    char topologyFileName[BUFSIZE], **daemonArgv = NULL;
    bool boolRet;
    StatError_t statError;

    if (isLaunched_ == false)
    {
        printMsg(STAT_NOT_LAUNCHED_ERROR, __FILE__, __LINE__, "BEs not launched yet.\n");
        return STAT_NOT_LAUNCHED_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\tLaunching MRNet tree\n");
    gStartTime.setTime();

    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Creating MRNet topology file\n");
    statError = createTopology(topologyFileName, topologyType, topologySpecification, nodeList, cpPolicy);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to create topology file\n");
        return statError;
    }
    gEndTime.setTime();
    addPerfData("\tCreate Topology File Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tMRNet topology %s created\n", topologyFileName);

    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tInitializing MRNet...\n");
    gStartTime.setTime();

    if (applicationOption_ == STAT_SERIAL_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
    {

#ifdef STAT_GDB_BE
        if (applicationOption_ == STAT_SERIAL_GDB_ATTACH)
        {
            // On PPC64LE systems the FE environment is not passed to the daemons.
            // We need to send PYTHONPATH for the GDB BE component.
            // We also need to send the GDB path since this variable isn't propagated either.
            const char *gdbCommand, *pythonPath;
            pythonPath = getenv("PYTHONPATH");
            if (pythonPath == NULL)
                pythonPath = ":";
            gdbCommand = getenv("STAT_GDB");
            if (gdbCommand == NULL)
                gdbCommand = "gdb";
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Using STAT GDB attach %s and PYTHONPATH %s\n", gdbCommand, pythonPath);
            daemonArgc += 4;
            daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
            if (daemonArgv == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
                return STAT_ALLOCATE_ERROR;
            }
            daemonArgv[daemonArgc - 5] = strdup("-P");
            daemonArgv[daemonArgc - 4] = strdup(pythonPath);
            daemonArgv[daemonArgc - 3] = strdup("-G");
            daemonArgv[daemonArgc - 2] = strdup(gdbCommand);
            daemonArgv[daemonArgc - 1] = NULL;
        }
#endif
        statError = addDaemonSerialProcArgs(daemonArgc, daemonArgv);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to add daemon process args\n");              free(daemonArgv);
            return statError;
        }

        statError = addDaemonLogArgs(daemonArgc, daemonArgv);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to add daemon logging args\n");              free(daemonArgv);
            return statError;
        }

        daemonArgc += 1;
        daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
        if (daemonArgv == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s realloc failed to allocate for daemon argv\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }

        daemonArgv[daemonArgc - 2] = strdup("-M");
        if (daemonArgv[daemonArgc - 2] == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), daemonArgc - 2);
            return STAT_ALLOCATE_ERROR;
        }
        daemonArgv[daemonArgc - 1] = NULL;

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\tCalling CreateNetworkFE with %d args:\n", daemonArgc);
        for (i = 0; i < daemonArgc; i++)
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\targv[%d] = %s\n", i, daemonArgv[i]);
        network_ = Network::CreateNetworkFE(topologyFileName, toolDaemonExe_, (const char **)daemonArgv);
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\tCalled CreateNetworkFE with %d args:\n", daemonArgc);
        isConnected_ = true;
        if (daemonArgv != NULL)
        {
            for (i = 0; i < daemonArgc; i++)
            {
                if (daemonArgv[i] == NULL)
                    break;
                free(daemonArgv[i]);
            }
            free(daemonArgv);
        }
    } /* if (applicationOption_ == STAT_SERIAL_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH) */
    else
    {
        statError = createMRNetNetwork(topologyFileName);
        if (statError != STAT_OK)
        {
            return statError;
        }


    } /* else branch of if (applicationOption_ == STAT_SERIAL_ATTAC || applicationOption_ == STAT_SERIAL_GDB_ATTACHH) */

    if (network_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Network initialization failure\n");
        return STAT_MRNET_ERROR;
    }
    if (network_->has_Error() == true)
    {
        isConnected_ = false;
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "MRNet reported a Network error with message: %s\n", network_->get_ErrorStr(network_->get_Error()));
        return STAT_MRNET_ERROR;
    }

    gEndTime.setTime();
    addPerfData("\tMRNet Constructor Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tMRNet initialized\n");

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Registering topology event callback with MRNet\n");
    gNumCallbacks = 0;
    boolRet = network_->register_EventCallback(Event::TOPOLOGY_EVENT, TopologyEvent::TOPOL_ADD_BE, beConnectCb, NULL);
    if (boolRet == false)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to register MRNet BE connect event callback\n");
        return STAT_MRNET_ERROR;
    }

    boolRet = network_->register_EventCallback(Event::TOPOLOGY_EVENT, TopologyEvent::TOPOL_REMOVE_NODE, nodeRemovedCb, (void *)this);
    if (boolRet == false)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to register MRNet node removed callback\n");
        return STAT_MRNET_ERROR;
    }

    boolRet = network_->register_EventCallback(Event::TOPOLOGY_EVENT, TopologyEvent::TOPOL_CHANGE_PARENT, topologyChangeCb, (void *)this);
    if (boolRet == false)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to register MRNet topology change callback\n");
        return STAT_MRNET_ERROR;
    }

    leafInfo_.networkTopology = network_->get_NetworkTopology();
    if (leafInfo_.networkTopology == NULL)
    {
        network_->print_error("Failed to get MRNet Network Topology");
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Network topology gather failure\n");
        return STAT_MRNET_ERROR;
    }
    topologySize_ = leafInfo_.networkTopology->get_NumNodes();

    if (applicationOption_ == STAT_SERIAL_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
        topologySize_ -= nApplNodes_; /* We need topologySize_ to not include BEs */

    leafInfo_.daemons = applicationNodeMultiSet_;

    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tConnecting to daemons...\n");
    if (isStatBench_ == false)
    {
        /* for STATbench we do this when creating traces since the proctab is modified */
        gStartTime.setTime();
        statError = createDaemonRankMap();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to create daemon rank map\n");
            return statError;
        }
        gEndTime.setTime();
        addPerfData("\tCreate Daemon Rank Map Time", (gEndTime - gStartTime).getDoubleTime());
    }

    leafInfo_.networkTopology->get_Leaves(leafInfo_.leafCps);

    if (applicationOption_ != STAT_SERIAL_ATTACH && applicationOption_ != STAT_SERIAL_GDB_ATTACH)
    {
        gStartTime.setTime();
        statError = sendDaemonInfo();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to Send Info to Daemons\n");
            return statError;
        }
        gEndTime.setTime();
        addPerfData("\tLaunchmon Send Time", (gEndTime - gStartTime).getDoubleTime());

        if (blocking == true)
        {
            statError = connectMrnetTree(blocking);
            if (statError != STAT_OK)
            {
                printMsg(statError, __FILE__, __LINE__, "Failed to connect MRNet tree\n");
                return statError;
            }
        }
    }

    return STAT_OK;
}


StatError_t STAT_FrontEnd::connectMrnetTree(bool blocking)
{
    static int sConnectTimeout = -1, sConnectAttempt = 0;
    char *connectTimeoutString;

    if (network_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "MRNet Network object is NULL.\n");
        return STAT_MRNET_ERROR;
    }

    if (sConnectTimeout == -1)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Connecting MRNet to daemons\n");
        gStartTime.setTime();
        connectTimeoutString = getenv("STAT_CONNECT_TIMEOUT");
        if (connectTimeoutString != NULL)
            sConnectTimeout = atoi(connectTimeoutString);
        else
            sConnectTimeout = 999999;
    }

    if (blocking == false)
    {
        sConnectAttempt = sConnectAttempt + 1;
        if (daemonsHaveExited())
        {
            printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
            return STAT_DAEMON_ERROR;
        }
        if (sConnectAttempt < sConnectTimeout * 100)
        {
            if (gNumCallbacks < nApplNodes_ * nDaemonsPerNode_)
            {
                usleep(10000);
                return STAT_PENDING_ACK;
            }
        }
    }
    else
    {
        for (sConnectAttempt = 0; sConnectAttempt < sConnectTimeout * 100; sConnectAttempt++)
        {
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                return STAT_DAEMON_ERROR;
            }
            if (gNumCallbacks == nApplNodes_ * nDaemonsPerNode_)
                break;
            usleep(10000);
        }
    }

    /* Make sure all expected BEs registered within timeout limit */
    if (sConnectAttempt >= sConnectTimeout * 100 || gNumCallbacks < nApplNodes_ * nDaemonsPerNode_)
    {
        printMsg(STAT_WARNING, __FILE__, __LINE__, "Connection timed out after %d/%d seconds with %d of %d Backends reporting.\n", sConnectAttempt/100, sConnectTimeout, gNumCallbacks, nApplNodes_);
        if (gNumCallbacks == 0)
            return STAT_DAEMON_ERROR;
        printMsg(STAT_WARNING, __FILE__, __LINE__, "Continuing with available subset.\n");
    }

    gEndTime.setTime();
    sConnectTimeout = -1;
    sConnectAttempt = 0;
    addPerfData("\tConnect to Daemons Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tDaemons connected\n");

    return STAT_OK;
}


StatError_t STAT_FrontEnd::setupConnectedMrnetTree()
{
    int filterId, intRet, tag;
    char fullTopologyFile[BUFSIZE];
    StatError_t statError;
    PacketPtr packet;
#ifdef STAT_FGFS
    int filterId2;
    bool boolRet;
    Stream *fgfsStream;
#endif

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting up STAT based on connected MRNet tree\n");

    /* Now that we're fully connected, determine the BE merge order */
    if (isStatBench_ == false)
    {
        statError = setRanksList();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to set ranks list\n");
            return statError;
        }
    }

    /* Dump the fully connected topology to a file */
    snprintf(fullTopologyFile, BUFSIZE, "%s/%s.fulltop", outDir_, filePrefix_);
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Outputting full MRNet topology file to %s\n", fullTopologyFile);
    leafInfo_.networkTopology->print_TopologyFile(fullTopologyFile);

    gStartTime.setTime();
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tConfiguring MRNet connection...\n");
    broadcastCommunicator_ = network_->get_BroadcastCommunicator();
    if (broadcastCommunicator_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to get MRNet broadcast communicator\n");
        return STAT_MRNET_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Loading STAT filter into MRNet\n");
    if (filterPath_ == NULL)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Filter path not set\n");
        return STAT_ARG_ERROR;
    }

    filterId = network_->load_FilterFunc(filterPath_, "filterInit");
    if (filterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure for path %s, function filterInit\n", filterPath_);
        return STAT_FILTERLOAD_ERROR;
    }

    broadcastStream_ = network_->new_Stream(broadcastCommunicator_, TFILTER_SUM, SFILTER_WAITFORALL, filterId);
    if (broadcastStream_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to setup MRNet broadcast stream\n");
        return STAT_MRNET_ERROR;
    }

    /* Send an initial message using the broadcast stream */
    if (broadcastStream_->send(PROT_SEND_BROADCAST_STREAM, "%uc %s %d", logging_, logOutDir_, mrnetOutputLevel_) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to send on broadcast stream\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to flush broadcast stream\n");
        return STAT_MRNET_ERROR;
    }
    intRet = broadcastStream_->recv(&tag, packet, true);
    if (intRet == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to receive broadcast stream acks\n");
        return STAT_MRNET_ERROR;
    }
    if (packet->unpack("%d", &intRet) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to unpack acknowledgement packet\n");
        return STAT_MRNET_ERROR;
    }
    if (intRet != 0)
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "%d daemons reported an error\n", intRet);
        return STAT_DAEMON_ERROR;
    }

    filterId = network_->load_FilterFunc(filterPath_, "statMerge");
    if (filterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure for path %s, function statMerge\n", filterPath_);
        return STAT_FILTERLOAD_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating MRNet stream with STAT filter\n");
    mergeStream_ = network_->new_Stream(broadcastCommunicator_, filterId, SFILTER_WAITFORALL);
    if (mergeStream_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to setup STAT merge stream\n");
        return STAT_MRNET_ERROR;
    }

    gEndTime.setTime();
    addPerfData("\tStream Creation and Filter Load Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tMRNet connection configured\n");
    isConnected_ = true;
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Tool daemons launched and connected!\n");

    statError = checkVersion();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to validate version number with the daemons.  The STAT FrontEnd is at version %d %d %d\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
        return statError;
    }

#ifdef STAT_FGFS
    gStartTime.setTime();

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting up FGFS\n");
    if (fgfsFilterPath_ == NULL)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "FGFS Filter path not set\n");
        return STAT_ARG_ERROR;
    }

    filterId = network_->load_FilterFunc(fgfsFilterPath_, FGFS_UP_FILTER_FN_NAME);
    if (filterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure for path %s, function %s\n", fgfsFilterPath_, FGFS_UP_FILTER_FN_NAME);
        return STAT_FILTERLOAD_ERROR;
    }

    filterId2 = network_->load_FilterFunc(fgfsFilterPath_, FGFS_DOWN_FILTER_FN_NAME);
    if (filterId2 == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure for path %s, function %s\n", fgfsFilterPath_, FGFS_DOWN_FILTER_FN_NAME);
        return STAT_FILTERLOAD_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating MRNet stream with the FGFS filter\n");
    fgfsStream = network_->new_Stream(broadcastCommunicator_, filterId, SFILTER_WAITFORALL, filterId2);
    if (fgfsStream == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to setup FGFS stream\n");
        return STAT_MRNET_ERROR;
    }

    /* Send an empty message using the FGFS stream */
    if (fgfsStream->send(PROT_SEND_FGFS_STREAM, "%auc", "x", 2) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to send on fgfs stream\n");
        return STAT_MRNET_ERROR;
    }
    if (fgfsStream->flush() == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to flush fgfs stream\n");
        return STAT_MRNET_ERROR;
    }

    boolRet = MRNetCommFabric::initialize((void *)network_, (void *)fgfsStream);
    if (boolRet == false)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to initialize FGFS CommFabric\n");
        return STAT_MRNET_ERROR;
    }
    fgfsCommFabric_ = new MRNetCommFabric();
    boolRet = AsyncGlobalFileStatus::initialize(fgfsCommFabric_);
    if (boolRet == false)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to initialize AsyncGlobalFileStatus\n");
        return STAT_MRNET_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "FGFS setup complete\n");
    gEndTime.setTime();
    addPerfData("\tFGFS Setup Time", (gEndTime - gStartTime).getDoubleTime());

    gStartTime.setTime();
    statError = sendFileRequestStream();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "sendFileRequestStream reported an error\n");
        return statError;
    }
    gEndTime.setTime();
    addPerfData("\tFile Broadcast Setup Time", (gEndTime - gStartTime).getDoubleTime());
#endif

    gTotalgEndTime.setTime();
    addPerfData("\tTotal MRNet Launch Time", (gTotalgEndTime - gTotalgStartTime).getDoubleTime());

    /* If we're STATBench, release the job launcher in case we need to debug */
    if (isStatBench_)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "STATBench releasing the job launcher\n");
        if (isLaunched_ == true)
        {
            detachFromLauncher("failed to detach from launcher... this probably means that the helper daemons exited normally\n");
            isLaunched_ = false;
        }
    }

    return STAT_OK;
}


#ifdef STAT_FGFS
StatError_t STAT_FrontEnd::sendFileRequestStream()
{
    int upstreamFileRequestFilterId, downstreamFileRequestFilterId;
    PacketPtr packet;
    StatError_t statError;

    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    upstreamFileRequestFilterId = network_->load_FilterFunc(filterPath_, "fileRequestUpStream");
    if (upstreamFileRequestFilterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure\n");
        return STAT_FILTERLOAD_ERROR;
    }

    downstreamFileRequestFilterId = network_->load_FilterFunc(filterPath_, "fileRequestDownStream");
    if (downstreamFileRequestFilterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure\n");
        return STAT_FILTERLOAD_ERROR;
    }

    fileRequestStream_ = network_->new_Stream(broadcastCommunicator_, upstreamFileRequestFilterId, SFILTER_DONTWAIT, downstreamFileRequestFilterId);
    if (fileRequestStream_ == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to create STAT file request stream\n");
        return STAT_MRNET_ERROR;
    }

    /* Broadcast a dummy message to BEs on the file request stream */
    if (fileRequestStream_->send(PROT_FILE_REQ, "%auc %s", "X", 2, "X") == -1 )
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send file request stream message\n");
        return STAT_MRNET_ERROR;
    }
    if (fileRequestStream_->flush() == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_FILE_REQ_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = NULL;
    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive file stream setup ack\n");
        return statError;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Received acknowledgements from all BEs\n");
    return STAT_OK;
}


StatError_t STAT_FrontEnd::waitForFileRequests(unsigned int &streamId, int &returnTag, PacketPtr &packetPtr, int &intRetVal, vector<Stream *> expectedStreams)
{
    char *receiveFileName = NULL;
    Stream *stream;
    StatError_t statError;

    while (1)
    {
        // first check for file request
        intRetVal = fileRequestStream_->recv(&returnTag, packetPtr, false);
        if (intRetVal == 0)
        {
            // no pending file request, check for other messages to forward
            vector<Stream *>::iterator iter;
            for (iter = expectedStreams.begin(); iter != expectedStreams.end(); iter++)
            {
                stream = *iter;
                intRetVal = stream->recv(&returnTag, packetPtr, false);
                if (intRetVal == 0)
                    continue;
                else if (intRetVal < 0)
                {
                    printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "stream::recv() failure for stream ID %d\n", stream->get_Id());
                    return STAT_MRNET_ERROR;
                }
                streamId = stream->get_Id();
                printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "waitForFileRequests returing tag %d, stream ID %d\n", returnTag, streamId);
                return STAT_OK;
            }
            return STAT_PENDING_ACK;
        }
        else if (intRetVal < 0)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "fileRequestStream_::recv() failure\n");
            return STAT_MRNET_ERROR;
        }

        // file request received
        if (packetPtr->unpack("%s", &receiveFileName) == -1)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "packetPtr::unpack() failure\n");
            return STAT_MRNET_ERROR;
        }

        statError = serveFileRequest(receiveFileName);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to serve file %s\n", receiveFileName);
            isPendingAck_ = false;
            return statError;
        }

        if (receiveFileName != NULL)
            free(receiveFileName);
    }
    return STAT_OK;
}

StatError_t STAT_FrontEnd::serveFileRequest(const char *receiveFileName)
{
    int tag, intRet;
    long signedFileSize;
    uint64_t fileSize;
    char *fileContents = NULL, errorMsg[BUFSIZE];
    FILE *file = NULL;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "received request for file %s\n", receiveFileName);

    file = fopen(receiveFileName, "r");
    if (file == NULL)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "%s: Failed to open file %s\n", strerror(errno), receiveFileName);
        snprintf(errorMsg, BUFSIZE, "STAT FE failed to open file %s", receiveFileName);
        fileSize = strlen(errorMsg) + 1;
        fileContents = strdup(errorMsg);
        if (fileContents == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: failed to malloc %lu bytes\n", strerror(errno), fileSize);
            return STAT_ALLOCATE_ERROR;
        }
        tag = PROT_LIB_REQ_ERR;
    }
    else
    {
        intRet = fseek(file, 0, SEEK_END);
        if (intRet == -1)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: failed to fseek file %s\n", strerror(errno), receiveFileName);
            fclose(file);
            return STAT_FILE_ERROR;
        }
        signedFileSize = ftell(file);
        if (signedFileSize < 0)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: ftell returned %ld for %s\n", strerror(errno), signedFileSize, receiveFileName);
            fclose(file);
            return STAT_ALLOCATE_ERROR;
        }
        fileSize = signedFileSize;
        intRet = fseek(file, 0, SEEK_SET);
        if (intRet == -1)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: failed to fseek file %s\n", strerror(errno), receiveFileName);
            fclose(file);
            return STAT_FILE_ERROR;
        }
        fileContents = (char *)malloc(fileSize * sizeof(char));
        if (fileContents == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: failed to malloc %lu bytes for contents\n", strerror(errno), fileSize);
            fclose(file);
            return STAT_ALLOCATE_ERROR;
        }
        intRet = fread(fileContents, fileSize, 1, file);
        if (intRet < 1)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: failed to fread %d, %d bytes for file %s, with return code %d\n", strerror(errno), fileSize, 1, receiveFileName, intRet);
            fclose(file);
            return STAT_ALLOCATE_ERROR;
        }
        intRet = ferror(file);
        if (intRet == -1)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: ferror returned %d on fread of %s\n", strerror(errno), intRet, receiveFileName);
            fclose(file);
            return STAT_FILE_ERROR;
        }
        fclose(file);
        tag = PROT_LIB_REQ_RESP;
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "sending contents of file %s, length %lu bytes\n", receiveFileName, fileSize);
    if (fileRequestStream_->send(tag, "%Ac %s", fileContents, fileSize, receiveFileName) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to send file contents\n");
        return STAT_MRNET_ERROR;
    }
    if (fileRequestStream_->flush() == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "failed to flush file contents\n");
        return STAT_MRNET_ERROR;
    }
    if (fileContents != NULL)
        free(fileContents);
    return STAT_OK;
}

StatError_t STAT_FrontEnd::checkFileRequest()
{
    int tag, intRetVal;
    char *receiveFileName = NULL;
    StatError_t statError;
    PacketPtr packetPtr;

    intRetVal = fileRequestStream_->recv(&tag, packetPtr, false);
    if (intRetVal == 0)
        return STAT_PENDING_ACK;
    else if (intRetVal < 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "network::recv() failure\n");
        return STAT_MRNET_ERROR;
    }

    if (tag != PROT_LIB_REQ)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "checkFileRequest received unexpected tag %d\n", tag);
        return STAT_OK;
    }
    if (packetPtr->unpack("%s", &receiveFileName) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "packetPtr::unpack() failure\n");
        return STAT_MRNET_ERROR;
    }

    statError = serveFileRequest(receiveFileName);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to serve file %s\n", receiveFileName);
        isPendingAck_ = false;
        return statError;
    }

    if (receiveFileName != NULL)
        free(receiveFileName);

    return STAT_OK;
}

#endif /* STAT_FGFS */


void STAT_FrontEnd::setNodeListFile(char *nodeListFile)
{
   nodeListFile_ = nodeListFile;
}


char *STAT_FrontEnd::getLastErrorMessage()
{
    return lastErrorMessage_;
}


const char *STAT_FrontEnd::getInstallPrefix()
{
    char *envValue;
    envValue = getenv("STAT_PREFIX");
    if (envValue != NULL)
        return envValue;
    return STAT_PREFIX;
}


void STAT_FrontEnd::getVersion(int *version)
{
    version[0] = STAT_MAJOR_VERSION;
    version[1] = STAT_MINOR_VERSION;
    version[2] = STAT_REVISION_VERSION;
}



StatError_t STAT_FrontEnd::startLog(unsigned int logType, char *logOutDir, bool withPid)
{
    int intRet;
    char fileName[BUFSIZE];

    logging_ = logType;
    snprintf(logOutDir_, BUFSIZE, "%s", logOutDir);

    intRet = mkdir(logOutDir_, S_IRUSR | S_IWUSR | S_IXUSR);
    if (intRet == -1 && errno != EEXIST)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: mkdir failed to create log directory %s\n", strerror(errno), logOutDir);
        return STAT_FILE_ERROR;
    }

    /* If we're logging the FE, open the log file */
    if (logging_ & STAT_LOG_FE)
    {
        if (withPid == true)
        {
            pid_t myPid = getpid();
            snprintf(fileName, BUFSIZE, "%s/%s.STATD.%d.log", logOutDir_, hostname_, myPid);
        }
        else
            snprintf(fileName, BUFSIZE, "%s/%s.STAT.log", logOutDir_, hostname_);
        gStatOutFp = fopen(fileName, "w");
        if (gStatOutFp == NULL)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to open FE log file %s\n", strerror(errno), fileName);
            return STAT_FILE_ERROR;
        }
        if (logging_ & STAT_LOG_MRN)
            mrn_printf_init(gStatOutFp);
    }

    return STAT_OK;
}


StatError_t STAT_FrontEnd::receiveAck(bool blocking)
{
    //TODO: break this into receiveAck and processAck?
    int tag, intRet;
    unsigned int streamId = 0;
    StatError_t statError;
    PacketPtr packet;
    vector<Stream *> expectedStreams; //TODO: make this part of the FE class?

    if (isPendingAck_ == false)
        return STAT_OK;

    /* If we're pending stack traces, call the receive function */
    if (pendingAckTag_ == PROT_SEND_LAST_TRACE_RESP || pendingAckTag_ == PROT_SEND_TRACES_RESP)
    {
        statError = receiveStackTraces(blocking);
        if (statError != STAT_OK && statError != STAT_PENDING_ACK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive stack traces\n");
            isPendingAck_ = false;
        }
        return statError;
    }

    /* Receive an acknowledgement packet that all daemons have completed */
    expectedStreams.push_back(broadcastStream_);
    expectedStreams.push_back(mergeStream_);
    do
    {
        if (hasFatalError_ == true)
        {
            printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "A Fatal Error has been detected\n");
            isPendingAck_ = false;
            return STAT_DAEMON_ERROR;
        }
#ifdef STAT_FGFS
        statError = waitForFileRequests(streamId, tag, packet, intRet, expectedStreams);
        if (statError == STAT_PENDING_ACK)
        {
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                isPendingAck_ = false;
                return STAT_DAEMON_ERROR;
            }
            if (blocking == true)
                usleep(1000);
            continue;
        }
        else if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Unable to process file requests\n");
            return statError;
        }
        if (streamId == broadcastStream_->get_Id())
        {
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Received message on broadcast stream %d\n", broadcastStream_->get_Id());
#else
        intRet = broadcastStream_->recv(&tag, packet, false);
#endif
        if (intRet == 0)
        {
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                isPendingAck_ = false;
                return STAT_DAEMON_ERROR;
            }
            if (blocking == true)
                usleep(1000);
        }
#ifdef STAT_FGFS
        }
#endif
    } while (intRet == 0 && blocking == true);

    /* Check for errors */
    if (intRet == 0 && blocking == false)
        return STAT_PENDING_ACK;
    else if (intRet == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to receive acks\n");
        isPendingAck_ = false;
        return STAT_MRNET_ERROR;
    }
    if (tag != pendingAckTag_)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Received unexpected tag %d, expecting %d\n", tag, pendingAckTag_);
        isPendingAck_ = false;
        return STAT_MRNET_ERROR;
    }

    /* Unpack the ack and check for errors */
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Successfully received ack with tag %d\n", tag);
    if (packet->unpack("%d", &intRet) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to unpack acknowledgement packet\n");
        isPendingAck_ = false;
        return STAT_MRNET_ERROR;
    }
    if (intRet != 0)
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "%d daemons reported an error\n", intRet);
        isPendingAck_ = false;
        return STAT_DAEMON_ERROR;
    }

    isPendingAck_ = false;
    if (pendingAckCb_ != NULL)
        (*this.*pendingAckCb_)();
    return STAT_OK;
}


StatError_t STAT_FrontEnd::setDaemonNodes()
{
    set<MRN::NetworkTopology::Node *> nodes;
    set<MRN::NetworkTopology::Node *>::iterator nodeIter;
    string prettyHost;
    int intRet;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating daemon node list: \n");
    leafInfo_.networkTopology->get_BackEndNodes(nodes);
    if (nodes.size() <= 0)
    {
#ifdef DYSECTAPI
        daemonsKilled_ = true;
        return STAT_APPLICATION_EXITED;
#else
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "No daemons are connected\n");
        return STAT_DAEMON_ERROR;
#endif
    }
    leafInfo_.daemons.clear();
    for (nodeIter = nodes.begin(); nodeIter != nodes.end(); nodeIter++)
    {
        intRet = XPlat::NetUtils::GetHostName((*nodeIter)->get_HostName(), prettyHost);
        if (intRet != 0)
            prettyHost = (*nodeIter)->get_HostName();

#ifdef STAT_ALIAS_SUFFIX
        // TODO: this is an ugly way of dealing with aliased hostnames
        size_t pub;
        pub = prettyHost.find(STAT_ALIAS_SUFFIX);
        if (pub != string::npos)
            prettyHost = prettyHost.substr(0, pub);
#endif

        leafInfo_.daemons.insert(prettyHost);
        printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "%s=%s, ", (*nodeIter)->get_HostName().c_str(), prettyHost.c_str());
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "\n");
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Daemon node list created\n");

    return STAT_OK;
}

StatError_t STAT_FrontEnd::createDaemonRankMap()
{
    unsigned int i, j, mrnetRank, rank;
    int k, l;
    const char *currentNode;
    IntList_t *daemonRanks, *newDaemonRanks;
    map<string, vector<int> > tempMap;
    map<string, vector<int> >::iterator tempMapIter;
    map<string, int> hostToMrnetRankMap;
    set<NetworkTopology::Node *> backEndNodes;
    set<NetworkTopology::Node *>::iterator backEndNodesIter;
    NetworkTopology::Node *node;
    string::size_type dotPos;
    string tempString;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating daemon rank map\n");

    /* First, create a map containing all daemons, with the MPI ranks that each daemon debugs */
    for (i = 0; i < nApplProcs_; i++)
    {
        currentNode = getHostnameForProc(i);
        tempMap[currentNode].push_back(getMpiRankForProc(i));
    }

    /* Next populate the hostToMrnetRankMap */
    leafInfo_.networkTopology->get_BackEndNodes(backEndNodes);
    if (isStatBench_ == true)
    {
        for (backEndNodesIter = backEndNodes.begin(), tempMapIter = tempMap.begin(); backEndNodesIter != backEndNodes.end() && tempMapIter != tempMap.end(); backEndNodesIter++, tempMapIter++)
        {
            node = *backEndNodesIter;
            hostToMrnetRankMap[tempMapIter->first] = node->get_Rank();
        }
    }
    else
    {
        for (backEndNodesIter = backEndNodes.begin(); backEndNodesIter != backEndNodes.end(); backEndNodesIter++)
        {
            node = *backEndNodesIter;
            tempString = node->get_HostName();
            dotPos = tempString.find_first_of(".");
            if (dotPos != string::npos)
                tempString = tempString.substr(0, dotPos);

#ifdef STAT_ALIAS_SUFFIX
            // TODO: this is an ugly way of dealing with aliased hostnames
            size_t pub;
            pub = tempString.find(STAT_ALIAS_SUFFIX);
            if (pub != string::npos)
                tempString = tempString.substr(0, pub);
#endif

            /* TODO: this won't work if we move to multiple daemons per node */
            hostToMrnetRankMap[tempString.c_str()] = node->get_Rank();
        }
    }

    /* Next sort each daemon's rank list and store it in the IntList_t ranks map */
    for (tempMapIter = tempMap.begin(), i = 0; tempMapIter != tempMap.end(); tempMapIter++, i++)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Daemon %s, %d ranks:", tempMapIter->first.c_str(), (tempMapIter->second).size());
        sort((tempMapIter->second).begin(), (tempMapIter->second).end());
        daemonRanks = (IntList_t *)malloc(sizeof(IntList_t));
        if (daemonRanks == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for daemonRanks\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
        daemonRanks->count = (tempMapIter->second).size();
        daemonRanks->list = (int *)malloc(daemonRanks->count * sizeof(int));
        if (daemonRanks->list == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for daemonRanks->list\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
        for (j = 0; j < (tempMapIter->second).size(); j++)
        {
            printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "%d, ", (tempMapIter->second)[j]);
            daemonRanks->list[j] = (tempMapIter->second)[j];
        }
        if (hostToMrnetRankMap.size() == 0) /* MRNet BEs not connected */
        {
            if (nDaemonsPerNode_ > 1)
            {
                l = -1;
                for (j = 0; j < nDaemonsPerNode_; j++)
                {
                    mrnetRank = topologySize_ + i + nApplNodes_ * j;
                    newDaemonRanks = (IntList_t *)malloc(sizeof(IntList_t));
                    if (newDaemonRanks == NULL)
                    {
                        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for newDaemonRanks\n", strerror(errno));
                        return STAT_ALLOCATE_ERROR;
                    }
                    newDaemonRanks->count = daemonRanks->count / nDaemonsPerNode_;
                    if (daemonRanks->count % nDaemonsPerNode_ > j)
                        newDaemonRanks->count = newDaemonRanks->count + 1;
                    newDaemonRanks->list = (int *)malloc(newDaemonRanks->count * sizeof(int));
                    if (newDaemonRanks->list == NULL)
                    {
                        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for newDaemonRanks->list\n", strerror(errno));
                        return STAT_ALLOCATE_ERROR;
                    }
                    for (k = 0; k < newDaemonRanks->count; k++)
                    {
                        l++;
                        rank = daemonRanks->list[l];
                        printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "%d, ", rank);
                        newDaemonRanks->list[k] = rank;
                    }
                    printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "; ");
                    mrnetRankToMpiRanksMap_[mrnetRank] = newDaemonRanks;
                }
            }
            else
                mrnetRankToMpiRanksMap_[topologySize_ + i] = daemonRanks;
        }
        else /* MRNet BEs connected */
        {
            if (hostToMrnetRankMap.find(tempMapIter->first) != hostToMrnetRankMap.end())
                mrnetRankToMpiRanksMap_[hostToMrnetRankMap[tempMapIter->first]] = daemonRanks;
            else
                for (k = 0; k < daemonRanks->count; k++)
                    missingRanks_.insert(daemonRanks->list[k]);
        }
        printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "\n");
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Daemon rank map created\n");
    return STAT_OK;
}

StatError_t STAT_FrontEnd::setCommNodeList(const char *nodeList, bool checkAccess = true)
{
    char numString[BUFSIZE], nodeName[BUFSIZE], *nodeRange;
    unsigned int num1 = 0, num2, startPos, endPos, i, j;
    bool isRange = false;
    string baseNodeName, nodes, list;
    string::size_type openBracketPos, closeBracketPos, commaPos, finalPos;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating communication node list\n");

    if (nodeList == NULL)
    {
        /* There may be enough resources where the STAT FE is being run so
           we'll return OK for now... we will check for sufficient resources
           when we create the topology file anyway */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "No nodes specified for communication processes\n");
        return STAT_OK;
    }

    list = nodeList;
    while (1)
    {
        openBracketPos = list.find_first_of("[");
        closeBracketPos = list.find_first_of("]");
        commaPos = list.find_first_of(",");

        if (openBracketPos == string::npos && commaPos == string::npos)
            finalPos = list.length(); /* Last one, just a single node */
        else if (commaPos < openBracketPos)
            finalPos = commaPos; /* just a single node */
        else
            finalPos = closeBracketPos + 1;
        nodes = list.substr(0, finalPos);
        openBracketPos = nodes.find_first_of("[");
        closeBracketPos = nodes.find_first_of("]");

        if (openBracketPos == string::npos && closeBracketPos == string::npos)
        {
            /* This is a single node */
            strncpy(nodeName, nodes.c_str(), BUFSIZE);
            if (strcmp(nodeName, "") != 0)
            {
                if (checkAccess == true)
                {
                    if (checkNodeAccess(nodeName))
                    {
                        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s added to communication node list\n", nodeName);
                        communicationNodeSet_.insert(nodeName);
                    }
                    else
                        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s not accessable\n", nodeName);
                }
                else
                    communicationNodeSet_.insert(nodeName);
            }
        }
        else
        {
            /* This is a list of nodes */
            /* Parse the node list string string e.g.: xxx[0-15,12,23,26-35] */
            nodeRange = strdup(nodes.substr(openBracketPos + 1, closeBracketPos - (openBracketPos + 1)).c_str());
            if (nodeRange == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed on call to set node range with strdup()\n", strerror(errno));
                return STAT_ARG_ERROR;
            }

            /* Get the machine name */
            baseNodeName = nodes.substr(0, openBracketPos);

            /* Decode the node list string  */
            for (i = 0; i < strlen(nodeRange); i++)
            {
                if (nodeRange[i] == ',')
                    continue;
                else if (isdigit(nodeRange[i]))
                {
                    startPos = i;
                    while (isdigit(nodeRange[i]))
                    {
                        i++;
                        if (i >= strlen(nodeRange))
                            break;
                    }
                    endPos = i - 1;
                }
                else
                {
                    printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Invalid node list %s\nEncountered unexpected character '%c'\n", nodeList, nodeRange[i]);
                    free(nodeRange);
                    return STAT_ARG_ERROR;
                }

                memcpy(numString, nodeRange + startPos, endPos - startPos + 1);
                numString[endPos - startPos + 1] = '\0';

                if (isRange)
                {
                    isRange = false;
                    num2 = atoi(numString);
                    for (j = num1; j <= num2; j++)
                    {
                        snprintf(nodeName, BUFSIZE, "%s%u", baseNodeName.c_str(), j);
                        if (checkAccess == true)
                        {
                            if (checkNodeAccess(nodeName))
                            {
                                printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s added to communication node list\n", nodeName);
                                communicationNodeSet_.insert(nodeName);
                            }
                            else
                                printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s not accessable\n", nodeName);
                        }
                        else
                            communicationNodeSet_.insert(nodeName);
                    }
                }
                else
                {
                    num1 = atoi(numString);
                    if (i < strlen(nodeRange))
                    {
                        if (nodeRange[i] == '-')
                        {
                            isRange = true;
                            continue;
                        }
                    }
                    snprintf(nodeName, BUFSIZE, "%s%u", baseNodeName.c_str(), num1);
                    if (checkAccess == true)
                    {
                        if (checkNodeAccess(nodeName))
                        {
                            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s added to communication node list\n", nodeName);
                            communicationNodeSet_.insert(nodeName);
                        }
                        else
                            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Node %s not accessable\n", nodeName);
                        }
                    else
                        communicationNodeSet_.insert(nodeName);
                } /* else branch of if (isRange) */
                i = i - 1;
            } /* for (i = 0; i < strlen(nodeRange); i++) */

            if (nodeRange != NULL)
                free(nodeRange);
        } /* else branch of if (openBracketPos == ...) */
        if (finalPos >= list.length())
            break;
        else
            list = list.substr(finalPos + 1, list.length() - 1);
    } /* while (1) */

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Communication node list created with %d nodes\n", communicationNodeSet_.size());

    return STAT_OK;
}


StatError_t STAT_FrontEnd::createOutputDir()
{
    int intRet, fileNameCount;
    char cwd[BUFSIZE], resultsDirectory[BUFSIZE], *homeDirectory;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating output directory\n");

    /* Look for the stat_results directory and create if it doesn't exist */
    if (getcwd(cwd, BUFSIZE) == NULL)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "%s: getcwd failed\n", strerror(errno));
        return STAT_SYSTEM_ERROR;
    }
    snprintf(resultsDirectory, BUFSIZE, "%s/stat_results", cwd);
    intRet = mkdir(resultsDirectory, S_IRUSR | S_IWUSR | S_IXUSR);
    if (intRet == -1 && errno != EEXIST)
    {
        /* User may not have write privileges in CWD, try the user's home directory */
        homeDirectory = getenv("HOME");
        if (homeDirectory == NULL)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "Failed to create output directory in current working directory and failed to get $HOME as an alternative.\n");
            return STAT_FILE_ERROR;
        }
        snprintf(resultsDirectory, BUFSIZE, "%s/stat_results", homeDirectory);
        intRet = mkdir(resultsDirectory, S_IRUSR | S_IWUSR | S_IXUSR);
        if (intRet == -1 && errno != EEXIST)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: Failed to create output directory in current working directory %s and failed to create output directory in $HOME %s as an alternative.\n", strerror(errno), cwd, homeDirectory);
            return STAT_FILE_ERROR;
        }
    }

    /* Create run-specific results directory with a unique name */
    for (fileNameCount = 0; fileNameCount < STAT_MAX_FILENAME_ID; fileNameCount++)
    {
        if (jobId_ == NULL)
            snprintf(outDir_, BUFSIZE, "%s/%s.%04d", resultsDirectory, applExe_, fileNameCount);
        else
            snprintf(outDir_, BUFSIZE, "%s/%s.%s.%04d", resultsDirectory, applExe_, jobId_, fileNameCount);
        intRet = mkdir(outDir_, S_IRUSR | S_IWUSR | S_IXUSR);
        if (intRet == 0)
            break;
        else if (intRet == -1 && errno != EEXIST)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: mkdir failed to create run specific directory %s\n", strerror(errno), outDir_);
            return STAT_FILE_ERROR;
        }
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Output directory %s created\n", outDir_);

    /* Generate the file prefix for output files */
    if (jobId_ == NULL)
        snprintf(filePrefix_, BUFSIZE, "%s.%04d", applExe_, fileNameCount);
    else
        snprintf(filePrefix_, BUFSIZE, "%s.%s.%04d", applExe_, jobId_, fileNameCount);
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generated file prefix: %s\n", outDir_);

    return STAT_OK;
}


StatError_t STAT_FrontEnd::createTopology(char *topologyFileName, StatTopology_t topologyType, char *topologySpecification, char *nodeList, StatCpPolicy_t cpPolicy)
{
    unsigned int i, j, counter, layer, parentIter = 0, childIter, fanout = 0, parentCount = 1, procsNeeded = 0, desiredDepth = 0, desiredMaxFanout = 0, size;
    FILE *file;
    char tmp[BUFSIZE];
    char *envValue;
    string topology, appNodes;
    vector<string> treeList;
    set<string>::iterator communicationNodeSetIter;
    multiset<string>::iterator applicationNodeMultiSetIter;
    string topoIter, current;
    string::size_type dashPos, lastPos;
    StatError_t statError;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__,
             "Creating MRNet topology file for %d processes on %d nodes\n",
             nApplProcs_, nApplNodes_);

    envValue = getenv("STAT_CHECK_NODE_ACCESS");
    if (envValue != NULL)
        checkNodeAccess_ = true;

    /* Set parameters based on requested topology */
    if (topologyType == STAT_TOPOLOGY_DEPTH)
    {
        desiredMaxFanout = STAT_MAX_FANOUT;
        desiredDepth = atoi(topologySpecification);
    }
    else if (topologyType == STAT_TOPOLOGY_FANOUT)
        desiredMaxFanout = atoi(topologySpecification);
    else if (topologyType == STAT_TOPOLOGY_USER)
        topology = topologySpecification;
    else
        desiredMaxFanout = STAT_MAX_FANOUT;

    /* Set the communication node list if we're not using a flat 1 to N tree */
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting communication node list\n");
    if ((desiredMaxFanout < nApplProcs_ && desiredMaxFanout > 0) || topology != "" || desiredDepth != 0)
    {
        /* Add application nodes to list if requested */
        if (cpPolicy == STAT_CP_SHAREAPPNODES)
        {
            appNodes = "";
            for (applicationNodeMultiSetIter = applicationNodeMultiSet_.begin(); applicationNodeMultiSetIter != applicationNodeMultiSet_.end(); applicationNodeMultiSetIter++)
                appNodes += *applicationNodeMultiSetIter + ",";
            appNodes = appNodes.substr(0, appNodes.size() - 1); /* to remove last comma */
            statError = setCommNodeList(appNodes.c_str(), false);
            if (statError != STAT_OK)
            {
                printMsg(statError, __FILE__, __LINE__, "Failed to set the global node list\n");
                return statError;
            }
        }
        if (communicationNodeSet_.size() == 0)
        {
            if (nodeList == NULL)
            {
                statError = setNodeListFromConfigFile(&nodeList);
                if (statError != STAT_OK)
                    printMsg(statError, __FILE__, __LINE__, "Failed to get node list from config file\n");
            }
            else
            {
                if (strcmp(nodeList, "") == 0)
                {
                    statError = setNodeListFromConfigFile(&nodeList);
                    if (statError != STAT_OK)
                        printMsg(statError, __FILE__, __LINE__, "Failed to get node list from config file\n");
                }
            }

            statError = setCommNodeList(nodeList, checkNodeAccess_);
            if (statError != STAT_OK)
            {
                printMsg(statError, __FILE__, __LINE__, "Failed to set the global node list\n");
                return statError;
            }
        }

        if (cpPolicy == STAT_CP_EXCLUSIVE)
        {
            set<string> temp1, temp2;
            set_difference(communicationNodeSet_.begin(), communicationNodeSet_.end(), applicationNodeMultiSet_.begin(), applicationNodeMultiSet_.end(), inserter(temp1, temp1.end()));
            temp2.insert(hostname_);
            communicationNodeSet_.clear();
            set_difference(temp1.begin(), temp1.end(), temp2.begin(), temp2.end(), inserter(communicationNodeSet_, communicationNodeSet_.end()));
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Exclusive communication node list created with %d nodes\n", communicationNodeSet_.size());
        }
    }

    /* Set the requested topology and check if there are enough CPs specified */
    if (topology == "")
    {
        /* Determine the depth and fanout */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting topology depth %d fanout %d node count %d\n", desiredDepth, desiredMaxFanout, nApplNodes_);

        if (desiredDepth == 0)
        {
            /* Find the desired depth based on the fanout and number of app nodes */
            for (desiredDepth = 1; desiredDepth < 1024; desiredDepth++)
            {
                fanout = (int)ceil(pow((float)nApplNodes_, (float)1.0 / (float)desiredDepth));
                if (fanout <= desiredMaxFanout)
                    break;
            }
        }
        else
            fanout = (int)ceil(pow((float)nApplNodes_, (float)1.0 / (float)desiredDepth));

        /* Determine the number of processes needed */
        procsNeeded = 0;
        for (i = 1; i < desiredDepth; i++)
        {
            if (i == 1)
            {
                snprintf(tmp, BUFSIZE, "%d", fanout);
                topology = tmp;
            }
            else
            {
                snprintf(tmp, BUFSIZE, "-%d", (int)ceil(pow((float)fanout, (float)i)));
                topology += tmp;
            }
            procsNeeded += (int)ceil(pow((float)fanout, (float)i));
        }
        if (procsNeeded <= communicationNodeSet_.size() * procsPerNode_)
        {
            /* We have enough CPs, so we can have our desired depth */
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Requested topology set with depth %d, fanout %d\n", desiredDepth, fanout);
        }
        else
        {
            /* There aren't enough CPs, so make a 2-deep tree with as many CPs as we have */
            size = communicationNodeSet_.size() * procsPerNode_;
            if (size == 0)
                printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Not enough processes for specified topology.  %d processes needed for depth of %d and fanout of %d.  Reverting to flat topology\n", procsNeeded, desiredDepth, fanout);
            else
            {
                if (size > (unsigned int)sqrt(nApplNodes_))
                    size = (unsigned int)sqrt(nApplNodes_);
                if (topologyType == STAT_TOPOLOGY_AUTO)
                    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Not enough processes for automatic topology.  %d processes needed for depth of %d and fanout of %d.  Reverting to tree with one layer of %d communication processes\n", procsNeeded, desiredDepth, fanout, communicationNodeSet_.size() * procsPerNode_);
                else
                    printMsg(STAT_WARNING, __FILE__, __LINE__, "Not enough processes specified for the requested topology depth %d, fanout %d: %d processes needed, %d processes specified.  Reverting to tree with one layer of %d communication processes.  Next time, please specify more resources with --nodes and --procs or request the use of application nodes with --appnodes.\n", desiredDepth, fanout, procsNeeded, communicationNodeSet_.size() * procsPerNode_, size);
            }
            snprintf(tmp, BUFSIZE, "%d", size);
            topology = tmp;
        }
    }
    else
    {
        procsNeeded = 0;
        topoIter = topology.c_str();
        while (1)
        {
            dashPos = topoIter.find_first_of("-");
            if (dashPos == string::npos)
                lastPos = topoIter.length();
            else
                lastPos = dashPos;
            current = topoIter.substr(0, lastPos);
            layer = atoi(current.c_str());
            procsNeeded += layer;
            if (lastPos >= topoIter.length())
                break;
            topoIter = topoIter.substr(lastPos + 1);
        }
        size = communicationNodeSet_.size() * procsPerNode_;
        if (procsNeeded > size)
        {
            if (size > (unsigned int)sqrt(nApplNodes_))
                size = (unsigned int)sqrt(nApplNodes_);
            if (size > 0)
                printMsg(STAT_WARNING, __FILE__, __LINE__, "Not enough processes specified for the requested topology %s: %d processes needed, %d processes specified.  Reverting to tree with one layer of %d communication processes.  Next time, please specify more resources with --nodes and --procs.\n", topology.c_str(), procsNeeded, communicationNodeSet_.size() * procsPerNode_, size);
            else
                printMsg(STAT_WARNING, __FILE__, __LINE__, "Not enough processes specified for the requested topology %s: %d processes needed, %d processes specified.  Reverting to flat topology.  Next time, please specify more resources with --nodes and --procs.\n", topology.c_str(), procsNeeded, communicationNodeSet_.size() * procsPerNode_);
            snprintf(tmp, BUFSIZE, "%d", size);
            topology = tmp;
        }
    }

    /* Check if tool FE hostname is in application list and the communication
       node list, then we will later add it to the comm nodes */

    /* Add the FE to the root of the tree */
#ifdef BGL
    /* On BlueGene systems we need the network interface with the IO nodes */
    snprintf(tmp, BUFSIZE, "%s-io", hostname_);
    snprintf(hostname_, BUFSIZE, "%s", tmp);
#endif
    envValue = getenv("STAT_FE_HOSTNAME");
    if (envValue != NULL)
        snprintf(hostname_, BUFSIZE, "%s", envValue);
    snprintf(tmp, BUFSIZE, "%s:0", hostname_);
    treeList.push_back(tmp);

    if (procsPerNode_ <= 0)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "At least one process must be allowed per node... %d specified\n", procsPerNode_);
        return STAT_ARG_ERROR;
    }

    /* Add the nodes and IDs to the list of hosts */
    for (i = 1, counter = 0; i < procsPerNode_; i++)
    {
        for (communicationNodeSetIter = communicationNodeSet_.begin(); communicationNodeSetIter != communicationNodeSet_.end(); communicationNodeSetIter++, counter++)
        {
            if ((*communicationNodeSetIter) == hostname_)
                snprintf(tmp, BUFSIZE, "%s:%d", (*communicationNodeSetIter).c_str(), i + 1);
            else
                snprintf(tmp, BUFSIZE, "%s:%d", (*communicationNodeSetIter).c_str(), i);
            treeList.push_back(tmp);
        }
    }

    snprintf(topologyFileName, BUFSIZE, "%s/%s.top", outDir_, filePrefix_);
    file = fopen(topologyFileName, "w");
    if (file == NULL)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to create topology file %s\n", strerror(errno), topologyFileName);
        return STAT_FILE_ERROR;
    }

    /* Initialized vector iterators */
    if (topology == "") /* Flat topology */
    {
        if (applicationOption_ != STAT_SERIAL_ATTACH && applicationOption_ != STAT_SERIAL_GDB_ATTACH)
            fprintf(file, "%s;\n", treeList[0].c_str());
    }
    else if (topology == "0" && applicationOption_ != STAT_SERIAL_ATTACH && applicationOption_ != STAT_SERIAL_GDB_ATTACH) /* Flat topology */
        fprintf(file, "%s;\n", treeList[0].c_str());
    else
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating specified topology\n");
        topoIter = topology.c_str();
        parentIter = 0;
        parentCount = 1;
        childIter = 1;

        /* Parse the specification and create the topology */
        while (true)
        {
            dashPos = topoIter.find_first_of("-");
            if (dashPos == string::npos)
                lastPos = topoIter.length();
            else
                lastPos = dashPos;
            current = topoIter.substr(0, lastPos);
            layer = atoi(current.c_str());

            /* Loop around the parent's for this layer */
            for (i = 0; i < parentCount; i++)
            {
                if (parentIter >= treeList.size())
                {
                    printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Not enough resources for specified topology.  Please specify more resources with --nodes and --procs.\n");
                    fclose(file);
                    return STAT_ARG_ERROR;
                }
                fprintf(file, "%s =>", treeList[parentIter].c_str());

                /* Add the children for this layer */
                for (j = 0; j < (layer / parentCount) + (layer % parentCount > i ? 1 : 0); j++)
                {
                    if (childIter >= treeList.size())
                    {
                        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Not enough resources for specified topology.  Please specify more resources with --nodes and --procs.\n");
                        fclose(file);
                        return STAT_ARG_ERROR;
                    }
                    fprintf(file, "\n\t%s", treeList[childIter].c_str());
                    childIter++;
                }
                fprintf(file, ";\n\n");
                parentIter++;
            }

            parentCount = layer;
            if (lastPos >= topoIter.length())
                break;
            topoIter = topoIter.substr(lastPos + 1);
        }
    }

    if (applicationOption_ == STAT_SERIAL_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
    {
        applicationNodeMultiSetIter = applicationNodeMultiSet_.begin();
        for (i = 0; i < parentCount; i++)
        {
            fprintf(file, "%s =>", treeList[parentIter].c_str());

            /* Add the children for this layer */
            for (j = 0; j < (applicationNodeMultiSet_.size() / parentCount) + (applicationNodeMultiSet_.size() % parentCount > i ? 1 : 0); j++)
            {
                fprintf(file, "\n\t%s:%d", (*applicationNodeMultiSetIter).c_str(), procsPerNode_ + 1);
                applicationNodeMultiSetIter++;
            }
            fprintf(file, ";\n\n");
            parentIter++;
        }
    }

    fclose(file);
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "MRNet topology file created\n");
    return STAT_OK;
}

StatError_t STAT_FrontEnd::checkVersion()
{
    int filterId, tag, daemonCount, filterCount, intRet, major, minor, revision;
    Stream *stream;
    PacketPtr packet;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Comparing STAT FE version %d.%d.%d with the daemons\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);

    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting up version check filter\n");
    filterId = network_->load_FilterFunc(filterPath_, "STAT_checkVersion");
    if (filterId == -1)
    {
        printMsg(STAT_FILTERLOAD_ERROR, __FILE__, __LINE__, "load_FilterFunc() failure\n");
        return STAT_FILTERLOAD_ERROR;
    }

    stream = network_->new_Stream(broadcastCommunicator_, filterId, SFILTER_WAITFORALL);
    if (stream == NULL)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to get STAT version check stream\n");
        return STAT_MRNET_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Sending version message to daemons\n");
    if (stream->send(PROT_CHECK_VERSION, "%d %d %d", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send version check message\n");
        return STAT_MRNET_ERROR;
    }
    if (stream->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Receiving version check acknowledgements\n");
    do
    {
        intRet = stream->recv(&tag, packet, false);
        if (intRet == 0)
        {
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                return STAT_DAEMON_ERROR;
            }
            usleep(1000);
        }
    } while (intRet == 0);
    if (stream != NULL)
    {
        delete stream;
        stream = NULL;
    }
    if (intRet == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to receive version message acks\n");
        return STAT_MRNET_ERROR;
    }
    if (tag != PROT_CHECK_VERSION_RESP)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Unexpected tag %d, expecting PROT_CHECK_VERSION_RESP = %d\n", tag, PROT_CHECK_VERSION_RESP);
        return STAT_MRNET_ERROR;
    }

    if (packet->unpack("%d %d %d %d %d", &major, &minor, &revision, &daemonCount, &filterCount) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to unpack acknowledgement packet\n");
        return STAT_MRNET_ERROR;
    }

    if (filterCount != 0 || daemonCount != 0)
    {
        if (filterCount != 0)
            printMsg(STAT_VERSION_ERROR, __FILE__, __LINE__, "%d filters reported a version mismatch\n", filterCount);
        if (daemonCount != 0)
            printMsg(STAT_VERSION_ERROR, __FILE__, __LINE__, "%d daemons reported a version mismatch\n", daemonCount);
        return STAT_VERSION_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "All daemon versions are compatible\n");

    return STAT_OK;
}


void checkPendingActions(STAT_FrontEnd *statFE)
{
#ifdef STAT_FGFS
    StatError_t statError;
    statError = statFE->checkFileRequest();
    if (statError != STAT_OK && statError != STAT_PENDING_ACK)
      statFE->printMsg(statError, __FILE__, __LINE__, "Unable to process file requests\n");
#endif
}

StatError_t STAT_FrontEnd::attachApplication(bool blocking)
{
    StatError_t statError;
    PacketPtr packet;

    if (isAttached_ == true)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "STAT already attached to the application... ignoring request to attach\n");
        return STAT_OK;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, attach canceled\n");
        return statError;
    }

    gStartTime.setTime();
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Attaching to application...\n");

    if (broadcastStream_->send(PROT_ATTACH_APPLICATION, "%s %s", outDir_, filePrefix_) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send attach request\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_ATTACH_APPLICATION_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postAttachApplication;
    if (blocking == true)
    {
        statError = receiveAck(true);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive attach ack\n");
            return statError;
        }
    }
    return STAT_OK;
}

StatError_t STAT_FrontEnd::postAttachApplication()
{
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Attached!\n");
    isAttached_ = true;
    isRunning_ = false;
    gEndTime.setTime();
    addPerfData("Attach Time", (gEndTime - gStartTime).getDoubleTime());
    return STAT_OK;
}


StatError_t STAT_FrontEnd::pause(bool blocking)
{
    StatError_t statError;
    PacketPtr packet;

    if (isRunning_ == false)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "Application already paused... ignoring request to pause\n");
        return STAT_OK;
    }
    if (isAttached_ == false)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "STAT not attached to the application\n");
        return STAT_NOT_ATTACHED_ERROR;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, pause canceled\n");
        return statError;
    }

    gStartTime.setTime();
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Pausing the application...\n");

    if (broadcastStream_->send(PROT_PAUSE_APPLICATION, "") == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send pause request\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_PAUSE_APPLICATION_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postPauseApplication;
    if (blocking == true)
    {
        statError = receiveAck(blocking);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive pause ack\n");
            return statError;
        }
    }
    return STAT_OK;
}

StatError_t STAT_FrontEnd::postPauseApplication()
{
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Paused!\n");
    isRunning_ = false;
    gEndTime.setTime();
    addPerfData("Pause Time", (gEndTime - gStartTime).getDoubleTime());
    return STAT_OK;
}

StatError_t STAT_FrontEnd::resume(bool blocking)
{
    StatError_t statError;
    PacketPtr packet;

    if (isRunning_ == true)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "Application already running... ignoring request to resume\n");
        return STAT_OK;
    }
    if (isAttached_ == false)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "STAT not attached to the application\n");
        return STAT_NOT_ATTACHED_ERROR;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, resume canceled\n");
        return statError;
    }

    gStartTime.setTime();
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Resuming the application...\n");

    if (broadcastStream_->send(PROT_RESUME_APPLICATION, "") == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send resume request\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_RESUME_APPLICATION_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postResumeApplication;
    if (blocking == true)
    {
        statError = receiveAck(blocking);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive resume ack\n");
            return statError;
        }
    }
    return STAT_OK;
}

StatError_t STAT_FrontEnd::postResumeApplication()
{
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Resumed!\n");
    isRunning_ = true;
    gEndTime.setTime();
    addPerfData("Resume Time", (gEndTime - gStartTime).getDoubleTime());
    return STAT_OK;
}

bool STAT_FrontEnd::isRunning()
{
    return isRunning_;
}

StatError_t STAT_FrontEnd::sampleStackTraces(unsigned int sampleType, unsigned int nTraces, unsigned int traceFrequency, unsigned int nRetries, unsigned int retryFrequency, unsigned int maxDaemonNumThreads, bool blocking, const char *variableSpecification)
{
    StatError_t statError;

    if (isAttached_ == false)
    {
        printMsg(STAT_NOT_ATTACHED_ERROR, __FILE__, __LINE__, "STAT not attached to the application... ignoring request to gather samples\n");
        return STAT_NOT_ATTACHED_ERROR;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, sample canceled\n");
        return statError;
    }

    gStartTime.setTime();
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Sampling traces...\n");
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "%d traces with %d frequency, %d retries with %d frequency, type %d\n", nTraces, traceFrequency, nRetries, retryFrequency, sampleType);

    if (broadcastStream_->send(PROT_SAMPLE_TRACES, "%ud %ud %ud %ud %ud %ud %s", nTraces, traceFrequency, nRetries, retryFrequency, sampleType, maxDaemonNumThreads, variableSpecification) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send request to sample\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_SAMPLE_TRACES_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postSampleStackTraces;
    if (blocking == true)
    {
        statError = receiveAck(blocking);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive stack samples\n");
            return statError;
        }
    }

    return STAT_OK;
}

StatError_t STAT_FrontEnd::postSampleStackTraces()
{
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Traces sampled!\n");
    gEndTime.setTime();
    addPerfData("Sample Traces Time", (gEndTime - gStartTime).getDoubleTime());
    return STAT_OK;
}

StatError_t STAT_FrontEnd::gatherLastTrace(bool blocking, const char *altDotFilename)
{
    StatError_t statError;

    if (altDotFilename == NULL)
        snprintf(altDotFilename_, BUFSIZE, "NULL");
    else
        snprintf(altDotFilename_, BUFSIZE, altDotFilename);

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, gather canceled\n");
        return statError;
    }

    statError = gatherImpl(PROT_SEND_LAST_TRACE, blocking);
    return statError;
}

StatError_t STAT_FrontEnd::gatherTraces(bool blocking, const char *altDotFilename)
{
    StatError_t statError;

    if (altDotFilename == NULL)
        snprintf(altDotFilename_, BUFSIZE, "NULL");
    else
        snprintf(altDotFilename_, BUFSIZE, altDotFilename);

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, gather canceled\n");
        return statError;
    }

    statError = gatherImpl(PROT_SEND_TRACES, blocking);
    return statError;
}

StatError_t STAT_FrontEnd::gatherImpl(StatProt_t type, bool blocking)
{
    StatError_t statError = STAT_OK;

    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    gStartTime.setTime();
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Merging traces...\n");

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Sending request to daemons to send traces\n");
    if (mergeStream_->send(type, "") == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send request to gather traces\n");
        return STAT_MRNET_ERROR;
    }
    if (mergeStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    isPendingAck_ = true;
    if (type == PROT_SEND_LAST_TRACE)
        pendingAckTag_ = PROT_SEND_LAST_TRACE_RESP;
    else
        pendingAckTag_ = PROT_SEND_TRACES_RESP;

    if (blocking == true)
    {
        statError = receiveStackTraces(true);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive attach ack\n");
            return statError;
        }
    }

    return STAT_OK;
}

StatError_t STAT_FrontEnd::receiveStackTraces(bool blocking)
{
    static int sMergeCount = -1;
    int tag, totalWidth, intRet, dummyRank, offset, nodeId, numProcs;
    uint64_t byteArrayLen;
    unsigned int sampleType;
    char outFile[BUFSIZE], perfData[BUFSIZE], outSuffix[BUFSIZE], *byteArray = NULL;
    char **graphAttrKeys, **graphAttrValues, tmpStr[BUFSIZE];
    list<int>::iterator ranksIter;
    set<int>::iterator missingRanksIter;
    graphlib_graph_p stackTraces = NULL, sortedStackTraces = NULL, withMissingStackTraces = NULL;
    graphlib_error_t graphlibError;
    IntList_t *hostRanks;
    PacketPtr packet;
    StatBitVectorEdge_t *edge = NULL;
    StatCountRepEdge_t *countRepEdge = NULL;
    graphlib_nodeattr_t nodeAttr = {1,0,20,GRC_LIGHTGREY,0,0,(void *)gErrorLabel, -1};
    graphlib_edgeattr_t edgeAttr = {1,0,NULL,0,0,0};
    StatError_t statError;

    do
    {
        intRet = mergeStream_->recv(&tag, packet, false);
        if (intRet == 0)
        {
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                return STAT_DAEMON_ERROR;
            }
            if (blocking == true)
                usleep(1000);
        }
    } while (intRet == 0 && blocking == true);

    /* Check for errors */
    if (intRet == 0 && blocking == false)
        return STAT_PENDING_ACK;
    else if (intRet == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to gather stack traces\n");
        isPendingAck_ = false;
        return STAT_MRNET_ERROR;
    }

    sMergeCount++;
    if (tag == PROT_SEND_LAST_TRACE_RESP)
        snprintf(outSuffix, BUFSIZE, "2D");
    else
        snprintf(outSuffix, BUFSIZE, "3D");
    snprintf(perfData, BUFSIZE, "Gather %s Traces Time (receive and merge)", outSuffix);
    addPerfData(perfData, -1.0);
    isPendingAck_ = false;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Unpacking traces\n");
    if (packet->unpack("%Ac %d %d %ud", &byteArray, &byteArrayLen, &totalWidth, &dummyRank, &sampleType) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "stream::unpack(PROT_COLLECT_TRACES_RESP, \"%%auc\") failed\n");
        return STAT_MRNET_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Deserializing graph, byteArrayLen = %d\n", byteArrayLen);
    if (sampleType & STAT_SAMPLE_THREADS)
    {
        gStatBitVectorFunctions->edge_checksum = statCountRepEdgeCheckSum;
        gStatReorderFunctions->edge_checksum = statCountRepEdgeCheckSum;
    }
    if (sampleType & STAT_SAMPLE_COUNT_REP)
        graphlibError = graphlib_deserializeBasicGraph(&stackTraces, gStatCountRepFunctions, byteArray, byteArrayLen);
    else
        graphlibError = graphlib_deserializeBasicGraph(&stackTraces, gStatBitVectorFunctions, byteArray, byteArrayLen);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "deserializeBasicGraph() failed\n");
        return STAT_GRAPHLIB_ERROR;
    }

    gEndTime.setTime();
    addPerfData("\tMerge", (gEndTime - gStartTime).getDoubleTime());

    numProcs = getNumProcs();
    
    if (sampleType & STAT_SAMPLE_COUNT_REP)
    {
        sortedStackTraces = stackTraces;
        stackTraces = NULL;
    }
    else
    {
        /* Translate the graphs into rank ordered bit vector */
        /* First we need to set the bit vector length to the proctab size, just in case there are missing ranks */
        extern int gStatGraphRoutinesTotalWidth;
        gStatGraphRoutinesTotalWidth = statBitVectorLength(numProcs);

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating sorted graph\n");
        gStartTime.setTime();
        offset = 0;
        sortedStackTraces = statNewGraph(gStatReorderFunctions);
        if (sortedStackTraces == NULL)
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to create new graph\n");
            return STAT_GRAPHLIB_ERROR;
        }

        /* Copy the unordered graphs, but with empty bit vectors */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Copying graph with empty edges\n");
        graphlibError = graphlib_mergeGraphs(sortedStackTraces, stackTraces);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to merge graph with empty edge labels\n");
            return STAT_GRAPHLIB_ERROR;
        }

        /* Fill edge labels on a per daemon basis */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Filling in edges\n");
        for (ranksIter = remapRanksList_.begin(); ranksIter != remapRanksList_.end(); ranksIter++)
        {
            /* Fill edge labels for this daemon */
            hostRanks = mrnetRankToMpiRanksMap_[*ranksIter];
            gStatGraphRoutinesRanksList = hostRanks->list;
            gStatGraphRoutinesRanksListLength = hostRanks->count;
            gStatGraphRoutinesCurrentIndex = offset;
            graphlibError = graphlib_mergeGraphs(sortedStackTraces, stackTraces);
            if (GRL_IS_FATALERROR(graphlibError))
            {
                printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to fill edge labels\n");
                return STAT_GRAPHLIB_ERROR;
            }

            /* update offset, round up to the nearest bit vector count*/
            offset += statBitVectorLength(hostRanks->count);
        }

        gEndTime.setTime();
        addPerfData("Graph Rank Ordering Time", (gEndTime - gStartTime).getDoubleTime());
    }

    if (missingRanks_.size() > 0)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Adding %d missing ranks\n", missingRanks_.size());
        gStartTime.setTime();

        withMissingStackTraces = createRootedGraph(sampleType);
        if (withMissingStackTraces == NULL)
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error creating rooted graph\n");
            return STAT_GRAPHLIB_ERROR;
        }
        int functionIndex;
        map<string, string>::iterator nodeAttrIter;

        nodeAttr.attr_values = (void **)calloc(1, gNumNodeAttrs * sizeof(void *));
        if (nodeAttr.attr_values == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error callocing %d nodeAttr.attr_values\n", strerror(errno), gNumNodeAttrs);
            return STAT_ALLOCATE_ERROR;
        }
        graphlibError = graphlib_getNodeAttrIndex(withMissingStackTraces, "function", &functionIndex);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to get node attr index for function\n");
            free(nodeAttr.attr_values);
            return STAT_GRAPHLIB_ERROR;
        }
        nodeAttr.attr_values[functionIndex] = statCopyNodeAttr("function", nodeAttr.label);
        nodeId = statStringHash((char *)nodeAttr.label);
        graphlibError = graphlib_addNode(withMissingStackTraces, nodeId, &nodeAttr);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error adding node to graph\n");
            free(nodeAttr.attr_values);
            return STAT_GRAPHLIB_ERROR;
        }
        free(nodeAttr.attr_values);

        map<string, string>::iterator edgeAttrIter;
        edgeAttr.attr_values = (void **)calloc(1, gNumEdgeAttrs * sizeof(void *));
        if (edgeAttr.attr_values == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error callocing %d edgeAttr.attr_values\n", strerror(errno), gNumEdgeAttrs);
            return STAT_ALLOCATE_ERROR;
        }
        if (sampleType & STAT_SAMPLE_COUNT_REP)
        {
            countRepEdge = (StatCountRepEdge_t *)malloc(sizeof(StatCountRepEdge_t));
            if (countRepEdge == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to initialize edge\n", strerror(errno));
                free(edgeAttr.attr_values);
                return STAT_ALLOCATE_ERROR;
            }
            countRepEdge->count = missingRanks_.size();
            countRepEdge->checksum = 0;
            for (missingRanksIter = missingRanks_.begin(); missingRanksIter != missingRanks_.end(); missingRanksIter++)
                countRepEdge->checksum += *missingRanksIter + 1;
            countRepEdge->representative = *(missingRanks_.begin());
            edgeAttr.label = (void *)countRepEdge;
        }
        else
        {
            edge = initializeBitVectorEdge(numProcs);
            if (edge == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to initialize edge\n");
                free(edgeAttr.attr_values);
                return STAT_ALLOCATE_ERROR;
            }
            for (missingRanksIter = missingRanks_.begin(); missingRanksIter != missingRanks_.end(); missingRanksIter++)
                edge->bitVector[*missingRanksIter / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(*missingRanksIter % STAT_BITVECTOR_BITS);
            edgeAttr.label = (void *)edge;
        }

        int index;
        if (sampleType & STAT_SAMPLE_COUNT_REP)
        {
            graphlib_getEdgeAttrIndex(withMissingStackTraces, "count", &index);
            edgeAttr.attr_values[index] = statCopyEdgeAttr("count", (void *)&countRepEdge->count);
            graphlib_getEdgeAttrIndex(withMissingStackTraces, "rep", &index);
            edgeAttr.attr_values[index] = statCopyEdgeAttr("rep", (void *)&countRepEdge->representative);
            graphlib_getEdgeAttrIndex(withMissingStackTraces, "sum", &index);
            edgeAttr.attr_values[index] = statCopyEdgeAttr("sum", (void *)&countRepEdge->checksum);
        }
        else if (sampleType & STAT_SAMPLE_THREADS)
        {
            graphlib_getEdgeAttrIndex(withMissingStackTraces, "bv", &index);
            edgeAttr.attr_values[index] = statCopyEdgeAttr("bv", (void *)edge);
        }
        else
        {
            graphlib_getEdgeAttrIndex(withMissingStackTraces, "bv", &index);
            edgeAttr.attr_values[index] = statCopyEdgeAttr("bv", (void *)edge);
        }

        graphlibError = graphlib_addDirectedEdge(withMissingStackTraces, 0, nodeId, &edgeAttr);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error adding edge to graph\n");
            statFreeEdgeAttrs(edgeAttr.attr_values, withMissingStackTraces);
            if (countRepEdge != NULL)
                statFreeCountRepEdge(countRepEdge);
            return STAT_GRAPHLIB_ERROR;
        }
        if (countRepEdge != NULL)
            statFreeCountRepEdge(countRepEdge);
        if (edge != NULL)
            statFreeEdge(edge);
        statFreeEdgeAttrs(edgeAttr.attr_values, withMissingStackTraces);
        graphlibError = graphlib_mergeGraphs(withMissingStackTraces, sortedStackTraces);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to merge graph with missing tasks\n");
            free(edgeAttr.attr_values);
            return STAT_GRAPHLIB_ERROR;
        }
        graphlibError = graphlib_delGraph(sortedStackTraces);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error deleting graph\n");
            free(edgeAttr.attr_values);
            return STAT_GRAPHLIB_ERROR;
        }
        sortedStackTraces = withMissingStackTraces;

        gEndTime.setTime();
        addPerfData("Add Missing Ranks Time", (gEndTime - gStartTime).getDoubleTime());
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Exporting %s graph to dot\n", outSuffix);
    gStartTime.setTime();
    if (sampleType & STAT_SAMPLE_COUNT_REP)
        graphlibError = graphlib_colorGraphByLeadingEdgeAttr(sortedStackTraces, "sum");
    else if (sampleType & STAT_SAMPLE_THREADS)
        graphlibError = graphlib_colorGraphByLeadingEdgeAttr(sortedStackTraces, "tbvsum");
    else
        graphlibError = graphlib_colorGraphByLeadingEdgeAttr(sortedStackTraces, "bv");
    if (GRL_IS_FATALERROR(graphlibError))
    {
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "graphlib error coloring graph by leading edge\n");
        return STAT_GRAPHLIB_ERROR;
    }
    graphlibError = graphlib_scaleNodeWidth(sortedStackTraces, 80, 160);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "graphlib error scaling node width\n");
        return STAT_GRAPHLIB_ERROR;
    }
    if (strcmp(altDotFilename_, "NULL") != 0)
        snprintf(outFile, BUFSIZE, "%s/%02d_%s.%s.dot", outDir_, sMergeCount, altDotFilename_, outSuffix);
    else
        snprintf(outFile, BUFSIZE, "%s/%02d_%s.%s.dot", outDir_, sMergeCount, filePrefix_, outSuffix);
    snprintf(lastDotFileName_, BUFSIZE, "%s", outFile);

    graphAttrKeys = (char **)malloc(sizeof(char *));
    if (graphAttrKeys == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to allocate for graph attr keys\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }
    graphAttrKeys[0] = strdup("type");
    graphAttrValues = (char **)malloc(sizeof(char *));
    if (graphAttrValues == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to allocate for graph attr values\n", strerror(errno));
        free(graphAttrKeys[0]);
        return STAT_ALLOCATE_ERROR;
    }
    snprintf(tmpStr, BUFSIZE, "stat_%d_%d", STAT_MAJOR_VERSION, STAT_MINOR_VERSION);
    graphAttrValues[0] = strdup(tmpStr);
    graphlibError = graphlib_exportAttributedGraph(outFile, GRF_DOT, sortedStackTraces, 1, graphAttrKeys, graphAttrValues);
    free(graphAttrKeys[0]);
    free(graphAttrValues[0]);
    free(graphAttrKeys);
    free(graphAttrValues);
    if (GRL_IS_FATALERROR(graphlibError))
    {
        printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "graphlib error exporting graph to dot format\n");
        return STAT_GRAPHLIB_ERROR;
    }

    gEndTime.setTime();
    addPerfData("Export Graph Files Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Traces merged!\n");

    statError = dumpPerf();
    if (statError != STAT_OK)
        printMsg(statError, __FILE__, __LINE__, "Failed to dump performance results\n");

    if (sampleType & STAT_SAMPLE_THREADS)
    {
        gStatBitVectorFunctions->edge_checksum = statEdgeCheckSum;
        gStatReorderFunctions->edge_checksum = statEdgeCheckSum;
    }

    if (stackTraces != NULL)
    {
        graphlibError = graphlib_delGraph(stackTraces);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error deleting graph\n");
            return STAT_GRAPHLIB_ERROR;
        }
    }
    if (sortedStackTraces != NULL)
    {
        graphlibError = graphlib_delGraph(sortedStackTraces);
        if (GRL_IS_FATALERROR(graphlibError))
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Error deleting graph\n");
            return STAT_GRAPHLIB_ERROR;
        }
    }
    if (byteArray != NULL)
        free(byteArray);

    return STAT_OK;
}


char *STAT_FrontEnd::getNodeInEdge(int nodeId)
{
    int tag, intRet, totalWidth, dummyRank, offset;
    unsigned int sampleType;
    uint64_t byteArrayLen;
    char *byteArray = NULL, *edgeLabel;
    list<int>::iterator ranksIter;
    set<int>::iterator missingRanksIter;
    IntList_t *hostRanks;
    PacketPtr packet;
    StatBitVectorEdge_t *unorderedEdge = NULL, *orderedEdge;

    gStartTime.setTime();

    /* Make sure we're in the expected state */
    if (isAttached_ == false)
    {
        printMsg(STAT_NOT_ATTACHED_ERROR, __FILE__, __LINE__, "STAT not attached to the application... ignoring request to gather samples\n");
        return NULL;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return NULL;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return NULL;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return NULL;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Getting incoming-edge for node %d\n", nodeId);

    if (nodeId == statStringHash(gErrorLabel))
    {
        orderedEdge = initializeBitVectorEdge(getNumProcs());
        if (orderedEdge == NULL)
        {
            printMsg(STAT_GRAPHLIB_ERROR, __FILE__, __LINE__, "Failed to initialize edge\n");
            return NULL;
        }
        for (missingRanksIter = missingRanks_.begin(); missingRanksIter != missingRanks_.end(); missingRanksIter++)
            orderedEdge->bitVector[*missingRanksIter / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(*missingRanksIter % STAT_BITVECTOR_BITS);
    }
    else
    {
        if (mergeStream_->send(PROT_SEND_NODE_IN_EDGE, "%d", nodeId) == -1)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send request to gather edge label\n");
            return NULL;
        }
        if (mergeStream_->flush() != 0)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
            return NULL;
        }

        do
        {
            intRet = mergeStream_->recv(&tag, packet, false);
            if (intRet == 0)
            {
                if (daemonsHaveExited())
                {
                    printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                    return NULL;
                }
                usleep(1000);
            }
        } while (intRet == 0);
        if (intRet == -1)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to gather edge label\n");
            return NULL;
        }
        if (tag != PROT_SEND_NODE_IN_EDGE_RESP)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to gather edge label.  Unexpected tag %d, expecting tag %d\n", tag, PROT_SEND_NODE_IN_EDGE_RESP);
            return NULL;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Unpacking traces\n");
        if (packet->unpack("%Ac %d %d %ud", &byteArray, &byteArrayLen, &totalWidth, &dummyRank, &sampleType) == -1)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "stream::unpack(PROT_SEND_NODE_IN_EDGE_RESP) failed\n");
            return NULL;
        }

        statDeserializeEdge((void **)&unorderedEdge, byteArray, byteArrayLen);
        if (unorderedEdge == NULL)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to deserialize unordered edge\n");
            return NULL;
        }

        /* Create an empty bit vector for reordering */
        orderedEdge = (StatBitVectorEdge_t *)statCopyEdgeInitializeEmpty(unorderedEdge);
        if (orderedEdge == NULL)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "statCopyEdgeInitializeEmpty failed\n");
            statFreeEdge((void *)unorderedEdge);
            return NULL;
        }

        /* Fill edge label on a per daemon basis */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Filling in edges\n");
        offset = 0;
        for (ranksIter = remapRanksList_.begin(); ranksIter != remapRanksList_.end(); ranksIter++)
        {
            /* Fill edge labels for this daemon */
            hostRanks = mrnetRankToMpiRanksMap_[*ranksIter];
            gStatGraphRoutinesRanksList = hostRanks->list;
            gStatGraphRoutinesRanksListLength = hostRanks->count;
            gStatGraphRoutinesCurrentIndex = offset;
            statMergeEdgeOrdered(orderedEdge, unorderedEdge);
            offset += statBitVectorLength(hostRanks->count);
        }
        statFreeEdge((void *)unorderedEdge);
    }
    edgeLabel = statEdgeToText((void *)orderedEdge);
    statFreeEdge((void *)orderedEdge);

    if (byteArray != NULL)
        free(byteArray);

    gEndTime.setTime();
    addPerfData("\tGather Edge", (gEndTime - gStartTime).getDoubleTime());

    return edgeLabel;
}


char *STAT_FrontEnd::getLastDotFilename()
{
    return lastDotFileName_;
}


StatError_t STAT_FrontEnd::addPerfData(const char *buf, double time)
{
    pair<string, double> perfData(buf, time);
    performanceData_.push_back(perfData);
    return STAT_OK;
}


StatError_t STAT_FrontEnd::dumpPerf()
{
    int i, size;
    static int sCount = 0;
    char perfFileName[BUFSIZE];
    FILE *perfFile;
    bool isUsageLogging = false;
    char usageLogFile[BUFSIZE], *envValue;
    double launchTime = -1.0, attachTime = -1.0, sampleTime = -1.0, mergeTime = -1.0, orderTime = -1.0, exportTime = -1.0;
    struct timeval timeStamp;
    time_t currentTime;
    char timeBuf[BUFSIZE], *userName = NULL;
    uid_t uid;
    struct passwd *pwd;

    size = performanceData_.size();
    if (size <= 0)
        return STAT_OK;

    if (strcmp(outDir_, "NULL") == 0)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "Output directory not created.  Performance results not written.\n");
        return STAT_FILE_ERROR;
    }

    snprintf(perfFileName, BUFSIZE, "%s/%s.perf", outDir_, filePrefix_);
    perfFile = fopen(perfFileName, "a");
    if (perfFile ==NULL)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to create performance results file\n", strerror(errno));
        return STAT_FILE_ERROR;
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Dumping performance info to %s\n", perfFileName);

    /* Dump header info only on first invocation */
    if (sCount == 0)
    {
        fprintf(perfFile, "STAT version %d.%d.%d\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
        fprintf(perfFile, "Application: %s\n", applExe_);
        fprintf(perfFile, "Number of Nodes: %d\n", nApplNodes_);
        fprintf(perfFile, "Number of Processes: %d\n\n", nApplProcs_);
        fprintf(perfFile, "All times reported in seconds\n\n");
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Performance Results:\n");
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "STAT version %d.%d.%d\n", STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION);
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Application: %s\n", applExe_);
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Number of Nodes: %d\n", nApplNodes_);
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Number of Processes: %d\n\n", nApplProcs_);
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "All times reported in seconds\n\n");
    }
    sCount++;

    /* Write out the results in the list */
    for (i = 0; i < size; i++)
    {
        if (performanceData_[i].second != -1.0)
            fprintf(perfFile, "%s: %.3f\n", performanceData_[i].first.c_str(), performanceData_[i].second);
        else
            fprintf(perfFile, "%s:\n", performanceData_[i].first.c_str());
    }
    fclose(perfFile);
    if (verbose_ == STAT_VERBOSE_FULL || logging_ & STAT_LOG_FE)
    {
        for (i = 0; i < size; i++)
        {
            if (performanceData_[i].second != -1.0)
                printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "%s: %.3lf\n", performanceData_[i].first.c_str(), performanceData_[i].second);
            else
                printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "%s:\n", performanceData_[i].first.c_str());
        }
    }

    snprintf(usageLogFile, BUFSIZE, "/dev/null");
#ifdef STAT_USAGELOG
    snprintf(usageLogFile, BUFSIZE, STAT_USAGELOG);
    isUsageLogging = true;
#endif
    envValue = getenv("STAT_USAGE_LOG");
    if (envValue != NULL)
    {
        snprintf(usageLogFile, BUFSIZE, envValue);
        isUsageLogging = true;
    }
    if (isUsageLogging == true && sCount == 1)
    {
        for (i = 0; i < size; i++)
        {
            if (performanceData_[i].first.find("Total MRNet Launch Time") != string::npos)
                launchTime = performanceData_[i].second;
            else if (performanceData_[i].first.find("Attach Time") != string::npos)
                attachTime = performanceData_[i].second;
            else if (performanceData_[i].first.find("Sample") != string::npos && sampleTime == -1.0)
                sampleTime = performanceData_[i].second;
            else if (performanceData_[i].first.find("Merge") != string::npos && mergeTime == -1.0)
                mergeTime = performanceData_[i].second;
            else if (performanceData_[i].first.find("Ordering") != string::npos && orderTime == -1.0)
                orderTime = performanceData_[i].second;
            else if (performanceData_[i].first.find("Export") != string::npos && exportTime == -1.0)
                exportTime = performanceData_[i].second;
        }
        perfFile = fopen(usageLogFile, "a");
        if (perfFile == NULL)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to open usage log file %s\n", strerror(errno), usageLogFile);
            performanceData_.clear();
            return STAT_OK;
        }
        gettimeofday(&timeStamp, NULL);
        currentTime = timeStamp.tv_sec;
        strftime(timeBuf, BUFSIZE, "%Y-%m-%d-%T", localtime(&currentTime));
        uid = getuid();
        pwd = getpwuid(uid);
        if (pwd != NULL)
        {
            if (pwd->pw_name != NULL)
                userName = strdup(pwd->pw_name);
            else
            {
                if (getenv("USER") != NULL)
                    userName = strdup(getenv("USER"));
            }
        }
        if (userName == NULL)
            userName = strdup("NULL");
        fprintf(perfFile, "%s %s %s %d.%d.%d %d %d %.3f %.3f %.3f %.3f %.3f %.3f\n", timeBuf, hostname_, userName, STAT_MAJOR_VERSION, STAT_MINOR_VERSION, STAT_REVISION_VERSION, nApplNodes_, nApplProcs_, launchTime, attachTime, sampleTime, mergeTime, orderTime, exportTime);
        fclose(perfFile);
        if (userName != NULL)
            free(userName);
    }

    performanceData_.clear();

    return STAT_OK;
}

bool STAT_FrontEnd::isAttached()
{
    return isAttached_;
}

StatError_t STAT_FrontEnd::detachApplication(bool blocking)
{
    return detachApplication(NULL, 0, blocking);
}


StatError_t STAT_FrontEnd::detachApplication(int *stopList, int stopListSize, bool blocking)
{
    StatError_t statError;
    PacketPtr packet;

    /* Make sure we're in the expected state */
    if (isAttached_ == false)
    {
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "STAT not attached to the application... ignoring request to detach\n");
        return STAT_OK;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
#ifndef DYSECTAPI
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
#endif
    }
    if (daemonsHaveExited())
    {
#ifndef DYSECTAPI
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
#endif
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, detach canceled\n");
        return statError;
    }

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Detaching from application...\n");

#ifdef DYSECTAPI
    if (daemonsKilled_)
        return STAT_OK;
#endif

    if (broadcastStream_->send(PROT_DETACH_APPLICATION, "%aud", stopList, stopListSize) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send detach request\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_DETACH_APPLICATION_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postDetachApplication;
    if (blocking == true)
    {
        statError = receiveAck(blocking);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive detach ack\n");
            return statError;
        }
    }
    return STAT_OK;
}


StatError_t STAT_FrontEnd::postDetachApplication()
{
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Detached!\n");
    isAttached_ = false;
    isRunning_ = false;
    return STAT_OK;
}


StatError_t STAT_FrontEnd::shutdownMrnetTree()
{
    if (broadcastStream_->send(PROT_EXIT, "") == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send exit message\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    return STAT_OK;
}


void STAT_FrontEnd::printMsg(StatError_t statError, const char *sourceFile, int sourceLine, const char *fmt, ...)
{
    va_list arg;
    char timeString[BUFSIZE];
    const char *timeFormat = "%b %d %T";
    time_t currentTime;
    struct tm *localTime;

    /* If this is a log message and we're not logging, then return */
    if (statError == STAT_LOG_MESSAGE && !(logging_ & STAT_LOG_FE))
        return;

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
        vsnprintf(lastErrorMessage_, BUFSIZE, fmt, arg);
        va_end(arg);
        fprintf(stderr, "%s", lastErrorMessage_);
    }
    else if ((statError == STAT_STDOUT && verbose_ >= STAT_VERBOSE_STDOUT) || (statError == STAT_VERBOSITY && verbose_ == STAT_VERBOSE_FULL))
    {
        /* Print the message to stdout */
        va_start(arg, fmt);
        vfprintf(stdout, fmt, arg);
        va_end(arg);
        fflush(stdout);
    }

    /* Print the message to the log */
    if (logging_ & STAT_LOG_FE && gStatOutFp != NULL)
    {
        if (logging_ & STAT_LOG_MRN)
        {
            char msg[BUFSIZE];
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


StatError_t STAT_FrontEnd::terminateApplication(bool blocking)
{
    StatError_t statError;
    PacketPtr packet;

    /* Make sure we're in the expected state */
    if (isAttached_ == false)
    {
        printMsg(STAT_NOT_ATTACHED_ERROR, __FILE__, __LINE__, "STAT not attached to the application... ignoring request to terminate\n");
        return STAT_NOT_ATTACHED_ERROR;
    }
    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }
    if (isKilled())
    {
        printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
        return STAT_APPLICATION_EXITED;
    }
    if (daemonsHaveExited())
    {
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
        return STAT_DAEMON_ERROR;
    }

    statError = receiveAck(true);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to receive pending ack, terminate canceled\n");
        return statError;
    }

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Terminating Application ...");

    if (broadcastStream_->send(PROT_TERMINATE_APPLICATION, "") == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send terminate request\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    pendingAckTag_ = PROT_TERMINATE_APPLICATION_RESP;
    isPendingAck_ = true;
    pendingAckCb_ = &STAT_FrontEnd::postTerminateApplication;
    if (blocking == true)
    {
        statError = receiveAck(blocking);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to receive terminate ack\n");
            return statError;
        }
    }
    return STAT_OK;
}


StatError_t STAT_FrontEnd::postTerminateApplication()
{
    isRunning_ = false;
    isAttached_ = false;
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\tApplication terminated!\n");
    return STAT_OK;
}


void STAT_FrontEnd::printCommunicationNodeList()
{
    set<string>::iterator communicationNodeSetIter;

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\tCommunication Node List: ");
    for (communicationNodeSetIter = communicationNodeSet_.begin(); communicationNodeSetIter != communicationNodeSet_.end(); communicationNodeSetIter++)
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "%s, ", communicationNodeSetIter->c_str());
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\n");
}


void STAT_FrontEnd::printApplicationNodeList()
{
    multiset<string>::iterator applicationNodeMultiSetIter;

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\tApplication Node List: ");
    for (applicationNodeMultiSetIter = applicationNodeMultiSet_.begin(); applicationNodeMultiSetIter != applicationNodeMultiSet_.end(); applicationNodeMultiSetIter++)
        printMsg(STAT_STDOUT, __FILE__, __LINE__, "%s, ", applicationNodeMultiSetIter->c_str());
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\n");
}


StatError_t STAT_FrontEnd::setNodeListFromConfigFile(char **nodeList)
{
    int count = 0, ch;
    char fileName[BUFSIZE], input[BUFSIZE], *nodefile;
    FILE *file;
    string nodes;

    if (nodeListFile_ == NULL)
    {
       snprintf(fileName, BUFSIZE, "%s/etc/STAT/nodes.txt", getInstallPrefix());
       nodefile = fileName;
    }
    else
       nodefile = nodeListFile_;
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting node list from file %s\n", nodefile);
    file = fopen(nodefile, "r");
    if (file != NULL)
    {
        while (fscanf(file, "%s", input) != EOF)
        {
            if (input[0] == '#')
            {
                while (1)
                {
                    ch = fgetc(file);
                    if ((char)ch == '\n' || ch == EOF)
                        break;
                }
                continue;
            }
            if (count != 0)
                nodes += ",";
            nodes += input;
            count++;
        }
        *nodeList = strdup(nodes.c_str());
        if (*nodeList == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) for nodeList\n", strerror(errno), nodes.c_str());
            fclose(file);
            return STAT_ALLOCATE_ERROR;
        }
        fclose(file);
    }
    else
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "No config file %s\n", nodefile);

    return STAT_OK;
}


unsigned int STAT_FrontEnd::getLauncherPid()
{
    return launcherPid_;
}


unsigned int STAT_FrontEnd::getNumApplProcs()
{
    return nApplProcs_;
}


unsigned int STAT_FrontEnd::getNumApplNodes()
{
    return nApplNodes_;
}


StatError_t STAT_FrontEnd::setJobId(const char *jobId)
{
    if (jobId != NULL)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting job ID to %s\n", jobId);
        if (jobId_ != NULL)
            free(jobId_);
        jobId_ = strdup(jobId);
        if (jobId_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to set job ID path with strdup() to %s\n", strerror(errno), jobId);
            return STAT_ALLOCATE_ERROR;
        }
    }
    return STAT_OK;
}


const char *STAT_FrontEnd::getJobId()
{
    return jobId_;
}


const char *STAT_FrontEnd::getApplExe()
{
    return applExe_;
}


StatError_t STAT_FrontEnd::setToolDaemonExe(const char *toolDaemonExe)
{
    if (toolDaemonExe != NULL)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting tool daemon path to %s\n", toolDaemonExe);
        if (toolDaemonExe_ != NULL)
            free(toolDaemonExe_);
        toolDaemonExe_ = strdup(toolDaemonExe);
        if (toolDaemonExe_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to set tool daemon path with strdup() to %s\n", strerror(errno), toolDaemonExe);
            return STAT_ALLOCATE_ERROR;
        }
    }
    return STAT_OK;
}


const char *STAT_FrontEnd::getToolDaemonExe()
{
    return toolDaemonExe_;
}


StatError_t STAT_FrontEnd::setOutDir(const char *outDir)
{
    snprintf(outDir_, BUFSIZE, "%s", outDir);
    return STAT_OK;
}


const char *STAT_FrontEnd::getOutDir()
{
    return outDir_;
}


StatError_t STAT_FrontEnd::setFilePrefix(const char *filePrefix)
{
    snprintf(filePrefix_, BUFSIZE, "%s", filePrefix);
    return STAT_OK;
}


const char *STAT_FrontEnd::getFilePrefix()
{
    return filePrefix_;
}


void STAT_FrontEnd::setProcsPerNode(unsigned int procsPerNode)
{
    procsPerNode_ = procsPerNode;
}


unsigned int STAT_FrontEnd::getProcsPerNode()
{
    return procsPerNode_;
}


StatError_t STAT_FrontEnd::setFilterPath(const char *filterPath)
{
    if (filterPath != NULL)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Setting filter path to %s\n", filterPath);
        if (filterPath_ != NULL)
            free(filterPath_);
        filterPath_ = strdup(filterPath);
        if (filterPath_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to set filter path with strdup() to %s\n", strerror(errno), filterPath_);
            return STAT_ALLOCATE_ERROR;
        }
    }
    return STAT_OK;
}


const char *STAT_FrontEnd::getFilterPath()
{
    return filterPath_;
}


const char *STAT_FrontEnd::getRemoteNode()
{
    return remoteNode_;
}


StatError_t STAT_FrontEnd::addLauncherArgv(const char *launcherArg)
{
    launcherArgc_++;
    launcherArgv_ = (char **)realloc(launcherArgv_, launcherArgc_ * sizeof(char *));
    if (launcherArgv_ == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to realloc launcherArgv_ to %d\n", strerror(errno), launcherArgc_);
        return STAT_ALLOCATE_ERROR;
    }
    launcherArgv_[launcherArgc_ - 2] = strdup(launcherArg);
    if (launcherArgv_[launcherArgc_ - 2] == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to copy arg to launcherArgv_ with strdup() to %s\n", strerror(errno), launcherArg);
        return STAT_ALLOCATE_ERROR;
    }

    launcherArgv_[launcherArgc_ - 1] = NULL;

    return STAT_OK;
}

const char **STAT_FrontEnd::getLauncherArgv()
{
    return (const char **)launcherArgv_;
}


unsigned int STAT_FrontEnd::getLauncherArgc()
{
    return launcherArgc_;
}


void STAT_FrontEnd::setVerbose(StatVerbose_t verbose)
{
    verbose_ = verbose;
}


StatVerbose_t STAT_FrontEnd::getVerbose()
{
    return verbose_;
}


void STAT_FrontEnd::setApplicationOption(StatLaunch_t applicationOption)
{
    applicationOption_ = applicationOption;
}


StatLaunch_t STAT_FrontEnd::getApplicationOption()
{
    return applicationOption_;
}


#if (defined(HAVE_GETRLIMIT) && defined(HAVE_SETRLIMIT))
StatError_t increaseSysLimits()
{
    struct rlimit rLim[1];

    if (getrlimit(RLIMIT_NOFILE, rLim) < 0)
    {
        fprintf(stderr, "%s: getrlimit failed\n", strerror(errno));
        return STAT_SYSTEM_ERROR;
    }
    else if (rLim->rlim_cur < rLim->rlim_max)
    {
        rLim->rlim_cur = rLim->rlim_max;
        if (setrlimit(RLIMIT_NOFILE, rLim) < 0)
        {
            fprintf(stderr, "%s: Unable to increase max no. files\n", strerror(errno));
            return STAT_SYSTEM_ERROR;
        }
    }

    if (getrlimit(RLIMIT_NPROC, rLim) < 0)
    {
        fprintf(stderr, "%s: getrlimit failed\n", strerror(errno));
        return STAT_SYSTEM_ERROR;
    }
    else if (rLim->rlim_cur < rLim->rlim_max)
    {
        rLim->rlim_cur = rLim->rlim_max;
        if (setrlimit (RLIMIT_NPROC, rLim) < 0)
        {
            fprintf(stderr, "%s: Unable to increase max no. files\n", strerror(errno));
            return STAT_SYSTEM_ERROR;
        }
    }

    return STAT_OK;
}
#endif


StatError_t STAT_FrontEnd::setRanksList()
{
    unsigned int i;
    int intRet, numProcs;
    map<int, RemapNode_t*> childOrder;
    map<int, RemapNode_t*>::iterator childOrderIter;
    list<int>::iterator remapRanksListIter;
    multiset<string>::iterator applicationNodeMultiSetIter;
    RemapNode_t *root;
    set<string> daemonIpAddrs, daemonSet;
    set<string>::iterator daemonIter;
    XPlat::NetUtils::NetworkAddress networkAddress;
    StatError_t statError;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating the merge order ranks list\n");

    /* First we need to generate the list of STAT BE daemons */
    statError = setDaemonNodes();
#ifdef DYSECTAPI
    if (statError == STAT_APPLICATION_EXITED)
        return statError;
#endif
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to set daemon nodes\n");
        return statError;
    }

    /* Generate the rank tree */
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Building the MRNet topology rank tree\n");
    root = buildRemapTree(leafInfo_.networkTopology->get_Root());
    if (root == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to Build Remap Tree\n");
        return STAT_ALLOCATE_ERROR;
    }

    /* Build the ordered ranks list based on the rank tree */
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating the ranks list based on the rank tree\n");
    remapRanksList_.clear();
    statError = buildRanksList(root);
    if (statError != STAT_OK)
    {
        freeRemapTree(root);
        printMsg(statError, __FILE__, __LINE__, "Failed to build ranks list\n");
        return statError;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Ranks list generated:");
    for (remapRanksListIter = remapRanksList_.begin(); remapRanksListIter != remapRanksList_.end(); remapRanksListIter++)
        printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "%d, ", *remapRanksListIter);
    printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "\n");

    freeRemapTree(root);

    /* Fill in the missing ranks */
    if (nApplNodes_ != leafInfo_.daemons.size())
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Detected %d of %d nodes reporting, generating missing ranks list\n", leafInfo_.daemons.size(), nApplNodes_);

        /* Also create a list of daemon IP addresses to compare */
        for (daemonIter = leafInfo_.daemons.begin(); daemonIter != leafInfo_.daemons.end(); daemonIter++)
        {
            intRet = XPlat::NetUtils::GetNetworkAddress(*daemonIter, networkAddress);
            if (intRet == 0)
                daemonIpAddrs.insert(networkAddress.GetString());
        }

        for (applicationNodeMultiSetIter = applicationNodeMultiSet_.begin(); applicationNodeMultiSetIter != applicationNodeMultiSet_.end(); applicationNodeMultiSetIter++)
            if (leafInfo_.daemons.find(*applicationNodeMultiSetIter) == leafInfo_.daemons.end() && daemonIpAddrs.find(*applicationNodeMultiSetIter) == daemonIpAddrs.end())
                daemonSet.insert(*applicationNodeMultiSetIter);
        numProcs = getNumProcs();
        for (i = 0; i < numProcs; i++)
            if (daemonSet.find(getHostnameForProc(i)) != daemonSet.end())
                missingRanks_.insert(getMpiRankForProc(i));
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Missing ranks list generated with %d tasks\n", missingRanks_.size());
    }

    return STAT_OK;
}


void STAT_FrontEnd::setHasFatalError(bool hasFatalError)
{
    hasFatalError_ = hasFatalError;
}


RemapNode_t *STAT_FrontEnd::buildRemapTree(NetworkTopology::Node *node)
{
    int i;
    set<NetworkTopology::Node *> children;
    set<NetworkTopology::Node *>::iterator childrenIter;
    map<int, RemapNode_t *> childOrder;
    map<int, RemapNode_t *>::iterator childOrderIter;
    RemapNode_t *remapNodeRet, *child;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Remap tree MRNet rank %d\n", node->get_Rank());
    remapNodeRet = (RemapNode_t *)malloc(sizeof(RemapNode_t));
    if (remapNodeRet == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for remapNodeRet\n", strerror(errno));
        return NULL;
    }
    remapNodeRet->numChildren = 0;
    remapNodeRet->children = NULL;
    remapNodeRet->lowRank = -1;
    if (mrnetRankToMpiRanksMap_.find(node->get_Rank()) != mrnetRankToMpiRanksMap_.end())
        remapNodeRet->lowRank = node->get_Rank(); /* This is a BE so return its rank */
    else
    {
        /* This is not a BE so find the lowest rank among this node's children */
        children = node->get_Children();
        for (childrenIter = children.begin(), i = 0; childrenIter != children.end(); childrenIter++, i++)
        {
            child = buildRemapTree(*childrenIter);
            if (child == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to Build Remap Tree\n");
                return NULL;
            }
            if (i == 0)
                remapNodeRet->lowRank = child->lowRank;
            else if (child->lowRank < remapNodeRet->lowRank)
                remapNodeRet->lowRank = child->lowRank;
            childOrder[child->lowRank] = child;
        }

        /* Add the children for this node, ordered by lowest rank */
        remapNodeRet->numChildren = childOrder.size();
        if (remapNodeRet->numChildren != 0)
        {
            remapNodeRet->children = (RemapNode_t **)malloc(remapNodeRet->numChildren * sizeof(RemapNode_t));
            if (remapNodeRet->children == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: malloc failed to allocate for remapNodeRet->children\n", strerror(errno));
                return NULL;
            }
            for (childOrderIter = childOrder.begin(), i = 0; childOrderIter != childOrder.end(); childOrderIter++, i++)
                remapNodeRet->children[i] = childOrderIter->second;
        }
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Remap tree MRNet rank %d with %d children and low rank %d\n", node->get_Rank(), remapNodeRet->numChildren, remapNodeRet->lowRank);

    return remapNodeRet;
}


StatError_t STAT_FrontEnd::buildRanksList(RemapNode_t *node)
{
    int i;

    if (node->numChildren != 0)
    {
        /* Not a daemon so traverse children in rank order */
        for (i = 0; i < node->numChildren; i++)
            buildRanksList(node->children[i]);
    }
    else
    {
        /* If this is a daemon then add its rank to the remap ranks list */
        if (mrnetRankToMpiRanksMap_.find(node->lowRank) != mrnetRankToMpiRanksMap_.end())
            remapRanksList_.push_back(node->lowRank);
    }

    return STAT_OK;
}


StatError_t STAT_FrontEnd::freeRemapTree(RemapNode_t *node)
{
    int i;

    if (node->numChildren != 0)
    {
        /* Not a daemon so traverse children in rank order */
        for (i = 0; i < node->numChildren; i++)
            freeRemapTree(node->children[i]);
        if (node->children != NULL)
            free(node->children);
        if (node != NULL)
            free(node);
    }
    else
    {
        /* This is a daemon so just free it up */
        if (node != NULL)
            free(node);
    }

    return STAT_OK;
}


int statPack(void *data, void *buf, int bufMax, int *bufLen)
{
    int total = 0, len, port, rank, daemonCount, daemonRank;
    unsigned int i, j, nLeaves;
    char *ptr = (char *)buf, *daemonCountPtr, *childCountPtr;
    unsigned int nNodes, depth, minFanout, maxFanout;
    double averageFanout, stdDevFanout;
    LeafInfo_t *leafInfo = (LeafInfo_t *)data;
    NetworkTopology *networkTopology = leafInfo->networkTopology;
    NetworkTopology::Node *node;
    set<string>::iterator daemonIter;
    string currentHost;

    /* Get the Network Topology tree statisctics (we really just want # nodes */
    networkTopology->get_TreeStatistics(nNodes, depth, minFanout, maxFanout, averageFanout, stdDevFanout);

    /* reserve space for the number of daemons (to be calculated later) */
    daemonCountPtr = ptr;
    ptr += sizeof(int);
    total += sizeof(int);

    /* pack up the number of parent nodes */
    nLeaves = leafInfo->leafCps.size();
    memcpy(ptr, (void *)&nLeaves, sizeof(unsigned int));
    ptr += sizeof(unsigned int);
    total += sizeof(int);

    /* write the data one parent at a time */
    daemonCount = 0;
    daemonIter = leafInfo->daemons.begin();
    for (i = 0; i < nLeaves; i++)
    {
        /* get the parent info */
        node = leafInfo->leafCps[i];
        port = node->get_Port();
        currentHost = node->get_HostName();
        rank = node->get_Rank();

        /* calculate the amount of data */
        len = strlen(currentHost.c_str()) + 1;
        total += 3 * sizeof(int) + len;
        if (total > bufMax)
        {
            fprintf(stderr, "Exceeded maximum packing buffer\n");
            return -1;
        }

        /* write the parent host name, port, rank and child count */
        memcpy(ptr, (void *)currentHost.c_str(), len);
        ptr += len;
        memcpy(ptr, (void *)&port, sizeof(int));
        ptr += sizeof(int);
        memcpy(ptr, (void *)&rank, sizeof(int));
        ptr += sizeof(int);
        childCountPtr = ptr;
        ptr += sizeof(unsigned int);

        for (j = 0; j < (leafInfo->daemons.size() / nLeaves) + (leafInfo->daemons.size() % nLeaves > i ? 1 : 0); j++)
        {
            if (daemonIter == leafInfo->daemons.end())
                break;
            len = strlen(daemonIter->c_str()) + 1;
            total += sizeof(int) + len;

            if (total > bufMax)
            {
                fprintf(stderr, "Exceeded maximum packing buffer\n");
                return -1;
            }

            /* copy the daemon host name */
            memcpy(ptr, (void *)(daemonIter->c_str()), len);
            ptr += len;

            /* copy the daemon rank */
            daemonRank = daemonCount + nNodes;
            memcpy(ptr, (void *)(&daemonRank), sizeof(int));
            ptr += sizeof(int);
            daemonIter++;
            daemonCount++;
        }
        memcpy(childCountPtr, (void *)&j, sizeof(unsigned int));
    }

    /* write the daemon count to the appropriate location */
    memcpy(daemonCountPtr, (void *)&daemonCount, sizeof(int));

    *bufLen = total;
    return 0;
}

StatError_t STAT_FrontEnd::addDaemonLogArgs(int &daemonArgc, char ** &daemonArgv)
{
    int current;

    if (daemonArgc == 0)
        daemonArgc = 1;
    current = daemonArgc - 1;

    if (logging_ & STAT_LOG_BE || logging_ & STAT_LOG_SW || logging_ & STAT_LOG_SWERR)
    {
        daemonArgc += 2;
        daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
        if (daemonArgv == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }

        daemonArgv[current] = strdup("-L");
        if (daemonArgv[current] == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[0]\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
        current++;

        daemonArgv[current] = strdup(logOutDir_);
        if (daemonArgv[current] == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) argv[%d]\n", strerror(errno), logOutDir_, current);
            return STAT_ALLOCATE_ERROR;
        }
        current++;

        if (logging_ & STAT_LOG_BE)
        {
            daemonArgc += 2;
            daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
            if (daemonArgv == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
                return STAT_ALLOCATE_ERROR;
            }

            daemonArgv[current] = strdup("-l");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;

            daemonArgv[current] = strdup("BE");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;
        }

        if (logging_ & STAT_LOG_SW)
        {
            daemonArgc += 2;
            daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
            if (daemonArgv == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
                return STAT_ALLOCATE_ERROR;
            }

            daemonArgv[current] = strdup("-l");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;

            daemonArgv[current] = strdup("SW");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;
        }
        else if (logging_ & STAT_LOG_SWERR)
        {
            daemonArgc += 2;
            daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
            if (daemonArgv == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
                return STAT_ALLOCATE_ERROR;
            }

            daemonArgv[current] = strdup("-l");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;

            daemonArgv[current] = strdup("SWERR");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;
        }

        if (logging_ & STAT_LOG_MRN)
        {
            daemonArgc += 3;
            daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
            if (daemonArgv == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
                return STAT_ALLOCATE_ERROR;
            }

            daemonArgv[current] = strdup("-o");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;

            daemonArgv[current] = (char *)malloc(8 * sizeof(char));
            snprintf(daemonArgv[current], 8, "%d", mrnetOutputLevel_);
            current++;

            daemonArgv[current] = strdup("-m");
            if (daemonArgv[current] == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), current);
                return STAT_ALLOCATE_ERROR;
            }
            current++;
        }

        daemonArgv[current] = NULL;
    }
    return STAT_OK;
}

void STAT_FrontEnd::setNDaemonsPerNode(int nDaemonsPerNode)
{
    nDaemonsPerNode_ = nDaemonsPerNode;
}

int STAT_FrontEnd::getNDaemonsPerNode()
{
    return nDaemonsPerNode_;
}


/**********************
 * STATBench Routines
**********************/


StatError_t STAT_FrontEnd::statBenchCreateStackTraces(unsigned int maxDepth, unsigned int nTasks, unsigned int nTraces, unsigned int functionFanout, int nEqClasses, unsigned int sampleType)
{
    int tag, intRet;
    unsigned int i, j;
    StatError_t statError;
    PacketPtr packet;

    addPerfData("STATBench Setup Time", -1.0);
    gTotalgStartTime.setTime();

    /* Rework the proc table to represent the STATBench emulated application */
    gStartTime.setTime();

    statError = STATBench_resetProctab(nTasks);
    if (statError != STAT_OK)
    {
        return statError;
    }

    gEndTime.setTime();
    addPerfData("\tProctable Creation Time", (gEndTime - gStartTime).getDoubleTime());

    /* Generate the daemon rank map */
    gStartTime.setTime();
    statError = createDaemonRankMap();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to create daemon rank map\n");
        return statError;
    }
    gEndTime.setTime();
    addPerfData("\tCreate Daemon Rank Map Time", (gEndTime - gStartTime).getDoubleTime());

    /* Update the application node list and daemon ranks lists */
    gStartTime.setTime();
    statError = setRanksList();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to set ranks list\n");
        return statError;
    }
    gEndTime.setTime();
    addPerfData("\tRanks List Creation Time", (gEndTime - gStartTime).getDoubleTime());
    gTotalgEndTime.setTime();
    addPerfData("\tTotal STATBench Setup Time", (gTotalgEndTime - gTotalgStartTime).getDoubleTime());
    gStartTime.setTime();

    if (isConnected_ == false)
    {
        printMsg(STAT_NOT_CONNECTED_ERROR, __FILE__, __LINE__, "STAT daemons have not been launched\n");
        return STAT_NOT_CONNECTED_ERROR;
    }

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Generating traces with parameters %d %d %d %d %d %u...\n", maxDepth, nTasks, nTraces, functionFanout, nEqClasses, sampleType);

    if (broadcastStream_->send(PROT_STATBENCH_CREATE_TRACES, "%ud %ud %ud %ud %d %ud", maxDepth, nTasks, nTraces, functionFanout, nEqClasses, sampleType) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to send request to create traces\n");
        return STAT_MRNET_ERROR;
    }
    if (broadcastStream_->flush() != 0)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to flush message\n");
        return STAT_MRNET_ERROR;
    }

    do
    {
        intRet = broadcastStream_->recv(&tag, packet, false);
        if (intRet == 0)
        {
            if (isKilled())
            {
                printMsg(STAT_APPLICATION_EXITED, __FILE__, __LINE__, "LMON detected the application has exited\n");
                return STAT_APPLICATION_EXITED;
            }
            if (daemonsHaveExited())
            {
                printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "LMON detected the daemons have exited\n");
                return STAT_DAEMON_ERROR;
            }
            usleep(1000);
        }
    } while (intRet == 0);
    if (intRet == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to receive ack\n");
        return STAT_MRNET_ERROR;
    }
    if (tag != PROT_STATBENCH_CREATE_TRACES_RESP)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Unexpected tag %d\n, expecting PROT_STATBENCH_CREATE_TRACES_RESP = %d\n", tag, PROT_STATBENCH_CREATE_TRACES_RESP);
    }

    if (packet->unpack("%d", &intRet) == -1)
    {
        printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to unpack acknowledgement packet\n");
        return STAT_MRNET_ERROR;
    }
    if (intRet != 0)
    {
        printMsg(STAT_SAMPLE_ERROR, __FILE__, __LINE__, "%d daemons reported an error\n", intRet);
        return STAT_SAMPLE_ERROR;
    }

    printMsg(STAT_STDOUT, __FILE__, __LINE__, "\t\tTraces generated!\n");

    gEndTime.setTime();
    addPerfData("Create Traces Time", (gEndTime - gStartTime).getDoubleTime());
    return STAT_OK;
}

#ifdef DYSECTAPI
StatError_t STAT_FrontEnd::dysectSetup(const char *dysectApiSessionPath, int dysectTimeout, int argc, char **argv)
{
    StatError_t statError;

    /* TODO: Refactoring work: Move sequence to Dysect::FE class */
    /* TODO: GUI will need to setenv("STAT_GROUP_OPS", "1", 1); prior to attach*/
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Setting up frontend session '%s'...\n", dysectApiSessionPath);
    dysectFrontEnd_ = new DysectAPI::FE((const char*)dysectApiSessionPath, this, dysectTimeout, argc, argv);
    if (dysectFrontEnd_->isLoaded() == false)
    {
        printMsg(STAT_DYSECT_ERROR, __FILE__, __LINE__, "Failed to load Dysect session %s\n", dysectApiSessionPath);
        return STAT_DYSECT_ERROR;
    }

    if (dysectFrontEnd_->requestBackendSetup((const char*)dysectApiSessionPath, argc, argv) != DysectAPI::OK)
    {
        printMsg(STAT_DYSECT_ERROR, __FILE__, __LINE__, "Failed to setup backends with session %s\n", dysectApiSessionPath);
        return STAT_DYSECT_ERROR;
    }
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Dysect session setup complete\n");

    statError = resume();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to resume application\n");
        return statError;
    }
    return STAT_OK;
}

StatError_t STAT_FrontEnd::dysectListen(bool blocking)
{
    DysectAPI::DysectErrorCode dysectError;

    if (dysectFrontEnd_ == NULL)
    {
        printMsg(STAT_DYSECT_ERROR, __FILE__, __LINE__, "Dysect not setup\n");
        return STAT_DYSECT_ERROR;
    }
    do
    {
        dysectError = dysectFrontEnd_->handleEvents(blocking);
    } while (dysectError == DysectAPI::SessionCont && blocking == true);
    if (dysectError == DysectAPI::OK)
        return STAT_OK;
    if (blocking == false)
    {
        if (dysectError == DysectAPI::SessionCont)
            return STAT_PENDING_ACK;
    }
    return STAT_DYSECT_ERROR;
}
StatError_t STAT_FrontEnd::dysectStop()
{
    dysectFrontEnd_->requestBackendShutdown();
    delete dysectFrontEnd_;
    return STAT_OK;
}
#endif
