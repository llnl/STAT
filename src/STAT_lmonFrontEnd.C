#include "STAT_lmonFrontEnd.h"

using namespace std;
using namespace MRN;

//! the LMON status bit vector to detect daemon and application exit
static int gsLmonState = 0;

// start timer for composite operations
extern STAT_timer gTotalgStartTime;

// start timer for individual operations
extern STAT_timer gStartTime;

// end timer for individual operations
extern STAT_timer gEndTime;


STAT_lmonFrontEnd::STAT_lmonFrontEnd()
    : lmonSession_(-1), proctab_(nullptr), proctabSize_(0)
{
    char *envValue = getenv("STAT_LMON_REMOTE_LOGIN");
    if (envValue != NULL)
        setenv("LMON_REMOTE_LOGIN", envValue, 1);

    /* Set LMON_DEBUG_BES */
    envValue = getenv("STAT_LMON_DEBUG_BES");
    if (envValue != NULL)
        setenv("LMON_DEBUG_BES", envValue, 1);

    /* Set the launchmon and mrnet_commnode paths based on STAT env vars */
    envValue = getenv("STAT_LMON_PREFIX");
    if (envValue != NULL)
        setenv("LMON_PREFIX", envValue, 1);
    envValue = getenv("STAT_LMON_LAUNCHMON_ENGINE_PATH");
    if (envValue != NULL)
        setenv("LMON_LAUNCHMON_ENGINE_PATH", envValue, 1);
    envValue = getenv("STAT_LMON_NEWLAUNCHMON_ENGINE_PATH");
    if (envValue != NULL)
        setenv("LMON_NEWLAUNCHMON_ENGINE_PATH", envValue, 1);
}

STAT_lmonFrontEnd::~STAT_lmonFrontEnd()
{
    unsigned int i;
    if (proctab_ != NULL)
    {
        for (i = 0; i < proctabSize_; i++)
        {
            if (proctab_[i].pd.executable_name != NULL)
                free(proctab_[i].pd.executable_name);
            if (proctab_[i].pd.host_name != NULL)
                free(proctab_[i].pd.host_name);
        }
        free(proctab_);
        proctab_ = NULL;
    }
}

StatError_t STAT_lmonFrontEnd::setupForSerialAttach()
{
#ifdef BGL
    printMsg(STAT_WARNING, __FILE__, __LINE__, "Serial attach not supported on Bluegene systems\n");
    return STAT_WARNING;
#endif
    printMsg(STAT_STDOUT, __FILE__, __LINE__, "Setting up environment for serial attach...\n");
    gsLmonState = gsLmonState | 0x00000002;
    nApplProcs_ = proctabSize_;

    return launchDaemons();
}

StatError_t STAT_lmonFrontEnd::launchDaemons()
{
    int i, daemonArgc = 1;
    unsigned int proctabSize, j;
    char **daemonArgv = NULL, tempString[BUFSIZE];
    static bool iIsFirstRun = true;
    string applName;
    set<string> exeNames;
    lmon_rc_e lmonRet;
    lmon_rm_info_t rmInfo;
    StatError_t statError;

    /* Initialize performance timer */
    addPerfData("MRNet Launch Time", -1.0);
    gTotalgStartTime.setTime();
    gStartTime.setTime();

    /* Increase the max proc and max fd limits for MRNet threads */
#if (defined(HAVE_GETRLIMIT) && defined(HAVE_SETRLIMIT))
    statError = increaseSysLimits();
    if (statError != STAT_OK)
        printMsg(statError, __FILE__, __LINE__, "Failed to increase limits... attemting to run with current configuration\n");
#endif

#ifdef STAT_GDB_BE
        if (applicationOption_ == STAT_GDB_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
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


    if (applicationOption_ != STAT_SERIAL_ATTACH && applicationOption_ != STAT_SERIAL_GDB_ATTACH)
    {
        if (iIsFirstRun == true)
        {
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Initializing LaunchMON\n");
            lmonRet = LMON_fe_init(LMON_VERSION);
            if (lmonRet != LMON_OK)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to initialize Launchmon\n");
                return STAT_LMON_ERROR;
            }
        }
        iIsFirstRun = false;

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Creating LaunchMON session\n");
        lmonRet = LMON_fe_createSession(&lmonSession_);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to create Launchmon session\n");
            return STAT_LMON_ERROR;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Registering pack function with LaunchMON\n");
        lmonRet = LMON_fe_regPackForFeToBe(lmonSession_, statPack);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to register pack function\n");
            return STAT_LMON_ERROR;
        }

        gsLmonState = 0;
        if (isStatBench_ == false)
        {
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Registering status CB function with LaunchMON\n");
            lmonRet = LMON_fe_regStatusCB(lmonSession_, lmonStatusCb);
            if (lmonRet != LMON_OK)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to register status CB function\n");
                return STAT_LMON_ERROR;
            }
        }
        else
            gsLmonState = gsLmonState | 0x00000002;

        if (toolDaemonExe_ == NULL)
        {
            printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Tool daemon path not set\n");
            return STAT_ARG_ERROR;
        }
        daemonArgc += 2;
        daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
        if (daemonArgv == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s malloc failed to allocate for daemon argv\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
        daemonArgv[daemonArgc - 3] = strdup("-d");
        snprintf(tempString, BUFSIZE, "%d", nDaemonsPerNode_);
        daemonArgv[daemonArgc - 2] = strdup(tempString);
        daemonArgv[daemonArgc - 1] = NULL;

        statError = addDaemonLogArgs(daemonArgc, daemonArgv);
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to add daemon logging args\n");
            return statError;
        }
        for (i = 0; i < daemonArgc; i++)
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "daemonArgv[%d] = %s\n", i, daemonArgv[i]);

        if (applicationOption_ == STAT_ATTACH || applicationOption_ == STAT_GDB_ATTACH)
        {
            if (launcherPid_ == 0)
            {
                printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Launcher PID not set\n");
                return STAT_ARG_ERROR;
            }

            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Attaching to job launcher and spawning daemons\n");
            lmonRet = LMON_fe_attachAndSpawnDaemons(lmonSession_, remoteNode_, launcherPid_, toolDaemonExe_, daemonArgv, NULL, NULL);
            if (lmonRet != LMON_OK)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to attach to job launcher and spawn daemons\n");
                return STAT_LMON_ERROR;
            }
        }
        else
        {
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launching the application and spawning daemons\n");
            i = 0;
            while (1)
            {
                if (launcherArgv_[i] == NULL)
                    break;
                printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launcher argv[%d] = %s\n", i, launcherArgv_[i]);
                i++;
            }
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "remote node = %s, daemon = %s\n", remoteNode_, toolDaemonExe_);
            lmonRet = LMON_fe_launchAndSpawnDaemons(lmonSession_, remoteNode_, launcherArgv_[0], launcherArgv_, toolDaemonExe_, daemonArgv, NULL, NULL);
            if (lmonRet != LMON_OK)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to launch job and spawn daemons\n");
                return STAT_LMON_ERROR;
            }

            lmonRet = LMON_fe_getRMInfo(lmonSession_, &rmInfo);
            if (lmonRet != LMON_OK)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get resource manager info\n");
                return STAT_LMON_ERROR;
            }
            launcherPid_ = rmInfo.rm_launcher_pid;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Gathering the process table\n");
        lmonRet = LMON_fe_getProctableSize(lmonSession_, &proctabSize_);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get Process Table Size\n");
            return STAT_LMON_ERROR;
        }
        proctab_ = (MPIR_PROCDESC_EXT *)malloc(proctabSize_ * sizeof(MPIR_PROCDESC_EXT));
        if (proctab_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to allocate memory for the process table\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
        lmonRet = LMON_fe_getProctable(lmonSession_, proctab_, &proctabSize, proctabSize_);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get Process Table\n");
            return STAT_LMON_ERROR;
        }

        /* Get the resource manager information from LaunchMON */
        lmonRet = LMON_fe_getRMInfo(lmonSession_, &lmonRmInfo_);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get RM Info\n");
            return STAT_LMON_ERROR;
        }
    } /* if (applicationOption_ != STAT_SERIAL_ATTACH && applicationOption_ != STAT_SERIAL_GDB_ATTACH) */

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Gathering application information\n");

    /* Iterate over all unique exe names and concatenate to app name */
    for (j = 0; j < proctabSize_; j++)
    {
        if (exeNames.find(proctab_[j].pd.executable_name) != exeNames.end())
            continue;
        exeNames.insert(proctab_[j].pd.executable_name);
        if (j > 0)
            applName += "_";
        applName += basename(proctab_[j].pd.executable_name);
    }
    applExe_ = strdup(applName.c_str());
    if (applExe_ == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to set applExe_ with strdup()\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }
    nApplProcs_ = proctabSize_;
    isLaunched_ = true;
    gEndTime.setTime();
    addPerfData("\tLaunchmon Launch Time", (gEndTime - gStartTime).getDoubleTime());
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tTool daemons launched\n");

    if (strcmp(outDir_, "NULL") == 0 || strcmp(filePrefix_, "NULL") == 0)
    {
        statError = createOutputDir();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to create output directory\n");
            return statError;
        }
    }

    statError = dumpProctab();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to dump process table to a file\n");
        return statError;
    }

    /* Set the application node list based on the process table */
    printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "\tPopulating application node list\n");
    gStartTime.setTime();
    if (isStatBench_ == true)
        statError = STATBench_setAppNodeList();
    else
        statError = setAppNodeList();
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to set app node list\n");
        return statError;
    }
    gEndTime.setTime();
    addPerfData("\tApp Node List Creation Time", (gEndTime - gStartTime).getDoubleTime());

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

    return STAT_OK;
}

StatError_t STAT_lmonFrontEnd::sendDaemonInfo()
{
    lmon_rc_e lmonRet;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Sending MRNet connection info to daemons\n");

    lmonRet = LMON_fe_sendUsrDataBe(lmonSession_, (void *)&leafInfo_);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to send data to BE\n");
        return STAT_LMON_ERROR;
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "MRNet connection info successfully sent to daemons\n");

    return STAT_OK;
}

StatError_t STAT_lmonFrontEnd::addDaemonSerialProcArgs(int& daemonArgc, char ** &daemonArgv)
{
    int statArgc = 1 + 2 * proctabSize_, currentArg;
    unsigned int j;
    char temp[BUFSIZE];
    
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\tSetting up daemon args for serial attach and MRNet launch\n");
    daemonArgc += statArgc;
    daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
    if (daemonArgv == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to allocate %d bytes for argv\n", strerror(errno), daemonArgc);
        return STAT_ALLOCATE_ERROR;
    }
    daemonArgv[daemonArgc - statArgc - 1] = strdup("-s");
    if (daemonArgv[0] == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[0]\n", strerror(errno));
        free(daemonArgv);
        return STAT_ALLOCATE_ERROR;
    }
    for (j = 0; j < proctabSize_; j++)
    {
        currentArg = daemonArgc - statArgc - 1 + 2 * j + 1;
        daemonArgv[currentArg] = strdup("-p");
        if (daemonArgv[currentArg] == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup argv[%d]\n", strerror(errno), currentArg);
            free(daemonArgv);
            return STAT_ALLOCATE_ERROR;
        }
        snprintf(temp, BUFSIZE, "%s@%s:%d", proctab_[j].pd.executable_name, proctab_[j].pd.host_name, proctab_[j].pd.pid);
        currentArg++;
        daemonArgv[currentArg] = strdup(temp);
        if (daemonArgv[currentArg] == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) argv[%d]\n", strerror(errno), temp, currentArg);
            free(daemonArgv);
            return STAT_ALLOCATE_ERROR;
        }
    }

    daemonArgv[daemonArgc - 1] = NULL;

    return STAT_OK;
}

bool STAT_lmonFrontEnd::daemonsHaveExited()
{
    return !WIFBESPAWNED(gsLmonState);
}

bool STAT_lmonFrontEnd::isKilled()
{
    return WIFKILLED(gsLmonState);
}

void STAT_lmonFrontEnd::detachFromLauncher(const char *errMsg)
{
    lmon_rc_e lmonRet = LMON_fe_detach(lmonSession_);
    if (lmonRet != LMON_OK)
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, errMsg);
}

StatError_t STAT_lmonFrontEnd::dumpProctab()
{
    unsigned int i;
    FILE *file;
    char fileName[BUFSIZE];

    snprintf(fileName, BUFSIZE, "%s/%s.ptab", outDir_, filePrefix_);
    file = fopen(fileName, "w");
    if (file == NULL)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to create ptab file %s\n", strerror(errno), fileName);
        return STAT_FILE_ERROR;
    }

    /* Write the job launcher host and PID */
    if (remoteNode_ != NULL)
        fprintf(file, "%s:%d\n", remoteNode_, launcherPid_);
    else
        fprintf(file, "%s:%d\n", hostname_, launcherPid_);

    /* Write out the MPI ranks */
    if (verbose_ == STAT_VERBOSE_FULL || logging_ & STAT_LOG_FE)
        printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "Process Table:\n");
    for (i = 0; i < nApplProcs_; i++)
    {
        fprintf(file, "%d %s:%d %s\n", proctab_[i].mpirank, proctab_[i].pd.host_name, proctab_[i].pd.pid, proctab_[i].pd.executable_name);
        if (verbose_ == STAT_VERBOSE_FULL || logging_ & STAT_LOG_FE)
            printMsg(STAT_VERBOSITY, __FILE__, __LINE__, "%s: %s, pid: %d, MPI rank: %d\n", proctab_[i].pd.host_name, proctab_[i].pd.executable_name, proctab_[i].pd.pid, proctab_[i].mpirank);
    }

    fclose(file);
    return STAT_OK;
}

StatError_t STAT_lmonFrontEnd::setAppNodeList()
{
    unsigned int i;
    char *currentNode;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating application node list: ");
    applicationNodeMultiSet_.clear();
    for (i = 0; i < nApplProcs_; i++)
    {
        currentNode = proctab_[i].pd.host_name;
        if (applicationNodeMultiSet_.find(currentNode) == applicationNodeMultiSet_.end())
        {
            applicationNodeMultiSet_.insert(currentNode);
            printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "%s, ", currentNode);
        }
    }
    printMsg(STAT_LOG_MESSAGE, __FILE__, -1, "\n");
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Application node list created\n");
    nApplNodes_ = applicationNodeMultiSet_.size();

    return STAT_OK;
}

StatError_t STAT_lmonFrontEnd::STATBench_setAppNodeList()
{
    unsigned int i;
    char *currentNode;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating application node list\n");

    applicationNodeMultiSet_.clear();
    for (i = 0; i < nApplProcs_; i++)
    {
        currentNode = proctab_[i].pd.host_name;
        applicationNodeMultiSet_.insert(currentNode);
    }
    nApplNodes_ = nApplProcs_;

    return STAT_OK;
}

StatError_t STAT_lmonFrontEnd::STATBench_resetProctab(unsigned int nTasks)
{
    unsigned int i, j;
    char name[BUFSIZE];

    for (int i = 0; i < proctabSize_; i++)
    {
        if (proctab_[i].pd.executable_name != NULL)
            free(proctab_[i].pd.executable_name);
        if (proctab_[i].pd.host_name != NULL)
            free(proctab_[i].pd.host_name);
    }
    free(proctab_);
    proctab_ = NULL;

    nApplProcs_ = nApplNodes_ * nTasks;
    proctabSize_ = nApplProcs_;
    proctab_ = (MPIR_PROCDESC_EXT *)malloc(nApplNodes_ * nTasks * sizeof(MPIR_PROCDESC_EXT));
    if (proctab_ == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to create proctable\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }

    for (i = 0; i < nApplNodes_; i++)
    {
        snprintf(name, BUFSIZE, "a%d", i);
        for (j = 0; j < nTasks; j++)
        {
            proctab_[i * nTasks + j].mpirank = i * nTasks + j;
            proctab_[i * nTasks + j].pd.host_name = strdup(name);
            if (proctab_[i * nTasks + j].pd.host_name == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to strdup(%s) for proctab_[%d]\n", strerror(errno), name, i * nTasks + j);
                return STAT_ALLOCATE_ERROR;
            }
            proctab_[i * nTasks + j].pd.executable_name = NULL;
        }
    }

    return STAT_OK;
}



bool checkAppExit()
{
    return WIFKILLED(gsLmonState);
}

bool checkDaemonExit()
{
    return !WIFBESPAWNED(gsLmonState);
}

int lmonStatusCb(int *status)
{
    gsLmonState = *status;
    return 0;
}

StatError_t STAT_lmonFrontEnd::createMRNetNetwork(const char *topologyFileName)
{
    network_ = Network::CreateNetworkFE(topologyFileName, NULL, NULL);

    return STAT_OK;
}

void STAT_lmonFrontEnd::shutDown()
{
    int killed, detached;
    StatError_t statError;
    lmon_rc_e lmonRet;

    statError = dumpPerf();
    if (statError != STAT_OK)
        printMsg(statError, __FILE__, __LINE__, "Failed to dump performance results\n");

    if (network_ != NULL && isConnected_ == true)
        shutdownMrnetTree();

    killed = WIFKILLED(gsLmonState);
    detached = !WIFBESPAWNED(gsLmonState);
    if (isLaunched_ == true && killed == 0 && detached == 0)
    {
        if (lmonSession_ != -1)
        {
            lmonRet = LMON_fe_detach(lmonSession_);
            if (lmonRet != LMON_OK)
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Launchmon failed to detach from launcher... have the daemons exited?\n");
        }
    }
    isLaunched_ = false;
    gsLmonState = gsLmonState &(~0x00000002);
}

StatError_t STAT_lmonFrontEnd::addSerialProcess(const char *pidString)
{
    unsigned int pid;
    char *remoteNode = NULL;
    string::size_type delimPos;
    string remotePid = pidString, remoteHost;

    proctabSize_++;
    proctab_ = (MPIR_PROCDESC_EXT *)realloc(proctab_, proctabSize_ * sizeof(MPIR_PROCDESC_EXT));
    if (proctab_ == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Failed to allocate memory for the process table\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }

    delimPos = remotePid.find_first_of("@");
    if (delimPos != string::npos)
    {
        proctab_[proctabSize_ - 1].pd.executable_name = strdup(remotePid.substr(0, delimPos).c_str());
        if (proctab_[proctabSize_ - 1].pd.executable_name == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to set executable_name from %s\n", strerror(errno), remotePid.substr(0, delimPos).c_str());
            return STAT_ALLOCATE_ERROR;
        }
        remotePid = remotePid.substr(delimPos + 1, remotePid.length());
    }
    else
    {
        proctab_[proctabSize_ - 1].pd.executable_name = strdup("output");
        if (proctab_[proctabSize_ - 1].pd.executable_name == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to set executable_name\n", strerror(errno));
            return STAT_ALLOCATE_ERROR;
        }
    }

    delimPos = remotePid.find_first_of(":");
    if (delimPos != string::npos)
    {
        remoteHost = remotePid.substr(0, delimPos);
        remoteNode = strdup(remoteHost.c_str());
        if (remoteNode == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup remote node from %s\n", strerror(errno), remotePid.c_str());
            return STAT_ALLOCATE_ERROR;
        }
        pid = atoi((remotePid.substr(delimPos + 1, remotePid.length())).c_str());
    }
    else
        pid = atoi(remotePid.c_str());

    proctab_[proctabSize_ - 1].mpirank = proctabSize_ - 1;
    if (remoteNode != NULL)
    {
        if (strcmp(remoteNode, "localhost") == 0 || strcmp(remoteNode, "") == 0)
        {
            proctab_[proctabSize_ - 1].pd.host_name = strdup(hostname_);
            if (proctab_[proctabSize_ - 1].pd.host_name == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) to host_name\n", strerror(errno), hostname_);
                free(remoteNode);
                return STAT_ALLOCATE_ERROR;
            }
        }
        else
            proctab_[proctabSize_ - 1].pd.host_name = strdup(remoteNode);
        free(remoteNode);
    }
    else
    {
        proctab_[proctabSize_ - 1].pd.host_name = strdup(hostname_);
        if (proctab_[proctabSize_ - 1].pd.host_name == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed to strdup(%s) to host_name\n", strerror(errno), hostname_);
            return STAT_ALLOCATE_ERROR;
        }
    }
    proctab_[proctabSize_ - 1].pd.pid = pid;
    return STAT_OK;
}

bool STAT_lmonFrontEnd::checkNodeAccess(const char *node)
{
    /* MRNet CPs launched through alps */
    if (lmonRmInfo_.rm_supported_types != NULL)
    {
        if (lmonRmInfo_.rm_supported_types[lmonRmInfo_.index_to_cur_instance] == RC_alps)
            return true;
    }
    char command[BUFSIZE], testOutput[BUFSIZE], runScript[BUFSIZE], checkHost[BUFSIZE], *rsh = NULL, *envValue;
    FILE *output, *temp;

    /* TODO: this is generally not good... a quick check for clusters where each node has the same hostname prefix */
    /* A first level filter, compare the first 3 letters */
    gethostname(checkHost, BUFSIZE);
    if (strncmp(node, checkHost, 3) != 0 && strcmp(node, "localhost") != 0)
        return false;

    envValue = getenv("XPLAT_RSH");
    if (envValue != NULL)
        rsh = strdup(envValue);
    if (rsh == NULL)
    {
        rsh = strdup("rsh");
        if (rsh == NULL)
        {
             printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: failed to set rsh command\n", strerror(errno));
            return false;
        }
    }

    /* Use run_one_sec script if available to tolerate rsh slowness/hangs */
    snprintf(runScript, BUFSIZE, "%s/bin/run_one_sec", getInstallPrefix());
    temp = fopen(runScript, "r");
    if (temp == NULL)
        snprintf(command, BUFSIZE, "%s %s hostname 2>1", rsh, node);
    else
    {
        snprintf(command, BUFSIZE, "%s %s %s hostname 2>1", runScript, rsh, node);
        fclose(temp);
    }
    output = popen(command, "r");
    free(rsh);
    if (output == NULL)
        return false;
    fgets(testOutput, sizeof(testOutput), output);
    pclose(output);
    if (strcmp(testOutput, "") == 0)
        return false;
    return true;
}


int STAT_lmonFrontEnd::getNumProcs()
{
    return proctabSize_;
}

const char *STAT_lmonFrontEnd::getHostnameForProc(int procIdx)
{
    return proctab_[procIdx].pd.host_name;
}
int STAT_lmonFrontEnd::getMpiRankForProc(int procIdx)
{
    return proctab_[procIdx].mpirank;
}

#ifndef USE_CTI
STAT_FrontEnd *STAT_FrontEnd::make()
{
    return new STAT_lmonFrontEnd();
}
#endif

