#include "STAT_ctiFrontEnd.h"
#include <fstream>

#define ctiError()   (printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "CTI Error: %s\n", cti_error_str()),STAT_SYSTEM_ERROR)

STAT_ctiFrontEnd::STAT_ctiFrontEnd() : appId_(0), session_(0), hosts_(nullptr),
                                       tasksPerPE_(1)
{
    std::string temp;
    if (XPlat::NetUtils::GetLocalHostName(temp) == 0)
        snprintf(hostname_, BUFSIZE, "%s", temp.c_str());

    // Ensure CTI can initialize
    if (auto hostname = cti_getHostname())
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "CTI initialized with hostname %s\n", hostname);
        free(hostname);
    }
    else
    {
        ctiError();
        exit(STAT_SYSTEM_ERROR);
    }
}

STAT_ctiFrontEnd::~STAT_ctiFrontEnd()
{
    if (hosts_)
        cti_destroyHostsList(hosts_);

    if (appId_ != 0 && cti_appIsValid(appId_))
        cti_deregisterApp(appId_);
}

StatError_t STAT_ctiFrontEnd::setupForSerialAttach()
{
    printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Serial launch is not supported in CTI\n");
    return STAT_ARG_ERROR;
}

StatError_t STAT_ctiFrontEnd::attach()
{
    // register the app with CTI
    void* vops = nullptr;
    auto wlm = cti_open_ops(&vops);

    if (!vops)
        return ctiError();

    switch (wlm)
    {
    case CTI_WLM_SLURM:
    {
        auto ops = static_cast<cti_slurm_ops_t *>(vops);
        cti_srunProc_t *srunInfo = ops->getJobInfo(launcherPid_);
        if (!srunInfo)
            return ctiError();

        appId_ = ops->registerJobStep(srunInfo->jobid, srunInfo->stepid);
        free(srunInfo);

        if (!appId_)
            return ctiError();

        break;
    }
    case CTI_WLM_SSH:
    {
        auto ops = static_cast<cti_ssh_ops_t *>(vops);
        appId_ = ops->registerLauncherPid((pid_t)launcherPid_);
        if (!appId_)
        {
            return ctiError();
        }
        break;
    }
    case CTI_WLM_ALPS:
    {
        auto ops = static_cast<cti_alps_ops_t *>(vops);
        uint64_t apid = ops->getApid((pid_t)launcherPid_);
        if (apid == 0)
            return ctiError();
        appId_ = ops->registerApid(apid);
        if (appId_ == 0)
            return ctiError();
        break;
    }

    case CTI_WLM_PALS:
    {
        auto ops = static_cast<cti_pals_ops_t*>(vops);

        char* apid = ops->getApid((pid_t)launcherPid_);
        if (!apid)
            return ctiError();

        appId_ = ops->registerApid(apid);

        free(apid);
        if (!appId_)
            return ctiError();
        break;
    }

    case CTI_WLM_FLUX:
    {
        auto ops = static_cast<cti_flux_ops_t*>(vops);

        char* jobid = ops->getJobid((pid_t)launcherPid_);
        if (!jobid)
            return ctiError();

        appId_ = ops->registerJob(jobid);

        free(jobid);
        if (!appId_)
            return ctiError();
        break;
    }

    default:
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "Unsupported Cray WLM!\n");
        return STAT_SYSTEM_ERROR;
    }
    return STAT_OK;
}

StatError_t STAT_ctiFrontEnd::launch()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launching application with CTI\n");

    // launcherArgv_ is null terminated.   But the stat documentation tells you to include
    // the launcher in the argument list, so I'll ignore the first one.
    if (launcherArgc_ < 3)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "No application given for launch\n");
        return STAT_ARG_ERROR;
    }

    const char* env[] = { nullptr };
    appId_ = cti_launchAppBarrier(launcherArgv_+1, -1, -1, nullptr, nullptr, env);
    //appId_ = cti_launchApp(launcherArgv_+1, -1, -1, nullptr, nullptr, env);
    if (!appId_)
        return ctiError();

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Application launched successfully\n");
    return STAT_OK;
}

StatError_t STAT_ctiFrontEnd::postAttachApplication()
{
    if (applicationOption_ == STAT_LAUNCH && appId_)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Releasing app barrier\n");
        if (cti_releaseAppBarrier(appId_))
            return ctiError();
        //printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "No Releasing app barrier\n");
    }
    return STAT_FrontEnd::postAttachApplication();
}


// Launch a back-end process on every node where the application is running. We don't
// yet have an MRNet network, so the one link we'll have available is the back-end nodes
// can access files the front end ships in the CTI manifest.
StatError_t STAT_ctiFrontEnd::launchDaemons()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launching daemons with CTI\n");
    
    if (!toolDaemonExe_)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Tool daemon path not set\n");
        return STAT_ARG_ERROR;
    }

    if (applicationOption_ == STAT_SERIAL_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Serial launch is not supported in CTI\n");
        return STAT_ARG_ERROR;
    }

    // initialize cti app id
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Attaching to job\n");

    StatError_t statError = STAT_OK;
    if (applicationOption_ == STAT_ATTACH || applicationOption_ == STAT_GDB_ATTACH)
    {
        statError = attach();
        if (statError != STAT_OK)
            return statError;

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Attached to job\n");

    }
    else if (applicationOption_ == STAT_LAUNCH)
    {
        statError = launch();
        if (statError != STAT_OK)
        {
            return statError;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launched the job\n");
    }
    else
    {
        printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Launch option %d it not supported by CTI\n",
                 applicationOption_);
        return STAT_ARG_ERROR;
    }
    

    /* Increase the max proc and max fd limits for MRNet threads */

#if (defined(HAVE_GETRLIMIT) && defined(HAVE_SETRLIMIT))
    statError = increaseSysLimits();
    if (statError != STAT_OK)
        printMsg(statError, __FILE__, __LINE__, "Failed to increase limits... attempting to run with current configuration\n");
#endif

    int daemonArgc = 1;
    char **daemonArgv = nullptr;

    if (applicationOption_ == STAT_GDB_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
    {
        const char* pythonPath = getenv("PYTHONPATH");
        if (!pythonPath)
            pythonPath = ":";

        const char* gdbCommand = getenv("STAT_GDB");
        if (!gdbCommand)
            gdbCommand = "gdb";

        daemonArgc += 4;
        daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
        daemonArgv[daemonArgc - 5] = strdup("-P");
        daemonArgv[daemonArgc - 4] = strdup(pythonPath);
        daemonArgv[daemonArgc - 3] = strdup("-G");
        daemonArgv[daemonArgc - 2] = strdup(gdbCommand);
        daemonArgv[daemonArgc - 1] = NULL;
    }

    daemonArgc += 2;
    daemonArgv = (char **)realloc(daemonArgv, daemonArgc * sizeof(char *));
    daemonArgv[daemonArgc - 3] = strdup("-d");
    char tempString[BUFSIZE];
    snprintf(tempString, BUFSIZE, "%d", nDaemonsPerNode_);
    daemonArgv[daemonArgc - 2] = strdup(tempString);
    daemonArgv[daemonArgc - 1] = NULL;

    statError = addDaemonLogArgs(daemonArgc, daemonArgv);
    if (statError != STAT_OK)
    {
        printMsg(statError, __FILE__, __LINE__, "Failed to add daemon logging args\n");
        return statError;
    }

    for (int i = 0; i < daemonArgc; i++)
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "daemonArgv[%d] = %s\n", i, daemonArgv[i]);

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Create CTI session\n");

    session_ = cti_createSession(appId_);
    if (!session_)
        return ctiError();

    cti_manifest_id_t manifest = cti_createManifest(session_);
    if (!manifest)
        return ctiError();

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Adding binary %s to manifest\n", toolDaemonExe_);
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "LD_LIBRARY_PATH=%s\n", getenv("LD_LIBRARY_PATH"));
    
    if (cti_addManifestBinary(manifest, toolDaemonExe_))
        return ctiError();

    if (cti_addManifestLibrary(manifest, filterPath_))
        return ctiError();

    if (applicationOption_ == STAT_GDB_ATTACH || applicationOption_ == STAT_SERIAL_GDB_ATTACH)
    {
        char* cpFilterPath = strdup(filterPath_);
        char* libDir = dirname(cpFilterPath);
        std::string cudaLib;
        std::string gdbLib;

        const char* pythonPath = getenv("PYTHONPATH");
        if (!pythonPath)
            return ctiError();

        std::string python_path = pythonPath;
        if (python_path.find("python3.6") != std::string::npos)
        {
          cudaLib = std::string(libDir) + "/python3.6/site-packages/cuda_gdb.py";
          gdbLib  = std::string(libDir) + "/python3.6/site-packages/gdb.py";
        }
        else if (python_path.find("python3.9") != std::string::npos)
        {
          cudaLib = std::string(libDir) + "/python3.9/site-packages/cuda_gdb.py";
          gdbLib  = std::string(libDir) + "/python3.9/site-packages/gdb.py";
        }
        free(cpFilterPath);

        if (cti_addManifestFile(manifest, cudaLib.c_str()))
            return ctiError();
        if (cti_addManifestFile(manifest, gdbLib.c_str()))
            return ctiError();
    }

    char* cpExe = strdup(toolDaemonExe_);
    char* toolExe = basename(cpExe);

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launching tool %s.\n", toolExe);

    if (cti_execToolDaemon(manifest, toolExe, daemonArgv, nullptr))
    {
        free(cpExe);
        return ctiError();
    }

    free(cpExe);

    isLaunched_ = true;

    statError = getProcInfo();
    if (statError != STAT_OK)
        return statError;

    if (strcmp(outDir_, "NULL") == 0 || strcmp(filePrefix_, "NULL") == 0)
    {
        statError = createOutputDir();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to create output directory\n");
            return statError;
        }
    }

    return STAT_OK;
}

// Assemble the back-end process information which will include the rank to node mapping and
// and a the label based on the name of the executable(s).
StatError_t STAT_ctiFrontEnd::getProcInfo()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Getting proc info and applExe from CTI.\n");

    nApplProcs_ = cti_getNumAppPEs(appId_);
    if (!nApplProcs_)
        return ctiError();

    hosts_ = cti_getAppHostsPlacement(appId_);
    if (!hosts_)
        return ctiError();

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Done.\n");

    nApplNodes_ = hosts_->numHosts;
    procToHost_.reserve(nApplNodes_);

    // Note: the CTI implementation is not currently trying to track the real MPI rank.  It's just
    // assigning a rank value based based on the order of the backend nodes.   We could add a little
    // protocol to fix up the bookkeeping after the backends are connected to MRNet, but I'm not
    // sure it's really worth it.
    int totNumPEs = 0;
    for (int i=0, totNumPEs=0; i<nApplNodes_; ++i)
    {
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "host %s has %d pes\n",
                 hosts_->hosts[i].hostname, hosts_->hosts[i].numPes);
                 
        totNumPEs += hosts_->hosts[i].numPes;
        procToHost_.push_back(totNumPEs);
    }

    std::string applName;
    if (cti_binaryList_t* binList = cti_getAppBinaryList(appId_))
    {
        for (char** binIt = binList->binaries; *binIt; ++binIt)
        {
            char* bin = *binIt;
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "binary = %s\n", bin);

            if (!applName.empty())
                applName += "_";
            applName += basename(bin);
        }
        
        cti_destroyBinaryList(binList);
    }

    if (applName.empty())
    {
        printMsg(STAT_CTI_ERROR, __FILE__, __LINE__, "did not get application name\n");
        applName = "unknown";
    }

    applExe_ = strdup(applName.c_str());

    return STAT_OK;
}

StatError_t STAT_ctiFrontEnd::sendDaemonInfo()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "sending daemon info\n");
    
    // the backend processes can't connect to mrnet without the parent information, so
    // we need to ship enough information via the cti manifest to infer the connectivity
    if (leafInfo_.leafCps.empty())
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "MRNet tree was not created\n");
        return STAT_SYSTEM_ERROR;
    }

    std::string connectFileDir = std::string(outDir_) + "/" + filePrefix_ + ".daemons";
    if (mkdir(connectFileDir.c_str(), S_IRWXU))
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "could not create directory %s\n",
                 connectFileDir.c_str());
        return STAT_SYSTEM_ERROR;
    }

    std::string connectFile = connectFileDir + "/daemoninfo.txt";
    std::ofstream str(connectFile.c_str());
    if (!str)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "could not create file %s\n",
                 connectFile.c_str());
        return STAT_SYSTEM_ERROR;
    }


    int numParents = leafInfo_.leafCps.size();
    int numHosts = hosts_->numHosts;

    // print out the hosts to parent index info
    str << numHosts << "\n";
    for (int i=0; i<numHosts; ++i)
    {
        auto host = hosts_->hosts[i];
        int rank = i + topologySize_;
        int parentIdx = (numParents * i) / numHosts;
        str << host.hostname << " " << rank << " " << parentIdx << "\n";
    }


    // mrnet parent nodes
    str << leafInfo_.leafCps.size() << "\n";

    for ( auto node : leafInfo_.leafCps)
    {
        str << node->get_HostName() << " " << node->get_Port() << " " << node->get_Rank() << "\n";
    }

    if (!str)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "writing daemon info file %s failed\n",
                 connectFile.c_str());
        return STAT_SYSTEM_ERROR;
    }
    str.close();

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "sending manifest with file %s\n",
             connectFile.c_str());
    
    // and ship the file.
    cti_manifest_id_t manifest = cti_createManifest(session_);
    if (!manifest)
        return ctiError();

    if (cti_addManifestFile(manifest, connectFile.c_str()))
        return ctiError();

    if (cti_sendManifest(manifest))
        return ctiError();

    return STAT_OK;
}

StatError_t STAT_ctiFrontEnd::createMRNetNetwork(const char* topologyFileName)
{
    cti_manifest_id_t manifest = cti_createManifest(session_);

    std::map<std::string, std::string> attrs;
    attrs["CRAY_CTI_APPID"] = std::to_string(appId_);
    attrs["CRAY_CTI_MID"] = std::to_string(manifest);
    
    // the filter should have already been sent in the manifest
    network_ = MRN::Network::CreateNetworkFE(topologyFileName, NULL, NULL, &attrs);
    return STAT_OK;
}


void STAT_ctiFrontEnd::detachFromLauncher(const char* errMsg)
{
    if (session_)
    {
        if (cti_destroySession(session_))
            printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "Detach failed %s\n", errMsg);
        else
            session_ = 0;
    }
}

void STAT_ctiFrontEnd::shutDown()
{
    
    if (network_ != NULL && isConnected_ == true)
        shutdownMrnetTree();

    detachFromLauncher("CTI failed to detach from launcher...\n");
    isLaunched_ = false;
}

bool STAT_ctiFrontEnd::daemonsHaveExited()
{
    return !cti_appIsValid(appId_);
}
bool STAT_ctiFrontEnd::isKilled()
{
    return !cti_appIsValid(appId_);
}

int STAT_ctiFrontEnd::getNumProcs()
{
    return nApplProcs_;
}
const char* STAT_ctiFrontEnd::getHostnameForProc(int procIdx)
{
    if (!hosts_)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "proc table is not initialized\n");
        return "invalid process";
    }

    // for STATBench, 
    procIdx /= tasksPerPE_;     

    if (procIdx < 0 || procIdx >= nApplProcs_)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "invalid procIdx in getHostnameForProc\n");
        return "invalid process";
    }

    auto hostIdx = std::upper_bound(procToHost_.begin(), procToHost_.end(), procIdx) -
        procToHost_.begin();

    return hosts_->hosts[hostIdx].hostname;
};

int STAT_ctiFrontEnd::getMpiRankForProc(int procIdx)
{
    static int cnt = 0;
    if (!cnt++)
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Using pseudo mpi rank for CTI\n");
    return procIdx;
}

StatError_t STAT_ctiFrontEnd::dumpProctab()
{
    char fileName[BUFSIZE];
    snprintf(fileName, BUFSIZE, "%s/%s.ptab", outDir_, filePrefix_);

    FILE* file = fopen(fileName, "w");
    if (!file)
    {
        printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: fopen failed to create ptab file %s\n", strerror(errno), fileName);
        return STAT_FILE_ERROR;
    }

    if (!hosts_)
    {
        fprintf(file, "host names are unavailable\n");
    }
    else
    {
        int totPEs = 0;
        for (int i=0, n=hosts_->numHosts; i<n; ++i)
        {
            int pes = hosts_->hosts[i].numPes;
            fprintf(file, "%d - %d : %s" , totPEs, totPEs+pes-1, hosts_->hosts[i].hostname);
            totPEs += pes;
        }
    }

    fclose(file);
    return STAT_OK;
}

bool STAT_ctiFrontEnd::checkNodeAccess(const char *node)
{
    /* MRNet CPs launched through alps */
    return true;
}

StatError_t STAT_ctiFrontEnd::addSerialProcess(const char *pidString)
{
    printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Serial launch is not supported in CTI\n");
    return STAT_ARG_ERROR;
}
StatError_t STAT_ctiFrontEnd::addDaemonSerialProcArgs(int& deamonArgc, char ** &deamonArgv)
{
    printMsg(STAT_ARG_ERROR, __FILE__, __LINE__, "Serial launch is not supported in CTI\n");
    return STAT_ARG_ERROR;
}

StatError_t STAT_ctiFrontEnd::setAppNodeList()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating application node list\n");
    applicationNodeMultiSet_.clear();

    if (!hosts_)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "host table is not available\n");
        return STAT_SYSTEM_ERROR;
    }

    int numHosts = hosts_->numHosts;
    for (int i=0; i<numHosts; ++i)
    {
        applicationNodeMultiSet_.insert(hosts_->hosts[i].hostname);
    }

    nApplNodes_ = numHosts;

    return STAT_OK;
}

StatError_t STAT_ctiFrontEnd::STATBench_setAppNodeList()
{
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Generating application node list\n");
    applicationNodeMultiSet_.clear();

    if (!hosts_)
    {
        printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "host table is not available\n");
        return STAT_SYSTEM_ERROR;
    }

    for (int i=0, n=hosts_->numHosts; i<n; ++i)
    {
        const std::string host = hosts_->hosts[i].hostname;
        for (int j=0, numPEs=hosts_->hosts[i].numPes; j<numPEs; ++j)
        {
            applicationNodeMultiSet_.insert(host);
        }
    }

    nApplNodes_ = nApplProcs_;

    return STAT_OK;
}
StatError_t STAT_ctiFrontEnd::STATBench_resetProctab(unsigned int nTasks)
{
    tasksPerPE_ = nTasks;
    nApplProcs_ = nApplNodes_ * nTasks;
    return STAT_OK;
}

const char* STAT_ctiFrontEnd::daemonExitedMsg()
{
    return "CTI detected the daemons have exited\n";
}

const char* STAT_ctiFrontEnd::appExitedMsg()
{
    return "CTI detected the daemons have exited\n";
}


#ifdef USE_CTI
STAT_FrontEnd* STAT_FrontEnd::make()
{
    return new STAT_ctiFrontEnd();
}
#endif

