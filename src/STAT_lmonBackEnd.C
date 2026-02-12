#include "STAT_lmonBackEnd.h"

using namespace std;
using namespace MRN;

STAT_lmonBackEnd::STAT_lmonBackEnd(StatDaemonLaunch_t launchType)
    : STAT_BackEnd(launchType)
{}

StatError_t STAT_lmonBackEnd::init(int *argc, char ***argv)
{
    if (launchType_ == STATD_LMON_LAUNCH)
    {
        lmon_rc_e lmonRet = LMON_be_init(LMON_VERSION, argc, argv);
        if (lmonRet != LMON_OK)
        {
            fprintf(stderr, "Failed to initialize Launchmon\n");
            return STAT_LMON_ERROR;
        }
    }

    return STAT_BackEnd::init(argc, argv);
}

StatError_t STAT_lmonBackEnd::finalize()
{
    lmon_rc_e lmonRet;

    if (launchType_ == STATD_LMON_LAUNCH)
    {
        lmonRet = LMON_be_finalize();
        if (lmonRet != LMON_OK)
        {
            fprintf(stderr, "Failed to finalize Launchmon\n");
            return STAT_LMON_ERROR;
        }
    }

    return STAT_OK;
}

// initialize launchmon
StatError_t STAT_lmonBackEnd::initLauncher()
{
    int size, lmonProcTabSize = 0;
    unsigned int i;
    lmon_rc_e lmonRet;
    MPIR_PROCDESC_EXT *lmonProcTab;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Registering unpack function with Launchmon\n");
    lmonRet = LMON_be_regUnpackForFeToBe(unpackStatBeInfo);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to register unpack function\n");
        return STAT_LMON_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launchmon handshaking\n");
    lmonRet = LMON_be_handshake(NULL);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to handshake with Launchmon\n");
        return STAT_LMON_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Waiting for launchmon to be ready\n");
    lmonRet = LMON_be_ready(NULL);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Launchmon failed to get ready\n");
        return STAT_LMON_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Allocating process table\n");
    lmonRet = LMON_be_getMyProctabSize(&size);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get Process Table Size\n");
        return STAT_LMON_ERROR;
    }

    lmonProcTab = (MPIR_PROCDESC_EXT *) malloc(size * sizeof(MPIR_PROCDESC_EXT));
    if (lmonProcTab == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error allocating process table\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Getting my process table entries\n");
    lmonRet = LMON_be_getMyProctab(lmonProcTab, &lmonProcTabSize, size);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to get Process Table\n");
        return STAT_LMON_ERROR;
    }

    proctab_ = (StatBackEndProcInfo_t *)malloc(size * sizeof(StatBackEndProcInfo_t));
    if (proctab_ == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error allocating process table\n", strerror(errno));
        return STAT_ALLOCATE_ERROR;
    }

    proctabSize_ = size;
    for (i = 0; i < size; ++i)
    {
        proctab_[i].executable_name = lmonProcTab[i].pd.executable_name;
        proctab_[i].host_name = lmonProcTab[i].pd.host_name;
        proctab_[i].pid = lmonProcTab[i].pd.pid;
        proctab_[i].mpirank = lmonProcTab[i].mpirank;
    }
    free(lmonProcTab);

    initialized_ = true;
    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Launchmon successfully initialized\n");

    return STAT_OK;
}

StatError_t STAT_lmonBackEnd::connect(int argc, char **argv)
{
    int i, newProctabSize, oldProctabSize, index;
    unsigned int j, k, nextNodeRank;
    bool found;
    char *param[6], parentPort[BUFSIZE], parentRank[BUFSIZE], myRank[BUFSIZE];
    pid_t pid;
    string prettyHost, leafPrettyHost;
    StatBackEndProcInfo_t *oldProctab, *newProctab;
    lmon_rc_e lmonRet;
    StatLeafInfoArray_t leafInfoArray;

    printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Prepearing to connect to MRNet\n");

    if (argc == 0 || argv == NULL)
    {
        if (initialized_ == false)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Trying to connect when not initialized\n");
            return STAT_LMON_ERROR;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Receiving connection information from FE\n");
        lmonRet = LMON_be_recvUsrData((void *)&leafInfoArray);
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to receive data from FE\n");
            return STAT_LMON_ERROR;
        }

        /* Master broadcast number of daemons */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Broadcasting size of connection info to daemons\n");
        lmonRet = LMON_be_broadcast((void *)&(leafInfoArray.size), sizeof(int));
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to broadcast num_leaves\n");
            return STAT_LMON_ERROR;
        }

        /* Non-masters allocate space for the MRNet connection info */
        if (LMON_be_amIMaster() == LMON_NO)
        {
            leafInfoArray.leaves = (StatLeafInfo_t *)malloc(leafInfoArray.size * sizeof(StatLeafInfo_t));
            if (leafInfoArray.leaves == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to allocate %d x %d = %dB of memory for leaf info\n", leafInfoArray.size, sizeof(StatLeafInfo_t), leafInfoArray.size*sizeof(StatLeafInfo_t));
                return STAT_ALLOCATE_ERROR;
            }
        }

        /* Master broadcast MRNet connection information to all daemons */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Broadcasting connection information to all daemons\n");
        lmonRet = LMON_be_broadcast((void *)(leafInfoArray.leaves), leafInfoArray.size * sizeof(StatLeafInfo_t));
        if (lmonRet != LMON_OK)
        {
            printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to broadcast num_leaves\n");
            return STAT_LMON_ERROR;
        }

        /* Find my MRNet personality  */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Searching for my connection information\n");
        found = false;
        XPlat::NetUtils::GetHostName(localHostName_, prettyHost);
        for (k = 0; k < leafInfoArray.size; k++)
        {
            XPlat::NetUtils::GetHostName(string(leafInfoArray.leaves[k].hostName), leafPrettyHost);
            if (prettyHost == leafPrettyHost || strcmp(leafPrettyHost.c_str(), localIp_) == 0)
            {
                found = true;
                parentHostName_ = strdup(leafInfoArray.leaves[k].parentHostName);
                if (parentHostName_ == NULL)
                {
                    printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed on call to strdup(%s) to parentHostName_)\n", strerror(errno), leafInfoArray.leaves[k].parentHostName);
                    return STAT_ALLOCATE_ERROR;
                }
                parentPort_ = leafInfoArray.leaves[k].parentPort;
                parentRank_ = leafInfoArray.leaves[k].parentRank;
                myRank_ = leafInfoArray.leaves[k].rank;
                break;
            }
        }
        if (found == false)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "Failed to find MRNet parent info\n");
            return STAT_MRNET_ERROR;
        }

        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Found MRNet connection info, parent hostname = %s, parent port = %d, my MRNet rank = %d\n", parentHostName_, parentPort_, myRank_);

        if (nDaemonsPerNode_ > 1)
        {
            nextNodeRank = myNodeRank_;
            pid = 1;
            for (j = 1; j < nDaemonsPerNode_ && pid > 0; j++)
            {
                nextNodeRank++;
                pid = fork();
                if (pid == 0)
                {
                    myNodeRank_ = nextNodeRank;
                    myRank_ = leafInfoArray.size * myNodeRank_ + myRank_;
                }
                else if (pid < 0)
                {
                    printMsg(STAT_SYSTEM_ERROR, __FILE__, __LINE__, "Failed to fork with pid %d\n", pid);
                    return STAT_SYSTEM_ERROR;
                }
            }
            newProctabSize = proctabSize_ / nDaemonsPerNode_;
            if (proctabSize_ % nDaemonsPerNode_ > myNodeRank_)
                newProctabSize++;
            newProctab = (StatBackEndProcInfo_t*)malloc(newProctabSize * sizeof(StatBackEndProcInfo_t));
            if (newProctab == NULL)
            {
                printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error allocating %d new process table\n", strerror(errno), newProctabSize);
                return STAT_ALLOCATE_ERROR;
            }
            index = myNodeRank_ * (proctabSize_ / nDaemonsPerNode_);
            if (proctabSize_ % nDaemonsPerNode_ > 0)
            {
                if (myNodeRank_ <= proctabSize_ % nDaemonsPerNode_)
                    index += myNodeRank_;
                else
                    index += proctabSize_ % nDaemonsPerNode_;
            }
            for (i = 0; i < newProctabSize; i++)
            {
                newProctab[i].mpirank = proctab_[index].mpirank;
                newProctab[i].pid = proctab_[index].pid;
                newProctab[i].executable_name = strdup(proctab_[index].executable_name);
                if (newProctab[i].executable_name == NULL)
                {
                    printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error allocating exe name for %d %d %s\n", strerror(errno), i, index, proctab_[i].executable_name);
                    return STAT_ALLOCATE_ERROR;
                }
                newProctab[i].host_name = strdup(proctab_[index].host_name);
                if (newProctab[i].host_name == NULL)
                {
                    printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s: Error allocating exe name for %d %d %s\n", strerror(errno), i, index, proctab_[i].host_name);
                    return STAT_ALLOCATE_ERROR;
                }
                index++;
            }

            oldProctabSize = proctabSize_;
            oldProctab = proctab_;
            proctabSize_ = newProctabSize;
            proctab_ = newProctab;
            for (i = 0; i < oldProctabSize; i++)
            {
                if (oldProctab[i].executable_name != NULL)
                    free(oldProctab[i].executable_name);
                if (oldProctab[i].host_name != NULL)
                    free(oldProctab[i].host_name);
            }
            free(oldProctab);
        }

        /* Connect to the MRNet Network */
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Connecting to MRNet network\n");
        snprintf(parentPort, BUFSIZE, "%d", parentPort_);
        snprintf(parentRank, BUFSIZE, "%d", parentRank_);
        snprintf(myRank, BUFSIZE, "%d", myRank_);
        param[0] = NULL;
        param[1] = parentHostName_;
        param[2] = parentPort;
        param[3] = parentRank;
        param[4] = localHostName_;
        param[5] = myRank;
        printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Calling CreateNetworkBE with 6 args:\n");
        for (i = 0; i < 6; i++)
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\targv[%d] = %s\n", i, param[i]);
        network_ = Network::CreateNetworkBE(6, param);
        if (network_ == NULL)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "CreateNetworkBE() failed\n");
            return STAT_MRNET_ERROR;
        }
        printMsg(STAT_LOG_MESSAGE,__FILE__,__LINE__,"CreateNetworkBE successfully completed\n");

        if (leafInfoArray.leaves != NULL)
            free(leafInfoArray.leaves);
    } /* if (argc == 0 || argv == NULL) */
    else
    {
        /* We were launched by MRNet */
        /* Store my connection information */
        parentHostName_ = strdup(argv[1]);
        if (parentHostName_ == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "%s Failed on call to strdup(%s) to parentHostName_\n", strerror(errno), argv[1]);
            return STAT_ALLOCATE_ERROR;
        }
        parentPort_ = atoi(argv[2]);
        parentRank_ = atoi(argv[3]);
        myRank_ = atoi(argv[5]);

        /* Connect to the MRNet Network */
        printMsg(STAT_LOG_MESSAGE, __FILE__ ,__LINE__, "Calling CreateNetworkBE with %d args:\n", argc);
        for (i = 0; i < argc; i++)
            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "\targv[%d] = %s\n", i, argv[i]);
        network_ = Network::CreateNetworkBE(6, argv);
        if (network_ == NULL)
        {
            printMsg(STAT_MRNET_ERROR, __FILE__, __LINE__, "CreateNetworkBE() failed\n");
            return STAT_MRNET_ERROR;
        }
        printMsg(STAT_LOG_MESSAGE,__FILE__,__LINE__,"CreateNetworkBE successfully completed\n");
    } /* else (argc == 0 || argv == NULL) */

    connected_ = true;

    return STAT_OK;
}

/******************
 * STATBench Code *
 ******************/

StatError_t STAT_lmonBackEnd::statBenchConnectInfoDump()
{
    int i, count, intRet, fd;
    unsigned int bytesWritten, j;
    char fileName[BUFSIZE], data[BUFSIZE], *ptr = NULL;
    string prettyHost, leafPrettyHost;
    lmon_rc_e lmonRet;
    StatLeafInfoArray_t leafInfoArray;

    /* Master daemon receive MRNet connection information */
    lmonRet = LMON_be_recvUsrData((void *)&leafInfoArray);
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to receive data from FE\n");
        return STAT_LMON_ERROR;
    }

    /* Master broadcast number of daemons */
    lmonRet = LMON_be_broadcast((void *)&(leafInfoArray.size), sizeof(int));
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to broadcast num_leaves\n");
        return STAT_LMON_ERROR;
    }

    /* Non-masters allocate space for the MRNet connection info */
    if (LMON_be_amIMaster() == LMON_NO)
    {
        leafInfoArray.leaves = (StatLeafInfo_t *)malloc(leafInfoArray.size * sizeof(StatLeafInfo_t));
        if (leafInfoArray.leaves == NULL)
        {
            printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to allocate memory for leaf info\n");
            return STAT_ALLOCATE_ERROR;
        }
    }

    /* Master broadcast MRNet connection information to all daemons */
    lmonRet = LMON_be_broadcast((void *)(leafInfoArray.leaves), leafInfoArray.size * sizeof(StatLeafInfo_t));
    if (lmonRet != LMON_OK)
    {
        printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "Failed to broadcast num_leaves\n");
        return STAT_LMON_ERROR;
    }

    /* Since we started the STATBench daemons under debugger control, */
    /* we must send a continue signal the STATBench daemons */
    for (i = 0; i < proctabSize_; i++)
        kill(proctab_[i].pid, SIGCONT);

    /* Find MRNet personalities for all STATBench Daemons on this node */
    for (j = 0, count = -1; j < leafInfoArray.size; j++)
    {
        XPlat::NetUtils::GetHostName(localHostName_, prettyHost);
        XPlat::NetUtils::GetHostName(string(leafInfoArray.leaves[j].hostName), leafPrettyHost);
        if (prettyHost == leafPrettyHost)
        {
            count++;
            if (count >= proctabSize_)
            {
                printMsg(STAT_LMON_ERROR, __FILE__, __LINE__, "The size of the leaf information for this node appears to be larger than the process table size\n");
                return STAT_LMON_ERROR;
            }

            /* Wait for the daemon emulator to create its fifo then write its connection info */
            snprintf(fileName, BUFSIZE, "/tmp/%s.%d.statbench.txt", localHostName_, count);
            while (1)
            {
                intRet = access(fileName, W_OK);
                if (intRet == 0)
                    break;
                usleep(1000);
            }

            fd = open(fileName, O_WRONLY);
            if (fd == -1)
            {
                printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: open() failed for %s\n", strerror(errno), fileName);
                remove(fileName);
                return STAT_FILE_ERROR;
            }

            snprintf(data, BUFSIZE, "%s %d %d %d %d", leafInfoArray.leaves[j].parentHostName, leafInfoArray.leaves[j].parentPort, leafInfoArray.leaves[j].parentRank, leafInfoArray.leaves[j].rank, proctab_[count].mpirank);
            bytesWritten = 0;
            while (bytesWritten < strlen(data))
            {
                ptr = data + bytesWritten;
                intRet = write(fd, ptr, strlen(ptr));
                if (intRet == -1)
                {
                    printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: write() to fifo %s failed\n", strerror(errno), fileName);
                    remove(fileName);
                    return STAT_FILE_ERROR;
                }
                bytesWritten += intRet;
            }

            intRet = close(fd);
            if (intRet != 0)
            {
                printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: close() failed for fifo %s, fd %d\n", strerror(errno), fileName, fd);
                remove(fileName);
                return STAT_FILE_ERROR;
            }
        }
    }

    for (i = 0; i <= count; i++)
    {
        snprintf(fileName, BUFSIZE, "/tmp/%s.%d.statbench.txt", localHostName_, i);
        intRet = remove(fileName);
        if (intRet != 0)
        {
            printMsg(STAT_FILE_ERROR, __FILE__, __LINE__, "%s: remove() failed for fifo %s\n", strerror(errno), fileName);
            return STAT_FILE_ERROR;
        }
    }

    if (leafInfoArray.leaves != NULL)
        free(leafInfoArray.leaves);

    return STAT_OK;
}

#ifdef STAT_GDB_BE
StatError_t STAT_lmonBackEnd::initGdb()
{
    PyObject *pName;
    const char *moduleName = "stat_cuda_gdb";
    Py_Initialize();
#if PY_MAJOR_VERSION >= 3
    pName = PyUnicode_FromString(moduleName);
#else
    pName = PyString_FromString(moduleName);
#endif
    if (pName == NULL)
    {
        fprintf(errOutFp_, "Cannot convert argument\n");
        return STAT_SYSTEM_ERROR;
    }

    gdbModule_ = PyImport_Import(pName);
    Py_DECREF(pName);
    if (gdbModule_ == NULL)
    {
        fprintf(errOutFp_, "Failed to import Python module %s\n", moduleName);
        PyErr_Print();
        return STAT_SYSTEM_ERROR;
    }
    usingGdb_ = true;

    return STAT_OK;
}
#endif

#ifndef USE_CTI
STAT_BackEnd* STAT_BackEnd::make(StatDaemonLaunch_t launchType)
{
    return new STAT_lmonBackEnd(launchType);
}
#endif
