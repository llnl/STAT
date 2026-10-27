#ifndef __STAT_CTIFRONTEND_H
#define __STAT_CTIFRONTEND_H

#include "STAT_FrontEnd.h"
#include "common_tools_fe.h"

// Implementation of the STAT front end to launch
// the stat daemons
class STAT_ctiFrontEnd : public STAT_FrontEnd
{
public:
    STAT_ctiFrontEnd();

    virtual ~STAT_ctiFrontEnd();

    virtual StatError_t setupForSerialAttach();

    virtual StatError_t launchDaemons();
    virtual StatError_t sendDaemonInfo();
    virtual StatError_t createMRNetNetwork(const char* topologyFileName);
    virtual void detachFromLauncher(const char* errMsg);
    virtual void shutDown();

    virtual bool daemonsHaveExited();
    virtual bool isKilled();
    virtual StatError_t postAttachApplication();

    virtual int getNumProcs();
    virtual const char* getHostnameForProc(int procIdx);
    virtual int getMpiRankForProc(int procIdx);
    virtual StatError_t dumpProctab();

    virtual bool checkNodeAccess(const char *node);

    virtual StatError_t addSerialProcess(const char *pidString);
    virtual StatError_t addDaemonSerialProcArgs(int& deamonArgc, char ** &deamonArgv);

    virtual StatError_t setAppNodeList();
    virtual StatError_t STATBench_setAppNodeList();
    virtual StatError_t STATBench_resetProctab(unsigned int nTasks);

    virtual bool haveGlobalRanks() { return true; }
    virtual const char* daemonExitedMsg();
    virtual const char* appExitedMsg();	

private:
    StatError_t attach();
    StatError_t launch();
    StatError_t getProcInfo();

private:
    cti_app_id_t appId_;        /*!< the CTI application ID */
    cti_session_id_t session_;  /*!< the CTI session for the daemons */

    cti_hostsList_t* hosts_;
    std::vector<int> procToHost_;
    int tasksPerPE_;
};



#endif
