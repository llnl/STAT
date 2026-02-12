#ifndef __STAT_LMONFRONTEND_H
#define __STAT_LMONFRONTEND_H

#include "STAT_FrontEnd.h"
#include "lmon_api/lmon_fe.h"

//! A callback function to detect daemon and application exit
/*!
    \param status - the input status vector
    \return 0 on success, -1 on error

    Determine if STAT is still connected to the resource manager
    and if the application is still running.
*/
int lmonStatusCb(int *status);

// Implementation of the STAT front end that uses launchmon to launch
// the stat daemons
class STAT_lmonFrontEnd : public STAT_FrontEnd
{
public:
    STAT_lmonFrontEnd();

    virtual ~STAT_lmonFrontEnd();

    virtual StatError_t setupForSerialAttach();

    virtual StatError_t launchDaemons();
    virtual StatError_t sendDaemonInfo();
    virtual void detachFromLauncher(const char *errMsg);
    virtual void shutDown();

    virtual StatError_t createMRNetNetwork(const char *topologyFileName);

    virtual bool daemonsHaveExited();
    virtual bool isKilled();

    virtual int getNumProcs();
    virtual const char *getHostnameForProc(int procIdx);
    virtual int getMpiRankForProc(int procIdx);
    virtual StatError_t dumpProctab();

    virtual bool checkNodeAccess(const char *node);

    virtual StatError_t addSerialProcess(const char *pidString);
    virtual StatError_t addDaemonSerialProcArgs(int& deamonArgc, char ** &deamonArgv);

    virtual StatError_t setAppNodeList();
    virtual StatError_t STATBench_setAppNodeList();
    virtual StatError_t STATBench_resetProctab(unsigned int nTasks);

private:
    //! validate the apid with CTI
    /*!
      return STAT_OK on success
      Performs validate of apid/launcherPid_ with CTI
    */
    StatError_t validateApidWithCTI();

    int lmonSession_;               /*!< the LaunchMON session ID */
    lmon_rm_info_t lmonRmInfo_;     /*!< the resource manager information from LMON */
    MPIR_PROCDESC_EXT *proctab_;    /*!< the process table */
    unsigned int proctabSize_;      /*!< the size of the process table */

#ifdef CRAYXT
    cti_app_id_t CTIAppId_;         /*!< the CTI application ID */
#endif
};



#endif
