#ifndef __STAT_LMONBACKEND_H
#define __STAT_LMONBACKEND_H

#include "STAT_BackEnd.h"

#include "lmon_api/lmon_be.h"

// Backend implementation using launchmon to start the tool daemons
class STAT_lmonBackEnd : public STAT_BackEnd
{
    public:

    STAT_lmonBackEnd(StatDaemonLaunch_t launchType);

    virtual StatError_t init(int *argc, char ***argv);
    virtual StatError_t finalize();
    
    //! Initialize and set up LaunchMON
    /*
      \return STAT_OK on success
    */
    virtual StatError_t initLauncher();

    virtual StatError_t connect(int argc = 0, char **argv = NULL);
    virtual StatError_t statBenchConnectInfoDump();

#ifdef STAT_GDB_BE
    virtual StatError_t initGdb();
#endif

};


#endif
