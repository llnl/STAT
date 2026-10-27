#ifndef __STAT_CTIBACKEND_H
#define __STAT_CTIBACKEND_H

#include "STAT_BackEnd.h"

// backend implementation class that uses cti to launch tool daemons
class STAT_ctiBackEnd : public STAT_BackEnd
{
public:

    STAT_ctiBackEnd(StatDaemonLaunch_t launchType);

    virtual StatError_t init(int *argc, char ***argv);
    virtual StatError_t finalize();
    
    //! Initialize and set up the cti
    /*
      \return STAT_OK on success
    */
    virtual StatError_t initLauncher();

#ifdef STAT_GDB_BE
    virtual StatError_t initGdb();
#endif

    virtual StatError_t connect(int argc = 0, char **argv = NULL);
    virtual StatError_t statBenchConnectInfoDump();
};


#endif
