# Running STAT in a Batch Script

A good way to run STAT is at the end of a batch script.  For example, if an application is estimated to take 10 hours to run and 12 hours are allocated, then you may consider your application hung if it is still running up to the 12th hour.  In such a situation, one may choose to run STAT in the last 10 minutes of the allocation to get diagnostic information about the job.

The following example script demonstrates how one might setup STAT to catch a hung job in a batch script.

```text
#!/bin/sh

# perform your batch script prologue/setup here

stat_wait_time_minutes=120
application_exited=0

#run the application and get the launcher PID
srun mpi_ringtopo &
pid=$!

# periodically check for application exit
for i in `seq ${stat_wait_time_minutes}`
do
    sleep 60
    ps -p ${pid}
    if test $? -eq 1 
    then
        # the application exited, so we're done!
        application_exited=1 
        break
    fi
done

# if the application is still running then invoke STAT 
if test ${application_exited} -eq 0
then
    /usr/local/bin/stat-cl -c ${pid}
    waitpid ${pid} # alternatively you may want to `kill -TERM ${pid}`
fi    

# perform your batch script epilogue/cleanup here
```

Within the for loop, the script will check every minute (sleep for 60 seconds between checks) to see if the application is still running by running \`ps\` on the PID of the job launcher.  If the application has exited, the script will break from the loop and perform any remaining operations in the batch script.  If the wait time, 120 minutes in this example, expires then STAT will be run to gather stack traces from the application.  The wait time should be set such that STAT has enough time to run (i.e., 10 minutes to be safe) within the batch script's allocated time.  Note the -c option to STAT gathers a "comprehensive" set of stack traces, with varying levels of detail.  After STAT completes, the script then waits for the application to exit.  Alternatively, you may want to kill the application if it isn't making any progress.
