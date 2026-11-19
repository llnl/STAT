# Using STAT with IO Watchdog and SLURM

STAT can be used in conjunction with the [IO Watchdog](<http://code.google.com/p/io-watchdog/>) utility, which monitors application output to detect hangs.  To enable STAT with the IO Watchdog, add the following to the file $HOME/.io-watchdogrc

```text
search /usr/local/tools/io-watchdog/actions
timeout = 20m
actions = STAT, kill
```

You will then need to run your application with the --io-watchdog ` srun ` option:

```text
% srun --io-watchdog mpi_application    
```

When STAT is invoked, it will create a stat\_results directory in the current working directory, as it would in a typical STAT run.  The outputted .dot files can then be viewed with ` stat-view `.  For more details about using IO Watchdog, refer to the IO Watchdog README file in /usr/local/tools/io-watchdog/README.
