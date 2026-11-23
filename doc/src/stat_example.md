# STAT Usage Example

The most typical usage is to invoke STAT on the job launcher's PID:

```text
% srun mpi_application arg1 arg2 &
[1] 16842

% ps
  PID TTY          TIME CMD
16755 pts/0    00:00:00 bash
16842 pts/0    00:00:00 srun
16871 pts/0    00:00:00 ps

% stat-cl 16842
```

Or for Flux, there is an additional step required:

```text
% flux run -n 8 mpi_application &
[1] 216842

% flux jobs -A
     JOBID USER     NAME       ST NTASKS NNODES  RUNTIME NODELIST
  fdB2peU7 lee218   a.out       R      8      2   12.89s fluke[6-7]
% flux job attach --debug fdB2peU7&
[3] 216853
% stat-cl 216853
```

You can also launch your application under STAT's control with the ` -C ` option.  All arguments after ` -C ` are used for job launch:

```text
% stat-cl -C srun mpi_application arg1 arg2
```

With the ` -a ` option (or when automatic topology is set as default), STAT will try to automatically create a scalable topology for large scale jobs.  However, if you wish you may manually specify a topology at larger scales.  For example, if you're running on 1024 nodes, you may want to try a fanout of sqrt(1024) = 32.  You will need to specify a list of nodes that contains enough processors to accommodate the ceil(1024/32) = 32 communication processes being launched with the ` --nodes ` option.  Be sure that you have login permissions to the specified nodes and that they contain the mrnet\_commnode executable and the STAT\_FilterDefinitions.so library.

```text
% stat-cl --fanout 32 --nodes atlas[1-4] --procs 8 16482
```

Upon successful completion, STAT will write its output to a stat\_results directory within the current working directory.  Each run creates a subdirectory named after the application with a unique integer ID.  STAT's output indicates the directory created with a message such as:

```text
Results written to /home/user/bin/stat_results/mpi_application.6
```

Within that directory will be one or more files with a .dot extension.  These .dot files can be viewed with ` stat-view `.
