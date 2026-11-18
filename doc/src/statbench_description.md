# Using the stat-bench Emulator

## Description

The Stack Trace Analysis Tool is a highly scalable, lightweight tool that gathers and merges stack traces from all of the processes of a parallel application.  ` stat-bench ` is a benchmark that can emulate STAT's performance.  By utilizing your entire parallel allocation (launching one stat-bench daemon emulator per core) and generating artificial stack traces, stat-bench is able model STAT's performance using less resources than an actual STAT run requires.  With various options, you can also map stat-bench to your target machine architecture and target application.  After completion, stat-bench will create a stat\_results directory in your current working directory.  This directory will contain a subdirectory for the current run, with the merged stack traces in DOT format as well as a performance results text file.  An example stat-bench generated prefix tree emulating 1M (1024\*1024) tasks can be seen in {numref}`statbench1m3d`.

```{figure} figs/statbench_1m_3d.png
:name: statbench1m3d
:alt: A stat-bench generated prefix tree emulating over 1 million tasks.
:width: 85%

A stat-bench generated prefix tree emulating over 1 million tasks.
```

```{include} statbench_options.md
:heading-offset: 1
```

```{include} statbench_example.md
:heading-offset: 1
```
