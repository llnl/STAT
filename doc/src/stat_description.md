# Using the stat-cl Command

## Description

STAT (the Stack Trace Analysis Tool) is a highly scalable, lightweight tool that gathers and merges stack traces from all of the processes of a parallel application.  After running the ` stat-cl ` command, STAT will create a stat\_results directory in your current working directory.  This directory will contain a subdirectory, based on your parallel application's executable name, with the merged stack traces in DOT format.

```{include} stat_options.md
:heading-offset: 1
```

```{include} stat_example.md
:heading-offset: 1
```

```{include} stat_pystack.md
:heading-offset: 1
```
