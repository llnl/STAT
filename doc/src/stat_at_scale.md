# Tips and Tricks Using STAT

## Running STAT at scale

STAT is highly scalable and its default analysis has been shown to run effectively on jobs even over one million MPI tasks. Even so, at extreme scales, there are several options that may make STAT's operation even more scalable. The first is to specify the underlying communication tree topology. By default, the ` stat-cl ` command and ` stat-gui ` will try to deploy the automatic topology, which defaults to a fanout of 64. STAT will, by default, try to co-locate the communication processes on the application nodes (or associated I/O node on BG systems). To avoid the default co-location option for the ` stat-cl ` script (remove the ` -A ` option). For the ` stat-gui ` GUI, create a preferences file and specify an alternative ` CP policy ` option. Refer to the options sections to learn about more topology options.

Typically, STAT launches one debug daemon per node. This can become a bottleneck if that daemon is resposible for debugging many target processes. STAT has a ` daemonspernode ` that allows users to request that multiple daemons be launched per node and distribute the target processes between them. This will help offload some of the debugging workload, however, be aware that this will increase STAT's memory usage per node.

Another consideration at scale is the granularity of debug information. At larger scales, you may prefer to start with coarse-grained analysis. For example, you may not need full task lists for the edge labels, but rather, would like to gather edge labels with just the task count and a representative rank. Within the ` stat-gui ` GUI, one can then request the full task list of a given edge via the left-click menu of the desired edge's target node. Note, with the ` stat-cl ` command, the full task lists would not be gathered and thus would not be available for post-mortem analysis via the stat-view GUI. The count and representative granularity will result in faster sampling and smaller output-file size.

The granularity of the stack traces themselves can also be adjusted to alleviate bottlenecks at scale. In particular, symbol resolution can be expensive, particularly when gathering traces at the function and line level of granularity, but even with the function only granularity. This can cause the many STAT daemons to perform many file operation requests at the same time, straining the target file system. To alleviate this issue, you may gather stack traces with the module and offset granularity. The ` stat-gui ` and ` stat-view ` GUIs can later translate the module and offset into function and line number via ` addr2line `. This translation feature is available only through a left-click of a module and offset node and will translate the entire prefix tree. Note that the file-system bottleneck can also be mitigated with the ` --with-fgfs ` configure option, to enable scalable file operations via the [FastGlobalFileStatus](<https://github.com/dongahn/FastGlobalFileStatus>) module.

```{include} io_watchdog.md
:heading-offset: 1
```

```{include} stat_batch_script.md
:heading-offset: 1
```
