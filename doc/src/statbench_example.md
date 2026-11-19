# stat-bench Usage Example

In the simplest form, you can invoke stat-bench, from within a parallel allocation, with no arguments.  This will run through with the default settings:

```text
% stat-bench
```

To model your target machine architecture, you can specify the number of tasks to emulate per daemon.  For instance if your target machine has 16-way SMP compute nodes:

```text
% stat-bench --numtasks 16
```

You may also want to model a specific application.  For instance, you may have a climate modeling code with 5 distinct task behaviors, or equivalence classes.  You can also specify the maximum call depth of your application, the average branching factor from a given function, and the number of distinct traces expected per task:

```text
% stat-bench --eqclasses 5 --maxdepth 17 --branch 5 --traces 4
```

At larger scales, you may want to employ a more scalable tree topology.  For example, if you're running 1024 daemon emulators, you may want to try a fanout of sqrt(1024) = 32.  You will need to specify a list of nodes that contains enough processors to accommodate the ceil(1024/32) = 32 communication processes being launched.  Be sure that you have login permissions to the specified nodes and that they contain the mrnet\_commnode executable and the STAT\_FilterDefinitions.so library:

```text
% stat-bench --fanout 32 --nodes atlas[1-4] --procs 8
```
