# Process Table

The application process table can be accessed through the stat-gui File-\>Properties menu item.  This window ({numref}`statguiProcTab`) lists the properties of the application, including the number of nodes, processes, the job launcher host and PID and a 4-tuple list of application process rank, host, PID, and executable.  The executable path in the 4-tuple is an index into the executable list at the top of the window in order to reduce duplication of text.  The 4-tuple process table list can be filtered by ranks or hosts.

```{figure} figs/statgui_proc_tab.png
:name: statguiProcTab
:alt: The properties window shows the application properties and lists the individual application processes.
:width: 76%
:figclass: use-figure-width

The properties window shows the application properties and lists the individual application processes.
```
