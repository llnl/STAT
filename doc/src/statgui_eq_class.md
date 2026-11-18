# Equivalence Classes and Subset Debugging

stat-gui can also serve as an interface to attach a full-featured debugger such as TotalView or DDT to a subset of application tasks.  This interface can be accessed through the "identify equivalence classes" **Eq C** button, which will pop up the equivalence classes window ({numref}`statguiEqClasses`).  You can then select a single representative, all, or none of an equivalence classes' tasks to form a subset of tasks. The **Attach to Subset** buttons will launch the specified debugger and attach to the subset of tasks (note, this detaches STAT from the application). There are two DDT subset-attach buttons. The preferred DDT method is to use **DDT bulk attach** which is more scalable than the **DDT host:PID attach**. If the bulk attach fails, then users may fall back to the host:PID attach. The **Debugger Options** button allows you to modify the debugger path.

```{figure} figs/statgui_eq_classes.png
:name: statguiEqClasses
:alt: The equivalence classes window. The colored task lists correspond to the nodes in the prefix tree.
:width: 92%

The equivalence classes window. The colored task lists correspond to the nodes in the prefix tree.
```
