# Using the stat-view GUI

## Description

` stat-view ` ({numref}`statviewScreenshot`) is a GUI for viewing STAT-outputted dot files.  ` stat-view ` provides easy navigation of the call prefix tree and also allows manipulation of the call tree to help focus on areas of interest.  Each node in the STAT call prefix tree represents a function call and the directed edges denote the calling sequence.  Further, the edges are labeled by the set of tasks that have taken that call path.  For simplification, stat-view will display the number of tasks in the set and truncate long task lists in the main display with "..." notation.  Similarly, long function node label names will be truncated with "..." notation.  The truncation length can be modified via the File-\>Preferences menu (this requires clicking the **Layout** button to rerender any already loaded graphs).  Nodes are colored based on the set of tasks of the incoming edge, providing a visual distinction when different tasks take different branches.

The ` stat-view ` GUI also allows you to view the application source files in the stack traces, when sampling is done at the source file and line number granularity. This may require the source file's path to be added to the search path, through **File** -\> **Add Search Paths**. If an application's source code is edited after STAT is run, the line numbers shown in the stack traces may not be accurate. To alleviate this problem, STAT can optionally cache the source files for the currently displayed .dot file. To cache files click on the **File** -\> **Add Search Paths** menu item. This will find and save the source files in the .dot file's stat\_results directory. The next time you open the .dot file with ` stat-view `, the source files will automatically be loaded from the cache.

```{raw} latex
\begingroup
\setkeys{Gin}{trim=0bp 427.5bp 2017.5bp 0bp,clip}
```

```{figure} figs/statview_screenshot.png
:name: statviewScreenshot
:alt: A screenshot of the stat-view GUI.
:width: 100%
:figclass: stat-view-screenshot

A screenshot of the stat-view GUI.
```

```{raw} latex
\endgroup
```

```{include} statview_node_menu.md
:heading-offset: 1
```

```{include} statview_toolbar.md
:heading-offset: 1
```
