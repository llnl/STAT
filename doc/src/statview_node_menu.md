# The stat-view Node Menu

By left clicking on a node in the call prefix tree you will get a window displaying the full list of tasks and the full frame label ({numref}`statguiMenu`).  This window also contains buttons that allow for the manipulation of the graph from that node.  Right clicking on a node provides a pop-up menu with the same options.  Note all of these operations are performed on the current visible state of the call prefix tree.

```{figure} figs/statgui_node_dialog.png
:name: statguiMenu
:alt: The node pop-up window
:width: 100%

The node pop-up window
```

The node operations are defined as follows:

**Advanced**
:   Display the full node and edge attributes.

**Join Equivalence Class**
:   collapses all of the descendent nodes with the same equivalence class into the current node and renders in a new tab.

**Collapse**
:   hide all of the descendents of the selected node.

**Collapse Depth**
:   collapse the entire tree to the depth of the selected node.

**Hide**
:   the same as **Collapse**, but also hides the selected node.

**Expand**
:   show (unhide) the immediate children of the selected node.

**Expand All**
:   show (unhide) all descendents of the selected node.

**Focus**
:   hide all nodes that are neither ancestors nor descendents of the selected node. (Note: This will not unhide any hidden ancestors.)

**View Source**
:   creates a popup window ({numref}`sourceView`) displaying the source file (only for stack traces with line number information). This may require the source file's path to be added to the search path, through **File** -\> **Add Search Paths**.

**Translate**
:   For traces with module and offset granularity, this button will translate all node labels into source file and line number information and open the resulting graph in a new tab.

**Temporally Order Children**
:   (prototype only) determine the temporal order of the node's children (only for stack traces with line number information). Requires the source file's path and all include paths to be added to the search path, through **File** -\> **Add Search Paths**.

**OK**
:   closes the pop-up window.

```{figure} figs/statgui_source_view.png
:name: sourceView
:alt: The source view window.  The colored arrows correspond to the nodes in the call prefix tree.
:width: 100%

The source view window.  The colored arrows correspond to the nodes in the call prefix tree.
```
