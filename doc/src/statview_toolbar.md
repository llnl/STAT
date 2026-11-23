# The stat-view Toolbar

The main window also has several tree manipulation options ({numref}`statviewToolbar`).  Note the initial click of a traversal operation operates on the original call prefix tree, while the remaining operations are performed on the current visible state of the call prefix tree.

```{figure} figs/statview_toolbar.png
:name: statviewToolbar
:alt: The stat-view tree manipulation toolbar.
:width: 100%

The stat-view tree manipulation toolbar.
```

The toolbar operations are defined as follows:

**Open**
:   Open a STAT generated .dot file

**Save As**
:   Save the current graph in .dot format, which can be displayed by stat-view or in an image format, such as PNG or PDF, which can be viewed on any computer with an image viewer

**Undo**
:   Undo the previous operation

**Redo**
:   Redo the undone operation

**Reset**
:   Revert to the original graph

**Layout**
:   Reset the layout of the current graph and open in a new tab.  This is useful for compacting wide trees after performing some pruning operations.

**Cut**
:   This feature ({numref}`statviewProgModel`) allows you to collapse the prefix tree below the implementation frames for various programming models. For instance, a user may wish to hide all calls that happen within the MPI library.  The programming models may be entered in a configuration file or added by the user. STATview looks for configuration files in $prefix/etc/STAT/STATview\_models.conf and in $HOME/.STATview\_models.conf. Programming models are specified as regular expressions, using Python's ` re ` module syntax, and the ` re.search ` function is used in favor of ` re.match `.

    ```{figure} figs/statview_prog_model.png
    :name: statviewProgModel
    :alt: The stat-view programming model cutting interface.
    :width: 68%
    :figwidth: 68%
    :figclass: fill-figure
    :align: center

    The stat-view programming model cutting interface.
    ```

**\[Cut\] MPI**
:   Collapse the MPI implementation frames below the MPI function call.

**\[Cut\] Pthreads**
:   Collapse the Pthread implementation frames below the Pthread function call.

**Join**
:   Join consecutive nodes of the same equivalence class into a single node and render in a new tab.  This is useful for condensing long call sequences.

**\[Traverse\] Eq C**
:   Traverse the prefix tree by expanding the leaves to the next equivalence class set.  The first click will display the top-level equivalence class.

**\[Traverse Longest\] Path**
:   Traversal focus on the next longest call path(s).  The first click will focus on the longest path.

**\[Traverse Shortest\] Path**
:   Traversal focus on the next shortest call path(s).  The first click will focus on the shortest path.

**\[Traverse Least\] Tasks**
:   Traversal focus on the path(s) with the next least visiting tasks.  The first click will focus on the path with the least visiting tasks.

**\[Traverse Most\] Tasks**
:   Traversal focus on the path(s) with the next most visiting tasks.  The first click will focus on the path with the most visiting tasks.

**\[Traverse Least\] TO**
:   Temporal Order traversal focus on the path(s) that have made the least execution progress in the application.  The first click will focus on the path that has made the least progress.

**\[Traverse Most\] TO**
:   Temporal Order traversal focus on the path(s) that have made the most execution progress in the application.  The first click will focus on the path that has made the most progress.

**Search**
:   Search for call paths containing specified text, taken by specified tasks, or from specified hosts.  Search text may be a regular expression, using the syntax described in [http://docs.python.org/library/re.html](<http://docs.python.org/library/re.html>).

**\[Identify\] Eq C**
:   Identify the equivalence classes of the visible graph.  After clicking on this button, a window will pop up showing the complete list of equivalence classes.
