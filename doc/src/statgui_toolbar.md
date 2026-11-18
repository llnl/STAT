# The stat-gui GUI Toolbar

In addition to the operations provided by stat-view, stat-gui provides a toolbar ({numref}`statguiToolbar`) to control STAT's operation.

```{figure} figs/statgui_toolbar.png
:name: statguiToolbar
:alt: The STAT GUI toolbar.
:width: 60px
:figclass: use-figure-width

The STAT GUI toolbar.
```

**Attach**
:   Attach to your application and gather an initial sample.

**ReAttach**
:   Reattach to the parallel application and gather an initial sample.

**Detach**
:   Detach from your application.

**Pause**
:   Put the application in a stopped state.

**Resume**
:   Set the application running.

**Sample**
:   Gather and merge a single stack trace from each task in your parallel application.  The application is left in a stopped state after sampling.

**Sample Multiple**
:   Gather and merge multiple stack traces from each task in your parallel application over time.  The application is left in a stopped state after sampling.
