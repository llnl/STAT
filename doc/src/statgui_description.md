# Using the stat-gui GUI

## Description

````{only} not man
   STAT includes a graphical user interface (GUI) to run STAT and to visualize STAT's outputted call prefix trees ({numref}`guiScreenshot`). This GUI provides a variety of operations to help focus on particular call paths and tasks of interest. It can also be used to identify the various equivalence classes and includes an interface to attach a heavyweight debugger to the representative subset of tasks.

   ```{figure} figs/statgui_screenshot.png
   :name: guiScreenshot
   :alt: A screenshot of the STAT GUI
   :width: 92%

   A screenshot of the STAT GUI
   ```
````

````{only} man
   stat-gui is a GUI that drives STAT, the Stack Trace Analysis Tool, and allows you to interactively control the sampling of stack traces from your parallel application.  stat-gui is built on stat-view and provides the same call tree manipulation operations.  For more on these operations see ` stat-view `.
````

## Quick walkthrough

```{include} statgui_tutorial.md
:start-line: 1
```

```{include} statgui_options.md
:heading-offset: 1
```

```{include} statgui_toolbar.md
:heading-offset: 1
```

```{include} statgui_sample_options.md
:heading-offset: 1
```

```{include} statgui_proc_tab.md
:heading-offset: 1
```

```{include} statgui_eq_class.md
:heading-offset: 1
```

```{include} stat_llnl_availability.md
:heading-offset: 1
```
