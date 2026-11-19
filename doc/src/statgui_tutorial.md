# Using the STAT GUI

After launching the GUI via the ` stat-gui ` command you will first need to attach to the application via the **Attach** button.  This will bring up the attach dialog ({numref}`statguiAttachDialog`).  You will need to select the job launcher (i.e., ` mpirun `, ` srun `, or equivalent process).  STAT will list processes owned by you on the localhost.  If your job launcher process is on a remote host, you will need to enter that hostname in the remote host text entry box.  Contact your local system administrator if you are not sure where to find your job launcher process.  Once the appropriate process has been selected, click on the **Attach** in the lower right hand corner.  STAT will then launch its daemons and gather an initial stack trace.

```{figure} figs/statgui_attach_dialog.png
:name: statguiAttachDialog
:alt: Use the attach dialog to select the job launcher process to attach to.
:width: 76%
:figclass: use-figure-width

Use the attach dialog to select the job launcher process to attach to.
```

Once STAT has displayed the initial merged stack trace, in the form of a prefix tree, first you may want to look for common buggy patterns.  This can be accomplished by using the analysis buttons in the toolbar across the top of the window.  This includes operations to look for outliers such as processes with the shortest or longest stack trace or for the stack trace that was exhibited by the least or most processes.  Note that these buttons are "traversal" buttons and they all initially operate on the full prefix tree.  For example, the first click of the **\[Shortest\] Path** button will display the shortest path and subsequent clicks will display the next shortest path.  Oftentimes bugs in parallel applications are triggered by a single or small subset of outliers in which case the **\[Least\] Tasks** button can quickly identify the outliers.  Another common behavior is for a small subset of processes to be hung and the rest of the processes to be blocked in an MPI barrier or collective.  In this case, the hung subset of tasks may have a shorter call path than the tasks in blocking in MPI, since the MPI implementation will usually be several frames deep.  In this case, the **\[Shortest\] Path** can be useful.

```{figure} figs/statgui_bgl208k_screenshot.png
:name: statguiBgl208kScreenshot
:alt: A screenshot of the STAT GUI.
:width: 92%

A screenshot of the STAT GUI.
```

Alternatively, you may wish to manually search through the stack traces.  There are several buttons to aid in this process too.  The **\[Traverse\] Eq C** will traverse the prefix tree, with each click traversing the down to the next point where there is a branch in equivalence classeses.  There is also a **Search** button to search for specific MPI ranks, for stack frames with specified text, or for tasks running on specified nodes.  Finally, left or right clicking on a node in the prefix tree gives you the option to expand or collapse the prefix tree.  Note also that you can zoom in and out of the prefix tree using the options in the **View** menu or by using the scroll button on your mouse.  You can also hold the left mouse button to "grab" the whitespace in the displayed prefix tree and move the focus around.

Another helpful button is the **Cut** button, which allows you to cut the tree at frames below a specified programming model's implementation.  For example, **\[Cut\] MPI** will cut any frames below an MPI function call, thus allowing you to focus on application code as opposed to the MPI implementation stack frames.  You can define your own programming model on the fly via the **Add Model** button.  Default programming models are defined in the installation $prefix/etc/STAT/STATview\_models.conf file or in the user $HOME/.STATview\_models.conf file.  Programming models are specified as regular expressions, using Python's ` re ` module syntax, and the ` re.search ` function is used in favor of ` re.match `.

By default, the initial sample will gather stack traces at the granularity of function names.  You can gather an additional sample with more detail, by clicking on the **Sample** button and selecting the **function and line** radio button.  Note this typically requires that the code be compiled with the -g flag to get the appropriate debug information.  After clicking **OK** a new prefix tree will be generated.  By gathering stack traces with line number information, you may now associate stack traces back to the source code using the **View Source** button after clicking on a node in the prefix tree ({numref}`sourceView`).

STAT was not intended to be a full-featured debugger, so you may ultimately need to employ another debugger such as DDT or TotalView for root cause analysis.  STAT includes an interface to launch either of these debuggers (where available) on a subset of the MPI tasks based on the equivalence classes that STAT identifies.  This interface can be accessed through the **\[Identify\] Eq C** button in the upper right hand corner of the window.  In order to allow the other debuggers to attach, STAT will first detach itself from the application.  Pinpointing a bug may require several iterations of running STAT on the entire application and running a full-featured debugger on a subset.  After detaching a full-featured debugger, you can quickly attach to your application again with the **ReAttach** button.
