# Usage

A DySectAPI probe session is constructed as a C++ program. You will need to include the "DysectAPI.h" header file, which contains the specifications for the various DySectAPI objects and for the session routines. Your session must define a DysectStatus DysectAPI::onProcStart(int argc, char \*\*argv) routine, which will create and link Probe objects. This section will only cover the high-level basics of constructing a debug session. For detailed information about the various constructs, you may refer to the header files or the reference guide.

A Probe consists of an Event, Condition, Domain, and an Action or list of Actions. Various Probe signatures exist, so not all components are required. Probes can be linked into a tree with the Probe::link(Probe \*) routine. Probe tree roots can be enabled via the Probe::setup() routine.

An Event is an occurence of interest. DySectAPI defines several event types: Location; Timer; Async. Furthermore, Events can be combined with And, Or, and Not releations. A Location, essentially a debugger breakpoint, specifies where to install a probe. DySectAPI supports three location specifications: a function name; a source file and line number; or a program counter address. A Location can also be marked as pending, which informs DySectAPI that the probe should not be enabled until the parent probe has been triggered. A Timer specifies a timeout period to wait before triggering the Probe. An Async can be various asynchronous events, such as a signal, crash, or process exit.

A Condition specifies the circumstances under which a probe is to be triggered. Most conditions evaluate a data expression, such as whether a specified variable equals a certain value or falls within a given range. Conditions can be combined with And, Or, and Not relations.

The Domain specifies the set of processes within which to install a probe. The supported domains are World (i.e., all processes), a rank-specified Group, or Inherit (from the parent probe). A Domain can also specify a timeout, which indicates the amount of time to wait after the first process encounters the probe's event before proceeding.

An Action indicates what to do once a probe has been triggered. The currently implemented Actions are: trace, null, totalview, depositCore, signal, loadLibrary, irpc, writeModuleVariable, stat, detach, detachAll, stackTrace, fullStackTrace, startTrace, stopTrace. The trace action prints out an aggregated trace message. The totalview action detaches DySectAPI from the target processes that triggered the probe and attaches the TotalView debugger to that set of processes. The depositCore action, which requires the libdepositcore library, causes the triggering processes to dump a core file. The signal action sends the specified integer signal to the triggered processes. The loadLibrary action loads the specified shared library into the triggered processes. The irpc action will invoke a function within the target processes. the writeModuleVariable action will modify the contents of a variable. The stat action will gather a Stack Trace Analysis Tool merged stack trace of the application. The detach action will detach DySectAPI from the triggered processes, while the detachAll action will detach DySectAPI from all application processes. The stackTrace and fullStackTrace actions will print an aggregated, text-based stack trace of the triggered processes. The startTrace and stopTrace actions indicate when to start and stop data tracing.

An example DySectAPI session can be seen below:

```text
/* File: session.C */
#include <DysectAPI.h>
#include <stdio.h>

/* Single entry for debug daemon */
DysectStatus DysectAPI::onProcStart(int argc, char **argv) {

  /* Probe creation */
  Probe* entry  = new Probe(Code::location("entry(foo)"),
                            Local::eval("argc >= 5"),
                            Domain::group("12,25,65-70", Wait::inf));

  /* Within 500 ms and call frame has not been left */
  Probe* timer  = new Probe(Event::And(Time::within(500),
                            Event::Not(Async::leaveFrame())),
                            Domain::group("..", 400),
                            Action::trace("Took more than 500ms to return from 
                                          foo()"));

  /* Event chain */
  entry->link(timer);

  /* Setup probe tree */
  entry->setup();

  return DysectOK;
}
```

The session can be compiled with the ` dysectc ` command.

```text
% dysectc session.C    
```

You will then need to run ` stat-cl ` and specify the generated session .so file with the ` stat-cl ` -X=path option, for example:

```text
% stat-cl -X $PWD/libsession.so -C srun mpi_application    
```
