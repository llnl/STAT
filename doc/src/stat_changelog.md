# Changelog

```{rubric} stat version 4.3.0
:name: stat-version-4-3-0
:class: changelog-version
```

- GUI update to support xdot 1.6 and GTK 3
- added tests
- udpates to cuda-gdb and gdb-oneapi drivers
- restore stat-bench functionality
- updates to build system
- numerous bug fixes

```{rubric} stat version 4.2.2
:name: stat-version-4-2-2
:class: changelog-version
```

- invoke python with -s to prevent user site-packages from interferring
  allow specification of GSETTINGS\_SCHEMA\_DIR for GUI

```{rubric} stat version 4.2.1
:name: stat-version-4-2-1
:class: changelog-version
```

- update to support dropping of local\_var.h from Dyninst 12

```{rubric} stat version 4.2.0
:name: stat-version-4-2-0
:class: changelog-version
```

- Core file debugging enhancements
- Backend refactoring
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 4.1.0
:name: stat-version-4-1-0
:class: changelog-version
```

- DynInst 10.2 support
- Initial rocgdb support
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 4.0.2
:name: stat-version-4-0-2
:class: changelog-version
```

- DynInst 10.X support
- Core file merging enhancements
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 4.0.1
:name: stat-version-4-0-1
:class: changelog-version
```

- Python 3 and xdot 0.9 support added
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 4.0.0
:name: stat-version-4-0-0
:class: changelog-version
```

- Added GDB and Cuda GDB backend support
- Core file merging enhancements
- GUI enhancements
- ability for Dysect API to dump lightweight core via callpath
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 3.0.1
:name: stat-version-3-0-1
:class: changelog-version
```

- Dyninst 9.3.0 support
- added ability to launch multiple daemons per node
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 3.0
:name: stat-version-3-0
:class: changelog-version
```

- added OpenMP OMPD support and ehanced thread display
- added data analytics to DySectAPI
- added DySectAPI probe tree visualizer
- fixes to FGFS and file broadcasting
- STAT now requires graphlib 3.0 to support generic edge labels
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 2.2
:name: stat-version-2-2
:class: changelog-version
```

- most edits for this release were bug fixes and enhancements to the prototype DySectAPI (refer to ChangeLog file in top-level directory for details of changes)
- DySectAPI session added to STAT GUI
- added module offset granularity
- cleaner daemon crash handling
- build system fixes
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} stat version 2.1
:name: stat-version-2-1
:class: changelog-version
```

- added prototype DySectAPI
- added prototype temporal ordering capability
- improved python wrapping, including new python-script command and a python-based test script to run through stat functionality
- modified cut/hide interface to allow for user-defined programming models
- modified dot file naming convention for easier sequencing.  also allow specification of dot filename on gather.
- new process table interface (via file-\>properties)
- added cp location policy
- improved bg/q support, including use of stackwalker timeouts.
- improved fgfs support, including adding bg/q support and re-enabling mrnet failure recovery when fgfs is in use
- better labeling of error tasks in stack traces
- as always, numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)

```{rubric} STAT version 2.0
:name: stat-version-2-0
:class: changelog-version
```

- The capitalized STAT commands have been deprecated in favor of all lowercase commands.  The ` STAT ` command is now ` stat-cl `, ` STATGUI ` is now ` stat-gui `, ` STATview ` is now ` stat-view `, and ` STATBench ` is now ` stat-bench `.
- Added optional build with Fast Global File Status plus target application binary file broadcasting
- Added Python-script-level debugging (Python must be built with -g and preferrably -O0)
- Added ability to manually specify serial processes to attach to
- Added count + representative level of detail
- Added join equivalence class GUI feature
- Added cut text GUI feature
- Added GUI preferences menu item
- (old) DynInst support has been removed.  STAT now strictly requires the Stackwalker component of DynInst.
- Graphlib 2.0 required (removed support for 1.X)
- MRNet version 3 or greater required (removed support for 1.X and 2.X)
- Added ability to manually specify list of processes
- Numerous other bug fixes and minor enhancements (refer to ChangeLog file in top-level directory)
