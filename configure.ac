#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_COPYRIGHT([Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC.])
AC_INIT([STAT], [4.2.2])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_SRCDIR([configure.ac])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([config])
AM_INIT_AUTOMAKE
AC_CANONICAL_HOST
AX_CODE_COVERAGE

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_FC
AC_PROG_INSTALL
AC_DISABLE_STATIC
AC_PROG_LIBTOOL
X_AC_RSH
X_AC_GDB

# Checks for libraries.
X_AC_TEMPORALORDERINGAPI
X_AC_PYTHON
AM_PATH_PYTHON
X_AC_GUI
X_AC_BOOST
X_AC_ARCH
X_AC_DEBUGLIBS
X_AC_GRAPHLIB
X_AC_LAUNCHMON
X_AC_RM_COMM
X_AC_MRNET
X_AC_FGFS
X_AC_CALLPATH

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([arpa/inet.h fcntl.h stdlib.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_SIZE_T
AC_HEADER_TIME
X_AC_ENABLE_DEBUG

AC_ARG_ENABLE(rpath,
  [AS_HELP_STRING([--enable-rpath],[Enable the use of rpaths, default=yes])],
  [WITH_RPATH=$enableval],
  [WITH_RPATH=yes]
)
if test "$WITH_RPATH" = "yes"
then
  LDFLAGS="$RPATH_FLAGS $LDFLAGS"
else
  LDFLAGS="$LDFLAGS"
fi

AC_ARG_ENABLE(statbench,
  [AS_HELP_STRING([--enable-statbench],[Enable the compilation of STATBench, the STAT performance emulator])],
  [WITH_STATBENCH=$enableval],
  [WITH_STATBENCH=no]
)

AM_CONDITIONAL([ENABLE_STATBENCH], [test "$WITH_STATBENCH" = yes])

AC_ARG_WITH(procspernode,
  [AS_HELP_STRING([--with-procspernode=num],[Set the maximum number of communication processes per node equal to num, defaults to 1 if not specified])],
  [AC_DEFINE_UNQUOTED([STAT_PROCS_PER_NODE], [$withval], [The max number of CPs per node])],
  [AC_DEFINE([STAT_PROCS_PER_NODE], [1], [The max number of CPs per node])]
)

AC_ARG_WITH(usagelog,
  [AS_HELP_STRING([--with-usagelog=path],[Enable usage logging and write log file to path])],
  [CXXFLAGS="$CXXFLAGS -DSTAT_USAGELOG=\\\"${withval}\\\""],
  [CXXFLAGS="$CXXFLAGS"]
)

AC_ARG_WITH(alias-suffix,
  [AS_HELP_STRING([--with-alias-suffix=suffix],[drop suffix from the hostname alias])],
  [CXXFLAGS="$CXXFLAGS -DSTAT_ALIAS_SUFFIX=\\\"${withval}\\\""],
  [CXXFLAGS="$CXXFLAGS"]
)

AC_ARG_ENABLE(examples,
  [AS_HELP_STRING([--enable-examples],[Enable the compilation of example test codes])],
  [WITH_EXAMPLES=$enableval],
  [WITH_EXAMPLES=yes]
)
AM_CONDITIONAL([ENABLE_EXAMPLES], false)
if test "$WITH_EXAMPLES" = "yes"
then
  AM_CONDITIONAL([ENABLE_EXAMPLES], true)
  X_AC_MPICC
fi

# Checks for library functions.
#AC_FUNC_MALLOC
#AC_FUNC_REALLOC
AC_FUNC_SELECT_ARGTYPES
AC_FUNC_STRFTIME
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([getcwd gethostname gettimeofday mkdir mkfifo pow select strdup getrlimit setrlimit])

AC_SUBST(CXX)
AC_SUBST(CXXFLAGS)
AC_SUBST(LDFLAGS)
AC_SUBST(FELIBS)
AC_SUBST(BELIBS)
AC_SUBST(MWLIBS)
AC_SUBST(MRNETCOMMNODEBIN)
AC_SUBST(DOTBINDIR)
AC_SUBST(LAUNCHMONBIN)
AC_SUBST(NEWLAUNCHMONBIN)
AC_SUBST(LAUNCHMONPREFIX)
AC_SUBST(GRAPHLIBPREFIX)
AC_SUBST(STACKWALKERPREFIX)
AC_SUBST(DEPCOREPREFIX)
AC_SUBST(STATPYTHON)

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 examples/Makefile
                 examples/src/Makefile
                 examples/scripts/Makefile
                 doc/Makefile
                 doc/userguide/Makefile
                 doc/quickstart/Makefile
                 etc/Makefile
                 man/Makefile
                 scripts/STAT
                 scripts/stat-cl
                 scripts/Makefile])

if test "$WITH_GUI" = "yes"
then
  AC_CONFIG_FILES([scripts/STATGUI
                   scripts/STATview
                   scripts/stat-script
                   scripts/stat-gui
                   scripts/stat-view])
fi

if test "$WITH_STATBENCH" = "yes"
then
  AC_CONFIG_FILES([scripts/STATBench
                   scripts/stat-bench])
fi

AC_CONFIG_FILES([scripts/stat-core-merger])
AC_CONFIG_FILES([scripts/attach-helper])

AC_OUTPUT
