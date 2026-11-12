AC_DEFUN([X_AC_PYTHON], [
  AM_CONDITIONAL([ENABLE_PYTHON2], false)
  AC_MSG_CHECKING([Python executable])
  AC_ARG_WITH(python,
    [AS_HELP_STRING([--with-python=path],
      [Use the specified path to python, or specify via PYTHON env var]
    )],
    [PYTHON=${withval}],
    [PYTHON=$PYTHON]
  )
  if test -z "$PYTHON"; then
    AC_CHECK_PROGS(PYTHON, python python3, "no")
    if test "$PYTHON" = "no"; then
      AC_MSG_ERROR([Failed to find python or python3 in your path])
    else
      AC_MSG_RESULT($PYTHON)
    fi
  else  
    AC_MSG_RESULT($PYTHON)
  fi
  STATPYTHON=$PYTHON
  AC_ARG_WITH(stat-python,
    [AS_HELP_STRING([--with-stat-python=path],
      [Use the specified path to python in the STAT executable scripts]
    )],
    [STATPYTHON=${withval}],
    [STATPYTHON=$STATPYTHON]
  )
  AC_ARG_WITH(python-path,
    [AS_HELP_STRING([--with-python-python=paths],
      [Use the specified paths for PYTHONPATH in the STAT executable scripts]
    )],
    [STAT_PYTHONPATH=${withval}],
    [STAT_PYTHONPATH=$STAT_PYTHONPATH]
  )
  PYTHONPATH=$STAT_PYTHONPATH
  AC_SUBST(PYTHONPATH)
  AC_MSG_CHECKING([Python include dir])
  if test -z "$PYTHONINCLUDE"; then
    python_header_path=`$PYTHON -c "import sysconfig; \
      print(sysconfig.get_path('include'));"`
    AC_MSG_RESULT($python_header_path)
    PYTHONINCLUDE=$python_header_path
    CXXFLAGS="$CXXFLAGS -I$PYTHONINCLUDE/"
  fi
  AC_MSG_CHECKING([Python lib dir])
  if test -z "$PYTHONLIB"; then
    python_lib_path=`$PYTHON -c "import sysconfig; \
      print(sysconfig.get_config_var('LIBDIR'));"`
    AC_MSG_RESULT($python_lib_path)
    PYTHONLIB=$python_lib_path
    LDFLAGS="$LDFLAGS -L$PYTHONLIB -Wl,-rpath=$PYTHONLIB"
  fi
  AC_MSG_CHECKING([Python version])
  python_version=`$PYTHON -c "import platform; \
    print('{0}.{1}'.format(*platform.python_version_tuple()));"`
  python_minor_version=`echo $python_version | sed 's/.*\.\(.*\)/\1/'`
  if test $python_version '>' 2.99 ; then
    if test $python_minor_version -lt 8 ; then
      m="m"
      python_version=$python_version$m
    else
      python_version=$python_version
    fi  
  else
    AM_CONDITIONAL([ENABLE_PYTHON2], true)
  fi
  AC_MSG_RESULT($python_version)
  AM_COND_IF([ENABLE_GDB], [BELIBS="-lpython$python_version $BELIBS"], [])

  enable_pystack=no
  pystackcmd=""
  AC_ARG_WITH(pystack,
    [AS_HELP_STRING([--with-pystack=path],
      [Use explicit pystack executable path for PyStack backend]
    )],
    [if test "x$withval" = "xno"; then
      pystackcmd=""
    elif test "x$withval" = "xyes" || test "x$withval" = "x"; then
      AC_MSG_ERROR([--with-pystack requires an explicit path; use --with-pystack=/path/to/pystack or --without-pystack])
    elif test ! -f "$withval" || test ! -x "$withval"; then
      AC_MSG_ERROR([pystack executable not found or not executable: $withval])
    else
      pystackcmd="$withval"
      enable_pystack=yes
      BELIBS="-lpython$python_version $BELIBS"
    fi
    ]
  )
  AM_CONDITIONAL([ENABLE_PYSTACK], [test "x$enable_pystack" = "xyes"])
  AC_SUBST(pystackcmd)
])
