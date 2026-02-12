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
  AC_MSG_CHECKING([Python include dir])
  if test -z "$PYTHONINCLUDE"; then
    python_header_path=`$PYTHON -c "import distutils.sysconfig; \
      print(distutils.sysconfig.get_python_inc());"`
    AC_MSG_RESULT($python_header_path)
    PYTHONINCLUDE=$python_header_path
    CXXFLAGS="$CXXFLAGS -I$PYTHONINCLUDE/"
  fi
  AC_MSG_CHECKING([Python lib dir])
  if test -z "$PYTHONLIB"; then
    python_lib_path=`$PYTHON -c "import distutils.sysconfig; \
      print(distutils.sysconfig.get_python_lib());"`
    AC_MSG_RESULT($python_lib_path)
    PYTHONLIB=$python_lib_path
    LDFLAGS="$CXXFLAGS -L$PYTHONLIB/../../ -Wl,-rpath=$PYTHONLIB/../../"
  fi
  AC_MSG_CHECKING([Python version])
  python_version=`$PYTHON -c "import distutils.sysconfig; \
    print(distutils.sysconfig.get_python_version());"`
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
])
