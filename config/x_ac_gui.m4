AC_DEFUN([X_AC_GUI], [  
  AC_ARG_ENABLE(gui,
    [AS_HELP_STRING([--enable-gui],
      [Enable the compilation of the STAT GUI.]
    )],
    [WITH_GUI=$enableval],
    [WITH_GUI=yes]
  )  
  AM_CONDITIONAL([ENABLE_GUI], [test "$WITH_GUI" = yes])
  if test "$WITH_GUI" = yes
  then
     X_AC_SWIG
     X_AC_GRAPHVIZ
  fi  
  PYTHONPATH=$STAT_PYTHONPATH
  AC_SUBST(PYTHONPATH)
  GSETTINGS_SCHEMA_DIR=$STAT_GSETTINGS_SCHEMA_DIR
  AC_SUBST(GSETTINGS_SCHEMA_DIR)
])
