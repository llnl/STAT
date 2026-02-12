AC_DEFUN([X_AC_BOOST], [

  AC_MSG_CHECKING([boost installation])

  AC_ARG_WITH([boost],
    AS_HELP_STRING(--with-boost@<:@=boost installation path@:>@,specify the path to boost @<:@default=/usr@:>@), 
    [with_boost_path=$withval],
    [with_boost_path="check"]
  )

  boost_dflt_dir="/usr"
  boost_found="no" 
  if test "x$with_boost_path" != "xcheck"; then
    #boost root given
    #TODO: remove dependency on 1_37 below
    #
    if test -f "$with_boost_path"/include/boost/algorithm/string.hpp ; then
      AC_SUBST(BOOST_INCLUDE, -I$with_boost_path/include)
      if test -f "$with_boost_path"/lib64/libboost_timer.so ; then
        AC_SUBST(LIBBOOSTDIR, [$with_boost_path/lib64])
      else
        AC_SUBST(LIBBOOSTDIR, [$with_boost_path/lib])
      fi
      AC_SUBST(LIBBOOST,["-lboost_date_time -lboost_thread -lboost_filesystem -lboost_program_options -lboost_regex -lboost_system -lboost_system -lboost_wave"])
      AC_DEFINE(HAVE_BOOST_TO,1,[Define 1 if a compatible boost package is found])	
      boost_found="yes"
    elif test -f "$with_boost_path"/include/boost-1_37/boost/algorithm/string.hpp ; then 
      AC_SUBST(BOOST_INCLUDE, -I$with_boost_path/include/boost-1_37)
      AC_SUBST(LIBBOOSTDIR, [$with_boost_path/lib])
      AC_SUBST(LIBBOOST,["-lboost_date_time -lboost_thread -lboost_filesystem -lboost_program_options -lboost_regex -lboost_system -lboost_system -lboost_wave"])
      AC_DEFINE(HAVE_BOOST_TO,1,[Define 1 if a compatible boost package is found])	
      boost_found="yes $LIBBOOSTDIR"
    else
      boost_found="no"
    fi
  else 
    # boost path not given
    if test -f $boost_dflt_dir/include/boost/algorithm/string.hpp ; then
      AC_SUBST(BOOST_INCLUDE, -I$boost_dflt_dir/include)
      AC_DEFINE(HAVE_BOOST_TO,1,[Define 1 if a compatible boost package is found])	
      if test -f $boost_dflt_dir/lib64/libboost_timer.so ; then
        AC_SUBST(LIBBOOSTDIR, [$boost_dflt_dir/lib64])
      else
        AC_SUBST(LIBBOOSTDIR, [$boost_dflt_dir/lib])
      fi
      AC_SUBST(LIBBOOST,["-lboost_date_time -lboost_thread -lboost_filesystem -lboost_program_options -lboost_regex -lboost_system -lboost_system -lboost_wave"])
      boost_found="yes $LIBBOOSTDIR"
    else
      boost_found="no"
    fi
  fi
  if test "$boost_found" = no; then
    AC_MSG_ERROR([boost is required. Specify boost prefix with --with-boost])
  fi
  CXXFLAGS="$CXXFLAGS $BOOST_INCLUDE"
  LDFLAGS="$LDFLAGS -L$LIBBOOSTDIR -Wl,-rpath=$LIBBOOSTDIR $LIBBOOST"
  AC_MSG_RESULT($boost_found)
])
