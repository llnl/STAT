AC_DEFUN([X_AC_DEBUGLIBS], [
  [CXXFLAGS="$CXXFLAGS -std=c++11"]
  AC_ARG_ENABLE(stackwalker-rpm,
    [AS_HELP_STRING([--enable-stackwalker-rpm],[Enable the use of rpm-installed stackwalker, default=no])],
    [CXXFLAGS="$CXXFLAGS -I/usr/include/dyninst"
     LDFLAGS="$LDFLAGS -L/usr/lib64/dyninst"
     STACKWALKERPREFIX="${withval}"
     RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=/usr/lib64/dyninst"],
    [CXXFLAGS="$CXXFLAGS"
     STACKWALKERPREFIX="${withval}"]
  )

  AC_ARG_WITH(elfutils,
    [AS_HELP_STRING([--with-elfutils=prefix],
      [Add the compile and link search paths for elfutils]
    )],
    [CXXFLAGS="$CXXFLAGS -I${withval}/include"
     LDFLAGS="-L${withval}/lib -ldw $LDFLAGS"
     ELFUTILSPREFIX="${withval}"
     RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}/lib"],
    [CXXFLAGS="$CXXFLAGS"
     ELFUTILSPREFIX="${withval}"]
  )

  AC_ARG_WITH(stackwalker,
    [AS_HELP_STRING([--with-stackwalker=prefix],
      [Add the compile and link search paths for stackwalker]
    )],
    [CXXFLAGS="$CXXFLAGS -I${withval}/include"
     LDFLAGS="$LDFLAGS -L${withval}/lib"
     STACKWALKERPREFIX="${withval}"
     RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}/lib"],
    [CXXFLAGS="$CXXFLAGS"
     STACKWALKERPREFIX="${withval}"]
  )

  AC_ARG_WITH(ompd,
    [AS_HELP_STRING([--with-ompd=prefix],
      [Add the compile and link search paths for ompd]
    )],
    [CXXFLAGS="$CXXFLAGS -I${withval}/include"
     LDFLAGS="$LDFLAGS -L${withval}/lib"
     RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}/lib"],
    [CXXFLAGS="$CXXFLAGS"]
  )
  AM_CONDITIONAL([ENABLE_OPENMP], false)
  AC_ARG_WITH(omp-stackwalker,
    [AS_HELP_STRING([--with-omp-stackwalker=prefix],
      [Add the compile and link search paths for openmp stack walker. Must have --with-stackwalker and --with-ompd defined]
    )],
    [CXXFLAGS="$CXXFLAGS -I${withval}/include -DOMP_STACKWALKER"
     LDFLAGS="$LDFLAGS -L${withval}/lib"
     RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}/lib"
     AM_CONDITIONAL([ENABLE_OPENMP], true)],
    [CXXFLAGS="$CXXFLAGS"]
  )

  AC_LANG_PUSH(C++)
  AC_CHECK_HEADER(walker.h,
    [],
    [AC_MSG_ERROR([walker.h is required.  Specify stackwalker prefix with --with-stackwalker])],
    AC_INCLUDES_DEFAULT
  )

  AC_MSG_CHECKING([Checking Dyninst Version 9.3 or greater])
  dyninst_vers_93=no
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([#include "version.h"
    int main()
    {
      return 0;
    }])],
    [dyninst_vers_93=yes],
    []
  )

  if test "$dyninst_vers_93" = yes; then
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([#include "version.h"
      #if ((DYNINST_MAJOR_VERSION == 9 && DYNINST_MINOR_VERSION < 3) || DYNINST_MAJOR_VERSION <= 8)
        #error
      #endif
      int main()
      {
        return 0;
      }])],
      [CXXFLAGS="$CXXFLAGS -std=c++11"],
      [dyninst_vers_93=no]
    )
  fi
  AC_MSG_RESULT([$dyninst_vers_93])

  AC_MSG_CHECKING([Checking Dyninst Version 10.0 or greater])
  dyninst_vers_10=no
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([#include "dyninstversion.h"
    int main()
    {
      return 0;
    }])],
    [dyninst_vers_10=yes],
    []
  )

  if test "$dyninst_vers_10" = yes; then
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([#include "dyninstversion.h"
      #if DYNINST_MAJOR_VERSION < 10
        #error
      #endif
      int main()
      {
        return 0;
      }])],
      [CXXFLAGS="$CXXFLAGS -std=c++11"],
      [dyninst_vers_10=no]
    )
  fi
  AC_MSG_RESULT([$dyninst_vers_10])

  AC_CHECK_HEADER(local_var.h,
    [AC_DEFINE([LOCAL_VAR_H], [], [local_var.h])],
    [],
    AC_INCLUDES_DEFAULT
  )

  AC_CHECK_HEADER(Symtab.h,
    [],
    [AC_MSG_ERROR([Symtab.h is required.  Specify prefix with --with-stackwalker])],
    AC_INCLUDES_DEFAULT
  )
  AC_MSG_CHECKING(for libstackwalk)
  TMP_LDFLAGS=$LDFLAGS
  LDFLAGS="$LDFLAGS -ldyninstAPI -lstackwalk -lpcontrol -lparseAPI -linstructionAPI -lsymtabAPI -lcommon -ldynElf -ldynDwarf -lsymLite -lpthread"
  AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "walker.h"
    using namespace Dyninst;
    using namespace Dyninst::Stackwalker;
    Walker *walker;)],
    [libstackwalk_found=yes],
    [libstackwalk_found=no]
  )
  LDFLAGS=$TMP_LDFLAGS
  if test "$libstackwalk_found" = yes; then
    BELIBS="-ldyninstAPI -lstackwalk -lpcontrol -lparseAPI -linstructionAPI -lsymtabAPI -lcommon -ldynElf -ldynDwarf -lsymLite $BELIBS"
  else
    LDFLAGS="$LDFLAGS -lstackwalk -lsymtabAPI -lpcontrol -lparseAPI -linstructionAPI -lcommon -lpthread"
    AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "walker.h"
      using namespace Dyninst;
      using namespace Dyninst::Stackwalker;
      Walker *walker;)],
      [libstackwalk_found=yes],
      [libstackwalk_found=no]
    )
    LDFLAGS=$TMP_LDFLAGS
    if test "$libstackwalk_found" = yes; then
      BELIBS="-ldyninstAPI -lstackwalk -lsymtabAPI -lpcontrol -lparseAPI -linstructionAPI -lcommon $BELIBS"
    else
      LDFLAGS="$LDFLAGS -lstackwalk -lsymtabAPI -lcommon -lpthread"
      AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "walker.h"
        using namespace Dyninst;
        using namespace Dyninst::Stackwalker;
        Walker *walker;)],
        [libstackwalk_found=yes],
        [libstackwalk_found=no]
      )
      LDFLAGS=$TMP_LDFLAGS
      if test "$libstackwalk_found" = yes; then
        BELIBS="-ldyninstAPI -lstackwalk -lsymtabAPI -lcommon $BELIBS"
      else
        AC_MSG_ERROR([libstackwalk is required.  Specify libstackwalk prefix with --with-stackwalker])
      fi
    fi
  fi
  AC_SUBST(STACKWALKERPREFIX)
  AC_MSG_RESULT($libstackwalk_found)

  AC_MSG_CHECKING(for libompd_intel)
  TMP_LDFLAGS=$LDFLAGS
  LDFLAGS="$LDFLAGS -lompd_intel"
  AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "ompd.h"
    ompd_callbacks_t *table;)],
    [libompd_found=yes],
    [libompd_found=no]
  )
  LDFLAGS=$TMP_LDFLAGS
  if test "$libompd_found" = yes; then
    BELIBS="-lompd_intel $BELIBS"
  fi
  AC_MSG_RESULT($libompd_found)

  AC_MSG_CHECKING(for libomp_stackwalker)
  TMP_LDFLAGS=$LDFLAGS
  LDFLAGS="$LDFLAGS -lomp_stackwalker $BELIBS"
  AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "OpenMPStackWalker.h"
    OpenMPStackWalker* walker;)],
    [libomp_stackwalker_found=yes],
    [libomp_stackwalker_found=no]
  )
  LDFLAGS=$TMP_LDFLAGS
  if test "$libomp_stackwalker_found" = yes; then
    BELIBS="-lomp_stackwalker $BELIBS"
  fi
  AC_MSG_RESULT($libomp_stackwalker_found)

  AC_LANG_POP(C++)
])
