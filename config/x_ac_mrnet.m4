AC_DEFUN([X_AC_MRNET], [
  AC_LANG_PUSH(C++)
  AC_ARG_WITH(mrnet,
    [AS_HELP_STRING([--with-mrnet=prefix],
      [Add the compile and link search paths for mrnet]
    )],
    [CXXFLAGS="$CXXFLAGS -I${withval}/include"
      MRNETPREFIX="${withval}"
      LDFLAGS="$LDFLAGS -L${withval}/lib"
      RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}/lib"
    ],
    [CXXFLAGS="$CXXFLAGS"
      MRNETPREFIX=""
    ]
  )
  AC_ARG_WITH(extra-mrnet-rpath,
    [AS_HELP_STRING([--with-extra-mrnet-rpath=path],
      [Add additional rpath fro mrnet]
    )],
    [RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath=${withval}"],
    []
  )
  AC_MSG_CHECKING([setting the RPATH_FLAGS for mrnet])
  AC_MSG_RESULT([$RPATH_FLAGS])

  mrn_incs=`ls -d $MRNETPREFIX/lib/*/include` 
  for mrn_inc in $mrn_incs
  do
    CXXFLAGS="-I$mrn_inc $CXXFLAGS"
  done
  AC_CHECK_HEADER(mrnet/MRNet.h,
    [],
    [AC_MSG_ERROR([mrnet/MRNet.h is required.  Specify mrnet prefix with --with-mrnet])],
    AC_INCLUDES_DEFAULT
  )
  AC_MSG_CHECKING([Checking MRNet Version])
  mrnet_vers=-1
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([#include "mrnet/MRNet.h"
    using namespace MRN;
    using namespace std;
    int main()
    {
      uint64_t bufLength;
      DataType type;
      DataElement pkt;
      pkt.get_array(&type, &bufLength);
    }])],
    [AC_DEFINE([MRNET40], [], [MRNet 4.0])
      mrnet_vers=4.0
    ]
  )
  if test $mrnet_vers = -1; then
    AC_MSG_ERROR([STAT requires MRNet 4.0 or greater. Specify MRNet prefix with --with-mrnet])
  fi
  AC_MSG_RESULT([$mrnet_vers])
  AC_MSG_CHECKING(for libmrnet)
  TMP_LDFLAGS=$LDFLAGS
  AC_ARG_WITH(cray-xt,
    [],
    [mrnet_libs="-lmrnet -lxplat -lpthread -ldl -lalps -lalpsutil"],
    [mrnet_libs="-lmrnet -lxplat -lpthread -ldl"]
  )  
  LDFLAGS="$LDFLAGS $mrnet_libs"
  AC_LINK_IFELSE([AC_LANG_PROGRAM(#include "mrnet/MRNet.h"
    using namespace MRN;
    Network *net;)],
    [libmrnet_found=yes],
    [libmrnet_found=no]
  )
  LDFLAGS=$TMP_LDFLAGS
  AC_MSG_RESULT($libmrnet_found)
  if test "$libmrnet_found" = yes; then
    FELIBS="$FELIBS $mrnet_libs"
    BELIBS="$BELIBS $mrnet_libs"
  else
    AC_MSG_ERROR([libmrnet is required.  Specify libmrnet prefix with --with-mrnet])
  fi
  AC_LANG_POP(C++)
  AC_PATH_PROG([MRNETCOMMNODEBIN], [mrnet_commnode], [no], [$MRNETPREFIX/bin$PATH_SEPARATOR$MRNETPREFIX/libexec$PATH_SEPARATOR$PATH])
  if test $MRNETCOMMNODEBIN = no; then
    AC_MSG_ERROR([the mrnet_commnode executable is required.  Specify mrnet prefix with --with-mrnet])
  fi
])
AC_SUBST(MRNETCOMMNODEBIN)
