AC_DEFUN([X_AC_RSH], [
  AC_ARG_WITH(rsh,
    [AS_HELP_STRING([--with-rsh=cmd],
      [Use cmd as the default remote shell]
    )],
    [withrsh="$withval"],
    [withrsh="rsh ssh"]
  )
  AC_PATH_PROGS(rshcmd,$withrsh,[/usr/bin/$withrsh],/usr/local/bin:/usr/bin:/bin)
])
