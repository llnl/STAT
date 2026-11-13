AC_DEFUN([X_AC_GDB], [
  AM_CONDITIONAL([ENABLE_GDB], false)
  AC_ARG_ENABLE(gdb,
    [AS_HELP_STRING([--enable-gdb],[Enable to include GDB BE prototype])],
    [ENABLE_GDB="yes"
      AM_CONDITIONAL([ENABLE_GDB], true)
    ]
  )

  AC_ARG_WITH(gdb,
    [AS_HELP_STRING([--with-gdb=path],
      [Use path for gdb backend]
    )],
    [withgdb="$withval"
      AM_CONDITIONAL([ENABLE_GDB], true)
    ],
    [withgdb=gdb]
  )
  AC_PATH_PROG(gdbcmd,$withgdb,[gdb],/,/usr/local/bin:/usr/bin:/bin/usr/local/bin)

])
