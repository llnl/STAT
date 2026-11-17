AC_DEFUN([X_AC_BUILD_INFO], [
  AC_ARG_VAR([STAT_BUILD_COMMIT], [Source Git commit, supplied when building a copied source tree])
  AC_ARG_VAR([STAT_BUILD_DIRTY], [Whether source had tracked changes: true, false, or unknown])

  AC_MSG_CHECKING([for source build provenance])

  # Explicit provenance takes precedence. Git is optional, and source archives must
  # not inherit the revision of an enclosing repository.
  if test -z "$STAT_BUILD_COMMIT" && test -e "$srcdir/.git" && command -v git >/dev/null 2>&1; then
    STAT_BUILD_COMMIT=`git -C "$srcdir" rev-parse --verify HEAD 2>/dev/null` || STAT_BUILD_COMMIT=

    # Detect tracked changes only for the revision discovered above. Porcelain status
    # handles timestamp-only changes; a failed probe leaves the dirty state unknown.
    if test -n "$STAT_BUILD_COMMIT" && test -z "$STAT_BUILD_DIRTY"; then
      if stat_build_status=`git -C "$srcdir" status --porcelain \
          --untracked-files=no 2>/dev/null`; then
        case "$stat_build_status" in
          "") STAT_BUILD_DIRTY=false ;;
          *)  STAT_BUILD_DIRTY=true ;;
        esac
      fi
    fi
  fi

  # Missing Git, source archives and failed probes all have unknown provenance.
  : "${STAT_BUILD_COMMIT:=unknown}"
  : "${STAT_BUILD_DIRTY:=unknown}"
  AC_MSG_RESULT([$STAT_BUILD_COMMIT (modified: $STAT_BUILD_DIRTY)])

  # Share the display suffix between native and Python tools; keep the full revision above.
  stat_build_label=`printf '%.12s' "$STAT_BUILD_COMMIT"`
  if test "$STAT_BUILD_DIRTY" = true; then
    stat_build_label="$stat_build_label|modified"
  fi
  STAT_BUILD_SUFFIX=" @<:@$stat_build_label@:>@"
  AC_SUBST([STAT_BUILD_SUFFIX])
  AC_DEFINE_UNQUOTED([STAT_BUILD_SUFFIX], ["$STAT_BUILD_SUFFIX"], [Build provenance for version displays])

  AC_SUBST([STAT_BUILD_COMMIT])
  AC_SUBST([STAT_BUILD_DIRTY])
])
