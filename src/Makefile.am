# what flags you want to pass to the C compiler & linker
AM_CFLAGS = -DSTAT_PREFIX=\"$(prefix)\" $(CODE_COVERAGE_CFLAGS)
SWIGFLAGS = -DSTAT_PREFIX=\"$(prefix)\"
AM_CXXFLAGS = -DSTAT_PREFIX=\"$(prefix)\" $(CODE_COVERAGE_CXXFLAGS)
if ENABLE_OPENMP
AM_CXXFLAGS += -DOMP_STACKWALKER
SWIGFLAGS += -DOMP_STACKWALKER
endif
if ENABLE_GDB
AM_CXXFLAGS += -DSTAT_GDB_BE
SWIGFLAGS += -DSTAT_GDB_BE
endif
AM_LDFLAGS = $(CODE_COVERAGE_LIBS)

include_HEADERS = STAT.h STAT_FrontEnd.h STAT_lmonFrontEnd.h STAT_BackEnd.h STAT_timer.h STAT_GraphRoutines.h STAT_IncMRNet.h MRNetSymbolReader.h STAT_CircularLogs.h

if ENABLE_STATBENCH
stat_bin_programs_statbench = STATBenchbin STATBenchD
STATBenchbin_SOURCES = STATBench.C STAT.h STAT_FrontEnd.h STAT_lmonFrontEnd.h STAT_GraphRoutines.h
STATBenchD_SOURCES = STATBenchD.C STAT.h STAT_BackEnd.h STAT_GraphRoutines.h
STATBenchbin_LDADD = libstatfe.la @FELIBS@
STATBenchD_LDADD = libstatbe.la @BELIBS@
endif

if ENABLE_GRAPHLIB20
stat_graphlib_sources = STAT_GraphRoutines.h STAT_GraphRoutines.C
endif

if ENABLE_FGFS
stat_be_la_sources_fgfs = MRNetSymbolReader.C
endif

bin_PROGRAMS = STATbin STATD $(stat_bin_programs_statbench)
lib_LTLIBRARIES = libstatfe.la libstatbe.la

if ENABLE_CALLPATH
lib_LTLIBRARIES += libcallpathwrap.la
libcallpathwrap_la_SOURCES = callpathWrap.h callpathWrap.C
libcallpathwrap_la_CPPFLAGS = -I@ADEPTUTILSPREFIX@/include -I@CALLPATHPREFIX@/include -I@MPIH_HEADERPATH@ -g -O0
libcallpathwrap_la_LDFLAGS = -L@CALLPATHPREFIX@/lib -Wl,-rpath=@CALLPATHPREFIX@/lib -lcallpath -Wl,-init=depositlightcorewrap_init -ldl
endif

STATbin_SOURCES = STAT.C STAT.h STAT_FrontEnd.h STAT_lmonFrontEnd.h STAT_GraphRoutines.h
STATD_SOURCES = STATD.C STAT.h STAT_BackEnd.h STAT_GraphRoutines.h
STATbin_LDADD = libstatfe.la @FELIBS@
STATD_LDADD = libstatbe.la @BELIBS@


stat_be_la_sources_base = STAT_BackEnd.C STAT_BackEnd.h STAT_lmonBackEnd.C STAT_lmonBackEnd.h STAT.h STAT_timer.h STAT_timer.C STAT_CircularLogs.C
stat_fe_la_sources_base = STAT_FrontEnd.C STAT_lmonFrontEnd.C STAT_FrontEnd.h  STAT_lmonFrontEnd.h STAT.h STAT_timer.h STAT_timer.C


libstatbe_la_SOURCES = $(stat_be_la_sources_base) $(stat_graphlib_sources) $(stat_be_la_sources_fgfs)
libstatfe_la_SOURCES = $(stat_fe_la_sources_base) $(stat_graphlib_sources) 

pkglib_LTLIBRARIES = STAT_FilterDefinitions.la
pkglibdir = $(libdir)

STAT_FilterDefinitions_la_SOURCES = STAT_FilterDefinitions.C $(stat_graphlib_sources)
STAT_FilterDefinitions_la_LDFLAGS = -module -avoid-version $(CODE_COVERAGE_LIBS)
STAT_FilterDefinitions_la_LIBADD = @MWLIBS@

pyexec_LTLIBRARIES = _STATmerge.la
_STATmerge_la_SOURCES = STAT_merge.C STAT_GraphRoutines.C
_STATmerge_la_CPPFLAGS = -I./
_STATmerge_la_LDFLAGS = -module -avoid-version
_STATmerge_la_LIBADD = -llnlgraph $(CODE_COVERAGE_LIBS)

python_PYTHON =
if ENABLE_GUI
GUI_SWIG_SOURCES = STAT.i
pyexec_LTLIBRARIES += _STAT.la
python_PYTHON += STAT.py
_STAT_la_SOURCES = STAT_wrap.C $(GUI_SWIG_SOURCES)
_STAT_la_LDFLAGS = -module -avoid-version
_STAT_la_LIBADD = libstatfe.la @FELIBS@ $(CODE_COVERAGE_LIBS)

STAT_wrap.C:$(GUI_SWIG_SOURCES) Makefile
	$(SWIG) -c++ -python $(SWIGFLAGS) -o $@ $<

STAT.py:$(GUI_SWIG_SOURCES) Makefile
	$(MAKE) $(AM_MAKEFLAGS) STAT_wrap.C
endif

#if ENABLE_TO
#TO_SWIG_SOURCES = to.i
#pyexec_LTLIBRARIES += _tomod.la
#python_PYTHON += tomod.py
#_tomod_la_SOURCES = to_wrap.C to.C to.h $(TO_SWIG_SOURCES)
#_tomod_la_CPPFLAGS = -I./ $(BOOST_INCLUDE) $(ROSE_INCLUDE) $(TOAPI_INCLUDE)
#_tomod_la_LDFLAGS = -module -avoid-version -L$(LIBBOOSTDIR) -L$(LIBJAVADIR) -L$(LIBROSEDIR) -L$(LIBTOAPIDIR) $(LIBTOAPI) -Wl,-rpath=$(LIBBOOSTDIR) -Wl,-rpath=$(LIBROSEDIR) -Wl,-rpath=$(LIBTOAPIDIR) -Wl,-rpath=$(LIBJAVADIR)
#
#to_wrap.C:$(TO_SWIG_SOURCES) Makefile
#	$(SWIG) -c++ -python -o $@ $<
#
#tomod.py:$(TO_SWIG_SOURCES) Makefile
#	$(MAKE) $(AM_MAKEFLAGS) to_wrap.C
#endif

clean-local:
	-rm -rf STAT_wrap.C STAT.py to_wrap.C tomod.py
