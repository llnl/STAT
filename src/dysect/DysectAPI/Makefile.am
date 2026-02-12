# what flags you want to pass to the C compiler & linker
AM_CXXFLAGS = -DSTAT_PREFIX=\"$(prefix)\"
AM_LDFLAGS =
AUTOMAKE_OPTIONS = subdir-objects

include_HEADERS=DysectAPI.h
lib_LTLIBRARIES = libDysectAPIFE.la libDysectAPIBE.la libDysectAPIFD.la

dysectapi_includes = -I$(abs_top_srcdir)/src -I$(abs_top_srcdir)/src/dysect -I$(abs_top_srcdir)/src/dysect/libDysectAPI/include
dysectapi_sources = DysectAPI.h DysectAPI.C
dysectapi_aggregate_sources = Aggregates/Data.C Aggregates/Aggregate.C Aggregates/Location.C Aggregates/CmpAgg.C Aggregates/DescVar.C Aggregates/StrAgg.C Aggregates/RankListAgg.C Aggregates/TimeListAgg.C Aggregates/BucketAgg.C Aggregates/RankBucketAgg.C Aggregates/RankBitmap.C Aggregates/DataStackTrace.C Aggregates/CollectValuesAgg.C Aggregates/CountInvocationsAgg.C Aggregates/AverageAgg.C

libDysectAPIFE_la_SOURCES = DysectAPIFE.C DysectAPIFE.h Aggregates/LocationFE.C Aggregates/DescVarFE.C Aggregates/CmpAggFE.C Aggregates/RankListAggFE.C  $(dysectapi_sources) $(dysectapi_aggregate_sources) Aggregates/TimeListAggFE.C Aggregates/BucketAggFE.C Aggregates/RankBucketAggFE.C Aggregates/DataStackTraceFE.C
libDysectAPIFE_la_CPPFLAGS = -g -O2 -fPIC $(dysectapi_includes) -D__DYSECT_IS_FRONTEND
libDysectAPIFE_la_LDFLAGS = --export-dynamic

libDysectAPIBE_la_SOURCES = DysectAPIBE.C DysectAPIBE.h ProcWait.C ProcWait.h ProcMap.C ProcMap.h DysectAPIProcessMgr.C Aggregates/AggregateBE.C Aggregates/LocationBE.C Aggregates/DescVarBE.C Aggregates/CmpAggBE.C Aggregates/RankListAggBE.C Aggregates/TimeListAggBE.C Aggregates/BucketAggBE.C Aggregates/RankBucketAggBE.C Aggregates/DataStackTraceBE.C TraceAPIInstr.C TraceAPIInstr.h $(dysectapi_sources) $(dysectapi_aggregate_sources)
libDysectAPIBE_la_CPPFLAGS = -g -O2 -fPIC $(dysectapi_includes) -D__DYSECT_IS_BACKEND
libDysectAPIBE_la_LDFLAGS = --export-dynamic -lrt

libDysectAPIFD_la_SOURCES = DysectAPIFilter.C DysectAPIFilter.h Aggregates/LocationFE.C Aggregates/DescVarCP.C Aggregates/CmpAggFE.C Aggregates/RankListAggFE.C $(dysectapi_aggregate_sources) Aggregates/TimeListAggFE.C  Aggregates/BucketAggFE.C Aggregates/RankBucketAggFE.C Aggregates/DataStackTraceFE.C
libDysectAPIFD_la_CPPFLAGS = -g -O2 -fPIC $(dysectapi_includes)
libDysectAPIFD_la_LDFLAGS = --export-dynamic -lrt

if ENABLE_DEPCORE
lib_LTLIBRARIES += libdepositcorewrap.la
libdepositcorewrap_la_SOURCES = libdepositcoreWrap.C
libdepositcorewrap_la_CPPFLAGS = -g -O0 -fPIC -I$(DEPCOREPREFIX)/include 
libdepositcorewrap_la_LDFLAGS = -L$(DEPCOREPREFIX)/lib -Wl,-rpath=$(DEPCOREPREFIX)/lib -Wl,-init=depositcorewrap_init -ldl -ldepositcore
endif
