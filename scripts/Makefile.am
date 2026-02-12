dist_bin_SCRIPTS = run_one_sec
bin_SCRIPTS = STAT stat-cl stat-script stat-core-merger attach-helper
python_PYTHON = STATmain.py stat_merge_base.py STATmerge.py core_file_merger.py bg_core_backtrace.py attach_helper.py STAThelper.py
if ENABLE_GDB
python_PYTHON += stat_cuda_gdb.py cuda_gdb.py gdb.py roc_gdb.py oneapi_gdb.py
endif
if ENABLE_GUI
python_PYTHON += STATGUI.py STATview.py
if ENABLE_PYTHON2
python_PYTHON += xdot.py
endif
bin_SCRIPTS += STATGUI STATview stat-gui stat-view
endif
if ENABLE_STATBENCH
bin_SCRIPTS += STATBench stat-bench
endif
