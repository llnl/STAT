# Unit Tests

Assuming STAT is installed in `/path/to/stat/install` with Python 3.XY, first set the Python
module search path as:

```sh
export PYTHONPATH="/path/to/stat/install/lib/python3.XY/site-packages:${PYTHONPATH}"
```

Make sure to adjust `python3.XY` to match its version and `lib` to match the installation as
some systems use `lib64`. Use the Python interpreter and runtime environment used to build STAT,
with its Python and native-library dependencies available.

This directory contains tests under `unit/test_*.py` and some of these require installed STAT
Python extensions like `_STATmerge`. Check we can import modules and then run these tests from
the STAT source root as:

```sh
python3 -c 'import _STATmerge'
python3 -m unittest discover -s tests/unit -t .
```

GTK tests need a display; on a headless host, run the discovery command under `xvfb-run -a`, as CI does.
These unit checks do not establish live GUI or cluster acceptance.
Unit tests run from a repository checkout; they are not included in source archives or installed by `make install`.

## Integration tests

The [integration suite](integration/README.md) checks rank membership, call paths and process/thread counts
for native and Python workloads using an installed STAT. It supports local MPI and Slurm profiles,
including Docker, nested Enroot and Slurm/Pyxis execution.
