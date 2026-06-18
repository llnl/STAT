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
