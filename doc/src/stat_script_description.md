# Using the stat-script Python Interface

## Description

` stat-script ` is a Python interface for STAT. When invoked without any arguments, this will create an interactive Python environment with the appropriate paths set to import the ` STAT ` module. You can run ` dir(STAT) ` to see the available functions, flags, etc. running ` help() ` (i.e., ` help(STAT.attach) `) will display the description of the function as well as the arguments required. Like the ` python ` command, ` stat-script ` can also be supplied with a python script file to execute. You may refer to examples/scripts/script\_test.py in the STAT source or share/STAT/examples/bin/script\_test.py in the installation directory for an example of how the stat scripting interface can be used.
