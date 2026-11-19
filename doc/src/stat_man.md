2018-02-02

# `stat-cl`

1

`stat-cl`

invoke the Stack Trace Analysis Tool.

`stat-cl` [*OPTIONS*] *PID*

`stat-cl` [*OPTIONS*] -C *COMMAND*...

where

[*OPTIONS*]

represents zero or more stat-cl options.
*PID*

is the PID of the parallel job launcher for the target application to attach to.
*COMMAND*...

is the command to launch the application.

## DESCRIPTION

```{include} stat_description.md
:start-after: "## Description"
:end-before: "```{include}"
```

## OPTIONS

```{include} stat_options.md
:start-after: "# stat-cl Options"
```

## EXAMPLE

```{include} stat_example.md
:start-after: "# STAT Usage Example"
```

## AUTHOR

Written by
Gregory
L.
Lee

lee218@llnl.gov

## COPYRIGHT

```{include} stat_copyright.md
```

## SEE ALSO

`stat-gui`(1), `stat-view`(1), `stat-bench`(1)
