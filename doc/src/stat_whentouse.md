# When to use STAT

STAT is most effective for diagnosing parallel applications that are hung (i.e., deadlock or livelock), however, its analysis can also be useful for debugging other errors.  The outputted, merged stack traces indicate where in the code all of the processes are at a given point in time, giving the user insight into where the bug may be.  The merged traces also form process equivalence classes, which can be used to select a subset of tasks to feed into a full-featured parallel debugger such as DDT or TotalView for root cause analysis.
