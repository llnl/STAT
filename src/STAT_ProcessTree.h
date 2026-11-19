#ifndef STAT_PROCESS_TREE_H
#define STAT_PROCESS_TREE_H

#include <set>
#include <string>

// Linux process inspection for daemon sampling and process control.
namespace StatProcessTree
{
    //! Return application descendants, excluding exited processes and filesystem-helper subtrees.
    std::set<int> getProcessTreeDescendants(int pid);

    //! Read whether Linux reports a stopped target; return false with errno on failure.
    bool isProcessStopped(int pid, bool &stopped);

    //! Identify container filesystem helpers that must remain running during sampling.
    bool isContainerFilesystemHelper(int pid);
}

#endif
