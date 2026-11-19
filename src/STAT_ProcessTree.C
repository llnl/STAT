#include "STAT_ProcessTree.h"
#include "STAT.h"

#include <climits>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <sstream>

using namespace std;

namespace StatProcessTree
{

bool isProcessStopped(int pid, bool &stopped)
{
    char path[BUFSIZE], line[BUFSIZE], state;
    snprintf(path, BUFSIZE, "/proc/%d/status", pid);
    FILE *status = fopen(path, "r");
    if (status == NULL)
    {
        if (errno == ENOENT)
            errno = ESRCH;
        return false;
    }

    while (fgets(line, sizeof(line), status) != NULL)
    {
        if (sscanf(line, "State: %c", &state) == 1)
        {
            fclose(status);
            stopped = (state == 'T' || state == 't');
            if (state == 'Z' || state == 'X' || state == 'x')
            {
                errno = ESRCH;
                return false;
            }
            return true;
        }
    }
    fclose(status);
    errno = EIO;
    return false;
}

static bool parseIntString(const char *text, int &value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (text == end || parsed <= 0 || parsed > INT_MAX)
        return false;
    value = (int)parsed;
    return true;
}

static void addTaskChildren(int pid, int tid, set<int> &children)
{
    char path[BUFSIZE];
    snprintf(path, BUFSIZE, "/proc/%d/task/%d/children", pid, tid);

    ifstream in(path);
    if (!in.good())
        return;

    int child = 0;
    while (in >> child)
    {
        if (child > 0 && child != pid)
            children.insert(child);
    }
}

static set<int> getDirectChildren(int pid)
{
    set<int> children;
    char taskPath[BUFSIZE];
    snprintf(taskPath, BUFSIZE, "/proc/%d/task", pid);

    DIR *taskDir = opendir(taskPath);
    if (taskDir == NULL)
        return children;

    struct dirent *entry = NULL;
    while ((entry = readdir(taskDir)) != NULL)
    {
        int tid = 0;
        if (parseIntString(entry->d_name, tid))
            addTaskChildren(pid, tid, children);
    }
    closedir(taskDir);

    return children;
}

// Discovery and helper filtering overlap with stat_pystack_process_tree.py because C++ needs
// targets for stop/resume control, while Python independently selects targets for stack collection.
// TODO: share discovery between control and collection so implementation cannot drift apart.

set<int> getProcessTreeDescendants(int pid)
{
    set<int> descendants, pending = getDirectChildren(pid);

    while (!pending.empty())
    {
        int child = *pending.begin();
        pending.erase(pending.begin());
        bool stopped;
        // Match Python traversal: never enter a helper subtree or sample a confirmed exited process.
        if (child == pid || isContainerFilesystemHelper(child) ||
            (!isProcessStopped(child, stopped) && errno == ESRCH) ||
            descendants.insert(child).second == false)
            continue;

        set<int> grandchildren = getDirectChildren(child);
        for (set<int>::iterator grandchildIter = grandchildren.begin(); grandchildIter != grandchildren.end(); grandchildIter++)
        {
            if (*grandchildIter > 0 && descendants.count(*grandchildIter) == 0)
                pending.insert(*grandchildIter);
        }
    }

    return descendants;
}

static string readProcText(int pid, const char *name)
{
    char path[BUFSIZE];
    snprintf(path, BUFSIZE, "/proc/%d/%s", pid, name);

    ifstream in(path);
    if (!in.good())
        return string();

    stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

static string getProcComm(int pid)
{
    string comm = readProcText(pid, "comm");
    while (!comm.empty() && (comm[comm.size() - 1] == '\n' || comm[comm.size() - 1] == '\r'))
        comm.erase(comm.size() - 1, 1);
    return comm;
}

// Duplicates the Python helper filter: C++ must not stop these processes; Python must not sample them.
// Keep exclusions aligned with is_container_filesystem_helper() in stat_pystack_process_tree.py.
bool isContainerFilesystemHelper(int pid)
{
    static const set<string> helpers = {"squashfuse", "fuse-overlayfs", "fuse-shim"};
    string command = readProcText(pid, "cmdline");
    size_t separator = command.find('\0');
    string name = command.substr(0, separator);
    name = name.substr(name.find_last_of('/') + 1);
    // Enroot names its shell shim through argv[0]; application arguments are not process identities.
    if (helpers.count(getProcComm(pid)) != 0 || helpers.count(name) != 0)
        return true;

    // Preserve the stopped shell sentinel seen in container launches; match its complete command.
    return (name == "sh" || name == "dash" || name == "bash") && separator != string::npos &&
           command.substr(separator + 1) == string("-c") + '\0' + "kill -STOP $$ ; exit 0" + '\0';
}

} // namespace StatProcessTree
