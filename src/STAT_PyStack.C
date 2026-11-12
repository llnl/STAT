/**
 * STAT daemon integration for PyStack sampling.
 *
 * This file stops and resumes application process trees, calls the Python adapter,
 * and converts its returned traces into STAT graphs. The adapter handles PyStack
 * execution, output parsing, and Python application discovery.
 */

#include "STAT_BackEnd.h"
#include "STAT_ProcessTree.h"

#include <unistd.h>
#include <chrono>

using namespace std;

// --- Process stop and resume helpers ---

/** Stop one target and wait up to one second for Linux to confirm its state.
 * changed records whether STAT sent SIGSTOP, even if confirmation later fails;
 * callers use it to undo their own stops without releasing pre-existing ones.
 */
static bool stopPyStackProcess(int pid, bool &changed)
{
    bool stopped;
    changed = false;
    if (!StatProcessTree::isProcessStopped(pid, stopped))
        return false;
    if (stopped)
        return true;

    if (kill(pid, SIGSTOP) == -1)
        return false;
    changed = true;

    const auto deadline = chrono::steady_clock::now() + chrono::seconds(1);
    do
    {
        if (!StatProcessTree::isProcessStopped(pid, stopped))
            return false;
        if (stopped)
            return true;
        usleep(1000);
    } while (chrono::steady_clock::now() < deadline);

    errno = ETIMEDOUT;
    return false;
}


/** Continue a target, treating an already-exited process as successful cleanup.
 * Include the caller's context in diagnostics when SIGCONT fails.
 */
bool STAT_BackEnd::continuePyStackProcess(int pid, const char *context)
{
    if (kill(pid, SIGCONT) == 0 || errno == ESRCH)
        return true;

    printMsg(STAT_WARNING, __FILE__, __LINE__, "Failed to SIGCONT pid %d %s: %s\n", pid, context, strerror(errno));
    return false;
}


/** Stop eligible descendants of each root and record which stops STAT owns.
 * Skip roots and previously tracked descendants; remember the owning rank so
 * detach can honor requests to leave that rank's process tree stopped.
 */
StatError_t STAT_BackEnd::stopDescendantProcessesForSampling()
{
    set<int> rootPids;
    for (unsigned int i = 0; i < proctabSize_; i++)
        rootPids.insert(proctab_[i].pid);

    for (unsigned int i = 0; i < proctabSize_; i++)
    {
        set<int> descendants = StatProcessTree::getProcessTreeDescendants(proctab_[i].pid);
        for (set<int>::iterator descendantIter = descendants.begin(); descendantIter != descendants.end(); descendantIter++)
        {
            int childPid = *descendantIter;
            if (rootPids.count(childPid) != 0 || stoppedDescendants_.count(childPid) != 0)
                continue;

            bool changed;
            bool stopped = stopPyStackProcess(childPid, changed);

            // Remember only our stops, including a queued stop whose confirmation timed out.
            if (changed)
                stoppedDescendants_[childPid] = proctab_[i].mpirank;

            if (!stopped)
            {
                if (errno == ESRCH)
                {
                    stoppedDescendants_.erase(childPid);
                    continue;
                }
                printMsg(STAT_PAUSE_ERROR, __FILE__, __LINE__, "Failed to stop descendant pid %d of target pid %d: %s\n", childPid, proctab_[i].pid, strerror(errno));
                return STAT_PAUSE_ERROR;
            }
        }
    }

    return STAT_OK;
}


/** Release descendant stops owned by STAT, retaining failed cleanup for retry.
 * leaveStoppedRanks transfers those retained stops to the user during detach.
 */
StatError_t STAT_BackEnd::resumeStoppedDescendantProcesses(const char *context,
    const set<unsigned int> &leaveStoppedRanks)
{
    StatError_t error = STAT_OK;

    for (map<int, unsigned int>::iterator child = stoppedDescendants_.begin(); child != stoppedDescendants_.end();)
    {
        // Detach transfers explicitly retained stops to the user; failed cleanup remains ours to retry.
        if (leaveStoppedRanks.count(child->second) != 0 || continuePyStackProcess(child->first, context))
            child = stoppedDescendants_.erase(child);
        else
        {
            error = STAT_RESUME_ERROR;
            ++child;
        }
    }

    return error;
}


// --- Application pause, resume, and detach ---

/** Freeze roots and descendants before collection, including Python processes beneath wrappers. */
StatError_t STAT_BackEnd::pausePythonStackProcessTrees()
{
    return pausePyStack(true);
}


/** Stop roots and, when requested, descendants before reporting the application paused.
 * On failure, roll back only stops introduced by this attempt. Keep any stops
 * whose cleanup fails available for a later Resume or Detach.
 */
StatError_t STAT_BackEnd::pausePyStack(bool withChildren)
{
    const bool wasRunning = isRunning_;
    set<int> newlyStoppedRoots;
    const map<int, unsigned int> previousDescendants = stoppedDescendants_;
    StatError_t error = STAT_OK;

    for (unsigned int i = 0; i < proctabSize_; i++)
    {
        bool changed;
        bool stopped = stopPyStackProcess(proctab_[i].pid, changed);
        if (changed)
            newlyStoppedRoots.insert(proctab_[i].pid);
        if (!stopped)
        {
            error = (errno == ESRCH) ? STAT_APPLICATION_EXITED : STAT_PAUSE_ERROR;
            printMsg(error, __FILE__, __LINE__, "Failed to stop Python stack target pid %d: %s\n", proctab_[i].pid, strerror(errno));
            break;
        }
    }

    if (error == STAT_OK && withChildren)
        error = stopDescendantProcessesForSampling();

    if (error == STAT_OK)
    {
        isRunning_ = false;
        return STAT_OK;
    }

    // Undo this attempt without releasing stops retained from an earlier sample or another owner.
    bool restored = true;
    for (map<int, unsigned int>::iterator child = stoppedDescendants_.begin(); child != stoppedDescendants_.end();)
    {
        if (previousDescendants.count(child->first) != 0)
            ++child;
        else if (continuePyStackProcess(child->first, "after failed pause"))
            child = stoppedDescendants_.erase(child);
        else
        {
            restored = false;
            ++child;
        }
    }

    for (set<int>::iterator pid = newlyStoppedRoots.begin(); pid != newlyStoppedRoots.end(); ++pid)
    {
        if (!continuePyStackProcess(*pid, "after failed pause"))
            restored = false;
    }

    // On incomplete rollback, Resume/Detach can retry every root plus the retained descendants.
    isRunning_ = wasRunning && restored;
    return error;
}


/** Resume tracked descendants before roots and report any continuation failure.
 * An explicit Resume also continues roots that were already stopped before STAT's pause.
 */
StatError_t STAT_BackEnd::resumePyStack()
{
    StatError_t error = resumeStoppedDescendantProcesses("after resume");

    for (unsigned int i = 0; i < proctabSize_; i++)
    {
        // Resume is an explicit request to continue all application roots, including pre-existing stops.
        if (!continuePyStackProcess(proctab_[i].pid, "after resume"))
            error = STAT_RESUME_ERROR;
    }

    isRunning_ = (error == STAT_OK);
    return error;
}


/** Release application processes except ranks explicitly requested to remain stopped.
 * Apply the same retained-rank policy to descendants and roots; preserve failed
 * descendant cleanup entries so continuation can be retried.
 */
StatError_t STAT_BackEnd::detachPyStack(unsigned int *stopArray, int stopArrayLen)
{
    set<unsigned int> leaveStoppedRanks;
    for (int i = 0; i < stopArrayLen; i++)
        leaveStoppedRanks.insert(stopArray[i]);

    StatError_t error = resumeStoppedDescendantProcesses("after detach", leaveStoppedRanks);

    for (unsigned int i = 0; i < proctabSize_; i++)
    {
        if (leaveStoppedRanks.count(proctab_[i].mpirank) == 0 &&
            !continuePyStackProcess(proctab_[i].pid, "after detach"))
            error = STAT_RESUME_ERROR;
    }

    isRunning_ = (error == STAT_OK && leaveStoppedRanks.empty());
    return error == STAT_OK ? STAT_OK : STAT_DETACH_ERROR;
}


// --- Rank identity and frame metadata ---

/** Decode function@source:line and return the frame identity selected by sampleType.
 * Detailed modes include source and line in identity. Function-only mode retains
 * real source filenames as display metadata without splitting equivalent functions.
 */
static string getPythonFrameName(map<string, string> &nodeAttrs, const string &frame, unsigned int sampleType)
{
    string function = frame;
    string source;
    string line;
    string::size_type atPos = frame.find("@");
    string::size_type linePos = frame.find_last_of(":");

    if (atPos != string::npos)
    {
        function = frame.substr(0, atPos);
        if (linePos != string::npos && linePos > atPos)
        {
            source = frame.substr(atPos + 1, linePos - atPos - 1);
            line = frame.substr(linePos + 1);
        }
    }

    nodeAttrs["function"] = function;
    if (!source.empty() && !line.empty() &&
        ((sampleType & STAT_SAMPLE_LINE) || (sampleType & STAT_SAMPLE_PC) ||
         (sampleType & STAT_SAMPLE_MODULE_OFFSET)))
    {
        nodeAttrs["source"] = source;
        nodeAttrs["line"] = line;
        return function + "@" + source + ":" + line;
    }

    if (!source.empty() && source != "stat-process-tree")
    {
        // Retain filename metadata for display without changing function-only merging.
        nodeAttrs["source"] = source;
    }

    return function;
}


// --- Python adapter calls ---

/** Request one batch in proctab order; return a new reference, or NULL for serial fallback.
 * This helper owns the temporary argument objects and validates the result count.
 * Pass resume_stopped=false so Python collects stacks without resuming targets.
 * STAT's C++ backend controls when application execution resumes.
 */
PyObject *STAT_BackEnd::collectPyStackBatch(const char *nativeMode)
{
    const bool sampleChildren = (sampleType_ & STAT_SAMPLE_CHILDREN) != 0;
    PyObject *batchFunc, *batchEntries, *batchArgs;
    PyObject *batchResults = NULL;

    batchFunc = PyObject_GetAttrString(pyStackModule_, "get_trace_batch");
    if (batchFunc != NULL && PyCallable_Check(batchFunc))
    {
        bool entriesValid = true;
        batchEntries = PyList_New(proctabSize_);
        if (batchEntries == NULL)
        {
            PyErr_Clear();
            entriesValid = false;
        }

        for (unsigned int j = 0; entriesValid && j < proctabSize_; j++)
        {
            PyObject *entry = Py_BuildValue(
                "(iis)",
                proctab_[j].pid,
                proctab_[j].mpirank,
                proctab_[j].host_name == NULL ? "" : proctab_[j].host_name);
            if (entry == NULL)
            {
                PyErr_Clear();
                entriesValid = false;
                break;
            }
            // The list takes ownership of entry; results retain this same proctab ordering.
            PyList_SET_ITEM(batchEntries, j, entry);
        }

        if (entriesValid)
        {
            // Carry native mode per request so attached sessions can change it.
            batchArgs = Py_BuildValue(
                "(Oiiis)",
                batchEntries,
                sampleChildren ? 1 : 0,
                (sampleType_ & STAT_SAMPLE_THREADS) != 0,
                0,
                nativeMode);
            if (batchArgs != NULL)
            {
                batchResults = PyObject_CallObject(batchFunc, batchArgs);
                Py_DECREF(batchArgs);
            }
            else
                PyErr_Clear();
        }

        Py_XDECREF(batchEntries);

        // Per-target lookup below is safe only when the batch matches the complete local proctab.
        if (batchResults != NULL &&
            (!PyList_Check(batchResults) ||
             PyList_Size(batchResults) != proctabSize_))
        {
            printMsg(STAT_WARNING, __FILE__, __LINE__,
                     "PyStack batch returned malformed results; falling back to serial sampling\n");
            Py_DECREF(batchResults);
            batchResults = NULL;
        }
        else if (batchResults == NULL && PyErr_Occurred())
        {
            PyErr_Clear();
            printMsg(STAT_WARNING, __FILE__, __LINE__,
                     "PyStack batch call failed; falling back to serial sampling\n");
        }
    }
    else
    {
        if (PyErr_Occurred())
            PyErr_Clear();
    }
    Py_XDECREF(batchFunc);

    return batchResults;
}


/** Return one target's trace as a new reference, retrying unusable batch entries serially.
 * NULL skips this target after diagnostics. nonPythonTarget distinguishes a confirmed
 * non-Python target from an ordinary collection failure for the final backend guidance.
 */
PyObject *STAT_BackEnd::collectPyStackTarget(unsigned int targetIndex, PyObject *batchResults,
    PyObject *sampleFunc, const char *nativeMode, bool &nonPythonTarget)
{
    const bool sampleChildren = (sampleType_ & STAT_SAMPLE_CHILDREN) != 0;
    const char *sampleFunctionName = sampleChildren ? "get_trace_tree" : "get_trace";
    const char *nonPythonTargetPrefix = "PyStack did not recognize";
    PyObject *pArgs = NULL, *pValue = NULL;

    nonPythonTarget = false;

    if (batchResults != NULL)
    {
        PyObject *batchItem = PyList_GetItem(batchResults, targetIndex);
        if (batchItem != NULL && PyTuple_Check(batchItem) &&
            PyTuple_Size(batchItem) == 2)
        {
            PyObject *traceValue = PyTuple_GetItem(batchItem, 0);
            PyObject *errorValue = PyTuple_GetItem(batchItem, 1);
            const char *batchError = NULL;
            if (errorValue != NULL && PyUnicode_Check(errorValue))
                batchError = PyUnicode_AsUTF8(errorValue);
            if (batchError != NULL && batchError[0] != '\0')
            {
                if (strstr(batchError, nonPythonTargetPrefix) != NULL)
                {
                    // A native target cannot produce a Python stack on serial retry.
                    nonPythonTarget = true;
                    return NULL;
                }
                printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__,
                         "PyStack batch rank %d pid %d failed (%s); retrying serially\n",
                         proctab_[targetIndex].mpirank, proctab_[targetIndex].pid, batchError);
            }
            else if (traceValue != NULL && PyUnicode_Check(traceValue))
            {
                const char *batchTrace = PyUnicode_AsUTF8(traceValue);
                if (batchTrace != NULL && batchTrace[0] != '\0')
                {
                    // The tuple owns traceValue; give the caller its own reference to release.
                    pValue = traceValue;
                    Py_INCREF(pValue);
                }
            }
            if (PyErr_Occurred())
                PyErr_Clear();
        }
    }

    // Retry an unavailable or failed batch entry using the corresponding single-target API.
    if (pValue == NULL)
    {
        if (sampleChildren == true)
        {
            pArgs = Py_BuildValue("(iisiis)", proctab_[targetIndex].pid, proctab_[targetIndex].mpirank, proctab_[targetIndex].host_name == NULL ? "" : proctab_[targetIndex].host_name, (sampleType_ & STAT_SAMPLE_THREADS) != 0, 0, nativeMode);
        }
        else
        {
            pArgs = Py_BuildValue("(iiis)", proctab_[targetIndex].pid, 0, (sampleType_ & STAT_SAMPLE_THREADS) != 0, nativeMode);
        }
        if (!pArgs)
        {
            PyErr_Clear();
            printMsg(STAT_WARNING, __FILE__, __LINE__, "Failed to generate pArgs for pid %d\n", proctab_[targetIndex].pid);
            return NULL;
        }
        // Both serial APIs receive resume_stopped=0: collection must leave process control to STAT.
        pValue = PyObject_CallObject(sampleFunc, pArgs);
        Py_DECREF(pArgs);
    }

    if (pValue == NULL)
    {
        PyObject *errorType = NULL, *errorValue = NULL, *errorTraceback = NULL;
        PyObject *errorString = NULL;
        const char *errorText = "unknown Python exception";
        // Capture the exception before further Python calls; keep its text alive through logging.
        PyErr_Fetch(&errorType, &errorValue, &errorTraceback);
        PyErr_NormalizeException(&errorType, &errorValue, &errorTraceback);
        if (errorValue != NULL)
        {
            errorString = PyObject_Str(errorValue);
            if (errorString != NULL)
            {
                const char *convertedError = PyUnicode_AsUTF8(errorString);
                if (convertedError != NULL)
                    errorText = convertedError;
            }
        }
        if (errorString == NULL || PyErr_Occurred())
            PyErr_Clear();
        if (strstr(errorText, nonPythonTargetPrefix) != NULL)
            nonPythonTarget = true;
        else
            printMsg(STAT_WARNING, __FILE__, __LINE__, "%s call failed for rank %d pid %d: %s\n",
                     sampleFunctionName, proctab_[targetIndex].mpirank, proctab_[targetIndex].pid, errorText);
        Py_XDECREF(errorString);
        Py_XDECREF(errorType);
        Py_XDECREF(errorValue);
        Py_XDECREF(errorTraceback);
        return NULL;
    }

    return pValue;
}


// --- Trace decoding and graph construction ---

/** Add one target's adapter output to the current 2D graph.
 * Process markers supply distinct-PID counts, rank markers select global worker bits,
 * and #endtrace starts a new thread path. threadIndex is shared across this sample.
 */
StatError_t STAT_BackEnd::addPyStackTrace(unsigned int targetIndex, const char *trace, int &threadIndex)
{
    const bool sampleChildren = (sampleType_ & STAT_SAMPLE_CHILDREN) != 0;
    StatBitVectorEdge_t *edge = initializeBitVectorEdge(proctabSize_);
    if (edge == NULL)
    {
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to initialize edge\n");
        return STAT_ALLOCATE_ERROR;
    }
    edge->bitVector[targetIndex / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(targetIndex % STAT_BITVECTOR_BITS);

    // Python owns the original UTF-8 text; strtok requires a writable copy.
    char *traceBuffer = strdup(trace);
    if (traceBuffer == NULL)
    {
        statFreeEdge(edge);
        printMsg(STAT_ALLOCATE_ERROR, __FILE__, __LINE__, "Failed to copy Python stack trace buffer\n");
        return STAT_ALLOCATE_ERROR;
    }

    static int threadCountWarning = 0;
    const auto recordThread = [&]() {
        if (find(threadIds_.begin(), threadIds_.end(), threadIndex) == threadIds_.end())
        {
            threadIds_.push_back(threadIndex);
            if (threadCountWarning == 0 && threadIds_.size() >= threadBvLength_)
            {
                printMsg(STAT_WARNING, __FILE__, __LINE__, "Number of threads exceeded %d limit\n", threadBvLength_);
                threadCountWarning++;
            }
        }
    };
    recordThread();

    int prevId = 0;
    string path;
    char *currentFrame;

    // Decode process/rank boundaries before adding each frame to the current call path.
    bool processRankTraces = false;
    pid_t currentProcessId = -1;

    currentFrame = strtok(traceBuffer, "\n");
    while (currentFrame != NULL)
    {
        int processRank, worldSize;
        int parsedProcessId;
        // A PID changes process-count attribution, but does not itself assign a new application rank.
        if (sscanf(currentFrame, "#process#%d", &parsedProcessId) == 1)
        {
            currentProcessId = parsedProcessId;
            path = "";
            prevId = 0;
        }
        else if (sscanf(currentFrame, "#processrank#%d#%d", &processRank, &worldSize) == 2)
        {
            processRankTraces = true;
            path = "";
            prevId = 0;
            if (processRank >= 0 && worldSize > processRank &&
                (processRankCount_ == 0 || processRankCount_ == worldSize))
            {
                processRankCount_ = worldSize;
                sampleType_ |= STAT_SAMPLE_PROCESS_RANKS;
                // Worker ranks are global, so replace the initial local-proctab bit vector.
                statFreeEdge(edge);
                edge = initializeBitVectorEdge(processRankCount_);
                if (edge == NULL)
                {
                    free(traceBuffer);
                    return STAT_ALLOCATE_ERROR;
                }
                edge->bitVector[processRank / STAT_BITVECTOR_BITS] |= STAT_GRAPH_BIT(processRank % STAT_BITVECTOR_BITS);
            }
        }
        else if (strcmp(currentFrame, "#endtrace") == 0)
        {
            path = "";
            threadIndex++;
            prevId = 0;
            recordThread();
            // Keep one stack only for a single-process, non-threaded trace; trees still need later processes.
            if (!(sampleType_ & STAT_SAMPLE_THREADS) && !sampleChildren && !processRankTraces)
                break;
        }
        else
        {
            map<string, string> nodeAttrs;
            string name = getPythonFrameName(nodeAttrs, currentFrame, sampleType_);
            // Hash the selected call path before adding display attributes, which must not split nodes.
            path += name;
            int nodeId = statStringHash(path.c_str());
            nodes2d_[nodeId] = name;
            for (map<string, string>::iterator nodeAttrsIter = nodeAttrs.begin(); nodeAttrsIter != nodeAttrs.end(); nodeAttrsIter++)
                nodeIdToAttrs_[nodeId][nodeAttrsIter->first] = nodeAttrsIter->second;
            update2dEdge(prevId, nodeId, edge, threadIndex, currentProcessId);
            prevId = nodeId;
        }
        currentFrame = strtok(NULL, "\n");
    }

    free(traceBuffer);
    statFreeEdge(edge);

    return STAT_OK;
}


// --- Sampling lifecycle ---

/** Collect samples while keeping process control separate from Python calls and graph decoding.
 * Each iteration freezes targets, collects in proctab order, merges into 3D history,
 * then resumes between samples or when wasRunning requests a final resume.
 * traceFrequency is in milliseconds; native mode is selected independently for each request.
 */
StatError_t STAT_BackEnd::samplePyStackTraces(unsigned int nTraces, unsigned int traceFrequency,
                                            bool wasRunning, const char *pystackNativeMode)
{
    const bool sampleChildren = (sampleType_ & STAT_SAMPLE_CHILDREN) != 0;
    const char *sampleFunctionName = sampleChildren ? "get_trace_tree" : "get_trace";
    const char *nativeMode = pystackNativeMode == NULL ? "Python Only" : pystackNativeMode;

    // Resolve the single-target API once; every sample can use it when batch collection falls back.
    PyObject *sampleFunc = PyObject_GetAttrString(pyStackModule_, sampleFunctionName);
    if (sampleFunc == NULL || !PyCallable_Check(sampleFunc))
    {
        if (PyErr_Occurred())
            PyErr_Print();
        Py_XDECREF(sampleFunc);
        printMsg(STAT_DAEMON_ERROR, __FILE__, __LINE__, "Failed to load function %s from PyStack module\n", sampleFunctionName);
        return STAT_DAEMON_ERROR;
    }

    for (unsigned int sample = 0; sample < nTraces; sample++)
    {
        // Start a fresh 2D sample without discarding 3D history, then freeze roots and descendants.
        clear2dNodesAndEdges();
        StatError_t statError = pausePythonStackProcessTrees();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Failed to pause Python stack process trees\n");
            Py_DECREF(sampleFunc);
            return statError;
        }

        // Collect all local targets together; a missing batch result still permits serial collection.
        PyObject *batchResults = collectPyStackBatch(nativeMode);
        unsigned int collectedTargetCount = 0, nonPythonTargetCount = 0;
        int threadIndex = 1;

        // Consume results in proctab order. The helper retries failed entries and diagnoses skipped targets.
        for (unsigned int target = 0; target < proctabSize_; target++)
        {
            bool nonPythonTarget;
            PyObject *traceValue = collectPyStackTarget(target, batchResults, sampleFunc, nativeMode, nonPythonTarget);
            if (nonPythonTarget)
                nonPythonTargetCount++;
            if (traceValue == NULL)
                continue;

            // Keep the Python result alive while its UTF-8 text is converted into graph nodes and edges.
            const char *trace = PyUnicode_AsUTF8(traceValue);
            if (trace == NULL)
            {
                printMsg(STAT_WARNING, __FILE__, __LINE__, "Failed to convert %s result for pid %d\n", sampleFunctionName, proctab_[target].pid);
                PyErr_Print();
                Py_DECREF(traceValue);
                continue;
            }

            // Report an unexpected empty result instead of accepting it as a successful sample.
            if (trace[0] == '\0')
            {
                printMsg(STAT_WARNING, __FILE__, __LINE__, "Empty PyStack result for rank %d pid %d\n",
                         proctab_[target].mpirank, proctab_[target].pid);
                Py_DECREF(traceValue);
                continue;
            }

            printMsg(STAT_LOG_MESSAGE, __FILE__, __LINE__, "Result of %s call: %s\n", sampleFunctionName, trace);
            statError = addPyStackTrace(target, trace, threadIndex);
            Py_DECREF(traceValue);
            if (statError != STAT_OK)
                break;
            collectedTargetCount++;
        }

        // Every target reference has been released; the batch can now be discarded, even on decode failure.
        Py_XDECREF(batchResults);

        // A decode failure aborts this sample before history merge; rank conflicts discard its partial graph.
        if (statError != STAT_OK)
        {
            Py_DECREF(sampleFunc);
            if (statError == STAT_SAMPLE_ERROR && wasRunning)
                resumePyStack();
            return statError;
        }

        // Collection errors and empty results must not produce a successful empty sample.
        const bool noStacks = collectedTargetCount == 0;
        if (noStacks)
            printMsg(STAT_WARNING, __FILE__, __LINE__, "PyStack collected no usable stacks from %d local targets\n", proctabSize_);

        if (nonPythonTargetCount != 0 && !noStacks)
            printMsg(STAT_WARNING, __FILE__, __LINE__,
                     "PyStack did not recognize %d sampled target%s as Python; use --backend dyninst for native binaries\n",
                     nonPythonTargetCount, nonPythonTargetCount == 1 ? "" : "s");

        // Accumulate the completed sample before allowing application execution between observations.
        statError = noStacks ? STAT_OK : update3dNodesAndEdges();
        if (statError != STAT_OK)
        {
            printMsg(statError, __FILE__, __LINE__, "Error updating 3d nodes and edges for trace %d of %d\n", sample + 1, nTraces);
            Py_DECREF(sampleFunc);
            return statError;
        }

        // Run between samples; after the final sample, keep paused targets paused unless wasRunning is set.
        const bool isLastTrace = (sample == nTraces - 1);
        if (!isLastTrace || wasRunning)
        {
            statError = resume();
            if (statError != STAT_OK)
            {
                printMsg(statError, __FILE__, __LINE__, "Failed to resume processes\n");
                Py_DECREF(sampleFunc);
                return statError;
            }
        }
        if (!isLastTrace)
            usleep(traceFrequency * 1000);

        // Report collection failure after the normal resume decision, without merging an empty sample.
        if (noStacks)
        {
            Py_DECREF(sampleFunc);
            return STAT_NO_SAMPLES_ERROR;
        }
    }

    Py_DECREF(sampleFunc);
    return STAT_OK;
}
