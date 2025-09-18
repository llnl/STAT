"""@package core_file_merger
An implementation for merging core files into a .dot
format suitable for the Stack Trace Analysis Tool.

Adapted from a script initially written by Dane Gardner
"""

"""
TODO: There are still issues with gdb subprocesses remaining after the
      script has exited; seems to be only when the user sends an
      interrupt
"""

__copyright__ = """Copyright (c) 2007-2022, Lawrence Livermore National Security, LLC."""
__license__ = """Produced at the Lawrence Livermore National Laboratory
Written by Dane Gardner, Gregory Lee <lee218@llnl.gov>, Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Dane Gardner", "Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 2
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

###############################################################################
import signal, os, sys

try:
    from stat_merge_base import StatTrace, StatMerger, StatMergerArgs
except Exception as e:
    sys.stderr.write("The following required library is missing: stat_merge_base\n%s\n" % (repr(e)))
    sys.exit(1)

import subprocess, re, threading, glob, logging
from datetime import datetime

high_rank = 9999999

class Gdb(object):
    def __init__ (self, options):
        """Constructor"""

        self.directory =  options["directory"]
        self.coredir =    (options["coredir"])    and options["coredir"]    or self.directory
        self.exedir =     (options["exedir"])     and options["exedir"]     or self.directory
        self.sourcepath = (options["sourcepath"]) and options["sourcepath"] or ''
        self.objectpath = (options["objectpath"]) and options["objectpath"] or ''
        self.executable = options["exe"]
        self.corefile = None
        self.subprocess = None

    def open(self, corefile, executable = None):
        """Opens the gdb subprocess using the given arguments
           \param corefile the corefile to open
           \param executable the executable file to use, defaults to None and
                  will be extracted from the core file"""

        args = []
        args.append('gdb')
        args.append('-ex')
        args.append("'set pagination 0'")
        args.append('-ex')
        args.append("'cd %s'" %(self.directory))
        args.append('-ex')
        args.append("'path %s'" %(self.objectpath))
        args.append('-ex')
        args.append("'directory %s'" %(self.sourcepath))
        args.append('-ex')
        args.append("'set filename-display absolute'")
        if corefile:
            self.corefile = corefile
            if os.path.isabs(self.corefile):
                args.append("--core=%s" %(self.corefile))
            else:
                args.append("--core=%s/%s" %(self.coredir, self.corefile))
        if executable and executable != "NULL":
            self.executable = executable
            if os.path.isabs(self.executable):
                args.append("%s" %(self.executable))
            else:
                args.append("%s/%s" %(self.exedir, self.executable))
        self.subprocess = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return self.readlines()

    def close(self):
        """Closes the gdb subprocess"""
        try:
            try:
                self.subprocess.communicate("quit\n")
                self.subprocess = None
            except:
                pass
        finally:
            self.kill()   #Ensure closure

    def kill(self):
        try:
            if self.subprocess:
                logging.warn("Killing gdb subprocess")
                os.kill(self.subprocess.pid, signal.SIGKILL)
                self.subprocess = None
        except OSError:
            pass


    def __del__(self):
        """Destructor"""
        self.close()


    def readlines(self):
        """Simply reads the lines, until we get to the '(gdb)' prompt
           prevents blocking from a lack of EOF with a stdout.readline() loop"""
        line = ''
        lines = []
        while True:
            ch = self.subprocess.stdout.read(1).decode('utf-8')
            if ch == '\n':
                lines.append(line)
                ch = ''
                line = ''
            elif ch == ' ':    #Check for the prompt, and return if we get it
                if '(gdb)' in line or '(cuda-gdb)' in line:
                    break
            line += ch
        return lines


    def communicate(self, command):
        """Sends the command to gdb, and returns a list of outputted lines
           \param command the command to send to gdb
           \returns a list of lines from a merged stdout/stderr stream"""
        if not command.endswith('\n'):
            command += '\n'
        self.subprocess.stdin.write(command.encode('utf-8'))
        self.subprocess.stdin.flush()
        return self.readlines()


class CudaGdb(Gdb):
    def __init__ (self, options):
        Gdb.__init__(self, options)

    def open(self, corefile, executable = None):
        args = []
        args.append('cuda-gdb')
        args.append('-ex')
        target_string = "target cudacore "

        if corefile:
            self.corefile = corefile
            if os.path.isabs(self.corefile):
                target_string += " %s" % (self.corefile)
            else:
                target_string += " %s/%s" % (self.coredir, self.corefile)
        args.append(target_string)

        self.subprocess = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return self.readlines()

class GdbOneapi(Gdb):

    def __init__ (self, options):
        Gdb.__init__(self, options)
        self.executable = "GPU"

    def open(self, corefile, executable = None):
        args = []
        args.append('gdb-oneapi')
        args.append('-ex')
        args.append("set pagination 0")
        args.append('-ex')
        args.append("cd %s" %(self.directory))
        args.append('-ex')
        args.append("path %s" %(self.objectpath))
        args.append('-ex')
        args.append("directory %s" %(self.sourcepath))
        args.append('-ex')
        args.append("set filename-display absolute")
        args.append('-ex')
        target_string = "target core "
        self.corefile = corefile
        if os.path.isabs(self.corefile):
            target_string += " %s" % (self.corefile)
        else:
            target_string += " %s/%s" % (self.coredir, self.corefile)
        args.append(target_string)
        self.subprocess = subprocess.Popen(args, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return self.readlines()


###############################################################################
class CoreFile:
    """RegEx for finding information about a frame from a back trace.
         group(1) = value of the frame number
         group(2) = name of the function
         group(3) = source file and line number in format "file:line"  --this value may be None"""
    __reFrame = re.compile(r"#(\d+)\s+(?:0x\S+\s+in\s+)?(\S+)\s+\(.*\)(?:\s+at\s+(\S+:\d+))?")
    __options = None
    _next_rank = 0
    _files = []

    def __init__ (self, coreFile, options):
        if CoreFile.__options is None and not options is None:
            CoreFile.__options = options
        self.coreData = {'coreFile': coreFile,
                         'rank': CoreFile._next_rank,
                         'rankSize': None,
                         'traces': []}
        if coreFile not in CoreFile._files:
            CoreFile._next_rank += 1
            CoreFile._files.append(coreFile)

    def add_functions(self, functions):
        """If the functions array isn't empty, reverse it to get a proper trace, and
           append it to merge
           \param functions a list of strings, each containing a function name"""

        if len(functions):    #If we've got a trace from the last one, add it to the graph
            logging.debug("Appending trace")
            trace = []
            while len(functions):    #Reverse the trace
                trace.append(functions.pop())
            self.coreData['traces'].append(trace)

    def get_function_value(self, gdb, function_name, var_position):
        """Run through the source and the stack frames to find a value for the
           requested variable, if possible (i.e. function was called).
           \param gdb the open gdb subprocess
           \paran function_name string with the desired function call name
           \param var_position int with desired variable position
           \warning Ensure that symbols are loaded before calling this function"""

        re_thread = re.compile("\* (\d) Thread")
        #re_thread = re.compile("\*?\s+(\d+)\s+process\s+\d+\s+") #old format
        lines = gdb.communicate("info threads")
        threads = []
        for line in lines:
            rexp = re_thread.match(line)
            if rexp:
                threads.append(rexp.group(1))
        if threads:
            logging.debug("Found threads")
            logging.log(logging.INSANE, "Threads found: %s" %(threads))
        else:
            logging.warn("No threads found")

        #Iterate over the threads
        while threads:
            thread = threads.pop()
            logging.debug("Now searching thread %s" %(thread))

            #Enter the thread
            lines = gdb.communicate("thread %s" %(thread))
            in_thread = False
            for line in lines:
                if re.match("\[Switching to thread\s+%s" %thread, line):
                    in_thread = True
                    break
            if in_thread:
                logging.debug("Successfully entered thread %s" %(thread))
            else:
                logging.warn("Cannot enter thread")
                continue


            #Get frames for this thread
            lines = gdb.communicate("info stack")
            frames = []
            for line in lines:
                rexp = CoreFile.__reFrame.match(line)
                if rexp:
                    frames.append(rexp.group(1))
            if frames:
                logging.debug("Found frames")
                logging.log(logging.INSANE, "Frames found: %s" %(frames))
            else:
                logging.warn("No frames found")

            #Step through the frames
            while frames:
                frame = frames.pop()
                logging.debug("Now searching frame %s" %(frame))

                #Enter the frame
                lines = gdb.communicate("frame %s"%(frame))
                inFrame = False
                for line in lines:
                    rexp = CoreFile.__reFrame.match(line)
                    if rexp and rexp.group(1) == frame:
                        inFrame = True
                        break
                if inFrame:
                    logging.debug("Successfully entered frame %s" %(frame))
                else:
                    logging.warn("Cannot enter frame")
                    continue

                #List the first line of source so we can do a forward-search in gdb
                lines = gdb.communicate("info line 1")
                has_source = False
                for line in lines:
                    if 'Line 1 of "' in line:
                        has_source = True
                        break
                if has_source:
                    logging.debug("Frame has source")
                else:
                    logging.warn("Frame has no source")
                    continue

                #Search the source for a call to function_name
                """NOTE: This doesn't search the first line, however, it is safe to assume that
                         it wouldn't be on the first line"""
                keep_searching = True
                var_names = []
                re_vars = re.compile(r"(\d+)\s*%s\s*\((.*)\)\s*;"
                %(function_name))
                while keep_searching:
                    lines = gdb.communicate("forward-search \s*%s\s*(.*)\s*;" %(function_name))
                    for line in lines:
                        if 'Expression not found' in line:
                            keep_searching = False
                            break
                        else:
                            rexp = re_vars.search(line)
                            if rexp:
                                vars = rexp.group(2).replace('&','').replace('*','').replace(' ','').replace('\t','').split(',')  #HACK:  Ugly, ugly, ugly
                                if var_position < len(vars):
                                    var_names.append( vars[var_position] )
                                    break
                if len(var_names) > 0:
                    logging.debug("var_names = %s" %(var_names))
                else:
                    logging.warn("No variable names were recovered")
                    continue

                #Get the values of the local variables, so we can return the variable value
                lines = gdb.communicate("info locals")
                var_value = ""
                for line in lines:
                    for var_name in var_names:
                        if var_name in line:
                            rexp = re.search("\s*%s\s+=\s+(\S+.*)"%(var_name), line)
                            if rexp:
                                var_value = rexp.group(1)
                                break
                    if var_value:
                        break

                if var_value:
                    logging.debug("var_value = %s"%(var_value))
                    return var_value
                else:
                    logging.warn("Getting a value for var_value failed")
                    continue
        return None

    def process_core(self):
        """Processes the core file"""

        #Set up the options variables
        withline =   CoreFile.__options["withline"]
        force =      CoreFile.__options["force"]
        directory =  CoreFile.__options["directory"]
        coredir =    (CoreFile.__options["coredir"])    and CoreFile.__options["coredir"]    or directory
        exedir =     (CoreFile.__options["exedir"])     and CoreFile.__options["exedir"]     or directory
        sourcepath = (CoreFile.__options["sourcepath"]) and CoreFile.__options["sourcepath"] or ''
        objectpath = (CoreFile.__options["objectpath"]) and CoreFile.__options["objectpath"] or ''

        #Open gdb against the core file
        logging.info("Connecting gdb to the core file (%s)"%self.coreData['coreFile'])
        if CoreFile.__options['cuda'] == 1:
            gdb = CudaGdb(CoreFile.__options)
        elif CoreFile.__options['oneapi'] == 1:
            gdb = GdbOneapi(CoreFile.__options)
        else:
            gdb = Gdb(CoreFile.__options)
        executable = gdb.executable

#        #Find the executable that generated the core file to begin with
        if executable == "NULL":
            lines = gdb.open(self.coreData['coreFile'])
            for line in lines:
                if 'Core was generated by' in line:
                    rexp = re.search(r"Core was generated by `(.+)'\.", line)
                    if rexp: executable = rexp.group(1).split()[0]
            if not executable:
                logging.critical("Error: Cannot discover executable that generated core file %s" %self.coreData['coreFile'])
                return False

            #Exit gdb
            logging.info("Disconnecting gdb from the core file (%s)"%self.coreData['coreFile'])
            gdb.close()

            #Reconnect to gdb using executable
            logging.info("Reconnecting gdb to the core file (%s) AND the executable (%s)"%(self.coreData['coreFile'],executable))

        lines = gdb.open(self.coreData['coreFile'], executable)
        # check for architecture type
        coretype = 'unknown'
        for line in lines:
            if line.find('ppc64le') != -1:
                coretype = 'ppc64le'
                break
            elif line.find('x86_64') != -1:
                coretype = 'x86_64'
                break

        #Check for gdb errors
        logging.debug("Checking for gdb errors")
        symbols_loaded = True
        for line in lines:
            if '(no debugging symbols found)' in line:
                pass # some .so's may have this message, which may be OK
                #symbols_loaded = False
                #if withline:
                #  logging.critical("The --withlines argument was specified, however there is no symbol table compiled into the executable.\n" +
                #                 "Try running without this argument.")
                #  if not force:
                #      sys.exit(2)
            elif 'warning: exec file is newer than core file.' in line:
                logging.critical("GDB: The executable (%s/%s) is newer than the core file (%s/%s).\n" %(exedir, executable, coredir, self.coreData['coreFile']) +
                               "      Try using touch to change the timestamps.")
                if not force:
                    sys.exit(2)
            elif 'warning: core file may not match specified executable file.' in line:
                logging.critical("GDB: The executable (%s/%s) may not match the core file (%s/%s).\nThis may result in incorrect traces or hangs in this utility" %(exedir, executable, coredir, self.coreData['coreFile']))
                if not force:
                    logging.critical("Use -r to run anyway\n")
                    sys.exit(2)
            elif "%s: No such file or directory."%(executable) in line:
                logging.critical("GDB: The executable (%s/%s) doesn't exist." %(exedir,executable))
                if not force:
                    sys.exit(2)

        # at some point we needed to gobble up an extra command prompt:
        # as of 01/30/19 this appears to be necessary on PPC64 LE w/ vanilla GDB
        # as of 01/30/19 this appears to be unnecessary on PPC64 LE w/ vanilla GDB
#        if CoreFile.__options['cuda'] != 1 and coretype == 'ppc64le':
#            lines2 = gdb.readlines()
#            lines += lines2

        if symbols_loaded:
            #Find the value for the returned size from MPI_Comm_size()
            #Currently removed for speed, we aren't using it right now
            """
            if self.coreData['rankSize'] is None:
              logging.debug("Find a value for the highest rank")
              rankSizeValue = self.get_function_value(gdb, 'MPI_Comm_size', 1)
              if rankSizeValue and rankSizeValue.isdigit() and int(rankSizeValue) >= 0:
                self.coreData['rankSize'] = int(rankSizeValue)
                logging.debug("Value found from stack, using: %i" %(self.coreData['rankSize']))
              else:
                logging.info("Could not find a highest rank value")
            """

            #Find the value for the returned rank from MPI_Comm_rank()
            logging.debug("Find a value for the current rank")
            rank_value = self.get_function_value(gdb, 'MPI_Comm_rank', 1)
            if rank_value and rank_value.isdigit() and int(rank_value) >= 0:
                self.coreData['rank'] = int(rank_value)
                logging.debug("Value found from stack, using: %i" %(self.coreData['rank']))
            else:
                logging.info("Could not find a rank value, using default: %i" %(self.coreData['rank']))

        ### PARSE THE STACK TRACE ###
        logging.debug("Parsing the stack trace")

        #Get stack trace from all threads
        lines = gdb.communicate("thread apply all bt")

        #Ensure we have a stack trace to parse
        if not lines:
            logging.critical("No backtrace returned")
            if not force:
                sys.exit(2)


        #Iterate over each line and find what we need
        in_thread = False
        functions = []
        for line in lines:
            #Start a new Thread listing
            if 'Thread' in line:
                logging.debug("functions %s" %functions)
                self.add_functions(functions)
                logging.debug("Found new thread")
                in_thread,functions = True,[]

            #In some cases, gdb will quit the stack trace early
            elif 'Backtrace stopped: ' in line and CoreFile.__options['oneapi'] != 1:
                logging.critical("GDB: Backtrace stopped")
                if not force:
                    sys.exit(2)

            #If we're in a Thread listing, see if we match a stack frame
            elif in_thread:
                rexp = CoreFile.__reFrame.match(line)
                if rexp:
                    if withline and rexp.group(3):
                        function = "%s@%s" %(rexp.group(2),rexp.group(3))
                    else:
                        function = rexp.group(2)
                    logging.log(logging.INSANE, "Found function %s from %s"%(function, line))  #This is a very verbose debug!
                    function = function.replace('<', '\<').replace('>', '\>')
                    functions.append( function )

        #Merge this process with the rest
        self.add_functions(functions)

        #Exit gdb
        logging.info("Disconnecting gdb from the core file (%s) AND the executable (%s)"%(self.coreData['coreFile'],executable))
        gdb.close()

        return True


###############################################################################
class CoreMergerArgs(StatMergerArgs):
    def __init__(self):
        StatMergerArgs.__init__(self)

        self.arg_map["output"] = StatMergerArgs.StatMergerArgElement("o", True, str, "coredump.dot", "the output filename")
        self.arg_map["pattern"] = StatMergerArgs.StatMergerArgElement("p", True, str, "^core.[0-9]+$", "the core file regex pattern")
        self.arg_map["directory"] = StatMergerArgs.StatMergerArgElement("d", True, str, os.getcwd(), "the output directory")
        self.arg_map["coredir"] = StatMergerArgs.StatMergerArgElement("D", True, str, os.getcwd(), "the core file directory")
        self.arg_map["exedir"] = StatMergerArgs.StatMergerArgElement("e", True, str, os.getcwd(), "the exe file directory")
        self.arg_map["sourcepath"] = StatMergerArgs.StatMergerArgElement("s", True, str, os.getcwd(), "the source file directory")
        self.arg_map["objectpath"] = StatMergerArgs.StatMergerArgElement("O", True, str, os.getcwd(), "the object file directory")
        self.arg_map["loglevel"] = StatMergerArgs.StatMergerArgElement("L", True, str, "error", "the verbosity level (critical|error|warning|info|verbose|debug|insane)")
        self.arg_map["logfile"] = StatMergerArgs.StatMergerArgElement("F", True, str, "stdout", "the log file name (defaults to stdout)")
        self.arg_map["withline"] = StatMergerArgs.StatMergerArgElement("i", False, int, 0, "whether to gather source line number")
        self.arg_map["force"] = StatMergerArgs.StatMergerArgElement("r", False, int, 0, "whether to force parsing on warnings and errors")
        self.arg_map["threads"] = StatMergerArgs.StatMergerArgElement("T", False, int, 1, "max number of threads")
        self.arg_map["cuda"] = StatMergerArgs.StatMergerArgElement("C", False, int, 0, "set if running on cuda cores")
        self.arg_map["oneapi"] = StatMergerArgs.StatMergerArgElement("n", False, int, 0, "set if using gdb-oneapi")

        self.arg_map["jobid"] = self.StatMergerArgElement("j", False, None, None, "[LW] delineate traces based on Job ID in the core file")
        self.arg_map["exe"] = StatMergerArgs.StatMergerArgElement("x", True, str, "NULL", "[LW] the executable path")
        self.arg_map["addr2line"] = StatMergerArgs.StatMergerArgElement("a", True, str, "NULL", "[LW] the path to addr2line")

        self.usage_msg_synopsis = '\nThis tool will merge the stack traces from the user-specified core files and output 2 .dot files, one with just function names, the other with function names + line number information\n\nNOTE: for lightweight core files, the -x option is required and the -x option is ignored for full core files.  Options marked with "[LW]" are for lightweight core files\n'
        self.usage_msg_command = '\nUSAGE:\n\t(LW only) python %s [options] -x <exe_path> -c <corefile>*\n\t(LW only) python %s [options] -x <exe_path> -c <core_files_dir>\n\tpython %s [options] -c <corefile>*\n' % (sys.argv[0], sys.argv[0], sys.argv[0])
        self.usage_msg_examples = '\nEXAMPLES:\n\t(LW only) python %s -x a.out -c core.0 core.1\n\t(LW only) python %s -x a.out -c core.*\n\t(LW only) python %s -x a.out -c ./\n\t(LW only) python %s -x a.out -c core_dir\n\tpython %s -c *.core\n' % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])

    def is_valid_file(self, file_path):
        if file_path.find('.core') == 0:
            return True
        return False

    def error_check(self, options):
        StatMergerArgs.error_check(self, options)
        if options["high"] != -1:
            options["high"] += 1
            high_rank = options["high"] # use this for all tasks we can't find the rank of
        new_loglevel = init_logging(options["loglevel"], options["logfile"])    #Initialize the logging module
        options["loglevel"] = new_loglevel


class CoreTrace(StatTrace):
    def get_traces(self):
        self.rank = 0
        self.tid = 0
        line_number_traces = []
        function_only_traces = []
        core_file = CoreFile(self.file_path, self.options)
        core_file.process_core()
        trace = core_file.coreData["traces"]
        self.rank = core_file.coreData["rank"]
        function_only_traces = trace

        self.options["withline"] = 1
        core_file2 = CoreFile(self.file_path, self.options)
        core_file2.process_core()
        trace2 = core_file2.coreData["traces"]
        self.options["withline"] = 0
        line_number_traces = trace2

        return [function_only_traces, line_number_traces]


class CoreMerger(StatMerger):
    def get_high_rank(self, trace_files):
        # determine the highest ranked task for graphlib initialization
        return high_rank


def init_logging(input_loglevel, input_logfile):
    """Initialize the logging module"""

    logging.VERBOSE = 25
    logging.addLevelName(logging.VERBOSE, 'VERBOSE')
    logging.INSANE = 5
    logging.addLevelName(logging.INSANE, 'INSANE')

    logLevels = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'verbose': logging.VERBOSE,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'insane': logging.INSANE,
                 'critical': logging.CRITICAL}
    logStreams = {'stdout': sys.stdout, 'stderr': sys.stderr}

    #NOTE: We're changing the options value from a string name to an int... this gets passed on to everything afterward!
    input_loglevel = logLevels.get(input_loglevel, logging.NOTSET)

    log_format = '%(relativeCreated)-8d %(module)-12s:%(lineno)-5s %(levelname)-8s (%(name)s) %(message)s'
    log_date = None

    if input_logfile == 'stdout' or input_logfile == 'stderr':
        log_stream = logStreams.get(input_logfile, logging.NOTSET)
        log_file = None
    else:
        log_stream = None
        log_file = input_logfile

    log_file_enabled = False
    try:
        logging.basicConfig(level=input_loglevel, format=log_format, datefmt=log_date, filename=log_file, stream=log_stream, filemode='w')
        log_file_enabled = True
    except ValueError:
        pass
    if log_file_enabled is False:
        try:
            logging.basicConfig(level=input_loglevel, format=log_format, datefmt=log_date, filename=log_file, filemode='w')
            log_file_enabled = True
        except ValueError:
            pass
        if log_file_enabled is False:
            try:
                logging.basicConfig(level=input_loglevel, format=log_format, datefmt=log_date, filemode='w')
            except Exception as e:
                sys.stderr.write('unable to config log file %s:\n%s\n' %(log_file, str(e)))
    logging.getLogger().name = "MainThread"
    logging.log(logging.VERBOSE, "Processing started at %s" %(datetime.now()))
    return input_loglevel


###############################################################################
def STATmerge_main(arg_list):
    core_file_type = 'full'
    sys.argv = sys.argv[1:]
    try:
        is_dir = False
        files = arg_list[arg_list.index("-c") + 1:]
        first_file = files[0]
        if os.path.isdir(first_file):
            is_dir = True
            filenames = os.listdir(first_file)
            files = []
            for filename in filenames:
                if filename.find('core') != -1 and not os.path.isdir(filename):
                    files.append(first_file + '/' + filename)
        empty_files = []
        file_types = {}
        file_types['full'] = False
        file_types['lw'] = False
        for filename in files:
            if os.stat(filename).st_size == 0:
                empty_files.append(filename)
            else:
                p = subprocess.Popen(['file', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8', universal_newlines=True)
                output, error = p.communicate()
                if output.find('core file') != -1:
                    file_types['full'] = True
                elif output.find('ASCII text') != -1:
                    f = open(filename, "r")
                    line = f.readline()
                    f.close()
                    if line.find("LIGHTWEIGHT COREFILE") != -1 or line.find("Summary") != -1:
                        file_types['lw'] = True
                        core_file_type = 'lightweight'
        if file_types['full'] == True and file_types['lw'] == True:
            sys.stderr.write('Detected both full and lightweight core files. Please only specify files of one type\n')
            sys.exit(2)
        if empty_files != []:
            sys.stderr.write("Warning: ignoring empty files %s\n" %repr(empty_files))
            for filename in empty_files:
                files.remove(filename)
        arg_list = arg_list[:arg_list.index("-c") + 1]
        sys.argv = sys.argv[:sys.argv.index("-c") + 1]
        arg_list += files
        sys.argv += files
    except Exception as e:
        sys.stderr.write('failed to determine core file type: %s\n' %e)

    if core_file_type == 'lightweight':
        try:
            from bg_core_backtrace import BgCoreTrace, BgCoreMerger, BgCoreMergerArgs
        except Exception as e:
            sys.stderr.write("The following library is missing: bg_core_backtrace\n")
            sys.stderr.write("Lightweight corefile analysis will not be enabled\n")
            sys.stderr.write("%s\n" % (repr(e)))
            sys.exit(1)
        merger = BgCoreMerger(BgCoreTrace, BgCoreMergerArgs)
    else:
        merger = CoreMerger(CoreTrace, CoreMergerArgs)
    ret = merger.run()
    if ret != 0:
        sys.stderr.write('Merger failed\n')
        sys.exit(ret)

if __name__ == '__main__':
    sys.stderr.write('WARNING: core_file_merger.py should not be directly invoked. This has been replaced by STATmain.py and the "merge" subcommand.\n')
    sys.exit(1)
