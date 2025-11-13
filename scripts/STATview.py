#!/bin/env python

"""@package STATview
Visualizes dot graphs outputted by STAT."""

__copyright__ = """Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC."""
__license__ = """Produced at the Lawrence Livermore National Laboratory
Written by Gregory Lee <lee218@llnl.gov>, Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 2
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import string
import sys
import shutil
import os
import math
import re
import subprocess
import traceback
import shelve
from collections import defaultdict
import copy
import ctypes

HAVE_DLOPEN = hasattr(sys, 'getdlopenflags')
if HAVE_DLOPEN is True:
    dlflags = sys.getdlopenflags()
    new_dlflags = ctypes.RTLD_GLOBAL | dlflags
    sys.setdlopenflags(new_dlflags)

(MODEL_INDEX_HIDE, MODEL_INDEX_NAME, MODEL_INDEX_CASESENSITIVE, MODEL_INDEX_REGEX, MODEL_INDEX_EDITABLE, MODEL_INDEX_NOTEDITABLE, MODEL_INDEX_CALLBACK, MODEL_INDEX_ICON, MODEL_INDEX_BUTTONNAME) = range(9)

# Make sure $DISPLAY is set (not good for Windows!)
if os.name != 'nt':
    if not "DISPLAY" in os.environ:
        raise Exception('$DISPLAY is not set.  Ensure that X11 forwarding is enabled.\n')

# Check for required modules

try:
    import xdot
except:
    raise Exception('STATview requires xdot\nxdot can be downloaded from https://github.com/jrfonseca/xdot.py\nWhen installing STAT via Spack, please be sure to `spack activate py-xdot`')
# old xdot vs new xdot compatibility wrapper
try:
    xdot_ui_actions = xdot.ui.actions
    xdot_ui_elements = xdot.ui.elements
    xdot_ui_window = xdot.ui.window
    xdot_dot_parser = xdot.dot.parser
    xdot_dot_lexer = xdot.dot.lexer
except:
    xdot_ui_actions = xdot
    xdot_ui_elements = xdot
    xdot_ui_window = xdot
    xdot_dot_parser = xdot
    xdot_dot_lexer = xdot

has_gtk = False
import_error = ''
try:
    import gtk
    import gobject
    import pango
    gtk_wrap_new_with_label_from_widget = gtk.RadioButton
    gtk_wrap_new_from_widget = gtk.RadioButton
    has_gtk = True
except ImportError as e:
    import_error = e
    pass #raise Exception('%s\nSTATview requires gtk and gobject\n' % repr(e))
except RuntimeError as e:
    import_error = e
    pass #raise Exception('%s\nThere was a problem loading the gtk and gobject module.\nIs X11 forwarding enabled?\n' % repr(e))
except Exception as e:
    import_error = e
    pass #raise Exception('%s\nThere was a problem loading the gtk module.\n' % repr(e))

if has_gtk == False:
    try:
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk as gtk
        from gi.repository import Gdk as gdk
        from gi.repository import GObject as gobject

        # wrap some old functionality
        gtk.FILE_CHOOSER_ACTION_OPEN = gtk.FileChooserAction.OPEN
        gtk.FILE_CHOOSER_ACTION_SAVE = gtk.FileChooserAction.SAVE
        gtk.RESPONSE_CANCEL = gtk.ResponseType.CANCEL
        gtk.RESPONSE_OK = gtk.ResponseType.OK
        gtk.WRAP_CHAR = gtk.WrapMode.CHAR
        gtk.WRAP_WORD = gtk.WrapMode.WORD
        gtk.STATE_SELECTED = gtk.StateType.SELECTED
        gtk.STATE_NORMAL = gtk.StateType.NORMAL
        gtk.STATE_PRELIGHT = gtk.StateType.PRELIGHT
        gtk.SHADOW_ETCHED_IN = gtk.ShadowType.ETCHED_IN
        gtk.SHADOW_ETCHED_OUT = gtk.ShadowType.ETCHED_OUT
        gtk.SHADOW_OUT = gtk.ShadowType.OUT
        gtk.CELL_RENDERER_PRELIT = gtk.CellRendererState.PRELIT
        gtk.POLICY_AUTOMATIC = gtk.PolicyType.AUTOMATIC
        gtk.SELECTION_SINGLE = gtk.SelectionMode.SINGLE
        gtk.CELL_RENDERER_MODE_ACTIVATABLE = gtk.CellRendererMode.ACTIVATABLE
        gtk.TREE_VIEW_COLUMN_FIXED = gtk.TreeViewColumnSizing.FIXED
        gdk.BUTTON_PRESS = gdk.EventType.BUTTON_PRESS
        gtk.FILE_CHOOSER_ACTION_OPEN = gtk.FileChooserAction.OPEN
        gtk.POS_TOP = gtk.PositionType.TOP
        gtk.combo_box_new_text = gtk.ComboBoxText
        gtk.gdk = gdk
        gtk_wrap_new_with_label_from_widget = gtk.RadioButton.new_with_label_from_widget
        gtk_wrap_new_from_widget = gtk.RadioButton.new_from_widget
    except Exception as e2:
        raise Exception('%s\n%s\nThere was a problem loading the gtk module.\n' % (repr(import_error), repr(e2)))

try:
    import STAThelper
    from STAThelper import which, color_to_string, DecomposedNode, decompose_node, HAVE_PYGMENTS, is_mpi, escaped_label, has_source_and_not_collapsed, has_module_offset_and_not_collapsed, label_has_source, label_has_module_offset, label_collapsed, translate, expr, node_attr_to_label, edge_attr_to_label, get_truncated_edge_label, get_num_tasks
except Exception as e:
    raise Exception('%s\nThere was a problem loading the STAThelper module.\n' % repr(e))
if HAVE_PYGMENTS:
    import pygments
    from pygments.lexers import CLexer
    from pygments.lexers import CppLexer
    from pygments.lexers import FortranLexer
    from STAThelper import STATviewFormatter
# Check for optional modules
## A variable to determine whether we have the temporal ordering module
HAVE_TOMOD = True
try:
    import tomod
except Exception as e:
    HAVE_TOMOD = False

## A variable to determine wehther we have the lru_cache decorator
HAVE_FUNCTOOLS=True
try:
    from functools import lru_cache
except ImportError:
    HAVE_FUNCTOOLS=False
if HAVE_FUNCTOOLS==False:
    try:
        from backports.functools_lru_cache import lru_cache
        HAVE_FUNCTOOLS=True
    except:
        pass

## The location of the STAT logo image
try:
    STAT_LOGO = os.path.join(os.path.dirname(__file__), '../../../share/stat/STATlogo.gif')
except:
    STAT_LOGO = 'STATlogo.gif'

## A variable to enable or disable scroll bars (not 100% functional)
use_scroll_bars = False

## A global variable to track the main window object
window = None

## The global map of source line info to lexicographical string
lex_map = {}

## The source and include search paths for temporal ordering and source view
search_paths = {}
search_paths['source'] = []
search_paths['source'].append(os.getcwd())
search_paths['include'] = ["/usr/local/tools/mvapich-gnu/include"]
#search_paths['include'] = []

## A global table to avoid redundant task list generation
task_label_to_list = {}

## A global table to avoid unnecessary parsing of long label strings
task_label_id_to_list = {}

## A global table to avoid unnecessary parsing of long label strings
task_label_to_id = {}

## A counter to keep track of unique label IDs
next_label_id = -1

## A default name for the source cache directory
cache_directory = 'stat_source_cache'

def my_lower(text):
    return text.lower()
string.lower = my_lower

## A decorator that uses the requested decorator upon a condition
class conditional_decorator(object):
    def __init__(self, dec, condition):
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            return func
        return self.decorator(func)


## \param label - the edge label
#  \return the list of ranks
#
#  \n
@conditional_decorator(lambda func: lru_cache(maxsize=None, typed=True)(func), (HAVE_FUNCTOOLS is True))
def get_task_list(label):
    try:
        label = label.decode('utf-8')
    except:
        pass
    if label == '' or label == None:
        return []
    if label[0] != '[':
        # this is just a count and representative
        label = label[label.find(':') + 2:label.find(']')]
        return [int(label)]
    ret = []
    if label in task_label_to_list:
        ret = task_label_to_list[label]
    else:
        ret = STAThelper.get_task_list(label)
        task_label_to_list[label] = ret
        global next_label_id
        next_label_id += 1
        task_label_id_to_list[next_label_id] = ret
        task_label_to_id[label] = next_label_id
    return ret


## \param task_list - the list of tasks
#  \return the string representation
#
#  \n
def list_to_string(task_list):
    """Translate a list of tasks into a range string."""
    global next_label_id
    for key, value in task_label_to_list.items():
        if task_list == value:
            return key
    ret = ''
    in_range = False
    first_iteration = True
    last_val = -1
    for task in task_list:
        if in_range:
            ret += '%d,%d' % (last_val, task)
            in_range = False
        else:
            if first_iteration:
                ret += '%d' % (task)
                first_iteration = False
            else:
                if task == last_val + 1:
                    in_range = True
                    ret += '-'
                else:
                    ret += ',%d' % (task)
        last_val = task
    if in_range:
        ret += '%d' % (task)
    task_label_to_list['[' + ret + ']'] = task_list
    next_label_id += 1
    task_label_id_to_list[next_label_id] = task_list
    task_label_to_id['[' + ret + ']'] = next_label_id
    return ret


## \param dot_filename - the input .dot file
## \param truncate - whether to truncate from "front" or "back"
## \param max_node_name - the max length of the node name to output
#  \return the created temporary dot file name
#
#  \n
def create_temp(dot_filename, truncate, max_node_name):
    """Create a temporary dot file with "..." truncated edge labels
    and compressed node labels."""
    temp_dot_filename = '_temp.dot'
    try:
        temp_dot_file = open(temp_dot_filename, 'w')
    except:
        home_dir = os.environ.get("HOME")
        temp_dot_filename = '%s/_temp.dot' % home_dir
        try:
            temp_dot_file = open(temp_dot_filename, 'w')
        except Exception as e:
            show_error_dialog('Failed to open temp dot file %s for writing' % temp_dot_filename, exception=e)
            return None
    temp_dot_file.write('digraph G {\n\tnode [shape=record,style=filled,labeljust=c,height=0.2];\n')
    try:
        with open(dot_filename, 'r') as dot_file:
            contents = dot_file.read()
            try:
                dot_code = bytes(contents, 'utf-8')
            except:
                dot_code = contents
            parser = STATDotParser(dot_code)
            parser.parse()
            for i, attr in enumerate(parser.graph_attrs.keys()):
                if i == 0:
                    temp_dot_file.write('\tgraph [')
                else:
                    temp_dot_file.write(', ')
                temp_dot_file.write('%s="%s"' %(attr, parser.graph_attrs[attr]))
                if i == len(parser.graph_attrs.keys()) - 1:
                    temp_dot_file.write('];\n')
            for node in parser.nodes:
                id, attrs = node
                attrs["label"] = node_attr_to_label(attrs, False)
                output_line = '\t%s [' % id
                for key in attrs.keys():
                    if key == 'function' or key == 'label':
                        if attrs[key].find('{') != -1:
                            attrs[key] = attrs[key].replace('{', '\{').replace('}', '\}')
                    if key == 'label':
                        label = escaped_label(attrs[key])
                        original_label = label
                        final_label = ''
                        label_lines = label.split('\\n')
                        for i, label in enumerate(label_lines):
                            if len(label) > max_node_name and truncate == "front":
                                # clip long node names at the front (preserve most significant characters)
                                if label[2 - max_node_name] == '\\':
                                    label = '...\\%s' % label[3 - max_node_name:]
                                else:
                                    label = '...%s' % label[3 - max_node_name:]
                            if len(label) > max_node_name and truncate == "rear":
                                # clip long node names at the rear (preserve least significant characters)
                                if label[max_node_name - 1] == '\\':
                                    label = '%s' % label[:max_node_name - 2]
                                else:
                                    label = '%s...' % label[:max_node_name - 3]
                            final_label += label
                            if i != len(label_lines) - 1:
                                final_label += '\\n'
                        output_line += '%s="%s",' % (key, final_label)
                    else:
                        output_line += '%s="%s",' % (key, attrs[key])
                output_line += 'originallabel="%s"];\n' % original_label
                temp_dot_file.write(output_line)
            for edge in parser.edges:
                src_id, dst_id, attrs = edge
                attrs["label"] = edge_attr_to_label(attrs)
                output_line = '\t%s -> %s [' % (src_id, dst_id)
                for key in attrs.keys():
                    if key == 'label':
                        if attrs["label"] == '':
                            pass
                        original_label = attrs["label"]
                        label = get_truncated_edge_label(attrs)
                        output_line += '%s="%s",' % (key, label)
                    else:
                        output_line += '%s="%s",' % (key, attrs[key])
                output_line += 'originallabel="%s"]\n' % original_label
                temp_dot_file.write(output_line)
    except IOError as e:
        show_error_dialog('Failed to open dot file %s' % dot_filename, exception=e)
        return None
    except Exception as e:
        show_error_dialog('Failed to create temporary dot file %s\n %s' % (dot_filename, repr(e)), exception=e)
        return None
    finally:
        temp_dot_file.write('}\n')
        temp_dot_file.close()

    return temp_dot_filename


## \param file_dir - the .dot file directory
## \param source - the source file name
#  \return the cache file path
#
#  \n
def get_cache_file_path(file_dir, source):
    cache_directory_path = os.path.abspath(os.path.join(file_dir, cache_directory))
    cache_file_path = cache_directory_path + '/' + source
    return cache_file_path


## \param current_graph - the currently opened graph object
#  \return directory and filename of the current graph's .dot file
#
#  \n
def dot_file_path_split(current_graph):
    file_path = current_graph.cur_filename
    if file_path is None or file_path == '':
        return (None, None)
    elif file_path == 'redraw.dot':
        return (None, None)
    file_dir, file_name = os.path.split(file_path)
    return (file_dir, file_name)

def run_gtk_main_loop(iters = 100):
    iter = 0
    while gtk.events_pending() and iter < iters:
        iter += 1
        try:
            gtk.main_iteration(False)
        except:
            gtk.main_iteration()


class CellRendererButtonPixbuf(gtk.CellRendererPixbuf):
    __gproperties__ = {"callable": (gobject.TYPE_PYOBJECT, "callable property", "callable property", gobject.PARAM_READWRITE)}
    _button_width = 20
    _button_height = 30

    def __init__(self):
        if hasattr(self, '__gobject_init__'):
            self.__gobject_init__()
        gtk.CellRendererPixbuf.__init__(self)
        self.set_property("xalign", 0.5)
        self.set_property("mode", gtk.CELL_RENDERER_MODE_ACTIVATABLE)
        self.callable = None
        self.table = None

    def do_set_property(self, pspec, value):
        if pspec.name == "callable":
            if callable(value):
                self.callable = value
            else:
                raise TypeError("callable property must be callable!")
        else:
            raise AttributeError("Unknown property %s" % pspec.name)

    def do_get_property(self, pspec):
        if pspec.name == "callable":
            return self.callable
        else:
            raise AttributeError("Unknown property %s" % pspec.name)

    def do_get_size(self, wid, cell_area):
        xpad = self.get_property("xpad")
        ypad = self.get_property("ypad")
        if not cell_area:
            x, y = 0, 0
            w = 2 * xpad + self._button_width
            h = 2 * ypad + self._button_height
        else:
            w = 2 * xpad + cell_area.width
            h = 2 * ypad + cell_area.height
            xalign = self.get_property("xalign")
            yalign = self.get_property("yalign")
            x = max(0, xalign * (cell_area.width - w))
            y = max(0, yalign * (cell_area.height - h))
        return (x, y, w, h)

    def do_activate(self, event, wid, path, bg_area, cell_area, flags):
        cb = self.get_property("callable")
        if cb is not None:
            cb(path)
        return True


class CellRendererButtonText(gtk.CellRendererText):
    __gproperties__ = {"callable": (gobject.TYPE_PYOBJECT, "callable property", "callable property", gobject.PARAM_READWRITE)}
    _button_width = 40
    _button_height = 30

    def __init__(self):
        self.__gobject_init__()
        gtk.CellRendererText.__init__(self)
        self.set_property("xalign", 0.5)
        self.set_property("mode", gtk.CELL_RENDERER_MODE_ACTIVATABLE)
        self.callable = None
        self.table = None

    def do_set_property(self, pspec, value):
        if pspec.name == "callable":
            if callable(value):
                self.callable = value
            else:
                raise TypeError("callable property must be callable!")
        else:
            raise AttributeError("Unknown property %s" % pspec.name)

    def do_get_property(self, pspec):
        if pspec.name == "callable":
            return self.callable
        else:
            raise AttributeError("Unknown property %s" % pspec.name)

    def do_get_size(self, wid, cell_area):
        xpad = self.get_property("xpad")
        ypad = self.get_property("ypad")
        if not cell_area:
            x, y = 0, 0
            w = 2 * xpad + self._button_width
            h = 2 * ypad + self._button_height
        else:
            w = 2 * xpad + cell_area.width
            h = 2 * ypad + cell_area.height
            xalign = self.get_property("xalign")
            yalign = self.get_property("yalign")
            x = max(0, xalign * (cell_area.width - w))
            y = max(0, yalign * (cell_area.height - h))
        return (x, y, w, h)

    def do_activate(self, event, wid, path, bg_area, cell_area, flags):
        cb = self.get_property("callable")
        if cb is not None:
            cb(path)
        return True


## STAT GUI's wait dialog.
class STAT_wait_dialog(object):
    """STAT GUI's wait dialog.

    List subtasks with progress/activity bars and overall progress.
    """

    def __init__(self):
        """The constructor"""
        self.tasks = []
        self.task_progress_bars = []
        self.current_task = 0
        self.wait_dialog = None
        self.cancelled = False
        self.timer = None
        self.progress_bar = None

    ## \param self - the instance
    #  \param fun - the function to run
    #  \param args - the argument list for the function
    #  \param task_list - [optional] the list of subtasks
    #  \param parent - [optional] the parent dialog
    #  \param cancelable - [optional] whether to allow a cancel action
    #  \return the return value of the function that was run
    #
    #  \n
    def show_wait_dialog_and_run(self, fun, args, task_list=[], parent=None, cancelable=False):
        """Display a wait dialog and run the specified function.

        Note, the specified function should NOT create any new windows or do
        any drawing, otherwise it will hang.
        """
        if self.wait_dialog != None:
            self.wait_dialog.destroy()
        self.wait_dialog = gtk.Dialog('Please Wait', parent)
        self.current_task = 0
        self.tasks = []
        self.task_progress_bars = []
        if len(task_list) != 0:
            self.timer = gobject.timeout_add(75, self.update_active_bar)
        for task in task_list:
            task_label = gtk.Label(task)
            task_label.set_line_wrap(False)
            hbox = gtk.HBox()
            hbox.pack_start(task_label, True, False, 0)
            active_bar = gtk.ProgressBar()
            active_bar.set_fraction(0.0)
            self.task_progress_bars.append(active_bar)
            hbox.pack_start(active_bar, False, False, 5)
            self.wait_dialog.vbox.pack_start(hbox, True, True, 0)
        if len(task_list) != 1:
            if len(task_list) == 0:
                label = gtk.Label("Please Wait...")
            else:
                separator = gtk.HSeparator()
                self.wait_dialog.vbox.pack_start(separator, False, False, 5)
                label = gtk.Label("Overall Progress")
            label.set_line_wrap(False)
            hbox = gtk.HBox()
            hbox.pack_start(label, True, False, 0)
            self.progress_bar = gtk.ProgressBar()
            self.progress_bar.set_fraction(0.0)
            hbox.pack_start(self.progress_bar, False, False, 5)
            self.wait_dialog.vbox.pack_start(hbox, True, True, 0)
        if cancelable is True:
            self.wait_dialog.vbox.pack_start(gtk.HSeparator(), False, False, 5)
            hbox = gtk.HButtonBox()
            button = gtk.Button(stock=gtk.STOCK_CANCEL)
            button.connect("clicked", lambda w: self.on_cancel(w))
            hbox.pack_end(button, False, False, 0)
            self.wait_dialog.vbox.pack_start(hbox, True, True, 0)
        self.wait_dialog.show_all()
        ret = self.run_and_destroy_wait_dialog(fun, (args), parent)
        return ret

    def on_cancel(self, widget):
        """Callback to handle clicking of cancel button."""
        self.cancelled = True

    def destroy(self):
        if self.wait_dialog is not None:
            self.wait_dialog.destroy()
            self.wait_dialog = None

    ## \param self - the instance
    #  \param fun - the function to run
    #  \param args - the tuple arguments for the \a fun function
    #  \param parent - [optional] the parent dialog
    #  \return the return value of the function that was run
    #
    #  \n
    def run_and_destroy_wait_dialog(self, fun, args, parent=None):
        """Run the specified command and destroy any pending wait dialog."""
        try:
            ret = fun(*args)
        except Exception as e:
            ret = False
            show_error_dialog('Unexpected error:  %s\n%s\n%s\n' % (sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]), exception=e)
        if len(self.task_progress_bars) != 0:
            gobject.source_remove(self.timer)
        self.destroy()
        return ret

    ## \param self - the instance
    #  \param fraction - the new fraction
    #
    #  \n
    def update(self, fraction):
        """Mark the current task as complete and advance to the next task."""
        if self.current_task >= len(self.task_progress_bars):
            return True
        self.task_progress_bars[self.current_task].set_fraction(1.0)
        self.current_task += 1
        self.progress_bar.set_fraction(fraction)
        run_gtk_main_loop()

    ## \param self - the instance
    #  \param fraction - the new fraction
    #
    #  \n
    def update_progress_bar(self, fraction):
        """Update the fraction of the progress bar."""
        self.progress_bar.set_fraction(fraction)
        run_gtk_main_loop()

    def update_active_bar(self):
        """Register activity on the active bar."""
        if self.current_task >= len(self.task_progress_bars):
            return True
        self.task_progress_bars[self.current_task].pulse()
        return True

## The globally available STAT wait dialog
stat_wait_dialog = STAT_wait_dialog()


## \param text - the error text
#  \param parent - [optional] the parent dialog
#  \param exception - [optional] the exception object
#
#  \n
def show_error_dialog(text, parent=None, exception=None):
    """Display an error dialog."""

    # print traceback information to the terminal
    if exception is not None:
        traceback.print_exc()

    # create error dialog with error message
    if parent == None:
        global window
        parent = window
    error_dialog = gtk.Dialog('Error', parent)
    label = gtk.Label(text)
    label.set_line_wrap(True)
    error_dialog.vbox.pack_start(label, True, True, 0)
    button = gtk.Button(stock=gtk.STOCK_OK)
    button.connect("clicked", lambda w, d: error_dialog.destroy(), "ok")
    error_dialog.vbox.pack_start(button, True, True, 0)
    error_dialog.show_all()
    error_dialog.run()


def load_model_cuts(filename):
    models = {}
    models_case_sensitive = {}
    model_name = None
    try:
        with open(filename, 'r') as model_file:
            for line in model_file:
                line = line.strip('\n')
                if line == '':
                    continue
                if line[0] == '#':
                    continue
                if line.find("model =") == 0 or line.find("model=") == 0:
                    model_name = line[line.find('"') + 1:line.rfind('"')]
                    line = line[line.rfind('"') + 1:].split()
                    case_sensitive = line[0].lower()
                    models[model_name] = ''
                    models_case_sensitive[model_name] = False
                    if case_sensitive == "true":
                        models_case_sensitive[model_name] = True
                    continue
                if model_name is not None:
                    models[model_name] += line + ';'
    except IOError as e:
        #sys.stderr.write('%s\nfailed to open models file %s\n' % (repr(e), filename))
        pass
    except Exception as e:
        traceback.print_exc()
        sys.stderr.write('%s\nfailed to process models file %s\n' % (repr(e), filename))
    model_list = []
    for model in models:
        # toggled, name, case sensitive, regexs, editable, not editable
        model_list.append([False, model, models_case_sensitive[model], models[model], False, True])
    return model_list


def re_search(function_name, search_text_and_match_case):
    search_text, match_case = search_text_and_match_case
    """Function to test whether a search string matches a re"""
    if match_case is False:
        search_text = string.lower(search_text)
        function_name = string.lower(function_name)
    if re.search(search_text, function_name) is not None:
        return True
    return False


## Overloaded DragAction for use with scroll bars.
class STATPanAction(xdot_ui_actions.DragAction):
    """Overloaded DragAction for use with scroll bars."""

    def drag(self, deltax, deltay):
        """Update the scroll bars."""
        if use_scroll_bars:
            if hasattr(self, 'last_deltax') and hasattr(self, 'last_deltay'):
                deltax = deltax + self.last_deltax
                deltay = deltay + self.last_deltay
            self.last_deltax = deltax
            self.last_deltay = deltay
            x = self.dot_widget.hadj.get_value()
            y = self.dot_widget.vadj.get_value()
            rect = self.dot_widget.get_allocation()
            allocation = self.dot_widget.dotsw.get_allocation()
            hspan = allocation[2]
            x += deltax
            if x < 0:
                x = 0
            elif x + hspan > rect.width:
                x = rect.width - hspan
            self.dot_widget.dotsw.get_hscrollbar().emit('change-value', gtk.SCROLL_JUMP, x)
            vspan = allocation[3]
            y += deltay
            if y < 0:
                y = 0
            elif y + vspan > rect.height:
                y = rect.height - vspan
            self.dot_widget.dotsw.get_vscrollbar().emit('change-value', gtk.SCROLL_JUMP, y)
        else:
            xdot_ui_actions.DragAction.drag(self, deltax, deltay)


## A shape composed of multiple sub shapes.
class STATCompoundShape(xdot_ui_elements.CompoundShape):
    """A shape composed of multiple sub shapes.

    Adds a hide attribute to the CompoundShape class.
    """

    def __init__(self, shapes):
        """The constructor"""
        xdot_ui_elements.CompoundShape.__init__(self, shapes)
        self.hide = False

    def draw(self, cr, highlight=False):
        """Draw the compound shape if not hidden.

        Hidden shapes are not completely hidden, rather they are made opaque.
        """
        a_val = 1.0
        if self.hide is True:
            a_val = 0.07
        for shape in self.shapes:
            color = shape.pen.color
            r, g, b, p = color
            shape.pen.color = (r, g, b, a_val)
            color = shape.pen.fillcolor
            r, g, b, p = color
            shape.pen.fillcolor = (r, g, b, a_val)
        xdot_ui_elements.CompoundShape.draw(self, cr, highlight)


## A new object to gather the edge label.
class EdgeLabel(object):
    """A new object to gather the edge label."""

    def __init__(self, item, edge_label, highlight=None):
        """The constructor"""
        self.item = item
        self.edge_label = edge_label
        if highlight is None:
            highlight = set([item])
        self.highlight = highlight


## A new object to gather the node label.
class Label(object):
    """A new object to gather the node label."""

    def __init__(self, item, label, highlight=None):
        """The constructor"""
        self.item = item
        if highlight is None:
            highlight = set([item])
        self.highlight = highlight


## The base class for STAT graph nodes and edges.
class STATElement(STATCompoundShape):
    """The base class for STAT graph nodes and edges."""

    def __init__(self, shapes):
        """The constructor"""
        STATCompoundShape.__init__(self, shapes)

    def get_edge_label(self, x, y):
        """Get the edge label (should be overloaded by derrived class)."""
        return None

    def get_label(self, x, y):
        """Get the label (should be overloaded by derrived class)."""
        return None

    def get_jump(self, x, y):
        """Get the jump (should be overloaded by derrived class)."""
        return None

    def draw(self, cr, highlight=False):
        """Draw the object."""
        STATCompoundShape.draw(self, cr, highlight)


## A node in the STAT graph.
class STATNode(STATElement):
    """A node in the STAT graph.

    Derrived from the STATElement to include the hide attribute.
    It adds several STAT specific members to the xdot Node class.
    """

    def __init__(self, x, y, x1, y1, x2, y2, shapes, label, attrs):
        """The constructor"""
        STATElement.__init__(self, shapes)
        self.x = x
        self.y = y
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.attrs = {}
        for key in attrs:
            try:
                self.attrs[key] = attrs[key].decode('utf-8')
            except:
                self.attrs[key] = attrs[key]
        try:
            label = label.decode('utf-8')
        except:
            pass
        self.attrs["label"] = label
        self.source_dir = None
        self.edge_label = None
        self.edge_label_id = -2
        self.hide = False
        self.in_edge = None
        self.out_edges = []
        self.node_name = None
        self.num_tasks = -1
        self.num_leaf_tasks = -1
        self.undo = []
        self.redo = []
        self.is_leaf = False
        self.depth = -1
        self.eq_depth = -1
        self.to_color_index = -1
        self.temporally_ordered = False

    def __repr__(self):
        ret = "STAT Node:\n"
        ret += "node_name(%s)\n" % self.node_name
        ret += "edge_label(%d: %s)\n" % (self.edge_label_id, self.edge_label)
        ret += "hide(%s)\n" % str(self.hide)
        ret += "num_tasks(%d)\n" % self.num_tasks
        ret += "attrs:\n"
        for key in self.attrs.keys():
            ret += "\tattrs[%s] = %s\n" % (key, self.attrs[key])
        ret += "\nin_edge:\n%s\n" % str(self.in_edge)
        for edge in self.out_edges:
            ret += "\nout_edge:\n%s\n" % str(edge)
        return ret

    def is_inside(self, x, y):
        """Determine if the specified coordinates are in this node."""
        return self.x1 <= x and x <= self.x2 and self.y1 <= y and y <= self.y2

    def get_label(self, x, y):
        """Get the node's label."""
        if self.attrs["label"] is None:
            return None
        if self.is_inside(x, y):
            return Label(self, self.attrs["label"])
        return None

    def get_url(self, x, y):
        """Dummy function to satisfy xdot motion notify."""
        return None

    def get_edge_label(self, x, y):
        """Get the node's incoming edge label."""
        if self.edge_label is None:
            return None
        if self.is_inside(x, y):
            return EdgeLabel(self, self.edge_label)
        return None

    def get_jump(self, x, y):
        """Get the jump object if specified coordintes are in the node."""
        if self.hide is True:
            return None
        if self.is_inside(x, y):
            return xdot_ui_elements.Jump(self, self.x, self.y)
        return None

    def draw(self, cr, highlight=False):
        """Draw the node."""
        STATElement.draw(self, cr, highlight)

    def get_text(self):
        """Return the text shape for the node."""
        for shape in self.shapes:
            if isinstance(shape, xdot_ui_elements.TextShape):
                return shape.t
        return ''

    def set_text(self, text):
        """Set the text for the text shape of this node."""
        for shape in self.shapes:
            if isinstance(shape, xdot_ui_elements.TextShape):
                shape.t = text
                try:
                    shape.layout.set_text(text)
                except:
                    pass
                return True
        return False

    #  \return the task list
    #
    #  \n
    def get_node_task_list(self):
        """Get the task list corresponding to the node's edge label.

        First see if we have this label indexed to avoid duplicate generation."""

        if self.edge_label_id in task_label_id_to_list:
            return task_label_id_to_list[self.edge_label_id]
        ret = get_task_list(self.edge_label)
        if self.edge_label != '' and self.edge_label[0] == '[': # not the head node and not count + rep
            self.edge_label_id = task_label_to_id[self.edge_label]
        return get_task_list(self.edge_label)

    #  \return the task list
    #
    #  \n
    def get_leaf_tasks(self):
        """Get the list of tasks that ended on this node."""
        in_set = set(self.get_node_task_list())
        out_set = set([])
        for edge in self.out_edges:
            out_set |= set(edge.dst.get_node_task_list())
        return sorted(list(in_set - out_set))

    #  \return the task count
    #
    #  \n
    def get_num_leaf_tasks(self):
        """Get the number of tasks that ended on this node."""
        if self.num_leaf_tasks == -1:
            out_sum = 0
            for edge in self.out_edges:
                out_sum += get_num_tasks(edge.attrs["label"])
            num_tasks = get_num_tasks(self.edge_label)
            if out_sum < num_tasks:
                self.num_leaf_tasks = num_tasks - out_sum
            else:
                self.num_leaf_tasks = 0
        return self.num_leaf_tasks

    #  \return the task count
    #
    #  \n
    def get_num_visual_leaf_tasks(self):
        """Get the number of tasks that ended on this node."""
        out_sum = 0
        for edge in self.out_edges:
            if edge.hide is False:
                out_sum += get_num_tasks(edge.attrs["label"])
        num_tasks = get_num_tasks(self.edge_label)
        if out_sum < num_tasks:
            return num_tasks - out_sum
        else:
            return 0

    def can_join_eq_c(self):
        if self.hide is True:
            return False
        ret = False
        if len(self.out_edges) == 1 and self.in_edge is not None:
            if self.out_edges[0].attrs["label"] == self.in_edge.attrs["label"] and self.out_edges[0].hide is False:
                ret = True
        return ret


## An edge in the STAT graph.
class STATEdge(STATElement):
    """An edge in the STAT graph.

    Derrived from the STATElement to include the hide attribute.
    It adds several STAT specific members to the xdot Edge class.
    """

    def __init__(self, src, dst, points, shapes, label, attrs):
        """The constructor."""
        STATElement.__init__(self, shapes)
        self.src = src
        self.dst = dst
        self.points = points
        self.attrs = {}
        for key in attrs:
            try:
                self.attrs[key] = attrs[key].decode('utf-8')
            except:
                self.attrs[key] = attrs[key]
        self.attrs["label"] = label
        self.hide = False
        self.undo = []
        self.redo = []

    RADIUS = 10

    def __repr__(self):
        ret = "STATEdge:\n"
        try:
            ret += "src(%s)\n" % str(self.src.attrs["label"])
        except:
            ret += "src(?)\n"
        try:
            ret += "dst(%s)\n" % str(self.dst.attrs["label"])
        except:
            ret += "dst(%s)\n" % str(self.dst.attrs["label"])
        ret += "hide(%s)\n" % str(self.hide)
        ret += "attrs:\n"
        for key in self.attrs.keys():
            ret += "\tattrs[%s] = %s\n" % (key, self.attrs[key])
        return ret

    def get_jump(self, x, y):
        """Get the jump object if specified coordintes are in the edge."""
        if xdot_ui_elements.square_distance(x, y, *self.points[0]) <= self.RADIUS*self.RADIUS:
            return xdot_ui_elements.Jump(self, self.dst.x, self.dst.y, highlight=set([self, self.dst]))
        if xdot_ui_elements.square_distance(x, y, *self.points[-1]) <= self.RADIUS*self.RADIUS:
            return xdot_ui_elements.Jump(self, self.src.x, self.src.y, highlight=set([self, self.dst]))
        return None


## A STAT graph object.
class STATGraph(xdot_ui_elements.Graph):
    """A STAT graph object.

    Derrived from xdot's Graph class and adds several STAT specific
    operations for tree manipulation and traversal.
    """

    def __init__(self, width=1, height=1, shapes=(), nodes=(), edges=()):
        """The constructor."""
        xdot_ui_elements.Graph.__init__(self, width, height, shapes, nodes, edges)
        for node in self.nodes:
            if self.is_leaf(node):
                node.is_leaf = True
            node.depth = self.get_node_depth(node)
            node.eq_depth = self.get_node_eq_depth(node)
        self.action_history = []
        self.undo_action_history = []
        self.cur_filename = ''
        self.to_var_visit_list = []
        self.source_view_window = None
        self.source_view_notebook = None
        self.source_view_tabs = []

    def get_node(self, x, y):
        """Get the node that the specified coordinates fall into."""
        for node in self.nodes:
            if node.is_inside(x, y):
                return node
        return None

    def get_edge_label(self, x, y):
        """Get the edge label of the node that the coordinates fall into."""
        for node in self.nodes:
            edge_label = node.get_edge_label(x, y)
            if edge_label is not None:
                return edge_label
        return None

    def get_label(self, x, y):
        """Get the label of the node that the coordinates fall into."""
        for node in self.nodes:
            if not hasattr(node, "get_label"):
                return None
            label = node.get_label(x, y)
            if label is not None:
                return label
        return None

    def set_undo_list(self):
        """Save modifiable node attributes to a list for undo operations."""
        for node in self.nodes:
            node_text = node.get_text()
            node.undo.append((node.hide, node_text, node.to_color_index, node.attrs.get("lex_string"), node.temporally_ordered))
            node.redo = []
        for edge in self.edges:
            edge_colors = []
            for shape in edge.shapes:
                r, g, b, p = shape.pen.color
                r2, g2, b2, p2 = shape.pen.fillcolor
                edge_colors.append((r, g, b, p, r2, g2, b2, p2))
            edge.undo.append((edge.hide, edge_colors))
            edge.redo = []

    def undo(self, update_redo=True):
        """Undo the modifications of the previous operation.

        Restore previous attributes.
        """
        if len(self.nodes) == 0:
            return True
        if len(self.nodes[0].undo) == 0:
            return True
        if update_redo:
            op = self.action_history.pop()
            self.undo_action_history.append(op)
        for node in self.nodes:
            hide, node_text, to_color_index, lex_string, temporally_ordered = node.undo.pop()
            old_node_text = node.get_text()
            node.set_text(node_text)
            if update_redo:
                node.redo.append((node.hide, old_node_text, node.to_color_index, node.attrs.get("lex_string"), node.temporally_ordered))
            node.hide = hide
            node.to_color_index = to_color_index
            if lex_string is not None:
                node.attrs["lex_string"] = lex_string
            node.temporally_ordered = temporally_ordered
            if hasattr(node, 'eq_collapsed_out_edges'):
                del node.eq_collapsed_out_edges
            if "eq_collapsed_label" in node.attrs:
                del node.attrs["eq_collapsed_label"]
        for edge in self.edges:
            edge_colors = []
            for shape in edge.shapes:
                r, g, b, p = shape.pen.color
                r2, g2, b2, p2 = shape.pen.fillcolor
                edge_colors.append((r, g, b, p, r2, g2, b2, p2))
            if update_redo:
                edge.redo.append((edge.hide, edge_colors))
            if len(edge.undo) > 0:
                hide, edge_colors = edge.undo.pop()
            for i in range(len(edge.shapes)):
                r, g, b, p, r2, g2, b2, p2 = edge_colors[i]
                edge.shapes[i].pen.color = (r, g, b, p)
                edge.shapes[i].pen.fillcolor = (r2, g2, b2, p2)
            edge.hide = hide
        return True

    def redo(self, widget):
        """Redo the modifications of the previous operation.

        Restore previous undo attributes.
        """
        if len(self.nodes) == 0:
            return True
        if len(self.nodes[0].redo) == 0:
            return True
        op = self.undo_action_history.pop()
        self.action_history.append(op)
        for node in self.nodes:
            hide, node_text, to_color_index, lex_string, temporally_ordered = node.redo.pop()
            old_node_text = node.get_text()
            node.set_text(node_text)
            node.undo.append((node.hide, old_node_text, node.to_color_index, node.attrs.get("lex_string"), node.temporally_ordered))
            node.hide = hide
            node.to_color_index = to_color_index
            if lex_string is not None:
                node.attrs["lex_string"] = lex_string
            node.temporally_ordered = temporally_ordered
        for edge in self.edges:
            shapes = []
            for shape in edge.shapes:
                r, g, b, p = shape.pen.color
                r2, g2, b2, p2 = shape.pen.fillcolor
                shapes.append((r, g, b, p, r2, g2, b2, p2))
            edge.undo.append((edge.hide, shapes))
            hide, shapes = edge.redo.pop()
            for i in range(len(edge.shapes)):
                r, g, b, p, r2, g2, b2, p2 = shapes[i]
                edge.shapes[i].pen.color = (r, g, b, p)
                edge.shapes[i].pen.fillcolor = (r2, g2, b2, p2)
            edge.hide = hide
        return True

    def expand(self, node):
        """Show any children nodes of the specified node."""
        modified = False
        if node is None:
            return False
        for edge in node.out_edges:
            if edge.dst.hide is True:
                edge.dst.hide = False
                modified = True
            edge.hide = False
        return modified

    def collapse(self, node, orig=False):
        """Hide any children nodes of the specified node."""
        modified = False
        if node is None:
            return False
        if orig is False:
            if node.hide is False:
                node.hide = True
                modified = True
        if node.in_edge is not None and orig is False:
            if node.in_edge.hide is False:
                node.in_edge.hide = True
                modified = True
        for edge in node.out_edges:
            edge.hide = True
            ret = self.collapse(edge.dst)
            if ret is True:
                modified = True
        return modified

    def collapse_depth(self, node):
        """Hide all nodes below the depth of the specified node."""
        modified = False
        if node is None:
            return False
        for cnode in self.nodes:
            if cnode.depth == node.depth:
                ret = self.collapse(cnode, True)
                if ret is True:
                    modified = True
        return modified

    def join_eq_c(self, node, root=False):
        """Collapse descendent nodes of the same eq class into a single node."""
        modified = False
        if node is None or node.hide is True:
            return False, (None, {})
        if not node.can_join_eq_c():
            node.hide = True
            node.in_edge.hide = True
            return True, (node, node.attrs)
        modified, (leaf_node, attrs) = self.join_eq_c(node.out_edges[0].dst)
        if root is True:
            #node.out_edges[0].hide
            node.eq_collapsed_out_edges = []
            if leaf_node is not None:
                for edge in leaf_node.out_edges:
                    new_edge = STATEdge(node, edge.dst, edge.points, edge.shapes, edge.attrs["label"], edge.attrs)
                    new_edge.hide = edge.hide
                    edge.hide = True
                    node.eq_collapsed_out_edges.append(new_edge)
                    new_edge.dst.eq_collapsed_in_edge = new_edge
                    self.edges.append(new_edge)
                node.attrs["eq_collapsed_label"] = node.attrs["label"] + '\\n' + attrs["label"]
        else:
            node.hide = True
            node.in_edge.hide = True
        for attr in ["label", "function", "source", "line", "module", "offset", "pc", "vars"]:
            if attr in node.attrs:
                node.attrs[attr] = node.attrs[attr] + '\\n' + attrs[attr]
        return modified, (leaf_node, node.attrs)

    def expand_all(self, node):
        """Show all descendents of the specified node."""
        modified = False
        if node is None:
            return False
        if node.hide is True:
            node.hide = False
            modified = True
        if node.in_edge is not None:
            node.in_edge.hide = False
        for edge in node.out_edges:
            edge.hide = False
            ret = self.expand_all(edge.dst)
            if ret is True:
                modified = True
        return modified

    def visible_children(self, node):
        """Find all visible (not hidden) children of the specified node."""
        ret_nodes = []
        ret_edges = []
        if node is not None:
            if node.hide is False:
                ret_nodes.append(node)
                if node.in_edge is not None:
                    ret_edges.append(node.in_edge)
            for edge in node.out_edges:
                if edge.hide is False:
                    n, e = self.visible_children(edge.dst)
                    ret_nodes += n
                    ret_edges += e
        return ret_nodes, ret_edges

    def focus(self, node):
        """Focus on a node

        Hide all nodes that are neither descendents nor ancestors of the
        specified node.
        """
        if node is None:
            return False
        show_nodes, show_edges = self.visible_children(node)
        num_hidden_nodes = 0
        for temp in self.nodes:
            if temp.hide is True:
                num_hidden_nodes += 1
            temp.hide = True
        for temp in self.edges:
            temp.hide = True
        temp_node = node
        while 1:
            temp_node.hide = False
            if temp_node.in_edge is not None:
                temp_node.in_edge.hide = False
                temp_node = temp_node.in_edge.src
            else:
                break
        for temp in show_nodes:
            temp.hide = False
        for temp in show_edges:
            temp.hide = False
        num_hidden_nodes2 = 0
        for temp in self.nodes:
            if temp.hide is True:
                num_hidden_nodes2 += 1
        if num_hidden_nodes == num_hidden_nodes2:
            return False
        else:
            return True

    def view_source(self, node, item=0):
        """Generate a window that displays the source code for the node."""
        # find the source file name and line number
        if not "source" in node.attrs.keys():
            show_error_dialog('Cannot determine source file, please run STAT with the -i option to get source file and line number information\n')
            return
        if node.attrs["source"] == "(null)" or node.attrs["source"] == "?":
            show_error_dialog('Cannot determine source file, please run STAT with the -i option to get source file and line number information\n')
            return

        source = node.attrs["source"].split('\\n')[item]
        cur_line_num = int(node.attrs["line"].split('\\n')[item].strip(":"))

        # get the node font and background colors
        for shape in node.shapes:
            if isinstance(shape, xdot_ui_elements.TextShape):
                font_color = shape.pen.color
            else:
                fill_color = shape.pen.fillcolor
        font_color_string = color_to_string(font_color)
        fill_color_string = color_to_string(fill_color)

        # find all nodes that are in this source file
        line_nums = []
        line_nums.append((cur_line_num, fill_color_string, font_color_string))
        for node_iter in self.nodes:
            if not "source" in node_iter.attrs:
                continue
            sources = node_iter.attrs["source"].split('\\n')
            for i, this_source in enumerate(sources):
                if this_source in ["(null)", "?"]:
                    continue
                if this_source == source:
                    for shape in node_iter.shapes:
                        if isinstance(shape, xdot_ui_elements.TextShape):
                            font_color = shape.pen.color
                        else:
                            fill_color = shape.pen.fillcolor
                    font_color_string = color_to_string(font_color)
                    fill_color_string = color_to_string(fill_color)
                    line_nums.append((int(node_iter.attrs["line"].split('\\n')[i].strip(":")), fill_color_string, font_color_string))
        error_msg = ''
        file_dir, file_name = dot_file_path_split(self)
        source_full_path = None
        if file_dir is not None and file_name is not None:
            cache_file_path = get_cache_file_path(file_dir, source)
            if os.path.exists(cache_file_path):
                source_full_path = cache_file_path
        if source_full_path == None:
            if os.path.isabs(source):
                if os.path.exists(source):
                    source_full_path = source
                else:
                    error_msg = 'Full path %s does not exist\n' % source
            if source_full_path is None:
                # find the full path to the file
                source = os.path.basename(source)
                for sp in search_paths['source']:
                    if not os.path.exists(sp):
                        continue
                    tmp_source_full_path = sp + '/' + source
                    if os.path.exists(tmp_source_full_path):
                        source_full_path = tmp_source_full_path
                        break
                if source_full_path is None:
                    show_error_dialog('%sFailed to find file "%s" in search paths.  Please add the source file search path for this file\n' % (error_msg, source))
                    return True

        # create the source view window
        if self.source_view_window is None:
            self.source_view_window = gtk.Window()
            self.source_view_window.set_default_size(800, 600)
            self.source_view_window.set_title('Source View %s' % source)
            self.source_view_window.connect('destroy', self.on_source_view_destroy)
        if self.source_view_notebook is None:
            self.source_view_notebook = gtk.Notebook()
            self.source_view_window.add(self.source_view_notebook)
        frame = gtk.Frame(label="")
        source_view = gtk.TextView()
        source_view.set_wrap_mode(False)
        source_view.set_editable(False)
        source_view.set_cursor_visible(False)
        source_string = ''
        count = 0
        source_view.get_buffer().create_tag("monospace", family="monospace")
        iterator = source_view.get_buffer().get_iter_at_offset(0)

        # determine the max number of ==> pointers so we can make space
        line_count_map = {}
        line_num_tuples = []
        for lineNum, fill_color_string, font_color_string in line_nums:
            if lineNum in line_count_map:
                if not (lineNum, fill_color_string, font_color_string) in line_num_tuples:
                    line_count_map[lineNum] += 1
            else:
                line_count_map[lineNum] = 1
            line_num_tuples.append((lineNum, fill_color_string, font_color_string))
        max_count = 0
        for line_count in line_count_map.values():
            if line_count > max_count:
                max_count = line_count

        # print the actual text with line nums and ==> arrows
        try:
            if HAVE_PYGMENTS:
                STAThelper.pygments_lines = []
            with open(source_full_path, 'r') as file:
                if HAVE_PYGMENTS:
                    cpp_file_patterns = ['.C', '.cpp', '.c++', '.cc', '.cxx', '.hpp', '.h++', '.hh', '.hxx']
                    fortran_file_patterns = ['.f', '.F', '.f90', '.F90']
                    source_extension = os.path.splitext(os.path.basename(source_full_path))[1]
                    if source_extension in cpp_file_patterns:
                        pygments.highlight(file.read(), CppLexer(), STATviewFormatter())
                    elif source_extension in fortran_file_patterns:
                        pygments.highlight(file.read(), FortranLexer(), STATviewFormatter())
                    else:  # default to C
                        pygments.highlight(file.read(), CLexer(), STATviewFormatter())
                    lines = STAThelper.pygments_lines
                    source_view.get_buffer().create_tag('bold_tag', weight=pango.WEIGHT_BOLD)
                    source_view.get_buffer().create_tag('italics_tag', style=pango.STYLE_ITALIC)
                    source_view.get_buffer().create_tag('underline_tag', underline=pango.UNDERLINE_SINGLE)
                else:
                    lines = file.readlines()
        except IOError as e:
            show_error_dialog('%s\nFailed to open file "%s"\n' % (repr(e), source_full_path), exception=e)
        except Exception as e:
            show_error_dialog('%s\nFailed to process source file "%s"\n' % (repr(e), source_full_path), exception=e)
        width = len(str(len(lines)))
        cur_line_iter = source_view.get_buffer().get_iter_at_offset(0)
        cur_line_mark = source_view.get_buffer().create_mark('cur_line', cur_line_iter, True)
        for line in lines:
            count += 1
            line_count = 0
            line_num_tuples = []
            if count == cur_line_num:
                cur_line_iter = source_view.get_buffer().get_iter_at_offset(0)
                cur_line_iter.set_line(cur_line_num)
                cur_line_mark = source_view.get_buffer().create_mark('cur_line', cur_line_iter, True)
            for lineNum, fill_color_string, font_color_string in line_nums:
                if count == lineNum and not (lineNum, fill_color_string, font_color_string) in line_num_tuples:
                    line_count += 1
                    line_num_tuples.append((lineNum, fill_color_string, font_color_string))
            for i in range(max_count - line_count):
                source_string = "   "
                source_view.get_buffer().insert_with_tags_by_name(iterator, source_string, "monospace")
            line_num_tuples = []
            for lineNum, fill_color_string, font_color_string in line_nums:
                if count == lineNum and not (lineNum, fill_color_string, font_color_string) in line_num_tuples:
                    foreground = gtk.gdk.color_parse(font_color_string)
                    background = gtk.gdk.color_parse(fill_color_string)
                    fore_color_tag = "color_fore%d%s" % (lineNum, font_color_string)
                    try:
                        source_view.get_buffer().create_tag(fore_color_tag, foreground_gdk=foreground)
                    except:
                        pass
                    back_color_tag = "color_back%d%s" % (lineNum, fill_color_string)
                    try:
                        source_view.get_buffer().create_tag(back_color_tag, background_gdk=background)
                    except:
                        pass
                    source_string = "==>"
                    source_view.get_buffer().insert_with_tags_by_name(iterator, source_string, fore_color_tag, back_color_tag, "monospace")
                    line_num_tuples.append((lineNum, fill_color_string, font_color_string))
            if HAVE_PYGMENTS:
                source_string = "%0*d| " % (width, count)
                source_view.get_buffer().insert_with_tags_by_name(iterator, source_string, "monospace")
                for item in line:
                    source_string, format_tuple = item
                    pygments_color, bold, italics, underline = format_tuple
                    foreground = gtk.gdk.color_parse(pygments_color)
                    fore_color_tag = "color_fore%d%s" % (lineNum, pygments_color)
                    try:
                        source_view.get_buffer().create_tag(fore_color_tag, foreground_gdk=foreground)
                    except:
                        pass
                    args = [iterator, source_string, fore_color_tag, "monospace"]
                    if bold:
                        args.append('bold_tag')
                    if italics:
                        args.append('italics_tag')
                    if underline:
                        args.append('underline_tag')
                    source_view.get_buffer().insert_with_tags_by_name(*tuple(args))
            else:
                source_string = "%0*d| " % (width, count)
                source_string += line
                source_view.get_buffer().insert_with_tags_by_name(iterator, source_string, "monospace")

        # create the tab label box
        tab_label_box = gtk.EventBox()
        label = gtk.Label(os.path.basename(source))
        tab_label_box.add(label)
        self.source_view_tabs.append(tab_label_box)
        tab_label_box.connect('event', self.on_source_tab_clicked, tab_label_box)
        label.show()

        # add the sw to the notebook and set current page
        self.source_view_notebook.append_page(frame, tab_label_box)
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        sw.add(source_view)
        frame.add(sw)
        self.source_view_window.show_all()
        self.source_view_window.present()
        pages = self.source_view_notebook.get_n_pages()
        self.source_view_notebook.set_current_page(pages - 1)

        # run iterations to get sw generated so we can scroll
        run_gtk_main_loop()
        try:
            source_view.scroll_to_mark(cur_line_mark, 0.0, True)
        except:
            source_view.scroll_to_mark(cur_line_mark, 0.0, True, 0.0, 0.0)

    def on_source_view_destroy(self, action):
        """Clean up source view state."""
        self.source_view_window = None
        self.source_view_notebook = None
        self.source_view_tabs = []

    def on_source_tab_clicked(self, widget, event, child):
        """Switch the focus to the clicked tab."""
        if event.type == gtk.gdk.BUTTON_PRESS:
            n = -1
            for tab in self.source_view_tabs:
                n += 1
                if child == tab:
                    break
            self.source_view_notebook.set_current_page(n)
            if event.button == 3:
                options = ['Close Tab']
                menu = gtk.Menu()
                for option in options:
                    menu_item = gtk.MenuItem(option)
                    menu.append(menu_item)
                    menu_item.connect('activate', self.menu_item_response, option)
                    menu_item.show()
                try:
                    menu.popup(None, None, None, None, event.button, event.time)
                except:
                    menu.popup(None, None, None, event.button, event.time)

    def menu_item_response(self, widget, string):
        """Handle tab menu responses."""
        if string == "Close Tab":
            if len(self.source_view_tabs) == 1:
                self.source_view_window.destroy()
            else:
                page = self.source_view_notebook.get_current_page()
                self.source_view_notebook.remove_page(page)
                self.source_view_tabs.remove(self.source_view_tabs[page])

    def get_lex(self, node, just_add=False):
        """Generate a lexicographical string for the specified node."""
        if node is None:
            return False
        if not has_source_and_not_collapsed(node.attrs["label"]):
            return False
        if node.attrs["label"].find('libc_start') != -1:
            node.attrs["lex_string"] = "(null)"
            return True
        if is_mpi(node.attrs["function"]):
            node.attrs["lex_string"] = "(null)"
            return True
        lex_map_index = node.attrs["source"] + node.attrs["line"]
        if node.attrs.get("lex_string") is None:
            if node.attrs["source"] == "(null)":
                return False
            if lex_map_index in lex_map.keys():
                lex_string = lex_map[lex_map_index]
                if lex_string.find('$') != -1 and node.attrs["vars"] != '(null)':
                    input_val = node.attrs["vars"][node.attrs["vars"].find('=') + 1:]
                    lex_string = lex_string[:lex_string.find('$')] + input_val + lex_string[lex_string.find(')') + 1:]
                node.attrs["lex_string"] = lex_string
                return True
            source = node.attrs["source"]
            line = node.attrs["line"].strip(":")
            if os.path.isabs(source):
                node.source_dir, source = os.path.split(source)
                node.source_dir += '/'
            else:
                found = False
                for sp in search_paths['source']:
                    if not os.path.exists(sp):
                        continue
                    if os.path.exists(sp + '/' + source):
                        found = True
                        break
                if found is False:
                    show_error_dialog('Failed to find file "%s" in search paths.  Please add the source file search path for this file\n' % source)
                    return False
                node.source_dir = sp + '/'
            to_input = tomod.to_tuple()
            to_input.basepath = node.source_dir
            to_input.filename = source
            to_input.line = int(line)
            if just_add is True:
                tomod.add_program_point(to_input)
                return True
            lex_string = tomod.get_lex_string(to_input)
            if lex_string.find('$') != -1 and node.attrs["vars"] != '(null)':
                input_val = node.attrs["vars"][node.attrs["vars"].find('=') + 1:]
                lex_string = lex_string[:lex_string.find('$')] + input_val + lex_string[lex_string.find(')') + 1:]
            node.attrs["lex_string"] = lex_string
            lex_map[lex_map_index] = lex_string
            if lex_string == '':
                return True

    def get_children_temporal_order(self, node):
        """Generate temporal strings for the children of the specified node."""
        if node is None:
            return False
        if node.out_edges == []:
            return False
        found = False
        for edge in node.out_edges:
            if self.get_to_string(edge.dst) == '':
                found = True
                break
        if found is False:
            return False
        source_map = defaultdict(list)
        progress = 0.0
        #import time
        #t11 = time.time()
        error_nodes = []
        tomod.clear_program_points()
        for edge in node.out_edges:
            temp_node = edge.dst
            ret = self.get_lex(temp_node, True)
            if ret is False:
                return False
        tomod.run_to()
        for edge in node.out_edges:
            temp_node = edge.dst
            if temp_node.hide is True:
                temp_node.hide = False
                temp_node.in_edge.hide = False
            #t1 = time.time()
            ret = self.get_lex(temp_node)
            if ret is False:
                return False
            #t2 = time.time()
            #print t2 - t1
            if temp_node.attrs.get("lex_string") is None or ret is False:
                error_nodes.append(temp_node.attrs["label"])
                continue
            if not has_source_and_not_collapsed(temp_node.attrs["label"]):
                error_nodes.append(temp_node.attrs["label"])
                continue
            if temp_node.attrs["source"] in ["(null)", "?"] or temp_node.attrs["line"] in ["(null)", ":?"]:
                error_nodes.append(temp_node.attrs["label"])
                continue
            source = temp_node.attrs["source"]
            line = temp_node.attrs["line"].strip(":")
            if not "lex_string" in node.attrs.keys():
                temp_node.attrs["lex_string"] = ""
            temp_string = temp_node.attrs["lex_string"][temp_node.attrs["lex_string"].find('#') + 1:]
            temp_string = temp_string[:temp_string.find('#')]
            # temp_string is now the line number offset of the function
            index_string = source + '#' + temp_string
            to_input = tomod.to_tuple()
            to_input.basepath = temp_node.attrs["lex_string"]
            to_input.filename = source
            to_input.line = int(line)
            source_map[index_string].append((to_input, temp_node))
            progress += 1.0 / len(node.out_edges)
            if progress > 1.0:
                progress = 1.0
            stat_wait_dialog.update_progress_bar(progress)
        #t12 = time.time()
        #print t12 - t11
        #t1 = time.time()
        if error_nodes != []:
            show_error_dialog('Failed to create lexicographical string for %s.  Please be sure to gather stack traces with line number information and be sure to include the appropriate search paths\n' % str(error_nodes))
            return False
        for key, source in source_map.items():
            tomod.clear_program_points()
            found = False
            skip_node_rename_list = []
            for to_input_tuple, node in source:
                if node.attrs["lex_string"].find('$') != -1 or node.attrs["label"].find('$') != -1:
                    # check if this is the first time visiting this variable
                    if node not in self.to_var_visit_list:
                        if node.attrs["label"].find('=') != -1:
                            found = True
                        else:
                            temp = node.attrs["lex_string"][node.attrs["lex_string"].find('$') + 1:]
                            var = temp[:temp.find('(')]
                            if var != 'iter#':
                                skip_node_rename_list.append(node)
                        self.to_var_visit_list.append(node)
                tomod.add_program_point(to_input_tuple)
            if found is True:
                for to_input_tuple, node in source:
                    # set lex string to None so we visit this next time around
                    if "lex_string" in node.attrs.keys():
                        del node.attrs["lex_string"]
                return False
            for to_input_tuple, node in source:
                if node in skip_node_rename_list:
                    continue
                temporal_string = tomod.get_temporal_string(to_input_tuple)
                parent = node.in_edge.src
                parent_temporal_string = ''
                if parent is not None:
                    parent_temporal_string = self.get_to_string(parent)
                node.temporally_ordered = True
                if parent_temporal_string == '':
                    node.attrs["temporal_string"] = node.attrs["function"] + "@T" + temporal_string
                    node.set_text(node.attrs["temporal_string"])
                    node.attrs["label"] = node.attrs["temporal_string"]
                else:
                    node.attrs["temporal_string"] = node.attrs["function"] + "@T" + parent_temporal_string + '.' + temporal_string
                    node.set_text(node.attrs["temporal_string"])
                    node.attrs["label"] = node.attrs["temporal_string"]
        self.color_temporally_ordered_edges()
        #t2 = time.time()
        #print t2 - t1
        return True

# TODO
#    def update_children_temporal_string(self, node, temporal_string):
#        if not self.has_to_descendents():
#            return True
#        for edge in node.out_edges:
#            return False

    def get_to_string(self, node):
        """Get the temporal string from the specified node."""
        temporal_string = node.attrs.get("temporal_string", "")
        ret_string = temporal_string[temporal_string.find('@T') + 2:]
        return ret_string

    def has_to_descendents(self, node):
        """Determine whether the specified node has children that have been temporally ordered."""
        for edge in node.out_edges:
            temp_node = edge.dst
            name = self.get_to_string(temp_node)
            if name != '':
                return True
            if self.has_to_descendents(temp_node):
                return True
        return False

    ## \param self - the instance
    #  \param color_index - a value from 0 to 255
    #  \return a tuple of (red, blue) color values from 0.0 to 1.0
    #
    #  \n
    def to_color_index_to_rb(self, color_index):
        """Translate a color index into red and blue color values."""
        red = 255.0/255.0
        blue = 255.0/255.0
        if color_index <= 255:
            blue = float(color_index)/255.0
        else:
            red = float(511 - color_index)/255.0
        return red, blue

    ## \param self - the instance
    #  \param nodes - a list of nodes to check
    #  \param leaves_only - [optional] only find leaf nodes
    #  \return a tuple of (TO leaves, TO strings, TO node ancestors)
    #
    #  \n
    def get_to_list(self, nodes, leaves_only=False):
        """Find all temporally ordered nodes and their ancestors."""
        to_leaves = []  # list of (node, to_leaf_string) tuples
        temporal_string_list = []
        node_list = []
        # generate list of all to leafs
        max_len = 1
        duplicates = 0
        for node in nodes:
            if self.has_to_descendents(node) is False or leaves_only is False:
                temporal_string = self.get_to_string(node)
                if temporal_string == '':
                    continue
                temporal_string = temporal_string.replace('.?', '')
                for num_string in temporal_string.split('.'):
                    if len(num_string) > max_len:
                        max_len = len(num_string)
                if temporal_string in temporal_string_list:
                    duplicates += 1
                else:
                    temporal_string_list.append(temporal_string)
                to_leaves.append((temporal_string, node))
                node_list.append(node)
        # sort the list by temporal_strings
        if max_len > 1:
            # pad with leading zeros if necessary
            for i in range(len(to_leaves)):
                temporal_string, node = to_leaves[i]
                new_temporal_string = ''
                for num_string in temporal_string.split('.'):
                    new_temporal_string += '%0*d' % (max_len, int(num_string))
                to_leaves[i] = new_temporal_string, node
                temporal_string_list[temporal_string_list.index(temporal_string)] = new_temporal_string
        to_leaves.sort()
        temporal_string_list.sort()
        return to_leaves, temporal_string_list, node_list

    def color_temporally_ordered_edges(self):
        """Color the edges based on temporal ordering.

        From red (least progress) to blue (most progress).
        """
        to_leaves, temporal_string_list, node_list = self.get_to_list(self.nodes, True)
        # assign colors
        maxDepth = 0
        for i in range(len(to_leaves)):
            temporal_string, node = to_leaves[i]
            if node.depth > maxDepth:
                maxDepth = node.depth
        depths = range(maxDepth + 1)
        depths.remove(0)
        try:
            depths.remove(1)
        except:
            pass
        depths.reverse()
        for i in depths:
            for j in range(len(to_leaves)):
                temporal_string, node = to_leaves[j]
                if node.depth != i:
                    continue
                if (len(temporal_string_list) == 1):
                    node.to_color_index = 0
                else:
                    node.to_color_index = 511 * temporal_string_list.index(temporal_string) / (len(temporal_string_list) - 1)
                red, blue = self.to_color_index_to_rb(node.to_color_index)
                green = 0.0
                for shape in node.in_edge.shapes:
                    r, g, b, p = shape.pen.color
                    shape.pen.color = (red, green, blue, p)
                    r, g, b, p = shape.pen.fillcolor
                    shape.pen.fillcolor = (red, green, blue, p)
            # assign colors up the tree too
            for node in self.nodes:
                if node.depth == i and not node in node_list and self.has_to_descendents(node):
                    min_to_color_index = 511
                    for edge in node.out_edges:
                        temp_node = edge.dst
                        if temp_node is None:
                            continue
                        if temp_node.to_color_index == -1:
                            continue
                        if temp_node.to_color_index < min_to_color_index:
                            min_to_color_index = temp_node.to_color_index
                    node.to_color_index = min_to_color_index
                    if node.to_color_index == -1:
                        continue
                    red, blue = self.to_color_index_to_rb(node.to_color_index)
                    green = 0.0
                    for shape in node.in_edge.shapes:
                        r, g, b, p = shape.pen.color
                        shape.pen.color = (red, 0, blue, p)
                        r, g, b, p = shape.pen.fillcolor
                        shape.pen.fillcolor = (red, 0, blue, p)

    def get_node_depth(self, node):
        """Determine the depth of the specified node."""
        if not hasattr(node, "in_edge"):
            return 1
        if node.in_edge is not None:
            return 1 + self.get_node_depth(node.in_edge.src)
        else:
            return 1

    def get_node_eq_depth(self, node):
        """Determine the equivalence class depth.

        i.e., number of colors along the path of the specified node."""
        if not hasattr(node, "in_edge"):
            return 1
        if node.in_edge is not None:
            parent_node = node.in_edge.src
            if parent_node.in_edge is None:
                return 1
            if node.in_edge.attrs["label"] != parent_node.in_edge.attrs["label"]:
                return 1 + self.get_node_eq_depth(node.in_edge.src)
            return self.get_node_eq_depth(node.in_edge.src)
        else:
            return 1

    ## \param self - the instance
    #  \param filename - the output file name
    #
    #  \n
    def save_folded_trace(self, filename):
        """Save the current graph as a folded trace.
        The folded trace can then be used to generate a
        flame graph (https://github.com/brendangregg/FlameGraph)"""

        try:
            with open(filename, 'w') as f:
                for node in self.nodes:
                    if node.hide:
                        continue
                    #if node.is_leaf:
                    if self.node_is_visual_leaf(node):
                        temp_node = node
                        count = node.get_num_visual_leaf_tasks()
                        #count = node.get_num_leaf_tasks()
                        trace = node.attrs["label"]
                        x = 0
                        while 1:
                            if temp_node.in_edge == None:
                                break
                            temp_node = temp_node.in_edge.src
                            if temp_node == None:
                                break
                            trace = temp_node.attrs["label"] + ";" + trace
                        f.write(trace + " " + str(count) + "\n")
        except IOError as e:
            show_error_dialog('%s\nFailed to open file "%s" for writing' % (repr(e), filename), exception=e)
            return False
        except Exception as e:
            show_error_dialog('%s\nFailed to process file "%s" for writing' % (repr(e), filename), exception=e)
            return False
        return True

    ## \param self - the instance
    #  \param filename - the output file name
    #  \param full_edge_label - [optional] whether to save full edge labels, defaults to True
    #  \param full_node_label - [optional] whether to save full node labels, defaults to True
    #  \param translate_module_offset - [optional] whether to translate module offset traces to source + line, defaults to False
    #
    #  \n
    def save_dot(self, filename, full_edge_label=True, full_node_label=True, translate_module_offset=False, invert=False):
        """Save the current graph as a dot file."""
        invert_color = False
        if not invert is False:
            invert_color = invert.attrs["fillcolor"]
        try:
            f2 = open("test.perf-folded", "w")
            with open(filename, 'w') as f:
                f.write('digraph G {\n')
                f.write('\tgraph [type="stat_%d_%d"];\n' %(__version_major__, __version_minor__))
                f.write('\tnode [shape=record,style=filled,labeljust=c,height=0.2];\n')
                for node in self.nodes:
                    if node.hide:
                        continue
                    for shape in node.shapes:
                        if isinstance(shape, xdot_ui_elements.TextShape):
                            font_color = shape.pen.color
                            if invert_color is not False and node.attrs['fillcolor'] == invert_color:
                                if node.attrs['fontcolor'] == '#000000':
                                    node.attrs['fontcolor'] = '#FFFFFF'
                                elif node.attrs['fontcolor'] == '#FFFFFF':
                                    node.attrs['fontcolor'] = '#000000'
                            node_text = shape.t
                        else:
                            fill_color = shape.pen.fillcolor

                    if translate_module_offset is True and node.attrs["module"] != "(null)":
                        eq_collapsed_label = ''
                        for i, label in enumerate(node.attrs["label"].split('\\n')):
                            module_offset = "%s%s" %(node.attrs["module"].split('\\n')[i], node.attrs["offset"].split('\\n')[i])
                            node_text = re.sub(expr, translate, module_offset)
                            function = node_text[0:node_text.rfind('@')]
                            source = node_text[node_text.rfind('@') + 1:node_text.rfind(':')]
                            line = node_text[node_text.rfind(':') + 1:]
                            if i == 0:
                                node.attrs["function"] = function
                                node.attrs["source"] = source
                                node.attrs["line"] = line
                                eq_collapsed_label = function + '@' + source + ':' + line
                            else:
                                node.attrs["function"] += '\\n' + function
                                node.attrs["source"] += '\\n' + source
                                node.attrs["line"] += '\\n' + line
                                eq_collapsed_label += '\\n' + function + '@' + source + ':' + line
                        if "eq_collapsed_label" in node.attrs.keys():
                            node.attrs["eq_collapsed_label"] = eq_collapsed_label
                    elif full_node_label is True:
                        node_text = node_attr_to_label(node.attrs)
                        if "eq_collapsed_label" in node.attrs.keys():
                            node.attrs["eq_collapsed_label"] = node_text
                    f.write('\t%s [pos="0,0", label="%s"' % (node.node_name, escaped_label(node_text)))
                    for attr in node.attrs.keys():
                        if attr not in ["pos", "label", "_draw_", "_ldraw_", "width", "height", "rects", "temporal_string", "lex_string"]:
                            f.write(', %s="%s"' %(attr, node.attrs[attr]))
                    f.write('];\n')
                for edge in self.edges:
                    if edge.hide:
                        continue
                    if full_edge_label is True:
                        output_label = edge.dst.edge_label
                    else:
                        output_label = get_truncated_edge_label(edge.attrs)
                    if output_label.find(':') == -1:
                        num_threads = -1
                        if "tbv" in edge.attrs and edge.attrs["tbv"] != "(null)":
                            num_threads = int(edge.attrs["tbv"])
                        elif "tcount" in edge.attrs and edge.attrs["tcount"] != "(null)":
                            num_threads = int(edge.attrs["tcount"])
                        counts_string = '%d' %num_threads
                        if num_threads != -1:
                            counts_string += '(%d)' %num_threads
                        counts_string += ':'
                        output_label = counts_string + output_label
                    f.write('\t%s -> %s [label="%s"' % (edge.src.node_name, edge.dst.node_name, output_label))
                    for attr in edge.attrs.keys():
                        if attr not in ["label", "_draw_", "_ldraw_", "_hdraw_", "pos", "lp"]:
                            f.write(', %s="%s"' %(attr, edge.attrs[attr]))
                    f.write(']\n')
                f.write('}\n')
        except IOError as e:
            show_error_dialog('%s\nFailed to open file "%s" for writing' % (repr(e), filename), exception=e)
            return False
        except Exception as e:
            show_error_dialog('%s\nFailed to process file "%s" for writing' % (repr(e), filename), exception=e)
            return False
        return True

    def save_file(self, filename):
        """Save the current graph into a dot file or image file."""
        if os.path.splitext(filename)[1] == '.dot':
            ret = self.save_dot(filename)
        elif os.path.splitext(filename)[1] == '.perf-folded':
            ret = self.save_folded_trace(filename)
        else:
            temp_dot_filename = os.path.splitext(filename)[0] + '_tmp.dot'
            ret = self.save_dot(temp_dot_filename, False, False)
            if ret is True:
                file_format = '-T' + os.path.splitext(filename)[1][1:]
                proc = subprocess.Popen(["dot", file_format, temp_dot_filename, "-o", filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                stdout_output, stderr_output = proc.communicate()
                if stderr_output != '':
                    sys.stderr.write('dot outputted error message: %s\n' % stderr_output)
                if proc.returncode != 0:
                    show_error_dialog('Failed to process file "%s" for writing: %s' % (temp_dot_filename, stderr_output))
            os.remove(temp_dot_filename)
        return ret

    def on_original_graph(self, widget):
        """Restore the call prefix tree to its original state."""
        widget.user_zoom = False
        modified = False
        for node in self.nodes:
            if "lex_string" in node.attrs:
                del node.attrs["lex_string"]
            if "temporal_string" in node.attrs:
                del node.attrs["temporal_string"]
            node.temporally_ordered = False
            if node.hide is True:
                node.hide = False
                modified = True
        for edge in self.edges:
            if edge.hide is True:
                edge.hide = False
                modified = True
            for shape in edge.shapes:
                r, g, b, p = shape.pen.color
                shape.pen.color = (0.0, 0.0, 0.0, p)
                r, g, b, p = shape.pen.fillcolor
                shape.pen.fillcolor = (0.0, 0.0, 0.0, p)
        return modified

    def hide_mpi(self):
        """Hide the MPI implementation frames."""
        return self.hide_generic(is_mpi)

    def hide_re(self, search_text, match_case):
        return self.hide_generic(re_search, search_text, match_case)

    def hide_generic(self, func, *args):
        """Hide frames that match the specified function."""
        modified = False
        for node in self.nodes:

            function_names = node.attrs["function"].split('\\n')
            for i, function_name in enumerate(function_names):
                if args != ():
                    hide = func(function_name, args)
                else:
                    hide = func(function_name)
                if hide is True:
                    ret = self.collapse(node, True)
                    if ret is True:
                        modified = True
                        break
        return modified

    def traverse(self, widget, traversal_depth):
        """Traverse the call prefix tree by equivalence class depth."""
        show_list = []
        max_eq_depth = 0
        for node in self.nodes:
            if node.eq_depth > max_eq_depth:
                max_eq_depth = node.eq_depth
            if node.eq_depth <= traversal_depth:
                show_list.append(node)
        if traversal_depth > max_eq_depth:
            return False
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in show_list:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        return True

    def to_traverse_least_progress(self, widget, traversal_depth):
        """Traverse the call prefix tree by least progress."""
        if traversal_depth == 1:
            for node in self.nodes:
                node.hide = True
            for edge in self.edges:
                edge.hide = True
        for node in self.nodes:
            if node.in_edge is None:
                root = node
                node.hide = False
#                for edge in root.out_edges:
#                    edge.hide = False
                if root.out_edges is not None:
                    if root.out_edges[0].dst.attrs["label"].find('start') != -1 and root.out_edges[0].dst.attrs["label"].find('libc') == -1:
                        root = root.out_edges[0].dst
                        root.temporally_ordered = True
                break
        ret = self._traverse_progress(root, 'least')
        return ret

    def to_traverse_most_progress(self, widget, traversal_depth):
        """Traverse the call prefix tree by most progress."""
        if traversal_depth == 1:
            for node in self.nodes:
                node.hide = True
            for edge in self.edges:
                edge.hide = True
        for node in self.nodes:
            if node.in_edge is None:
                root = node
                node.hide = False
                for edge in root.out_edges:
                    edge.hide = False
                if root.out_edges is not None:
                    for edge in root.out_edges:
                        if edge.dst.attrs["label"].find('start') != -1 and edge.dst.attrs["label"].find('libc') == -1:
                            edge.dst.temporally_ordered = True
                            root = edge.dst
                            edge.hide = False
                            break
#                    if root.out_edges[0].dst.attrs["label"].find('start') != -1 and root.out_edges[0].dst.attrs["label"].find('libc') == -1:
#                        root = root.out_edges[0].dst
                break
        ret = self._traverse_progress(root, 'most')
        return ret

    def _traverse_progress(self, node, least_or_most):
        """Generic implementation for traversal by progress."""
        node.hide = False
        if node.in_edge != None:
            node.in_edge.hide = False
        if node.out_edges is None or node.out_edges == []:
            return False
        if len(node.out_edges) == 0:
            return True
        found_lex_string = False
        for edge in node.out_edges:
            if edge.dst.temporally_ordered is True:
                edge.hide = False
                edge.dst.hide = False
                found_lex_string = True
        if found_lex_string is False:
            if is_mpi(node.attrs["function"]):
                return False
            # we want to TO this node's children
            ret = self.get_children_temporal_order(node)
            if ret is False:
                return ret
            if len(node.out_edges) == 1:
                if node.out_edges[0].dst.attrs["lex_string"].find('$') != -1:
                    return True
                self._traverse_progress(node.out_edges[0].dst, least_or_most)
            else:
                #print 'multiple children => done'
                return True
        else:
            # we need to traverse down the tree further
            if len(node.out_edges) == 1:
                #print 'one child, keep going'
                ret = self._traverse_progress(node.out_edges[0].dst, least_or_most)
                return ret
            elif len(node.out_edges) != 0:
                #print 'multiple children... choose'
                dst_nodes = []
                for edge in node.out_edges:
                    dst_nodes.append(edge.dst)
                to_leaves, temporal_string_list, node_list = self.get_to_list(dst_nodes)
                if least_or_most == 'most':
                    to_leaves.reverse()
                for temporal_string, temp_node in to_leaves:
                    ret = self._traverse_progress(temp_node, least_or_most)
                    if ret is True:
                        return ret
                return False
        return True

    def focus_tasks(self, task):
        """Hide all call paths that are not taken by the specified tasks."""
        task = task.strip(' ')
        task = '[' + task + ']'
        task_list_set = set(get_task_list(task))
        for node in self.nodes:
            if node.node_name == '0':
                continue
            if task_list_set & set(node.get_node_task_list()) == set([]):
                node.hide = True
        for edge in self.edges:
            if task_list_set & set(edge.dst.get_node_task_list()) == set([]):
                edge.hide = True
        return True

    def focus_text(self, text, case_sensitive=True):
        """Hide all call paths that do not contain the search text."""
        for node in self.nodes:
            if node.node_name == '0':
                self.depth_first_search_text(node, text, case_sensitive)
        return True

    def depth_first_search_text(self, node, text, case_sensitive=True):
        """search for specified text in a depth first order."""
        ret = False
        if node.hide is True:
            return False
        for edge in node.out_edges:
            if self.depth_first_search_text(edge.dst, text, case_sensitive):
                ret = True
        search_text = text
        node_text = node.attrs["label"]
        if case_sensitive is False:
            search_text = string.lower(text)
            node_text = string.lower(node.attrs["label"])
        if re.search(search_text, node_text) is not None:
            ret = True
        if ret is False and node.node_name != '0':
            node.hide = True
            node.in_edge.hide = True
        return ret

    def longest_path(self, widget, longest_depth):
        """Traverse the call prefix tree by longest path."""
        longest_map = {}
        for node in self.nodes:
            if node.is_leaf:
                try:
                    longest_map[node.depth].append(node)
                except:
                    longest_map[node.depth] = []
                    longest_map[node.depth].append(node)
        keys = list(longest_map.keys())
        if len(keys) < longest_depth:
            return False
        keys.sort()
        keys.reverse()
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in longest_map[keys[longest_depth - 1]]:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        return True

#    def longest_path(self, widget):
#        longest = 0
#        long_list = []
#        for node in self.nodes:
#            if node.hide is True:
#                continue
#            if node.depth == longest:
#                long_list.append(node)
#            elif node.depth > longest:
#                longest = node.depth
#                long_list = []
#                long_list.append(node)
#        for node in self.nodes:
#            node.hide = True
#        for edge in self.edges:
#            edge.hide = True
#        for node in long_list:
#            temp_node = node
#            while 1:
#                temp_node.hide = False
#                if temp_node.in_edge is not None:
#                    temp_node.in_edge.hide = False
#                    temp_node = temp_node.in_edge.src
#                else:
#                    break

    def is_leaf(self, node):
        """Determine if the node is the leaf of any task's call path."""
        if not hasattr(node, "node_name"):
            return False
        if node.node_name == '0':
            return False
        child_task_list = []
        for edge in node.out_edges:
            if edge.dst is not None:
                child_task_list += edge.dst.get_node_task_list()
        if set(child_task_list) == set(node.get_node_task_list()):
            return False
        return True

    def shortest_path(self, widget, shortest_depth):
        """Traverse the call prefix tree by shortest path."""
        shortest_map = {}
        for node in self.nodes:
            if node.is_leaf:
                try:
                    shortest_map[node.depth].append(node)
                except:
                    shortest_map[node.depth] = []
                    shortest_map[node.depth].append(node)
        keys = list(shortest_map.keys())
        if len(keys) < shortest_depth:
            return False
        keys.sort()
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in shortest_map[keys[shortest_depth - 1]]:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        return True

#    def shortest_path(self, widget):
#        shortest = 999999
#        short_list = []
#        for node in self.nodes:
#            if node.hide is True:
#                continue
#            if node.is_leaf:
#                if node.depth == shortest:
#                    short_list.append(node)
#                elif node.depth < shortest:
#                    shortest = node.depth
#                    short_list = []
#                    short_list.append(node)
#        for node in self.nodes:
#            node.hide = True
#        for edge in self.edges:
#            edge.hide = True
#        for node in short_list:
#            temp_node = node
#            while 1:
#                temp_node.hide = False
#                if temp_node.in_edge is not None:
#                    temp_node.in_edge.hide = False
#                    temp_node = temp_node.in_edge.src
#                else:
#                    break

    def least_tasks(self, widget, least_tasks):
        """Traverse the call prefix tree by least tasks."""
        least_map = {}
        for node in self.nodes:
            task_count = node.get_num_leaf_tasks()
            if task_count == 0:
                continue
            try:
                least_map[task_count].append(node)
            except:
                least_map[task_count] = []
                least_map[task_count].append(node)
        keys = list(least_map.keys())
        if len(keys) < least_tasks:
            return False
        keys.sort()
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in least_map[keys[least_tasks - 1]]:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        return True

    def most_tasks(self, widget, most_tasks):
        """Traverse the call prefix tree by most tasks."""
        most_map = {}
        for node in self.nodes:
            task_count = node.get_num_leaf_tasks()
            if task_count == 0:
                continue
            try:
                most_map[task_count].append(node)
            except:
                most_map[task_count] = []
                most_map[task_count].append(node)
        keys = list(most_map.keys())
        if len(keys) < most_tasks:
            return False
        keys.sort()
        keys.reverse()
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in most_map[keys[most_tasks - 1]]:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        return True

    def single_task_path(self, widget):
        """Hides any call path whose leaf was visited by multpile tasks."""
        single_task_list = []
        hide_count = 0
        for node in self.nodes:
            if node.hide is True:
                hide_count += 1
            if node.num_tasks == 1 and node.hide is False:
                single_task_list.append(node)
        for node in self.nodes:
            node.hide = True
        for edge in self.edges:
            edge.hide = True
        for node in single_task_list:
            temp_node = node
            while 1:
                temp_node.hide = False
                if temp_node.in_edge is not None:
                    temp_node.in_edge.hide = False
                    temp_node = temp_node.in_edge.src
                else:
                    break
        hide_count2 = 0
        for node in self.nodes:
            if node.hide is True:
                hide_count2 += 1
        if hide_count == hide_count2:
            return False
        else:
            return True
        return True

    def adjust_dims(self):
        """Readjust the viewing area based on visible nodes."""
        minx = 999999.9
        miny = 999999.9
        maxx = -999999.9
        maxy = -999999.9
        for node in self.nodes:
            if node.x1 < minx and node.hide is False:
                minx = node.x1
            if node.y1 < miny and node.hide is False:
                miny = node.y1
        if use_scroll_bars:
            for node in self.nodes:
                if node.x2 > maxx and node.hide is False:
                    maxx = node.x2
                if node.y2 > maxy and node.hide is False:
                    maxy = node.y2
            self.visible_width = int(math.ceil(maxx) - math.floor(minx))
            self.visible_height = int(math.ceil(maxy) - math.floor(miny))
            self.maxx = maxx
            self.maxy = maxy
            #self.width = int(math.ceil(maxx))
            #self.height = int(math.ceil(maxy))
        else:
            for node in self.nodes:
                node.x1 = node.x1 - minx
                node.x2 = node.x2 - minx
                node.y1 = node.y1 - miny
                node.y2 = node.y2 - miny
                node.x = node.x - minx
                node.y = node.y - miny
                for shape in node.shapes:
                    try:
                        shape.x = shape.x - minx
                        shape.y = shape.y - miny
                    except:
                        pass
                    try:
                        points = []
                        for x, y in shape.points:
                            points.append((x - minx, y - miny))
                        shape.points = points
                    except:
                        pass
                    if node.x2 > maxx and node.hide is False:
                        maxx = node.x2
                    if node.y2 > maxy and node.hide is False:
                        maxy = node.y2
            for edge in self.edges:
                points = []
                for x, y in edge.points:
                    points.append((x - minx, y - miny))
                edge.points = points
                for shape in edge.shapes:
                    try:
                        shape.x = shape.x - minx
                        shape.y = shape.y - miny
                    except:
                        pass
                    try:
                        points = []
                        for x, y in shape.points:
                            points.append((x - minx, y - miny))
                        shape.points = points
                    except:
                        pass
            self.width = int(math.ceil(maxx))
            self.height = int(math.ceil(maxy))

    def node_is_visual_leaf(self, node):
        """Determine if the node is a leaf of the visible tree."""
        if node.hide is True:
            return False
        for edge in node.out_edges:
            if edge.dst.hide is False:
                return False
        return True

    def node_is_visual_eq_leaf(self, node):
        """Determine if the node is an eq class leaf of the visible tree."""
        if node.hide is True or node.node_name == '0':
            return False

        task_list = []
        if node.edge_label.find(':') != -1:
            num_leaf_tasks = node.get_num_visual_leaf_tasks()
            if num_leaf_tasks != 0:
                return True
            else:
                return False
        else:
            for edge in node.out_edges:
                if edge.dst.hide is False:
                    task_list += edge.dst.get_node_task_list()
            if set(node.get_node_task_list()) == set(task_list):
                return False
            return True

    def identify_num_eq_classes(self, widget):
        """Find all equivalence classes (based on color) of the call tree."""
        leaves = []
        for node in self.nodes:
            if self.node_is_visual_eq_leaf(node):
                leaves.append(node)
        task_leaf_list = []
        leaf_task_sets = []
        for node in leaves:
            leaf_task_sets.append(set(node.get_node_task_list()))
        for node in leaves:
            # get the node font and background colors
            for shape in node.shapes:
                if isinstance(shape, xdot_ui_elements.TextShape):
                    font_color = shape.pen.color
                else:
                    fill_color = shape.pen.fillcolor
            font_color_string = color_to_string(font_color)
            fill_color_string = color_to_string(fill_color)

            # adjust task list for eq class leaves with children
            task_list = node.get_node_task_list()
            task_list_set = set(task_list)
            for edge in node.out_edges:
                if edge.dst.hide is True:
                    continue
                if set(edge.dst.get_node_task_list()) in leaf_task_sets:
                    task_list_set = task_list_set - set(edge.dst.get_node_task_list())
            task_list = list(task_list_set)
            if (task_list, fill_color_string, font_color_string) not in task_leaf_list:
                task_leaf_list.append((task_list, fill_color_string, font_color_string))
        return task_leaf_list

    def identify_real_num_eq_classes(self, widget):
        """Finds all equivalence classes (based on visited paths) of the call tree."""
        leaves = []
        for node in self.nodes:
            if node.hide is False:
                leaves.append(node)
        task_leaf_map = {}
        for node in leaves:
            task_list = node.get_node_task_list()
            for task in task_list:
                try:
                    task_leaf_map[task] += '#%s' % node.node_name
                except:
                    task_leaf_map[task] = node.node_name
        num_eq_classes = {}
        for task in task_leaf_map:
            try:
                num_eq_classes[task_leaf_map[task]].append(task)
            except:
                num_eq_classes[task_leaf_map[task]] = [task]
        return num_eq_classes


class STATDotParser(xdot_dot_parser.DotParser):

    def __init__(self, dot_code):
        self.nodes = []
        self.edges = []
        self.graph_attrs = {}
        xdot_dot_parser.DotParser.__init__(self, xdot_dot_lexer.DotLexer(buf=dot_code))

    def handle_graph(self, attrs):
        new_attrs = {}
        for key in attrs:
            if type(attrs[key]) != bytes:
                self.graph_attrs.update(attrs)
                return
            new_attrs[key] = attrs[key].decode("utf-8")
        self.graph_attrs.update(new_attrs)

    def handle_node(self, id, attrs):
        new_attrs = {}
        for key in attrs:
            if type(attrs[key]) != bytes:
                self.nodes.append((id, attrs))
                return
            new_attrs[key] = attrs[key].decode("utf-8")
        self.nodes.append((id.decode("utf-8"), new_attrs))

    def handle_edge(self, src_id, dst_id, attrs):
        new_attrs = {}
        for key in attrs:
            if type(attrs[key]) != bytes:
                self.edges.append((src_id, dst_id, attrs))
                return
            new_attrs[key] = attrs[key].decode("utf-8")
        self.edges.append((src_id.decode("utf-8"), dst_id.decode("utf-8"), new_attrs))


## The STAT XDot Parser.
class STATXDotParser(xdot_dot_parser.XDotParser):
    """The STAT XDot Parser.

    Derrived form the XDotParser and overrides some XDotParser methods
    to build a STATGraph.
    """
    # TODO: use attrs to add STAT-specific values to nodes and edges

    def __init__(self, xdotcode):
        """The constructor."""
        self.graph_attrs = {}
        xdot_dot_parser.XDotParser.__init__(self, xdotcode)

    def parse(self):
        """Parse the dot file."""
        xdot_dot_parser.DotParser.parse(self)
        if ("type" in self.graph_attrs.keys()) and (self.graph_attrs["type"] == "dysect"):
            raise Exception('This is a DySectAPI .dot graph, open with dysect-view')
        return STATGraph(self.width, self.height, (), self.nodes, self.edges)

    def handle_graph(self, attrs):
        self.graph_attrs.update(attrs)

        new_attrs = {}
        for key in attrs:
            if type(attrs[key]) != bytes:
                xdot_dot_parser.XDotParser.handle_graph(self, attrs)
                return
            new_attrs[key] = attrs[key].decode("utf-8")
        xdot_dot_parser.XDotParser.handle_graph(self, attrs)

    def handle_node(self, node_id, attrs):
        """Handle a node attribute to create a STATNode."""
        try:
            node_id = node_id.decode('utf-8')
        except:
            pass
        new_id = node_id.strip('"')
        xdot_dot_parser.XDotParser.handle_node(self, new_id, attrs)
        node = self.node_by_name[new_id]
        label = attrs.get('originallabel', None)
        if label == None:
            return False
        stat_node = STATNode(node.x, node.y, node.x1, node.y1, node.x2, node.y2, node.shapes, label, attrs)
        stat_node.node_name = new_id
        self.node_by_name[new_id] = stat_node
        self.nodes.pop()
        self.nodes.append(stat_node)

    def handle_edge(self, src_id, dst_id, attrs):
        """Handle an edge attribute to create a STATNode."""
        try:
            src_id = src_id.decode('utf-8')
            dst_id = dst_id.decode('utf-8')
        except:
            pass
        new_src_id = src_id.strip('"')
        new_dst_id = dst_id.strip('"')
        xdot_dot_parser.XDotParser.handle_edge(self, new_src_id, new_dst_id, attrs)
        label = attrs.get('label', None)
        if label == None:
            return False
        try:
            label = label.decode('utf-8')
        except:
            pass
        edge = self.edges.pop()
        stat_edge = STATEdge(edge.src, edge.dst, edge.points, edge.shapes, label, attrs)
        self.edges.append(stat_edge)
        stat_edge.src.out_edges.append(stat_edge)
        stat_edge.dst.in_edge = stat_edge
        new_dst_label = attrs.get('originallabel', None)
        try:
            new_dst_label = new_dst_label.decode('utf-8')
        except:
            pass
        if new_dst_label == None:
            new_dst_label = label# attrs["label"]
        if src_id == '0':
            self.node_by_name[new_src_id].edge_label = ''
            self.node_by_name[new_src_id].num_tasks = get_num_tasks(new_dst_label)
        stat_edge.attrs["label"] = new_dst_label
        self.node_by_name[new_dst_id].edge_label = new_dst_label
        self.node_by_name[new_dst_id].num_tasks = get_num_tasks(new_dst_label)


## The STATNullAction overloads the xdot NullAction.
class STATNullAction(xdot_ui_actions.DragAction):
    """The STATNullAction overloads the xdot NullAction.

    Allows highlighting the entire call path of a given node.
    """

    def on_motion_notify(self, event):
        """Highlight the call path of the node/edge that is moused over."""
        dot_widget = self.dot_widget
        item = dot_widget.get_label(event.x, event.y)
        if item is None:
            item = dot_widget.get_jump(event.x, event.y)
        if item is not None:
            node = dot_widget.get_node(event.x, event.y)
            if node is not None:
                if not hasattr(node, "hide"):
                    return False
                if node.hide is False:
                    if hasattr(dot_widget.window, "set_cursor"):
                        dot_widget.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.ARROW))
                    else:
                        dot_widget.get_window().set_cursor(gdk.Cursor(gdk.CursorType.ARROW))
                    highlight_list = []
                    while node is not None:
                        #highlight_list.append(node)
                        edge = node.in_edge
                        if edge is None:
                            break
                        highlight_list.append(edge)
                        node = edge.src
                    dot_widget.set_highlight(highlight_list)
            else:
                if hasattr(dot_widget.window, "set_cursor"):
                    dot_widget.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.ARROW))
                else:
                    dot_widget.get_window().set_cursor(gdk.Cursor(gdk.CursorType.ARROW))
                dot_widget.set_highlight(None)
        else:
            if hasattr(dot_widget.window, "set_cursor"):
                dot_widget.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.ARROW))
            else:
                dot_widget.get_window().set_cursor(gdk.Cursor(gdk.CursorType.ARROW))
            dot_widget.set_highlight(None)


## PyGTK widget that draws STAT generated dot graphs.
class STATDotWidget(xdot_ui_window.DotWidget):
    """PyGTK widget that draws STAT generated dot graphs.

    Derrived from xdot's DotWidget class.
    """

    def __init__(self):
        """The constructor."""
        xdot_ui_window.DotWidget.__init__(self)
        self.graph = STATGraph()
        try:
            self.window = self
        except:
            # TODO: for some reason python 2 + xdot.py doesn't like this
            pass
        self.drag_action = STATNullAction(self)
        self.user_zoom = False

    def set_dotcode(self, dotcode, filename='<stdin>', truncate="front", max_node_name=64, center=True):
        """Set the dotcode for the widget.

        Create a temporary dot file with truncated edge labels from the
        specified dotcode and generates xdotcode with layout information.
        """
        if filename != '<stdin>':
            temp_dot_filename = create_temp(filename, truncate, max_node_name)
            if temp_dot_filename is None:
                return False
        try:
            f = open(temp_dot_filename, 'r')
            contents = f.read()
            # We need to break lines so they are no longer than 16K characters
            new_contents = ''
            for line in contents.split('\n'):
                while len(line) > 16000:
                    new_contents += line[:16000] + '\\\n'
                    line = line[16000:]
                new_contents += line + '\n'
            try:
                dotcode2 = bytes(new_contents, 'utf-8')
            except:
                dotcode2 = new_contents
            f.close()
        except Exception as e:
            show_error_dialog('Failed to read temp dot file %s' % temp_dot_filename, self, exception=e)
            return False
        os.remove(temp_dot_filename)
        xdot_ui_window.DotWidget.set_dotcode(self, dotcode2, filename)
        self.graph.cur_filename = filename
        return True

    def set_xdotcode(self, xdotcode, center=True):
        """Parse the xdot code to create a STAT Graph."""
        parser = STATXDotParser(xdotcode)
        self.graph = parser.parse()
        self.graph.visible_width = self.graph.width
        self.graph.visible_height = self.graph.height
        self.zoom_image(self.zoom_ratio, center=True)

    def update(self):
        pass

    def on_area_button_release(self, area, event):
        """Handle the clicking of a node."""
        self.drag_action.on_button_release(event)
        self.drag_action = STATNullAction(self)
        if event.button == 1 and self.is_click(event):
            x, y = int(event.x), int(event.y)
            label = self.get_label(x, y)
            if label is not None:
                self.emit('clicked', 'left', event)
            else:
                jump = self.get_jump(x, y)
                if jump is not None:
                    self.animate_to(jump.x, jump.y)
            return True
        if event.button == 2 and self.is_click(event):
            x, y = int(event.x), int(event.y)
            jump = self.get_jump(x, y)
            if jump is not None:
                self.animate_to(jump.x, jump.y)
            return True
        if event.button == 3 and self.is_click(event):
            x, y = int(event.x), int(event.y)
            label = self.get_label(x, y)
            if label is not None:
                self.emit('clicked', 'right', event)
            else:
                jump = self.get_jump(x, y)
                if jump is not None:
                    self.animate_to(jump.x, jump.y)
            return True
        if event.button == 1 or event.button == 2 or event.button == 3:
            return True
        return False

    def get_node(self, x, y):
        """Get the node that contains the specified coordinates."""
        x, y = self.window2graph(x, y)
        return self.graph.get_node(x, y)

    def get_edge(self, x, y):
        """Get the edge that contains the specified coordinates."""
        x, y = self.window2graph(x, y)
        return self.graph.get_edge(x, y)

    def get_edge_label(self, x, y):
        """Get the edge label that contains the specified coordinates."""
        x, y = self.window2graph(x, y)
        return self.graph.get_edge_label(x, y)

    def get_label(self, x, y):
        """Get the label that contains the specified coordinates."""
        x, y = self.window2graph(x, y)
        return self.graph.get_label(x, y)

    def queue_draw(self):
        """Overloaded queue_draw to set size request for scroll window."""
        if use_scroll_bars:
            width = int(math.ceil(self.graph.width * self.zoom_ratio)) + 2 * self.ZOOM_TO_FIT_MARGIN
            height = int(math.ceil(self.graph.height * self.zoom_ratio)) + 2 * self.ZOOM_TO_FIT_MARGIN
            self.set_size_request(max(width, 1), max(height, 1))
        xdot_ui_window.DotWidget.queue_draw(self)

    def is_click(self, event, click_fuzz=4, click_timeout=1.0):
        if event.x != self.pressx or event.y != self.pressy:
            self.user_zoom = True
        return xdot_ui_window.DotWidget.is_click(self, event, click_fuzz, click_timeout)

    def get_drag_action(self, event):
        """Overloaded get_drag_action for scroll window."""
        if use_scroll_bars:
            state = event.state
            if event.button in (1, 2):  # left or middle button
                if state & gtk.gdk.CONTROL_MASK:
                    return xdot_ui_actions.ZoomAction
                elif state & gtk.gdk.SHIFT_MASK:
                    return xdot_ui_actions.ZoomAreaAction
                else:
                    return STATPanAction
            return xdot_ui_actions.NullAction
        else:
            return xdot_ui_window.DotWidget.get_drag_action(self, event)

    def zoom_to_fit(self):
        """Overloaded zoom_to_fit for scroll window."""
        self.user_zoom = False
        if use_scroll_bars:
            #rect = self.dotsw.get_allocation()
            rect = self.viewport.get_allocation()
            rect.x += self.ZOOM_TO_FIT_MARGIN
            rect.y += self.ZOOM_TO_FIT_MARGIN
            rect.width -= 2 * self.ZOOM_TO_FIT_MARGIN
            rect.height -= 2 * self.ZOOM_TO_FIT_MARGIN
            zoom_ratio = min(
                float(rect.width)/float(self.graph.visible_width),
                float(rect.height)/float(self.graph.visible_height)
            )
            if self.graph.width == self.graph.visible_width and self.graph.height == self.graph.visible_height:
                self.zoom_image(zoom_ratio, center=True)
            else:
                x = zoom_ratio * (self.graph.maxx - self.graph.visible_width / 2)
                y = zoom_ratio * (self.graph.maxy - self.graph.visible_height / 2)
                self.zoom_image(zoom_ratio, False, (x, y))
            self.zoom_to_fit_on_resize = True
        else:
            return xdot_ui_window.DotWidget.zoom_to_fit(self)

    def on_zoom_fit(self, action):
        self.user_zoom = False
        self.graph.adjust_dims()
        self.zoom_to_fit()

    def on_area_scroll_event(self, area, event):
        self.user_zoom = True
        return xdot_ui_window.DotWidget.on_area_scroll_event(self, area, event)

    def on_key_press_event(self, widget, event):
        if event.keyval in (gtk.keysyms.Page_Up, gtk.keysyms.plus, gtk.keysyms.equal, gtk.keysyms.KP_Add, gtk.keysyms.Page_Down, gtk.keysyms.minus, gtk.keysyms.KP_Subtract, gtk.keysyms.Left, gtk.keysyms.Right, gtk.keysyms.Up, gtk.keysyms.Down):
            self.user_zoom = True
        if event.keyval == gtk.keysyms.r:
            global window
            window.on_toolbar_action(None, 'Reset', self.graph.on_original_graph, (widget,))
            return
        ret = xdot_ui_window.DotWidget.on_key_press_event(self, widget, event)
        if event.keyval == gtk.keysyms.r:
            self.user_zoom = False
            self.zoom_to_fit()
        return ret

    def on_zoom_in(self, action):
        self.user_zoom = True
        return xdot_ui_window.DotWidget.on_zoom_in(self, action)

    def on_zoom_out(self, action):
        self.user_zoom = True
        return xdot_ui_window.DotWidget.on_zoom_out(self, action)

    def zoom_image(self, zoom_ratio, center=False, pos=None):
        """Overloaded zoom_image for scroll window."""
        if use_scroll_bars:
            old_hadj_value = self.hadj.get_value()
            old_vadj_value = self.vadj.get_value()
            adj_zoom_ratio = zoom_ratio / self.zoom_ratio
            self.zoom_ratio = zoom_ratio
            self.zoom_to_fit_on_resize = False
            self.queue_draw()
            if center is False and pos is None:
                allocation = self.viewport.get_allocation()
                hspan = allocation[2]
                hcenter = old_hadj_value + hspan / 2 - self.ZOOM_TO_FIT_MARGIN
                vspan = allocation[3]
                vcenter = old_vadj_value + vspan / 2 - self.ZOOM_TO_FIT_MARGIN
                pos = (hcenter, vcenter)
            if center is True:
                self.x = self.graph.visible_width/2
                self.y = self.graph.visible_height/2
            elif pos is not None:
                x, y = pos
                allocation = self.viewport.get_allocation()
                hspan = allocation[2]
                vspan = allocation[3]
                hpos = x * adj_zoom_ratio - hspan / 2.0 + self.ZOOM_TO_FIT_MARGIN
                self.dotsw.get_hscrollbar().get_adjustment().set_value(hpos)
                vpos = y * adj_zoom_ratio - vspan / 2.0 + self.ZOOM_TO_FIT_MARGIN
                self.dotsw.get_vscrollbar().get_adjustment().set_value(vpos)
        else:
            return xdot_ui_window.DotWidget.zoom_image(self, zoom_ratio, center, pos)


## The window object containing the STATDotWidget
class STATDotWindow(xdot_ui_window.DotWindow):
    """The window object containing the STATDotWidget

    The STATDotWindow is derrived from xdot's DotWindow class.
    It adds STAT specific operations to manipulate the graph.
    """

    ## A notebook tab within the window, contains the STAT widget.
    class Tab(object):
        """A notebook tab within the window, contains the STAT widget."""

        def __init__(self):
            """The constructor."""
            self.widget = STATDotWidget()
            frame = gtk.Frame()
            frame.set_size_request(60, 60)
            if use_scroll_bars:
                self.widget.dotsw = gtk.ScrolledWindow()
                self.widget.dotsw.set_policy(gtk.POLICY_ALWAYS, gtk.POLICY_ALWAYS)
                self.widget.viewport = gtk.Viewport()
                self.widget.viewport.add(self.widget)
                self.widget.dotsw.add(self.widget.viewport)
                self.widget.hadj = self.widget.dotsw.get_hadjustment()
                self.widget.vadj = self.widget.dotsw.get_vadjustment()
                frame.add(self.widget.dotsw)
            else:
                frame.add(self.widget)
            self.hpaned = gtk.HPaned()
            self.hpaned.pack1(frame, True, True)
            self.history_view = gtk.TextView()
            self.history_view.set_wrap_mode(False)
            self.history_view.set_editable(False)
            self.history_view.set_cursor_visible(False)
            self.history_view.get_buffer().set_text('')
            sw = gtk.ScrolledWindow()
            sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
            sw.add(self.history_view)
            frame = gtk.Frame(label="Command History")
            frame.set_size_request(150, 60)
            frame.add(sw)
            self.hpaned.pack2(frame, False, True)
            self.label = gtk.Label('new tab')

    ui = ''
    ui += '<ui>\n'
    ui += '    <menubar name="MenuBar">\n'
    ui += '        <menu action="FileMenu">\n'
    ui += '            <menuitem action="Open"/>\n'
    ui += '            <menuitem action="SaveAs"/>\n'
    ui += '            <separator/>\n'
    ui += '            <menuitem action="Prefs"/>\n'
    ui += '            <menuitem action="SavePrefs"/>\n'
    ui += '            <menuitem action="LoadPrefs"/>\n'
    ui += '            <separator/>\n'
    ui += '            <menuitem action="SearchPath"/>\n'
    ui += '            <menuitem action="CacheSources"/>\n'
    ui += '            <separator/>\n'
    ui += '            <menuitem action="NewTab"/>\n'
    ui += '            <menuitem action="ResetLayout"/>\n'
    ui += '            <menuitem action="CloseTab"/>\n'
    ui += '            <separator/>\n'
    ui += '            <menuitem action="Quit"/>\n'
    ui += '        </menu>\n'
    ui += '        <menu action="EditMenu">\n'
    ui += '            <menuitem action="Undo"/>\n'
    ui += '            <menuitem action="Redo"/>\n'
    ui += '        </menu>\n'
    ui += '        <menu action="ViewMenu">\n'
    ui += '            <menuitem action="ZoomIn"/>\n'
    ui += '            <menuitem action="ZoomOut"/>\n'
    ui += '            <menuitem action="ZoomFit"/>\n'
    ui += '            <menuitem action="Zoom100"/>\n'
    ui += '        </menu>\n'
    ui += '        <menu action="HelpMenu">\n'
    ui += '            <menuitem action="UserGuide"/>\n'
    ui += '            <menuitem action="About"/>\n'
    ui += '        </menu>\n'
    ui += '    </menubar>\n'
    ui += '    <toolbar name="ToolBar">\n'
    ui += '        <toolitem action="Open"/>\n'
    ui += '        <toolitem action="SaveAs"/>\n'
    ui += '        <separator/>\n'
    ui += '        <toolitem action="Undo"/>\n'
    ui += '        <toolitem action="Redo"/>\n'
    ui += '        <toolitem action="OriginalGraph"/>\n'
    ui += '        <toolitem action="ResetLayout"/>\n'
#    ui += '        <toolitem action="HideMPI"/>\n'
    ui += '        <toolitem action="HideModel"/>\n'
#    ui += '        <toolitem action="HideText"/>\n'
    ui += '        <toolitem action="Join"/>\n'
    ui += '        <toolitem action="TraverseGraph"/>\n'
    ui += '        <toolitem action="ShortestPath"/>\n'
    ui += '        <toolitem action="LongestPath"/>\n'
    ui += '        <toolitem action="LeastTasks"/>\n'
    ui += '        <toolitem action="MostTasks"/>\n'
    if HAVE_TOMOD:
        ui += '        <toolitem action="TOTraverseLeastProgress"/>\n'
        ui += '        <toolitem action="TOTraverseMostProgress"/>\n'
    ui += '        <toolitem action="Search"/>\n'
    ui += '        <toolitem action="IdentifyEqClasses"/>\n'
    ui += '    </toolbar>\n'
    ui += '</ui>\n'

    def __init__(self, input_actions=None):
        """The constructor."""
        gtk.Window.__init__(self)
        self.set_title('STATview')
        self.set_default_size(1048, 600)
        uimanager = gtk.UIManager()
        accelgroup = uimanager.get_accel_group()
        self.add_accel_group(accelgroup)
        actiongroup = gtk.ActionGroup('Actions')
        actions = []
        actions.append(('FileMenu', None, '_File'))
        actions.append(('EditMenu', None, '_Edit'))
        actions.append(('ViewMenu', None, '_View'))
        actions.append(('HelpMenu', None, '_Help'))
        actions.append(('Open', gtk.STOCK_OPEN, '_Open', '<control>O', 'Open a file', self.on_open))
        actions.append(('SaveAs', gtk.STOCK_SAVE_AS, '_SaveAs', '<control>S', 'Save current view as a file', self.on_save_as))
        actions.append(('Prefs', gtk.STOCK_PROPERTIES, 'Preferences', '<control>P', 'Load saved preference settings', self.on_prefs))
        actions.append(('SavePrefs', gtk.STOCK_SAVE_AS, 'Save Pr_eferences', '<control>E', 'Load saved preference settings', self.on_save_prefs))
        actions.append(('LoadPrefs', gtk.STOCK_OPEN, '_Load Preferences', '<control>L', 'Save current preference settings', self.on_load_prefs))
        actions.append(('SearchPath', gtk.STOCK_ADD, '_Add Search Paths', '<control>A', 'Add search paths for application source and header files', self.on_modify_search_paths))
        actions.append(('CacheSources', gtk.STOCK_ADD, '_Cache Source Files', '<control>U', 'Cache all source files', self.on_cache_sources))
        actions.append(('NewTab', gtk.STOCK_NEW, '_New Tab', '<control>T', 'Create new tab', lambda x: self.menu_item_response(gtk.STOCK_NEW, 'New Tab')))
        actions.append(('CloseTab', gtk.STOCK_CLOSE, 'Close Tab', '<control>W', 'Close current tab', lambda x: self.menu_item_response(None, 'Close Tab')))
        actions.append(('Quit', gtk.STOCK_QUIT, '_Quit', '<control>Q', 'Quit', self.on_destroy))
        actions.append(('Undo', gtk.STOCK_UNDO, '_Undo', '<control>Z', 'Undo operation', lambda a: self.on_toolbar_action(a, None, self.get_current_graph().undo, ())))
        actions.append(('Redo', gtk.STOCK_REDO, '_Redo', '<control>R', 'Redo operation', lambda a: self.on_toolbar_action(a, None, self.get_current_graph().redo, (self.get_current_widget(), ))))
        actions.append(('OriginalGraph', gtk.STOCK_HOME, 'Reset', None, 'Revert to original graph', lambda a: self.on_toolbar_action(a, 'Original Graph', self.get_current_graph().on_original_graph, (self.get_current_widget(), ))))
        actions.append(('ResetLayout', gtk.STOCK_REFRESH, 'Layout', None, 'Reset the layout of the current graph and open in a new tab', lambda a: self.on_reset_layout()))
        actions.append(('HideModel', gtk.STOCK_CUT, 'Cut', None, 'Hide the details of various programming models', self.on_hide_model))
#        actions.append(('HideMPI', gtk.STOCK_CUT, 'MPI', None, 'Hide the MPI implementation', lambda a: self.on_toolbar_action(a, 'Hide MPI', self.get_current_graph().hide_mpi, ())))
        actions.append(('Join', gtk.STOCK_GOTO_TOP, 'Join', None, 'Join consecutive nodes of the same equivalence class into a single node and render in a new tab', self.on_join_eq_classes))
        actions.append(('TraverseGraph', gtk.STOCK_GO_DOWN, 'Eq C', None, 'Traverse the graphs equivalence classes', self.on_traverse_graph))
        if HAVE_TOMOD:
            actions.append(('TOTraverseMostProgress', gtk.STOCK_MEDIA_NEXT, 'TO', None, 'Traverse the graph based on the most progressed temporal ordering', self.on_to_traverse_most_progress))
            actions.append(('TOTraverseLeastProgress', gtk.STOCK_MEDIA_PREVIOUS, 'TO', None, 'Traverse the graph based on the least progressed temporal ordering', self.on_to_traverse_least_progress))
        actions.append(('ShortestPath', gtk.STOCK_GOTO_TOP, 'Path', None, 'Traverse the [next] shortest path', self.on_shortest_path))
        actions.append(('LongestPath', gtk.STOCK_GOTO_BOTTOM, 'Path', None, 'Traverse the [next] longest path', self.on_longest_path))
        actions.append(('Search', gtk.STOCK_FIND, 'Search', None, 'Search for callpaths by text, tasks, or hosts', self.on_search))
#        actions.append(('HideText', gtk.STOCK_CUT, 'Text', None, 'Cut the call graph based on a regular expression', self.on_hide_text))
        actions.append(('LeastTasks', gtk.STOCK_GOTO_FIRST, 'Tasks', None, 'Traverse the path with the [next] least tasks visited', self.on_least_tasks))
        actions.append(('MostTasks', gtk.STOCK_GOTO_LAST, 'Tasks', None, 'Traverse the path with the [next] most tasks visited', self.on_most_tasks))
        actions.append(('IdentifyEqClasses', gtk.STOCK_SELECT_COLOR, 'Eq C', None, 'Identify the equivalence classes of the current graph', self.on_identify_num_eq_classes))
        actions.append(('ZoomIn', gtk.STOCK_ZOOM_IN, None, None, None, self.on_zoom_in))
        actions.append(('ZoomOut', gtk.STOCK_ZOOM_OUT, None, None, None, self.on_zoom_out))
        actions.append(('ZoomFit', gtk.STOCK_ZOOM_FIT, None, None, None, self.on_zoom_fit))
        actions.append(('Zoom100', gtk.STOCK_ZOOM_100, None, None, None, self.on_zoom_100))
        actions.append(('UserGuide', gtk.STOCK_HELP, None, None, None, self.on_user_guide))
        actions.append(('About', gtk.STOCK_ABOUT, None, None, None, self.on_about))
        if input_actions is not None:
            for action in input_actions:
                action_text = action[0]
                for a in actions:
                    if a[0] == action_text:
                        actions.remove(a)
                        break
                actions.append(action)
        actiongroup.add_actions(actions)
        uimanager.insert_action_group(actiongroup, 0)
        uimanager.add_ui_from_string(self.ui)
        toolbar = uimanager.get_widget('/ToolBar')
        menubar = uimanager.get_widget('/MenuBar')
        self.vbox = gtk.VBox()
        self.add(self.vbox)
        self.vbox.pack_start(menubar, False, True, 0)
        hbox = gtk.HBox()
        hbox.pack_start(toolbar, True, True, 0)
        image = gtk.Image()
        try:
            if hasattr(gtk.gdk, "pixbuf_new_from_file"):
                pixbuf = gtk.gdk.pixbuf_new_from_file(STAT_LOGO)
            elif hasattr(gtk, "Image"):
                image = gtk.Image()
                image.set_from_file(STAT_LOGO)
                pixbuf = image.get_pixbuf()
            image.set_from_pixbuf(pixbuf)
            hbox.pack_start(image, False, True, 0)
        except gobject.GError:
            pass
        self.vbox.pack_start(hbox, False, True, 0)
        self.tabs = []
        self.notebook = gtk.Notebook()
        self.create_new_tab()
        self.notebook.set_tab_pos(gtk.POS_TOP)
        self.notebook.set_scrollable(True)
        try:
            self.hbox.pack_start(self.notebook, True, True, 0)
        except:
            self.hbox = gtk.HBox()
            self.hbox.pack_start(self.notebook, True, True, 0)
        self.vbox.pack_start(self.hbox, True, True, 0)
        if HAVE_TOMOD is True:
            for path in search_paths['include']:
                tomod.add_include_path(path)
        self.search_types = []
        help_string = """Search for a specified task, range
of tasks, or comma-separated list
of tasks.  Example task lists:
0
0-10
0-10,12,15-20"""
        self.search_types.append(('tasks', self.search_tasks, help_string))
        help_string = """Search for callpaths containing the
specified text, which may be
entered as a regular expression"""
        self.search_types.append(('text', self.search_for_text, 'Search for callpaths containing the\nspecified text, which may be\nentered as a regular expression'))
        if not hasattr(self, "types"):
            self.types = {}
        self.types["truncate"] = ["front", "rear"]
        if not hasattr(self, "options"):
            self.options = {}
        self.options["truncate"] = "front"
        self.options["max node name"] = 64
        self.options["maintain user zoom"] = True
        self.combo_boxes = {}
        self.spinners = {}
        self.entries = {}
        self.show_all()

    def update_option(self, w, label, parent_window, option):
        """Generate text entry dialog to update the specified option."""
        dialog = gtk.Dialog('Update %s' % option, parent_window)
        entry = gtk.Entry()
        entry.set_max_length(1024)
        entry.set_text(self.options[option])
        entry.connect("activate", lambda w: self.on_update_option(w, entry, label, dialog, option))
        dialog.vbox.pack_start(entry, True, True, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: dialog.destroy())
        hbox.pack_start(button, False, False, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", lambda w: self.on_update_option(w, entry, label, dialog, option))
        hbox.pack_start(button, False, False, 0)
        dialog.vbox.pack_start(hbox, False, False, 0)
        dialog.show_all()
        dialog.run()

    def on_update_option(self, w, entry, label, dialog, option):
        """Callback to update the specified option."""
        self.options[option] = entry.get_text()
        entry.set_text('')
        label.set_text('%s: %s' % (option, self.options[option]))
        dialog.destroy()

    def pack_entry_and_button(self, entry_text, function, frame, dialog, button_text, box, fill=False, center=False, pad=0):
        """Generates a text entry and activation button."""
        hbox = gtk.HBox()
        entry = gtk.Entry()
        entry.set_max_length(1024)
        entry.set_text(entry_text)
        entry.connect("activate", lambda w: function(*(w, frame, dialog, entry)))
        hbox.pack_start(entry, True, True, 0)
        button = gtk.Button(button_text)
        button.connect("clicked", lambda w: function(*(w, frame, dialog, entry)))
        hbox.pack_start(button, False, False, 0)
        box.pack_start(hbox, fill, center, pad)
        return entry

    def pack_radio_buttons(self, box, option):
        """Pack a set of radio buttons for a specified option."""
        for option_type in self.types[option]:
            if option_type == self.types[option][0]:
                radio_button = gtk_wrap_new_with_label_from_widget(None, option_type)
            else:
                radio_button = gtk_wrap_new_with_label_from_widget(radio_button, option_type)
            if option_type == self.options[option]:
                radio_button.set_active(True)
                self.toggle_radio_button(None, (option, option_type))
            radio_button.connect('toggled', self.toggle_radio_button, (option, option_type))
            box.pack_start(radio_button, False, False, 0)

    def toggle_radio_button(self, action, data):
        """Callback to toggle on/off a radio button."""
        option, option_type = data
        self.options[option] = option_type

    def pack_spinbutton(self, box, option):
        """Pack a spin button into the spcified box for the specified option."""
        hbox = gtk.HBox()
        label = gtk.Label(option)
        hbox.pack_start(label, False, False, 0)
        adj = gtk.Adjustment(1.0, 0.0, 1000000.0, 1.0, 100.0, 0.0)
        spinner = gtk.SpinButton(adjustment=adj, climb_rate=0, digits=0)
        spinner.set_value(self.options[option])
        hbox.pack_start(spinner, False, False, 0)
        box.pack_start(hbox, False, False, 0)
        self.spinners[option] = spinner

    def pack_combo_box(self, box, option):
        """Pack a combo box into the spcified box for the specified option."""
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Specify %s:" % option), False, False, 0)
        combo_box = gtk.combo_box_new_text()
        for type in self.types[option]:
            combo_box.append_text(type)
        combo_box.set_active(self.types[option].index(self.options[option]))
        hbox.pack_start(combo_box, False, False, 10)
        self.combo_boxes[option] = combo_box
        box.pack_start(hbox, False, False, 0)

    def pack_check_button(self, box, option, center=False, expand=False, pad=0):
        """Pack a check button into the specified box for the specified option."""
        check_button = gtk.CheckButton(option)
        if self.options[option] is True:
            check_button.set_active(True)
        else:
            check_button.set_active(False)
        check_button.connect('toggled', lambda w: self.on_toggle_check_button(w, option))
        box.pack_start(check_button, center, expand, pad)

    def on_toggle_check_button(self, widget, option):
        """Callback to toggle on/off a check button."""
        if self.options[option] is True:
            self.options[option] = False
        else:
            self.options[option] = True

    def pack_string_option(self, box, option, parent_window):
        """Pack a button into the specified box that generates a text entry dialog for the specified option."""
        hbox = gtk.HBox()
        label = gtk.Label('%s: %s' % (option, self.options[option]))
        hbox.pack_start(label, False, False, 0)
        box.pack_start(hbox, False, False, 10)
        button = gtk.Button('Modify %s' % option)
        button.connect("clicked", lambda w: self.update_option(w, label, parent_window, option))
        box.pack_start(button, False, False, 0)

    def update_prefs_cb(self, w, dialog):
        dialog.hide()
        try:
            self.options['max node name'] = int(self.spinners['max node name'].get_value())
            self.options['truncate'] = self.types["truncate"][self.combo_boxes['truncate'].get_active()]
        except:
            pass

    def on_prefs(self, action):
        dialog = gtk.Dialog('Preferences', self)
        options = ["truncate", "max node name", "maintain user zoom"]
        for option in options:
            if type(self.options[option]) == int:
                self.pack_spinbutton(dialog.vbox, option)
            elif type(self.options[option]) == bool:
                self.pack_check_button(dialog.vbox, option)
            elif type(self.options[option]) == str:
                if option in self.types:
                    self.pack_combo_box(dialog.vbox, option)
                else:
                    self.pack_string_option(dialog.vbox, option, dialog)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: dialog.destroy())
        hbox.pack_start(button, False, True, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", lambda w: self.update_prefs_cb(w, dialog))
        hbox.pack_start(button, False, True, 0)
        dialog.vbox.pack_start(hbox, False, False, 0)
        dialog.show_all()
        dialog.run()

    def on_load_prefs(self, action):
        """Load user-saved preferences from a file."""
        myaction = gtk.FILE_CHOOSER_ACTION_OPEN
        chooser = gtk.FileChooserDialog(title="Load Preferences", action=myaction, buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE_AS, gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        file_filter = gtk.FileFilter()
        file_filter.set_name('STAT Prefs File')
        file_filter.add_pattern("*.SPF")
        chooser.add_filter(file_filter)
        file_filter = gtk.FileFilter()
        file_filter.set_name('All files')
        file_filter.add_pattern("*")
        chooser.add_filter(file_filter)
        chooser.set_current_folder('%s/.STAT' % os.environ['HOME'])
        if chooser.run() == gtk.RESPONSE_OK:
            filename = chooser.get_filename()
            try:
                shelf = shelve.open(filename)
                for key in self.options.keys():
                    self.options[key] = shelf[key]
                shelf.close()
            except IOError as e:
                show_error_dialog('%s\nFailed to load preferences file %s\n' % (repr(e), filename), self)
            except Exception as e:
                show_error_dialog('%s\nFailed to process preferences file %s\n' % (repr(e), filename), self)
        chooser.destroy()

    def on_save_prefs(self, action):
        """Save user preferences to a file."""
        chooser = gtk.FileChooserDialog(title="Save Preferences", action=gtk.FILE_CHOOSER_ACTION_SAVE, buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE_AS, gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        chooser.set_do_overwrite_confirmation(True)
        file_filter = gtk.FileFilter()
        file_filter.set_name('STAT Prefs File')
        file_filter.add_pattern("*.SPF")
        chooser.add_filter(file_filter)
        file_filter = gtk.FileFilter()
        file_filter.set_name('All files')
        file_filter.add_pattern("*")
        chooser.add_filter(file_filter)
        chooser.set_current_folder('%s/.STAT' % os.environ['HOME'])
        if chooser.run() == gtk.RESPONSE_OK:
            file_filter = chooser.get_filter()
            ext = ''
            if file_filter.get_name() == 'STAT Prefs File':
                ext = '.SPF'
            filename = chooser.get_filename()
            if filename[-4:] != ".SPF":
                filename = filename + ext
            saved_options = {}
            if os.path.exists(filename):
                # do not delete options that aren't currently set
                try:
                    shelf = shelve.open(filename)
                    for key in self.options.keys():
                        if not key in self.options:
                            saved_options[key] = shelf[key]
                    shelf.close()
                except:
                    pass
            try:
                shelf = shelve.open(filename)
                for key in saved_options.keys():
                    shelf[key] = self.options[key]
                for key in self.options.keys():
                    shelf[key] = self.options[key]
                shelf.close()
            except IOError as e:
                show_error_dialog('%s\nFailed to save preferences file %s\n' % (repr(e), filename), self)
            except Exception as e:
                show_error_dialog('%s\nFailed to save preferences file %s\n' % (repr(e), filename), self)
        chooser.destroy()

    def on_destroy(self, action):
        """Destroy the window."""
        gtk.main_quit()

    def create_new_tab(self, page=-1):
        """Create a new notebook tab."""
        new_tab = STATDotWindow.Tab()
        tab_label_box = gtk.EventBox()
        tab_label_box.add(new_tab.label)
        tab_label_box.connect('event', self.on_tab_clicked, new_tab.widget)
        new_tab.label.show()
        new_tab.widget.connect('clicked', self.on_node_clicked)
        if page == -1:
            page = len(self.tabs)
        self.notebook.insert_page(new_tab.hpaned, tab_label_box, page)
        self.tabs.insert(page, new_tab)
        self.set_focus(new_tab.widget)
        self.show_all()

    def on_tab_clicked(self, widget, event, child):
        """Switch the focus to the clicked tab."""
        if event.type == gtk.gdk.BUTTON_PRESS:
            n = -1
            for tab in self.tabs:
                n += 1
                if child == tab.widget:
                    break
            self.notebook.set_current_page(n)
            if hasattr(event.button, 'button'):
                button = event.button.button
            else:
                button = event.button
            if button == 3:
                options = ['New Tab', 'Close Tab']
                menu = gtk.Menu()
                for option in options:
                    menu_item = gtk.MenuItem(option)
                    menu.append(menu_item)
                    menu_item.connect('activate', self.menu_item_response, option)
                    menu_item.show()
                try:
                    menu.popup(None, None, None, None, button, event.time)
                except:
                    menu.popup(None, None, None, event.button, event.time)

    def get_current_widget(self):
        """Get the widget of the current tab."""
        return self.tabs[self.notebook.get_current_page()].widget

    def get_current_graph(self):
        """Get the graph of the current tab."""
        return self.tabs[self.notebook.get_current_page()].widget.graph

    def menu_item_response(self, widget, option):
        """Handle tab menu responses."""
        if option == "New Tab":
            page = self.notebook.get_current_page()
            self.create_new_tab(page + 1)
            self.notebook.set_current_page(page + 1)
        elif option == "Close Tab":
            if len(self.tabs) == 1:
                self.create_new_tab()
                page = 0
            else:
                page = self.notebook.get_current_page()
            self.notebook.remove_page(page)
            self.tabs.remove(self.tabs[page])

    def on_user_guide(self, action):
        """Display the user guide."""
        user_guide_path = os.path.join(os.path.dirname(__file__), '../../../doc/stat_userguide.pdf')
        if not os.path.exists(user_guide_path):
            show_error_dialog('Failed to find STAT user guide %s' % user_guide_path, self)
            return False
        pdfviewer = which('evince')
        if pdfviewer is None:
            pdfviewer = which('xpdf')
            if pdfviewer is None:
                pdfviewer = which('acroread')
                if pdfviewer is None:
                    show_error_dialog('Failed to find PDF viewer', self)
                return False
        if os.fork() == 0:
            subprocess.call([pdfviewer, user_guide_path])
            sys.exit(0)

    def on_about(self, action):
        """Display info about STAT"""
        about_dialog = gtk.AboutDialog()
        about_dialog.set_name('STATview')
        about_dialog.set_authors(__author__)
        about_dialog.set_copyright(__copyright__)
        about_dialog.set_license(__license__)
        about_dialog.set_wrap_license(80)
        try:
            if hasattr(gtk.gdk, "pixbuf_new_from_file"):
                pixbuf = gtk.gdk.pixbuf_new_from_file(STAT_LOGO)
            elif hasattr(gtk, "Image"):
                image = gtk.Image()
                image.set_from_file(STAT_LOGO)
                pixbuf = image.get_pixbuf()
            about_dialog.set_logo(pixbuf)
        except gobject.GError:
            pass
        about_dialog.set_website('https://github.com/lee218llnl/STAT')
        about_dialog.show_all()
        about_dialog.run()
        about_dialog.destroy()

    def on_zoom_in(self, action):
        """Zoom in on the current tab's graph."""
        self.get_current_widget().on_zoom_in(action)

    def on_zoom_out(self, action):
        """Zoom out of the current tab's graph."""
        self.get_current_widget().on_zoom_out(action)

    def on_zoom_fit(self, action):
        """Zoom to best fit of the current tab's graph."""
        self.get_current_widget().on_zoom_fit(action)

    def on_zoom_100(self, action):
        """Zoom to full size of the current tab's graph."""
        self.get_current_widget().on_zoom_100(action)

    def set_dotcode(self, dotcode, filename='<stdin>', page=-1):
        """Set the dotcode of the specified tab's widget."""
        try:
            if self.tabs[page].widget.set_dotcode(dotcode, filename, self.options["truncate"], self.options["max node name"]):
                self.tabs[page].label.set_text(os.path.basename(filename))
                self.tabs[page].widget.zoom_to_fit()
        except Exception as e:
            show_error_dialog('%s\nFailed to open file:\n\n%s\n\nPlease be sure the file exists and is a valid STAT outputted dot file.' % (repr(e), filename), exception=e)

    def open_file(self, filename):
        """Open a dot file and set the dotcode for the current tab."""
        try:
            with open(filename, 'r') as f:
                stat_wait_dialog.show_wait_dialog_and_run(self.set_dotcode, (f.read(), filename, self.notebook.get_current_page()), ['Opening DOT File'], self)
        except IOError as e:
            show_error_dialog('%s\nFailed to open file:\n\n%s\n\nPlease be sure the file exists and is a valid STAT outputted dot file.' % (repr(e), filename), exception=e)
        except Exception as e:
            show_error_dialog('%s\nFailed to process file:\n\n%s\n\nPlease be sure the file exists and is a valid STAT outputted dot file.' % (repr(e), filename), exception=e)
        self.show_all()
        self.update_history()

    def on_open(self, action):
        """Callback to generate an open file dialog."""
        chooser = gtk.FileChooserDialog(title="Open dot File",
                                        action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_OPEN,
                                                 gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        chooser.set_current_folder(os.getcwd())
        chooser.set_select_multiple(True)
        file_filter = gtk.FileFilter()
        file_filter.set_name("Graphviz dot files")
        file_filter.add_pattern("*.dot")
        chooser.add_filter(file_filter)
        file_filter = gtk.FileFilter()
        file_filter.set_name("All files")
        file_filter.add_pattern("*")
        chooser.add_filter(file_filter)
        if chooser.run() == gtk.RESPONSE_OK:
            filenames = chooser.get_filenames()
            chooser.destroy()
            for filename in filenames:
                if filename != filenames[0]:
                    self.create_new_tab(self.notebook.get_current_page() + 1)
                self.notebook.set_current_page(self.notebook.get_current_page() + 1)
                self.open_file(filename)
        else:
            chooser.destroy()

    def update_history(self):
        """Update the action history of the current tab."""
        history_string = ''
        for command in self.get_current_graph().action_history:
            history_string += command + '\n'
        self.tabs[self.notebook.get_current_page()].history_view.get_buffer().set_text(history_string)
        self.show_all()

    def save_file(self, filename):
        """Save the current graph to a dot or image file."""
        ret = self.get_current_graph().save_file(filename)
        return ret

    def save_type_toggle_cb(self, widget, data=None):
        """Callback to toggle the file format."""
        self.ext_type = data

    def file_extension_ok_cb(self, widget, data=None):
        """Callback to confirm the file format."""
        self.ext_dialog.destroy()
        self.chooser.destroy()
        self.on_save_as(None, data[0], os.path.basename('%s.%s' % (data[1], self.ext_type)))

    def on_save_as(self, action, directory='', filename=''):
        """Callback to generate save dialog."""
        if self.get_current_graph().cur_filename == '':
            return False
        file_extensions = ['dot graph file', 'gif image file', 'pdf document file', 'png image file', 'jpg image file', 'fig figure file', 'ps image file', 'perf-folded trace file']
        short_file_extensions = []
        for file_extension in file_extensions:
            short_file_extensions.append('.' + file_extension.split()[0])
        self.chooser = gtk.FileChooserDialog(title="Save File",
                                             action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                             buttons=(gtk.STOCK_CANCEL,
                                                      gtk.RESPONSE_CANCEL,
                                                      gtk.STOCK_SAVE_AS,
                                                      gtk.RESPONSE_OK))
        self.chooser.set_default_response(gtk.RESPONSE_OK)
        self.chooser.set_current_folder(os.getcwd())
        self.chooser.set_do_overwrite_confirmation(True)
        for extension in file_extensions:
            file_filter = gtk.FileFilter()
            file_filter.set_name(extension)
            file_filter.add_pattern("*.%s" % extension.split()[0])
            self.chooser.add_filter(file_filter)
        file_filter = gtk.FileFilter()
        file_filter.set_name("All files")
        file_filter.add_pattern("*")
        self.chooser.add_filter(file_filter)
        if directory != '':
            self.chooser.set_current_folder(directory)
        if filename != '':
            self.chooser.set_current_name(filename)
        if self.chooser.run() == gtk.RESPONSE_OK:
            file_filter = self.chooser.get_filter()
            file_type = file_filter.get_name()
            folder = self.chooser.get_current_folder()
            filename = self.chooser.get_filename()
            if not os.path.splitext(filename)[1] in short_file_extensions and file_type == 'All files':
                self.ext_type = 'dot'
                self.ext_dialog = gtk.Dialog("Choose Extension", self)
                self.ext_dialog.set_default_size(300, 80)
                ext_frame = gtk.Frame(label="File type explanation")
                explanation = 'Saving as an AT&T dot format graph file allows the file to be reopened by this application.  Saving as an image or document file generates a file that can be easily viewed by other users and on other systems, but can not be manipulated by this application.'
                ext_explanation = gtk.Label(explanation)
                ext_explanation.set_line_wrap(True)
                ext_frame.add(ext_explanation)
                self.ext_dialog.vbox.pack_start(ext_frame, False, False, 0)
                ext_vbox2 = gtk.VBox(False, 0)
                for extension in file_extensions:
                    if extension == file_extensions[0]:
                        radio_button = gtk_wrap_new_with_label_from_widget(None, extension)
                        radio_button.set_active(True)
                    else:
                        radio_button = gtk_wrap_new_with_label_from_widget(radio_button, extension)
                    radio_button.connect("toggled", self.save_type_toggle_cb, extension.split()[0])
                    ext_vbox2.pack_start(radio_button, False, True, 5)

                self.ext_dialog.vbox.pack_start(ext_vbox2, False, False, 0)
                ext_hbox2 = gtk.HBox(False, 0)
                ok_button = gtk.Button(stock=gtk.STOCK_OK)
                ok_button.connect("clicked", self.file_extension_ok_cb, [folder, filename])
                ext_hbox2.pack_end(ok_button, False, False, 0)
                self.ext_dialog.vbox.pack_end(ext_hbox2, False, False, 0)
                self.ext_dialog.show_all()
            elif not os.path.splitext(filename)[1] in short_file_extensions and file_type != 'All files':
                ret = self.save_file(filename + '.' + file_type.split()[0])
                if ret is False:
                    show_error_dialog('failed to save file "%s"' % filename + '.' + file_type.split()[0], self)
                self.chooser.destroy()
            else:
                ret = self.save_file(filename)
                if ret is False:
                    show_error_dialog('failed to save file "%s"' % filename + '.' + file_type.split()[0], self)
                self.chooser.destroy()
        else:
            self.chooser.destroy()

    def on_reset_layout(self, invert=False):
        temp_dot_filename = 'redraw.dot'
        try:
            temp_dot_file = open(temp_dot_filename, 'w')
        except:
            home_dir = os.environ.get("HOME")
            temp_dot_filename = '%s/redraw.dot' % home_dir
            try:
                temp_dot_file = open(temp_dot_filename, 'w')
            except Exception as e:
                show_error_dialog('Failed to open temp dot file %s for writing' % temp_dot_filename, exception=e)
                return False
        temp_dot_file.close()
        self.get_current_graph().save_dot(temp_dot_filename, invert=invert)
        page = self.notebook.get_current_page()
        self.create_new_tab(page + 1)
        self.notebook.set_current_page(page + 1)
        self.open_file(temp_dot_filename)
        os.remove(temp_dot_filename)
        return True

    def on_translate(self):
        if hasattr(self, "my_dialog"):
            self.my_dialog.destroy()
        temp_dot_filename = 'translated.dot'
        try:
            temp_dot_file = open(temp_dot_filename, 'w')
        except:
            home_dir = os.environ.get("HOME")
            temp_dot_filename = '%s/tranlsated.dot' % home_dir
            try:
                temp_dot_file = open(temp_dot_filename, 'w')
            except Exception as e:
                show_error_dialog('Failed to open temp dot file %s for writing' % temp_dot_filename, exception=e)
                return False
        temp_dot_file.close()
        self.get_current_graph().save_dot(temp_dot_filename, True, True, True)
        page = self.notebook.get_current_page()
        self.create_new_tab(page + 1)
        self.notebook.set_current_page(page + 1)
        self.open_file(temp_dot_filename)
        os.remove(temp_dot_filename)
        return True

    def on_cache_sources(self, action):
        """Callback to cache all source files found in this .dot file."""
        current_graph = self.get_current_graph()
        file_dir, file_name = dot_file_path_split(current_graph)
        if file_dir is None or file_dir == '':
            show_error_dialog('unable to determine directory of .dot file, STAT will not cache sources')
            return False
        source_files = set()
        for node in current_graph.nodes:
            source_file_path = node.attrs['source']
            if source_file_path != '' and os.path.exists(source_file_path):
                source_files.add(source_file_path)
        cache_directory_path = os.path.abspath(os.path.join(file_dir, cache_directory))
        if not os.path.exists(cache_directory_path):
            try:
                os.makedirs(cache_directory_path)
            except Exception as e:
                show_error_dialog('Failed to create cache directory %s' % cache_directory_path, exception=e)
                return False
        for source_file in source_files:
            cache_file_path = cache_directory_path + '/' + source_file
            cache_file_dir, cache_file_name = os.path.split(cache_file_path)
            if not os.path.exists(cache_file_path):
                try:
                    if not os.path.exists(cache_file_dir):
                        os.makedirs(cache_file_dir)
                    shutil.copyfile(source_file, cache_file_path)
                except Exception as e:
                    show_error_dialog('Failed to copy %s to %s' % (source_file, cache_file_path), exception=e)
                    return False


    def on_modify_search_paths(self, action):
        """Callback to generate dialog to modify search paths."""
        search_dialog = gtk.Dialog('add file search path', self)
        frame = gtk.Frame(label="Current Search Paths")
        hpaned = gtk.HPaned()
        path_views = {}
        for path_type2 in ["source", "include"]:
            frame2 = gtk.Frame(label=path_type2)
            path_view = gtk.TextView()
            path_view.set_size_request(400, 200)
            path_view.set_wrap_mode(False)
            path_view.set_editable(False)
            path_view.set_cursor_visible(False)
            paths_string = ''
            for path in search_paths[path_type2]:
                paths_string += path + '\n'
            path_view.get_buffer().set_text(paths_string)
            sw = gtk.ScrolledWindow()
            sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
            sw.add(path_view)
            path_views[path_type2] = path_view
            frame2.add(sw)
            if path_type2 == "source":
                hpaned.pack1(frame2, True, True)
            else:
                hpaned.pack2(frame2, True, True)
        frame.add(hpaned)
        search_dialog.vbox.pack_start(frame, True, True, 0)
        separator = gtk.HSeparator()
        search_dialog.vbox.pack_start(separator, False, True, 5)
        vbox = gtk.VBox(False, 0)
        frame = gtk.Frame(label="Add Search Paths")
        entry = gtk.Entry()
        entry.set_max_length(65536)
        check_buttons = {}
        source_check_button = gtk.CheckButton("Add to Source search path")
        source_check_button.set_active(True)
        source_check_button.connect('toggled', lambda x: x)
        vbox.pack_start(source_check_button, False, False, 0)
        check_buttons["source"] = source_check_button
        include_check_button = gtk.CheckButton("Add to Include search path")
        include_check_button.set_active(False)
        include_check_button.connect('toggled', lambda x: x)
        vbox.pack_start(include_check_button, False, False, 0)
        check_buttons["include"] = include_check_button
        recurse_check_button = gtk.CheckButton("Recursively add sub directories")
        recurse_check_button.set_active(False)
        recurse_check_button.connect('toggled', lambda x: x)
        vbox.pack_start(recurse_check_button, False, False, 0)
        check_buttons["recurse"] = recurse_check_button
        entry.connect("activate", self.on_search_path_enter_cb, entry, path_views, check_buttons, search_dialog)
        vbox.pack_start(entry, False, False, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(" Add ")
        button.connect("clicked", self.on_search_path_enter_cb, entry, path_views, check_buttons, search_dialog)
        hbox.pack_start(button, False, False, 0)
        vbox.pack_end(hbox, False, False, 0)
        frame.add(vbox)
        search_dialog.vbox.pack_start(frame, False, False, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: search_dialog.destroy())
        hbox.pack_end(button, False, False, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", lambda w: self.on_search_path_enter_cb(w, entry, path_views, check_buttons, search_dialog, True))
        hbox.pack_end(button, False, False, 0)
        search_dialog.vbox.pack_end(hbox, False, False, 0)
        search_dialog.set_focus(entry)
        search_dialog.show_all()

    def on_search_path_enter_cb(self, widget, entry, path_views, check_buttons, search_dialog, destroy=False):
        """Callback to handle activation of search path text entry."""
        path = entry.get_text()
        entry.set_text('')
        paths = []
        if path == '' and destroy is True:
            search_dialog.hide()
            return
        elif path == '' and destroy is False:
            return
        if not os.path.exists(path):
            show_error_dialog('directory "%s" does not exist' % path, self)
            return
        if check_buttons["recurse"].get_active():
            for p, d, f in os.walk(path):
                paths.append(p)
        else:
            paths.append(path)
        for path in paths:
            for path_type2 in ["source", "include"]:
                if not check_buttons[path_type2].get_active():
                    continue
                if path_type2 == 'include':
                    tomod.add_include_path(path)
                search_paths[path_type2].append(path)
                paths_string = ''
                for path in search_paths[path_type2]:
                    paths_string += path + '\n'
                path_views[path_type2].get_buffer().set_text(paths_string)
        if destroy is True:
            search_dialog.hide()
            return
        search_dialog.show_all()

    def on_toolbar_action(self, action, name, function, args):
        """Callback to handle action initiated from the toolbar."""
        if self.get_current_graph().cur_filename == '':
            return False
        if name is not None:
            self.get_current_graph().set_undo_list()
        ret = function(*args)
        if ret is True:
            if name is not None:
                self.get_current_graph().action_history.append(name)
            self.update_history()
            global window
            if self.get_current_widget().user_zoom is True and window.options["maintain user zoom"] is True and (function == self.get_current_graph().undo or function == self.get_current_graph().redo):
                self.get_current_widget().queue_draw()
            else:
                self.get_current_graph().adjust_dims()
                self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return True

    def on_traverse_graph(self, action):
        """Callback to handle pressing of eq class traversal button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        traversal_depth = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Traverse Graph ') != -1:
                traversal_depth = int(action_history_top.strip('Traverse Graph ')) + 1
        except:
            traversal_depth = 1
        ret = self.get_current_graph().traverse(self.get_current_widget(), traversal_depth)
        if ret is True:
            self.get_current_graph().action_history.append('Traverse Graph %d' % traversal_depth)
            self.update_history()
            self.get_current_graph().adjust_dims()
            self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return ret

    def on_to_traverse_least_progress(self, action):
        """Callback to handle pressing of least progress traversal button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        traversal_depth = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Traverse Least Progress ') != -1:
                traversal_depth = int(action_history_top.strip('Traverse Least Progress ')) + 1
        except:
            traversal_depth = 1
        ret = self.get_current_graph().to_traverse_least_progress(self.get_current_widget(), traversal_depth)
        self.get_current_graph().action_history.append('Traverse Least Progress %d' % traversal_depth)
        self.update_history()
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()
        return ret

    def on_to_traverse_most_progress(self, action):
        """Callback to handle pressing of most progress traversal button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        traversal_depth = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Traverse Most Progress ') != -1:
                traversal_depth = int(action_history_top.strip('Traverse Most Progress ')) + 1
        except:
            traversal_depth = 1
        ret = self.get_current_graph().to_traverse_most_progress(self.get_current_widget(), traversal_depth)
        self.get_current_graph().action_history.append('Traverse Most Progress %d' % traversal_depth)
        self.update_history()
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()
        return ret

    def search_tasks(self, text, dummy=None):
        self.get_current_graph().focus_tasks(text)

    def search_for_text(self, text, match_case_check_box):
        self.get_current_graph().focus_text(text, match_case_check_box.get_active())
        highlight_list = []
        search_text = text
        if match_case_check_box.get_active() is False:
            search_text = string.lower(text)
        for node in self.get_current_graph().nodes:
            node_text = node.attrs["label"]
            if match_case_check_box.get_active() is False:
                node_text = string.lower(node.attrs["label"])
            if node_text.find(search_text) != -1 and node.hide is False:
                highlight_list.append(node)
        self.tabs[self.notebook.get_current_page()].widget.set_highlight(highlight_list)

    def on_search_enter_cb(self, widget, arg):
        """Callback to handle activation of focus task text entry."""
        entry, combo_box, match_case_check_box = arg
        text = entry.get_text()
        s_type, search_cb, search_help = self.search_types[combo_box.get_active()]
        self.get_current_graph().set_undo_list()
        self.get_current_graph().action_history.append('Search for %s: %s' % (s_type, text))
        self.update_history()
        search_cb(text, match_case_check_box)
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()
        self.task_dialog.destroy()
        return True

    def on_search_type_toggled(self, combo_box, label):
        s_type, search_cb, search_help = self.search_types[combo_box.get_active()]
        label.set_text(search_help)

    def on_hide_model(self, action):
        """Callback to handle pressing of cut model button."""
        if self.get_current_graph().cur_filename == '':
            return False
        if not hasattr(self, 'programming_models'):
            self.programming_models = []
            site_models_path = os.path.dirname(os.path.realpath(__file__)) + '/../../../etc/STAT/STATview_models.conf'
            self.programming_models += load_model_cuts(site_models_path)
            user_models_path = '%s/.STATview_models.conf' % (os.environ.get('HOME'))
            self.programming_models += load_model_cuts(user_models_path)
        self.model_dialog = gtk.Dialog('Hide Programming Model Frames', self)
        self.model_dialog.set_default_size(400, 400)
        frame = gtk.Frame(label="Programming Models")
        vbox = gtk.VBox()

        sw = gtk.ScrolledWindow()
        sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        list_store = gtk.ListStore(gobject.TYPE_BOOLEAN, gobject.TYPE_STRING, gobject.TYPE_BOOLEAN, gobject.TYPE_STRING, gobject.TYPE_BOOLEAN, gobject.TYPE_BOOLEAN, gobject.TYPE_PYOBJECT, gobject.TYPE_STRING, gobject.TYPE_STRING, gobject.TYPE_STRING)
        for programming_model in self.programming_models:
            iterator = list_store.append()
            list_store.set(iterator, MODEL_INDEX_NAME, programming_model[MODEL_INDEX_NAME], MODEL_INDEX_CASESENSITIVE, programming_model[MODEL_INDEX_CASESENSITIVE], MODEL_INDEX_REGEX, programming_model[MODEL_INDEX_REGEX], MODEL_INDEX_EDITABLE, programming_model[MODEL_INDEX_EDITABLE], MODEL_INDEX_NOTEDITABLE, programming_model[MODEL_INDEX_NOTEDITABLE], MODEL_INDEX_CALLBACK, self.hide_model_callback, MODEL_INDEX_ICON, gtk.STOCK_CUT)
        treeview = gtk.TreeView(list_store)
        treeview.set_rules_hint(False)
        treeview.set_enable_search(False)
        treeview.get_selection().set_mode(gtk.SELECTION_SINGLE)

        # column for the hide button
        renderer = CellRendererButtonPixbuf()
        renderer.set_property('cell-background', 'grey')
        column = gtk.TreeViewColumn(" Click\nto hide")
        column.set_fixed_width(20)
        column.pack_start(renderer, True)
        column.set_attributes(renderer, stock_id=MODEL_INDEX_ICON, callable=MODEL_INDEX_CALLBACK, cell_background_set=MODEL_INDEX_NOTEDITABLE)
        treeview.append_column(column)

        # name column
        renderer = gtk.CellRendererText()
        renderer.connect("edited", self.on_cell_edited, list_store, MODEL_INDEX_NAME)
        renderer.set_property('cell-background', 'grey')
        column = gtk.TreeViewColumn("Programming\n     Model", renderer, text=MODEL_INDEX_NAME, editable=MODEL_INDEX_EDITABLE, cell_background_set=MODEL_INDEX_NOTEDITABLE)
        treeview.append_column(column)

        # column for case sensitive toggles
        renderer = gtk.CellRendererToggle()
        renderer.connect('toggled', self.case_sensitive_toggled, list_store)
        renderer.set_property('cell-background', 'grey')
        column = gtk.TreeViewColumn('   Case\nSensitive', renderer, active=MODEL_INDEX_CASESENSITIVE, cell_background_set=MODEL_INDEX_NOTEDITABLE)
        column.set_sizing(gtk.TREE_VIEW_COLUMN_FIXED)
        column.set_fixed_width(70)
        treeview.append_column(column)

        # regex column
        renderer = gtk.CellRendererText()
        renderer.connect("edited", self.on_cell_edited, list_store, MODEL_INDEX_REGEX)
        renderer.set_property('cell-background', 'grey')
        column = gtk.TreeViewColumn("Regex", renderer, text=MODEL_INDEX_REGEX, editable=MODEL_INDEX_EDITABLE, cell_background_set=MODEL_INDEX_NOTEDITABLE)
        treeview.append_column(column)

        sw.add(treeview)
        vbox.pack_start(sw, True, True, 0)
        hbox = gtk.HBox(True, 4)
        button = gtk.Button("_Add Model")
        button.connect("clicked", self.on_add_item_clicked, list_store)
        hbox.pack_start(button, True, True, 0)
        button = gtk.Button("_Remove Model")
        button.connect("clicked", self.on_remove_item_clicked, treeview)
        hbox.pack_start(button, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        frame.add(vbox)
        self.model_dialog.vbox.pack_start(frame, True, True, 0)

        separator = gtk.HSeparator()
        self.model_dialog.vbox.pack_start(separator, False, False, 10)
        hbox = gtk.HBox(True, 4)
        button = gtk.Button("_Done")
        button.connect("clicked", lambda w: self.model_dialog.destroy())
        hbox.pack_start(button, True, True, 0)
        self.model_dialog.vbox.pack_start(hbox, False, False, 0)

        self.model_dialog.show_all()
        return True

    def hide_model_callback(self, path):
        """Callback to handle activation of hide model button."""
        path = int(path)
        self.get_current_graph().set_undo_list()
        self.get_current_graph().action_history.append('Hide Model: %s' % self.programming_models[path][MODEL_INDEX_NAME])
        self.update_history()
        for regex in self.programming_models[path][MODEL_INDEX_REGEX].split(';'):
            if regex == '':
                continue
            self.tabs[self.notebook.get_current_page()].widget.graph.hide_re(regex, self.programming_models[path][MODEL_INDEX_CASESENSITIVE])
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()
        return True

    def case_sensitive_toggled(self, cell, path, list_store):
        iterator = list_store.get_iter((int(path),))
        if iterator:
            path = list_store.get_path(iterator)[0]
            if self.programming_models[path][MODEL_INDEX_EDITABLE] is False:
                return True
        toggled = list_store.get_value(iterator, MODEL_INDEX_CASESENSITIVE)
        toggled = not toggled
        self.programming_models[int(path)][MODEL_INDEX_CASESENSITIVE] = toggled
        list_store.set(iterator, MODEL_INDEX_CASESENSITIVE, toggled)

    def on_add_item_clicked(self, button, list_store):
        new_item = [False, "Name", True, "Regex", True, False]
        self.programming_models.append(new_item)
        iterator = list_store.append()
        list_store.set(iterator, MODEL_INDEX_NAME, new_item[MODEL_INDEX_NAME], MODEL_INDEX_CASESENSITIVE, new_item[MODEL_INDEX_CASESENSITIVE], MODEL_INDEX_REGEX, new_item[MODEL_INDEX_REGEX], MODEL_INDEX_EDITABLE, new_item[MODEL_INDEX_EDITABLE], MODEL_INDEX_NOTEDITABLE, new_item[MODEL_INDEX_NOTEDITABLE], MODEL_INDEX_CALLBACK, self.hide_model_callback, MODEL_INDEX_ICON, gtk.STOCK_CUT)

    def on_remove_item_clicked(self, button, treeview):
        selection = treeview.get_selection()
        list_store, iterator = selection.get_selected()
        if iterator:
            path = list_store.get_path(iterator)[0]
            if self.programming_models[path][MODEL_INDEX_EDITABLE] is True:
                list_store.remove(iterator)
                del self.programming_models[path]

    def on_cell_edited(self, widget, path, new_text, list_store, column):
        list_store[int(path)][column] = new_text
        self.programming_models[int(path)][column] = new_text

    def on_search(self, action):
        """Callback to handle pressing of focus task button."""
        #TODO-count-rep: does not work if we just have count + 1 representative
        if self.get_current_graph().cur_filename == '':
            return False
        self.task_dialog = gtk.Dialog('Search', self)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Search for:"), False, False, 0)
        label = gtk.Label()
        combo_box = gtk.combo_box_new_text()
        for search_type in self.search_types:
            s_type, search_cb, search_help = search_type
            combo_box.append_text(s_type)
        combo_box.set_active(0)
        combo_box.connect('changed', lambda w: self.on_search_type_toggled(w, label))
        hbox.pack_start(combo_box, False, False, 10)
        self.task_dialog.vbox.pack_start(hbox, False, False, 0)
        match_case_check_box = gtk.CheckButton("Match Case")
        match_case_check_box.set_active(True)
        match_case_check_box.connect('toggled', lambda x: x)
        entry = gtk.Entry()
        entry.set_max_length(65536)
        entry.connect("activate", self.on_search_enter_cb, (entry, combo_box, match_case_check_box))
        self.task_dialog.vbox.pack_start(entry, False, False, 0)
        self.task_dialog.vbox.pack_start(match_case_check_box, False, False, 0)
        separator = gtk.HSeparator()
        self.task_dialog.vbox.pack_start(separator, False, False, 0)
        s_type, search_cb, search_help = self.search_types[combo_box.get_active()]
        label.set_text(search_help)
        self.task_dialog.vbox.pack_start(label, True, True, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.task_dialog.destroy())
        hbox.pack_start(button, False, False, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", self.on_search_enter_cb, (entry, combo_box, match_case_check_box))
        hbox.pack_start(button, False, False, 0)
        self.task_dialog.vbox.pack_end(hbox, False, False, 0)
        self.task_dialog.show_all()
        return True

    def on_hide_text_enter_cb(self, widget, arg):
        """Callback to handle activation of focus task text entry."""
        entry, match_case_check_box = arg
        text = entry.get_text()
        self.get_current_graph().set_undo_list()
        self.get_current_graph().action_history.append('Cut text: %s' % (text))
        self.update_history()
        self.get_current_graph().hide_re(text, match_case_check_box.get_active())
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()
        self.task_dialog.destroy()
        return True

    def on_hide_text(self, action):
        """Callback to handle pressing of cut text button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.task_dialog = gtk.Dialog('Cut', self)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label("Cut below:"), False, False, 0)
        match_case_check_box = gtk.CheckButton("Match Case")
        match_case_check_box.set_active(True)
        match_case_check_box.connect('toggled', lambda x: x)
        entry = gtk.Entry()
        entry.set_max_length(65536)
        entry.connect("activate", self.on_hide_text_enter_cb, (entry, match_case_check_box))
        self.task_dialog.vbox.pack_start(entry, False, False, 0)
        self.task_dialog.vbox.pack_start(match_case_check_box, False, False, 0)
        help_string = """Cut the tree below frames matching
the specified text, which may be
enterered as a regular expression.
        """
        separator = gtk.HSeparator()
        self.task_dialog.vbox.pack_start(separator, False, False, 0)
        label = gtk.Label()
        label.set_text(help_string)
        self.task_dialog.vbox.pack_start(label, True, True, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.task_dialog.destroy())
        hbox.pack_start(button, False, False, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", self.on_hide_text_enter_cb, (entry, match_case_check_box))
        hbox.pack_start(button, False, False, 0)
        self.task_dialog.vbox.pack_end(hbox, False, False, 0)
        self.task_dialog.show_all()
        return True

    def on_longest_path(self, action):
        """Callback to handle pressing of longest path button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        longest_depth = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Longest Path ') != -1:
                longest_depth = int(action_history_top.strip('Longest Path ')) + 1
        except:
            longest_depth = 1
        ret = self.get_current_graph().longest_path(self.get_current_widget(), longest_depth)
        if ret is True:
            self.get_current_graph().action_history.append('Longest Path %d' % longest_depth)
            self.update_history()
            self.get_current_graph().adjust_dims()
            self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return ret

    def on_shortest_path(self, action):
        """Callback to handle pressing of shortest path button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        shortest_depth = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Shortest Path ') != -1:
                shortest_depth = int(action_history_top.strip('Shortest Path ')) + 1
        except:
            shortest_depth = 1
        ret = self.get_current_graph().shortest_path(self.get_current_widget(), shortest_depth)
        if ret is True:
            self.get_current_graph().action_history.append('Shortest Path %d' % shortest_depth)
            self.update_history()
            self.get_current_graph().adjust_dims()
            self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return ret

    def on_least_tasks(self, action):
        """Callback to handle pressing of least tasks button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        least_tasks = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Least Tasks ') != -1:
                least_tasks = int(action_history_top.strip('Least Tasks ')) + 1
        except:
            least_tasks = 1
        ret = self.get_current_graph().least_tasks(self.get_current_widget(), least_tasks)
        if ret is True:
            self.get_current_graph().action_history.append('Least Tasks %d' % least_tasks)
            self.update_history()
            self.get_current_graph().adjust_dims()
            self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return ret

    def on_most_tasks(self, action):
        """Callback to handle pressing of most tasks button."""
        if self.get_current_graph().cur_filename == '':
            return False
        self.get_current_graph().set_undo_list()
        most_tasks = 1
        try:
            action_history_top = self.get_current_graph().action_history[-1]
            if action_history_top.find('Most Tasks ') != -1:
                most_tasks = int(action_history_top.strip('Most Tasks ')) + 1
        except:
            most_tasks = 1
        ret = self.get_current_graph().most_tasks(self.get_current_widget(), most_tasks)
        if ret is True:
            self.get_current_graph().action_history.append('Most Tasks %d' % most_tasks)
            self.update_history()
            self.get_current_graph().adjust_dims()
            self.get_current_widget().zoom_to_fit()
        else:
            self.get_current_graph().undo(False)
        return ret

    def on_join_eq_classes(self, action):
        """Callback to handle pressing of collapse equivalence classes button."""
        graph = self.get_current_graph()
        if graph.cur_filename == '':
            return False
        num_edges = len(graph.edges)
        graph.set_undo_list()
        old_attrs = {}
        for node in graph.nodes:
            old_attrs[node.node_name] = (copy.copy(node.attrs))
        for node in graph.nodes:
            if node.node_name == '0':
                ret = self.join_eq_classes(node)
                break
        if ret is True:
            self.on_reset_layout()
            for node in graph.nodes:
                node.attrs = copy.copy(old_attrs[node.node_name])
            for i in range(len(graph.edges) - num_edges):
                graph.edges.pop()
            graph.undo(False)
        return ret

    def join_eq_classes(self, node):
        """Recursively collapse equivalence classes."""
        found = False
        if node.hide is True:
            return False
        if node.can_join_eq_c() and node.node_name != '0':
            self.get_current_graph().join_eq_c(node, True)
            found = True
        if found:
            edges = node.eq_collapsed_out_edges
        else:
            edges = node.out_edges
        for edge in edges:
            ret = self.join_eq_classes(edge.dst)
            if ret is True:
                found = True
        return found

    def on_identify_num_eq_classes(self, action):
        """Callback to handle pressing of equivalence classes button."""
        if self.get_current_graph().cur_filename == '':
            return False
        num_eq_classes = self.get_current_graph().identify_real_num_eq_classes(self.get_current_widget())
        #num_eq_classes = self.get_current_graph().identify_num_eq_classes(self.get_current_widget())
        eq_dialog = gtk.Dialog("Equivalence Classes", self)
        my_frame = gtk.Frame(label="%d Equivalence Classes:" % (len(num_eq_classes)))
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        task_view = gtk.TextView()
        task_view.set_editable(False)
        task_view.set_cursor_visible(False)
        task_view_buffer = task_view.get_buffer()
        task_view.set_wrap_mode(gtk.WRAP_WORD)
        eq_string = ''
        for eq_class in num_eq_classes:
            current_class = '['
            prev = -2
            in_range = False
            task_count = 0
            for num in num_eq_classes[eq_class]:
                task_count += 1
                if num == prev + 1:
                    in_range = True
                else:
                    if in_range is True:
                        current_class += '-%d' % prev
                    if prev != -2:
                        current_class += ','
                    current_class += '%d' % num
                    in_range = False
                prev = num
            if in_range is True:
                current_class += '-%d' % num
            current_class += ']\n'
            if task_count == 1:
                eq_string += '%d task: %s' % (task_count, current_class)
            else:
                eq_string += '%d tasks: %s' % (task_count, current_class)
        task_view_buffer.set_text(eq_string.replace(",", ", "))
        sw.add(task_view)
        my_frame.add(sw)
        eq_dialog.vbox.pack_start(my_frame, True, True, 5)
        self.separator = gtk.HSeparator()
        eq_dialog.vbox.pack_start(self.separator, False, True, 5)
        box2 = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", lambda w, d: eq_dialog.destroy(), "ok")
        box2.pack_end(button, False, True, 5)
        eq_dialog.vbox.pack_end(box2, False, False, 0)
        eq_dialog.show_all()
        return True

    def on_node_clicked(self, widget, button_clicked, event):
        """Callback to handle clicking of a node."""
        if isinstance(event, STATNode):
            node = event
        else:
            node = widget.get_node(int(event.x), int(event.y))
        try:
            label = node.attrs["label"].decode('utf-8')
            tasks = node.edge_label.decode('utf-8')
        except:
            label = node.attrs["label"]
            tasks = node.edge_label
        label = label.replace('\\n', '\n').replace('\\<', '<').replace('\\>', '>')
        if node.hide is True:
            return True
        options = ['Join Equivalence Class', 'Collapse', 'Collapse Depth', 'Hide', 'Expand', 'Expand All', 'Focus', 'View Source', 'Translate', 'Invert EQ Font']

        if HAVE_TOMOD is True:
            options.append('Temporally Order Children')
        if button_clicked == 'left':
            if hasattr(self, 'get_full_edge_label') and tasks.find(':') != -1:
                options.append('Get Full Edge Label')
            try:
                self.my_dialog.destroy()
            except:
                pass
            self.my_dialog = gtk.Dialog("Node", self)
            vpaned1 = gtk.VPaned()
            my_frame = gtk.Frame(label="Stack Frame:")
            sw = gtk.ScrolledWindow()
            sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
            text_view = gtk.TextView()
            for shape in node.shapes:
                if isinstance(shape, xdot_ui_elements.TextShape):
                    font_color = shape.pen.color
                else:
                    fill_color = shape.pen.fillcolor
            font_color_string = color_to_string(font_color)
            fill_color_string = color_to_string(fill_color)
            text_view.set_editable(False)
            text_view.set_cursor_visible(False)
            text_view_buffer = text_view.get_buffer()
            text_view.set_wrap_mode(gtk.WRAP_CHAR)
            text_view_buffer.create_tag("monospace", family="monospace")
            iterator = text_view_buffer.get_iter_at_offset(0)
            fore_color_tag = "color_fore%s" % (font_color_string)
            foreground = gtk.gdk.color_parse(font_color_string)
            text_view_buffer.create_tag(fore_color_tag, foreground_gdk=foreground)
            background = gtk.gdk.color_parse(fill_color_string)
            back_color_tag = "color_back%s" % (fill_color_string)
            text_view_buffer.create_tag(back_color_tag, background_gdk=background)
            text_view_buffer.insert_with_tags_by_name(iterator, label, fore_color_tag, back_color_tag, "monospace")
            sw.add(text_view)
            my_frame.add(sw)
            vpaned1.add1(my_frame)
            #TODO: make frames resizable
            if tasks.find(':') == -1:
                leaf_tasks = node.get_leaf_tasks()
                num_leaf_tasks = node.get_num_leaf_tasks()
                num_tasks = node.num_tasks
                if num_leaf_tasks != 0 and num_leaf_tasks == len(leaf_tasks):
                    vpaned2 = gtk.VPaned()
                    if num_leaf_tasks == 1:
                        my_frame = gtk.Frame(label="%d Leaf Task:" % (num_leaf_tasks))
                    else:
                        my_frame = gtk.Frame(label="%d Leaf Tasks:" % (num_leaf_tasks))
                    sw = gtk.ScrolledWindow()
                    sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
                    task_view = gtk.TextView()
                    task_view.set_editable(False)
                    task_view.set_cursor_visible(False)
                    task_view_buffer = task_view.get_buffer()
                    task_view.set_wrap_mode(gtk.WRAP_WORD)
                    x = list_to_string(leaf_tasks)
                    task_view_buffer.set_text(list_to_string(leaf_tasks).replace(",", ", ").strip('[').strip(']'))
                    sw.add(task_view)
                    my_frame.add(sw)
                    if num_tasks != num_leaf_tasks:
                        vpaned2.add1(my_frame)
                    else:
                        vpaned1.add2(my_frame)
                if num_tasks != num_leaf_tasks:
                    if num_tasks == 1:
                        my_frame = gtk.Frame(label="%d Total Task:" % (num_tasks))
                    else:
                        my_frame = gtk.Frame(label="%d Total Tasks:" % (num_tasks))
                    sw = gtk.ScrolledWindow()
                    sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
                    task_view = gtk.TextView()
                    task_view.set_editable(False)
                    task_view.set_cursor_visible(False)
                    task_view_buffer = task_view.get_buffer()
                    task_view.set_wrap_mode(gtk.WRAP_WORD)
                    task_view_buffer.set_text(tasks.replace(",", ", ").strip('[').strip(']'))
                    sw.add(task_view)
                    my_frame.add(sw)
                    try:
                        vpaned2.add2(my_frame)
                        vpaned1.add2(vpaned2)
                    except:
                        vpaned1.add2(my_frame)
            else:
                num_tasks = node.num_tasks
                num_leaf_tasks = node.get_num_leaf_tasks()
                my_frame = gtk.Frame(label="Node summary:")
                sw = gtk.ScrolledWindow()
                sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
                summary_view = gtk.TextView()
                summary_view.set_editable(False)
                summary_view.set_cursor_visible(False)
                task_view_buffer = summary_view.get_buffer()
                summary_view.set_wrap_mode(gtk.WRAP_WORD)
                task_view_buffer.set_text("%d total tasks, %d leaf tasks, representative = %d" % (num_tasks, num_leaf_tasks, get_task_list(tasks)[0]))
                sw.add(summary_view)
                my_frame.add(sw)
                vpaned1.add2(my_frame)
            self.my_dialog.vbox.pack_start(vpaned1, True, True, 5)

            expander = gtk.Expander()
            expander.set_label(label="Advanced")
            sw = gtk.ScrolledWindow()
            sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
            attributes_view = gtk.TextView()
            attributes_view.set_editable(False)
            attributes_view.set_cursor_visible(False)
            attributes_view_buffer = attributes_view.get_buffer()
            attributes_view.set_wrap_mode(gtk.WRAP_WORD)
            attributes_text = str(node)
            attributes_view_buffer.set_text(attributes_text)
            sw.add(attributes_view)
            #expander.add(sw)
            #self.my_dialog.vbox.pack_start(expander, True, True, 5)
            # TODO: When the sw is in an expander, it doesn't resize
            self.my_dialog.vbox.pack_start(sw, True, True, 5)

            self.separator = gtk.HSeparator()
            self.my_dialog.vbox.pack_start(self.separator, False, True, 5)
            box2 = gtk.HButtonBox()
            for option in options:
                button = gtk.Button(option.replace(' ', '\n'))
                if option != 'View Source' and option != 'Get Full Edge Label' and option != 'Translate' and option != 'Invert EQ Font':
                    button.connect("clicked", self.manipulate_cb, option, node)
                if option == 'View Source':
                    if not label_has_source(node.attrs["label"]):
                        button.set_sensitive(False)
                    elif label_has_source(node.attrs["label"]) and label_collapsed(node.attrs["label"]):
                        button.connect("clicked", lambda w, n: self.select_source(n), node)
                    else:
                        button.connect("clicked", self.manipulate_cb, option, node)
                elif option == 'Translate':
                    if ("offset" in node.attrs and node.attrs["offset"].find("(null)") != -1) or ("module" in node.attrs and node.attrs["module"].find("(null)") != -1) or ("source" in node.attrs and node.attrs["source"].find("(null)") == -1):
                        button.set_sensitive(False)
                    else:
                        button.connect("clicked", lambda x:self.on_translate())
                elif option == 'Join Equivalence Class':
                    if not node.can_join_eq_c():
                        button.set_sensitive(False)
                elif option == 'Get Full Edge Label':
                    button.connect("clicked", lambda x, w, l, n: self.get_full_edge_label(w, l, n), widget, button_clicked, node)
                elif option == 'Invert EQ Font':
                    button.connect("clicked", lambda x:self.on_reset_layout(invert=node))
                box2.pack_start(button, False, True, 5)
            button = gtk.Button(stock=gtk.STOCK_OK)
            button.connect("clicked", lambda w, d: self.my_dialog.destroy(), "ok")
            box2.pack_end(button, False, True, 5)
            self.my_dialog.vbox.pack_end(box2, False, False, 0)
            self.my_dialog.show_all()
        elif button_clicked == 'right':
            menu = gtk.Menu()
            for option in options:
                menu_item = gtk.MenuItem(option)
                menu.append(menu_item)
                if option != 'View Source' and option != 'Get Full Edge Label' and option != 'Translate' and option != 'Invert EQ Font':
                    menu_item.connect('activate', self.manipulate_cb, option, node)
                if option == 'View Source':
                    if label_has_source(node.attrs["label"]):
                        if not label_collapsed(node.attrs["label"]):
                            menu_item.connect('activate', self.manipulate_cb, option, node)
                        else:
                            sub_menu = gtk.Menu()
                            sources = node.attrs["source"].split('\\n')
                            for i, source in enumerate(sources):
                                if source == "(null)":
                                    source = "?"
                                else:
                                    source = os.path.basename(source)
                                cur_line_num = node.attrs["line"].split('\\n')[i]
                                if cur_line_num == "(null)":
                                    cur_line_num = "?"
                                sub_menu_item = gtk.MenuItem(source + cur_line_num)
                                sub_menu_item.connect('button-release-event', lambda w, e, o, n, i2: self.get_current_graph().view_source(n, i2), option, node, i)
                                sub_menu.append(sub_menu_item)
                                sub_menu_item.show()

                            menu_item.set_submenu(sub_menu)
                    else:
                        menu_item.set_sensitive(False)
                elif option == 'Translate':
                    if label_has_module_offset(node.attrs["label"]):
                        menu_item.connect('activate', lambda w: self.on_translate())
                    else:
                        menu_item.set_sensitive(False)
                elif option == 'Join Equivalence Class':
                    if not node.can_join_eq_c():
                        menu_item.set_sensitive(False)
                elif option == 'Invert EQ Font':
                    menu_item.connect("activate", lambda x:self.on_reset_layout(invert=node))
                menu_item.show()
            try:
                menu.popup(None, None, None, None, event.button, event.time)
            except:
                menu.popup(None, None, None, event.button, event.time)
        return True

    def on_select_source_enter_cb(self, combo_box, node):
        """Callback to handle activation of source selection."""
        index = combo_box.get_active()
        self.my_dialog.destroy()
        self.get_current_graph().view_source(node, index)

    def select_source(self, node):
        self.my_dialog.destroy()
        frames = node.attrs["label"].split('\\n')
        if len(frames) == 1:
            self.get_current_graph().view_source(node)
        else:
            self.my_dialog = gtk.Dialog("Select Frame", self)
            hbox = gtk.HBox()
            hbox.pack_start(gtk.Label("Select a frame"), False, False, 0)
            combo_box = gtk.combo_box_new_text()
            for i, frame in enumerate(frames):
                source = node.attrs["source"].split('\\n')[i]
                if source == "(null)":
                    source = "?"
                else:
                    source = os.path.basename(source)
                cur_line_num = node.attrs["line"].split('\\n')[i]
                if cur_line_num == "(null)":
                    cur_line_num = "?"
                combo_box.append_text(source + cur_line_num)
            combo_box.set_active(0)
            hbox.pack_start(combo_box, False, False, 10)
            self.my_dialog.vbox.pack_start(hbox, True, True, 0)
            hbox = gtk.HButtonBox()
            button = gtk.Button(stock=gtk.STOCK_CANCEL)
            button.connect("clicked", lambda w: self.my_dialog.destroy())
            hbox.pack_end(button, False, False, 0)
            button = gtk.Button(stock=gtk.STOCK_OK)
            button.connect("clicked", lambda w: self.on_select_source_enter_cb(combo_box, node))
            hbox.pack_end(button, False, False, 0)
            self.my_dialog.vbox.pack_end(hbox, False, False, 0)
            self.my_dialog.show_all()

    def manipulate_cb(self, widget, data, node):
        """Callback to handle the request of a manipulation operation."""
        ret = True
        # setup
        if data != 'View Source':
            ret = self.get_current_graph().set_undo_list()

        # process command
        if data == 'Collapse':
            ret = self.get_current_graph().collapse(node, True)
        elif data == 'Collapse Depth':
            ret = self.get_current_graph().collapse_depth(node)
        elif data == 'Join Equivalence Class':
            old_attrs = copy.copy(node.attrs)
            num_edges = len(self.get_current_graph().edges)
            ret, (leaf_node, attrs) = self.get_current_graph().join_eq_c(node, True)
        elif data == 'Expand':
            ret = self.get_current_graph().expand(node)
        elif data == 'Expand All':
            ret = self.get_current_graph().expand_all(node)
        elif data == 'Hide':
            ret = self.get_current_graph().collapse(node)
        elif data == 'Focus':
            ret = self.get_current_graph().focus(node)
        elif data == 'View Source':
            ret = self.get_current_graph().view_source(node)
        elif data == 'Temporally Order Children':
            ret = stat_wait_dialog.show_wait_dialog_and_run(self.get_current_graph().get_children_temporal_order, (node,), [], self)
        elif data == 'View Source':
            ret = self.get_current_graph().save_dot(invert=node)

        # post process
        if data not in ['View Source', 'Join Equivalence Class', 'Invert EQ Font']:
            if ret is True:
                self.get_current_graph().action_history.append('%s: %s' % (data, node.attrs["label"]))
                self.update_history()
            else:
                self.get_current_graph().undo(False)
        if data not in ['Temporally Order Children', 'View Source', 'Join Equivalence Class'] and ret is True:
            global window
            if self.get_current_widget().user_zoom is True and window.options["maintain user zoom"] is True:
                self.get_current_widget().queue_draw()
            else:
                self.get_current_graph().adjust_dims()
                self.get_current_widget().zoom_to_fit()
        if data == 'Join Equivalence Class' and ret is True:
            old_graph = self.get_current_graph()
            self.on_reset_layout()
            node.attrs = copy.copy(old_attrs)
            for i in range(len(old_graph.edges) - num_edges):
                old_graph.edges.pop()
            old_graph.undo(False)
        try:
            self.my_dialog.destroy()
        except:
            pass


def STATview_main(args):
    """The STATview main."""
    global window
    window = STATDotWindow()
    if args.files:
        i = -1
        for filename in args.files:
            i += 1
            try:
                with open(filename, 'r') as f:
                    if i != 0:
                        window.create_new_tab()
                    stat_wait_dialog.show_wait_dialog_and_run(window.set_dotcode, (f.read(), filename, i), ['Opening DOT file'], window)
            except IOError as e:
                sys.stderr.write('\n%s\nfailed to open file %s\n\n' % (repr(e), filename))
                sys.stderr.write('usage:\n\tSTATview [<your_STAT_generated_graph.dot>]*\n\n')
                sys.exit(-1)
            except Exception as e:
                sys.stderr.write('\n%s\nfailed to proess file %s\n\n' % (repr(e), filename))
                sys.stderr.write('usage:\n\tSTATview [<your_STAT_generated_graph.dot>]*\n\n')
                sys.exit(-1)
    window.connect('destroy', window.on_destroy)
    #gtk.gdk.threads_init()
    gtk.main()


if __name__ == '__main__':
    sys.stderr.write('WARNING: STATview.py should not be directly invoked. This has been replaced by STATmain.py and the "view" subcommand.\n')
    sys.exit(1)
