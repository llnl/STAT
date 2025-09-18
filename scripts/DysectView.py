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
__author__ = ["Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"] "Jesper Puge Nielsen"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 2
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import STATview
import os.path
import sys
import gtk
import xdot

show_default_probes = False

def create_temp(dot_filename):
    """Create a temporary dot file."""

    temp_dot_filename = '_temp_dysect.dot'
    declaration_pos = {}
    try:
        temp_dot_file = open(temp_dot_filename, 'w')
    except:
        home_dir = os.environ.get("HOME")
        temp_dot_filename = '%s/_temp.dot' % home_dir
        try:
            temp_dot_file = open(temp_dot_filename, 'w')
        except Exception as e:
            show_error_dialog('Failed to open temp dot file %s for writing' % temp_dot_filename, exception=e)
            return (None, declaration_pos)
    temp_dot_file.write('digraph G {\n\tnode [shape=record,style=filled,labeljust=c,height=0.2];\n')
    try:
        with open(dot_filename, 'r') as dot_file:
            parser = STATview.STATDotParser(dot_file.read())
            parser.parse()
            if ("type" in parser.graph_attrs.keys()) and (parser.graph_attrs["type"] == "dysect"):
                probe_properties = ["event", "condition", "domain", "action"]
                probe_colors = ["#F98D87", "#E4C052", "#5088E7", "#ACC373"]
                hide_ids = []
                for node in parser.nodes:
                    id, attrs = node
                    if show_default_probes is False:
                        if "source" in attrs:
                            if attrs["source"] == "default_probes.cpp":
                                hide_ids.append(id)
                                continue
                    output_line = '\t%s [' % id
                    for key in attrs.keys():
                        if (key in probe_properties) or (key == "label"):
                            pass
                        else:
                            output_line += '%s="%s",' % (key, attrs[key])

                    # Store line numbers
                    line_number = attrs["line"] if "line" in attrs.keys() else None
                    source_file = attrs["source"] if "source" in attrs.keys() else None
                    declaration_pos[id] = (source_file, line_number)

                    # Generate the special label
                    if len(set(probe_properties) & set(attrs.keys())) == 0:
                        # The node is not a probe (probably the artificial root '/')
                        output_line += 'label = "%s", ' % attrs["label"]
                    else:
                        output_line += 'label = <<table border="1" cellborder="0" '
                        output_line += 'cellpadding="4" cellspacing="0" bgcolor="#F98D87">'
                        for i in range(len(probe_properties)):
#                            if attrs["source"] == "default_probes.cpp":
#                                continue
                            if (probe_properties[i] in attrs.keys()) and (attrs[probe_properties[i]] != "") and (attrs[probe_properties[i]] != "NULL"):
                                output_line += '<tr>'
                                output_line += '<td width="50" align="left" height="30" bgcolor="%s">' % probe_colors[i]
                                output_line += '<font face="verdana"><b>'
                                output_line += '%s</b></font></td>' % probe_properties[i]
                                output_line += '<td width="125" height="30" bgcolor="%s">' % probe_colors[i]
                                output_line += '<font face="Lucida Console;Courier New"><b>'
                                output_line += '%s</b></font></td>' % (attrs[probe_properties[i]]).replace(";", "<br />")
                                output_line += '</tr>'
                        output_line += '</table>>, '

                    output_line += 'penwidth="0",fillcolor="#ffffff",fontcolor="#1f1f1f"];\n'
                    temp_dot_file.write(output_line)
                for edge in parser.edges:
                    src_id, dst_id, attrs = edge
                    if src_id in hide_ids or dst_id in hide_ids:
                        continue
                    output_line = '\t%s -> %s [label=""]\n' % (src_id, dst_id)
                    temp_dot_file.write(output_line)
            else:
                STATview.show_error_dialog('Invalid dot file, must be dysect type')
                return (None, {})
    except IOError as e:
        STATview.show_error_dialog('Failed to open dot file %s' % dot_filename, exception=e)
        return (None, {})
    except Exception as e:
        STATview.show_error_dialog('Failed to create temporary dot file %s\n %s' % (dot_filename, repr(e)), exception=e)
        return (None, {})
    finally:
        temp_dot_file.write('}\n')
        temp_dot_file.close()

    return (temp_dot_filename, declaration_pos)

### The globally available DYSECT wait dialog
dysect_wait_dialog = STATview.STAT_wait_dialog()

## PyGTK widget that draws DySect generated session dot graphs.
class DysectDotWidget(STATview.STATDotWidget):
    """PyGTK widget that draws DySect generated session dot graphs.

    Derrived from STATview's DotWidget class.
    """

    def set_dotcode(self, dotcode, filename='<stdin>'):
        """Set the dotcode for the widget.

        Creates a temporary dot file and generates xdotcode with layout information.
        """
        if filename != '<stdin>':
            (temp_dot_filename, declaration_pos) = create_temp(filename)
            if temp_dot_filename is None:
                return False
            self.declaration_pos = declaration_pos
        try:
            f = file(temp_dot_filename, 'r')
            dotcode2 = f.read()
            f.close()
        except:
            STATview.show_error_dialog('Failed to read temp dot file %s' % temp_dot_filename, self, exception=e)
            return False
        os.remove(temp_dot_filename)
        xdot.DotWidget.set_dotcode(self, dotcode2, filename)
        self.graph.cur_filename = filename
        return True

## The window object containing the DysectDotWidget
class DysectDotWindow(STATview.STATDotWindow):
    """The window object containing the DysectDotWidget

    The DysectDotWindow is derrived from STATview's DotWindow class.
    """

    def __init__(self, filename, owner):
        """The constructor."""
        gtk.Window.__init__(self)
        self.options = {}
        self.owner = owner
        self.set_title('DysectAPI Probe Viewer')
        self.set_default_size(1048, 600)
        self.dot_renderer = DysectDotWidget()
        self.dot_renderer.connect('clicked', self.on_node_clicked)
        self.add(self.dot_renderer)
        self.open_file(filename)
        self.show_all()

    def set_dotcode(self, dotcode, filename='<stdin>'):
        """Set the dotcode of the specified tab's widget."""
        if self.dot_renderer.set_dotcode(dotcode, filename):
            self.dot_renderer.zoom_to_fit()

    def open_file(self, filename):
        """Open a dot file and set the dotcode for the current tab."""
        try:
            with open(filename, 'r') as f:
                dysect_wait_dialog.show_wait_dialog_and_run(self.set_dotcode, (f.read(), filename), ['Opening DOT File'], self)
        except IOError as e:
            STATview.show_error_dialog('%s\nFailed to open file:\n\n%s\n\nPlease be sure the file exists and is a valid Dysect outputted dot file.' % (repr(e), filename), exception=e)
        except IOError as e:
            STATview.show_error_dialog('%s\nFailed to process file:\n\n%s\n\nPlease be sure the file exists and is a valid Dysect outputted dot file.' % (repr(e), filename), exception=e)
        self.show_all()

    def on_node_clicked(self, widget, button_clicked, event):
        """Callback to handle clicking of a node."""
        if isinstance(event, STATview.STATNode):
            node = event
        else:
            node = widget.get_node(int(event.x), int(event.y))
        (filename, line_number) = self.dot_renderer.declaration_pos[node.node_name]
        if self.owner != None:
            self.owner.scroll_probe_window(filename, line_number)
        return True

def DysectView_main(argv):
    """The DysectView main."""
    global window
    if len(argv) != 2:
        sys.stderr.write('No file provided for probe sequence viewer!')
        sys.exit(-1)
    window = DysectDotWindow(argv[1], None)
    window.connect('destroy', window.on_destroy)
    #gtk.gdk.threads_init()
    gtk.main()


if __name__ == '__main__':
    DysectView_main(sys.argv)
