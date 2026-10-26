"""Check edge navigation, node opacity and keyboard zoom using installed STAT."""

import unittest

import cairo
import STATview as statview
from xdot.ui.pen import Pen


class ViewerCompatibilityTests(unittest.TestCase):
    def node(self, x):
        shape = statview.xdot_ui_elements.EllipseShape(Pen(), x, 20, 8, 8, filled=True)
        return statview.STATNode(x, 20, x-8, 12, x+8, 28, [shape], 'frame', {})

    def test_edge_navigation(self):
        """Create an edge and check that direct and xdot calls jump to its destination."""
        edge = statview.STATEdge(self.node(10), self.node(90),
                                [(10, 20), (90, 20)], [], '1:[0]', {})
        graph = statview.xdot_ui_elements.Graph(100, 40, [], [], [edge])

        self.assertEqual(edge.get_jump(10, 20).x, 90)
        self.assertEqual(edge.get_jump(10, 20, False).x, 90)
        self.assertEqual(graph.get_jump(10, 20).x, 90)

    def test_hidden_node_drawing(self):
        """Draw a node with xdot and check its alpha when hidden and visible."""
        node = self.node(10)
        graph = statview.xdot_ui_elements.Graph(100, 40, [], [node], [])
        context = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 40))

        for hidden, alpha in ((True, 0.07), (False, 1.0)):
            node.hide = hidden
            graph.draw(context)

            self.assertEqual(node.shapes[0].pen.color[3], alpha)
            self.assertEqual(node.shapes[0].pen.fillcolor[3], alpha)

    def test_keyboard_zoom(self):
        """Send a GTK plus-key event and check that STAT records manual zoom."""
        widget = statview.STATDotWidget()
        self.addCleanup(widget.destroy)
        event = statview.gtk.gdk.Event.new(statview.gtk.gdk.EventType.KEY_PRESS)
        event.keyval = statview.gtk.gdk.keyval_from_name('plus')
        event.state = statview.gtk.gdk.ModifierType(0)

        widget.on_key_press_event(widget, event)

        self.assertTrue(widget.user_zoom)


if __name__ == '__main__':
    unittest.main()
