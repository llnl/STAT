"""Test DOT escaping without changing function names."""

import os
from pathlib import Path
import tempfile
import unittest

import STATmerge
import STATview as statview


class DotEscapingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(directory.name)

    def functions(self, filename):
        parser = statview.STATDotParser(Path(filename).read_bytes())
        parser.parse()
        return {node: attrs['function'] for node, attrs in parser.nodes}

    def test_native_output_and_rewrite_preserve_functions(self):
        """Write a trace and check that opening its DOT twice preserves function names."""

        # This lambda name was corrupted in a saved PyTorch workload graph.
        function = ('native:at::native::_local_scalar_dense_cuda(at::Tensor const&)::'
                    '{lambda()#1}::operator()() const')

        # These operator names can appear with C++11 examples.
        frames = [function, 'operator"" _bytes(unsigned long long)', 'operator|(Flags, Flags)']

        # Add a stack for rank 0/thread 0. These calls return 0 on success.
        self.assertEqual(STATmerge.Init_Graphlib(2), 0)
        graph = STATmerge.New_Graph()
        self.assertEqual(STATmerge.Add_Trace(graph, 0, 0, frames), 0)

        self.assertEqual(STATmerge.Output_Graph(graph, 'input.dot'), 0)
        original_functions = self.functions('input.dot')

        # STAT adds '/' as the root; every supplied function must remain intact.
        self.assertCountEqual(original_functions.values(), ['/'] + frames)

        # First open checks the original rewrite; reopen checks that processing
        # already-written DOT does not add another layer of backslash escapes.
        for operation in ('first open', 'reopen'):
            with self.subTest(operation=operation):
                rewritten = statview.create_temp('input.dot', 'rear', 10000)
                self.assertEqual(self.functions(rewritten), original_functions)
                Path('input.dot').write_bytes(Path(rewritten).read_bytes())

    def test_save_preserves_literal_operator(self):
        """Save a C++ literal operator and check that its quotes remain in the name."""
        function = 'operator"" _bytes(unsigned long long)'
        attrs = {'function': function, 'source': '(null)'}
        # Saving a node's metadata needs no layout, shapes or edges.
        node = statview.STATNode(0, 0, 0, 0, 0, 0, [], function, attrs)
        node.node_name = '1'
        node.edge_label = ''  # A standalone node has no incoming task edge.
        graph = statview.STATGraph(nodes=[node])

        self.assertTrue(graph.save_dot('saved.dot'))
        self.assertEqual(self.functions('saved.dot'), {'1': function})

    def test_record_labels(self):
        """Protect Python and C++ names from Graphviz's record-field syntax."""
        # Python frame/source names from dots/, plus the compiled C++ operator.
        cases = (
            ('<module>', r'\<module\>'),
            ('<lambda>', r'\<lambda\>'),
            ('<dynamic_function>', r'\<dynamic_function\>'),
            ('operator|(Flags, Flags)', r'operator\|(Flags, Flags)'),
            (r'\<lambda\>', r'\<lambda\>'),  # Do not escape an existing label twice.
        )

        for label, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(statview.escaped_label(label), expected)


if __name__ == '__main__':
    unittest.main()
