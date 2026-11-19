#!/usr/bin/env python3
"""Focused tests for process multiplicity in STAT edge labels."""

import unittest

from STAThelper import edge_attr_to_label, get_task_list, get_truncated_edge_label


class EdgeProcessCountTests(unittest.TestCase):
    def test_equal_process_and_rank_counts_keep_existing_label(self):
        attrs = {"bv": "[0-7]", "count": "(null)", "pcount": "8"}
        self.assertEqual(edge_attr_to_label(attrs), "[0-7]")

    def test_extra_processes_add_multiplicity_suffix(self):
        attrs = {"bv": "[0,2]", "count": "(null)", "pcount": "8"}
        attrs["label"] = edge_attr_to_label(attrs)
        self.assertEqual(attrs["label"], "[0,2]x8")
        self.assertEqual(get_truncated_edge_label(attrs), "2:[0,2]x8")

    def test_process_suffix_does_not_change_rank_selection(self):
        self.assertEqual(get_task_list("[0,2]x8"), [0, 2])
        self.assertEqual(get_task_list("2:[0]x8"), [0])


if __name__ == "__main__":
    unittest.main()
