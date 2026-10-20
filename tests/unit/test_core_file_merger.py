#!/bin/env python

"""@package core_file_merger_test
Test for core_file_merger package."""

__copyright__ = """Modifications Copyright (C) 2022-2026 Intel Corporation
SPDX-License-Identifier: BSD-3-Clause"""
__license__ = """Produced by Intel Corporation for Lawrence Livermore National Security, LLC.
Written by Abdul Basit Ijaz abdul.b.ijaz@intel.com
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Abdul Basit Ijaz <abdul.b.ijaz@intel.com>"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 3
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import unittest
from core_file_merger import CoreFile

# Access the private regex via Python name mangling.
_reFrame = CoreFile._CoreFile__reFrame


class TestReFrameRegex(unittest.TestCase):
    """Test RegEx __reFrame for finding information about a frame from a back trace.
         group(1) = value of the frame number
         group(2) = name of the function
         group(3) = source file and line number in format "file:line"
         __reFrame = re.compile(r"#(\d+)\s+(?:0x\S+\s+in\s+)?(.*?)\s+\(.*\)(?:\s+at\s+(\S+:\d+))?")
    """

    def _match(self, line):
        return _reFrame.match(line)

    def test_simple_function(self):
        """Tests a simple C function frame with address and source location."""
        m = self._match("#0  0x00001234 in foo (x=5) at foo.c:10")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '0')
        self.assertEqual(m.group(2), 'foo')
        self.assertEqual(m.group(3), 'foo.c:10')

    def test_no_address(self):
        """Tests a frame with no hex address prefix."""
        m = self._match("#3  foo (x=5) at foo.c:7")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '3')
        self.assertEqual(m.group(2), 'foo')
        self.assertEqual(m.group(3), 'foo.c:7')

    def test_no_source_location(self):
        """Tests a frame with no source location (group(3) should be None)."""
        m = self._match("#2  0x00001234 in foo (x=5)")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'foo')
        self.assertIsNone(m.group(3))

    def test_unknown_function(self):
        """Tests a frame with unknown function ??."""
        m = self._match("#4  0x00001234 in ?? ()")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '4')
        self.assertEqual(m.group(2), '??')

    def test_frame_number(self):
        """Tests that multi-digit frame numbers are captured correctly."""
        m = self._match("#42  0x00001234 in foo ()")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '42')

    def test_template_with_spaces(self):
        """Tests a C++ template function name containing spaces after commas."""
        m = self._match("#0  0x00001234 in Foo<int, double> (x=5) at foo.cpp:10")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'Foo<int, double>')
        self.assertEqual(m.group(3), 'foo.cpp:10')

    def test_nested_template_with_spaces(self):
        """Tests a deeply nested C++ template name with spaces."""
        m = self._match("#1  0x00001234 in Kokkos::Impl::Foo<double, Kokkos::SYCL>::bar (this=0x1) at foo.hpp:42")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'Kokkos::Impl::Foo<double, Kokkos::SYCL>::bar')
        self.assertEqual(m.group(3), 'foo.hpp:42')

    def test_operator_call(self):
        """Tests a C++ operator() function name."""
        m = self._match("#0  0x00001234 in foo::operator() (this=0x1) at foo.cpp:5")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'foo::operator()')
        self.assertEqual(m.group(3), 'foo.cpp:5')

    def test_lambda(self):
        """Tests a lambda function name containing () in the name."""
        m = self._match("#2  0x00001234 in main::{lambda()#1}::operator() (this=0x1) at foo.cpp:20")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'main::{lambda()#1}::operator()')
        self.assertEqual(m.group(3), 'foo.cpp:20')

    def test_nested_parens_in_args(self):
        """Tests that nested '(' inside argument values does not show into group(2)."""
        m = self._match("#0  0x00001234 in foo (a=bar (1)) at foo.c:10")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), 'foo')


if __name__ == '__main__':
    unittest.main()
