#!/bin/env python

"""@package STATview
Visualizes dot graphs outputted by STAT."""

__copyright__ = """Modifications Copyright (C) 2022-2025 Intel Corporation
SPDX-License-Identifier: BSD-3-Clause"""
__license__ = """Produced by Intel Corporation for Lawrence Livermore National Security, LLC.
Written by Matti Puputti matti.puputti@intel.com, M. Oguzhan Karakaya oguzhan.karakaya@intel.com
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Abdul Basit Ijaz <abdul.b.ijaz@intel.com>", "Matti Puputti <matti.puputti@intel.com>", "M. Oguzhan Karakaya <oguzhan.karakaya@intel.com>"]
__version_major__ = 4
__version_minor__ = 1
__version_revision__ = 2
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import unittest
from unittest.mock import Mock
from oneapi_gdb import OneAPIGdbDriver, parse_frameinfo_from_backtrace

class TestGDBParsing(unittest.TestCase):
    """
    Testing OneAPIGdbDriver and utility functions
    in oneapi_gdb module
    """

    def setUp(self):
        """Initializes environment for tests"""
        self.gdb_driver = OneAPIGdbDriver(0, 'debug', 'log.txt')
        self.gdb_driver.communicate = Mock()

    def test_oneapigdbdriver_class_statics(self):
        """
        Tests class static variables
        """
        self.assertEqual(OneAPIGdbDriver.gdb_command, "gdb-oneapi")

    def test_info_threads(self):
        """
        Tests get_thread_list method.
        """

        # Mock the communicate method to return sample thread data
        self.gdb_driver.communicate.return_value = [
            "1.1",
            "1.2",
            "2.1"
        ]

        result = self.gdb_driver.get_thread_list()

        # Verify the command was constructed correctly
        expected_cmd = (
            'thread apply all -q -c printf "%d.%d\\n", '
            '$_inferior, $_thread'
        )
        self.gdb_driver.communicate.assert_called_once_with(
            expected_cmd
        )

        # Verify results
        self.assertEqual(result, ["1.1", "1.2", "2.1"])

    def test_info_threads_empty_tid_list(self):
        """
        Tests get_thread_list method with empty response
        """

        self.gdb_driver.communicate.return_value = []
        result = self.gdb_driver.get_thread_list()

        self.assertEqual(result, [])

    def test_bt(self):
        """
        Tests parse_frameinfo_from_backtrace method.
        """

        info = parse_frameinfo_from_backtrace("#0  SubKernel0::FunctionX1 (a=0x75569dba0000, "
                                              "b=0x75569db70000, o=0x75569db60000, id=8) at /"
                                              "path/to/demo/app/./sub_kernel_0.h:37")
        self.assertEqual(info, {
            'function': 'SubKernel0::FunctionX1',
            'source': '/path/to/demo/app/./sub_kernel_0.h',
            'linenum': 37,
            'error': False})

        info = parse_frameinfo_from_backtrace("#1  0x00006000f319ee80 in SubKernel0::Function"
                                              "A (a=0x75569dba0000, b=0x75569db70000, o=0x755"
                                              "69db60000, local_id=8, id=8) at /path/to/demo/"
                                              "app/./sub_kernel_0.h:87")
        self.assertEqual(info, {
            'function': 'SubKernel0::FunctionA',
            'source': '/path/to/demo/app/./sub_kernel_0.h',
            'linenum': 87,
            'error': False})

        info = parse_frameinfo_from_backtrace("#2  0x00006000f31a2250 in SubKernel0::Function"
                                              "K (a=0x75569dba0000, b=0x75569db70000, o=0x755"
                                              "69db60000, local_id=8, group_id=0, global_id=8"
                                              ") at /path/to/demo/app/./sub_kernel_0.h:134")
        self.assertEqual(info, {
            'function': 'SubKernel0::FunctionK',
            'source': '/path/to/demo/app/./sub_kernel_0.h',
            'linenum': 134,
            'error': False})

        info = parse_frameinfo_from_backtrace("#3  0x00006000f31c27a0 in SubKernel<0>::SubKer"
                                              "nelImp<cl::sycl::accessor<float, 1, (cl::sycl:"
                                              ":access::mode)1024, (cl::sycl::access::target)"
                                              "2014, (cl::sycl::access::placeholder)0, cl::sy"
                                              "cl::ext::oneapi::accessor_property_list<> >, c"
                                              "l::sycl::accessor<float, 1, (cl::sycl::access:"
                                              ":mode)1025, (cl::sycl::access::target)2014, (c"
                                              "l::sycl::access::placeholder)0, cl::sycl::ext:"
                                              ":oneapi::accessor_property_list<> > const>(cl:"
                                              ":sycl::accessor<float, 1, (cl::sycl::access::m"
                                              "ode)1024, (cl::sycl::access::target)2014, (cl:"
                                              ":sycl::access::placeholder)0, cl::sycl::ext::o"
                                              "neapi::accessor_property_list<> > const&, cl::"
                                              "sycl::accessor<float, 1, (cl::sycl::access::mo"
                                              "de)1024, (cl::sycl::access::target)2014, (cl::"
                                              "sycl::access::placeholder)0, cl::sycl::ext::on"
                                              "eapi::accessor_property_list<> > const&, cl::s"
                                              "ycl::accessor<float, 1, (cl::sycl::access::mod"
                                              "e)1025, (cl::sycl::access::target)2014, (cl::s"
                                              "ycl::access::placeholder)0, cl::sycl::ext::one"
                                              "api::accessor_property_list<> > const&, cl::sy"
                                              "cl::nd_item<1>&) (in_acc_a=..., in_acc_b=..., "
                                              "out_acc=..., item=...) at /path/to/demo/app/./"
                                              "sub_kernel_0.h:154")
        self.assertEqual(info, {
            'function': 'SubKernel<...>::SubKernelImp<...>(...)',
            'source': '/path/to/demo/app/./sub_kernel_0.h',
            'linenum': 154,
            'error': False})

        info = parse_frameinfo_from_backtrace("#4  0x00006000f31c3040 in SimpleKernel<0, cl::"
                                              "sycl::accessor<float, 1, (cl::sycl::access::mo"
                                              "de)1024, (cl::sycl::access::target)2014, (cl::"
                                              "sycl::access::placeholder)0, cl::sycl::ext::on"
                                              "eapi::accessor_property_list<> >, cl::sycl::ac"
                                              "cessor<float, 1, (cl::sycl::access::mode)1025,"
                                              " (cl::sycl::access::target)2014, (cl::sycl::ac"
                                              "cess::placeholder)0, cl::sycl::ext::oneapi::ac"
                                              "cessor_property_list<> > >::operator()(cl::syc"
                                              "l::nd_item<1>) const (this=0x75569de02990, ite"
                                              "m=<>) at main.cpp:24")
        self.assertEqual(info, {
            'function': 'SimpleKernel<...>::operator()(...) const',
            'source': 'main.cpp',
            'linenum': 24,
            'error': False})

        info = parse_frameinfo_from_backtrace("#5  0x00006000f3197120 in typeinfo name for Si"
                                              "mpleKernel<0, cl::sycl::accessor<float, 1, (cl"
                                              "::sycl::access::mode)1024, (cl::sycl::access::"
                                              "target)2014, (cl::sycl::access::placeholder)0,"
                                              " cl::sycl::ext::oneapi::accessor_property_list"
                                              "<> >, cl::sycl::accessor<float, 1, (cl::sycl::"
                                              "access::mode)1025, (cl::sycl::access::target)2"
                                              "014, (cl::sycl::access::placeholder)0, cl::syc"
                                              "l::ext::oneapi::accessor_property_list<> > > ("
                                              "_arg_in_acc_a=cl::sycl::id<1> = {...}, _arg_in"
                                              "_acc_a=cl::sycl::id<1> = {...}, _arg_in_acc_a="
                                              "cl::sycl::id<1> = {...}, _arg_in_acc_a=cl::syc"
                                              "l::id<1> = {...}, _arg_in_acc_b=cl::sycl::id<1"
                                              "> = {...}, _arg_in_acc_b=cl::sycl::id<1> = {.."
                                              ".}, _arg_in_acc_b=cl::sycl::id<1> = {...}, _ar"
                                              "g_in_acc_b=cl::sycl::id<1> = {...}, _arg_out_a"
                                              "cc=cl::sycl::id<1> = {...}, _arg_out_acc=cl::s"
                                              "ycl::id<1> = {...}, _arg_out_acc=cl::sycl::id<"
                                              "1> = {...}, _arg_out_acc=cl::sycl::id<1> = {.."
                                              ".}) at //path/to/oneapi/compiler/../include/sy"
                                              "cl/CL/sycl/handler.hpp:939")
        self.assertEqual(info, {
            'function': 'typeinfo name for SimpleKernel<...>',
            'source': '//path/to/oneapi/compiler/../include/sycl/CL/sycl/handler.hpp',
            'linenum': 939,
            'error': False})

if __name__ == '__main__':
    unittest.main()
