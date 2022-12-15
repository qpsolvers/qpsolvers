#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

import io
import sys
import unittest

import numpy as np

from qpsolvers.utils import print_matrix_vector


class TestUtils(unittest.TestCase):

    """
    Test fixture for utility functions.
    """

    def setUp(self):
        self.G = np.array([[1.3, 2.1], [2.6, 0.3], [2.2, -1.6]])
        self.h = np.array([3.4, 1.8, -2.7]).reshape((3,))

    def test_print_matrix_vector(self):
        """
        Printing a matrix-vector pair outputs the proper labels.
        """
        def run_test(G, h):
            stdout_capture = io.StringIO()
            sys.stdout = stdout_capture
            print_matrix_vector(G, "ineq_matrix", h, "ineq_vector")
            sys.stdout = sys.__stdout__
            output = stdout_capture.getvalue()
            self.assertIn("ineq_matrix =", output)
            self.assertIn(str(G[0][1]), output)
            self.assertIn("ineq_vector =", output)
            self.assertIn(str(h[1]), output)

        run_test(self.G, self.h)
        run_test(self.G, self.h[:-1])
        run_test(self.G[:-1], self.h)
