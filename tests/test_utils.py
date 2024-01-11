#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for utility functions."""

import io
import sys
import unittest

import numpy as np

from qpsolvers.utils import print_matrix_vector


class TestUtils(unittest.TestCase):
    """Test fixture for utility functions."""

    def setUp(self):
        self.G = np.array([[1.3, 2.1], [2.6, 0.3], [2.2, -1.6]])
        self.h = np.array([3.4, 1.8, -2.7]).reshape((3,))

    def test_print_matrix_vector(self):
        """Printing a matrix-vector pair outputs the proper labels."""
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
