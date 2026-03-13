#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2024 Stéphane Caron and the qpsolvers contributors

"""Unit tests for pyqpmad."""

import unittest
import warnings

import numpy as np
import scipy.sparse as spa

from qpsolvers.exceptions import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.pyqpmad_ import pyqpmad_solve_qp

    class TestPyqpmad(unittest.TestCase):
        """Test fixture for the pyqpmad solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_not_sparse(self):
            """Raise a ProblemError on sparse problems."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            P = spa.csc_matrix(P)
            with self.assertRaises(ProblemError):
                pyqpmad_solve_qp(P, q, G, h, A, b)

        def test_box_constraints(self):
            """Test solving a problem with only box constraints."""
            P = np.array([[1.0, 0.0], [0.0, 1.0]])
            q = np.array([-1.0, -1.0])
            lb = np.array([0.0, 0.0])
            ub = np.array([0.5, 0.5])
            x = pyqpmad_solve_qp(P, q, lb=lb, ub=ub)
            self.assertIsNotNone(x)
            self.assertTrue(np.allclose(x, [0.5, 0.5]))

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping pyqpmad tests: {exn}")
