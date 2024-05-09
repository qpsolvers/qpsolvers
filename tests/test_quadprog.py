#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for quadprog."""

import unittest
import warnings

import numpy as np
import scipy.sparse as spa

from qpsolvers.exceptions import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.quadprog_ import quadprog_solve_qp

    class TestQuadprog(unittest.TestCase):
        """Test fixture for the quadprog solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_non_psd_cost(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            P -= np.eye(3)
            with self.assertRaises(ProblemError):
                quadprog_solve_qp(P, q, G, h, A, b)

        def test_quadprog_value_error(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            q = q[1:]  # raise "G and a must have the same dimension"
            self.assertIsNone(quadprog_solve_qp(P, q, G, h, A, b))

        def test_not_sparse(self):
            """Raise a ProblemError on sparse problems."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            P = spa.csc_matrix(P)
            with self.assertRaises(ProblemError):
                quadprog_solve_qp(P, q, G, h, A, b)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping quadprog tests: {exn}")
