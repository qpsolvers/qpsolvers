#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for qpSWIFT."""

import unittest
import warnings

import scipy.sparse as spa

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.qpswift_ import qpswift_solve_qp

    class TestQpSwift(unittest.TestCase):
        """Tests specific to qpSWIFT."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(qpswift_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_not_sparse(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            P = spa.csc_matrix(P)
            with self.assertRaises(ProblemError):
                qpswift_solve_qp(P, q, G, h, A, b, lb, ub)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping qpSWIFT tests: {exn}")
