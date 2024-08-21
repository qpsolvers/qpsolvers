#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 Lev Kozlov

"""Unit tests for qpax."""

import unittest
import warnings

import numpy as np

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.qpax_ import qpax_solve_qp

    class TestQpax(unittest.TestCase):
        """Tests specific to qpax."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(qpax_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_unbounded_below(self):
            problem = get_sd3310_problem()
            P, q, _, _, _, _, _, _ = problem.unpack()
            P -= np.eye(3)  # make problem unbounded
            with self.assertRaises(ProblemError):
                qpax_solve_qp(P, q)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping qpax tests: {exn}")
