#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for SCS."""

import unittest
import warnings

import numpy as np

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.scs_ import scs_solve_qp

    class TestSCS(unittest.TestCase):
        """Tests specific to SCS."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(scs_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_unbounded_below(self):
            problem = get_sd3310_problem()
            P, q, _, _, _, _, _, _ = problem.unpack()
            P -= np.eye(3)  # make problem unbounded
            with self.assertRaises(ProblemError):
                scs_solve_qp(P, q)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SCS tests: {exn}")
