#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for OSQP."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.osqp_ import osqp_solve_qp

    class TestOSQP(unittest.TestCase):
        """Tests specific to OSQP."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(osqp_solve_qp(P, q, G, h, A, b, lb, ub))


except ImportError:  # solver not installed
    warnings.warn("Skipping OSQP tests as the solver is not installed")
