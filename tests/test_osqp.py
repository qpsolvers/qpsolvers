#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for OSQP."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.osqp_ import osqp_solve_qp

    class TestOSQP(unittest.TestCase):
        """Tests specific to OSQP."""

        def test_problem(self):
            """Solve a standard QP with OSQP."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(osqp_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_raise_error_kwarg(self):
            """The `raise_error` OSQP kwarg should be forwarded to solve().

            See https://github.com/qpsolvers/qpsolvers/issues/380
            """
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(
                osqp_solve_qp(P, q, G, h, A, b, lb, ub, raise_error=True)
            )

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping OSQP tests: {exn}")
