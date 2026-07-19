#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for SIP."""

import unittest
import warnings

import numpy as np

from qpsolvers import Problem

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.sip_ import sip_solve_problem, sip_solve_qp

    class TestSIP(unittest.TestCase):
        """Tests specific to SIP."""

        def test_problem(self):
            """Solve an established qpsolvers test problem."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(sip_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_native_variable_bounds(self):
            """Pass variable bounds through without general inequalities."""
            problem = Problem(
                P=np.array([[4.0, 1.0], [1.0, 2.0]]),
                q=np.array([1.0, 1.0]),
                A=np.array([[1.0, 1.0]]),
                b=np.array([1.0]),
                lb=np.array([0.0, 0.0]),
                ub=np.array([0.7, 0.7]),
            )

            solution = sip_solve_problem(problem)

            self.assertTrue(solution.found)
            np.testing.assert_allclose(solution.x, [0.3, 0.7], atol=1e-5)
            self.assertEqual(solution.z.shape, (0,))
            self.assertEqual(solution.z_box.shape, (2,))
            self.assertGreater(solution.z_box[1], 0.0)

        def test_fixed_variable_bound(self):
            """Represent an exact fixed bound as an equality."""
            problem = Problem(
                P=np.eye(1),
                q=np.array([-2.0]),
                lb=np.array([1.0]),
                ub=np.array([1.0]),
            )

            solution = sip_solve_problem(problem)

            self.assertTrue(solution.found)
            np.testing.assert_allclose(solution.x, [1.0], atol=1e-5)
            self.assertEqual(solution.y.shape, (0,))
            np.testing.assert_allclose(solution.z_box, [1.0], atol=1e-5)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SIP tests: {exn}")
