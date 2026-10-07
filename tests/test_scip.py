#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for SCIP."""

import unittest
import warnings

import numpy as np

from qpsolvers.problem import Problem

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.scip_ import scip_solve_problem, scip_solve_qp

    class TestSCIP(unittest.TestCase):
        """Test fixture for the SCIP solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_problem(self):
            """Solve the sd3310 reference problem."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b)
            self.assertIsNotNone(x)
            self.assertTrue(
                np.allclose(
                    x, [0.30769231, -0.69230769, 1.38461538], atol=1e-4
                )
            )

        def test_infeasible(self):
            """Return None on an infeasible problem."""
            P = np.eye(2)
            q = np.zeros(2)
            G = np.array([[1.0, 0.0], [-1.0, 0.0]])
            h = np.array([-1.0, -1.0])
            x = scip_solve_qp(P, q, G, h)
            self.assertIsNone(x)

        def test_warm_start_used(self):
            """SCIP accepts the warm-start guess as an initial solution.

            With a solution limit of one, the first (and only) solution is
            the warm start if it was accepted (primal bound 3), or a
            heuristic one otherwise (the trivial heuristic finds the
            origin, objective 0).
            """
            P = np.eye(2)
            q = np.array([1.0, 1.0])
            lb = np.zeros(2)
            problem = Problem(P, q, lb=lb)
            solution = scip_solve_problem(
                problem,
                initvals=np.array([1.0, 1.0]),
                **{"limits/solutions": 1},
            )
            self.assertEqual(solution.extras["status"], "sollimit")
            self.assertAlmostEqual(
                solution.extras["primal_bound"], 3.0, places=6
            )

        def test_no_duals(self):
            """The interface leaves dual multipliers unset."""
            solution = scip_solve_problem(get_sd3310_problem())
            self.assertTrue(solution.found)
            self.assertIsNone(solution.z)
            self.assertIsNone(solution.y)
            self.assertIsNone(solution.z_box)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SCIP tests: {exn}")
