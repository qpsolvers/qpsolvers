#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for Solution class."""

import unittest

import numpy as np
import scipy.sparse as spa

from qpsolvers import Problem, Solution, solve_problem

from .problems import get_sd3310_problem


class TestSolution(unittest.TestCase):
    """Test fixture for solutions."""

    def test_found_default(self):
        """Found attribute is None if no solver was run."""
        solution = Solution(get_sd3310_problem())
        self.assertIsNone(solution.found)

    def test_residuals(self):
        """Test residuals at the solution of the SD3310 problem.

        Note
        ----
        This function uses DAQP to find a solution.
        """
        problem = get_sd3310_problem()
        solution = solve_problem(problem, solver="daqp")
        eps_abs = 1e-10
        self.assertLess(solution.primal_residual(), eps_abs)
        self.assertLess(solution.dual_residual(), eps_abs)
        self.assertLess(solution.duality_gap(), eps_abs)
        self.assertTrue(solution.is_optimal(eps_abs))

    def test_infinite_inequalities(self):
        """Disabled inequalities contribute zero to the dual objective."""
        diagonal = np.array([2.0, 3.0])
        target = np.array([1.0, 2.0])
        for sparse in [False, True]:
            for bounds in [
                [np.inf, 1.0],
                [0.5, np.inf],
                [np.inf, np.inf],
                [0.5, 1.0],
            ]:
                with self.subTest(sparse=sparse, bounds=bounds):
                    P, G = np.diag(diagonal), np.eye(2)
                    if sparse:
                        P, G = spa.csc_matrix(P), spa.csc_matrix(G)
                    h = np.array(bounds)
                    x = np.minimum(target, h)
                    problem = Problem(P, -diagonal * target, G, h)
                    solution = Solution(
                        problem,
                        found=True,
                        x=x,
                        z=diagonal * (target - x),
                    )
                    with np.errstate(invalid="raise"):
                        self.assertEqual(solution.duality_gap(), 0.0)
                        self.assertTrue(solution.is_optimal(1e-12))

    def test_infinite_inequality_nonzero_multiplier(self):
        """Nonzero multipliers on disabled inequalities remain nonoptimal."""
        problem = Problem(
            np.eye(2),
            np.array([-1.0, -1.0]),
            np.eye(2),
            np.array([np.inf, 2.0]),
        )
        solution = Solution(
            problem,
            found=True,
            x=np.array([0.0, 1.0]),
            z=np.array([1.0, 0.0]),
        )
        self.assertEqual(solution.primal_residual(), 0.0)
        self.assertEqual(solution.dual_residual(), 0.0)
        self.assertEqual(solution.duality_gap(), np.inf)
        self.assertFalse(solution.is_optimal(1e-12))

    def test_infinite_box_bound_nonzero_multiplier(self):
        """Nonzero multipliers on disabled box bounds remain nonoptimal."""
        for case in [
            dict(
                side="upper",
                q=np.array([-1.0, -1.0]),
                lb=None,
                ub=np.array([np.inf, 2.0]),
                z_box=np.array([1.0, 0.0]),
            ),
            dict(
                side="lower",
                q=np.array([1.0, -1.0]),
                lb=np.array([-np.inf, -2.0]),
                ub=None,
                z_box=np.array([-1.0, 0.0]),
            ),
        ]:
            with self.subTest(side=case["side"]):
                problem = Problem(
                    np.eye(2), case["q"], lb=case["lb"], ub=case["ub"]
                )
                # Not a valid solution: the optimum is (1, 1) in the upper
                # case and (-1, 1) in the lower case. The multiplier z_box
                # makes stationarity hold exactly by claiming an active bound
                # on x[0], but that bound is infinite. Primal and dual
                # residuals are both zero, so only the duality gap can
                # detect that this point is not optimal.
                solution = Solution(
                    problem,
                    found=True,
                    x=np.array([0.0, 1.0]),
                    z_box=case["z_box"],
                )
                self.assertEqual(solution.primal_residual(), 0.0)
                self.assertEqual(solution.dual_residual(), 0.0)
                self.assertEqual(solution.duality_gap(), np.inf)
                self.assertFalse(solution.is_optimal(1e-12))

    def test_undefined_optimality(self):
        """Residuals are infinite unless solution is fully defined."""
        solution = Solution(get_sd3310_problem())

        # solution is fully undefined
        self.assertEqual(solution.primal_residual(), np.inf)
        self.assertEqual(solution.dual_residual(), np.inf)
        self.assertEqual(solution.duality_gap(), np.inf)

        # solution was not found
        solution.found = False
        self.assertEqual(solution.primal_residual(), np.inf)
        self.assertEqual(solution.dual_residual(), np.inf)
        self.assertEqual(solution.duality_gap(), np.inf)

        solution.found = True
        solution.x = np.array([1.0, 2.0, 3.0])
        self.assertNotEqual(solution.primal_residual(), np.inf)
        self.assertEqual(solution.dual_residual(), np.inf)
        self.assertEqual(solution.duality_gap(), np.inf)

        solution.z = np.array([-1.0, -2.0, -3.0])
        self.assertEqual(solution.dual_residual(), np.inf)
        self.assertEqual(solution.duality_gap(), np.inf)

        solution.y = np.array([0.0])
        self.assertNotEqual(solution.dual_residual(), np.inf)
        self.assertNotEqual(solution.duality_gap(), np.inf)

        # solution is now fully defined
        self.assertGreater(solution.primal_residual(), 1.0)
        self.assertGreater(solution.dual_residual(), 10.0)
        self.assertGreater(solution.duality_gap(), 10.0)
