#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for Solution class."""

import unittest

import numpy as np

from qpsolvers import Solution, solve_problem

from .problems import get_sd3310_problem


class TestSolution(unittest.TestCase):
    """Test fixture for solutions."""

    def test_found_default(self):
        solution = Solution(get_sd3310_problem())
        self.assertFalse(solution.found)

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

    def test_undefined_optimality(self):
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
