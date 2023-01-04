#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

import unittest

import numpy as np

from qpsolvers import Solution, solve_problem

from .problems import get_sd3310_problem


class TestSolution(unittest.TestCase):

    """
    Test fixture for solutions.
    """

    def test_empty(self):
        solution = Solution(get_sd3310_problem())
        self.assertTrue(solution.is_empty)

    def test_residuals(self):
        problem = get_sd3310_problem()
        solution = solve_problem(problem, solver="quadprog")
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
