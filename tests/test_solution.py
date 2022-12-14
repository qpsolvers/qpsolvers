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
