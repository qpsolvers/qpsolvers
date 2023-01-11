#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 Inria
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

"""
Tests for the `solve_problem` function.
"""

import unittest

from numpy.linalg import norm

from qpsolvers import available_solvers, solve_problem

from .solved_problems import get_maros_meszaros_qptest


class TestSolveProblem(unittest.TestCase):

    """
    Test fixture for a variety of quadratic programs.
    """

    @staticmethod
    def get_test_maros_meszaros_qptest(solver):
        """
        Get test function for the QPTEST problem.

        Parameters
        ----------
        solver : string
            Name of the solver to test.

        Returns
        -------
        test : function
            Test function for that solver.
        """

        def test(self):
            solution = get_maros_meszaros_qptest()
            problem = solution.problem
            result = solve_problem(problem, solver=solver)
            tolerance = 1e-8
            self.assertLess(norm(result.x - solution.x), tolerance)
            self.assertLess(norm(result.z - solution.z), tolerance)
            self.assertLess(norm(result.z_box - solution.z_box), tolerance)

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestSolveProblem,
        f"test_maros_meszaros_qptest_{solver}",
        TestSolveProblem.get_test_maros_meszaros_qptest(solver),
    )
