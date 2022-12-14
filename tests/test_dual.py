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

"""
Tests both primal and dual solutions to a set of problems.
"""

import unittest
import warnings

from qpsolvers import available_solvers, solve_problem
from numpy.linalg import norm

from .solved_problems import get_qpsut01


class TestDualMultipliers(unittest.TestCase):

    """
    Test fixture for primal and dual solutions.

    Notes
    -----
    Solver-specific tests are implemented in static methods called
    ``get_test_{foo}`` that return the test function for a given solver. The
    corresponding test function ``test_{foo}_{solver}`` is then added to the
    fixture below the class definition.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    @staticmethod
    def get_test_qpsut01(solver):
        """
        Get test function for a given solver.

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
            ref_solution = get_qpsut01()
            problem = ref_solution.problem
            solution = solve_problem(problem, solver=solver)
            eps_abs = 1e-8
            self.assertLess(norm(solution.x - ref_solution.x), eps_abs)
            self.assertLess(norm(solution.y - ref_solution.y), eps_abs)
            self.assertLess(norm(solution.z - ref_solution.z), eps_abs)
            self.assertLess(norm(solution.z_box - ref_solution.z_box), eps_abs)

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestDualMultipliers,
        f"test_qpsut01_{solver}",
        TestDualMultipliers.get_test_01(solver),
    )
