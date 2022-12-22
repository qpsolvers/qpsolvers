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

import math
import unittest

from numpy.linalg import norm

from qpsolvers import available_solvers, solve_problem

from .solved_problems import get_qpsut01, get_qpsut02, get_qpsut03


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

    @staticmethod
    def get_test_qpsut01(solver: str):
        """
        Get test function for a given solver.

        Parameters
        ----------
        solver :
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
            eps_abs = (
                5e-1
                if solver == "osqp"
                else 5e-3
                if solver == "proxqp"
                else 1e-4
                if solver == "ecos"
                else 1e-6
                if solver in ["cvxopt", "qpswift", "scs"]
                else 1e-7
            )
            self.assertLess(norm(solution.x - ref_solution.x), eps_abs)
            # NB: in general the dual solution is not unique (that's why the
            # other tests check residuals). This test only works because the
            # dual solution is unique in this particular problem.
            self.assertLess(norm(solution.y - ref_solution.y), eps_abs)
            self.assertLess(norm(solution.z - ref_solution.z), eps_abs)
            self.assertLess(norm(solution.z_box - ref_solution.z_box), eps_abs)

        return test

    @staticmethod
    def get_test_qpsut02(solver: str):
        """
        Get test function for a given solver.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        test : function
            Test function for that solver.
        """

        def test(self):
            ref_solution = get_qpsut02()
            problem = ref_solution.problem
            solution = solve_problem(problem, solver=solver)
            eps_abs = (
                5e-2
                if solver == "ecos"
                else 5e-4
                if solver in ["proxqp", "scs"]
                else 1e-4
                if solver == "cvxopt"
                else 1e-5
                if solver in ["highs", "osqp"]
                else 5e-7
                if solver == "qpswift"
                else 1e-7
                if solver == "gurobi"
                else 1e-8
            )
            self.assertLess(norm(solution.x - ref_solution.x), eps_abs)
            self.assertLess(solution.primal_residual(), eps_abs)
            self.assertLess(solution.dual_residual(), eps_abs)
            self.assertLess(solution.duality_gap(), eps_abs)

        return test

    @staticmethod
    def get_test_duality_gap(solver: str):
        """
        Get test function for a given solver.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        test : function
            Test function for that solver.
        """

        def test(self):
            ref_solution = get_qpsut03()
            problem = ref_solution.problem
            solution = solve_problem(problem, solver=solver)
            self.assertFalse(math.isnan(solution.duality_gap()))

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestDualMultipliers,
        f"test_qpsut01_{solver}",
        TestDualMultipliers.get_test_qpsut01(solver),
    )
    setattr(
        TestDualMultipliers,
        f"test_qpsut02_{solver}",
        TestDualMultipliers.get_test_qpsut02(solver),
    )
    if solver != "cvxopt":
        # See https://github.com/stephane-caron/qpsolvers/issues/137
        setattr(
            TestDualMultipliers,
            f"test_duality_gap_{solver}",
            TestDualMultipliers.get_test_duality_gap(solver),
        )
