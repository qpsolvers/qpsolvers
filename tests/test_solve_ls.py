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
Tests for the main `solve_qp` function.
"""

import unittest
import warnings

from numpy import allclose, array, dot
from numpy.linalg import norm
from qpsolvers import available_solvers, solve_ls


class TestSolveLS(unittest.TestCase):

    """
    Test fixture for the README example problem.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.R = array([[1.0, 2.0, 0.0], [2.0, 3.0, 4.0], [0.0, 4.0, 1.0]])
        self.s = array([3.0, 2.0, 3.0])
        self.G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        self.h = array([3.0, 2.0, -2.0]).reshape((3,))
        self.A = array([1.0, 1.0, 1.0])
        self.b = array([1.0])

    def get_problem(self):
        """
        Get problem as a sextuple of values to unpack.

        Returns
        -------
        R :
            Least-squares matrix.
        s :
            Least-squares vector.
        G :
            Linear inequality matrix.
        h :
            Linear inequality vector.
        A :
            Linear equality matrix.
        b :
            Linear equality vector.
        """
        return self.R, self.s, self.G, self.h, self.A, self.b

    @staticmethod
    def get_test(solver: str):
        """
        Get test function for a given solver.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        :
            Test function for that solver.
        """

        def test(self):
            R, s, G, h, A, b = self.get_problem()
            x = solve_ls(R, s, G, h, A, b, solver=solver)
            x_sp = solve_ls(R, s, G, h, A, b, solver=solver, sym_proj=True)
            self.assertIsNotNone(x)
            self.assertIsNotNone(x_sp)
            known_solution = array([2.0 / 3, -1.0 / 3, 2.0 / 3])
            sol_tolerance = 1e-5 if solver == "ecos" else 1e-6
            ineq_tolerance = 1e-7 if solver == "scs" else 1e-9
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(norm(x_sp - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestSolveLS, "test_{}".format(solver), TestSolveLS.get_test(solver)
    )


if __name__ == "__main__":
    unittest.main()
