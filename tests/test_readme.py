#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stéphane Caron <stephane.caron@normalesup.org>
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
import warnings

from numpy import allclose, array, dot, random
from numpy.linalg import norm
from qpsolvers import available_solvers, solve_qp


class ReadmeProblem(unittest.TestCase):

    """
    Test fixture for the README example problem.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        M = array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        self.P = dot(M.T, M)  # this is a positive definite matrix
        self.q = dot(array([3.0, 2.0, 3.0]), M).reshape((3,))
        self.G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        self.h = array([3.0, 2.0, -2.0]).reshape((3,))
        self.A = array([1.0, 1.0, 1.0])
        self.b = array([1.0])

    def get_problem(self):
        """
        Get problem as a sextuple of values to unpack.

        Returns
        -------
        P : numpy.ndarray
            Symmetric quadratic-cost matrix .
        q : numpy.ndarray
            Quadratic-cost vector.
        G : numpy.ndarray
            Linear inequality matrix.
        h : numpy.ndarray
            Linear inequality vector.
        A : numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix
            Linear equality matrix.
        b : numpy.ndarray
            Linear equality vector.
        """
        return self.P, self.q, self.G, self.h, self.A, self.b

    @staticmethod
    def get_test(solver):
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
            P, q, G, h, A, b = self.get_problem()
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            sol_tolerance = 1e-4 if solver == "ecos" else 1e-8
            self.assertTrue(norm(x - known_solution) < sol_tolerance)
            self.assertTrue(max(dot(G, x) - h) <= 1e-10)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    @staticmethod
    def get_test_no_cons(solver):
        """
        Get test function for a given solver. In this variant, there is
        no equality nor inequality constraint.

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
            P, q, G, h, A, b = self.get_problem()
            x = solve_qp(P, q, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.64705882, -1.17647059, -1.82352941])
            sol_tolerance = 1e-3 if solver == "ecos" else 1e-6
            self.assertTrue(norm(x - known_solution) < sol_tolerance)

        return test

    @staticmethod
    def get_test_no_eq(solver):
        """
        Get test function for a given solver. In this variant, there is
        no equality constraint.

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
            P, q, G, h, A, b = self.get_problem()
            x = solve_qp(P, q, G, h, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.49025721, -1.57755261, -0.66484801])
            sol_tolerance = 1e-3 if solver == "ecos" else 1e-6
            ineq_tolerance = 1e-7 if solver == "scs" else 1e-10
            self.assertTrue(norm(x - known_solution) < sol_tolerance)
            self.assertTrue(max(dot(G, x) - h) <= ineq_tolerance)

        return test

    @staticmethod
    def get_test_no_ineq(solver):
        """
        Get test function for a given solver. In this variant, there is
        no inequality constraint.

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
            P, q, G, h, A, b = self.get_problem()
            x = solve_qp(P, q, A=A, b=b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.28026906, -1.55156951, 2.27130045])
            sol_tolerance = 1e-5 if solver in ["ecos", "scs"] else 1e-8
            self.assertTrue(norm(x - known_solution) < sol_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    @staticmethod
    def get_test_warmstart(solver):
        """
        Get test function for a given solver. This variant warm starts.

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
            P, q, G, h, A, b = self.get_problem()
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            initvals = known_solution + 0.1 * random.random(3)
            x = solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                solver=solver,
                initvals=initvals,
                verbose=True,  # increases coverage
            )
            self.assertIsNotNone(x)
            sol_tolerance = 1e-4 if solver == "ecos" else 1e-8
            self.assertTrue(norm(x - known_solution) < sol_tolerance)
            self.assertTrue(max(dot(G, x) - h) <= 1e-10)
            self.assertTrue(allclose(dot(A, x), b))

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        ReadmeProblem, "test_{}".format(solver), ReadmeProblem.get_test(solver)
    )
    setattr(
        ReadmeProblem,
        "test_no_cons_{}".format(solver),
        ReadmeProblem.get_test_no_cons(solver),
    )
    setattr(
        ReadmeProblem,
        "test_no_eq_{}".format(solver),
        ReadmeProblem.get_test_no_eq(solver),
    )
    setattr(
        ReadmeProblem,
        "test_no_ineq_{}".format(solver),
        ReadmeProblem.get_test_no_ineq(solver),
    )
    setattr(
        ReadmeProblem,
        "test_warmstart_{}".format(solver),
        ReadmeProblem.get_test_warmstart(solver),
    )


if __name__ == "__main__":
    unittest.main()
