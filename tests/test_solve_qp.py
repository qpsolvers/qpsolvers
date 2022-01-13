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

import scipy

from numpy import allclose, array, dot, ones, random
from numpy.linalg import norm
from scipy.sparse import csc_matrix

from qpsolvers import available_solvers, sparse_solvers
from qpsolvers import solve_qp, solve_safer_qp
from qpsolvers.exceptions import SolverNotFound


class TestSolveQP(unittest.TestCase):

    """
    Test fixture for the README example problem.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def get_dense_problem(self):
        """
        Get dense problem as a sextuple of values to unpack.

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
        M = array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        P = dot(M.T, M)  # this is a positive definite matrix
        q = dot(array([3.0, 2.0, 3.0]), M).reshape((3,))
        G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = array([3.0, 2.0, -2.0]).reshape((3,))
        A = array([1.0, 1.0, 1.0])
        b = array([1.0])
        return P, q, G, h, A, b

    def get_sparse_problem(self):
        """
        Get sparse problem as a quadruplet of values to unpack.

        Returns
        -------
        P : scipy.sparse.csc_matrix
            Symmetric quadratic-cost matrix .
        q : numpy.ndarray
            Quadratic-cost vector.
        G : scipy.sparse.csc_matrix
            Linear inequality matrix.
        h : numpy.ndarray
            Linear inequality vector.
        """
        n = 150
        M = scipy.sparse.lil_matrix(scipy.sparse.eye(n))
        for i in range(1, n - 1):
            M[i, i + 1] = -1
            M[i, i - 1] = 1
        P = csc_matrix(M.dot(M.transpose()))
        q = -ones((n,))
        G = csc_matrix(-scipy.sparse.eye(n))
        h = -2.0 * ones((n,))
        return P, q, G, h

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            x_sp = solve_qp(P, q, G, h, A, b, solver=solver, sym_proj=True)
            self.assertIsNotNone(x)
            self.assertIsNotNone(x_sp)
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            sol_tolerance = 1e-4 if solver == "ecos" else 1e-8
            ineq_tolerance = 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(norm(x_sp - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    @staticmethod
    def get_test_all_shapes(solver):
        """
        Get test function for a given solver. This variant tries all possible
        shapes for matrix and vector parameters.

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
            P, q, G, h, _, _ = self.get_dense_problem()
            h0 = array([h[0]])
            A = array([[1.0, 0.0, 0.0], [0.0, 0.4, 0.5]])
            b = array([-0.5, -1.2])
            b0 = array([b[0]])
            lb = array([-1.0, -1.0, -1.0])
            ub = array([+1.0, +1.0, +1.0])

            cases = [
                {"P": P, "q": q},
                {"P": P, "q": q, "G": G, "h": h},
                {"P": P, "q": q, "A": A, "b": b},
                {"P": P, "q": q, "G": G[0], "h": h0},
                {"P": P, "q": q, "A": A[0], "b": b0},
                {"P": P, "q": q, "G": G, "h": h, "A": A, "b": b},
                {"P": P, "q": q, "G": G[0], "h": h0, "A": A, "b": b},
                {"P": P, "q": q, "G": G, "h": h, "A": A[0], "b": b0},
                {"P": P, "q": q, "G": G[0], "h": h0, "A": A[0], "b": b0},
                {
                    "P": P,
                    "q": q,
                    "G": G[0],
                    "h": h0,
                    "A": A[0],
                    "b": b0,
                    "lb": lb,
                    "ub": ub,
                },
            ]

            for (i, case) in enumerate(cases):
                quadprog_solution = solve_qp(solver="quadprog", **case)
                for solver in available_solvers:
                    x = solve_qp(solver=solver, **case)
                    self.assertLess(norm(x - quadprog_solution), 2e-4)

        return test

    @staticmethod
    def get_test_bounds(solver):
        """
        Get test function for a given solver. This variant adds vector bounds.

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
            P, q, G, h, A, b = self.get_dense_problem()
            lb = array([-1.0, -2.0, -0.5])
            ub = array([1.0, -0.2, 1.0])
            x = solve_qp(P, q, G, h, A, b, lb, ub, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.41463415, -0.41463415, 1.0])
            sol_tolerance = 1e-6 if solver in ["ecos", "scs"] else 1e-8
            ineq_tolerance = 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.64705882, -1.17647059, -1.82352941])
            sol_tolerance = 1e-3 if solver == "ecos" else 1e-6
            self.assertLess(norm(x - known_solution), sol_tolerance)

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, G, h, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.49025721, -1.57755261, -0.66484801])
            sol_tolerance = 1e-3 if solver == "ecos" else 1e-6
            ineq_tolerance = 1e-7 if solver == "scs" else 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, A=A, b=b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.28026906, -1.55156951, 2.27130045])
            sol_tolerance = 1e-5 if solver in ["ecos", "scs"] else 1e-8
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    @staticmethod
    def get_test_one_ineq(solver):
        """
        Get test function for a given solver. In this variant, there is
        only one inequality constraint.

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
            P, q, G, h, A, b = self.get_dense_problem()
            G, h = G[1], h[1].reshape((1,))
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            sol_tolerance = (
                1e-5 if solver == "scs" else
                1e-6 if solver in ["cvxopt", "ecos"] else 1e-8
            )
            ineq_tolerance = 1e-7 if solver == "scs" else 1e-8
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(dot(G, x) - h, ineq_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    @staticmethod
    def get_test_safer(solver):
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
            P, q, G, h, _, _ = self.get_dense_problem()
            if solver in sparse_solvers:
                with self.assertRaises(AssertionError):
                    solve_safer_qp(P, q, G, h, sr=1e-4, solver=solver)
                return
            x = solve_safer_qp(P, q, G, h, sr=1e-4, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.49021915, -1.57749935, -0.66477954])
            sol_tolerance = 1e-4 if solver in ["ecos", "scs"] else 1e-6
            ineq_tolerance = 1e-7 if solver == "scs" else 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)

        return test

    @staticmethod
    def get_test_sparse(solver):
        """
        Get test function for a given solver. This variant tests a sparse
        problem.

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
            P, q, G, h = self.get_sparse_problem()
            x = solve_qp(P, q, G, h, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([2.0] * 149 + [3.0])
            sol_tolerance = 1e-3 if solver == "gurobi" else 1e-7
            ineq_tolerance = 1e-7
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(G * x - h), ineq_tolerance)

        return test

    @staticmethod
    def get_test_sparse_bounds(solver):
        """
        Get test function for a given solver. This variant tests a sparse
        problem with additional vector lower and upper bounds.

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
            P, q, G, h = self.get_sparse_problem()
            lb = +2.2 * ones(q.shape)
            ub = +2.4 * ones(q.shape)
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([2.2] * 149 + [2.4])
            sol_tolerance = 1e-3 if solver == "gurobi" else 1e-8
            ineq_tolerance = 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(G * x - h), ineq_tolerance)

        return test

    @staticmethod
    def get_test_sparse_unfeasible(solver):
        """
        Get test function for a given solver. This variant tests an unfeasible
        sparse problem with additional vector lower and upper bounds.

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
            P, q, G, h = self.get_sparse_problem()
            lb = +0.5 * ones(q.shape)
            ub = +1.5 * ones(q.shape)
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
            self.assertIsNone(x)

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
            P, q, G, h, A, b = self.get_dense_problem()
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
            ineq_tolerance = 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertTrue(allclose(dot(A, x), b))

        return test

    def test_solver_not_found(self):
        P, q, G, h, A, b = self.get_dense_problem()
        with self.assertRaises(SolverNotFound):
            solve_qp(P, q, G, h, A, b, solver="ideal")


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestSolveQP, "test_{}".format(solver), TestSolveQP.get_test(solver)
    )
    setattr(
        TestSolveQP,
        "test_all_shapes_{}".format(solver),
        TestSolveQP.get_test_all_shapes(solver),
    )
    setattr(
        TestSolveQP,
        "test_bounds_{}".format(solver),
        TestSolveQP.get_test_bounds(solver),
    )
    setattr(
        TestSolveQP,
        "test_no_cons_{}".format(solver),
        TestSolveQP.get_test_no_cons(solver),
    )
    setattr(
        TestSolveQP,
        "test_no_eq_{}".format(solver),
        TestSolveQP.get_test_no_eq(solver),
    )
    setattr(
        TestSolveQP,
        "test_no_ineq_{}".format(solver),
        TestSolveQP.get_test_no_ineq(solver),
    )
    setattr(
        TestSolveQP,
        "test_one_ineq_{}".format(solver),
        TestSolveQP.get_test_one_ineq(solver),
    )
    setattr(
        TestSolveQP,
        "test_safer_{}".format(solver),
        TestSolveQP.get_test_safer(solver),
    )
    if solver in sparse_solvers:
        setattr(
            TestSolveQP,
            "test_sparse_{}".format(solver),
            TestSolveQP.get_test_sparse(solver),
        )
        setattr(
            TestSolveQP,
            "test_sparse_bounds_{}".format(solver),
            TestSolveQP.get_test_sparse_bounds(solver),
        )
        setattr(
            TestSolveQP,
            "test_sparse_unfeasible_{}".format(solver),
            TestSolveQP.get_test_sparse_unfeasible(solver),
        )
    setattr(
        TestSolveQP,
        "test_warmstart_{}".format(solver),
        TestSolveQP.get_test_warmstart(solver),
    )


if __name__ == "__main__":
    unittest.main()
