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
from numpy import array, dot, ones, random
from numpy.linalg import norm
from scipy.sparse import csc_matrix

from qpsolvers import (
    available_solvers,
    dense_solvers,
    solve_qp,
    solve_safer_qp,
    sparse_solvers,
)
from qpsolvers.exceptions import NoSolverSelected, SolverNotFound

# Raising a ValueError when the problem is unbounded below is desired but not
# achieved by some solvers. Here are the behaviors observed as of March 2022.
# Unit tests only cover solvers that raise successfully:
behavior_on_unbounded = {
    "raise_value_error": ["cvxopt", "ecos", "quadprog", "scs"],
    "return_crazy_solution": ["qpoases"],
    "return_none": ["osqp"],
}


class TestSolveQP(unittest.TestCase):

    """
    Test fixture for a variety of quadratic programs.

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

    def get_dense_problem(self):
        """
        Get dense problem as a sextuple of values to unpack.

        Returns
        -------
        P : numpy.ndarray
            Symmetric cost matrix .
        q : numpy.ndarray
            Cost vector.
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
            Symmetric cost matrix.
        q : numpy.ndarray
            Cost vector.
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

    def test_no_solver_selected(self):
        """
        Check that NoSolverSelected is raised when applicable.
        """
        P, q, G, h, A, b = self.get_dense_problem()
        with self.assertRaises(NoSolverSelected):
            solve_qp(P, q, G, h, A, b, solver=None)

    def test_solver_not_found(self):
        """
        Check that SolverNotFound is raised when the solver does not exist.
        """
        P, q, G, h, A, b = self.get_dense_problem()
        with self.assertRaises(SolverNotFound):
            solve_qp(P, q, G, h, A, b, solver="ideal")

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
            sol_tolerance = (
                2e-4
                if solver == "osqp"
                else 5e-4
                if solver == "scs"
                else 1e-4
                if solver == "ecos"
                else 5e-6
                if solver == "proxqp"
                else 1e-8
            )
            eq_tolerance = 1e-10
            ineq_tolerance = (
                2e-4
                if solver == "scs"
                else 5e-6
                if solver == "proxqp"
                else 1e-10
            )
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(norm(x_sp - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertLess(max(dot(A, x) - b), eq_tolerance)
            self.assertLess(min(dot(A, x) - b), eq_tolerance)

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

        Note
        ----
        This function relies on "quadprog" to find groundtruth solutions.
        """

        def test(self):
            P, q, G, h, _, _ = self.get_dense_problem()
            A = array([[1.0, 0.0, 0.0], [0.0, 0.4, 0.5]])
            b = array([-0.5, -1.2])
            lb = array([-0.5, -2, -0.8])
            ub = array([+1.0, +1.0, +1.0])

            ineq_variants = ((None, None), (G, h), (G[0], array([h[0]])))
            eq_variants = ((None, None), (A, b), (A[0], array([b[0]])))
            box_variants = ((None, None), (lb, None), (None, ub), (lb, ub))
            cases = [
                {
                    "P": P,
                    "q": q,
                    "G": G_case,
                    "h": h_case,
                    "A": A_case,
                    "b": b_case,
                    "lb": lb_case,
                    "ub": ub_case,
                }
                for (G_case, h_case) in ineq_variants
                for (A_case, b_case) in eq_variants
                for (lb_case, ub_case) in box_variants
            ]

            for (i, test_case) in enumerate(cases):
                no_inequality = "G" not in test_case or test_case["G"] is None
                if no_inequality and solver == "qpswift":
                    # QPs without inequality constraints not handled by qpSWIFT
                    continue
                test_comp = {
                    k: v.shape if v is not None else "None"
                    for k, v in test_case.items()
                }
                quadprog_solution = solve_qp(solver="quadprog", **test_case)
                self.assertIsNotNone(
                    quadprog_solution,
                    f"Baseline failed on parameters: {test_comp}",
                )
                solver_solution = solve_qp(solver=solver, **test_case)
                sol_tolerance = (
                    2e-2
                    if solver == "proxqp"
                    else 1e-3
                    if solver == "scs"
                    else 2e-3
                    if solver == "osqp"
                    else 5e-4
                    if solver == "ecos"
                    else 2e-4
                )
                self.assertLess(
                    norm(solver_solution - quadprog_solution),
                    sol_tolerance,
                    f"Solver failed on parameters: {test_comp}",
                )

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
            sol_tolerance = (
                2e-3
                if solver == "proxqp"
                else 5e-3
                if solver == "osqp"
                else 5e-5
                if solver == "scs"
                else 1e-6
                if solver == "ecos"
                else 1e-8
            )
            eq_tolerance = 1e-5 if solver == "proxqp" else 1e-10
            ineq_tolerance = 1e-5 if solver == "proxqp" else 1e-10
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertLess(max(dot(A, x) - b), eq_tolerance)
            self.assertLess(min(dot(A, x) - b), eq_tolerance)

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
            sol_tolerance = (
                1e-3
                if solver == "ecos"
                else 1e-5
                if solver == "osqp"
                else 1e-6
            )
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
            sol_tolerance = (
                1e-3
                if solver == "ecos"
                else 2e-6
                if solver == "osqp"
                else 1e-6
            )
            ineq_tolerance = (
                2e-6
                if solver == "osqp"
                else 1e-6
                if solver == "proxqp"
                else 1e-7
                if solver == "scs"
                else 1e-10
            )
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
            sol_tolerance = (
                5e-4
                if solver == "osqp"
                else 1e-5
                if solver in ["ecos", "scs"]
                else 1e-6
                if solver == "highs"
                else 1e-7
                if solver == "proxqp"
                else 1e-8
            )
            eq_tolerance = 1e-9
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(A, x) - b), eq_tolerance)
            self.assertLess(min(dot(A, x) - b), eq_tolerance)

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
                5e-4
                if solver == "osqp"
                else 1e-5
                if solver == "scs"
                else 5e-6
                if solver == "proxqp"
                else 1e-6
                if solver in ["cvxopt", "ecos"]
                else 5e-8
                if solver == "qpswift"
                else 1e-8
            )
            eq_tolerance = 5e-10 if solver in ["osqp", "scs"] else 1e-10
            ineq_tolerance = (
                5e-6
                if solver == "proxqp"
                else 1e-7
                if solver == "scs"
                else 2e-8
                if solver == "qpswift"
                else 1e-8
            )
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertLess(max(dot(A, x) - b), eq_tolerance)
            self.assertLess(min(dot(A, x) - b), eq_tolerance)

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
            if solver not in dense_solvers:
                with self.assertRaises(NotImplementedError):
                    solve_safer_qp(P, q, G, h, sr=1e-4, solver=solver)
                return
            x = solve_safer_qp(P, q, G, h, sr=1e-4, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.49021915, -1.57749935, -0.66477954])
            sol_tolerance = (
                2e-3
                if solver == "proxqp"  # test params not applied
                else 1e-4
                if solver in ["ecos", "scs"]
                else 1e-6
            )
            ineq_tolerance = (
                1e-7
                if solver == "scs"
                else 1e-6
                if solver == "proxqp"
                else 1e-10
            )
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
            sol_tolerance = (
                5e-3
                if solver == "cvxopt"
                else 1e-3
                if solver == "gurobi"
                else 5e-4
                if solver == "osqp"
                else 1e-4
                if solver == "scs"
                else 2e-5
                if solver == "proxqp"
                else 1e-6
                if solver == "highs"
                else 1e-7
            )
            ineq_tolerance = (
                1e-4
                if solver == "scs"
                else 2e-5
                if solver == "proxqp"
                else 5e-7
                if solver == "osqp"
                else 1e-7
            )
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
            sol_tolerance = (
                1e-3
                if solver == "gurobi"
                else 1e-3
                if solver == "osqp"
                else 5e-6
                if solver == "proxqp"
                else 1e-7
                if solver in ["cvxopt", "scs"]
                else 1e-8
            )
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
            if solver == "cvxopt":
                # Skipping this test for CVXOPT for now
                # See https://github.com/cvxopt/cvxopt/issues/229
                return
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
            sol_tolerance = (
                5e-4
                if solver == "scs"
                else 2e-4
                if solver == "osqp"
                else 1e-4
                if solver == "ecos"
                else 5e-6
                if solver == "proxqp"
                else 1e-8
            )
            eq_tolerance = 1e-4 if solver == "osqp" else 1e-10
            ineq_tolerance = (
                2e-4
                if solver in ["osqp", "scs"]
                else 5e-6
                if solver == "proxqp"
                else 1e-10
            )
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(dot(G, x) - h), ineq_tolerance)
            self.assertLess(max(dot(A, x) - b), eq_tolerance)
            self.assertLess(min(dot(A, x) - b), eq_tolerance)

        return test

    @staticmethod
    def get_test_raise_on_unbounded_below(solver):
        """
        Check that a ValueError is raised when the problem is unbounded below.

        Parameters
        ----------
        solver : string
            Name of the solver to test.

        Returns
        -------
        test : function
            Test function for that solver.

        Notes
        -----
        Detecting non-convexity is not a trivial problem and most solvers leave
        it to the user. See for instance the `recommendation from OSQP
        <https://osqp.org/docs/interfaces/status_values.html#status-values>`_.
        We only run this test for functions that successfully detect unbounded
        problems when the eigenvalues of :math:`P` are close to zero.
        """

        def test(self):
            v = array([5.4, -1.2, -1e-2, 1e4])
            P = dot(v.reshape(4, 1), v.reshape(1, 4))
            q = array([-1.0, -2, 0, 3e-4])
            # q is in the nullspace of P, so the problem is unbounded below
            with self.assertRaises(ValueError):
                solve_qp(P, q, solver=solver)

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(TestSolveQP, f"test_{solver}", TestSolveQP.get_test(solver))
    setattr(
        TestSolveQP,
        f"test_all_shapes_{solver}",
        TestSolveQP.get_test_all_shapes(solver),
    )
    setattr(
        TestSolveQP,
        f"test_bounds_{solver}",
        TestSolveQP.get_test_bounds(solver),
    )
    if solver != "qpswift":
        # QPs without inequality constraints are not handled by qpSWIFT
        setattr(
            TestSolveQP,
            f"test_no_cons_{solver}",
            TestSolveQP.get_test_no_cons(solver),
        )
    setattr(
        TestSolveQP,
        f"test_no_eq_{solver}",
        TestSolveQP.get_test_no_eq(solver),
    )
    if solver != "qpswift":
        # QPs without inequality constraints are not handled by qpSWIFT
        setattr(
            TestSolveQP,
            f"test_no_ineq_{solver}",
            TestSolveQP.get_test_no_ineq(solver),
        )
    setattr(
        TestSolveQP,
        f"test_one_ineq_{solver}",
        TestSolveQP.get_test_one_ineq(solver),
    )
    setattr(
        TestSolveQP,
        f"test_safer_{solver}",
        TestSolveQP.get_test_safer(solver),
    )
    if solver in sparse_solvers:
        setattr(
            TestSolveQP,
            f"test_sparse_{solver}",
            TestSolveQP.get_test_sparse(solver),
        )
        setattr(
            TestSolveQP,
            f"test_sparse_bounds_{solver}",
            TestSolveQP.get_test_sparse_bounds(solver),
        )
        setattr(
            TestSolveQP,
            f"test_sparse_unfeasible_{solver}",
            TestSolveQP.get_test_sparse_unfeasible(solver),
        )
    setattr(
        TestSolveQP,
        f"test_warmstart_{solver}",
        TestSolveQP.get_test_warmstart(solver),
    )
    if solver in behavior_on_unbounded["raise_value_error"]:
        setattr(
            TestSolveQP,
            f"test_raise_on_unbounded_below_{solver}",
            TestSolveQP.get_test_raise_on_unbounded_below(solver),
        )
