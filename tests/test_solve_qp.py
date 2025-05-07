#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for the `solve_qp` function."""

import unittest
import warnings

import numpy as np
import scipy
from numpy import array, dot, ones, random
from numpy.linalg import norm
from scipy.sparse import csc_matrix

from qpsolvers import (
    NoSolverSelected,
    ProblemError,
    SolverNotFound,
    available_solvers,
    solve_qp,
    sparse_solvers,
)

from .problems import get_qpmad_demo_problem

# Raising a ValueError when the problem is unbounded below is desired but not
# achieved by some solvers. Here are the behaviors observed as of March 2022.
# Unit tests only cover solvers that raise successfully:
behavior_on_unbounded = {
    "raise": ["cvxopt", "kvxopt", "ecos", "quadprog", "scs"],
    "return_crazy_solution": ["qpoases"],
    "return_none": ["osqp"],
}


class TestSolveQP(unittest.TestCase):
    """Test fixture for a variety of quadratic programs.

    Solver-specific tests are implemented in static methods called
    ``get_test_{foo}`` that return the test function for a given solver. The
    corresponding test function ``test_{foo}_{solver}`` is then added to the
    fixture below the class definition.
    """

    def setUp(self):
        """Prepare test fixture."""
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def get_dense_problem(self):
        """Get dense problem as a sextuple of values to unpack.

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
        """Get sparse problem as a quadruplet of values to unpack.

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
        """Check that NoSolverSelected is raised when applicable."""
        P, q, G, h, A, b = self.get_dense_problem()
        with self.assertRaises(NoSolverSelected):
            solve_qp(P, q, G, h, A, b, solver=None)

    def test_solver_not_found(self):
        """SolverNotFound is raised when the solver does not exist."""
        P, q, G, h, A, b = self.get_dense_problem()
        with self.assertRaises(SolverNotFound):
            solve_qp(P, q, G, h, A, b, solver="ideal")

    @staticmethod
    def get_test(solver: str):
        """Get test function for a given solver.

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            x_sp = solve_qp(P, q, G, h, A, b, solver=solver)
            self.assertIsNotNone(x, f"{solver=}")
            self.assertIsNotNone(x_sp, f"{solver=}")
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            sol_tolerance = (
                5e-4
                if solver in ["osqp", "qpalm", "scs"]
                else (
                    1e-4
                    if solver in ["ecos", "jaxopt_osqp"]
                    else 5e-6 if solver in ["proxqp", "qpax", "sip"] else 1e-8
                )
            )
            eq_tolerance = 1e-5 if solver in ["jaxopt_osqp", "sip"] else 1e-10
            ineq_tolerance = (
                2e-4
                if solver in ["jaxopt_osqp", "qpalm", "scs"]
                else 5e-6 if solver in ["proxqp", "qpax"] else 1e-10
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(
                norm(x_sp - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(G, x) - h), ineq_tolerance, f"{solver=}")
            self.assertLess(max(dot(A, x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(dot(A, x) - b), eq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_all_shapes(solver: str):
        """Get test function for a given solver.

        This variant tries all possible shapes for matrix and vector
        parameters.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        :
            Test function for that solver.

        Note
        ----
        This function uses DAQP to find groundtruth solutions.
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

            for i, test_case in enumerate(cases):
                no_inequality = "G" not in test_case or test_case["G"] is None
                if no_inequality and solver in ["qpswift", "qpax"]:
                    # QPs without inequality constraints are not handled by
                    # qpSWIFT or qpax
                    continue
                no_equality = "A" not in test_case or test_case["A"] is None
                if no_equality and solver in ["qpax"]:
                    # QPs without equality constraints not handled by qpax
                    continue
                has_one_equality = (
                    "A" in test_case
                    and test_case["A"] is not None
                    and test_case["A"].ndim == 1
                )
                has_lower_box = (
                    "lb" in test_case and test_case["lb"] is not None
                )
                if has_one_equality and has_lower_box and solver == "hpipm":
                    # Skipping this test for HPIPM for now
                    # See https://github.com/giaf/hpipm/issues/136
                    continue
                test_comp = {
                    k: v.shape if v is not None else "None"
                    for k, v in test_case.items()
                }
                daqp_solution = solve_qp(solver="daqp", **test_case)
                self.assertIsNotNone(
                    daqp_solution,
                    f"Baseline failed on parameters: {test_comp}",
                )
                solver_solution = solve_qp(solver=solver, **test_case)
                sol_tolerance = (
                    2e-2
                    if solver == "proxqp"
                    else (
                        5e-3
                        if solver in ["jaxopt_osqp"]
                        else (
                            2e-3
                            if solver in ["osqp", "qpalm", "scs"]
                            else 5e-4 if solver == "ecos" else 2e-4
                        )
                    )
                )
                self.assertLess(
                    norm(solver_solution - daqp_solution),
                    sol_tolerance,
                    f"Solver failed on parameters: {test_comp}",
                )

        return test

    @staticmethod
    def get_test_bounds(solver: str):
        """Get test function for a given solver.

        This variant adds vector bounds.

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
            P, q, G, h, A, b = self.get_dense_problem()
            lb = array([-1.0, -2.0, -0.5])
            ub = array([1.0, -0.2, 1.0])
            x = solve_qp(P, q, G, h, A, b, lb, ub, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.41463415, -0.41463415, 1.0])
            sol_tolerance = (
                2e-3
                if solver == "proxqp"
                else (
                    5e-3
                    if solver in ["jaxopt_osqp", "osqp"]
                    else (
                        5e-5
                        if solver == "scs"
                        else (
                            1e-6
                            if solver in ["qpalm", "ecos", "qpax", "sip"]
                            else 1e-8
                        )
                    )
                )
            )
            eq_tolerance = 1e-5 if solver in ["proxqp", "sip"] else 1e-10
            ineq_tolerance = 1e-5 if solver == "proxqp" else 1e-10
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(G, x) - h), ineq_tolerance, f"{solver=}")
            self.assertLess(max(dot(A, x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(dot(A, x) - b), eq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_no_cons(solver: str):
        """Get test function for a given solver.

        In this variant, there is no equality nor inequality constraint.

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.64705882, -1.17647059, -1.82352941])
            sol_tolerance = (
                1e-3
                if solver == "ecos"
                else 1e-5 if solver in ["osqp", "qpalm"] else 1e-6
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )

        return test

    @staticmethod
    def get_test_no_eq(solver: str):
        """Get test function for a given solver.

        In this variant, there is no equality constraint.

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, G, h, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([-0.49025721, -1.57755261, -0.66484801])
            sol_tolerance = (
                1e-3
                if solver in ["ecos", "jaxopt_osqp"]
                else (
                    1e-4
                    if solver in ["qpalm", "sip"]
                    else 1e-5 if solver == "osqp" else 1e-6
                )
            )
            ineq_tolerance = (
                1e-3
                if solver == "jaxopt_osqp"
                else (
                    1e-5
                    if solver == "osqp"
                    else (
                        1e-6
                        if solver == "proxqp"
                        else 1e-7 if solver == "scs" else 1e-10
                    )
                )
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(G, x) - h), ineq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_no_ineq(solver: str):
        """Get test function for a given solver.

        In this variant, there is no inequality constraint.

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
            P, q, G, h, A, b = self.get_dense_problem()
            x = solve_qp(P, q, A=A, b=b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.28026906, -1.55156951, 2.27130045])
            sol_tolerance = (
                1e-3
                if solver in ["jaxopt_osqp", "osqp", "qpalm"]
                else (
                    1e-5
                    if solver in ["ecos", "scs", "qpax"]
                    else (
                        1e-6
                        if solver == "highs"
                        else 1e-7 if solver == "proxqp" else 1e-8
                    )
                )
            )
            eq_tolerance = (
                1e-5
                if solver == "jaxopt_osqp"
                else (
                    1e-6
                    if solver in ["qpax", "sip"]
                    else 1e-7 if solver == "osqp" else 1e-9
                )
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(A, x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(dot(A, x) - b), eq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_one_ineq(solver: str):
        """Get test function for a given solver.

        In this variant, there is only one inequality constraint.

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
            P, q, G, h, A, b = self.get_dense_problem()
            G, h = G[1], h[1].reshape((1,))
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([0.30769231, -0.69230769, 1.38461538])
            sol_tolerance = (
                1e-3
                if solver in ["jaxopt_osqp", "qpalm"]
                else (
                    5e-4
                    if solver == "osqp"
                    else (
                        1e-5
                        if solver == "scs"
                        else (
                            5e-6
                            if solver in ["proxqp", "sip"]
                            else (
                                1e-6
                                if solver
                                in ["cvxopt", "kvxopt", "ecos", "qpax"]
                                else 5e-8 if solver == "qpswift" else 1e-8
                            )
                        )
                    )
                )
            )
            eq_tolerance = (
                1e-3
                if solver in ["jaxopt_osqp"]
                else (
                    1e-4
                    if solver in ["qpalm", "qpax", "sip"]
                    else 1e-8 if solver in ["osqp", "scs"] else 1e-10
                )
            )
            ineq_tolerance = (
                1e-5
                if solver == "proxqp"
                else (
                    1e-6
                    if solver == "qpax"
                    else (1e-7 if solver in ["qpswift", "scs"] else 1e-8)
                )
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(G, x) - h), ineq_tolerance, f"{solver=}")
            self.assertLess(max(dot(A, x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(dot(A, x) - b), eq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_sparse(solver: str):
        """Get test function for a given solver.

        This variant tests a sparse problem.

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
            P, q, G, h = self.get_sparse_problem()
            kwargs = {}
            tol_solvers = ("osqp", "proxqp", "qpalm", "scs")
            if solver in tol_solvers:
                kwargs["eps_abs"] = 2e-4
            x = solve_qp(P, q, G, h, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([2.0] * 149 + [3.0])
            sol_tolerance = (
                5e-3
                if solver == "cvxopt" or solver == "kvxopt"
                else (
                    2e-3
                    if solver in ["osqp", "qpalm", "qpax"]
                    else (
                        1e-3
                        if solver in ["gurobi", "piqp"]
                        else (
                            5e-4
                            if solver in ["clarabel", "mosek"]
                            else (
                                1e-4
                                if solver in ["scs", "sip"]
                                else (
                                    2e-5
                                    if solver == "proxqp"
                                    else 1e-6 if solver == "highs" else 1e-7
                                )
                            )
                        )
                    )
                )
            )
            ineq_tolerance = 1e-4 if solver in tol_solvers else 1e-7
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(G * x - h), ineq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_sparse_bounds(solver: str):
        """Get test function for a given solver.

        This variant tests a sparse problem with additional vector lower and
        upper bounds.

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
            P, q, G, h = self.get_sparse_problem()
            lb = +2.2 * ones(q.shape)
            ub = +2.4 * ones(q.shape)
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
            self.assertIsNotNone(x)
            known_solution = array([2.2] * 149 + [2.4])
            sol_tolerance = (
                1e-3
                if solver in ["gurobi", "osqp", "qpalm", "sip"]
                else (
                    5e-6
                    if solver in ["mosek", "proxqp"]
                    else (
                        1e-7 if solver in ["cvxopt", "kvxopt", "scs"] else 1e-8
                    )
                )
            )
            ineq_tolerance = 1e-10
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(G * x - h), ineq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_sparse_unfeasible(solver: str):
        """Get test function for a given solver.

        This variant tests an unfeasible sparse problem with additional vector
        lower and upper bounds.

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
            P, q, G, h = self.get_sparse_problem()
            lb = +0.5 * ones(q.shape)
            ub = +1.5 * ones(q.shape)
            if solver == "cvxopt" or solver == "kvxopt":
                # Skipping this test for CVXOPT and KVXOPT for now
                # See https://github.com/cvxopt/cvxopt/issues/229
                return
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
            self.assertIsNone(x)

        return test

    @staticmethod
    def get_test_warmstart(solver: str):
        """Get test function for a given solver.

        This variant warm starts.

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
                1e-3
                if solver in ["osqp", "qpalm", "scs"]
                else (
                    1e-4
                    if solver in ["ecos", "jaxopt_osqp"]
                    else 5e-6 if solver in ["proxqp", "qpax", "sip"] else 1e-8
                )
            )
            eq_tolerance = (
                1e-4 if solver in ["jaxopt_osqp", "osqp", "sip"] else 1e-10
            )
            ineq_tolerance = (
                1e-3
                if solver in ["osqp", "qpalm", "scs"]
                else (
                    1e-5
                    if solver in ["jaxopt_osqp", "proxqp", "qpax"]
                    else 1e-10
                )
            )
            self.assertLess(
                norm(x - known_solution), sol_tolerance, f"{solver=}"
            )
            self.assertLess(max(dot(G, x) - h), ineq_tolerance, f"{solver=}")
            self.assertLess(max(dot(A, x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(dot(A, x) - b), eq_tolerance, f"{solver=}")

        return test

    @staticmethod
    def get_test_raise_on_unbounded_below(solver: str):
        """ValueError is raised when the problem is unbounded below.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        :
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
            with self.assertRaises(ProblemError):
                solve_qp(P, q, solver=solver)

        return test

    @staticmethod
    def get_test_qpmad_demo(solver: str):
        """Get test function for a given solver.

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
            problem = get_qpmad_demo_problem()
            P, q, G, h, _, _, lb, ub = problem.unpack()
            x = solve_qp(P, q, G, h, lb=lb, ub=ub, solver=solver)
            known_solution = array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                    -0.71875,
                ]
            )
            sol_tolerance = (
                1e-2
                if solver in ["jaxopt_osqp", "osqp"]
                else (
                    5e-4
                    if solver in ["qpalm", "scs", "qpax"]
                    else (
                        2e-5
                        if solver == "proxqp"
                        else (
                            1e-6
                            if solver
                            in [
                                "cvxopt",
                                "kvxopt",
                                "mosek",
                                "qpswift",
                                "piqp",
                                "sip",
                            ]
                            else 1e-8
                        )
                    )
                )
            )
            self.assertIsNotNone(x)
            self.assertLess(np.linalg.norm(x - known_solution), sol_tolerance)

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
    if solver not in ["qpswift"]:
        # qpSWIFT: https://github.com/qpSWIFT/qpSWIFT/issues/2
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
    if solver not in ["qpswift"]:
        # qpSWIFT: https://github.com/qpSWIFT/qpSWIFT/issues/2
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
    if solver in behavior_on_unbounded["raise"]:
        setattr(
            TestSolveQP,
            f"test_raise_on_unbounded_below_{solver}",
            TestSolveQP.get_test_raise_on_unbounded_below(solver),
        )
    setattr(
        TestSolveQP,
        f"test_qpmad_demo_{solver}",
        TestSolveQP.get_test_qpmad_demo(solver),
    )
