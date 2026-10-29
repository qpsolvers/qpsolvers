#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the `solve_ls` function."""

import unittest
import warnings

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

from qpsolvers import available_solvers, solve_ls, sparse_solvers
from qpsolvers.exceptions import NoSolverSelected, SolverNotFound
from qpsolvers.problems import get_sparse_least_squares


class TestSolveLS(unittest.TestCase):
    def setUp(self):
        """Prepare test fixture."""
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def get_problem_and_solution(self):
        """Get least-squares problem and its primal solution.

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
        solution :
            Known solution.
        """
        R = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 4.0], [0.0, 4.0, 1.0]])
        s = np.array([3.0, 2.0, 3.0])
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        A = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0])
        solution = np.array([2.0 / 3, -1.0 / 3, 2.0 / 3])
        return R, s, G, h, A, b, solution

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
            R, s, G, h, A, b, solution = self.get_problem_and_solution()
            kwargs = {"eps_abs": 1e-9, "eps_rel": 0.0} if solver == "sip" else {}
            x = solve_ls(
                R, s, G, h, A, b, solver=solver, sparse_conversion=False, **kwargs
            )
            x_sp = solve_ls(
                R, s, G, h, A, b, solver=solver, sparse_conversion=False, **kwargs
            )
            self.assertIsNotNone(x, f"{solver=}")
            self.assertIsNotNone(x_sp, f"{solver=}")
            sol_tolerance = (
                5e-3
                if solver in ["jaxopt_osqp", "osqp"]
                else (
                    2e-5
                    if solver == "proxqp"
                    else (
                        1e-5
                        if solver in ["ecos", "qpalm", "qpax"]
                        else 1e-6
                    )
                )
            )
            eq_tolerance = (
                1e-4
                if solver == "jaxopt_osqp"
                else (
                    2e-6
                    if solver in ["qpalm", "qpax"]
                    else 1e-7 if solver in ["osqp", "qtqp"] else 1e-9
                )
            )
            ineq_tolerance = (
                1e-3
                if solver == "osqp"
                else (
                    1e-5
                    if solver == "proxqp"
                    else (
                        2e-7
                        if solver in ["scs", "qpax"]
                        else 1e-8 if solver == "qtqp" else 1e-9
                    )
                )
            )
            self.assertLess(norm(x - solution), sol_tolerance, f"{solver=}")
            self.assertLess(norm(x_sp - solution), sol_tolerance, f"{solver=}")
            self.assertLess(max(G.dot(x) - h), ineq_tolerance, f"{solver=}")
            self.assertLess(max(A.dot(x) - b), eq_tolerance, f"{solver=}")
            self.assertLess(min(A.dot(x) - b), eq_tolerance, f"{solver=}")

        return test

    def test_no_solver_selected(self):
        """Check that NoSolverSelected is raised when applicable."""
        R, s, G, h, A, b, _ = self.get_problem_and_solution()
        with self.assertRaises(NoSolverSelected):
            solve_ls(R, s, G, h, A, b, solver=None)

    def test_solver_not_found(self):
        """SolverNotFound is raised when the solver does not exist."""
        R, s, G, h, A, b, _ = self.get_problem_and_solution()
        with self.assertRaises(SolverNotFound):
            solve_ls(R, s, G, h, A, b, solver="ideal")

    def test_sparse_warm_start(self):
        """Warm-start the sparse conversion with column vectors."""
        R = spa.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
        s = np.array([[0.7], [0.2], [0.9], [0.5]])
        initvals = np.array([[0.4], [0.3]])
        for solver in ["proxqp", "scs"]:
            if solver not in available_solvers:
                continue
            with self.subTest(solver=solver):
                x = solve_ls(
                    R, s, lb=np.zeros(2), initvals=initvals, solver=solver
                )
                self.assertIsNotNone(x)
                self.assertLess(norm(x - np.array([0.7, 0.2])), 1e-4)

    @staticmethod
    def get_test_integer_normal_equations(solver: str):
        """Check integer products and negation before solver conversion."""

        def test(self):
            for dtype in (np.uint8, np.int8):
                for sparse in (False, True) if solver in sparse_solvers else (False,):
                    for weighted in (False, True):
                        with self.subTest(
                            dtype=dtype, sparse=sparse, weighted=weighted
                        ):
                            R = np.array(
                                [[10, 0], [0, 20], [10, 0], [0, 20]], dtype=dtype
                            )
                            s = np.array([20, 60, 40, 100], dtype=dtype)
                            W = (
                                np.diag(np.array([1, 2, 3, 4], dtype=dtype))
                                if weighted
                                else None
                            )
                            if sparse:
                                R = spa.csc_matrix(R)
                            R_original, s_original = R.copy(), s.copy()
                            W_original = W.copy() if W is not None else None
                            kwargs = (
                                {"eps_abs": 1e-8, "eps_rel": 1e-8}
                                if solver == "osqp"
                                else {}
                            )
                            if solver == "ecos":
                                kwargs = {
                                    "abstol": 1e-10,
                                    "reltol": 1e-10,
                                    "feastol": 1e-10,
                                }
                            x = solve_ls(
                                R,
                                s,
                                lb=np.zeros(2),
                                ub=np.full(2, 10.0),
                                W=W,
                                solver=solver,
                                sparse_conversion=False,
                                **kwargs,
                            )
                            # Independent weighted means for the two columns.
                            expected = [3.5, 13 / 3] if weighted else [3.0, 4.0]
                            self.assertIsNotNone(x)
                            np.testing.assert_allclose(x, expected, atol=5e-3, rtol=0)
                            if sparse:
                                self.assertEqual((R != R_original).nnz, 0)
                            else:
                                np.testing.assert_array_equal(R, R_original)
                            np.testing.assert_array_equal(s, s_original)
                            if W is not None:
                                np.testing.assert_array_equal(W, W_original)

        return test

    @staticmethod
    def get_test_mixed_sparse_args(solver: str):
        """Get test function for mixed sparse problems with a given solver.

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
            _, s, G, h, A, b, _ = self.get_problem_and_solution()
            n = len(s)

            R_csc = spa.eye(n, format="csc")
            x_csc = solve_ls(
                R_csc, s, G, h, A, b, solver=solver, sparse_conversion=False
            )
            self.assertIsNotNone(x_csc, f"{solver=}")

            R_dia = spa.eye(n)
            x_dia = solve_ls(
                R_dia, s, G, h, A, b, solver=solver, sparse_conversion=False
            )
            self.assertIsNotNone(x_dia, f"{solver=}")

            x_np_dia = solve_ls(
                R_dia,
                s,
                G,
                h,
                A,
                b,
                W=np.eye(n),
                solver=solver,
                sparse_conversion=False,
            )
            self.assertIsNotNone(x_np_dia, f"{solver=}")

            sol_tolerance = 1e-8
            self.assertLess(norm(x_csc - x_dia), sol_tolerance, f"{solver=}")
            self.assertLess(
                norm(x_csc - x_np_dia), sol_tolerance, f"{solver=}"
            )

        return test

    @staticmethod
    def get_test_medium_sparse(solver: str, sparse_conversion: bool, **kwargs):
        """Get test function for a large sparse problem with a given solver.

        Parameters
        ----------
        solver :
            Name of the solver to test.
        sparse_conversion :
            Conversion strategy boolean.

        Returns
        -------
        :
            Test function for that solver.
        """

        def test(self):
            R, s, G, h, A, b, _, _ = get_sparse_least_squares(n=1500)
            x = solve_ls(
                R,
                s,
                G,
                h,
                A,
                b,
                solver=solver,
                sparse_conversion=sparse_conversion,
                **kwargs,
            )
            self.assertIsNotNone(x, f"{solver=}")

        return test

    @staticmethod
    def get_test_large_sparse(
        solver: str,
        sparse_conversion: bool,
        obj_scaling: float = 1.0,
        **kwargs,
    ):
        """Get test function for a large sparse problem with a given solver.

        Parameters
        ----------
        solver :
            Name of the solver to test.
        sparse_conversion :
            Whether to perform sparse or dense LS-to-QP conversion.
        obj_scaling:
            Scale objective matrices by this factor. Suitable values can help
            solvers that don't compute a preconditioner internally.

        Returns
        -------
        :
            Test function for that solver.
        """

        def test(self):
            R, s, G, h, A, b, _, _ = get_sparse_least_squares(n=15_000)
            x = solve_ls(
                obj_scaling * R,
                obj_scaling * s,
                G,
                h,
                A,
                b,
                solver=solver,
                sparse_conversion=sparse_conversion,
                **kwargs,
            )
            self.assertIsNotNone(x, f"{solver=}")

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestSolveLS, "test_{}".format(solver), TestSolveLS.get_test(solver)
    )
    setattr(
        TestSolveLS,
        "test_integer_normal_equations_{}".format(solver),
        TestSolveLS.get_test_integer_normal_equations(solver),
    )

for solver in sparse_solvers:
    setattr(
        TestSolveLS,
        "test_mixed_sparse_args_{}".format(solver),
        TestSolveLS.get_test_mixed_sparse_args(solver),
    )

for solver in sparse_solvers:  # loop complexity warning ;p
    if solver not in ["gurobi", "qtqp"]:
        # Gurobi: model too large for size-limited license
        # QTQP: slow convergence on medium/large problems (pure Python)
        kwargs = {}
        if solver == "mosek":
            try:
                import mosek

                kwargs["mosek"] = {mosek.dparam.intpnt_qo_tol_rel_gap: 1e-6}
            except ImportError:
                pass
        if solver != "copt":
            # COPT: Size limitation for non-commercial use
            setattr(
                TestSolveLS,
                "test_medium_sparse_dense_conversion_{}".format(solver),
                TestSolveLS.get_test_medium_sparse(
                    solver, sparse_conversion=False, **kwargs
                ),
            )

for solver in sparse_solvers:  # loop complexity warning ;p
    if solver not in ["copt", "cvxopt", "kvxopt", "gurobi", "qtqp"]:
        # COPT: Size limitation for non-commercial use
        # CVXOPT and KVXOPT: sparse conversion breaks rank assumption
        # Gurobi: model too large for size-limited license
        # QTQP: slow convergence on medium/large problems (pure Python)
        setattr(
            TestSolveLS,
            "test_medium_sparse_sparse_conversion_{}".format(solver),
            TestSolveLS.get_test_medium_sparse(solver, sparse_conversion=True),
        )

for solver in sparse_solvers:  # loop complexity warning ;p
    if solver not in ["gurobi", "highs", "qtqp"]:
        # Gurobi: model too large for size-limited license
        # HiGHS: model too large https://github.com/ERGO-Code/HiGHS/issues/992
        # QTQP: slow convergence on large problems (pure Python implementation)
        kwargs = {"eps_infeas": 1e-12} if solver == "scs" else {}
        if solver == "mosek":
            try:
                import mosek

                kwargs["mosek"] = {mosek.dparam.intpnt_qo_tol_rel_gap: 1e-7}
            except ImportError:
                pass
        if solver != "copt":
            # COPT: Size limitation for non-commercial use
            setattr(
                TestSolveLS,
                "test_large_sparse_problem_dense_conversion_{}".format(solver),
                TestSolveLS.get_test_large_sparse(
                    solver,
                    sparse_conversion=False,
                    obj_scaling=1e-3 if solver == "mosek" else 1.0,
                    **kwargs,
                ),
            )
        if solver not in ["copt", "cvxopt", "kvxopt"]:
            # COPT: Size limitation for non-commercial use
            # CVXOPT and KVXOPT: sparse conversion breaks rank assumption
            setattr(
                TestSolveLS,
                "test_large_sparse_problem_sparse_conversion_{}".format(
                    solver
                ),
                TestSolveLS.get_test_large_sparse(
                    solver, sparse_conversion=True, **kwargs
                ),
            )
