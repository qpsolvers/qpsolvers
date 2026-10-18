#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for the `solve_ls` function."""

import unittest
import warnings
from itertools import product
from unittest.mock import patch

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

from qpsolvers import available_solvers, solve_ls, solve_qp, sparse_solvers
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

    @unittest.skipUnless(
        any(s in available_solvers for s in ["proxqp", "scs"]),
        "ProxQP and SCS unavailable",
    )
    def test_sparse_warm_start(self):
        """Lift warm starts into residual-variable coordinates."""
        R = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
        expected = np.array([0.7, 0.2])
        s = R @ expected
        initial = np.array([0.4, 0.3])
        original = initial.copy()
        solvers = [s for s in ["proxqp", "scs"] if s in available_solvers]
        for solver, sparse, weighted in product(
            solvers, [False, True], [False, True]
        ):
            with self.subTest(solver=solver, sparse=sparse, weighted=weighted):
                with patch(
                    "qpsolvers.solve_ls.solve_qp", wraps=solve_qp
                ) as qp:
                    x = solve_ls(
                        spa.csc_matrix(R) if sparse else R,
                        s,
                        A=np.array([[1.0, 1.0]]),
                        b=np.array([0.9]),
                        lb=np.zeros(2),
                        ub=np.ones(2),
                        W=np.diag([1.0, 2.0, 3.0, 4.0]) if weighted else None,
                        initvals=initial,
                        sparse_conversion=True,
                        solver=solver,
                        eps_abs=1e-9,
                        eps_rel=0.0,
                    )
                np.testing.assert_allclose(x, expected, atol=1e-8)
                args, kwargs = qp.call_args
                guess = kwargs["initvals"]
                np.testing.assert_array_equal(guess[:2], original)
                # The added residual equations must hold at the guess.
                np.testing.assert_allclose(
                    args[4][-4:] @ guess, args[5][-4:], atol=1e-14
                )
                np.testing.assert_array_equal(initial, original)

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
