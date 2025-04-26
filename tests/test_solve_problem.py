#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2023 Inria

"""Unit tests for the `solve_problem` function."""

import unittest

import numpy as np
from numpy.linalg import norm

from qpsolvers import available_solvers, solve_problem
from qpsolvers.problems import (
    get_qpgurabs,
    get_qpgurdu,
    get_qpgureq,
    get_qpsut01,
    get_qpsut02,
    get_qpsut03,
    get_qpsut04,
    get_qpsut05,
    get_qptest,
)


class TestSolveProblem(unittest.TestCase):
    """Test fixture for primal and dual solutions of a variety of problems.

    Notes
    -----
    Solver-specific tests are implemented in static methods called
    ``get_test_{foo}`` that return the test function for a given solver. The
    corresponding test function ``test_{foo}_{solver}`` is then added to the
    fixture below the class definition.
    """

    @staticmethod
    def get_test_qpsut01(solver: str):
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
            problem, ref_solution = get_qpsut01()
            solution = solve_problem(problem, solver=solver)
            eps_abs = (
                5e-1
                if solver in ["jaxopt_osqp", "osqp", "qpalm"]
                else (
                    5e-3
                    if solver == "proxqp"
                    else (
                        1e-4
                        if solver == "ecos"
                        else (
                            5e-5
                            if solver in ["mosek", "qpax", "sip"]
                            else (
                                1e-6
                                if solver in ["cvxopt", "kvxopt", "qpswift", "scs"]
                                else 5e-7 if solver in ["gurobi"] else 1e-7
                            )
                        )
                    )
                )
            )
            self.assertLess(
                norm(solution.x - ref_solution.x), eps_abs, f"{solver=}"
            )
            # NB: in general the dual solution is not unique (that's why the
            # other tests check residuals). This test only works because the
            # dual solution is unique in this particular problem.
            self.assertLess(
                norm(solution.y - ref_solution.y), eps_abs, f"{solver=}"
            )
            self.assertLess(
                norm(solution.z - ref_solution.z), eps_abs, f"{solver=}"
            )
            self.assertLess(
                norm(solution.z_box - ref_solution.z_box),
                eps_abs,
                f"{solver=}",
            )

        return test

    @staticmethod
    def get_test_qpsut02(solver: str):
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
            problem, ref_solution = get_qpsut02()
            solution = solve_problem(problem, solver=solver)
            eps_abs = (
                5e-2
                if solver in ["ecos", "jaxopt_osqp", "qpalm"]
                else (
                    5e-4
                    if solver in ["proxqp", "scs", "qpax"]
                    else (
                        1e-4
                        if solver in ["cvxopt", "kvxopt", "qpax"]
                        else (
                            1e-5
                            if solver in ["highs", "osqp"]
                            else (
                                5e-7
                                if solver
                                in ["clarabel", "mosek", "qpswift", "piqp", "sip"]
                                else 1e-7 if solver in ["gurobi"] else 1e-8
                            )
                        )
                    )
                )
            )
            self.assertLess(
                norm(solution.x - ref_solution.x), eps_abs, f"{solver=}"
            )
            self.assertLess(solution.primal_residual(), eps_abs, f"{solver=}")
            self.assertLess(solution.dual_residual(), eps_abs, f"{solver=}")
            self.assertLess(solution.duality_gap(), eps_abs, f"{solver=}")

        return test

    @staticmethod
    def get_test_qpsut03(solver: str):
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
            problem, ref_solution = get_qpsut03()
            solution = solve_problem(problem, solver=solver)
            self.assertEqual(solution.x.shape, (4,), f"{solver=}")
            self.assertEqual(solution.y.shape, (0,), f"{solver=}")
            self.assertEqual(solution.z.shape, (0,), f"{solver=}")
            self.assertEqual(solution.z_box.shape, (4,), f"{solver=}")
            tolerance = (
                1e-1 if solver == "osqp" else 1e-2 if solver == "scs" else 1e-3
            )
            self.assertTrue(solution.is_optimal(tolerance), f"{solver=}")

        return test

    @staticmethod
    def get_test_qpsut04(solver: str):
        """
        Get test function for the QPSUT04 problem.

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
            problem, ref_solution = get_qpsut04()
            solution = solve_problem(problem, solver=solver)
            eps_abs = 2e-4 if solver in ["jaxopt_osqp", "osqp", "qpalm", "qpax", "sip"] else 1e-6
            self.assertLess(
                norm(solution.x - ref_solution.x), eps_abs, f"{solver=}"
            )
            self.assertLess(
                norm(solution.z - ref_solution.z), eps_abs, f"{solver=}"
            )
            self.assertTrue(np.isfinite(solution.duality_gap()), f"{solver=}")

        return test

    @staticmethod
    def get_test_qpsut05(solver: str):
        """
        Get test function for the QPSUT04 problem.

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
            problem, ref_solution = get_qpsut05()
            solution = solve_problem(problem, solver=solver)
            eps_abs = 2e-5 if solver == "ecos" else 1e-6
            self.assertLess(
                norm(solution.x - ref_solution.x), eps_abs, f"{solver=}"
            )
            self.assertTrue(np.isfinite(solution.duality_gap()), f"{solver=}")

        return test

    @staticmethod
    def get_test_qptest(solver: str):
        """Get test function for the QPTEST problem.

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
        ECOS fails to solve this problem.
        """

        def test(self):
            problem, solution = get_qptest()
            result = solve_problem(problem, solver=solver)
            tolerance = (
                1e1
                if solver == "gurobi"
                else (
                    1.0
                    if solver == "proxqp"
                    else (
                        2e-3
                        if solver == "osqp"
                        else (
                            5e-5
                            if solver in ["qpalm", "scs", "qpax"]
                            else (
                                1e-6
                                if solver == "mosek"
                                else (
                                    1e-7
                                    if solver == "highs"
                                    else (
                                        5e-7
                                        if solver == "cvxopt" or solver == "kvxopt"
                                        else (
                                            5e-8
                                            if solver == "clarabel"
                                            else 1e-8
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            self.assertIsNotNone(result.z_box, f"{solver=}")
            self.assertTrue(solution.is_optimal(tolerance), f"{solver=}")

        return test

    @staticmethod
    def get_test_infinite_box_bounds(solver: str):
        """Problem with some infinite box bounds.

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
            problem, _ = get_qpsut01()
            problem.lb[1] = -np.inf
            problem.ub[1] = +np.inf
            result = solve_problem(problem, solver=solver)
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            self.assertIsNotNone(result.z_box, f"{solver=}")

        return test

    @staticmethod
    def get_test_infinite_linear_bounds(solver: str):
        """Problem with some infinite linear bounds.

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
            problem, _ = get_qpsut01()
            problem.h[0] = +np.inf
            result = solve_problem(problem, solver=solver)
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            self.assertIsNotNone(result.z_box, f"{solver=}")

        return test

    @staticmethod
    def get_test_qpgurdu(solver: str):
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
            problem, _ = get_qpgurdu()
            result = solve_problem(problem, solver)
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            eps_abs = (
                6e-3
                if solver in ("jaxopt_osqp", "osqp", "qpax")
                else (
                    1e-3
                    if solver in ("scs", "sip")
                    else (
                        1e-4 if solver in ("ecos", "highs", "proxqp") else 1e-5
                    )
                )
            )
            self.assertLess(result.primal_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.dual_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.duality_gap(), eps_abs, f"{solver=}")

        return test

    @staticmethod
    def get_test_qpgurabs(solver: str):
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
            problem, _ = get_qpgurabs()
            result = solve_problem(problem, solver)
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            eps_abs = (
                0.2
                if solver == "osqp"
                else 3e-3 if solver in ["jaxopt_osqp", "proxqp"] else 1e-4
            )
            self.assertLess(result.primal_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.dual_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.duality_gap(), eps_abs, f"{solver=}")

        return test

    @staticmethod
    def get_test_qpgureq(solver: str):
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
            if solver == "ecos":
                return
            problem, _ = get_qpgureq()
            result = solve_problem(problem, solver)
            self.assertIsNotNone(result.x, f"{solver=}")
            self.assertIsNotNone(result.z, f"{solver=}")
            eps_abs = (
                0.01
                if solver in ["osqp", "qpax"]
                else 5e-3 if solver in ["jaxopt_osqp", "proxqp"] else 1e-4
            )
            self.assertLess(result.primal_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.dual_residual(), eps_abs, f"{solver=}")
            self.assertLess(result.duality_gap(), eps_abs, f"{solver=}")

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestSolveProblem,
        f"test_qpsut01_{solver}",
        TestSolveProblem.get_test_qpsut01(solver),
    )
    setattr(
        TestSolveProblem,
        f"test_qpsut02_{solver}",
        TestSolveProblem.get_test_qpsut02(solver),
    )
    if solver not in ["ecos", "mosek", "qpswift"]:
        # ECOS: https://github.com/embotech/ecos-python/issues/49
        # MOSEK: https://github.com/qpsolvers/qpsolvers/issues/229
        # qpSWIFT: https://github.com/qpsolvers/qpsolvers/issues/159
        # qpax: https://github.com/kevin-tracy/qpax/issues/5
        setattr(
            TestSolveProblem,
            f"test_qpsut03_{solver}",
            TestSolveProblem.get_test_qpsut03(solver),
        )
    setattr(
        TestSolveProblem,
        f"test_qpsut04_{solver}",
        TestSolveProblem.get_test_qpsut04(solver),
    )
    if solver not in ["osqp", "qpswift"]:
        # OSQP: see https://github.com/osqp/osqp-python/issues/104
        setattr(
            TestSolveProblem,
            f"test_qpsut05_{solver}",
            TestSolveProblem.get_test_qpsut05(solver),
        )
    if solver not in ["ecos", "qpswift"]:
        # ECOS: https://github.com/qpsolvers/qpsolvers/issues/160
        # qpSWIFT: https://github.com/qpsolvers/qpsolvers/issues/159
        # qpax: https://github.com/kevin-tracy/qpax/issues/4
        setattr(
            TestSolveProblem,
            f"test_qptest_{solver}",
            TestSolveProblem.get_test_qptest(solver),
        )
    if solver not in ["ecos", "qpswift"]:
        # See https://github.com/qpsolvers/qpsolvers/issues/159
        # See https://github.com/qpsolvers/qpsolvers/issues/160
        setattr(
            TestSolveProblem,
            f"test_infinite_box_bounds_{solver}",
            TestSolveProblem.get_test_infinite_box_bounds(solver),
        )
    if solver not in ["ecos", "qpswift", "scs"]:
        # See https://github.com/qpsolvers/qpsolvers/issues/159
        # See https://github.com/qpsolvers/qpsolvers/issues/160
        # See https://github.com/qpsolvers/qpsolvers/issues/161
        setattr(
            TestSolveProblem,
            f"test_infinite_linear_bounds_{solver}",
            TestSolveProblem.get_test_infinite_linear_bounds(solver),
        )
    setattr(
        TestSolveProblem,
        f"test_qpgurdu_{solver}",
        TestSolveProblem.get_test_qpgurdu(solver),
    )
    setattr(
        TestSolveProblem,
        f"test_qpgurabs_{solver}",
        TestSolveProblem.get_test_qpgurabs(solver),
    )
    setattr(
        TestSolveProblem,
        f"test_qpgureq_{solver}",
        TestSolveProblem.get_test_qpgureq(solver),
    )
