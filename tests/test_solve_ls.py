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
Tests for the `solve_lp` function.
"""

import unittest
import warnings

import numpy as np
import scipy.sparse as spa
from numpy.linalg import norm

from qpsolvers import available_solvers, solve_ls, sparse_solvers
from qpsolvers.exceptions import NoSolverSelected, SolverNotFound

from .problems import get_sparse_least_squares


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
        self.R = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 4.0], [0.0, 4.0, 1.0]])
        self.s = np.array([3.0, 2.0, 3.0])
        self.G = np.array(
            [[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]]
        )
        self.h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        self.A = np.array([1.0, 1.0, 1.0])
        self.b = np.array([1.0])
        self.known_solution = np.array([2.0 / 3, -1.0 / 3, 2.0 / 3])

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
            sol_tolerance = (
                5e-3
                if solver == "osqp"
                else 2e-5
                if solver == "proxqp"
                else 1e-5
                if solver == "ecos"
                else 1e-6
            )
            eq_tolerance = 1e-9
            ineq_tolerance = (
                1e-3
                if solver == "osqp"
                else 1e-5
                if solver == "proxqp"
                else 2e-7
                if solver == "scs"
                else 1e-9
            )
            self.assertLess(norm(x - self.known_solution), sol_tolerance)
            self.assertLess(norm(x_sp - self.known_solution), sol_tolerance)
            self.assertLess(max(G.dot(x) - h), ineq_tolerance)
            self.assertLess(max(A.dot(x) - b), eq_tolerance)
            self.assertLess(min(A.dot(x) - b), eq_tolerance)

        return test

    def test_no_solver_selected(self):
        """
        Check that NoSolverSelected is raised when applicable.
        """
        R, s, G, h, A, b = self.get_problem()
        with self.assertRaises(NoSolverSelected):
            solve_ls(R, s, G, h, A, b, solver=None)

    def test_solver_not_found(self):
        """
        Check that SolverNotFound is raised when the solver does not exist.
        """
        R, s, G, h, A, b = self.get_problem()
        with self.assertRaises(SolverNotFound):
            solve_ls(R, s, G, h, A, b, solver="ideal")

    @staticmethod
    def get_test_mixed_sparse_args(solver: str):
        """
        Get test function for mixed sparse problems with a given solver.

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
            _, s, G, h, A, b = self.get_problem()
            n = len(s)

            R_csc = spa.eye(n, format="csc")
            x_csc = solve_ls(R_csc, s, G, h, A, b, solver=solver)
            self.assertIsNotNone(x_csc)

            R_dia = spa.eye(n)
            x_dia = solve_ls(R_dia, s, G, h, A, b, solver=solver)
            self.assertIsNotNone(x_dia)

            x_np_dia = solve_ls(
                R_dia, s, G, h, A, b, W=np.eye(n), solver=solver
            )
            self.assertIsNotNone(x_np_dia)

            sol_tolerance = 1e-8
            self.assertLess(norm(x_csc - x_dia), sol_tolerance)
            self.assertLess(norm(x_csc - x_np_dia), sol_tolerance)

        return test

    @staticmethod
    def get_test_medium_sparse_problem(solver: str):
        """
        Get test function for a large sparse problem with a given solver.

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
            R, s, G, h, A, b, lb, ub = get_sparse_least_squares(n=1500)
            x = solve_ls(R, s, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)

        return test

    @staticmethod
    def get_test_large_sparse_problem(solver: str):
        """
        Get test function for a large sparse problem with a given solver.

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
            R, s, G, h, A, b, lb, ub = get_sparse_least_squares(n=15_000)
            x = solve_ls(R, s, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)

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
    if solver != "gurobi":
        # Gurobi: model too large for size-limited license
        setattr(
            TestSolveLS,
            "test_medium_sparse_problem_{}".format(solver),
            TestSolveLS.get_test_medium_sparse_problem(solver),
        )
    if solver not in ["gurobi", "highs", "scs"]:
        # Gurobi: model too large for size-limited license
        # HiGHS: model too large https://github.com/ERGO-Code/HiGHS/issues/992
        # SCS: issue reported in https://github.com/cvxgrp/scs/issues/234
        setattr(
            TestSolveLS,
            "test_large_sparse_problem_{}".format(solver),
            TestSolveLS.get_test_large_sparse_problem(solver),
        )
