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

import unittest
import warnings

from numpy import array, dot

from qpsolvers import (
    available_solvers,
    dense_solvers,
    solve_qp,
    solve_safer_qp,
)


class UnfeasibleProblem(unittest.TestCase):

    """
    Test fixture for an unfeasible quadratic program (inequality and equality
    constraints are inconsistent).
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        warnings.simplefilter("ignore", category=UserWarning)

    def get_dense_problem(self):
        """
        Get problem as a sextuple of values to unpack.

        Returns
        -------
        P :
            Symmetric cost matrix.
        q :
            Cost vector.
        G :
            Linear inequality matrix.
        h :
            Linear inequality vector.
        A :
            Linear equality matrix.
        b :
            Linear equality vector.
        """
        M = array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
        P = dot(M.T, M)  # this is a positive definite matrix
        q = dot(array([3.0, 2.0, 3.0]), M).reshape((3,))
        G = array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = array([3.0, 2.0, -2.0]).reshape((3,))
        A = array([1.0, 1.0, 1.0])
        b = array([42.0])
        return P, q, G, h, A, b

    @staticmethod
    def get_test(solver):
        """
        Closure of test function for a given solver.

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
            self.assertIsNone(x)

        return test

    @staticmethod
    def get_test_safer(solver):
        """
        Closure of test function for a given solver.

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
            G[0] = 0
            h[0] = -10000.0
            x = solve_safer_qp(P, q, G, h, sr=1e-2, solver=solver)
            self.assertIsNone(x)

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    if solver == "qpoases":
        # Unfortunately qpOASES returns an invalid solution in the face of this
        # problem being unfeasible. Skipping it.
        continue
    setattr(
        UnfeasibleProblem,
        "test_{}".format(solver),
        UnfeasibleProblem.get_test(solver),
    )
    if solver in dense_solvers:
        setattr(
            UnfeasibleProblem,
            "test_safer_{}".format(solver),
            UnfeasibleProblem.get_test_safer(solver),
        )
