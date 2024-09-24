#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests with unfeasible problems."""

import unittest
import warnings

from numpy import array, dot
from qpsolvers import available_solvers, solve_qp


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

    def get_unfeasible_problem(self):
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
    def get_test(solver: str):
        """
        Closure of test function for a given solver.

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
            P, q, G, h, A, b = self.get_unfeasible_problem()
            x = solve_qp(P, q, G, h, A, b, solver=solver)
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
