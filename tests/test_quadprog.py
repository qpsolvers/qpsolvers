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

from numpy import array, dot, eye
from qpsolvers import solve_qp


class ReadmeProblem(unittest.TestCase):

    """
    Test fixture for the README example problem.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
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

    def test_non_psd_cost(self):
        P, q, G, h, A, b = self.get_problem()
        P -= eye(3)
        with self.assertRaises(ValueError):
            solve_qp(P, q, G, h, A, b, solver="quadprog")

    def test_quadprog_value_error(self):
        P, q, G, h, A, b = self.get_problem()
        q = q[1:]  # raise quadprog's "G and a must have the same dimension"
        self.assertIsNone(solve_qp(P, q, G, h, A, b, solver="quadprog"))


if __name__ == "__main__":
    unittest.main()
