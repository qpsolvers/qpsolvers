#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stephane Caron <stephane.caron@normalesup.org>
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

from numpy import allclose, array, dot
from qpsolvers import available_solvers, solve_qp


class ExampleProblem(unittest.TestCase):

    """
    Test fixture for the README example problem.
    """

    def setUp(self):
        """
        Prepare test fixture.
        """
        M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
        self.P = dot(M.T, M)  # this is a positive definite matrix
        self.q = dot(array([3., 2., 3.]), M).reshape((3,))
        self.G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        self.h = array([3., 2., -2.]).reshape((3,))
        self.A = array([1., 1., 1.])
        self.b = array([1.])

    def get_problem(self):
        """
        Get problem as a sextuple of values to unpack.

        Returns
        -------
        P : numpy.array
            Symmetric quadratic-cost matrix .
        q : numpy.array
            Quadratic-cost vector.
        G : numpy.array
            Linear inequality matrix.
        h : numpy.array
            Linear inequality vector.
        A : numpy.array, scipy.sparse.csc_matrix or cvxopt.spmatrix
            Linear equality matrix.
        b : numpy.array
            Linear equality vector.
        """
        return self.P, self.q, self.G, self.h, self.A, self.b

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
            P, q, G, h, A, b = self.get_problem()
            print(solver)
            x = solve_qp(P, q, G, h, A, b, solver=solver)
            self.assertIsNotNone(x)
            self.assertTrue((dot(G, x) <= h).all())
            self.assertTrue(allclose(dot(A, x), b))
        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(ExampleProblem, 'test_{}'.format(solver),
            ExampleProblem.get_test(solver))


if __name__ == '__main__':
    unittest.main()
