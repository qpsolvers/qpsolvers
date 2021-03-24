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

from numpy import array, dot
from qpsolvers import solve_qp


class QuadProgTest(unittest.TestCase):

    def setUp(self):
        M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
        self.P = dot(M.T, M)  # this is a positive definite matrix
        self.q = dot(array([3., 2., 3.]), M).reshape((3,))
        self.G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
        self.h = array([3., 2., -2.]).reshape((3,))
        self.A = array([1., 1., 1.])
        self.b = array([1.])

    def get_problem(self):
        return self.P, self.q, self.G, self.h, self.A, self.b

    def test_feasible(self):
        P, q, G, h, A, b = self.get_problem()
        x = solve_qp(P, q, G, h, A, b, solver="quadprog")
        self.assertTrue((dot(G, x) <= h).all())

    def test_unfeasible(self):
        P, q, _, h, A, _ = self.get_problem()
        G_u = array([[1., 1., 1.], [2., 0., 1.], [-1., 2., -1.]])
        b_u = array([42.])
        x = solve_qp(P, q, G_u, h, A, b_u)
        self.assertIsNone(x)


if __name__ == '__main__':
    unittest.main()
