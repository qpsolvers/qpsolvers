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

from numpy import allclose, array, eye
from qpsolvers.concatenate_bounds import concatenate_bounds


class TestConcatenateBounds(unittest.TestCase):

    """
    Test fixture for `concatenate_bounds`.
    """

    def setUp(self):
        self.G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        self.h = array([3.0, 2.0, -2.0]).reshape((3,))
        self.lb = array([-1.0, -1.0, -1.0])
        self.ub = array([1.0, 1.0, 1.0])

    def check_concatenation(self, G, h, lb, ub):
        G2, h2 = concatenate_bounds(G, h, lb, ub)
        m = G.shape[0] if G is not None else 0
        k = lb.shape[0]
        self.assertTrue(allclose(G2[m:m + k, :], -eye(k)))
        self.assertTrue(allclose(h2[m:m + k], -lb))
        self.assertTrue(allclose(G2[m + k:m + 2 * k, :], eye(k)))
        self.assertTrue(allclose(h2[m + k:m + 2 * k], ub))

    def test_concatenate_bounds(self):
        G, h, lb, ub = self.G, self.h, self.lb, self.ub
        self.check_concatenation(G, h, lb, ub)

    def test_pure_bounds(self):
        self.check_concatenation(None, None, self.lb, self.ub)


if __name__ == "__main__":
    unittest.main()
