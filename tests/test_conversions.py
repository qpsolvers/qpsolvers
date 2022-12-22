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

import numpy as np

from qpsolvers.conversions import linear_from_box_inequalities


class TestConversions(unittest.TestCase):

    """
    Test fixture for box to linear inequality conversion.
    """

    def __test_linear_from_box_inequalities(self, G, h, lb, ub):
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub)
        m = G.shape[0] if G is not None else 0
        k = lb.shape[0]
        self.assertTrue(np.allclose(G2[m : m + k, :], -np.eye(k)))
        self.assertTrue(np.allclose(h2[m : m + k], -lb))
        self.assertTrue(np.allclose(G2[m + k : m + 2 * k, :], np.eye(k)))
        self.assertTrue(np.allclose(h2[m + k : m + 2 * k], ub))

    def test_concatenate_bounds(self):
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])
        self.__test_linear_from_box_inequalities(G, h, lb, ub)

    def test_pure_bounds(self):
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])
        self.__test_linear_from_box_inequalities(None, None, lb, ub)

    def test_skip_infinite_bounds(self):
        """
        TODO(scaron): infinite box bounds are skipped by the conversion.
        """
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-np.inf, -np.inf, -np.inf])
        ub = np.array([np.inf, np.inf, np.inf])
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub)
        if False:  # TODO(scaron): update behavior
            self.assertTrue(np.allclose(G2, G))
            self.assertTrue(np.allclose(h2, h))

    def test_skip_partial_infinite_bounds(self):
        """
        TODO(scaron): all values in the combined constraint vector are finite,
        even if some input box bounds are infinite.
        """
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-1.0, -np.inf, -1.0])
        ub = np.array([np.inf, 1.0, 1.0])
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub)
        if False:  # TODO(scaron): update behavior
            self.assertTrue(np.isfinite(h2).all())
