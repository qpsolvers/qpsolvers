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

from qpsolvers import Problem

from .problems import get_sd3310_problem


class TestProblem(unittest.TestCase):

    """
    Test fixture for problems.
    """

    def setUp(self):
        self.problem = get_sd3310_problem()

    def test_unpack(self):
        P, q, G, h, A, b, lb, ub = self.problem.unpack()
        self.assertEqual(P.shape, self.problem.P.shape)
        self.assertEqual(q.shape, self.problem.q.shape)
        self.assertEqual(G.shape, self.problem.G.shape)
        self.assertEqual(h.shape, self.problem.h.shape)
        self.assertEqual(A.shape, self.problem.A.shape)
        self.assertEqual(b.shape, self.problem.b.shape)
        self.assertIsNone(lb)
        self.assertIsNone(ub)

    def test_check_inequality_constraints(self):
        problem = get_sd3310_problem()
        P, q, G, h, A, b, _, _ = problem.unpack()
        with self.assertRaises(ValueError):
            Problem(P, q, G, None, A, b).check_constraints()
        with self.assertRaises(ValueError):
            Problem(P, q, None, h, A, b).check_constraints()

    def test_check_equality_constraints(self):
        problem = get_sd3310_problem()
        P, q, G, h, A, b, _, _ = problem.unpack()
        with self.assertRaises(ValueError):
            Problem(P, q, G, h, A, None).check_constraints()
        with self.assertRaises(ValueError):
            Problem(P, q, G, h, None, b).check_constraints()

    def test_cond(self):
        unconstrained = Problem(self.problem.P, self.problem.q)
        self.assertAlmostEqual(unconstrained.cond(), 124.257, places=4)
        self.assertGreater(self.problem.cond(), 200.0)
