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

from numpy import array
from qpsolvers.check_problem_constraints import check_problem_constraints


class TestCheckProblemConstraints(unittest.TestCase):

    """
    Test fixture for `check_problem_constraint`.
    """

    def setUp(self):
        self.G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        self.h = array([3.0, 2.0, -2.0]).reshape((3,))
        self.A = array([1.0, 1.0, 1.0])
        self.b = array([1.0])

    def test_partial_inequality(self):
        with self.assertRaises(ValueError):
            check_problem_constraints(self.G, None, self.A, self.b)
        with self.assertRaises(ValueError):
            check_problem_constraints(None, self.h, self.A, self.b)

    def test_partial_equality(self):
        with self.assertRaises(ValueError):
            check_problem_constraints(self.G, self.h, self.A, None)
        with self.assertRaises(ValueError):
            check_problem_constraints(self.G, self.h, None, self.b)


if __name__ == "__main__":
    unittest.main()
