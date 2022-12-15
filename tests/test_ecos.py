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

from qpsolvers.solvers import ecos_solve_qp

from .problems import get_sd3310_problem


class TestECOS(unittest.TestCase):

    """
    Tests specific to ECOS.
    """

    def test_problem(self):
        problem = get_sd3310_problem()
        P, q, G, h, A, b, lb, ub = problem.unpack()
        self.assertIsNotNone(ecos_solve_qp(P, q, G, h, A, b, lb, ub))
