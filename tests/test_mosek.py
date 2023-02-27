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

"""Unit tests for MOSEK."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.mosek_ import mosek_solve_qp

    class TestMOSEK(unittest.TestCase):
        """Tests specific to MOSEK."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(mosek_solve_qp(P, q, G, h, A, b, lb, ub))


except ImportError:  # solver not installed
    warnings.warn("Skipping MOSEK tests as the solver is not installed")
