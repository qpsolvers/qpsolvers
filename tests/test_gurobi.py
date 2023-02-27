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

"""Unit tests for Gurobi."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.gurobi_ import gurobi_solve_qp

    class TestGurobi(unittest.TestCase):
        """Test fixture for the Gurobi solver."""

        def test_gurobi_params(self):
            problem = get_sd3310_problem()
            x = gurobi_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                TimeLimit=0.1,
                FeasibilityTol=1e-8,
            )
            self.assertIsNotNone(x)


except ImportError:  # solver not installed
    warnings.warn("Skipping Gurobi tests as the solver is not installed")
