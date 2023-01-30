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

"""Unit tests for ECOS."""

import unittest
import warnings

import numpy as np

from qpsolvers.exceptions import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.ecos_ import ecos_solve_qp

    class TestECOS(unittest.TestCase):
        """Tests specific to ECOS."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(ecos_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_infinite_inequality(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            lb = np.array([-1.0, -np.inf, -1.0])
            ub = np.array([np.inf, 1.0, 1.0])
            with self.assertRaises(ProblemError):
                ecos_solve_qp(P, q, G, h, lb=lb, ub=ub)


except ImportError:  # solver not installed
    warnings.warn("Skipping ECOS tests as the solver is not installed")
