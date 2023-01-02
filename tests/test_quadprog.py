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
import warnings

import numpy as np
import scipy.sparse as spa

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.quadprog_ import quadprog_solve_qp

    class TestQuadprog(unittest.TestCase):

        """
        Test fixture for the quadprog solver.
        """

        def setUp(self):
            """
            Prepare test fixture.
            """
            warnings.simplefilter("ignore", category=UserWarning)

        def test_non_psd_cost(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            P -= np.eye(3)
            with self.assertRaises(ValueError):
                quadprog_solve_qp(P, q, G, h, A, b)

        def test_quadprog_value_error(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            q = q[1:]  # raise "G and a must have the same dimension"
            self.assertIsNone(quadprog_solve_qp(P, q, G, h, A, b))

        def test_not_sparse(self):
            """
            Raise a ProblemError on sparse problems.
            """
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            P = spa.csc_matrix(P)
            with self.assertRaises(ProblemError):
                quadprog_solve_qp(P, q, G, h, A, b)


except ImportError:  # solver not installed
    warnings.warn("Skipping quadprog tests as the solver is not installed")
