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

from qpsolvers import solve_qp

from .problems import get_sd3310_problem

try:
    import qpoases

    class TestQpOASES(unittest.TestCase):

        """
        Test fixture specific to the qpOASES solver.
        """

        def setUp(self):
            self.assertTrue(hasattr(qpoases, "PyQProblem"))

        def test_initvals(self):
            """
            Call the solver with a warm-start guess.
            """
            P, q, G, h, A, b = get_sd3310_problem()
            solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                solver="qpoases",
                initvals=q,
            )

        def test_params(self):
            """
            Call the solver with a time limit and other parameters.
            """
            P, q, G, h, A, b = get_sd3310_problem()
            solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                solver="qpoases",
                time_limit=0.1,
                terminationTolerance=1e-7,
            )
            solve_qp(
                P,
                q,
                solver="qpoases",
                time_limit=0.1,
                terminationTolerance=1e-7,
            )

        def test_unfeasible(self):
            P, q, G, h, A, b = get_sd3310_problem()
            lb = np.ones(q.shape)
            ub = -np.ones(q.shape)
            x = solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                lb,
                ub,
                solver="qpoases",
            )
            self.assertIsNone(x)


except ImportError:  # qpOASES not installed

    pass
