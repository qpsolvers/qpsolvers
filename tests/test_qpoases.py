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
    from qpsolvers.solvers.qpoases_ import qpoases_solve_qp

    class TestQpOASES(unittest.TestCase):

        """
        Test fixture specific to the qpOASES solver.
        """

        def test_initvals(self):
            """
            Call the solver with a warm-start guess.
            """
            problem = get_sd3310_problem()
            qpoases_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                initvals=problem.q,
            )

        def test_params(self):
            """
            Call the solver with a time limit and other parameters.
            """
            problem = get_sd3310_problem()
            qpoases_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                time_limit=0.1,
                terminationTolerance=1e-7,
            )
            qpoases_solve_qp(
                problem.P,
                problem.q,
                time_limit=0.1,
                terminationTolerance=1e-7,
            )

        def test_unfeasible(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            custom_lb = np.ones(q.shape)
            custom_ub = -np.ones(q.shape)
            x = qpoases_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                custom_lb,
                custom_ub,
            )
            self.assertIsNone(x)

        def test_not_sparse(self):
            """
            Raise a ProblemError on sparse problems.
            """
            problem = get_sd3310_problem()
            problem.P = spa.csc_matrix(problem.P)
            with self.assertRaises(ProblemError):
                qpoases_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                )


except ImportError:  # solver not installed
    warnings.warn("Skipping qpOASES tests as the solver is not installed")
