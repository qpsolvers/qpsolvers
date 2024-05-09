#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for qpOASES."""

import unittest
import warnings

import numpy as np
import scipy.sparse as spa

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.qpoases_ import qpoases_solve_qp

    class TestQpOASES(unittest.TestCase):
        """Test fixture specific to the qpOASES solver."""

        def test_initvals(self):
            """Call the solver with a warm-start guess."""
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
            """Call the solver with a time limit and other parameters."""
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
            """Raise a ProblemError on sparse problems."""
            problem = get_sd3310_problem()
            problem.P = spa.csc_matrix(problem.P)
            with self.assertRaises(ProblemError):
                qpoases_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                )

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping qpOASES tests: {exn}")
