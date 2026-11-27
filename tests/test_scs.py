#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for SCS."""

import unittest
import warnings

import numpy as np
from scipy.sparse import diags

from qpsolvers import ProblemError, solve_qp

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.scs_ import scs_solve_qp

    class TestSCS(unittest.TestCase):
        """Tests specific to SCS."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(scs_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_unbounded_below(self):
            problem = get_sd3310_problem()
            P, q, _, _, _, _, _, _ = problem.unpack()
            P -= np.eye(3)  # make problem unbounded
            with self.assertRaises(ProblemError):
                scs_solve_qp(P, q)

        def test_unconstrained_positive_definite(self):
            """Solve a bounded problem rejected at default LSQR accuracy."""
            weights = np.arange(1.0, 31.0)
            x = solve_qp(diags(weights, format="csc"), -weights, solver="scs")
            np.testing.assert_allclose(x, np.ones(30), atol=1e-6)

        def test_unconstrained_scaled(self):
            """Solve bounded problems at different objective scales."""
            for n in (10, 100):
                for scale in (1e-3, 1.0, 1e3):
                    with self.subTest(n=n, scale=scale):
                        weights = scale * np.arange(1.0, n + 1.0)
                        x = solve_qp(
                            diags(weights, format="csc"),
                            -weights,
                            solver="scs",
                            eps_abs=1e-8,
                            eps_rel=1e-8,
                        )
                        np.testing.assert_allclose(x, np.ones(n), atol=1e-6)

        def test_unconstrained_positive_semidefinite(self):
            """Allow a nullspace when the objective is bounded below."""
            weights = np.arange(30.0)
            x = solve_qp(diags(weights, format="csc"), -weights, solver="scs")
            np.testing.assert_allclose(x[1:], np.ones(29), atol=1e-6)

        def test_unconstrained_nullspace_unbounded(self):
            """Keep rejecting an objective that varies in the nullspace."""
            weights = np.arange(30.0)
            q = -weights
            q[0] = 1.0
            with self.assertRaises(ProblemError):
                solve_qp(diags(weights, format="csc"), q, solver="scs")

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SCS tests: {exn}")
