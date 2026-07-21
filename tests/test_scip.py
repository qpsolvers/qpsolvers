#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2024 Stéphane Caron and the qpsolvers contributors

"""Unit tests for SCIP."""

import unittest
import warnings

import numpy as np

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.scip_ import scip_solve_qp

    class TestSCIP(unittest.TestCase):
        """Test fixture for the SCIP solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_problem(self):
            """Solve the sd3310 reference problem."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b)
            self.assertIsNotNone(x)
            self.assertTrue(
                np.allclose(
                    x, [0.30769231, -0.69230769, 1.38461538], atol=1e-4
                )
            )

        def test_box_constraints(self):
            """Solve a problem with only box constraints."""
            P = np.array([[1.0, 0.0], [0.0, 1.0]])
            q = np.array([-1.0, -1.0])
            lb = np.array([0.0, 0.0])
            ub = np.array([0.5, 0.5])
            x = scip_solve_qp(P, q, lb=lb, ub=ub)
            self.assertIsNotNone(x)
            self.assertTrue(np.allclose(x, [0.5, 0.5], atol=1e-4))

        def test_forward_kwargs(self):
            """Forward keyword arguments to SCIP as parameters."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b, **{"limits/time": 30.0})
            self.assertIsNotNone(x)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SCIP tests: {exn}")
