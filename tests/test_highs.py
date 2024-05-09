#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for HiGHS."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.highs_ import highs_solve_qp

    class TestHiGHS(unittest.TestCase):
        """Test fixture for the HiGHS solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def test_highs_tolerances(self):
            problem = get_sd3310_problem()
            x = highs_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                time_limit=0.1,
                primal_feasibility_tolerance=1e-1,
                dual_feasibility_tolerance=1e-1,
            )
            self.assertIsNotNone(x)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping HiGHS tests: {exn}")
