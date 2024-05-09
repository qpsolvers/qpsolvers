#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

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

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping Gurobi tests: {exn}")
