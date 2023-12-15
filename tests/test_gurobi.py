#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
# SPDX-License-Identifier: LGPL-3.0-or-later

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
