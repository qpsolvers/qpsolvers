#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for COPT."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.copt_ import copt_solve_qp

    class TestCOPT(unittest.TestCase):
        """Test fixture for the COPT solver."""

        def test_copt_params(self):
            problem = get_sd3310_problem()
            x = copt_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                TimeLimit=0.1,
                FeasTol=1e-8,
            )
            self.assertIsNotNone(x)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping COPT tests: {exn}")
