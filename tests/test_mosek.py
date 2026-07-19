#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for MOSEK."""

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.mosek_ import mosek_solve_qp

    class TestMOSEK(unittest.TestCase):
        """Tests specific to MOSEK."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(mosek_solve_qp(P, q, G, h, A, b, lb, ub))

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping MOSEK tests: {exn}")
