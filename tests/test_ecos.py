#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for ECOS."""

import unittest
import warnings

import numpy as np

from qpsolvers.exceptions import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.ecos_ import ecos_solve_qp

    class TestECOS(unittest.TestCase):
        """Tests specific to ECOS."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(ecos_solve_qp(P, q, G, h, A, b, lb, ub))

        def test_infinite_inequality(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            lb = np.array([-1.0, -np.inf, -1.0])
            ub = np.array([np.inf, 1.0, 1.0])
            with self.assertRaises(ProblemError):
                ecos_solve_qp(P, q, G, h, lb=lb, ub=ub)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping ECOS tests: {exn}")
