#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest
import warnings

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.nppro_ import nppro_solve_qp

    class TestNPPro(unittest.TestCase):
        """Tests specific to NPPro."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(nppro_solve_qp(P, q, G, h, A, b, lb, ub))


except ImportError:  # solver not installed
    warnings.warn("Skipping NPPro tests as the solver is not installed")
