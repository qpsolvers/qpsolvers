#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for SIP."""

import unittest
import warnings

import numpy as np

from qpsolvers import ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.sip_ import sip_solve_qp

    class TestSIP(unittest.TestCase):
        """Tests specific to SIP."""

        def test_problem(self):
            problem = get_sd3310_problem()
            P, q, G, h, A, b, lb, ub = problem.unpack()
            self.assertIsNotNone(sip_solve_qp(P, q, G, h, A, b, lb, ub))

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SIP tests: {exn}")
