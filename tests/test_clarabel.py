#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for Clarabel."""

import unittest
import warnings

from qpsolvers.problems import get_qpsut01

try:
    import clarabel

    from qpsolvers.solvers.clarabel_ import clarabel_solve_problem

    class TestClarabel(unittest.TestCase):
        """Test fixture for the Clarabel.rs solver."""

        def test_time_limit(self):
            """Call Clarabel.rs with an infeasibly low time limit."""
            problem, ref_solution = get_qpsut01()
            solution = clarabel_solve_problem(problem, time_limit=1e-10)
            status = solution.extras["status"]
            self.assertEqual(status, clarabel.SolverStatus.MaxTime)
            # See https://github.com/oxfordcontrol/Clarabel.rs/issues/10
            self.assertFalse(status != clarabel.SolverStatus.MaxTime)

        def test_status(self):
            """Check that result status is consistent with its string repr.

            Context: https://github.com/oxfordcontrol/Clarabel.rs/issues/10
            """
            problem, _ = get_qpsut01()
            solution = clarabel_solve_problem(problem)
            status = solution.extras["status"]
            check_1 = str(status) != "Solved"
            check_2 = status != clarabel.SolverStatus.Solved
            self.assertEqual(check_1, check_2)

except ImportError as exn:  # in case the solver is not installed
    warnings.warn(f"Skipping Clarabel.rs tests: {exn}")
