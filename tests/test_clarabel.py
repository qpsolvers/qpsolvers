#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

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

except ImportError:  # solver not installed
    warnings.warn("Skipping Clarabel.rs tests as the solver is not installed")
