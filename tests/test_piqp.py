#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for PIQP."""

import unittest
import warnings

from qpsolvers.exceptions import ParamError, ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.piqp_ import piqp_solve_qp

    class TestPIQP(unittest.TestCase):
        """Test fixture specific to the PIQP solver."""

        def test_dense_backend(self):
            """Try the dense backend."""
            problem = get_sd3310_problem()
            sol = piqp_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                backend="dense",
            )
            self.assertIsNotNone(sol)

        def test_sparse_backend(self):
            """Try the sparse backend."""
            problem = get_sd3310_problem()
            sol = piqp_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                backend="sparse",
            )
            self.assertIsNotNone(sol)

        def test_invalid_backend(self):
            """Exception raised when asking for an invalid backend."""
            problem = get_sd3310_problem()
            with self.assertRaises(ParamError):
                piqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                    problem.A,
                    problem.b,
                    backend="invalid",
                )

        def test_invalid_problems(self):
            """Exception raised when asking for an invalid backend."""
            problem = get_sd3310_problem()
            with self.assertRaises(ProblemError):
                piqp_solve_qp(
                    problem.P,
                    problem.q,
                    None,
                    problem.h,
                    problem.A,
                    problem.b,
                    backend="sparse",
                )
            with self.assertRaises(ProblemError):
                piqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    None,
                    problem.A,
                    problem.b,
                    backend="sparse",
                )
            with self.assertRaises(ProblemError):
                piqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                    None,
                    problem.b,
                    backend="sparse",
                )
            with self.assertRaises(ProblemError):
                piqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                    problem.A,
                    None,
                    backend="sparse",
                )

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping PIQP tests: {exn}")
