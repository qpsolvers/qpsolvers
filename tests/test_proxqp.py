#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for ProxQP."""

import unittest
import warnings

from qpsolvers.exceptions import ParamError, ProblemError

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.proxqp_ import proxqp_solve_qp

    class TestProxQP(unittest.TestCase):
        """Test fixture specific to the ProxQP solver."""

        def test_dense_backend(self):
            """Try the dense backend."""
            problem = get_sd3310_problem()
            proxqp_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                backend="dense",
            )

        def test_sparse_backend(self):
            """Try the sparse backend."""
            problem = get_sd3310_problem()
            proxqp_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                backend="sparse",
            )

        def test_invalid_backend(self):
            """Exception raised when asking for an invalid backend."""
            problem = get_sd3310_problem()
            with self.assertRaises(ParamError):
                proxqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                    problem.A,
                    problem.b,
                    backend="invalid",
                )

        def test_double_warm_start(self):
            """Exception when two warm-start values are provided."""
            problem = get_sd3310_problem()
            with self.assertRaises(ParamError):
                proxqp_solve_qp(
                    problem.P,
                    problem.q,
                    problem.G,
                    problem.h,
                    problem.A,
                    problem.b,
                    initvals=problem.q,
                    x=problem.q,
                )

        def test_invalid_inequalities(self):
            """Check for inconsistent parameters.

            Raise an exception in an implementation-dependent inconsistent set
            of parameters. This may happen when :func:`proxqp_solve_qp` it is
            called directly.
            """
            problem = get_sd3310_problem()
            with self.assertRaises(ProblemError):
                proxqp_solve_qp(
                    problem.P,
                    problem.q,
                    G=problem.G,
                    h=None,
                    lb=problem.q,
                )

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping ProxQP tests: {exn}")
