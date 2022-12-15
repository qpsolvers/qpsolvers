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

import unittest

from .problems import get_sd3310_problem

try:
    from qpsolvers.solvers.proxqp_ import proxqp_solve_qp

    class TestProxQP(unittest.TestCase):

        """
        Test fixture specific to the ProxQP solver.
        """

        def test_dense_backend(self):
            """
            Try the dense backend.
            """
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
            """
            Try the sparse backend.
            """
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
            """
            Exception raised when asking for an invalid backend.
            """
            problem = get_sd3310_problem()
            with self.assertRaises(ValueError):
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
            """
            Raise an exception when two warm-start values are provided at the
            same time.
            """
            problem = get_sd3310_problem()
            with self.assertRaises(ValueError):
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
            """
            Raise an exception in an implementation-dependent inconsistent set
            of parameters. This may happen when :func:`proxqp_solve_qp` it is
            called directly.
            """
            problem = get_sd3310_problem()
            with self.assertRaises(ValueError):
                proxqp_solve_qp(
                    problem.P,
                    problem.q,
                    G=problem.G,
                    h=None,
                    lb=problem.q,
                )


except ImportError:  # ProxSuite not installed

    pass
