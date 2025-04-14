#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for KVXOPT."""

import unittest
import warnings
from typing import Tuple

import numpy as np
import scipy.sparse as spa
from numpy import array, ones
from numpy.linalg import norm
from qpsolvers.problems import get_qpsut01

from .problems import get_sd3310_problem

try:
    import kvxopt
    from qpsolvers.solvers.kvxopt_ import kvxopt_solve_problem, kvxopt_solve_qp

    class TestKVXOPT(unittest.TestCase):
        """Test fixture for the KVXOPT solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def get_sparse_problem(
            self,
        ) -> Tuple[kvxopt.matrix, np.ndarray, kvxopt.matrix, np.ndarray]:
            """Get sparse problem as a quadruplet of values to unpack.

            Returns
            -------
            P :
                Symmetric cost matrix.
            q :
                Cost vector.
            G :
                Linear inequality matrix.
            h :
                Linear inequality vector.
            """
            n = 150
            M = spa.lil_matrix(spa.eye(n))
            for i in range(1, n - 1):
                M[i, i + 1] = -1
                M[i, i - 1] = 1
            P = spa.csc_matrix(M.dot(M.transpose()))
            q = -ones((n,))
            G = spa.csc_matrix(-spa.eye(n))
            h = -2.0 * ones((n,))
            return P, q, G, h

        def test_sparse(self):
            """Test KVXOPT on a sparse problem."""
            P, q, G, h = self.get_sparse_problem()
            x = kvxopt_solve_qp(P, q, G, h)
            self.assertIsNotNone(x, 'solver="kvxopt"')
            known_solution = array([2.0] * 149 + [3.0])
            sol_tolerance = 1e-2  # aouch, not great!
            self.assertLess(
                norm(x - known_solution), sol_tolerance, 'solver="kvxopt"'
            )
            self.assertLess(max(G.dot(x) - h), 1e-10, 'solver="kvxopt"')

        def test_extra_kwargs(self):
            """Call KVXOPT with various solver-specific settings."""
            problem = get_sd3310_problem()
            x = kvxopt_solve_qp(
                problem.P,
                problem.q,
                problem.G,
                problem.h,
                problem.A,
                problem.b,
                maxiters=10,
                abstol=1e-1,
                reltol=1e-1,
                feastol=1e-2,
                refinement=3,
            )
            self.assertIsNotNone(x, 'solver="kvxopt"')

        def test_infinite_linear_bounds(self):
            """KVXOPT does not yield a domain error on infinite bounds."""
            problem, _ = get_qpsut01()
            problem.h[1] = +np.inf
            x = kvxopt_solve_problem(problem)
            self.assertIsNotNone(x, 'solver="kvxopt"')

        def test_infinite_box_bounds(self):
            """KVXOPT does not yield a domain error infinite box bounds."""
            problem, _ = get_qpsut01()
            problem.lb[1] = -np.inf
            problem.ub[1] = +np.inf
            x = kvxopt_solve_problem(problem)
            self.assertIsNotNone(x, 'solver="kvxopt"')

except ImportError as exn:  # in case the solver is not installed
    warnings.warn(f"Skipping KVXOPT tests: {exn}")