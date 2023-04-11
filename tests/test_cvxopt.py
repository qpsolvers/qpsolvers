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

"""Unit tests for CVXOPT."""

import unittest
import warnings
from typing import Tuple

import numpy as np
import scipy.sparse as spa
from numpy import array, ones
from numpy.linalg import norm

from .problems import get_sd3310_problem
from qpsolvers.problems import get_qpsut01

try:
    import cvxopt

    from qpsolvers.solvers.cvxopt_ import cvxopt_solve_problem, cvxopt_solve_qp

    class TestCVXOPT(unittest.TestCase):
        """Test fixture for the CVXOPT solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def get_sparse_problem(
            self,
        ) -> Tuple[cvxopt.matrix, np.ndarray, cvxopt.matrix, np.ndarray]:
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
            """Test CVXOPT on a sparse problem."""
            P, q, G, h = self.get_sparse_problem()
            x = cvxopt_solve_qp(P, q, G, h)
            self.assertIsNotNone(x)
            known_solution = array([2.0] * 149 + [3.0])
            sol_tolerance = 1e-2  # aouch, not great!
            self.assertLess(norm(x - known_solution), sol_tolerance)
            self.assertLess(max(G.dot(x) - h), 1e-10)

        def test_extra_kwargs(self):
            """Call CVXOPT with various solver-specific settings."""
            problem = get_sd3310_problem()
            x = cvxopt_solve_qp(
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
            self.assertIsNotNone(x)

        def test_infinite_linear_bounds(self):
            """CVXOPT does not yield a domain error on infinite bounds."""
            problem, _ = get_qpsut01()
            problem.h[1] = +np.inf
            x = cvxopt_solve_problem(problem)
            self.assertIsNotNone(x)

        def test_infinite_box_bounds(self):
            """CVXOPT does not yield a domain error infinite box bounds."""
            problem, _ = get_qpsut01()
            problem.lb[1] = -np.inf
            problem.ub[1] = +np.inf
            x = cvxopt_solve_problem(problem)
            self.assertIsNotNone(x)


except ImportError:  # solver not installed
    warnings.warn("Skipping CVXOPT tests as the solver is not installed")
