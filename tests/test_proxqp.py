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

import numpy as np

from qpsolvers import solve_qp

try:
    import proxsuite

    from qpsolvers import proxqp_solve_qp

    class TestproxQP(unittest.TestCase):

        """
        Test fixture specific to the ProxQP solver.
        """

        def get_dense_problem(self):
            """
            Get problem as a sextuple of values to unpack.

            Returns
            -------
            P :
                Symmetric quadratic-cost matrix .
            q :
                Quadratic-cost vector.
            G :
                Linear inequality matrix.
            h :
                Linear inequality vector.
            A :
                Linear equality matrix.
            b :
                Linear equality vector.
            """
            M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
            P = np.dot(M.T, M)  # this is a positive definite matrix
            q = np.dot(np.array([3.0, 2.0, 3.0]), M).reshape((3,))
            G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
            h = np.array([3.0, 2.0, -2.0]).reshape((3,))
            A = np.array([1.0, 1.0, 1.0])
            b = np.array([1.0])
            return P, q, G, h, A, b

        def test_dense_backend(self):
            """
            Try the dense backend.
            """
            P, q, G, h, A, b = self.get_dense_problem()
            solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                solver="proxqp",
                backend="dense",
            )

        def test_sparse_backend(self):
            """
            Try the sparse backend.
            """
            P, q, G, h, A, b = self.get_dense_problem()
            solve_qp(
                P,
                q,
                G,
                h,
                A,
                b,
                solver="proxqp",
                backend="sparse",
            )

        def test_invalid_backend(self):
            """
            Exception raised when asking for an invalid backend.
            """
            P, q, G, h, A, b = self.get_dense_problem()
            with self.assertRaises(ValueError):
                solve_qp(
                    P,
                    q,
                    G,
                    h,
                    A,
                    b,
                    solver="proxqp",
                    backend="invalid",
                )

        def test_double_warm_start(self):
            """
            Raise an exception when two warm-start values are provided at the
            same time.
            """
            P, q, G, h, A, b = self.get_dense_problem()
            with self.assertRaises(ValueError):
                solve_qp(
                    P,
                    q,
                    G,
                    h,
                    A,
                    b,
                    solver="proxqp",
                    initvals=q,
                    x=q,
                    avoid_unused_import_warning=proxsuite,
                )

        def test_invalid_inequalities(self):
            """
            Raise an exception in an implementation-dependent inconsistent set
            of parameters. This won't happen when the function is called by
            `solve_qp`, but it may happen when it is called directly.
            """
            P, q, G, _, _, _ = self.get_dense_problem()
            with self.assertRaises(ValueError):
                proxqp_solve_qp(P, q, G=G, h=None, lb=q)


except ImportError:  # ProxSuite is not installed

    pass


if __name__ == "__main__":
    unittest.main()
