#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Unit tests for the `solve_qp` function."""

import unittest
import warnings

import numpy as np

from qpsolvers import available_solvers
from qpsolvers.conversions import combine_linear_box_inequalities


class TestCombineLinearBoxInequalities(unittest.TestCase):
    def setUp(self):
        """Prepare test fixture."""
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    def get_dense_problem(self):
        """Get dense problem as a sextuple of values to unpack.

        Returns
        -------
        P : numpy.ndarray
            Symmetric cost matrix .
        q : numpy.ndarray
            Cost vector.
        G : numpy.ndarray
            Linear inequality matrix.
        h : numpy.ndarray
            Linear inequality vector.
        A : numpy.ndarray, scipy.sparse.csc_matrix or cvxopt.spmatrix
            Linear equality matrix.
        b : numpy.ndarray
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

    @staticmethod
    def get_test_all_shapes(solver: str):
        """Get test function for a given solver.

        This variant tries all possible shapes for matrix and vector
        parameters.

        Parameters
        ----------
        solver :
            Name of the solver to test.

        Returns
        -------
        :
            Test function for that solver.
        """

        def test(self):
            P, q, G, h, _, _ = self.get_dense_problem()
            A = np.array([[1.0, 0.0, 0.0], [0.0, 0.4, 0.5]])
            b = np.array([-0.5, -1.2])
            lb = np.array([-0.5, -2, -0.8])
            ub = np.array([+1.0, +1.0, +1.0])

            ineq_variants = ((None, None), (G, h), (G[0], np.array([h[0]])))
            eq_variants = ((None, None), (A, b), (A[0], np.array([b[0]])))
            box_variants = ((None, None), (lb, None), (None, ub), (lb, ub))
            cases = [
                {
                    "P": P,
                    "q": q,
                    "G": G_case,
                    "h": h_case,
                    "A": A_case,
                    "b": b_case,
                    "lb": lb_case,
                    "ub": ub_case,
                }
                for (G_case, h_case) in ineq_variants
                for (A_case, b_case) in eq_variants
                for (lb_case, ub_case) in box_variants
            ]

            for i, test_case in enumerate(cases):
                G = test_case["G"]
                h = test_case["h"]
                lb = test_case["lb"]
                ub = test_case["ub"]
                n = test_case["q"].shape[0]
                if G is None and lb is None and ub is None:
                    continue
                elif isinstance(G, np.ndarray) and G.ndim == 1:
                    G = G.reshape((1, G.shape[0]))
                C, u, l = combine_linear_box_inequalities(
                    G, h, lb, ub, n, use_csc=False
                )
                self.assertTrue(isinstance(C, np.ndarray))
                self.assertTrue(isinstance(u, np.ndarray))
                self.assertTrue(isinstance(l, np.ndarray))

        return test


# Generate test fixtures for each solver
for solver in available_solvers:
    setattr(
        TestCombineLinearBoxInequalities,
        f"test_all_shapes_{solver}",
        TestCombineLinearBoxInequalities.get_test_all_shapes(solver),
    )
