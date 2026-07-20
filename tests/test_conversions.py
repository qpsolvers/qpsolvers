#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for internal conversion functions."""

import unittest

import numpy as np
import scipy.sparse as spa

from qpsolvers import ProblemError
from qpsolvers.conversions import (
    linear_from_box_inequalities,
    put_infinite_inequalities_back,
    remove_infinite_inequalities,
)


class TestConversions(unittest.TestCase):
    """Test fixture for box to linear inequality conversion."""

    def __test_linear_from_box_inequalities(self, G, h, lb, ub):
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
        m = G.shape[0] if G is not None else 0
        k = lb.shape[0]
        self.assertTrue(np.allclose(G2[m : m + k, :], -np.eye(k)))
        self.assertTrue(np.allclose(h2[m : m + k], -lb))
        self.assertTrue(np.allclose(G2[m + k : m + 2 * k, :], np.eye(k)))
        self.assertTrue(np.allclose(h2[m + k : m + 2 * k], ub))

    def test_concatenate_bounds(self):
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])
        self.__test_linear_from_box_inequalities(G, h, lb, ub)

    def test_pure_bounds(self):
        lb = np.array([-1.0, -1.0, -1.0])
        ub = np.array([1.0, 1.0, 1.0])
        self.__test_linear_from_box_inequalities(None, None, lb, ub)

    def test_skip_infinite_bounds(self):
        """
        TODO(scaron): infinite box bounds are skipped by the conversion.
        """
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-np.inf, -np.inf, -np.inf])
        ub = np.array([np.inf, np.inf, np.inf])
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
        if False:  # TODO(scaron): update behavior
            self.assertTrue(np.allclose(G2, G))
            self.assertTrue(np.allclose(h2, h))

    def test_skip_partial_infinite_bounds(self):
        """
        TODO(scaron): all values in the combined constraint vector are finite,
        even if some input box bounds are infinite.
        """
        G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
        h = np.array([3.0, 2.0, -2.0]).reshape((3,))
        lb = np.array([-1.0, -np.inf, -1.0])
        ub = np.array([np.inf, 1.0, 1.0])
        G2, h2 = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
        if False:  # TODO(scaron): update behavior
            self.assertTrue(np.isfinite(h2).all())

    def test_sparse_conversion(self):
        """
        Box concatenation on a sparse problem without linear inequality
        constraints yields a sparse problem.
        """
        n = 1000
        lb = np.full((n,), -1.0)
        ub = np.full((n,), +1.0)
        G, h = linear_from_box_inequalities(
            None, None, lb, ub, use_sparse=True
        )
        self.assertTrue(isinstance(G, spa.csc_matrix))


class TestRemoveInfiniteInequalities(unittest.TestCase):
    """Test fixture for removing infinite linear inequalities."""

    def setUp(self):
        self.G = np.array(
            [[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]]
        )
        self.h = np.array([3.0, np.inf, -2.0])

    def test_removes_infinite_row(self):
        G, h, kept = remove_infinite_inequalities(self.G, self.h)
        self.assertTrue(np.array_equal(kept, [True, False, True]))
        self.assertTrue(np.allclose(G, self.G[[0, 2]]))
        self.assertTrue(np.allclose(h, self.h[[0, 2]]))

    def test_keeps_finite_rows_untouched(self):
        h = np.array([3.0, 2.0, -2.0])
        G, h_out, kept = remove_infinite_inequalities(self.G, h)
        self.assertTrue(kept.all())
        self.assertIs(G, self.G)  # returned as-is, no copy
        self.assertIs(h_out, h)

    def test_removes_infinite_row_sparse(self):
        G, h, kept = remove_infinite_inequalities(
            spa.csc_matrix(self.G), self.h
        )
        self.assertTrue(np.array_equal(kept, [True, False, True]))
        self.assertEqual(G.shape, (2, 3))
        self.assertTrue(np.allclose(G.toarray(), self.G[[0, 2]]))

    def test_negative_infinite_raises(self):
        h = np.array([3.0, -np.inf, -2.0])
        with self.assertRaises(ProblemError):
            remove_infinite_inequalities(self.G, h)

    def test_nan_raises(self):
        h = np.array([3.0, np.nan, -2.0])
        with self.assertRaises(ProblemError):
            remove_infinite_inequalities(self.G, h)

    def test_put_back_fills_out_zeros(self):
        _, _, kept = remove_infinite_inequalities(self.G, self.h)
        z = np.array([0.7, 0.3])  # multipliers for the two kept rows
        z_full = put_infinite_inequalities_back(z, kept)
        self.assertTrue(np.allclose(z_full, [0.7, 0.0, 0.3]))

    def test_put_back_identity_when_all_finite(self):
        kept = np.ones(3, dtype=bool)
        z = np.array([1.0, 2.0, 3.0])
        self.assertIs(put_infinite_inequalities_back(z, kept), z)
