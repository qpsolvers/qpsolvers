#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unit tests for Problem class."""

import tempfile
import unittest

import numpy as np
import scipy.sparse as spa

from qpsolvers import ActiveSet, Problem, ProblemError

from .problems import get_sd3310_problem


class TestProblem(unittest.TestCase):
    """Test fixture for problems."""

    def setUp(self):
        self.problem = get_sd3310_problem()

    def test_unpack(self):
        P, q, G, h, A, b, lb, ub = self.problem.unpack()
        self.assertEqual(P.shape, self.problem.P.shape)
        self.assertEqual(q.shape, self.problem.q.shape)
        self.assertEqual(G.shape, self.problem.G.shape)
        self.assertEqual(h.shape, self.problem.h.shape)
        self.assertEqual(A.shape, self.problem.A.shape)
        self.assertEqual(b.shape, self.problem.b.shape)
        self.assertIsNone(lb)
        self.assertIsNone(ub)

    def test_check_inequality_constraints(self):
        problem = get_sd3310_problem()
        P, q, G, h, A, b, _, _ = problem.unpack()
        with self.assertRaises(ProblemError):
            Problem(P, q, G, None, A, b).check_constraints()
        with self.assertRaises(ProblemError):
            Problem(P, q, None, h, A, b).check_constraints()

    def test_check_equality_constraints(self):
        problem = get_sd3310_problem()
        P, q, G, h, A, b, _, _ = problem.unpack()
        with self.assertRaises(ProblemError):
            Problem(P, q, G, h, A, None).check_constraints()
        with self.assertRaises(ProblemError):
            Problem(P, q, G, h, None, b).check_constraints()

    def test_cond(self):
        active_set = ActiveSet(
            G_indices=range(self.problem.G.shape[0]),
            lb_indices=[],
            ub_indices=[],
        )
        self.assertGreater(self.problem.cond(active_set), 200.0)

    def test_cond_unconstrained(self):
        unconstrained = Problem(self.problem.P, self.problem.q)
        active_set = ActiveSet()
        self.assertAlmostEqual(
            unconstrained.cond(active_set), 124.257, places=4
        )

    def test_cond_no_equality(self):
        no_equality = Problem(
            self.problem.P, self.problem.q, self.problem.G, self.problem.h
        )
        active_set = ActiveSet(G_indices=range(self.problem.G.shape[0]))
        self.assertGreater(no_equality.cond(active_set), 200.0)

    def test_cond_only_equalities(self):
        """Condition number with only equality constraints."""
        only_equalities = Problem(
            self.problem.P, self.problem.q, A=self.problem.A, b=self.problem.b
        )
        active_set = ActiveSet()  # equalities are always part of the KKT matrix
        self.assertAlmostEqual(
            only_equalities.cond(active_set), 93.7393, places=4
        )

    def test_cond_active_inequalities(self):
        """Condition number with active linear inequality constraints."""
        inequalities = Problem(
            self.problem.P, self.problem.q, self.problem.G, self.problem.h
        )
        active_set = ActiveSet(G_indices=range(self.problem.G.shape[0]))
        self.assertAlmostEqual(
            inequalities.cond(active_set), 13736.27, places=2
        )

    def test_cond_active_lower_bounds(self):
        """Condition number with active lower bounds.

        With ``lb = [1, 1, 1]`` the unconstrained optimum lies below all
        bounds, so all three lower bounds are active at the optimum.
        """
        lb = np.array([1.0, 1.0, 1.0])
        with_lb = Problem(self.problem.P, self.problem.q, lb=lb)
        active_set = ActiveSet(lb_indices=range(lb.size))
        self.assertAlmostEqual(
            with_lb.cond(active_set), 5989.643, places=3
        )

    def test_cond_active_upper_bounds(self):
        """Condition number with active upper bounds.

        Box-constraint rows are (signed) identity rows, so the condition
        number only depends on which bounds are active, not on their values.
        It therefore matches the lower-bound case with the same active set.
        """
        ub = np.array([-1.0, -1.0, -1.0])
        with_ub = Problem(self.problem.P, self.problem.q, ub=ub)
        active_set = ActiveSet(ub_indices=range(ub.size))
        self.assertAlmostEqual(
            with_ub.cond(active_set), 5989.643, places=3
        )

    def test_cond_sparse(self):
        sparse = Problem(spa.csc_matrix(self.problem.P), self.problem.q)
        active_set = ActiveSet()
        with self.assertRaises(ProblemError):
            sparse.cond(active_set)

    def test_check_matrix_shapes(self):
        Problem(np.eye(1), np.ones(1))
        Problem(np.array([1.0]), np.ones(1))

    def test_check_vector_shapes(self):
        Problem(np.eye(3), np.ones(shape=(3, 1)))
        Problem(np.eye(3), np.ones(shape=(1, 3)))
        Problem(np.eye(3), np.ones(shape=(3,)))
        with self.assertRaises(ProblemError):
            Problem(np.eye(3), np.ones(shape=(3, 2)))
        with self.assertRaises(ProblemError):
            Problem(np.eye(3), np.ones(shape=(3, 1, 1)))
        with self.assertRaises(ProblemError):
            Problem(np.eye(3), np.ones(shape=(1, 3, 1)))

    def test_save_load(self):
        problem = Problem(np.eye(3), np.ones(shape=(3, 1)))
        save_path = tempfile.mktemp() + ".npz"
        problem.save(save_path)
        reloaded = Problem.load(save_path)
        self.assertTrue(np.allclose(reloaded.P, problem.P))
        self.assertTrue(np.allclose(reloaded.q, problem.q))
        self.assertIsNone(reloaded.G)
        self.assertIsNone(reloaded.h)
        self.assertIsNone(reloaded.A)
        self.assertIsNone(reloaded.b)
        self.assertIsNone(reloaded.lb)
        self.assertIsNone(reloaded.ub)
