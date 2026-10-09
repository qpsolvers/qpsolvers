#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for PDHCG backend selection and QP solution conversion."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as spa

from qpsolvers import Problem, available_solvers, solve_problem, solve_qp

try:
    from pdhcg import built_devices

    HAS_CPU = "cpu" in built_devices()
except (ImportError, RuntimeError):
    HAS_CPU = False


@unittest.skipUnless("pdhcg" in available_solvers, "PDHCG is not installed")
class TestPDHCGDispatch(unittest.TestCase):
    """Check device routing without requiring a GPU."""

    def test_pdhcg_device_and_parameter_routing(self):
        """Device belongs to optimize; solver parameters go to setParams."""
        for device in (None, "cpu", "cuda"):
            with (
                self.subTest(device=device),
                patch("qpsolvers.solvers.pdhcg_.Model") as model_class,
            ):
                model = model_class.return_value
                model.Status = "ITERATION_LIMIT"
                problem = Problem(np.eye(2), np.ones(2))
                result = solve_problem(
                    problem,
                    solver="pdhcg",
                    device=device,
                    Threads=3,
                    FeasibilityTol=1e-7,
                    initvals=np.zeros(2),
                )
                model.setParams.assert_called_once_with(
                    Threads=3, FeasibilityTol=1e-7
                )
                if device is None:
                    model.optimize.assert_called_once_with()
                else:
                    model.optimize.assert_called_once_with(device=device)
                np.testing.assert_array_equal(
                    model.setWarmStart.call_args.kwargs["primal"], np.zeros(2)
                )
                self.assertFalse(result.found)
                for value in (result.x, result.y, result.z, result.z_box):
                    self.assertIsNone(value)


@unittest.skipUnless(HAS_CPU, "PDHCG CPU backend is not installed")
class TestPDHCGCPU(unittest.TestCase):
    """Exercise the real CPU solver and its public qpsolvers interface."""

    def setUp(self):
        """Use strict tolerances and a bounded runtime for small QPs."""
        self.options = dict(
            solver="pdhcg",
            device="cpu",
            Threads=2,
            TimeLimit=10,
            OptimalityTol=1e-9,
            FeasibilityTol=1e-9,
        )

    def test_pdhcg_dense_sparse_and_mixed_constraints(self):
        """Check primal/dual values with both constraint types and formats."""
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                P = spa.eye(2, format="csc") if sparse else np.eye(2)
                G = np.array([[1.0, 0.0]])
                A = spa.csr_matrix([[1.0, 1.0]]) if sparse else np.ones((1, 2))
                problem = Problem(
                    P,
                    np.array([-2.0, -1.0]),
                    G,
                    np.array([0.5]),
                    A,
                    np.array([1.0]),
                )
                result = solve_problem(problem, **self.options)
                self.assertTrue(result.found)
                np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-6)
                np.testing.assert_allclose(result.y, [0.5], atol=1e-6)
                np.testing.assert_allclose(result.z, [1.0], atol=1e-6)
                self.assertTrue(result.is_optimal(1e-6))
                self.assertGreaterEqual(result.build_time, 0.0)
                self.assertGreaterEqual(result.solve_time, 0.0)

    def test_pdhcg_bound_multiplier_signs(self):
        """Lower/upper multipliers have the qpsolvers sign convention."""
        problem = Problem(
            np.eye(3),
            np.array([1.0, -4.0, -0.5]),
            lb=np.array([1.0, -np.inf, -np.inf]),
            ub=np.array([np.inf, 2.0, np.inf]),
        )
        result = solve_problem(problem, **self.options)
        self.assertTrue(result.found)
        np.testing.assert_allclose(result.x, [1.0, 2.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(result.z_box, [-2.0, 2.0, 0.0], atol=1e-6)
        self.assertTrue(result.is_optimal(1e-6))

    def test_pdhcg_bound_multipliers_with_row_constraints(self):
        """Recover bound multipliers using both types of row duals."""
        problem = Problem(
            np.eye(3),
            np.array([0.0, -3.0, -2.0]),
            np.array([[1.0, 1.0, 0.0]]),
            np.array([3.0]),
            np.array([[0.0, 1.0, 1.0]]),
            np.array([5.0]),
            lb=np.array([1.0, -np.inf, -np.inf]),
        )
        result = solve_problem(problem, **self.options)
        self.assertTrue(result.found)
        np.testing.assert_allclose(result.x, [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(result.y, [-1.0], atol=1e-6)
        np.testing.assert_allclose(result.z, [2.0], atol=1e-6)
        np.testing.assert_allclose(result.z_box, [-3.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(result.obj, -5.0, places=6)
        self.assertTrue(result.is_optimal(1e-6))

    def test_pdhcg_one_sided_and_fixed_bounds(self):
        """Handle lower-only, upper-only and fixed variables."""
        for lb, ub, x, multiplier in [
            (np.array([1.0]), None, 1.0, -1.0),
            (None, np.array([-1.0]), -1.0, 1.0),
            (np.array([2.0]), np.array([2.0]), 2.0, -2.0),
        ]:
            with self.subTest(lb=lb, ub=ub):
                problem = Problem(np.eye(1), np.zeros(1), lb=lb, ub=ub)
                result = solve_problem(problem, **self.options)
                self.assertTrue(result.found)
                np.testing.assert_allclose(result.x, [x], atol=1e-6)
                np.testing.assert_allclose(
                    result.z_box, [multiplier], atol=1e-6
                )
                self.assertTrue(result.is_optimal(1e-6))

    def test_pdhcg_infeasible_has_no_solution(self):
        """Do not expose uninitialized multipliers after infeasibility."""
        problem = Problem(
            np.eye(1),
            np.zeros(1),
            np.array([[1.0], [-1.0]]),
            np.array([0.0, -1.0]),
        )
        result = solve_problem(problem, **self.options)
        self.assertFalse(result.found)
        self.assertEqual(result.extras["status"], "PRIMAL_INFEASIBLE")
        for value in (result.x, result.y, result.z, result.z_box):
            self.assertIsNone(value)
        self.assertIsNone(
            solve_qp(
                problem.P, problem.q, problem.G, problem.h, **self.options
            )
        )

    def test_pdhcg_unbounded_has_no_solution(self):
        """A linear recession direction is reported as dual infeasibility."""
        problem = Problem(np.zeros((1, 1)), np.array([-1.0]), lb=np.zeros(1))
        result = solve_problem(problem, **self.options)
        self.assertFalse(result.found)
        self.assertEqual(result.extras["status"], "DUAL_INFEASIBLE")
        self.assertIsNone(result.x)

    def test_pdhcg_iteration_limit_has_no_solution(self):
        """Only successful solves expose primal and dual solutions."""
        result = solve_problem(
            Problem(np.eye(2), -np.ones(2)), IterationLimit=0, **self.options
        )
        self.assertFalse(result.found)
        self.assertEqual(result.extras["status"], "ITERATION_LIMIT")
        for value in (result.x, result.y, result.z, result.z_box):
            self.assertIsNone(value)

    def test_pdhcg_warm_start_and_native_threads(self):
        """Forward a primal start and the native thread-count parameter."""
        options = {**self.options, "num_threads": 1}
        del options["Threads"]
        x = solve_qp(np.eye(2), -np.ones(2), initvals=np.ones(2), **options)
        np.testing.assert_allclose(x, np.ones(2), atol=1e-6)

    def test_pdhcg_direct_solver_entry_point(self):
        """The backend-specific solve_qp also forwards device and limits."""
        from qpsolvers.solvers.pdhcg_ import pdhcg_solve_qp

        options = dict(self.options)
        del options["solver"]
        x = pdhcg_solve_qp(np.eye(2), -np.ones(2), **options)
        np.testing.assert_allclose(x, np.ones(2), atol=1e-6)
        self.assertIsNone(
            pdhcg_solve_qp(np.eye(2), -np.ones(2), IterationLimit=0, **options)
        )

    def test_pdhcg_invalid_options(self):
        """Preserve backend and parameter validation errors."""
        for options, message in [
            ({"device": "missing"}, "missing"),
            ({"Threads": -1}, "num_threads"),
        ]:
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                solve_qp(np.eye(2), np.ones(2), **{**self.options, **options})
