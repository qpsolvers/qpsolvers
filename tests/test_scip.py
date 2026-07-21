#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2024 Stéphane Caron and the qpsolvers contributors

"""Unit tests for SCIP."""

import importlib
import unittest
import warnings
from unittest import mock

import numpy as np

from qpsolvers.exceptions import ParamError, ProblemError
from qpsolvers.problem import Problem

from .problems import get_sd3310_problem

try:
    import pyscipopt

    from qpsolvers.solvers import scip_
    from qpsolvers.solvers.scip_ import scip_solve_problem, scip_solve_qp

    class TestSCIP(unittest.TestCase):
        """Test fixture for the SCIP solver."""

        def setUp(self):
            """Prepare test fixture."""
            warnings.simplefilter("ignore", category=UserWarning)

        def get_box_problem(self):
            """Problem with box constraints, optimum at [0.5, 0.5]."""
            P = np.eye(2)
            q = np.array([-1.0, -1.0])
            lb = np.array([0.0, 0.0])
            ub = np.array([0.5, 0.5])
            return P, q, lb, ub

        def test_problem(self):
            """Solve the sd3310 reference problem."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b)
            self.assertIsNotNone(x)
            self.assertTrue(
                np.allclose(
                    x, [0.30769231, -0.69230769, 1.38461538], atol=1e-4
                )
            )

        def test_box_constraints(self):
            """Solve a problem with only box constraints."""
            P, q, lb, ub = self.get_box_problem()
            x = scip_solve_qp(P, q, lb=lb, ub=ub)
            self.assertIsNotNone(x)
            self.assertTrue(np.allclose(x, [0.5, 0.5], atol=1e-4))

        def test_mixed_infinite_bounds(self):
            """Solve with infinite entries in the bound vectors."""
            P = np.eye(2)
            q = np.array([1.0, -1.0])
            lb = np.array([0.0, -np.inf])
            ub = np.array([np.inf, 2.0])
            x = scip_solve_qp(P, q, lb=lb, ub=ub)
            self.assertIsNotNone(x)
            self.assertTrue(np.allclose(x, [0.0, 1.0], atol=1e-4))

        def test_solution_dtype(self):
            """The solution vector has a floating-point dtype."""
            P, q, lb, ub = self.get_box_problem()
            x = scip_solve_qp(P, q, lb=lb, ub=ub)
            self.assertIsNotNone(x)
            self.assertEqual(x.dtype, np.float64)

        def test_objective_value(self):
            """The solution reports the primal objective value."""
            P, q, lb, ub = self.get_box_problem()
            problem = Problem(P, q, lb=lb, ub=ub)
            solution = scip_solve_problem(problem)
            self.assertTrue(solution.found)
            self.assertIsNotNone(solution.obj)
            self.assertAlmostEqual(solution.obj, -0.75, places=6)

        def test_timings(self):
            """The solution reports build and solve times."""
            solution = scip_solve_problem(get_sd3310_problem())
            self.assertIsNotNone(solution.build_time)
            self.assertIsNotNone(solution.solve_time)
            self.assertGreater(solution.build_time, 0.0)
            self.assertGreater(solution.solve_time, 0.0)

        def test_forward_kwargs(self):
            """Forward keyword arguments to SCIP as parameters."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b, **{"limits/time": 30.0})
            self.assertIsNotNone(x)

        def test_bad_param(self):
            """An unknown SCIP parameter raises ParamError, not KeyError."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            with self.assertRaises(ParamError):
                scip_solve_qp(P, q, G, h, A, b, **{"limits/tyme": 10.0})

        def test_gap_limit_found(self):
            """Stopping on the gap limit still counts as a found solution."""
            problem = get_sd3310_problem()
            P, q, G, h, A, b, _, _ = problem.unpack()
            x = scip_solve_qp(P, q, G, h, A, b, **{"limits/gap": 10.0})
            self.assertIsNotNone(x)

        def test_initvals_column_vector(self):
            """A (n, 1) column-vector warm start is accepted."""
            P, q, lb, ub = self.get_box_problem()
            x = scip_solve_qp(
                P, q, lb=lb, ub=ub, initvals=np.full((2, 1), 0.25)
            )
            self.assertIsNotNone(x)
            self.assertTrue(np.allclose(x, [0.5, 0.5], atol=1e-4))

        def test_initvals_wrong_length(self):
            """A warm start of the wrong length raises a clear error."""
            P, q, lb, ub = self.get_box_problem()
            with self.assertRaises(ValueError):
                scip_solve_qp(P, q, lb=lb, ub=ub, initvals=np.ones(5))

        def test_unbounded(self):
            """Raise ProblemError on an unconstrained unbounded problem.

            Unconstrained problems are routed to the library's LSQR path,
            which raises when the problem is unbounded below (the behavior
            the test suite lists as desired in ``behavior_on_unbounded``).
            """
            P = np.diag([1.0, 0.0])
            q = np.array([0.0, -1.0])
            with self.assertRaises(ProblemError):
                scip_solve_qp(P, q)

        def test_unbounded_constrained(self):
            """Return None on a constrained unbounded problem."""
            P = np.diag([1.0, 0.0])
            q = np.array([0.0, -1.0])
            G = np.array([[1.0, 0.0]])
            h = np.array([10.0])
            x = scip_solve_qp(P, q, G, h)
            self.assertIsNone(x)

        def test_infeasible(self):
            """Return None on an infeasible problem."""
            P = np.eye(2)
            q = np.zeros(2)
            G = np.array([[1.0, 0.0], [-1.0, 0.0]])
            h = np.array([-1.0, -1.0])
            x = scip_solve_qp(P, q, G, h)
            self.assertIsNone(x)

        def test_time_limit_returns_none(self):
            """A solve stopped by the time limit is not reported as found.

            An incumbent typically exists when the time limit strikes (here
            the warm start guarantees one), but limit statuses must not be
            reported as success: qpsolvers convention is that ``found`` means
            the solver terminated with its success status.
            """
            rng = np.random.default_rng(0)
            n = 400
            M = rng.standard_normal((n, n))
            P = M @ M.T + np.eye(n)
            q = rng.standard_normal(n)
            lb = np.full(n, -1.0)
            ub = np.full(n, 1.0)
            problem = Problem(P, q, lb=lb, ub=ub)
            solution = scip_solve_problem(
                problem, initvals=np.zeros(n), **{"limits/time": 0.1}
            )
            self.assertEqual(solution.extras["status"], "timelimit")
            self.assertFalse(solution.found)
            self.assertIsNone(solution.x)

        def test_warm_start_used(self):
            """SCIP accepts the warm-start guess as an initial solution.

            The solution limit is set to one, so SCIP stops as soon as it
            holds one solution. If the warm start is accepted, it is that
            first solution and the primal bound matches its objective value
            (+3); if it were rejected, the first solution would come from a
            SCIP heuristic instead (the trivial heuristic finds the origin,
            objective 0). The guess has a positive objective value on
            purpose: solutions stored without a value for the epigraph
            variable get rejected in exactly this case.
            """
            P = np.eye(2)
            q = np.array([1.0, 1.0])
            lb = np.zeros(2)
            problem = Problem(P, q, lb=lb)
            solution = scip_solve_problem(
                problem,
                initvals=np.array([1.0, 1.0]),
                **{"limits/solutions": 1},
            )
            self.assertEqual(solution.extras["status"], "sollimit")
            self.assertFalse(solution.found)  # a limit status is not success
            self.assertAlmostEqual(
                solution.extras["primal_bound"], 3.0, places=6
            )

        def test_solve_problem_no_duals(self):
            """The interface reports status and bounds but leaves duals unset.

            SCIP solves the QP by spatial branch-and-bound on an epigraph
            reformulation and does not produce dual multipliers following
            QP conventions, so the solution's dual attributes stay None.
            """
            problem = get_sd3310_problem()
            solution = scip_solve_problem(problem)
            self.assertTrue(solution.found)
            self.assertEqual(solution.extras["status"], "optimal")
            self.assertIn("primal_bound", solution.extras)
            self.assertIn("dual_bound", solution.extras)
            self.assertIsNone(solution.z)
            self.assertIsNone(solution.y)
            self.assertIsNone(solution.z_box)

        def test_old_pyscipopt_rejected(self):
            """Importing the interface with pyscipopt < 6.1 raises ImportError.

            The registration in qpsolvers.solvers guards on ImportError, so
            raising here keeps 'scip' out of available_solvers instead of
            letting it crash at solve time with an opaque internal error.
            """
            try:
                with mock.patch.object(pyscipopt, "__version__", "6.0.0"):
                    with self.assertRaises(ImportError):
                        importlib.reload(scip_)
            finally:
                importlib.reload(scip_)

except ImportError as exn:  # solver not installed
    warnings.warn(f"Skipping SCIP tests: {exn}")
