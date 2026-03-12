#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Solver interface for `QTQP`_.

.. _QTQP: https://github.com/google-deepmind/qtqp

QTQP is a primal-dual interior point method for solving convex quadratic
programs (QPs), implemented in pure Python. It is developed by Google DeepMind.

**Warm-start:** this solver interface does not support warm starting ❄️
"""

import time
import warnings
from typing import List, Optional, Union

import numpy as np
import qtqp
import scipy.sparse as spa

from ..conversions import (
    ensure_sparse_matrices,
    linear_from_box_inequalities,
    split_dual_linear_box,
)
from ..problem import Problem
from ..solution import Solution
from ..solve_unconstrained import solve_unconstrained


def qtqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    r"""Solve a quadratic program using QTQP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        This argument is not used by QTQP.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded as options to QTQP. For instance, we
    can call ``qtqp_solve_qp(P, q, G, h, atol=1e-8, max_iter=200)``. QTQP
    options include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``atol``
         - Absolute tolerance for optimality convergence (default: 1e-7).
       * - ``rtol``
         - Relative tolerance for optimality convergence (default: 1e-8).
       * - ``atol_infeas``
         - Absolute tolerance for infeasibility detection (default: 1e-8).
       * - ``rtol_infeas``
         - Relative tolerance for infeasibility detection (default: 1e-9).
       * - ``max_iter``
         - Maximum number of iterations (default: 100).
       * - ``step_size_scale``
         - Scale factor in (0,1) for line search step size (default: 0.99).
       * - ``min_static_regularization``
         - Diagonal regularization on KKT for robustness (default: 1e-7).
       * - ``max_iterative_refinement_steps``
         - Maximum steps for iterative refinement (default: 50).
       * - ``linear_solver_atol``
         - Absolute tolerance for iterative refinement (default: 1e-12).
       * - ``linear_solver_rtol``
         - Relative tolerance for iterative refinement (default: 1e-12).
       * - ``linear_solver``
         - KKT solver backend (default: qtqp.LinearSolver.SCIPY).
       * - ``equilibrate``
         - Scale/equilibrate data for numerical stability (default: True).

    Check out the `QTQP repository
    <https://github.com/google-deepmind/qtqp>`_ for details.

    Lower values for absolute or relative tolerances yield more precise
    solutions at the cost of computation time.
    """
    build_start_time = time.perf_counter()
    if initvals is not None and verbose:
        warnings.warn("QTQP: warm-start values are ignored")

    P, q, G, h, A, b, lb, ub = problem.unpack()

    # Convert box constraints to linear inequalities
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=True)

    # Convert to CSC format as required by QTQP
    P, G, A = ensure_sparse_matrices("qtqp", P, G, A)

    # Check for unconstrained case
    if G is None and A is None:
        warnings.warn(
            "QP is unconstrained: solving with SciPy's LSQR rather than QTQP"
        )
        return solve_unconstrained(problem)

    # Build the constraint matrix in QTQP format
    # QTQP expects: a @ x + s = b
    # where s[:z] == 0 (equality constraints)
    #       s[z:] >= 0 (inequality constraints)

    constraint_matrices: List[spa.csc_matrix] = []
    constraint_vectors = []

    # Add equality constraints first (these form the zero cone)
    z = 0
    if A is not None and b is not None:
        constraint_matrices.append(A)
        constraint_vectors.append(b)
        z = A.shape[0]

    # Add inequality constraints (these form the nonnegative cone)
    if G is not None and h is not None:
        constraint_matrices.append(G)
        constraint_vectors.append(h)

    # QTQP requires at least one inequality constraint (z < m)
    if G is None and A is not None:
        warnings.warn(
            "QTQP cannot solve problems with only equality constraints; "
            "at least one inequality constraint is required"
        )
        solution = Solution(problem)
        solution.found = False
        return solution

    # Stack all constraints
    a_qtqp = spa.vstack(constraint_matrices, format="csc")
    b_qtqp = np.concatenate(constraint_vectors)

    # QTQP uses 'c' for the cost vector (qpsolvers uses 'q')
    c_qtqp = q

    # QTQP uses 'p' for the cost matrix (qpsolvers uses 'P')
    p_qtqp = P if P is not None else None

    # Create QTQP solver instance
    solver = qtqp.QTQP(
        p=p_qtqp,
        a=a_qtqp,
        b=b_qtqp,
        c=c_qtqp,
        z=z,
    )

    # Solve the problem
    solve_start_time = time.perf_counter()
    result = solver.solve(verbose=verbose, **kwargs)
    solve_end_time = time.perf_counter()

    # Build solution
    solution = Solution(problem)
    solution.extras = {
        "status": result.status,
        "stats": result.stats,
    }

    # Check solution status
    solution.found = result.status == qtqp.SolutionStatus.SOLVED
    if not solution.found:
        warnings.warn(f"QTQP terminated with status {result.status.value}")

    # Extract primal solution
    solution.x = result.x

    # Extract dual variables
    # result.y contains dual variables for all constraints
    # First z entries correspond to equality constraints (y)
    # Remaining entries correspond to inequality constraints (z)
    meq = z
    if meq > 0:
        solution.y = result.y[:meq]
    else:
        solution.y = np.empty((0,))

    # Extract inequality duals
    if G is not None:
        z_ineq = result.y[meq:]
        # Split dual variables for linear and box inequalities
        z_linear, z_box = split_dual_linear_box(z_ineq, lb, ub)
        solution.z = z_linear
        solution.z_box = z_box
    else:
        solution.z = np.empty((0,))
        solution.z_box = np.empty((0,))

    # Compute objective value
    if solution.found and solution.x is not None:
        solution.obj = (
            0.5 * solution.x @ (P @ solution.x) if P is not None else 0.0
        )
        solution.obj += q @ solution.x

    solution.build_time = solve_start_time - build_start_time
    solution.solve_time = solve_end_time - solve_start_time
    return solution


def qtqp_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using QTQP.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h                \\
                & A x = b                   \\
                & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `QTQP`_.

    Parameters
    ----------
    P :
        Symmetric cost matrix.
    q :
        Cost vector.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        This argument is not used by QTQP.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to the solver. For example, we can call
    ``qtqp_solve_qp(P, q, G, h, atol=1e-8, max_iter=200)``.

    QTQP is a primal-dual interior point method that solves convex quadratic
    programs. It requires the cost matrix P to be positive semidefinite.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qtqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
