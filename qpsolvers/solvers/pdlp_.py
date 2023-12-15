#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Solver interface for `PDLP`_.

.. _PDLP: https://developers.google.com/optimization/lp/pdlp_math

PDLP is a first-order method for convex quadratic programming aiming for
high-accuracy solutions and scaling to large problems. If you use PDLP in your
academic works, consider citing the corresponding paper [Applegate2021]_.
"""

from typing import Optional, Union

import numpy as np
import scipy.sparse as spa
from ortools.pdlp import solve_log_pb2, solvers_pb2
from ortools.pdlp.python import pdlp

from ..conversions import ensure_sparse_matrices
from ..problem import Problem
from ..solution import Solution


def pdlp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    eps_optimal_absolute: Optional[float] = None,
    eps_optimal_relative: Optional[float] = None,
    time_sec_limits: Optional[float] = None,
    verbosity_level: Optional[int] = None,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using PDLP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.
    eps_optimal_absolute :
        Absolute tolerance on the primal-dual residuals and duality gap.
    eps_optimal_relative :
        Relative tolerance on the primal-dual residuals and duality gap.
    time_sec_limits :
        Maximum computation time the solver is allowed, in seconds.
    verbosity_level :
        Verbosity level: 0 for no logging, 1 for summary statistics, 2 for
        per-iteration statistics, 3 for even more details and 4 for maximum
        details.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Note
    ----
    See also the `Mathematical background for PDLP
    <https://developers.google.com/optimization/lp/pdlp_math>`__ for details.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices(P, G, A)

    A_pdlp = None
    l_pdlp = None
    u_pdlp = None
    if G is not None and h is not None:
        A_pdlp = G
        l_pdlp = np.full(h.shape, -np.infty)
        u_pdlp = h
    if A is not None and b is not None:
        A_pdlp = A if A_pdlp is None else spa.vstack([A_pdlp, A], format="csc")
        l_pdlp = b if l_pdlp is None else np.hstack([l_pdlp, b])
        u_pdlp = b if u_pdlp is None else np.hstack([u_pdlp, b])

    qp = pdlp.QuadraticProgram()
    qp.objective_matrix = P
    qp.objective_vector = q
    qp.constraint_matrix = A_pdlp
    qp.constraint_lower_bounds = l_pdlp
    qp.constraint_upper_bounds = u_pdlp
    qp.variable_lower_bounds = lb
    qp.variable_upper_bounds = ub

    params = solvers_pb2.PrimalDualHybridGradientParams()
    optimality = params.termination_criteria.simple_optimality_criteria
    if eps_optimal_absolute is not None:
        optimality.eps_optimal_absolute = eps_optimal_absolute
    if eps_optimal_relative is not None:
        optimality.eps_optimal_relative = eps_optimal_relative
    if time_sec_limits is not None:
        params.termination_criteria.time_sec_limits = time_sec_limits
    if verbosity_level is not None:
        params.verbosity_level = verbosity_level
    else:  #
        params.verbosity_level = 1 if verbose else 0

    result = pdlp.primal_dual_hybrid_gradient(qp, params)
    solve_log = result.solve_log

    solution = Solution(problem)
    solution.extras = {
        "solve_log": solve_log,
        "solve_time_sec": solve_log.solve_time_sec,
    }
    solution.found = (
        solve_log.termination_reason
        == solve_log_pb2.TERMINATION_REASON_OPTIMAL
    )
    solution.x = result.primal_solution
    m = G.shape[0] if G is not None else 0
    meq = A.shape[0] if A is not None else 0
    y_pdlp = result.dual_solution
    solution.z = y_pdlp[:m] if G is not None else np.empty((0,))
    solution.y = y_pdlp[m : m + meq] if A is not None else np.empty((0,))
    if lb is not None or ub is not None:
        solution.z_box = result.reduced_costs
    return solution


def pdlp_solve_qp(
    P: Union[np.ndarray, spa.csc_matrix],
    q: Union[np.ndarray, spa.csc_matrix],
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    lb: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    ub: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using PDLP.

    The quadratic program is defined as:

    .. math::

        \begin{split}\begin{array}{ll}
        \underset{\mbox{minimize}}{x} &
            \frac{1}{2} x^T P x + q^T x \\
        \mbox{subject to}
            & G x \leq h                \\
            & A x = b                   \\
            & lb \leq x \leq ub
        \end{array}\end{split}

    It is solved using `PDLP
    <https://developers.google.com/optimization/lp/pdlp_math>`__.

    Parameters
    ----------
    P :
        Positive semidefinite cost matrix.
    q :
        Cost vector.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = pdlp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
