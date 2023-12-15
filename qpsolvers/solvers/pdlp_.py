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
        Absolute tolerance on the primal-dual residuals and duality gap. See
        *e.g.* [tolerances]_ for an overview of solver tolerances.
    eps_optimal_relative :
        Relative tolerance on the primal-dual residuals and duality gap. See
        *e.g.* [tolerances]_ for an overview of solver tolerances.
    time_sec_limits :
        Maximum computation time the solver is allowed, in seconds.

    Returns
    -------
    :
        Solution to the QP returned by the solver.

    Notes
    -----
    All other keyword arguments are forwarded as parameters to PDLP. For
    instance, you can call ``pdlp_solve_qp(P, q, G, h, num_threads=3,
    verbosity_level=2)``. For a quick overview, the solver accepts the
    following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``num_threads``
         - Number of threads to use (positive).
       * - ``verbosity_level``
         - Verbosity level from 0 (no logging) to 4 (extensive logging).
       * - ``initial_primal_weight``
         - Initial value of the primal weight (ratio of primal over  dual step
           sizes).
       * - ``l_inf_ruiz_iterations``
         - Number of L-infinity Ruiz rescaling iterations applied to the
           constraint matrix.
       * - ``l2_norm_rescaling``
         - If set to ``True``, applies L2-norm rescaling after Ruiz rescaling.

    This list is not exhaustive. Check out the solver's `Protocol Bufffers file
    <https://github.com/google/or-tools/blob/8768ed7a43f8899848effb71295a790f3ecbe2f2/ortools/pdlp/solvers.proto>`__
    for more. See also the `Mathematical background for PDLP
    <https://developers.google.com/optimization/lp/pdlp_math>`__.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices(P, G, A)
    n = P.shape[0]

    A_pdlp = None
    lc_pdlp = None
    uc_pdlp = None
    if G is not None and h is not None:
        A_pdlp = G
        lc_pdlp = np.full(h.shape, -np.infty)
        uc_pdlp = h
    if A is not None and b is not None:
        A_pdlp = A if A_pdlp is None else spa.vstack([A_pdlp, A], format="csc")
        lc_pdlp = b if lc_pdlp is None else np.hstack([lc_pdlp, b])
        uc_pdlp = b if uc_pdlp is None else np.hstack([uc_pdlp, b])
    lv_pdlp = lb if lb is not None else np.full((n,), -np.inf)
    uv_pdlp = ub if ub is not None else np.full((n,), +np.inf)

    qp = pdlp.QuadraticProgram()
    qp.objective_matrix = P
    qp.objective_vector = q
    if A_pdlp is not None:
        qp.constraint_matrix = A_pdlp
        qp.constraint_lower_bounds = lc_pdlp
        qp.constraint_upper_bounds = uc_pdlp
    qp.variable_lower_bounds = lv_pdlp
    qp.variable_upper_bounds = uv_pdlp

    params = solvers_pb2.PrimalDualHybridGradientParams()
    optimality = params.termination_criteria.simple_optimality_criteria
    if eps_optimal_absolute is not None:
        optimality.eps_optimal_absolute = eps_optimal_absolute
    if eps_optimal_relative is not None:
        optimality.eps_optimal_relative = eps_optimal_relative
    if time_sec_limits is not None:
        params.termination_criteria.time_sec_limits = time_sec_limits
    if verbose and "verbosity_level" not in kwargs:
        params.verbosity_level = 1 if verbose else 0
    for param, value in kwargs.items():
        setattr(params, param, value)

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
    eps_optimal_absolute: Optional[float] = None,
    eps_optimal_relative: Optional[float] = None,
    time_sec_limits: Optional[float] = None,
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
