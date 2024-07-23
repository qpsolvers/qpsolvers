#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2023 Stéphane Caron and the qpsolvers contributors

"""Solver interface for NPPro.

The NPPro solver implements an enhanced Newton Projection with Proportioning
method for strictly convex quadratic programming. Currently, it is designed for
dense problems only.
"""

import warnings
from typing import Optional

import nppro
import numpy as np

from ..problem import Problem
from ..solution import Solution


def nppro_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using NPPro.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.

    Returns
    -------
    :
        Solution returned by the solver.

    Notes
    -----
    All other keyword arguments are forwarded as options to NPPro. For
    instance, you can call ``nppro_solve_qp(P, q, G, h, MaxIter=15)``.
    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``MaxIter``
         - Maximum number of iterations.
       * - ``SkipPreprocessing``
         - Skip preprocessing phase or not.
       * - ``SkipPhaseOne``
         - Skip feasible starting point finding or not.
       * - ``InfVal``
         - Values are assumed to be infinite above this threshold.
       * - ``HessianUpdates``
         - Enable Hessian updates or not.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()

    n = P.shape[0]
    m_iq = G.shape[0] if G is not None else 0
    m_eq = A.shape[0] if A is not None else 0
    m = m_iq + m_eq

    A_ = None
    l_ = None
    u_ = None
    lb_ = np.full(q.shape, -np.inf)
    ub_ = np.full(q.shape, +np.inf)
    if G is not None and h is not None:
        A_ = G
        l_ = np.full(h.shape, -np.inf)
        u_ = h
    if A is not None and b is not None:
        A_ = A if A_ is None else np.vstack([A_, A])
        l_ = b if l_ is None else np.hstack([l_, b])
        u_ = b if u_ is None else np.hstack([u_, b])
    if lb is not None:
        lb_ = lb
    if ub is not None:
        ub_ = ub

    # Create solver object
    solver = nppro.CreateSolver(n, m)

    # Use options from input if provided, defaults otherwise
    max_iter = kwargs.get("MaxIter", 100)
    skip_preprocessing = kwargs.get("SkipPreprocessing", False)
    skip_phase_one = kwargs.get("SkipPhaseOne", False)
    inf_val = kwargs.get("InfVal", 1e16)
    hessian_updates = kwargs.get("HessianUpdates", True)

    # Set options
    solver.setOption_MaxIter(max_iter)
    solver.setOption_SkipPreprocessing(skip_preprocessing)
    solver.setOption_SkipPhaseOne(skip_phase_one)
    solver.setOption_InfVal(inf_val)
    solver.setOption_HessianUpdates(hessian_updates)

    x0 = np.full(q.shape, 0)
    if initvals is not None:
        x0 = initvals

    # Conversion to datatype supported by the solver's C++ interface
    P = np.asarray(P, order="C", dtype=np.float64)
    q = np.asarray(q, order="C", dtype=np.float64)
    A_ = np.asarray(A_, order="C", dtype=np.float64)
    l_ = np.asarray(l_, order="C", dtype=np.float64)
    u_ = np.asarray(u_, order="C", dtype=np.float64)
    lb_ = np.asarray(lb_, order="C", dtype=np.float64)
    ub_ = np.asarray(ub_, order="C", dtype=np.float64)
    x0 = np.asarray(x0, order="C", dtype=np.float64)

    # Call solver
    x, fval, exitflag, iter_ = solver.solve(P, q, A_, l_, u_, lb_, ub_, x0)

    # Store solution
    exitflag_success = 1
    solution = Solution(problem)
    solution.found = exitflag == exitflag_success and not np.isnan(x).any()
    if not solution.found:
        # The second condition typically handle positive semi-definite cases
        # that are not catched by the solver yet
        warnings.warn(f"NPPro exited with status {exitflag}")
    solution.x = x
    solution.z = None  # not available yet
    solution.y = None  # not available yet
    solution.z_box = None  # not available yet
    solution.extras = {
        "cost": fval,
        "iter": iter_,
    }
    return solution


def nppro_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using NPPro.

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

    It is solved using NPPro.

    Parameters
    ----------
    P :
        Positive definite cost matrix.
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

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    See the Notes section in :func:`nppro_solve_problem`.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = nppro_solve_problem(problem, initvals, **kwargs)
    return solution.x if solution.found else None
