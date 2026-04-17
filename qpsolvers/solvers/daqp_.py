#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `DAQP <https://github.com/darnstrom/daqp>`__.

DAQP is a dual active-set algorithm implemented in C [Arnstrom2022]_.
It has been developed to solve small/medium scale dense problems.

**Warm-start:** this solver interface supports warm starting 🔥
"""

import time
import warnings
from ctypes import c_int
from typing import Optional

import daqp
import numpy as np

from ..problem import Problem
from ..solution import Solution


def daqp_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using DAQP.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to DAQP. For instance, we can call
    ``daqp_solve_qp(P, q, G, h, u, primal_tol=1e-6, iter_limit=1000)``. DAQP
    settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``iter_limit``
         - Maximum number of iterations.
       * - ``primal_tol``
         - Primal feasibility tolerance.
       * - ``dual_tol``
         - Dual feasibility tolerance.

    Check out the `DAQP settings
    <https://darnstrom.github.io/daqp/parameters>`_ documentation for
    all available settings.
    """
    build_start_time = time.perf_counter()
    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by DAQP")

    H, f, G, h, A, b, lb, ub = problem.unpack_as_dense()

    # Determine constraint counts upfront to allow single pre-allocation
    meq = A.shape[0] if A is not None else 0
    mineq = G.shape[0] if G is not None else 0
    if ub is not None:
        ms = ub.size
    elif lb is not None:
        ms = lb.size
    else:
        ms = 0
    mtot = ms + mineq + meq

    # Build bupper/blower. When there are no box constraints and only one
    # constraint block, reuse the existing arrays directly (zero extra copy).
    if ms == 0 and (mineq == 0 or meq == 0):
        bupper = h if (mineq > 0) else (b if meq > 0 else np.zeros(0))
        blower = np.full(mineq + meq, -1e30)
    else:
        bupper = np.empty(mtot)
        blower = np.full(mtot, -1e30)
        if ms > 0:
            bupper[:ms] = ub if ub is not None else 1e30
            if lb is not None:
                blower[:ms] = lb
        if mineq > 0:
            bupper[ms : ms + mineq] = h
        if meq > 0:
            bupper[ms + mineq :] = b

    # Build constraint matrix; stack only when both blocks are present
    if mineq > 0 and meq > 0:
        Atot = np.empty((mineq + meq, f.size))
        Atot[:mineq] = G
        Atot[mineq:] = A
    elif mineq > 0:
        Atot = G  # type: ignore[assignment]
    elif meq > 0:
        Atot = A  # type: ignore[assignment]
    else:
        Atot = np.zeros((0, f.size))

    sense = np.zeros(mtot, dtype=c_int)
    sense[ms + mineq :] = 5

    solve_start_time = time.perf_counter()
    x, obj, exitflag, info = daqp.solve(
        H, f, Atot, bupper, blower, sense, primal_start=initvals, **kwargs
    )
    solve_end_time = time.perf_counter()

    solution = Solution(problem)
    solution.found = exitflag > 0
    if exitflag > 0:
        solution.x = x
        solution.obj = obj

        solution.z_box = info["lam"][:ms]
        solution.z = info["lam"][ms : ms + mineq]
        solution.y = info["lam"][ms + mineq :]
    solution.build_time = solve_start_time - build_start_time
    solution.solve_time = solve_end_time - solve_start_time
    return solution


def daqp_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Solve a quadratic program using DAQP.

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

    It is solved using `DAQP <https://pypi.python.org/pypi/daqp/>`__.

    Parameters
    ----------
    P :
        Symmetric cost matrix.
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
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to DAQP. For instance, we can call
    ``daqp_solve_qp(P, q, G, h, u, primal_tol=1e-6, iter_limit=1000)``. DAQP
    settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``iter_limit``
         - Maximum number of iterations.
       * - ``primal_tol``
         - Primal feasibility tolerance.
       * - ``dual_tol``
         - Dual feasibility tolerance.
       * - ``time_limit``
         - Time limit for solve run in seconds.

    Check out the `DAQP settings
    <https://darnstrom.github.io/daqp/parameters>`_ documentation for
    all available settings.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = daqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
