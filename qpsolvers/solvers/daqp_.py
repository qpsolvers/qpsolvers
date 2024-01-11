#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `DAQP <https://github.com/darnstrom/daqp>`__.

DAQP is a dual active-set algorithm implemented in C [Arnstrom2022]_.
It has been developed to solve small/medium scale dense problems.
"""

import warnings
from ctypes import c_int
from typing import Optional

import daqp
import numpy as np
from numpy import hstack, vstack

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
        Warm-start guess vector (not used).
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
    H, f, G, h, A, b, lb, ub = problem.unpack()

    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by DAQP")

    Atot: Optional[np.ndarray] = None
    bupper: Optional[np.ndarray] = None
    blower: Optional[np.ndarray] = None

    # General constraints
    if A is not None and b is not None:
        if G is not None and h is not None:
            Atot = vstack([G, A])
            bupper = hstack([h, b])
        else:
            Atot = A
            bupper = b
        meq = A.shape[0]
    else:  # no equality constraint
        if G is not None and h is not None:
            Atot = G
            bupper = h
        meq = 0

    if bupper is None:
        bupper = np.zeros((0,))
        Atot = np.zeros((0, f.size))

    mineq = bupper.size - meq
    blower = np.full(bupper.shape, -1e30)
    # Box constraints
    if ub is not None:
        bupper = hstack([ub, bupper])
        ms = ub.size
        if lb is not None:
            blower = hstack([lb, blower])
        else:
            blower = hstack([np.full(ms, -1e30), blower])
    else:  # No upper
        if lb is not None:
            ms = lb.size
            blower = hstack([lb, blower])
            bupper = hstack([np.full(ms, 1e30), bupper])
        else:
            ms = 0
    sense = np.zeros(bupper.shape, dtype=c_int)
    sense[ms + mineq :] = 5

    x, obj, exitflag, info = daqp.solve(
        H, f, Atot, bupper, blower, sense, **kwargs
    )

    solution = Solution(problem)
    solution.found = exitflag > 0
    if exitflag > 0:
        solution.x = x
        solution.obj = obj

        solution.z_box = info["lam"][:ms]
        solution.z = info["lam"][ms : ms + mineq]
        solution.y = info["lam"][ms + mineq :]
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
        Warm-start guess vector (not used).
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
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = daqp_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
