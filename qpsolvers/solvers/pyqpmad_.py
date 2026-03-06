#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `pyqpmad <https://github.com/ahoarau/pyqpmad>`__.

pyqpmad is a Python wrapper for qpmad, a C++ implementation of
Goldfarb-Idnani's dual active-set method [Goldfarb1983]_. It works best on
well-conditioned dense problems with a positive-definite Hessian.

`qpmad <https://github.com/asherikov/qpmad>`

**Warm-start:** this solver interface supports warm starting 🌡️
"""

import warnings
from typing import Optional

import numpy as np
import pyqpmad
from numpy import hstack, vstack

from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def pyqpmad_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using pyqpmad.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Initial guess for the primal solution. pyqpmad uses this as a
        warm-start if provided.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Raises
    ------
    ProblemError :
        If the problem has sparse matrices (pyqpmad is a dense solver).

    Notes
    -----
    Keyword arguments are forwarded as attributes of a
    ``pyqpmad.SolverParameters`` object. Supported settings include:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``tolerance``
         - Solver feasibility/optimality tolerance.
       * - ``max_iter``
         - Maximum number of iterations.

    Check out the `qpmad documentation
    <https://asherikov.github.io/qpmad/>`_ for all available settings.
    """
    if verbose:
        warnings.warn("pyqpmad does not support verbose output")

    if problem.has_sparse:
        raise ProblemError("pyqpmad does not support sparse matrices")

    P, q, G, h, A, b, lb, ub = problem.unpack_as_dense()
    n = q.shape[0]

    A_qpmad = None
    Alb_qpmad = None
    Aub_qpmad = None

    if A is not None and b is not None:
        A_qpmad = A
        Alb_qpmad = b
        Aub_qpmad = b

    if G is not None and h is not None:
        A_qpmad = G if A_qpmad is None else vstack([A_qpmad, G])
        lb_G = np.full(h.shape, -np.inf)
        Alb_qpmad = lb_G if Alb_qpmad is None else hstack([Alb_qpmad, lb_G])
        Aub_qpmad = h if Aub_qpmad is None else hstack([Aub_qpmad, h])

    # qpmad requires the Hessian in Fortran (column-major) order.
    H_qpmad = np.asfortranarray(P, dtype=np.float64)
    if A_qpmad is not None:
        A_qpmad = np.asfortranarray(A_qpmad, dtype=np.float64)

    # Build solver parameters from keyword arguments.
    params = pyqpmad.SolverParameters()
    for key, val in kwargs.items():
        if hasattr(params, key):
            setattr(params, key, val)
        elif verbose:
            warnings.warn(f"pyqpmad ignoring unknown parameter: {key!r}")

    # Initialize primal variable; warm start if initvals is provided.
    x = (
        np.array(initvals, dtype=np.float64)
        if initvals is not None
        else np.zeros(n, dtype=np.float64)
    )

    lb_qpmad = None
    ub_qpmad = None
    if lb is not None or ub is not None:
        lb_qpmad = np.asarray(lb, dtype=np.float64) if lb is not None else np.full(n, -np.inf)
        ub_qpmad = np.asarray(ub, dtype=np.float64) if ub is not None else np.full(n, np.inf)

    solver = pyqpmad.Solver()
    try:
        status = solver.solve(
            x,
            H_qpmad,
            np.asarray(q, dtype=np.float64),
            lb_qpmad,
            ub_qpmad,
            A_qpmad,
            Alb_qpmad,
            Aub_qpmad,
            params,
        )
    except Exception:
        status = None # To make solution.found = False

    solution = Solution(problem)
    solution.found = status == pyqpmad.ReturnStatus.OK
    if solution.found:
        solution.x = x

        n_simple = n if (lb is not None or ub is not None) else 0
        n_eq_orig = A.shape[0] if A is not None else 0
        n_ineq_orig = G.shape[0] if G is not None else 0

        # Initialise dual arrays (zeros covers inactive constraints).
        # z is always a non-None array (possibly empty) after a successful solve.
        solution.y = np.zeros(n_eq_orig)
        solution.z = np.zeros(n_ineq_orig)
        if n_simple > 0:
            solution.z_box = np.zeros(n)

        # Reconstruct z and z_box from the active-set inequality duals.
        # qpmad index ordering: 0..n_simple-1 are simple bounds (lb/ub),
        # n_simple..n_simple+n_eq_orig-1 are equality rows of A_qpmad,
        # n_simple+n_eq_orig.. are inequality rows of A_qpmad (from G).
        ineq_dual = solver.get_inequality_dual()
        for i in range(len(ineq_dual.dual)):
            ci = int(ineq_dual.indices[i])
            d = float(ineq_dual.dual[i])
            if ci < n_simple:
                # Active box bound: sign follows qpsolvers convention
                # (negative for lower, positive for upper)
                solution.z_box[ci] = -d if ineq_dual.is_lower[i] else d
            else:
                j = ci - n_simple  # row in A_qpmad
                if j >= n_eq_orig:  # inequality row from G
                    solution.z[j - n_eq_orig] = d

        # Compute equality duals y from KKT stationarity to avoid any
        # sign-convention ambiguity with qpmad's internal dual storage:
        #   P x + q + A' y + G' z + z_box = 0  =>  A' y = -(P x + q + G' z + z_box)
        if A is not None and n_eq_orig > 0:
            residual = P @ x + q
            if G is not None and solution.z is not None:
                residual = residual + G.T @ solution.z
            if solution.z_box is not None:
                residual = residual + solution.z_box
            solution.y, _, _, _ = np.linalg.lstsq(A.T, -residual, rcond=None)

        solution.extras = {"num_iterations": solver.get_num_iterations()}
    return solution


def pyqpmad_solve_qp(
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
    r"""Solve a quadratic program using pyqpmad.

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

    It is solved using `pyqpmad <https://pypi.org/project/pyqpmad/>`__, a
    Python wrapper for the `qpmad
    <https://github.com/asherikov/qpmad>`__ C++ solver.

    Parameters
    ----------
    P :
        Symmetric positive-definite cost matrix.
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
        Initial guess for the primal solution (warm start).
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = pyqpmad_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
