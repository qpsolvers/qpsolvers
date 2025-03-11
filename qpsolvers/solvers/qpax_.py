#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2024 Lev Kozlov
"""
Solver interface for `qpax <https://github.com/kevin-tracy/qpax>`__.

qpax is an open-source QP solver that can be combined with JAX's jit and vmap
functionality, as well as differentiated with reverse-mode differentiation. It
is based on a primal-dual interior point algorithm. If you are using qpax in
a scientific work, consider citing the corresponding paper [Tracy2024]_.
"""

import warnings
from typing import Optional

import numpy as np
import qpax
import scipy.sparse as spa

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def qpax_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using qpax.

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
        Solution to the QP returned by the solver.

    Notes
    -----
    All other keyword arguments are forwarded as options to qpax. For
    instance, you can call ``qpax_solve_qp(P, q, G, h, solver_tol=1e-5)``.
    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``solver_tol``
         - Tolerance for the solver.

    Note that `jax` by default uses 32-bit floating point numbers, which can
    lead to numerical instability. If you encounter numerical issues, consider
    using 64-bit floating point numbers by setting
    ```python
    import jax
    jax.config.update("jax_enable_x64", True)
    ```
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    n: int = q.shape[0]

    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by qpax")

    if G is None and h is not None:
        raise ProblemError(
            "Inconsistent inequalities: G is not set but h is set"
        )
    if G is not None and h is None:
        raise ProblemError("Inconsistent inequalities: G is set but h is None")
    if A is None and b is not None:
        raise ProblemError(
            "Inconsistent inequalities: A is not set but b is set"
        )
    if A is not None and b is None:
        raise ProblemError("Inconsistent inequalities: A is set but b is None")

    # construct the qpax problem
    G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
    if G is None:
        G = np.zeros((0, n))
        h = np.zeros((0,))

    # qpax does not support A, b to be None.
    A_qpax = np.zeros((0, n)) if A is None else A
    b_qpax = np.zeros((0)) if b is None else b

    # qpax does not support sparse matrices,
    # so we need to convert them to dense
    if isinstance(P, spa.csc_matrix):
        P = P.toarray()
    if isinstance(A_qpax, spa.csc_matrix):
        A_qpax = A_qpax.toarray()
    if isinstance(G, spa.csc_matrix):
        G = G.toarray()

    x, s, z, y, converged, iters1 = qpax.solve_qp(
        P,
        q,
        A_qpax,
        b_qpax,
        G,
        h,
        **kwargs,
    )

    solution = Solution(problem)
    solution.x = x
    solution.found = converged
    solution.y = y

    # split the dual variables into
    # the box constraints and the linear constraints
    solution.z, solution.z_box = split_dual_linear_box(
        z, problem.lb, problem.ub
    )

    # store information about the solution
    # and the resulted raw variables in the extras
    solution.extras = {
        "info": {
            "iterations": iters1,
            "converged": converged,
        },
        "variables": {
            "x": x,
            "y": y,
            "z": z,
            "s": s,
        },
    }

    return solution


def qpax_solve_qp(
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
    r"""Solve a quadratic program using qpax.

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

    It is solved using `qpax
    <https://github.com/kevin-tracy/qpax>`__.
    `Paper: <https://arxiv.org/pdf/2406.11749>`__.

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
    verbose :
        Set to `True` to print out extra information.
    initvals :
        Warm-start guess vector. Not used.

    Returns
    -------
    :
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpax_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
