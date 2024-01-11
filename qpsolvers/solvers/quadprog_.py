#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `quadprog <https://github.com/quadprog/quadprog>`__.

quadprog is a C implementation of the Goldfarb-Idnani dual algorithm
[Goldfarb1983]_. It works best on well-conditioned dense problems.
"""

import warnings
from typing import Optional

import numpy as np
from numpy import hstack, vstack
from quadprog import solve_qp

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def quadprog_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using quadprog.

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

    Raises
    ------
    ProblemError :
        If the cost matrix of the quadratic program if not positive definite,
        or if the problem is ill-formed in some way, for instance if some
        matrices are not dense.

    Note
    ----
    The quadprog solver only considers the lower entries of :math:`P`,
    therefore it will use a different cost than the one intended if a
    non-symmetric matrix is provided.

    Notes
    -----
    All other keyword arguments are forwarded to the quadprog solver. For
    instance, you can call ``quadprog_solve_qp(P, q, G, h, factorized=True)``.
    See the solver documentation for details.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if initvals is not None and verbose:
        warnings.warn("warm-start values are ignored by quadprog")
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
    qp_G = P
    qp_a = -q
    qp_C: Optional[np.ndarray] = None
    qp_b: Optional[np.ndarray] = None
    if A is not None and b is not None:
        if G is not None and h is not None:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
        else:
            qp_C = -A.T
            qp_b = -b
        meq = A.shape[0]
    else:  # no equality constraint
        if G is not None and h is not None:
            qp_C = -G.T
            qp_b = -h
        meq = 0

    solution = Solution(problem)
    try:
        x, obj, xu, iterations, y, iact = solve_qp(
            qp_G, qp_a, qp_C, qp_b, meq, **kwargs
        )
        solution.found = True
        solution.x = x
        solution.obj = obj

        z, z_box = split_dual_linear_box(y[meq:], lb, ub)
        solution.y = y[:meq] if meq > 0 else np.empty((0,))
        solution.z = z
        solution.z_box = z_box

        solution.extras = {
            "iact": iact,
            "iterations": iterations,
            "xu": xu,
        }
    except TypeError as error:
        raise ProblemError("problem has sparse matrices") from error
    except ValueError as error:
        solution.found = False
        error_message = str(error)
        if "matrix G is not positive definite" in error_message:
            # quadprog writes G the cost matrix that we write P in this package
            raise ProblemError("matrix P is not positive definite") from error
        if "no solution" not in error_message:
            warnings.warn(f"quadprog raised a ValueError: {error_message}")

    return solution


def quadprog_solve_qp(
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
    r"""Solve a quadratic program using quadprog.

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

    It is solved using `quadprog <https://pypi.python.org/pypi/quadprog/>`__.

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
        Primal solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = quadprog_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
