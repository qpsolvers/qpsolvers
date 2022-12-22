#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2022 Stéphane Caron and the qpsolvers contributors.
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with qpsolvers. If not, see <http://www.gnu.org/licenses/>.

"""
Solver interface for `quadprog <https://github.com/quadprog/quadprog>`__.

quadprog is a C implementation of the Goldfarb-Idnani dual algorithm
[Goldfarb1983]_. It works best on well-conditioned dense problems.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from numpy import hstack, vstack
from quadprog import solve_qp

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..problem import Problem
from ..solution import Solution


def quadprog_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """
    Solve a quadratic program using `quadprog
    <https://pypi.python.org/pypi/quadprog/>`_.

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
        G, h = linear_from_box_inequalities(G, h, lb, ub)
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
    try:
        x, obj, xu, iterations, y, iact = solve_qp(
            qp_G, qp_a, qp_C, qp_b, meq, **kwargs
        )
    except ValueError as e:
        error = str(e)
        if "matrix G is not positive definite" in error:
            # quadprog writes G the cost matrix that we write P in this package
            raise ValueError("matrix P is not positive definite") from e
        if "no solution" in error:
            return Solution(problem)
        warnings.warn(f"quadprog raised a ValueError: {e}")
        return Solution(problem)

    solution = Solution(problem)
    solution.x = x
    solution.obj = obj

    n = P.shape[0]
    m = qp_C.shape[1] - meq if qp_C is not None else 0
    z, ys, z_box = __convert_dual_multipliers(y, n, m, meq, lb, ub)
    solution.y = ys
    solution.z = z
    solution.z_box = z_box

    solution.extras = {
        "iact": iact,
        "iterations": iterations,
        "xu": xu,
    }
    return solution


def __convert_dual_multipliers(
    y: np.ndarray,
    n: int,
    m: int,
    meq: int,
    lb: Optional[np.ndarray],
    ub: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert dual multipliers from quadprog to qpsolvers QP formulation.

    Parameters
    ----------
    y :
        Dual multipliers from quadprog.
    n :
        Number of optimization variables.
    m :
        Number of (in)equality constraints.
    meq :
        Number of equality constraints.
    lb :
        Lower bound vector for box inequalities, if any.
    ub :
        Upper bound vector for box inequalities, if any.

    Returns
    -------
    :
        Tuple of dual multipliers :code:`z, ys, z_box` corresponding
        respectively to linear inequalities, linear equalities, and box
        inequalities.
    """
    z, ys, z_box = None, None, None
    if meq > 0:
        ys = y[:meq]
    z, z_box = split_dual_linear_box(y[meq:], lb, ub)
    return z, ys, z_box


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
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
            \\underset{x}{\\mbox{minimize}} &
                \\frac{1}{2} x^T P x + q^T x \\\\
            \\mbox{subject to}
                & G x \\leq h                \\\\
                & A x = b                    \\\\
                & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `quadprog <https://pypi.python.org/pypi/quadprog/>`_.

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
    warnings.warn(
        "The return type of this function will change "
        "to qpsolvers.Solution in qpsolvers v3.0",
        DeprecationWarning,
        stacklevel=2,
    )
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = quadprog_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x
