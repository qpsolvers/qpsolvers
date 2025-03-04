#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `MOSEK <https://www.mosek.com/>`__.

MOSEK is a solver for linear, mixed-integer linear, quadratic, mixed-integer
quadratic, quadratically constraint, conic and convex nonlinear mathematical
optimization problems. Its interior-point method is geared towards large scale
sparse problems, in particular for linear or conic quadratic programs.
"""

import warnings
from typing import Optional, Union

import mosek
import numpy as np
import scipy.sparse as spa

from ..problem import Problem
from ..solution import Solution
from ..solve_unconstrained import solve_unconstrained
from ..solvers.cvxopt_ import cvxopt_solve_problem


def mosek_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using MOSEK.

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
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
    """
    if problem.is_unconstrained:
        warnings.warn(
            "QP is unconstrained: solving with SciPy's LSQR rather than MOSEK"
        )
        return solve_unconstrained(problem)
    if "mosek" not in kwargs:
        kwargs["mosek"] = {}
    kwargs["mosek"][mosek.iparam.log] = 1 if verbose else 0
    solution = cvxopt_solve_problem(problem, "mosek", initvals, **kwargs)
    return solution


def mosek_solve_qp(
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
    r"""Solve a quadratic program using MOSEK.

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

    It is solved using the `MOSEK interface from CVXOPT
    <https://cvxopt.org/userguide/coneprog.html#optional-solvers>`_.

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
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = mosek_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
