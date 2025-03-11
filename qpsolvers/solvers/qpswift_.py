#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `qpSWIFT <https://github.com/qpSWIFT/qpSWIFT>`__.

qpSWIFT is a light-weight sparse Quadratic Programming solver targeted for
embedded and robotic applications. It employs Primal-Dual Interior Point method
with Mehrotra Predictor corrector step and Nesterov Todd scaling. For solving
the linear system of equations, sparse LDL' factorization is used along with
approximate minimum degree heuristic to minimize fill-in of the factorizations.
If you use qpSWIFT in your research, consider citing the corresponding paper
[Pandala2019]_.
"""

import warnings
from typing import Optional

import numpy as np
import qpSWIFT

from ..conversions import linear_from_box_inequalities, split_dual_linear_box
from ..exceptions import ProblemError
from ..problem import Problem
from ..solution import Solution


def qpswift_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    r"""Solve a quadratic program using qpSWIFT.

    Note
    ----
    This solver does not handle problems without inequality constraints.

    Parameters
    ----------
    problem :
        Quadratic program to solve.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution returned by the solver.

    Raises
    ------
    ProblemError :
        If the problem is ill-formed in some way, for instance if some matrices
        are not dense or the problem has no inequality constraint.

    Note
    ----
    **Rank assumptions:** qpSWIFT requires the QP matrices to satisfy the

    .. math::

        \begin{split}\begin{array}{cc}
        \mathrm{rank}(A) = p
        &
        \mathrm{rank}([P\ A^T\ G^T]) = n
        \end{array}\end{split}

    where :math:`p` is the number of rows of :math:`A` and :math:`n` is the
    number of optimization variables. This is the same requirement as
    :func:`cvxopt_solve_qp`, however qpSWIFT does not perform rank checks as it
    prioritizes performance. If the solver fails on your problem, try running
    CVXOPT on it for rank checks.

    Notes
    -----
    All other keyword arguments are forwarded as options to the qpSWIFT solver.
    For instance, you can call ``qpswift_solve_qp(P, q, G, h, ABSTOL=1e-5)``.
    See the solver documentation for details.

    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - ``MAXITER``
         - Maximum number of iterations needed.
       * - ``ABSTOL``
         - Absolute tolerance on the duality gap. See *e.g.* [Caron2022]_ for
           a primer on the duality gap and solver tolerances.
       * - ``RELTOL``
         - Relative tolerance on the residuals :math:`r_x = P x + G^T z + q`
           (dual residual), :math:`r_y = A x - b` (primal residual on equality
           constraints) and :math:`r_z = h - G x - s` (primal residual on
           inequality constraints). See equation (21) in [Pandala2019]_.
       * - ``SIGMA``
         - Maximum centering allowed.

    If a verbose output shows that the maximum number of iterations is reached,
    check e.g. (1) the rank of your equality constraint matrix and (2) that
    your inequality constraint matrix does not have zero rows.

    As qpSWIFT does not sanity check its inputs, it should be used with a
    little more care than the other solvers. For instance, make sure you don't
    have zero rows in your input matrices, as it can `make the solver
    numerically unstable <https://github.com/qpSWIFT/qpSWIFT/issues/3>`_.
    """
    if initvals is not None:
        warnings.warn("qpSWIFT: warm-start values are ignored")
    P, q, G, h, A, b, lb, ub = problem.unpack()
    if lb is not None or ub is not None:
        G, h = linear_from_box_inequalities(G, h, lb, ub, use_sparse=False)
    result: dict = {}
    kwargs.update(
        {
            "OUTPUT": 2,  # include "sol", "basicInfo" and "advInfo"
            "VERBOSE": 1 if verbose else 0,
        }
    )

    try:
        if G is not None and h is not None:
            if A is not None and b is not None:
                result = qpSWIFT.run(q, h, P, G, A, b, kwargs)
            else:  # no equality constraint
                result = qpSWIFT.run(q, h, P, G, opts=kwargs)
        else:  # no inequality constraint
            # See https://github.com/qpSWIFT/qpSWIFT/issues/2
            raise ProblemError("problem has no inequality constraint")
    except TypeError as error:
        raise ProblemError("problem has sparse matrices") from error

    basic_info = result["basicInfo"]
    adv_info = result["advInfo"]

    solution = Solution(problem)
    solution.extras = {
        "basicInfo": basic_info,
        "advInfo": adv_info,
    }
    solution.obj = adv_info["fval"]
    exit_flag = basic_info["ExitFlag"]
    solution.found = exit_flag == 0
    solution.x = result["sol"]
    solution.y = adv_info["y"] if A is not None else np.empty((0,))
    z, z_box = split_dual_linear_box(adv_info["z"], lb, ub)
    solution.z = z
    solution.z_box = z_box
    return solution


def qpswift_solve_qp(
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
    r"""Solve a quadratic program using qpSWIFT.

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

    It is solved using `qpSWIFT <https://github.com/qpSWIFT/qpSWIFT>`__.

    Note
    ----
    This solver does not handle problems without inequality constraints yet.

    Parameters
    ----------
    P :
        Symmetric cost matrix. Together with :math:`A` and :math:`G`, it should
        satisfy :math:`\mathrm{rank}([P\ A^T\ G^T]) = n`, see the rank
        assumptions below.
    q :
        Cost vector.
    G :
        Linear inequality constraint matrix. Together with :math:`P` and
        :math:`A`, it should satisfy :math:`\mathrm{rank}([P\ A^T\ G^T]) =
        n`, see the rank assumptions below.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix. It needs to be full row rank, and
        together with :math:`P` and :math:`G` satisfy
        :math:`\mathrm{rank}([P\ A^T\ G^T]) = n`. See the rank assumptions
        below.
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

    Raises
    ------
    ProblemError :
        If the problem is ill-formed in some way, for instance if some matrices
        are not dense or the problem has no inequality constraint.

    Note
    ----
    .. _qpSWIFT rank assumptions:

    **Rank assumptions:** qpSWIFT requires the QP matrices to satisfy the

    .. math::

        \begin{split}\begin{array}{cc}
        \mathrm{rank}(A) = p
        &
        \mathrm{rank}([P\ A^T\ G^T]) = n
        \end{array}\end{split}

    where :math:`p` is the number of rows of :math:`A` and :math:`n` is the
    number of optimization variables. This is the same requirement as
    :func:`cvxopt_solve_qp`, however qpSWIFT does not perform rank checks as it
    prioritizes performance. If the solver fails on your problem, try running
    CVXOPT on it for rank checks.

    Notes
    -----
    All other keyword arguments are forwarded as options to the qpSWIFT solver.
    For instance, you can call ``qpswift_solve_qp(P, q, G, h, ABSTOL=1e-5)``.
    See the solver documentation for details.

    For a quick overview, the solver accepts the following settings:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Effect
       * - MAXITER
         - Maximum number of iterations needed.
       * - ABSTOL
         - Absolute tolerance on the duality gap. See *e.g.* [Caron2022]_ for
           a primer on the duality gap and solver tolerances.
       * - RELTOL
         - Relative tolerance on the residuals :math:`r_x = P x + G^T z + q`
           (dual residual), :math:`r_y = A x - b` (primal residual on equality
           constraints) and :math:`r_z = h - G x - s` (primal residual on
           inequality constraints). See equation (21) in [Pandala2019]_.
       * - SIGMA
         - Maximum centering allowed.

    If a verbose output shows that the maximum number of iterations is reached,
    check e.g. (1) the rank of your equality constraint matrix and (2) that
    your inequality constraint matrix does not have zero rows.

    As qpSWIFT does not sanity check its inputs, it should be used with a
    little more care than the other solvers. For instance, make sure you don't
    have zero rows in your input matrices, as it can `make the solver
    numerically unstable <https://github.com/qpSWIFT/qpSWIFT/issues/3>`_.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = qpswift_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
