#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Solver interface for `HiGHS <https://github.com/ERGO-Code/HiGHS>`__.

HiGHS is an open source, serial and parallel solver for large scale sparse
linear programming (LP), mixed-integer programming (MIP), and quadratic
programming (QP). It is written mostly in C++11 and available under the MIT
licence. HiGHS's QP solver implements a Nullspace Active Set method. It works
best on moderately-sized dense problems. If you are using HiGHS in a scientific
work, consider citing the corresponding paper [Huangfu2018]_.
"""

import warnings
from typing import Optional, Union

import highspy
import numpy as np
import scipy.sparse as spa

from ..conversions import ensure_sparse_matrices
from ..problem import Problem
from ..solution import Solution


def __set_hessian(model: highspy.HighsModel, P: spa.csc_matrix) -> None:
    """Set Hessian :math:`Q` of the cost :math:`(1/2) x^T Q x + c^T x`.

    Parameters
    ----------
    model :
        HiGHS model.
    P :
        Positive semidefinite cost matrix.
    """
    model.hessian_.dim_ = P.shape[0]
    model.hessian_.start_ = P.indptr
    model.hessian_.index_ = P.indices
    model.hessian_.value_ = P.data


def __set_columns(
    model: highspy.HighsModel,
    q: np.ndarray,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
) -> None:
    r"""Set columns of the model.

    Columns consist of:

    - Linear part :math:`c` of the cost :math:`(1/2) x^T Q x + c^T x`
    - Box inequalities :math:`l \leq x \leq u``

    Parameters
    ----------
    model :
        HiGHS model.
    q :
        Cost vector.
    lb :
        Lower bound constraint vector.
    ub :
        Upper bound constraint vector.
    """
    n = q.shape[0]
    lp = model.lp_
    lp.num_col_ = n
    lp.col_cost_ = q
    lp.col_lower_ = lb if lb is not None else np.full((n,), -highspy.kHighsInf)
    lp.col_upper_ = ub if ub is not None else np.full((n,), highspy.kHighsInf)


def __set_rows(
    model: highspy.HighsModel,
    G: Optional[spa.csc_matrix] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[spa.csc_matrix] = None,
    b: Optional[np.ndarray] = None,
) -> None:
    r"""Set rows :math:`L \leq A x \leq U`` of the model.

    Parameters
    ----------
    model :
        HiGHS model.
    G :
        Linear inequality constraint matrix.
    h :
        Linear inequality constraint vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    """
    lp = model.lp_
    lp.num_row_ = 0
    row_list: list = []
    row_lower: list = []
    row_upper: list = []
    if G is not None:
        lp.num_row_ += G.shape[0]
        row_list.append(G)
        row_lower.append(np.full((G.shape[0],), -highspy.kHighsInf))
        row_upper.append(h)
    if A is not None:
        lp.num_row_ += A.shape[0]
        row_list.append(A)
        row_lower.append(b)
        row_upper.append(b)
    if not row_list:
        return
    row_matrix = spa.vstack(row_list, format="csc")
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = row_matrix.indptr
    lp.a_matrix_.index_ = row_matrix.indices
    lp.a_matrix_.value_ = row_matrix.data
    lp.a_matrix_.num_row_ = row_matrix.shape[0]
    lp.a_matrix_.num_col_ = row_matrix.shape[1]
    lp.row_lower_ = np.hstack(row_lower)
    lp.row_upper_ = np.hstack(row_upper)


def highs_solve_problem(
    problem: Problem,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Solution:
    """Solve a quadratic program using HiGHS.

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
        Solution returned by the solver.

    Notes
    -----
    Keyword arguments are forwarded to HiGHS as options. For instance, we
    can call ``highs_solve_qp(P, q, G, h, u, primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8)``. HiGHS settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``dual_feasibility_tolerance``
         - Dual feasibility tolerance.
       * - ``primal_feasibility_tolerance``
         - Primal feasibility tolerance.
       * - ``time_limit``
         - Run time limit in seconds.

    Check out the `HiGHS documentation <https://ergo-code.github.io/HiGHS/>`_
    for more information on the solver.
    """
    P, q, G, h, A, b, lb, ub = problem.unpack()
    P, G, A = ensure_sparse_matrices("highs", P, G, A)
    if initvals is not None:
        warnings.warn(
            "HiGHS: warm-start values are not available for this solver, "
            "see: https://github.com/qpsolvers/qpsolvers/issues/94"
        )

    model = highspy.HighsModel()
    __set_hessian(model, P)
    __set_columns(model, q, lb, ub)
    __set_rows(model, G, h, A, b)

    solver = highspy.Highs()
    if verbose:
        solver.setOptionValue("log_to_console", True)
        solver.setOptionValue("log_dev_level", highspy.HighsLogType.kVerbose)
        solver.setOptionValue(
            "highs_debug_level", highspy.HighsLogType.kVerbose
        )
    else:  # not verbose
        solver.setOptionValue("log_to_console", False)
    for option, value in kwargs.items():
        solver.setOptionValue(option, value)
    solver.passModel(model)
    solver.run()

    result = solver.getSolution()
    model_status = solver.getModelStatus()

    solution = Solution(problem)
    solution.found = model_status == highspy.HighsModelStatus.kOptimal
    solution.x = np.array(result.col_value)
    if G is not None:
        solution.z = -np.array(result.row_dual[: G.shape[0]])
        solution.y = (
            -np.array(result.row_dual[G.shape[0] :])
            if A is not None
            else np.empty((0,))
        )
    else:  # G is None
        solution.z = np.empty((0,))
        solution.y = (
            -np.array(result.row_dual) if A is not None else np.empty((0,))
        )
    solution.z_box = (
        -np.array(result.col_dual)
        if lb is not None or ub is not None
        else np.empty((0,))
    )
    return solution


def highs_solve_qp(
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
    r"""Solve a quadratic program using HiGHS.

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

    It is solved using `HiGHS <https://github.com/ERGO-Code/HiGHS>`__.

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
        Warm-start guess vector for the primal solution.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    Keyword arguments are forwarded to HiGHS as options. For instance, we
    can call ``highs_solve_qp(P, q, G, h, u, primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8)``. HiGHS settings include the following:

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Name
         - Description
       * - ``dual_feasibility_tolerance``
         - Dual feasibility tolerance.
       * - ``primal_feasibility_tolerance``
         - Primal feasibility tolerance.
       * - ``time_limit``
         - Run time limit in seconds.

    Check out the `HiGHS documentation <https://ergo-code.github.io/HiGHS/>`_
    for more information on the solver.
    """
    problem = Problem(P, q, G, h, A, b, lb, ub)
    solution = highs_solve_problem(problem, initvals, verbose, **kwargs)
    return solution.x if solution.found else None
