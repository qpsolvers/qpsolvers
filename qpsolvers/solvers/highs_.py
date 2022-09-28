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
Solver interface for `HiGHS <https://github.com/ERGO-Code/HiGHS>`__.

HiGHS is a high performance serial and parallel solver for large scale sparse
linear optimization problems of the form

.. math::

    \\mathrm{minimize } (1/2) x^T Q x + c^T x
    \\quad
    \\mathrm{subject to } L \\leq Ax \\leq U;
    \\quad
    l \\leq x \\leq u

where Q must be positive semi-definite and, if Q is zero, there may be a
requirement that some of the variables take integer values. Thus HiGHS can
solve linear programming (LP) problems, convex quadratic programming (QP)
problems, and mixed integer programming (MIP) problems. It is mainly written in
C++, but also has some C. It has been developed and tested on various Linux,
MacOS and Windows installations using both the GNU (g++) and Intel (icc) C++
compilers. Note that HiGHS requires (at least) version 4.9 of the GNU compiler.
It has no third-party dependencies.

HiGHS has primal and dual revised simplex solvers, originally written by Qi
Huangfu and further developed by Julian Hall. It also has an interior point
solver for LP written by Lukas Schork, an active set solver for QP written by
Michael Feldmeier, and a MIP solver written by Leona Gottwald. Other features
have been added by Julian Hall and Ivet Galabova, who manages the software
engineering of HiGHS and interfaces to C, C#, FORTRAN, Julia and Python.
"""

from typing import Optional

import highspy
import numpy as np
import scipy.sparse as spa

from .typing import DenseOrCSCMatrix, warn_about_sparse_conversion


def highs_solve_qp(
    P: DenseOrCSCMatrix,
    q: np.ndarray,
    G: Optional[DenseOrCSCMatrix] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[DenseOrCSCMatrix] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    **kwargs,
) -> Optional[np.ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b                    \\\\
            & lb \\leq x \\leq ub
        \\end{array}\\end{split}

    using `HiGHS <https://github.com/ERGO-Code/HiGHS>`__.

    Parameters
    ----------
    P :
        Positive semidefinite quadratic-cost matrix.
    q :
        Quadratic-cost vector.
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
    backend :
        ProxQP backend to use in ``[None, "dense", "sparse"]``. If ``None``
        (default), the backend is selected based on the type of ``P``.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Notes
    -----
    ...
    """
    if isinstance(P, np.ndarray):
        warn_about_sparse_conversion("P")
        P = spa.csc_matrix(P)
    if isinstance(G, np.ndarray):
        warn_about_sparse_conversion("G")
        G = spa.csc_matrix(G)
    if isinstance(A, np.ndarray):
        warn_about_sparse_conversion("A")
        A = spa.csc_matrix(A)

    model = highspy.HighsModel()
    hessian = model.hessian_

    # Hessian part of the cost
    hessian.dim_ = P.shape[0]
    hessian.start_ = P.indptr
    hessian.index_ = P.indices
    hessian.value_ = P.data

    # Linear part of the cost
    n = P.shape[1]
    lp = model.lp_
    lp.num_col_ = n
    lp.col_cost_ = q
    lp.col_lower_ = lb if lb is not None else np.full((n,), -highspy.kHighsInf)
    lp.col_upper_ = ub if ub is not None else np.full((n,), highspy.kHighsInf)

    # Row inequalities:  L <= A * x <+ U
    lp.num_row_ = 0
    row_list = []
    row_lower = []
    row_upper = []
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
    row_matrix = spa.vstack(row_list, format="csc")
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = row_matrix.indptr
    lp.a_matrix_.index_ = row_matrix.indices
    lp.a_matrix_.value_ = row_matrix.data
    lp.a_matrix_.num_row_ = row_matrix.shape[0]
    lp.a_matrix_.num_col_ = row_matrix.shape[1]
    lp.row_lower_ = np.hstack(row_lower)
    lp.row_upper_ = np.hstack(row_upper)

    solver = highspy.Highs()
    solver.passModel(model)
    solution = solver.getSolution()
    info = solver.getInfo()
    model_status = solver.getModelStatus()
    if model_status != highspy.HighsModelStatus.kOptimal:
        print(f"{info.ipm_iteration_count=}, {info.qp_iteration_count=}")
        return None
    return solution
