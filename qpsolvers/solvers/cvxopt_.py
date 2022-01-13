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

"""Solver interface for CVXOPT."""

from typing import Optional

from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp
from numpy import array, ndarray

from .typing import CvxoptReadyMatrix


options["show_progress"] = False  # disable cvxopt output


def cvxopt_matrix(M: CvxoptReadyMatrix) -> matrix:
    """
    Convert matrix to CVXOPT format.

    Parameters
    ----------
    M :
        Matrix in NumPy format.

    Returns
    -------
    :
        Matrix in CVXOPT format.
    """
    if isinstance(M, ndarray):
        return matrix(M)
    if isinstance(M, (spmatrix, matrix)):
        return M
    coo = M.tocoo()
    return spmatrix(
        coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=M.shape
    )


def cvxopt_solve_qp(
    P: CvxoptReadyMatrix,
    q: CvxoptReadyMatrix,
    G: Optional[CvxoptReadyMatrix] = None,
    h: Optional[CvxoptReadyMatrix] = None,
    A: Optional[CvxoptReadyMatrix] = None,
    b: Optional[CvxoptReadyMatrix] = None,
    solver: str = None,
    initvals: Optional[ndarray] = None,
    verbose: bool = False,
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = b
        \\end{array}\\end{split}

    using `CVXOPT <http://cvxopt.org/>`_.

    Parameters
    ----------
    P :
        Symmetric quadratic-cost matrix.
    q :
        Quadratic-cost vector.
    G :
        Linear inequality matrix.
    h :
        Linear inequality vector.
    A :
        Linear equality constraint matrix.
    b :
        Linear equality constraint vector.
    solver :
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals :
        Warm-start guess vector.
    verbose :
        Set to `True` to print out extra information.

    Returns
    -------
    :
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    CVXOPT only considers the lower entries of `P`, therefore it will use a
    different cost than the one intended if a non-symmetric matrix is provided.
    """
    options["show_progress"] = verbose
    args = [cvxopt_matrix(P), cvxopt_matrix(q)]
    kwargs = {"G": None, "h": None, "A": None, "b": None}
    if G is not None and h is not None:
        kwargs["G"] = cvxopt_matrix(G)
        kwargs["h"] = cvxopt_matrix(h)
    if A is not None and b is not None:
        kwargs["A"] = cvxopt_matrix(A)
        kwargs["b"] = cvxopt_matrix(b)
    sol = qp(*args, solver=solver, initvals=initvals, **kwargs)
    if "optimal" not in sol["status"]:
        return None
    return array(sol["x"]).reshape((q.shape[0],))
