#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2020 Stephane Caron <stephane.caron@normalesup.org>
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

"""Solver interface for ECOS"""

from typing import Any, Dict, Optional
from warnings import warn

from ecos import solve
from numpy import hstack, ndarray, sqrt, vstack, zeros
from numpy.linalg import cholesky
from scipy.sparse import csc_matrix


__exit_flag_meaning__ = {
    0: "OPTIMAL",
    1: "PRIMAL INFEASIBLE",
    2: "DUAL INFEASIBLE",
    -1: "MAXIT REACHED",
}


def convert_to_socp(P, q, G, h):
    """
    Convert the Quadratic Program defined by:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} & \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to} & G x \\leq h
        \\end{array}\\end{split}

    to an equivalent Second-Order Cone Program:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} & c^T_s y \\\\
        \\mbox{subject to} & G_s y \\leq_{\\cal K} h_s
        \\end{array}\\end{split}

    This function is adapted from ``ecosqp.m`` in the `ecos-matlab
    <https://github.com/embotech/ecos-matlab/>`_ repository. See the documentation in
    that script for details on this reformulation.

    Parameters
    ----------
    P : numpy.array
        Primal quadratic cost matrix.
    q : numpy.array
        Primal quadratic cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.

    Returns
    -------
    c_socp : array
        SOCP cost vector.
    G_socp : array
        SOCP inequality matrix.
    h_socp : array
        SOCP inequality vector.
    dims : dict
        Dimension dictionary used by ECOS.
    """
    n = P.shape[1]  # dimension of QP variable
    c_socp = hstack([zeros(n), 1])  # new SOCP variable stacked as [x, t]
    L = cholesky(P)

    scale = 1.0 / sqrt(2)
    G_quad = vstack(
        [
            scale * hstack([q, -1.0]),
            hstack([-L.T, zeros((L.shape[0], 1))]),
            scale * hstack([-q, +1.0]),
        ]
    )
    h_quad = hstack([scale, zeros(L.shape[0]), scale])

    dims: Dict[str, Any] = {"q": [L.shape[0] + 2]}
    if G is None:
        G_socp = G_quad
        h_socp = h_quad
        dims["l"] = 0
    else:
        G_socp = vstack([hstack([G, zeros((G.shape[0], 1))]), G_quad])
        h_socp = hstack([h, h_quad])
        dims["l"] = G.shape[0]

    G_socp = csc_matrix(G_socp)
    return c_socp, G_socp, h_socp, dims


def ecos_solve_qp(
    P, q, G=None, h=None, A=None, b=None, initvals=None, verbose: bool = False
) -> Optional[ndarray]:
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{split}\\begin{array}{ll}
        \\mbox{minimize} &
            \\frac{1}{2} x^T P x + q^T x \\\\
        \\mbox{subject to}
            & G x \\leq h                \\\\
            & A x = h
        \\end{array}\\end{split}

    using `ECOS <https://github.com/embotech/ecos>`_.

    Parameters
    ----------
    P : numpy.array
        Primal quadratic cost matrix.
    q : numpy.array
        Primal quadratic cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).
    verbose : bool, optional
        Set to `True` to print out extra information.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        warn("note that warm-start values ignored by this wrapper")
    c_socp, G_socp, h_socp, dims = convert_to_socp(P, q, G, h)
    if A is not None:
        A_socp = hstack([A, zeros((A.shape[0], 1))])
        A_socp = csc_matrix(A_socp)
        solution = solve(c_socp, G_socp, h_socp, dims, A_socp, b, verbose=verbose)
    else:
        solution = solve(c_socp, G_socp, h_socp, dims, verbose=verbose)
    flag = solution["info"]["exitFlag"]
    if flag != 0:
        warn(f"ECOS returned exit flag {flag} ({__exit_flag_meaning__[flag]})")
        return None
    return solution["x"][:-1]
