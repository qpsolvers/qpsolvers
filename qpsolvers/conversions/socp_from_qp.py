#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright 2016-2022 Stéphane Caron and the qpsolvers contributors

"""Convert quadratic programs to second-order cone programs."""

from typing import Any, Dict, Optional, Tuple

from numpy import hstack, ndarray, sqrt, vstack, zeros
from numpy.linalg import LinAlgError, cholesky
from scipy.sparse import csc_matrix

from ..exceptions import ProblemError


def socp_from_qp(
    P: ndarray, q: ndarray, G: Optional[ndarray], h: Optional[ndarray]
) -> Tuple[ndarray, ndarray, ndarray, Dict[str, Any]]:
    r"""Convert a quadratic program to a second-order cone program.

    The quadratic program is defined by:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                \frac{1}{2} x^T P x + q^T x \\
            \mbox{subject to}
                & G x \leq h
        \end{array}\end{split}

    The equivalent second-order cone program is:

    .. math::

        \begin{split}\begin{array}{ll}
            \underset{x}{\mbox{minimize}} &
                c^T_s y \\
            \mbox{subject to}
                & G_s y \leq_{\cal K} h_s
        \end{array}\end{split}

    This function is adapted from ``ecosqp.m`` in the `ecos-matlab
    <https://github.com/embotech/ecos-matlab/>`_ repository. See the
    documentation in that script for details on this reformulation.

    Parameters
    ----------
    P :
        Primal quadratic cost matrix.
    q :
        Primal quadratic cost vector.
    G :
        Linear inequality constraint matrix.
    h :
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
        Dimension dictionary used by SOCP solvers, where ``dims["l"]`` is the
        number of inequality constraints.

    Raises
    ------
    ValueError :
        If the cost matrix is not positive definite.
    """
    n = P.shape[1]  # dimension of QP variable
    c_socp = hstack([zeros(n), 1])  # new SOCP variable stacked as [x, t]
    try:
        L = cholesky(P)
    except LinAlgError as e:
        error = str(e)
        if "not positive definite" in error:
            raise ProblemError("matrix P is not positive definite") from e
        raise e  # other linear algebraic error

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
    if G is not None and h is not None:
        G_socp = vstack([hstack([G, zeros((G.shape[0], 1))]), G_quad])
        h_socp = hstack([h, h_quad])
        dims["l"] = G.shape[0]
    else:  # no linear inequality constraint
        G_socp = G_quad
        h_socp = h_quad
        dims["l"] = 0

    G_socp = csc_matrix(G_socp)
    return c_socp, G_socp, h_socp, dims
