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

from ecos import solve
from numpy import hstack, sqrt, vstack, zeros
from numpy.linalg import cholesky
from scipy.sparse import csc_matrix


def ecos_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using ECOS <https://www.embotech.com/ECOS>.

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

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    This function is adapted from ``ecosqp.m`` in the `ecos-matlab
    <https://github.com/embotech/ecos-matlab/>`_ repository.
    """
    n = P.shape[1]  # dimension of QP variable
    c_socp = hstack([zeros(n), 1])  # new SOCP variable stacked as [x, t]
    L = cholesky(P)

    scale = 1. / sqrt(2)
    G_quad = vstack([
        scale * hstack([q, -1.]),
        hstack([-L.T, zeros((L.shape[0], 1))]),
        scale * hstack([-q, +1.])])
    h_quad = hstack([
        scale,
        zeros(L.shape[0]),
        scale])

    dims = {'q': [L.shape[0] + 2]}
    if G is None:
        G_socp = G_quad
        h_socp = h_quad
        dims['l'] = 0
    else:
        G_socp = vstack([
            hstack([G, zeros((G.shape[0], 1))]),
            G_quad])
        h_socp = hstack([h, h_quad])
        dims['l'] = G.shape[0]

    G_socp = csc_matrix(G_socp)
    kwargs = {'verbose': False}
    if A is not None:
        A_socp = hstack([A, zeros((A.shape[0], 1))])
        A_socp = csc_matrix(A_socp)
        solution = solve(c_socp, G_socp, h_socp, dims, A_socp, b, **kwargs)
    else:
        solution = solve(c_socp, G_socp, h_socp, dims, **kwargs)
    return solution['x'][:-1]
