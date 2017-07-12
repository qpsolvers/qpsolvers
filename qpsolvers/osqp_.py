#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of qpsolvers.
#
# qpsolvers is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# qpsolvers is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# qpsolvers. If not, see <http://www.gnu.org/licenses/>.

from numpy import hstack, inf, ones, vstack
from osqp import OSQP
from scipy import sparse
from warnings import warn


def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using OSQP <https://github.com/oxfordcontrol/osqp>.

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    initvals : array, shape=(n,), optional
        Warm-start guess vector.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    n = q.shape[0]
    l = -inf * ones(n)
    qp_P = sparse.csc_matrix(P)
    qp_q = q
    if A is not None:
        qp_A = vstack([G, A])
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])
    else:  # no equality constraint
        qp_A = G
        qp_l = l
        qp_u = h
    qp_A = sparse.csc_matrix(qp_A)
    osqp = OSQP()
    osqp.setup(P=qp_P, q=qp_q, A=qp_A, l=qp_l, u=qp_u, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        warn("OSQP exited with status '%s'" % res.info.status)
    return res.x
