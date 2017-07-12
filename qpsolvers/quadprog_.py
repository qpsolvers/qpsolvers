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

from numpy import hstack, vstack
from quadprog import solve_qp
from warnings import warn


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using quadprog <https://pypi.python.org/pypi/quadprog/>.

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
        Warm-start guess vector (not used).

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        warn("warm-start values ignored by quadprog wrapper")
    qp_G = .5 * (P + P.T)   # quadprog assumes that P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -vstack([A, G]).T
        qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
