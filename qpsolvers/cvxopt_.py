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

from numpy import array
from cvxopt import matrix
from cvxopt.solvers import options, qp
from warnings import warn

options['show_progress'] = False  # disable cvxopt output


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, solver=None,
                    initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using CVXOPT <http://cvxopt.org/>.

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
    solver : string, optional
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals : array, shape=(n,), optional
        Warm-start guess vector.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    n = P.shape[1]
    # CVXOPT 1.1.7 only considers the lower entries of P
    # so we need to project on the symmetric part beforehand,
    # otherwise a wrong cost function will be used
    P = .5 * (P + P.T)
    # now we can proceed
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = qp(*args, solver=solver, initvals=initvals)
    if not ('optimal' in sol['status']):
        warn("QP optimum not found: %s" % sol['status'])
        return None
    return array(sol['x']).reshape((n,))
