#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2018 Stephane Caron <stephane.caron@normalesup.org>
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

import cvxopt.msk
import mosek

from .cvxopt_ import cvxopt_solve_qp

cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}


def mosek_solve_qp(P, q, G, h, A=None, b=None, initvals=None):
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
    P : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Symmetric quadratic-cost matrix.
    q : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Quadratic-cost vector.
    G : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality constraint matrix.
    h : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear inequality constraint vector.
    A : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint matrix.
    b : numpy.array, cvxopt.matrix or cvxopt.spmatrix
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector.

    Returns
    -------
    x : numpy.array
        Solution to the QP, if found, otherwise ``None``.
    """
    return cvxopt_solve_qp(P, q, G, h, A, b, 'mosek', initvals)
