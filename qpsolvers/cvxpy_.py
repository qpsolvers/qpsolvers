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

from cvxpy import Constant, Minimize, Problem, Variable, quad_form
from numpy import array


def cvxpy_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None,
                   solver=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    calling a given solver using the CVXPY <http://www.cvxpy.org/> modelling
    language.

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
    solver : string, optional
        Solver name in ``cvxpy.installed_solvers()``.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    if initvals is not None:
        print("CVXPY: note that warm-start values are ignored by wrapper")
    n = q.shape[0]
    x = Variable(n)
    P = Constant(P)  # see http://www.cvxpy.org/en/latest/faq/
    objective = Minimize(0.5 * quad_form(x, P) + q * x)
    constraints = []
    if G is not None:
        constraints.append(G * x <= h)
    if A is not None:
        constraints.append(A * x == b)
    prob = Problem(objective, constraints)
    prob.solve(solver=solver)
    x_opt = array(x.value).reshape((n,))
    return x_opt


def ecos_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using ECOS <https://www.embotech.com/ECOS> called via CVXPY
    <http://www.cvxpy.org/>.

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
    solver : string, optional
        Solver name in ``cvxpy.installed_solvers()``.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    return cvxpy_solve_qp(P, q, G, h, A, b, initvals, solver="ECOS")
