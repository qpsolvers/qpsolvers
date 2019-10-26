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

from numpy import empty
from gurobipy import Model, QuadExpr, GRB, GurobiError, quicksum, setParam


try:
    setParam('OutputFlag', 0)
except GurobiError as e:
    print("GurobiError: {}".format(e))


def get_nonzero_rows(M):
    nonzero_rows = {}
    rows, cols = M.nonzero()
    for ij in zip(rows, cols):
        i, j = ij
        if i not in nonzero_rows:
            nonzero_rows[i] = []
        nonzero_rows[i].append(j)
    return nonzero_rows


def gurobi_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using Gurobi <http://www.gurobi.com/>.

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
        print("Gurobi: note that warm-start values are ignored by wrapper")
    n = P.shape[1]
    model = Model()
    x = {
        i: model.addVar(
            vtype=GRB.CONTINUOUS,
            name='x_%d' % i,
            lb=-GRB.INFINITY,
            ub=+GRB.INFINITY)
        for i in range(n)
    }
    model.update()   # integrate new variables

    # minimize
    #     1/2 x.T * P * x + q * x
    obj = QuadExpr()
    rows, cols = P.nonzero()
    for i, j in zip(rows, cols):
        obj += 0.5 * x[i] * P[i, j] * x[j]
    for i in range(n):
        obj += q[i] * x[i]
    model.setObjective(obj, GRB.MINIMIZE)

    # subject to
    #     G * x <= h
    if G is not None:
        G_nonzero_rows = get_nonzero_rows(G)
        for i, row in G_nonzero_rows.items():
            model.addConstr(quicksum(G[i, j] * x[j] for j in row) <= h[i])

    # subject to
    #     A * x == b
    if A is not None:
        A_nonzero_rows = get_nonzero_rows(A)
        for i, row in A_nonzero_rows.items():
            model.addConstr(quicksum(A[i, j] * x[j] for j in row) == b[i])

    model.optimize()

    a = empty(n)
    for i in range(n):
        a[i] = model.getVarByName('x_%d' % i).x
    return a
