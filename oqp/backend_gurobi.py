#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of oqp.
#
# oqp is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# oqp is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# oqp. If not, see <http://www.gnu.org/licenses/>.

from numpy import empty
from gurobipy import Model, QuadExpr, GRB, quicksum, setParam


setParam('OutputFlag', 0)


def gurobi_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    using Gurobi <http://www.gurobi.com/>.
    """
    n = P.shape[1]
    model = Model()
    x = {
        i: model.addVar(
            vtype=GRB.CONTINUOUS,
            name='x_%d' % i,
            lb=-GRB.INFINITY,
            ub=+GRB.INFINITY)
        for i in xrange(n)
    }
    model.update()   # integrate new variables

    # minimize
    #     1/2 x.T * P * x + q * x
    obj = QuadExpr()
    for i in xrange(n):
        for j in xrange(n):
            obj += 0.5 * x[i] * P[i, j] * x[j]
        obj += q[i] * x[i]
    model.setObjective(obj, GRB.MINIMIZE)

    # subject to
    #     G * x <= h
    if G is not None:
        for i in xrange(n):
            model.addConstr(quicksum(G[i, j] * x[j] for j in xrange(n)) <= h[i])

    # subject to
    #     A * x == b
    if A is not None:
        for i in xrange(n):
            model.addConstr(quicksum(A[i, j] * x[j] for j in xrange(n)) == b[i])

    model.optimize()

    a = empty(n)
    for i in xrange(n):
        a[i] = model.getVarByName('x_%d' % i).x
    return a
