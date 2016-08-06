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

from cvxpy import Variable, Minimize, Problem, quad_form
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

    using CVXPY <http://www.cvxpy.org/>.
    """
    n = P.shape[0]
    x = Variable(n)
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
