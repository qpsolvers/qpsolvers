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


from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from oqp import solve_qp_cvxopt
from oqp import solve_qp_cvxpy
from oqp import solve_qp_gurobi
from oqp import solve_qp_qpoases
from oqp import solve_qp_quadprog


if __name__ == "__main__":
    M = array([
        [1., 2., 0.],
        [-8., 3., 2.],
        [0., 1., 1.]])
    P = dot(M.T, M)
    q = dot(array([3., 2., 3.]), M).reshape((3,))
    G = array([
        [1., 2., 1.],
        [2., 0., 1.],
        [-1., 2., -1.]])
    h = array([3., 2., -2.]).reshape((3,))

    u0 = solve_qp_cvxopt(P, q, G, h)
    u1 = solve_qp_gurobi(P, q, G, h)
    u2 = solve_qp_qpoases(P, q, G, h)
    u3 = solve_qp_cvxpy(P, q, G, h)
    u4 = solve_qp_quadprog(P, q, G, h)

    assert norm(u0 - u1) < 1e-4
    assert norm(u1 - u2) < 1e-4
    assert norm(u2 - u3) < 1e-4
    assert norm(u3 - u4) < 1e-4
    assert norm(u4 - u0) < 1e-4

    print "\nSYMBOLIC",
    print "\n========\n"
    for c in ["u1 = solve_qp_gurobi(P, q, G, h)",
              "u3 = solve_qp_cvxpy(P, q, G, h)"]:
        print c
        get_ipython().magic(u'timeit %s' % c)

    print "\nNUMERIC (COLD START)",
    print "\n====================\n"
    for c in ["u0 = solve_qp_cvxopt(P, q, G, h)",
              "u2 = solve_qp_qpoases(P, q, G, h)",
              "u4 = solve_qp_quadprog(P, q, G, h)"]:
        print c
        get_ipython().magic(u'timeit %s' % c)
