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

from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from qpsolvers import solve_qp
from os.path import basename


if __name__ == "__main__":
    if get_ipython() is None:
        print "Usage: ipython -i %s\n" % basename(__file__)
        exit()

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

    u0 = solve_qp(P, q, G, h, solver='cvxopt')
    u1 = solve_qp(P, q, G, h, solver='gurobi')
    u2 = solve_qp(P, q, G, h, solver='qpoases')
    u3 = solve_qp(P, q, G, h, solver='cvxpy')
    u4 = solve_qp(P, q, G, h, solver='quadprog')
    u5 = solve_qp(P, q, G, h, solver='mosek')

    assert norm(u0 - u1) < 1e-4
    assert norm(u1 - u2) < 1e-4
    assert norm(u2 - u3) < 1e-4
    assert norm(u3 - u4) < 1e-4
    assert norm(u4 - u0) < 1e-4

    print "\nSYMBOLIC",
    print "\n========\n"
    for c in ["u1 = solve_qp(P, q, G, h, solver='gurobi')",
              "u3 = solve_qp(P, q, G, h, solver='cvxpy')",
              "u5 = solve_qp(P, q, G, h, solver='mosek')"]:
        print c
        get_ipython().magic(u'timeit %s' % c)

    print "\nNUMERIC (COLD START)",
    print "\n====================\n"
    for c in ["u0 = solve_qp(P, q, G, h, solver='cvxopt')",
              "u2 = solve_qp(P, q, G, h, solver='qpoases')",
              "u4 = solve_qp(P, q, G, h, solver='quadprog')"]:
        print c
        get_ipython().magic(u'timeit %s' % c)
