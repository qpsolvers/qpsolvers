#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2021 Stéphane Caron <stephane.caron@normalesup.org>
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

"""
Test all available QP solvers on a dense quadratic program.
"""

from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from os.path import basename
from scipy.sparse import csc_matrix

from qpsolvers import dense_solvers, sparse_solvers
from qpsolvers import solve_qp


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
P_csc = csc_matrix(P)
G_csc = csc_matrix(G)


if __name__ == "__main__":
    if get_ipython() is None:
        print("Usage: ipython -i %s" % basename(__file__))
        exit()

    dense_instr = {
        solver: "u = solve_qp(P, q, G, h, solver='%s')" % solver
        for solver in dense_solvers}
    sparse_instr = {
        solver: "u = solve_qp(P_csc, q, G_csc, h, solver='%s')" % solver
        for solver in sparse_solvers}

    print("\nTesting all QP solvers on a dense quadratic program...")

    sol0 = solve_qp(P, q, G, h, solver=dense_solvers[0])
    abstol = 2e-4  # tolerance on absolute solution error
    for solver in dense_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
        delta = norm(sol - sol0)
        assert delta < abstol, "%s's solution offset by %.1e" % (solver, delta)
    for solver in sparse_solvers:
        sol = solve_qp(P_csc, q, G_csc, h, solver=solver)
        delta = norm(sol - sol0)
        assert delta < abstol, "%s's solution offset by %.1e" % (solver, delta)

    print("\nDense solvers\n-------------")
    for solver, instr in dense_instr.items():
        print("%s: " % solver, end='')
        get_ipython().magic('timeit %s' % instr)

    print("\nSparse solvers\n--------------")
    for solver, instr in sparse_instr.items():
        print("%s: " % solver, end='')
        get_ipython().magic('timeit %s' % instr)
