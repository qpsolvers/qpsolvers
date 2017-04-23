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

num_solvers = {}
solutions = {}
sym_solvers = {}

solvers = [
    ('cvxopt', num_solvers), ('gurobi', sym_solvers),
    ('qpoases', num_solvers), ('cvxpy', sym_solvers),
    ('quadprog', num_solvers), ('mosek', sym_solvers)]


# QP matrices
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


if __name__ == "__main__":
    if get_ipython() is None:
        print "Usage: ipython -i %s\n" % basename(__file__)
        exit()

    for (solver, out_dict) in solvers:
        try:
            solutions[solver] = solve_qp(P, q, G, h, solver=solver)
            out_dict[solver] = "u = solve_qp(P, q, G, h, solver='%s')" % solver
        except:
            pass

    sol0 = solutions.values()[0]
    for sol in solutions.values():
        assert norm(sol - sol0) < 1e-4

    print "\nSYMBOLIC",
    print "\n========"
    for solver, instr in sym_solvers.iteritems():
        print "\n%s:" % solver
        get_ipython().magic(u'timeit %s' % instr)

    print "\nNUMERIC (COLD START)",
    print "\n===================="
    for solver, instr in num_solvers.iteritems():
        print "\n%s:" % solver
        get_ipython().magic(u'timeit %s' % instr)
