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

import sys

from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from os.path import basename, dirname, realpath

try:
    from qpsolvers import available_solvers, matrix_solvers, symbolic_solvers
    from qpsolvers import solve_qp
except ImportError:  # run locally if not installed
    sys.path.append(dirname(realpath(__file__)) + '/..')
    from qpsolvers import available_solvers, matrix_solvers, symbolic_solvers
    from qpsolvers import solve_qp


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
        print "Usage: ipython -i %s" % basename(__file__)
        exit()

    mat_instr = {
        solver: "u = solve_qp(P, q, G, h, solver='%s')" % solver
        for solver in matrix_solvers}
    sym_instr = {
        solver: "u = solve_qp(P, q, G, h, solver='%s')" % solver
        for solver in symbolic_solvers}

    sol0 = solve_qp(P, q, G, h, solver=available_solvers[0])
    for solver in available_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
        assert norm(sol - sol0) < 1e-4

    print "\nMATRIX",
    print "\n------"
    for solver, instr in mat_instr.iteritems():
        print "%s: " % solver,
        get_ipython().magic(u'timeit %s' % instr)

    print "\nSYMBOLIC",
    print "\n--------"
    for solver, instr in sym_instr.iteritems():
        print "%s: " % solver,
        get_ipython().magic(u'timeit %s' % instr)
