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

import numpy
import scipy.sparse
import sys

from IPython import get_ipython
from numpy.linalg import norm
from os.path import basename, dirname, realpath
from scipy.sparse import csc_matrix

try:
    from qpsolvers import dense_solvers, sparse_solvers
    from qpsolvers import solve_qp
except ImportError:  # run locally if not installed
    sys.path.append(dirname(realpath(__file__)) + '/..')
    from qpsolvers import dense_solvers, sparse_solvers
    from qpsolvers import solve_qp


# QP matrices
n = 1000
P = scipy.sparse.lil_matrix(scipy.sparse.eye(n))
for i in xrange(1, n - 1):
    P[i, i + 1] = -1
    P[i, i - 1] = 1
P = csc_matrix(P)
q = -numpy.ones((n,))
G = csc_matrix(-scipy.sparse.eye(n))
h = -2 * numpy.ones((n,))
P_array = numpy.array(P.todense())
G_array = numpy.array(G.todense())


def check_same_solutions(tol=0.05):
    sol0 = solve_qp(P, q, G, h, solver=sparse_solvers[0])
    for solver in sparse_solvers:
        sol = solve_qp(P, q, G, h, solver=solver)
        relvar = norm(sol - sol0) / norm(sol0)
        assert relvar < tol, "%s's solution offset by %.1f%%" % (
            solver, 100. * relvar)
    for solver in dense_solvers:
        sol = solve_qp(P_array, q, G_array, h, solver=solver)
        relvar = norm(sol - sol0) / norm(sol0)
        assert relvar < tol, "%s's solution offset by %.1f%%" % (
            solver, 100. * relvar)


def time_dense_solvers():
    instructions = {
        solver: "u = solve_qp(P_array, q, G_array, h, solver='%s')" % solver
        for solver in dense_solvers}
    print "\nDense solvers",
    print "\n-------------"
    for solver, instr in instructions.iteritems():
        print "%s: " % solver,
        get_ipython().magic(u'timeit %s' % instr)


def time_sparse_solvers():
    instructions = {
        solver: "u = solve_qp(P, q, G, h, solver='%s')" % solver
        for solver in sparse_solvers}
    print "\nSparse solvers",
    print "\n--------------"
    for solver, instr in instructions.iteritems():
        print "%s: " % solver,
        get_ipython().magic(u'timeit %s' % instr)


if __name__ == "__main__":
    if get_ipython() is None:
        print "Usage: ipython -i %s" % basename(__file__)
        exit()
    check_same_solutions()
    time_dense_solvers()
    time_sparse_solvers()
