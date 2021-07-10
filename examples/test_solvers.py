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
Test all solvers on all combinations of inequality/equality API calls.
"""

from numpy import array, dot
from numpy.linalg import norm

from qpsolvers import available_solvers
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
h0 = array([h[0]])
A = array([
    [1., 0., 0.],
    [0., 0.4, 0.5]])
b = array([-0.5, -1.2])
b0 = array([b[0]])
lb = array([-1., -1., -1.])
ub = array([+1., +1., +1.])


if __name__ == "__main__":
    cases = [
        {'P': P, 'q': q},
        {'P': P, 'q': q, 'G': G, 'h': h},
        {'P': P, 'q': q, 'A': A, 'b': b},
        {'P': P, 'q': q, 'G': G[0], 'h': h0},
        {'P': P, 'q': q, 'A': A[0], 'b': b0},
        {'P': P, 'q': q, 'G': G, 'h': h, 'A': A, 'b': b},
        {'P': P, 'q': q, 'G': G[0], 'h': h0, 'A': A, 'b': b},
        {'P': P, 'q': q, 'G': G, 'h': h, 'A': A[0], 'b': b0},
        {'P': P, 'q': q, 'G': G[0], 'h': h0, 'A': A[0], 'b': b0},
        {'P': P, 'q': q, 'G': G[0], 'h': h0, 'A': A[0], 'b': b0, 'lb': lb,
         'ub': ub},
    ]

    for (i, case) in enumerate(cases):
        print("\nTest %1d\n======\n" % i)
        expected_sol = solve_qp(solver=available_solvers[0], **case)
        for solver in available_solvers:
            sol = solve_qp(solver=solver, **case)
            delta = norm(sol - expected_sol)
            print("%9s's solution: %s\toffset: %.1e" % (
                solver, sol.round(decimals=5), delta))
            critical_offset = 2e-4
            assert delta < critical_offset, \
                "%s's solution offset by %.1e on test #%d" % (solver, delta, i)
