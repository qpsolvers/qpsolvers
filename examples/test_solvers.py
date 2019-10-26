#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016-2017 Stephane Caron <stephane.caron@normalesup.org>
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

from __future__ import print_function  # Python 2 compatibility

import sys

from IPython import get_ipython
from numpy import array, dot
from numpy.linalg import norm
from os.path import basename, dirname, realpath

try:
    from qpsolvers import available_solvers
    from qpsolvers import solve_qp
except ImportError:  # run locally if not installed
    sys.path.append(dirname(realpath(__file__)) + '/..')
    from qpsolvers import available_solvers
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
A = array([
    [2., 0., 1.],
    [1., 0.5, 0.5]])
b = array([1., 1.])


if __name__ == "__main__":
    if get_ipython() is None:
        print("Usage: ipython -i %s" % basename(__file__))
        exit()

    cases = [
        {'P': P, 'q': q},
        {'P': P, 'q': q, 'G': G, 'h': h},
        {'P': P, 'q': q, 'A': A, 'b': b},
        {'P': P, 'q': q, 'A': A[0], 'b': b[0]},
        {'P': P, 'q': q, 'G': G, 'h': h, 'A': A, 'b': b},
    ]

    for (i, kwargs) in enumerate(cases):
        sol0 = solve_qp(solver=available_solvers[0], **kwargs)
        for solver in available_solvers:
            sol = solve_qp(solver=solver, **kwargs)
            delta = norm(sol - sol0)
            assert delta < 1e-4, \
                "%s's solution offset by %.1e on test #%d" % (solver, delta, i)
