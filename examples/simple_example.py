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
Test the "quadprog" QP solver on a small dense problem.
"""

from numpy import array, dot
from qpsolvers import solve_qp
from time import time

M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))

t_start = time()
solver = "quadprog"  # see qpsolvers.available_solvers
x_sol = solve_qp(P, q, G, h, solver=solver, verbose=True)
t_end = time()

print("")
print("    min. 1/2 x^T P x + q^T x")
print("    s.t. G * x <= h")
print("")
print("P =", P)
print("q =", q)
print("G =", G)
print("h =", h)
print("")
print("Solution: x =", x_sol)
print("Solve time:", 1000. * (t_end - t_start), "[ms]")
print("Solver:", solver)
