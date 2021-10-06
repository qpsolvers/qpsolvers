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

from numpy import array
from qpsolvers import solve_ls
from time import time

R = array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
s = array([3.0, 2.0, 3.0])
G = array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
h = array([3.0, 2.0, -2.0]).reshape((3,))

t_start = time()
solver = "quadprog"  # see qpsolvers.available_solvers
x_sol = solve_ls(R, s, G, h, solver=solver, verbose=True)
t_end = time()

print("")
print("    min. || R * x - s ||^2")
print("    s.t. G * x <= h")
print("")
print("R =", R)
print("s =", s)
print("G =", G)
print("h =", h)
print("")
print("Solution: x =", x_sol)
print("Solve time:", 1000.0 * (t_end - t_start), "[ms]")
print("Solver:", solver)
